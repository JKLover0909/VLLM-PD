"""Document parsing with page-aware PDF OCR and source metadata."""

import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import fitz
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class DocumentLimitError(ValueError):
    """Raised when an uploaded document exceeds a configured safety limit."""


class DocumentProcessingTimeout(TimeoutError):
    """Raised when document parsing exceeds its configured time budget."""


@dataclass
class TextChunk:
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    content_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"TextChunk(file='{self.source_file}', page={self.page_number}, "
            f"type='{self.content_type}', chars={len(self.text)})"
        )


class DocumentParser:
    """Parse supported documents while preserving PDF page provenance."""

    CHUNK_SIZE = 1400
    CHUNK_OVERLAP = 220
    MIN_NATIVE_PAGE_CHARS = 80
    MIN_CHUNK_CHARS = 20
    # Page-boundary markers embedded in curated OCR markdown ("<!-- Trang N -->"
    # or "<!-- Page N -->"). They align a curated .md page to the PDF page it
    # was transcribed from, so citations/previews keep pointing at the original.
    PAGE_MARKER_RE = re.compile(r"<!--\s*(?:Trang|Page)\s+(\d+)", re.IGNORECASE)
    MARKDOWN_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
    # Long directory-style tables (e.g. the employee roster) retrieve far better
    # when split by row instead of by character budget: each chunk then holds a
    # handful of complete rows with the header, keeping every field on one line
    # associated with its record instead of blending dozens of people together.
    BIG_TABLE_MIN_ROWS = 12
    TABLE_ROWS_PER_CHUNK = 15
    # Size cap for one table chunk. Sparse Excel exports (merged cells) can
    # produce rows of thousands of characters; packing a fixed 15 rows would
    # still exceed the prompt/embedding budget, so rows are packed until this
    # budget instead. Normal tables (short rows) keep the 15-row behaviour.
    TABLE_CHUNK_CHAR_BUDGET = 3600
    TABLE_SEPARATOR_CELL_RE = re.compile(r":?-{1,}:?")

    def __init__(self):
        self.max_pdf_pages = int(os.getenv("MAX_DOCUMENT_PAGES", "100"))
        self.processing_timeout = float(
            os.getenv("DOCUMENT_PROCESSING_TIMEOUT_SECONDS", "300")
        )
        self.ocr_device = os.getenv("DOCLING_DEVICE", "cuda").lower()
        self.ocr_threads = int(os.getenv("DOCLING_NUM_THREADS", "4"))
        self.ocr_languages = [
            language.strip()
            for language in os.getenv("DOCLING_OCR_LANGUAGES", "vi,en").split(",")
            if language.strip()
        ]

        logger.info(
            "Initializing Docling on device=%s threads=%s languages=%s...",
            self.ocr_device,
            self.ocr_threads,
            ",".join(self.ocr_languages),
        )
        accelerator_options = AcceleratorOptions(
            device=self.ocr_device,
            num_threads=self.ocr_threads,
        )
        pipeline_options = PdfPipelineOptions(
            accelerator_options=accelerator_options,
            document_timeout=self.processing_timeout,
            ocr_options=EasyOcrOptions(
                lang=self.ocr_languages,
                use_gpu=self.ocr_device.startswith("cuda"),
            ),
            ocr_batch_size=1,
            layout_batch_size=1,
            table_batch_size=1,
        )
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
                InputFormat.IMAGE: ImageFormatOption(
                    pipeline_options=pipeline_options,
                ),
            }
        )
        logger.info("Docling DocumentConverter initialized.")

    def process_file(
        self,
        file_path: str | Path,
        *,
        image_output_dir: str | Path | None = None,
        document_metadata: Dict[str, Any] | None = None,
        text_source: str | Path | None = None,
    ) -> List[TextChunk]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        metadata = dict(document_metadata or {})
        suffix = path.suffix.lower()
        if text_source:
            # Retrieval text comes from a curated OCR markdown file, while page
            # images / provenance still come from the original document so cited
            # documents keep displaying the genuine (stamped, signed) original.
            text_path = Path(text_source)
            if suffix == ".pdf":
                return self._process_pdf_with_text_source(
                    path, text_path, image_output_dir, metadata
                )
            return self._process_text_source(text_path, path.name, metadata)
        if suffix == ".pdf":
            return self._process_pdf(path, image_output_dir, metadata)
        return self._process_with_docling(path, metadata)

    def _split_markdown_pages(self, md_text: str) -> List[tuple[int, str]]:
        """Split curated markdown into (page_number, text) using page markers.

        Content before the first marker is folded into the first page. When no
        markers are present the whole document is treated as a single page.
        """
        marks = list(self.PAGE_MARKER_RE.finditer(md_text))
        if not marks:
            return [(1, md_text)]
        pages: List[tuple[int, str]] = []
        preamble = md_text[: marks[0].start()].strip()
        for i, match in enumerate(marks):
            page_number = int(match.group(1))
            end = marks[i + 1].start() if i + 1 < len(marks) else len(md_text)
            body = md_text[match.end():end]
            if i == 0 and preamble:
                body = f"{preamble}\n{body}"
            pages.append((page_number, body))
        return pages

    def _split_page_body(
        self,
        body: str,
        *,
        filename: str,
        page_number: int,
        start_index: int,
        metadata: Dict[str, Any],
    ) -> List[TextChunk]:
        """Chunk one curated page: row-wise for long tables, else by size."""
        if self._is_big_table(body):
            return self._split_markdown_table(
                body,
                filename=filename,
                page_number=page_number,
                start_index=start_index,
                metadata=metadata,
            )
        return self._split_text(
            body,
            filename=filename,
            page_number=page_number,
            start_index=start_index,
            metadata=metadata,
        )

    def _is_big_table(self, body: str) -> bool:
        rows = [
            line
            for line in body.splitlines()
            if self.MARKDOWN_TABLE_ROW.match(line)
        ]
        return len(rows) >= self.BIG_TABLE_MIN_ROWS

    def _compact_table_row(
        self,
        row: str,
        *,
        header_cells: List[str] | None = None,
    ) -> str:
        """Drop empty cells from an oversized sparse table row.

        Excel sheets with merged cells export rows that are thousands of
        characters of ``|  |  |`` padding around a handful of values. Plain
        positional compaction (just the surviving values, in order) throws
        away which column each value came from — for a row like "1億円以上"
        / "取締役会", the reader can no longer tell which is the threshold and
        which is the approver. When ``header_cells`` is given, each surviving
        value is prefixed with its column name ("基準: 1億円以上") so meaning
        survives compaction, not just the raw text. Rows of normal width are
        returned untouched.
        """
        if len(row) <= 300:
            return row
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        if header_cells is not None:
            labelled = [
                f"{header_cells[i]}: {cell}"
                if i < len(header_cells) and header_cells[i] and cell
                else cell
                for i, cell in enumerate(cells)
            ]
            filled = [cell for cell in labelled if cell.split(": ", 1)[-1]]
        else:
            filled = [cell for cell in cells if cell]
        if not filled:
            return ""
        # Keep the sparse row intact when most cells are filled: alignment
        # still matters for dense tables and compaction would corrupt it.
        if len(filled) / len(cells) > 0.5:
            return row
        return "| " + " | ".join(filled) + " |"

    def _split_markdown_table(
        self,
        body: str,
        *,
        filename: str,
        page_number: int,
        start_index: int,
        metadata: Dict[str, Any],
    ) -> List[TextChunk]:
        lines = [line for line in body.splitlines() if line.strip()]
        first_row_at = next(
            (i for i, line in enumerate(lines) if self.MARKDOWN_TABLE_ROW.match(line)),
            len(lines),
        )
        # Preamble text (title, notes) placed before the table. Lines that
        # appear *between* table rows are cell continuations from sparse
        # exports, not preamble — glueing them here once produced a single
        # 800k-char chunk. They are compacted as data rows below instead.
        preamble = "\n".join(lines[:first_row_at]).strip()[:600]
        table_rows: List[str] = []
        for line in lines[first_row_at:]:
            if self.MARKDOWN_TABLE_ROW.match(line):
                table_rows.append(line)
            elif "|" in line:
                table_rows.append(f"| {line.strip().strip('|').strip()} |")
            elif table_rows:
                # Plain continuation text: append to the previous row so the
                # record keeps its trailing note without becoming "preamble".
                table_rows[-1] = f"{table_rows[-1]} {line.strip()}"
        header_row, separator_row = table_rows[0], table_rows[1]
        if self._is_separator_row(header_row):
            # Docling omits the header for some sheets, emitting the separator
            # first; a giant |---|---| row is pure noise, keep a minimal one.
            header_cells: List[str] = []
            header_row, separator_row = "", "|---|---|---|"
        else:
            # Raw (uncompacted) header cells label data-row values below, so
            # a compacted data row reads "決裁者: 取締役会" instead of a bare
            # "取締役会" that lost which column it came from.
            header_cells = [
                cell.strip()
                for cell in header_row.strip().strip("|").split("|")
            ]
            header_row = self._compact_table_row(header_row)
            separator_row = "|" + "---|" * max(header_row.count("|") - 1, 2)
        header = [row for row in (header_row, separator_row) if row]
        data_rows = [
            compacted
            for row in table_rows[2:]
            if not self._is_separator_row(row)
            and (
                compacted := self._compact_table_row(
                    row, header_cells=header_cells
                )
            )
        ]

        chunks: List[TextChunk] = []
        chunk_index = start_index
        header_size = sum(len(row) + 1 for row in header)
        block_rows: List[str] = []
        block_size = header_size
        blocks: List[List[str]] = []
        for row in data_rows:
            over_budget = block_size + len(row) + 1 > self.TABLE_CHUNK_CHAR_BUDGET
            if block_rows and (over_budget or len(block_rows) >= self.TABLE_ROWS_PER_CHUNK):
                blocks.append(block_rows)
                block_rows = []
                block_size = header_size
            block_rows.append(row)
            block_size += len(row) + 1
        if block_rows:
            blocks.append(block_rows)

        for position, rows in enumerate(blocks):
            block = "\n".join(header + rows)
            # Repeat the preamble on the first chunk so document context (title
            # of the table) stays attached without bloating every chunk.
            if position == 0 and preamble:
                block = f"{preamble}\n{block}"
            if len(block.strip()) >= self.MIN_CHUNK_CHARS:
                chunks.append(
                    self._make_chunk(
                        block,
                        filename,
                        page_number,
                        chunk_index,
                        metadata,
                    )
                )
                chunk_index += 1
        return chunks

    def _is_separator_row(self, row: str) -> bool:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        return all(
            not cell or self.TABLE_SEPARATOR_CELL_RE.fullmatch(cell)
            for cell in cells
        )

    def _process_pdf_with_text_source(
        self,
        path: Path,
        text_path: Path,
        image_output_dir: str | Path | None,
        document_metadata: Dict[str, Any],
    ) -> List[TextChunk]:
        md_text = text_path.read_text(encoding="utf-8")
        pages = self._split_markdown_pages(md_text)

        output_dir = Path(image_output_dir) if image_output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        image_by_page: Dict[int, str] = {}
        with fitz.open(path) as pdf:
            if self.max_pdf_pages > 0 and pdf.page_count > self.max_pdf_pages:
                raise DocumentLimitError(
                    f"PDF has {pdf.page_count} pages; the limit is "
                    f"{self.max_pdf_pages} pages."
                )
            pdf_page_count = pdf.page_count
            if output_dir:
                for page_index, page in enumerate(pdf):
                    page_number = page_index + 1
                    target = output_dir / f"page-{page_number:04d}.png"
                    self._render_pdf_page(page, target)
                    image_by_page[page_number] = str(target.resolve())

        marker_pages = [number for number, _ in pages]
        if marker_pages != list(range(1, len(marker_pages) + 1)) or (
            len(marker_pages) != pdf_page_count
        ):
            logger.warning(
                "Curated text source '%s' has page markers %s that do not align "
                "with the %s pages of '%s'; citations may be off.",
                text_path.name,
                marker_pages,
                pdf_page_count,
                path.name,
            )

        chunks: List[TextChunk] = []
        chunk_index = 0
        for page_number, body in pages:
            page_metadata = {
                **document_metadata,
                "source": "curated-md",
                "ocr_method": "curated",
                "ocr_chars": len(body.strip()),
                "text_source": text_path.name,
            }
            image_path = image_by_page.get(page_number)
            if image_path:
                page_metadata["image_path"] = image_path
            page_chunks = self._split_page_body(
                body,
                filename=path.name,
                page_number=page_number,
                start_index=chunk_index,
                metadata=page_metadata,
            )
            chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        logger.info(
            "Document '%s' processed from curated text source '%s' into %s chunks.",
            path.name,
            text_path.name,
            len(chunks),
        )
        return chunks

    def _process_text_source(
        self,
        text_path: Path,
        display_name: str,
        document_metadata: Dict[str, Any],
    ) -> List[TextChunk]:
        """Use curated markdown as the text for a non-PDF source (html/docx)."""
        md_text = text_path.read_text(encoding="utf-8")
        chunks: List[TextChunk] = []
        chunk_index = 0
        for page_number, body in self._split_markdown_pages(md_text):
            page_metadata = {
                **document_metadata,
                "source": "curated-md",
                "ocr_method": "curated",
                "ocr_chars": len(body.strip()),
                "text_source": text_path.name,
            }
            page_chunks = self._split_page_body(
                body,
                filename=display_name,
                page_number=page_number,
                start_index=chunk_index,
                metadata=page_metadata,
            )
            chunks.extend(page_chunks)
            chunk_index += len(page_chunks)
        return chunks

    def _process_pdf(
        self,
        path: Path,
        image_output_dir: str | Path | None,
        document_metadata: Dict[str, Any],
    ) -> List[TextChunk]:
        output_dir = Path(image_output_dir) if image_output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        chunks: List[TextChunk] = []
        chunk_index = 0
        deadline = (
            time.monotonic() + self.processing_timeout
            if self.processing_timeout > 0
            else None
        )
        with fitz.open(path) as pdf:
            if self.max_pdf_pages > 0 and pdf.page_count > self.max_pdf_pages:
                raise DocumentLimitError(
                    f"PDF has {pdf.page_count} pages; the limit is "
                    f"{self.max_pdf_pages} pages."
                )

            for page_index, page in enumerate(pdf):
                self._check_deadline(deadline, path.name)
                page_number = page_index + 1
                native_text = page.get_text("text").strip()
                use_ocr = len(native_text) < self.MIN_NATIVE_PAGE_CHARS
                image_path = None

                if use_ocr or output_dir:
                    target = (
                        output_dir / f"page-{page_number:04d}.png"
                        if output_dir
                        else None
                    )
                    page_image = self._render_pdf_page(page, target)
                    if target:
                        image_path = str(target.resolve())
                    if use_ocr:
                        native_text = self._ocr_image(page_image)
                        self._check_deadline(deadline, path.name)
                    if not target:
                        page_image.unlink(missing_ok=True)

                page_metadata = {
                    **document_metadata,
                    "source": "docling-ocr" if use_ocr else "pymupdf",
                    "ocr_method": "docling" if use_ocr else "native",
                    "ocr_chars": len(native_text),
                }
                if image_path:
                    page_metadata["image_path"] = image_path

                page_chunks = self._split_text(
                    native_text,
                    filename=path.name,
                    page_number=page_number,
                    start_index=chunk_index,
                    metadata=page_metadata,
                )
                chunks.extend(page_chunks)
                chunk_index += len(page_chunks)

        logger.info(
            "Document '%s' processed into %s page-aware chunks.",
            path.name,
            len(chunks),
        )
        return chunks

    @staticmethod
    def _check_deadline(deadline: float | None, filename: str) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise DocumentProcessingTimeout(
                f"Processing '{filename}' exceeded the configured time limit."
            )

    def _process_with_docling(
        self,
        path: Path,
        document_metadata: Dict[str, Any],
    ) -> List[TextChunk]:
        try:
            result = self.converter.convert(path)
            markdown = result.document.export_to_markdown()
        except Exception:
            logger.exception("Error converting document %s with Docling", path.name)
            raise

        metadata = {**document_metadata, "source": "docling"}
        # Office exports (xlsx/docx/pptx) can be one huge markdown table;
        # _split_page_body routes those through row-wise table splitting
        # instead of the size-based splitter, which cannot break a single
        # oversized table row.
        chunks = self._split_page_body(
            markdown,
            filename=path.name,
            page_number=1,
            start_index=0,
            metadata=metadata,
        )
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            image_path = str(path.resolve())
            if not chunks:
                chunks = [
                    TextChunk(
                        text=(
                            f"Hình ảnh được tải lên: {path.name}. "
                            "Nội dung chi tiết cần được phân tích bằng mô hình vision."
                        ),
                        source_file=path.name,
                        page_number=1,
                        chunk_index=0,
                        content_type="image",
                        metadata={**metadata, "image_path": image_path},
                    )
                ]
            for chunk in chunks:
                chunk.content_type = "image"
                chunk.metadata["image_path"] = image_path
        return chunks

    def _render_pdf_page(self, page: fitz.Page, target: Path | None) -> Path:
        if target is None:
            handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            handle.close()
            target = Path(handle.name)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        pixmap.save(target)
        return target

    def _ocr_image(self, image_path: Path) -> str:
        try:
            result = self.converter.convert(image_path)
            return result.document.export_to_markdown().strip()
        except Exception:
            logger.exception("OCR failed for %s", image_path)
            return ""

    def _split_text(
        self,
        text: str,
        *,
        filename: str,
        page_number: int,
        start_index: int,
        metadata: Dict[str, Any],
    ) -> List[TextChunk]:
        cleaned = text.strip()
        if len(cleaned) < self.MIN_CHUNK_CHARS:
            return []

        # A single line longer than CHUNK_SIZE (minified export, one-line
        # sheet dump) would otherwise pass through as one oversized chunk,
        # blowing the embedding/prompt budget. Hard-split such lines.
        lines: List[str] = []
        for line in cleaned.splitlines():
            if len(line) <= self.CHUNK_SIZE:
                lines.append(line)
                continue
            for start in range(0, len(line), self.CHUNK_SIZE):
                lines.append(line[start:start + self.CHUNK_SIZE])
        chunks: List[TextChunk] = []
        current: List[str] = []
        current_size = 0
        chunk_index = start_index

        for line in lines:
            if current and current_size + len(line) + 1 > self.CHUNK_SIZE:
                chunk_text = "\n".join(current).strip()
                if len(chunk_text) >= self.MIN_CHUNK_CHARS:
                    chunks.append(
                        self._make_chunk(
                            chunk_text,
                            filename,
                            page_number,
                            chunk_index,
                            metadata,
                        )
                    )
                    chunk_index += 1

                overlap: List[str] = []
                overlap_size = 0
                for previous in reversed(current):
                    if overlap_size + len(previous) > self.CHUNK_OVERLAP:
                        break
                    overlap.insert(0, previous)
                    overlap_size += len(previous) + 1
                current = overlap
                current_size = overlap_size

            current.append(line)
            current_size += len(line) + 1

        final_text = "\n".join(current).strip()
        if len(final_text) >= self.MIN_CHUNK_CHARS:
            chunks.append(
                self._make_chunk(
                    final_text,
                    filename,
                    page_number,
                    chunk_index,
                    metadata,
                )
            )
        return chunks

    @staticmethod
    def _make_chunk(
        text: str,
        filename: str,
        page_number: int,
        chunk_index: int,
        metadata: Dict[str, Any],
    ) -> TextChunk:
        is_table = "|" in text and ("---|" in text or "|---" in text)
        return TextChunk(
            text=text,
            source_file=filename,
            page_number=page_number,
            chunk_index=chunk_index,
            content_type="table" if is_table else "text",
            metadata=dict(metadata),
        )
