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

    def _split_markdown_table(
        self,
        body: str,
        *,
        filename: str,
        page_number: int,
        start_index: int,
        metadata: Dict[str, Any],
    ) -> List[TextChunk]:
        table_rows = [
            line
            for line in body.splitlines()
            if self.MARKDOWN_TABLE_ROW.match(line)
        ]
        # Preamble text (title, notes) that is not part of the table itself.
        preamble = "\n".join(
            line
            for line in body.splitlines()
            if line.strip() and not self.MARKDOWN_TABLE_ROW.match(line)
        ).strip()
        header = table_rows[:2]  # column header + the |---| separator row
        data_rows = table_rows[2:]

        chunks: List[TextChunk] = []
        chunk_index = start_index
        for start in range(0, len(data_rows), self.TABLE_ROWS_PER_CHUNK):
            block_rows = data_rows[start:start + self.TABLE_ROWS_PER_CHUNK]
            parts = header + block_rows
            # Repeat the preamble on the first chunk so document context (title
            # of the table) stays attached without bloating every chunk.
            block = "\n".join(parts)
            if start == 0 and preamble:
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
        chunks = self._split_text(
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

        lines = cleaned.splitlines()
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
