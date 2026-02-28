"""
document_processor.py
----------------------
Xử lý tài liệu: PDF và ảnh (PNG/JPG).
- PDF: dùng PyMuPDF để extract text và embedded images
- Images: dùng EasyOCR để nhận diện text tiếng Việt + tiếng Anh
- Output: danh sách chunks có metadata (file, page, type)
"""

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import Counter

import fitz  # PyMuPDF
import easyocr
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Một đoạn text đã được extract và chunked."""
    text: str
    source_file: str
    page_number: int          # 0-indexed
    chunk_index: int
    content_type: str         # "text" | "ocr_image" | "ocr_page"
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"TextChunk(file='{self.source_file}', "
            f"page={self.page_number + 1}, "
            f"chars={len(self.text)})"
        )


class DocumentProcessor:
    """
    Xử lý tài liệu PDF và ảnh, trả về list TextChunk.
    """

    # EasyOCR reader — lazy init để tránh load model khi import
    _ocr_reader: Optional[easyocr.Reader] = None

    OCR_LANGUAGES = ["vi", "en"]  # Tiếng Việt + Tiếng Anh
    CHUNK_SIZE = 768              # tokens ≈ words (bge-m3 handle được 8192)
    CHUNK_OVERLAP = 96            # ~12% overlap để tránh mất context

    def __init__(self, use_gpu_ocr: bool = True):
        self.use_gpu_ocr = use_gpu_ocr

    @property
    def ocr_reader(self) -> easyocr.Reader:
        """Lazy load EasyOCR."""
        if DocumentProcessor._ocr_reader is None:
            logger.info(f"Loading EasyOCR (langs: {self.OCR_LANGUAGES})...")
            DocumentProcessor._ocr_reader = easyocr.Reader(
                self.OCR_LANGUAGES,
                gpu=self.use_gpu_ocr,
                verbose=False,
            )
            logger.info("EasyOCR ready.")
        return DocumentProcessor._ocr_reader

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def process_file(self, file_path: str | Path) -> list[TextChunk]:
        """
        Xử lý một file và trả về list chunks.
        Hỗ trợ: .pdf, .png, .jpg, .jpeg, .bmp, .tiff
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            raw_chunks = self._process_pdf(path)
        elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
            raw_chunks = self._process_image(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # Chunk text thành đoạn phù hợp
        chunks = []
        for rc in raw_chunks:
            sub_chunks = self._split_text(
                rc["text"],
                rc["source_file"],
                rc["page_number"],
                rc["content_type"],
            )
            chunks.extend(sub_chunks)

        logger.info(f"Processed '{path.name}': {len(chunks)} chunks")
        return chunks

    # ──────────────────────────────────────────────
    # PDF Processing
    # ──────────────────────────────────────────────

    def _process_pdf(self, path: Path) -> list[dict]:
        """Extract text và OCR từ PDF."""
        doc = fitz.open(str(path))
        raw = []

        for page_num, page in enumerate(doc):
            # 1. Extract native text
            text = page.get_text("text").strip()
            if len(text) > 50:  # có text thực sự
                raw.append({
                    "text": text,
                    "source_file": path.name,
                    "page_number": page_num,
                    "content_type": "text",
                })

            # 2. Extract embedded images và OCR
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img_np = np.array(img)

                    ocr_text = self._ocr_image_array(img_np)
                    color_text = self._describe_image_colors(img_np)
                    if ocr_text.strip():
                        raw.append({
                            "text": f"[Nội dung ảnh trang {page_num + 1}]\n{ocr_text}\n{color_text}",
                            "source_file": path.name,
                            "page_number": page_num,
                            "content_type": "ocr_image",
                        })
                    elif color_text.strip():
                        raw.append({
                            "text": f"[Ảnh trang {page_num + 1}]\n{color_text}",
                            "source_file": path.name,
                            "page_number": page_num,
                            "content_type": "ocr_image",
                        })
                except Exception as e:
                    logger.warning(f"OCR image error (page {page_num}, img {img_idx}): {e}")

            # 3. Nếu trang không có text → OCR toàn trang (scan PDF)
            if len(text) <= 50 and not image_list:
                try:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_np = np.array(img)
                    ocr_text = self._ocr_image_array(img_np)
                    if ocr_text.strip():
                        raw.append({
                            "text": f"[OCR trang {page_num + 1}]\n{ocr_text}",
                            "source_file": path.name,
                            "page_number": page_num,
                            "content_type": "ocr_page",
                        })
                except Exception as e:
                    logger.warning(f"Page OCR error (page {page_num}): {e}")

        doc.close()
        return raw

    # ──────────────────────────────────────────────
    # Image Processing
    # ──────────────────────────────────────────────

    def _process_image(self, path: Path) -> list[dict]:
        """OCR trực tiếp từ file ảnh."""
        img = Image.open(str(path)).convert("RGB")
        img_np = np.array(img)
        ocr_text = self._ocr_image_array(img_np)
        color_text = self._describe_image_colors(img_np)

        return [{
            "text": f"[Nội dung ảnh: {path.name}]\n{ocr_text}\n{color_text}",
            "source_file": path.name,
            "page_number": 0,
            "content_type": "ocr_image",
        }]

    def _ocr_image_array(self, img_np: np.ndarray) -> str:
        """Chạy EasyOCR trên numpy array, trả về text."""
        results = self.ocr_reader.readtext(img_np, detail=0, paragraph=True)
        return "\n".join(results)

    def _describe_image_colors(self, img_np: np.ndarray) -> str:
        """Mô tả màu chủ đạo từ ảnh để hỗ trợ câu hỏi thị giác cơ bản trong RAG."""
        if img_np.size == 0:
            return ""

        # Giảm kích thước để xử lý nhanh
        img_small = Image.fromarray(img_np).resize((128, 128))
        arr = np.array(img_small).reshape(-1, 3)

        # Lượng tử màu để gom nhóm gần nhau
        quantized = (arr // 32) * 32
        color_counts = Counter(map(tuple, quantized.tolist()))
        top_colors = color_counts.most_common(3)

        if not top_colors:
            return ""

        total = sum(count for _, count in top_colors)
        dominant_name = self._rgb_to_color_name(top_colors[0][0])

        parts = []
        for rgb, count in top_colors:
            ratio = (count / total) * 100 if total else 0
            name = self._rgb_to_color_name(rgb)
            parts.append(f"{name} ({ratio:.1f}%)")

        return (
            f"[Phân tích màu ảnh] Màu chủ đạo: {dominant_name}. "
            f"Top màu: {', '.join(parts)}."
        )

    def _rgb_to_color_name(self, rgb: tuple[int, int, int]) -> str:
        """Ánh xạ RGB gần đúng về tên màu tiếng Việt."""
        r, g, b = rgb
        palette = {
            "đỏ": (220, 40, 40),
            "cam": (235, 140, 40),
            "vàng": (230, 210, 50),
            "xanh lá": (60, 170, 70),
            "xanh dương": (50, 120, 220),
            "tím": (140, 80, 180),
            "hồng": (230, 120, 180),
            "nâu": (130, 90, 60),
            "đen": (25, 25, 25),
            "xám": (140, 140, 140),
            "trắng": (235, 235, 235),
        }

        def dist(c):
            return (r - c[0]) ** 2 + (g - c[1]) ** 2 + (b - c[2]) ** 2

        return min(palette, key=lambda name: dist(palette[name]))

    # ──────────────────────────────────────────────
    # Text Chunking
    # ──────────────────────────────────────────────

    def _split_text(
        self,
        text: str,
        source_file: str,
        page_number: int,
        content_type: str,
    ) -> list[TextChunk]:
        """
        Chia text thành chunks theo word count với overlap.
        Ưu tiên split theo paragraph trước.
        """
        # Tách theo paragraph
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_words: list[str] = []
        chunk_idx = 0

        for para in paragraphs:
            words = para.split()
            if len(current_words) + len(words) > self.CHUNK_SIZE:
                if current_words:
                    chunk_text = " ".join(current_words)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        source_file=source_file,
                        page_number=page_number,
                        chunk_index=chunk_idx,
                        content_type=content_type,
                    ))
                    chunk_idx += 1
                    # Overlap: giữ lại CHUNK_OVERLAP words cuối
                    current_words = current_words[-self.CHUNK_OVERLAP:] + words
                else:
                    current_words = words
            else:
                current_words.extend(words)

        # Chunk cuối
        if current_words:
            chunks.append(TextChunk(
                text=" ".join(current_words),
                source_file=source_file,
                page_number=page_number,
                chunk_index=chunk_idx,
                content_type=content_type,
            ))

        # Nếu text quá ngắn, vẫn tạo 1 chunk
        if not chunks and text.strip():
            chunks.append(TextChunk(
                text=text.strip(),
                source_file=source_file,
                page_number=page_number,
                chunk_index=0,
                content_type=content_type,
            ))

        return chunks
