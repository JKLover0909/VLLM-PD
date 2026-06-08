"""
src/rag/parser.py
-----------------
Xử lý tài liệu đa định dạng (PDF, DOCX, XLSX, hình ảnh) sử dụng Docling (IBM).
Đầu ra là danh sách các TextChunk có giữ nguyên cấu trúc Markdown (đặc biệt là bảng biểu).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class TextChunk:
    """Đoạn văn bản sau khi được trích xuất và phân nhỏ (chunked)."""
    text: str
    source_file: str
    page_number: int          # 1-indexed (bắt đầu từ 1)
    chunk_index: int
    content_type: str         # "text" | "table" | "mixed"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"TextChunk(file='{self.source_file}', "
            f"page={self.page_number}, "
            f"type='{self.content_type}', "
            f"chars={len(self.text)})"
        )


class DocumentParser:
    """
    Sử dụng Docling để chuyển đổi tài liệu sang cấu trúc Markdown,
    sau đó phân đoạn (chunking) một cách thông minh để giữ ngữ cảnh.
    """

    CHUNK_SIZE = 1000  # Ký tự (khoảng 200-250 từ)
    CHUNK_OVERLAP = 200

    def __init__(self):
        logger.info("Initializing Docling DocumentConverter...")
        # Docling tự động tải các mô hình layout và TableFormer trong lần chạy đầu tiên
        self.converter = DocumentConverter()
        logger.info("Docling DocumentConverter initialized.")

    def process_file(self, file_path: str | Path) -> List[TextChunk]:
        """
        Xử lý tệp bất kỳ hỗ trợ bởi Docling và trả về danh sách TextChunk.
        Hỗ trợ: PDF, DOCX, XLSX, PPTX, HTML, PNG, JPG, JPEG, v.v.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Converting document '{path.name}' using Docling...")
        try:
            conversion_result = self.converter.convert(path)
            # Trích xuất dạng markdown
            doc = conversion_result.document
            markdown_content = doc.export_to_markdown()
        except Exception as e:
            logger.error(f"Error converting document {path.name} with Docling: {e}")
            raise e

        # Tiến hành phân chunk dựa trên nội dung Markdown
        chunks = self._split_markdown(markdown_content, path.name, doc)
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            for chunk in chunks:
                chunk.content_type = "image"
                chunk.metadata.update(
                    {
                        "source": "docling",
                        "image_path": str(path.resolve()),
                    }
                )
        logger.info(f"Document '{path.name}' processed into {len(chunks)} chunks.")
        return chunks

    def _split_markdown(self, markdown_text: str, filename: str, doc: Any) -> List[TextChunk]:
        """
        Phân mảnh nội dung Markdown thành các khối (chunks) hợp lý.
        Nếu gặp bảng biểu hoặc khối dữ liệu lớn, cố gắng giữ nguyên trong cùng một chunk.
        """
        # Trích xuất các phân đoạn trang nếu có
        # Đối với các tài liệu nhiều trang như PDF, Docling lưu trữ vị trí trang
        # Ta tạm thời chia văn bản thành các dòng và nhóm lại
        lines = markdown_text.split("\n")
        
        chunks = []
        current_chunk_lines = []
        current_char_count = 0
        chunk_idx = 0
        current_page = 1

        # Cố gắng phát hiện số trang bằng các thẻ trang hoặc phân trang từ văn bản nếu có.
        # Hoặc chia đều. Đối với Docling, ta cũng có thể truy xuất trang thông qua phần tử (elements).
        # Cách đơn giản nhưng hiệu quả là parse theo dòng và kiểm soát độ dài.
        
        for line in lines:
            line_len = len(line)
            
            # Cố gắng cập nhật trang hiện tại nếu thấy chỉ báo trang hoặc giữ mặc định
            # (Thường trong markdown của Docling không tự sinh chỉ báo trang rõ ràng trừ khi ta tìm trong cấu trúc json)
            # Tạm thời để page = 1 và cập nhật nếu cần hoặc xử lý đơn giản.
            
            if current_char_count + line_len > self.CHUNK_SIZE and current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines)
                
                # Xác định xem chunk này chứa bảng biểu hay không
                content_type = "text"
                if "|" in chunk_text and "-|-" in chunk_text or "---|" in chunk_text:
                    content_type = "table"

                chunks.append(TextChunk(
                    text=chunk_text,
                    source_file=filename,
                    page_number=current_page,
                    chunk_index=chunk_idx,
                    content_type=content_type,
                    metadata={"source": "docling"}
                ))
                chunk_idx += 1
                
                # Giữ overlap: lấy khoảng 20% số dòng cuối cùng
                overlap_lines_count = max(1, len(current_chunk_lines) // 5)
                current_chunk_lines = current_chunk_lines[-overlap_lines_count:]
                current_char_count = sum(len(l) for l in current_chunk_lines)
            
            current_chunk_lines.append(line)
            current_char_count += line_len + 1  # Cộng thêm 1 cho ký tự newline

        # Chunk cuối cùng
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            content_type = "text"
            if "|" in chunk_text and "-|-" in chunk_text or "---|" in chunk_text:
                content_type = "table"
                
            chunks.append(TextChunk(
                text=chunk_text,
                source_file=filename,
                page_number=current_page,
                chunk_index=chunk_idx,
                content_type=content_type,
                metadata={"source": "docling"}
            ))

        return chunks
