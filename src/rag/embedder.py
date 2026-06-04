"""
src/rag/embedder.py
-------------------
Wrapper cho BAAI/bge-m3 embedding model chạy cục bộ trên GPU của Máy 2.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class Embedder:
    MODEL_NAME = "BAAI/bge-m3"
    EMBEDDING_DIM = 1024  # bge-m3 output dimension

    def __init__(self, device: Optional[str] = None):
        """
        Khởi tạo bge-m3 model.
        Tự động chọn GPU nếu khả dụng.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Loading {self.MODEL_NAME} on device: {device}...")

        try:
            self.model = SentenceTransformer(
                self.MODEL_NAME,
                device=device,
            )
            logger.info("Embedder BGE-M3 loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise e

    def embed(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode text(s) thành vector embeddings.

        Args:
            texts: Chuỗi hoặc danh sách chuỗi cần embed.
            batch_size: Số lượng texts xử lý cùng lúc.

        Returns:
            numpy array shape (N, 1024) đã được normalize (L2).
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # L2 normalize -> dot product = cosine similarity
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed một câu query, trả về list float (tiêu chuẩn cho Qdrant).
        """
        # BGE-M3 khuyên dùng instruction prefix đối với truy vấn để đạt độ chính xác cao nhất
        # instruction = "Represent this sentence for searching relevant passages: "
        # Lưu ý: BGE-M3 có thể không cần prefix nếu là đa ngữ, nhưng có prefix sẽ cải thiện kết quả tìm kiếm ngữ nghĩa.
        prefix = ""  # Để mặc định rỗng hoặc tùy chỉnh nếu cần thiết
        vector = self.embed(prefix + query)[0]
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed danh sách documents, trả về danh sách các list float.
        """
        vectors = self.embed(texts)
        return vectors.tolist()
