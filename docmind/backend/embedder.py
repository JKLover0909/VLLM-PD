"""
embedder.py
-----------
Wrapper cho BAAI/bge-m3 embedding model.
bge-m3 hỗ trợ 100+ ngôn ngữ, bao gồm tiếng Việt, context lên đến 8192 tokens.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Union
import logging

logger = logging.getLogger(__name__)


class Embedder:
    MODEL_NAME = "BAAI/bge-m3"
    EMBEDDING_DIM = 1024  # bge-m3 output dimension

    def __init__(self, device: Optional[str] = None):
        """
        Khởi tạo bge-m3 model.
        Tự động dùng GPU nếu có.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Loading {self.MODEL_NAME} on {device}...")

        self.model = SentenceTransformer(
            self.MODEL_NAME,
            device=device,
        )
        logger.info("Embedder ready.")

    def embed(self, texts: Union[str, list[str]], batch_size: int = 32) -> np.ndarray:
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

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # L2 normalize → dot product = cosine similarity
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed một câu query, thêm instruction prefix của bge-m3.
        bge-m3 recommend dùng instruction prefix cho query.
        """
        instruction = "Represent this sentence for searching relevant passages: "
        return self.embed(instruction + query)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed danh sách documents (không cần prefix)."""
        return self.embed(texts)
