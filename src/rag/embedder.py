"""
src/rag/embedder.py
-------------------
Wrapper cho BAAI/bge-m3 embedding model chạy cục bộ trên GPU của Máy 2.
"""

import logging
import os
import threading
from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    MODEL_NAME = "BAAI/bge-m3"
    EMBEDDING_DIM = 1024  # bge-m3 output dimension

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        dtype: Optional[str] = None,
    ):
        """
        Khởi tạo bge-m3 model.
        Tự động chọn GPU nếu khả dụng.
        """
        if device is None:
            device = os.getenv(
                "EMBEDDING_DEVICE",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA is unavailable; falling back to CPU embeddings.")
            device = "cpu"

        self.device = device
        self.batch_size = batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
        self.dtype = (
            dtype
            or os.getenv(
                "EMBEDDING_DTYPE",
                "float16" if device.startswith("cuda") else "float32",
            )
        ).lower()
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "EMBEDDING_DTYPE must be float16, bfloat16, or float32."
            )
        if not device.startswith("cuda") and self.dtype == "float16":
            logger.warning("float16 embeddings are not suitable for CPU; using float32.")
            self.dtype = "float32"

        self._encode_lock = threading.Lock()
        model_kwargs = {"dtype": self.dtype}
        logger.info(
            "Loading %s on device=%s dtype=%s batch_size=%s...",
            self.MODEL_NAME,
            self.device,
            self.dtype,
            self.batch_size,
        )

        try:
            self.model = SentenceTransformer(
                self.MODEL_NAME,
                device=self.device,
                model_kwargs=model_kwargs,
            )
            logger.info("Embedder BGE-M3 loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
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

        # SentenceTransformer/PyTorch inference is not guaranteed to be safe when
        # several request threads share one CUDA model. Serializing encode calls
        # also prevents concurrent batches from producing a VRAM spike.
        with self._encode_lock, torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size or self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
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
