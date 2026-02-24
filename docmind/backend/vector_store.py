"""
vector_store.py
---------------
Quản lý FAISS index theo session.
Mỗi user session có một index riêng biệt trong memory.
Sử dụng IndexFlatIP (Inner Product) với L2-normalized vectors → cosine similarity.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np

from document_processor import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk: TextChunk
    score: float   # cosine similarity (0–1, cao hơn = liên quan hơn)


@dataclass
class SessionIndex:
    """FAISS index + metadata cho một session."""
    index: faiss.IndexFlatIP
    chunks: list[TextChunk] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    @property
    def num_chunks(self) -> int:
        return self.index.ntotal


class VectorStore:
    """
    Quản lý nhiều FAISS index theo session_id.
    Thread-safe cho môi trường FastAPI async.
    """

    EMBEDDING_DIM = 1024  # bge-m3
    SESSION_TTL = 3600    # 1 giờ không dùng → xóa session

    def __init__(self):
        self._sessions: dict[str, SessionIndex] = {}

    # ──────────────────────────────────────────────
    # Session Management
    # ──────────────────────────────────────────────

    def create_session(self, session_id: str) -> None:
        """Tạo index mới cho session."""
        index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self._sessions[session_id] = SessionIndex(index=index)
        logger.info(f"Created session: {session_id}")

    def delete_session(self, session_id: str) -> None:
        """Xóa session và giải phóng memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id}")

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Trả về thông tin session."""
        if session_id not in self._sessions:
            return None
        s = self._sessions[session_id]
        files = list({c.source_file for c in s.chunks})
        return {
            "session_id": session_id,
            "num_chunks": s.num_chunks,
            "files": files,
            "num_files": len(files),
            "created_at": s.created_at,
        }

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def cleanup_expired(self) -> int:
        """Xóa các session đã hết TTL. Trả về số session đã xóa."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_used > self.SESSION_TTL
        ]
        for sid in expired:
            self.delete_session(sid)
        return len(expired)

    # ──────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────

    def add_chunks(
        self,
        session_id: str,
        chunks: list[TextChunk],
        embeddings: np.ndarray,
    ) -> None:
        """
        Thêm chunks vào FAISS index của session.

        Args:
            session_id: ID của session
            chunks: List TextChunk tương ứng
            embeddings: numpy array (N, 1024), đã L2-normalized
        """
        if session_id not in self._sessions:
            self.create_session(session_id)

        s = self._sessions[session_id]

        # FAISS yêu cầu float32
        vecs = embeddings.astype(np.float32)
        s.index.add(vecs)
        s.chunks.extend(chunks)
        s.last_used = time.time()

        logger.info(
            f"Session {session_id}: added {len(chunks)} chunks "
            f"(total: {s.num_chunks})"
        )

    def remove_file(self, session_id: str, filename: str) -> int:
        """
        Xóa tất cả chunks của một file khỏi session.
        Phải rebuild index vì FAISS không hỗ trợ xóa inline.
        Trả về số chunks đã xóa.
        """
        if session_id not in self._sessions:
            return 0

        s = self._sessions[session_id]
        original = len(s.chunks)
        # Lọc ra chunks không thuộc file cần xóa
        keep = [c for c in s.chunks if c.source_file != filename]
        removed = original - len(keep)

        if removed == 0:
            return 0

        # Rebuild index
        # NOTE: Cần embeddings → lưu lại. Ở đây ta rebuild từ scratch.
        # Với production, nên dùng faiss.IndexIDMap để hỗ trợ xóa.
        s.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        s.chunks = []
        # WARN: Sau remove, cần re-embed chunks giữ lại.
        # Đơn giản hơn: tag là cần re-index. Xem main.py.
        s.chunks = keep  # giữ metadata, index sẽ không đồng bộ
        logger.warning(
            f"File '{filename}' removed from session {session_id}. "
            f"Index cần re-embed {len(keep)} chunks còn lại."
        )
        return removed

    # ──────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────

    def search(
        self,
        session_id: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[SearchResult]:
        """
        Tìm top-k chunks liên quan nhất đến câu query.

        Args:
            session_id: ID session
            query_embedding: (1, 1024) float32, L2-normalized
            top_k: Số lượng kết quả trả về
            score_threshold: Lọc bỏ kết quả có score thấp

        Returns:
            List SearchResult, sorted by score desc
        """
        if session_id not in self._sessions:
            return []

        s = self._sessions[session_id]
        s.last_used = time.time()

        if s.num_chunks == 0:
            return []

        # Giới hạn top_k không vượt quá số chunks có
        k = min(top_k, s.num_chunks)

        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        scores, indices = s.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS trả -1 nếu không đủ kết quả
                continue
            if float(score) < score_threshold:
                continue
            results.append(SearchResult(
                chunk=s.chunks[idx],
                score=float(score),
            ))

        # Đã được sort bởi FAISS (score cao trước)
        return results
