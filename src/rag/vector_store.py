"""
src/rag/vector_store.py
-----------------------
Quản lý Vector Database sử dụng Qdrant thay thế cho FAISS.
Tận dụng tính năng Payload Filtering của Qdrant để tách biệt dữ liệu giữa các session của người dùng.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.rag.parser import TextChunk

logger = logging.getLogger(__name__)


class SearchResult:
    def __init__(self, chunk: TextChunk, score: float):
        self.chunk = chunk
        self.score = score

    def __repr__(self):
        return f"SearchResult(score={self.score:.4f}, chunk={self.chunk})"


class VectorStore:
    COLLECTION_NAME = "docmind_documents"
    EMBEDDING_DIM = 1024  # Kích thước vector của BGE-M3

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str | None = None,
    ):
        """
        Khởi tạo kết nối tới Qdrant Server.
        """
        if collection_name:
            self.COLLECTION_NAME = collection_name
        logger.info(
            "Connecting to Qdrant at %s:%s (collection=%s)...",
            host,
            port,
            self.COLLECTION_NAME,
        )
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Đảm bảo collection lưu trữ tài liệu đã được khởi tạo trong Qdrant.
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.COLLECTION_NAME not in collection_names:
                logger.info(f"Creating collection '{self.COLLECTION_NAME}' in Qdrant...")
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=self.EMBEDDING_DIM,
                        distance=models.Distance.COSINE
                    )
                )
                # Tạo index cho trường session_id để tối ưu hóa việc tìm kiếm lọc theo session
                self.client.create_payload_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name="session_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name="source_file",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info(f"Collection '{self.COLLECTION_NAME}' initialized successfully.")
            else:
                logger.info(f"Collection '{self.COLLECTION_NAME}' already exists.")
            # Index cho metadata.category phục vụ filter theo nhóm chủ đề
            # (Research/DocJP). Idempotent nên gọi cho cả collection đã tồn tại.
            try:
                self.client.create_payload_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name="metadata.category",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as index_exc:  # noqa: BLE001
                logger.debug(
                    "Payload index metadata.category not (re)created: %s", index_exc
                )
        except Exception as e:
            logger.error(f"Error checking/creating Qdrant collection: {e}")
            raise e

    def create_session(self, session_id: str) -> None:
        """
        Khởi tạo session mới. Đối với Qdrant, ta không cần khởi tạo thực tế gì đặc biệt
        vì các points được nhóm qua payload filtering bằng `session_id`.
        """
        logger.info(f"Session '{session_id}' registered in Qdrant context.")

    def delete_session(self, session_id: str) -> None:
        """
        Xóa mọi dữ liệu vector thuộc về session_id này.
        """
        try:
            logger.info(f"Deleting all vectors for session: {session_id}")
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="session_id",
                                match=models.MatchValue(value=session_id)
                            )
                        ]
                    )
                )
            )
        except Exception as e:
            logger.error(f"Error deleting session '{session_id}' in Qdrant: {e}")

    def add_chunks(self, session_id: str, chunks: List[TextChunk], embeddings: List[List[float]]) -> None:
        """
        Thêm danh sách TextChunk kèm vector tương ứng vào Qdrant.
        """
        if not chunks:
            return

        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                "session_id": session_id,
                "text": chunk.text,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "content_type": chunk.content_type,
                "metadata": chunk.metadata
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )

        logger.info(f"Uploading {len(points)} points to Qdrant for session '{session_id}'...")
        try:
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                wait=True,
                points=points
            )
            logger.info(f"Successfully indexed {len(points)} chunks for session '{session_id}'.")
        except Exception as e:
            logger.error(f"Error adding chunks to Qdrant: {e}")
            raise e

    def remove_file(self, session_id: str, filename: str) -> int:
        """
        Xóa tất cả các chunks thuộc về một file cụ thể trong một session.
        Qdrant hỗ trợ xóa trực tiếp thông qua payload filter mà không cần rebuild index.
        """
        try:
            # Trước tiên đếm xem có bao nhiêu chunks sẽ bị xóa để báo lại cho API
            count_res = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id)),
                        models.FieldCondition(key="source_file", match=models.MatchValue(value=filename))
                    ]
                ),
                limit=1000,
                with_payload=False,
                with_vectors=False
            )
            count = len(count_res[0])

            if count > 0:
                logger.info(f"Deleting {count} chunks for file '{filename}' in session '{session_id}'...")
                self.client.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id)),
                                models.FieldCondition(key="source_file", match=models.MatchValue(value=filename))
                            ]
                        )
                    )
                )
            return count
        except Exception as e:
            logger.error(f"Error removing file '{filename}' in session '{session_id}': {e}")
            return 0

    @staticmethod
    def _build_filter(
        session_id: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> models.Filter:
        """Build a session filter, optionally narrowed by metadata fields.

        ``metadata_filters`` keys address nested payload fields under
        ``metadata`` (e.g. ``{"category": "accounting"}`` filters on
        ``metadata.category``).
        """
        must: List[models.FieldCondition] = [
            models.FieldCondition(
                key="session_id",
                match=models.MatchValue(value=session_id),
            )
        ]
        for key, value in (metadata_filters or {}).items():
            if value is None:
                continue
            must.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value),
                )
            )
        return models.Filter(must=must)

    def search(
        self,
        session_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.3,
        *,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Tìm kiếm ngữ nghĩa (cosine similarity) lọc theo session_id.
        """
        try:
            # Query Qdrant. qdrant-client >= 1.10 dùng query_points; bản cũ
            # (1.9, đang chạy trong Docker image) chỉ có search().
            query_filter = self._build_filter(session_id, metadata_filters)
            if hasattr(self.client, "query_points"):
                hits = self.client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    query=query_embedding,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                ).points
            else:
                hits = self.client.search(
                    collection_name=self.COLLECTION_NAME,
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold
                )

            results = []
            for hit in hits:
                payload = hit.payload
                chunk = TextChunk(
                    text=payload.get("text", ""),
                    source_file=payload.get("source_file", ""),
                    page_number=payload.get("page_number", 1),
                    chunk_index=payload.get("chunk_index", 0),
                    content_type=payload.get("content_type", "text"),
                    metadata=payload.get("metadata", {})
                )
                results.append(SearchResult(chunk=chunk, score=hit.score))

            return results
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []

    def get_session_info(
        self,
        session_id: str,
        *,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin tổng hợp về session từ database.
        """
        try:
            # Scroll phân trang: một số session (vd. DocJP) có hơn 1000 chunks
            # nên một lượt scroll giới hạn 1000 sẽ đếm thiếu.
            scroll_filter = self._build_filter(session_id, metadata_filters)
            num_chunks = 0
            files: set[str] = set()
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.COLLECTION_NAME,
                    scroll_filter=scroll_filter,
                    limit=1000,
                    offset=offset,
                    with_payload=["source_file"],
                    with_vectors=False,
                )
                num_chunks += len(points)
                for p in points:
                    source_file = (p.payload or {}).get("source_file")
                    if source_file:
                        files.add(source_file)
                if offset is None:
                    break
            if num_chunks == 0:
                return None

            return {
                "session_id": session_id,
                "num_chunks": num_chunks,
                "files": sorted(files),
                "num_files": len(files)
            }
        except Exception as e:
            logger.error(f"Error getting session info from Qdrant: {e}")
            return None

    def get_file_metadata(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        """Return one metadata payload per indexed source file."""
        result: Dict[str, Dict[str, Any]] = {}
        offset = None
        try:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.COLLECTION_NAME,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="session_id",
                                match=models.MatchValue(value=session_id),
                            )
                        ]
                    ),
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    filename = payload.get("source_file")
                    if filename and filename not in result:
                        result[filename] = payload.get("metadata", {})
                if offset is None:
                    break
        except Exception as exc:
            logger.error("Error reading indexed file metadata: %s", exc)
        return result

    def get_page_image_path(
        self,
        session_id: str,
        filename: str,
        page_number: int,
    ) -> Optional[str]:
        """Return the indexed preview image path for one source page."""
        try:
            points, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=session_id),
                        ),
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=filename),
                        ),
                        models.FieldCondition(
                            key="page_number",
                            match=models.MatchValue(value=page_number),
                        ),
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                metadata = (point.payload or {}).get("metadata") or {}
                image_path = metadata.get("image_path")
                if image_path:
                    return str(image_path)
        except Exception as exc:
            logger.error(
                "Error reading page preview image for %s page %s in session %s: %s",
                filename,
                page_number,
                session_id,
                exc,
            )
        return None
