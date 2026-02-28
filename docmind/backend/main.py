"""
main.py
-------
FastAPI backend cho DocMind.
Cung cấp REST API để:
- Tạo/quản lý sessions
- Upload và index tài liệu
- Hỏi đáp qua RAG pipeline (streaming và non-streaming)
"""

import asyncio
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from document_processor import DocumentProcessor
from embedder import Embedder
from rag_pipeline import RAGPipeline
from vector_store import VectorStore
from vllm_client import VLLMClient

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "qwen2.5-vl-3b")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

UPLOAD_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Global singletons (khởi tạo khi app start)
# ──────────────────────────────────────────────
embedder: Optional[Embedder] = None
vector_store: Optional[VectorStore] = None
doc_processor: Optional[DocumentProcessor] = None
vllm_client: Optional[VLLMClient] = None
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo models khi app start, cleanup khi shutdown."""
    global embedder, vector_store, doc_processor, vllm_client, rag_pipeline

    logger.info("🚀 Starting DocMind backend...")
    embedder = Embedder()
    vector_store = VectorStore()
    doc_processor = DocumentProcessor()
    vllm_client = VLLMClient(base_url=VLLM_URL, model=VLLM_MODEL)
    rag_pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        vllm_client=vllm_client,
    )

    # Cleanup task: xóa session hết hạn mỗi 10 phút
    async def cleanup_loop():
        while True:
            await asyncio.sleep(600)
            n = vector_store.cleanup_expired()
            if n > 0:
                logger.info(f"Cleaned up {n} expired sessions")

    cleanup_task = asyncio.create_task(cleanup_loop())
    logger.info("✅ DocMind backend ready.")

    yield

    cleanup_task.cancel()
    logger.info("DocMind backend shutdown.")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(
    title="DocMind API",
    description="RAG-powered document Q&A with Vietnamese support",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class SessionResponse(BaseModel):
    session_id: str
    message: str


class QueryRequest(BaseModel):
    session_id: str
    question: str
    stream: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str


class SessionInfoResponse(BaseModel):
    session_id: str
    num_chunks: int
    files: list[str]
    num_files: int


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    vllm_ok = await vllm_client.health_check()
    return {
        "status": "ok",
        "vllm_connected": vllm_ok,
        "sessions": len(vector_store.list_sessions()),
    }


@app.post("/sessions", response_model=SessionResponse)
async def create_session():
    """Tạo session mới, trả về session_id."""
    session_id = str(uuid.uuid4())
    vector_store.create_session(session_id)
    return SessionResponse(
        session_id=session_id,
        message="Session created successfully",
    )


@app.get("/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str):
    """Thông tin về session."""
    info = vector_store.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionInfoResponse(**info)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Xóa session."""
    if not vector_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    vector_store.delete_session(session_id)
    # Xóa files đã upload của session
    session_dir = UPLOAD_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
    return {"message": f"Session {session_id} deleted"}


@app.post("/sessions/{session_id}/upload")
async def upload_document(
    session_id: str,
    file: UploadFile = File(...),
):
    """
    Upload và index tài liệu vào session.
    Hỗ trợ: PDF, PNG, JPG, JPEG, BMP, TIFF, WEBP.
    """
    if not vector_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate file type
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed}",
        )

    # Lưu file
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    file_path = session_dir / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    logger.info(f"Uploaded '{file.filename}' ({len(content)/1024:.1f} KB) to session {session_id}")

    try:
        # Process document → chunks
        chunks = doc_processor.process_file(file_path)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from this file.",
            )

        # Embed chunks
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_documents(texts)

        # Index vào FAISS
        vector_store.add_chunks(session_id, chunks, embeddings)

        return {
            "filename": file.filename,
            "num_chunks": len(chunks),
            "file_size_kb": round(len(content) / 1024, 1),
            "message": f"Successfully indexed {len(chunks)} chunks",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """
    Hỏi đáp qua RAG pipeline (non-streaming).
    """
    if not vector_store.session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    info = vector_store.get_session_info(req.session_id)
    if info["num_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed in this session. Please upload files first.",
        )

    try:
        answer, results = await rag_pipeline.query(
            session_id=req.session_id,
            question=req.question,
        )
        return QueryResponse(
            answer=answer,
            sources=rag_pipeline.format_sources(results),
            session_id=req.session_id,
        )
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    Hỏi đáp qua RAG pipeline (streaming SSE).
    Trả về Server-Sent Events: tokens được stream dần.
    """
    if not vector_store.session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    info = vector_store.get_session_info(req.session_id)
    if info["num_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Please upload files first.",
        )

    async def event_generator():
        import json
        try:
            token_stream, results = await rag_pipeline.query_stream(
                session_id=req.session_id,
                question=req.question,
            )
            # Gửi sources trước
            sources = rag_pipeline.format_sources(results)
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Stream tokens
            async for token in token_stream:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Done
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/sessions/{session_id}/files/{filename}")
async def remove_file(session_id: str, filename: str):
    """Xóa một file khỏi session và xóa khỏi disk."""
    if not vector_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    removed = vector_store.remove_file(session_id, filename)

    # Xóa file trên disk
    file_path = UPLOAD_DIR / session_id / filename
    if file_path.exists():
        file_path.unlink()

    return {
        "message": f"Removed '{filename}' ({removed} chunks) from session",
        "chunks_removed": removed,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
