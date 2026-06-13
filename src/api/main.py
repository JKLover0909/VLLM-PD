"""
src/api/main.py
---------------
FastAPI API Gateway chính cho Máy 2.
Quản lý các REST API phục vụ cho RAG (Upload, Index, Query) và AI Agent (LangGraph Execution).
"""

import asyncio
import json
import logging
import os
import secrets
import shutil
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load biến môi trường từ .env
load_dotenv()

from src.rag.parser import (
    DocumentLimitError,
    DocumentParser,
    DocumentProcessingTimeout,
)
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.rag_pipeline import RAGPipeline
from src.rag.web_search import WebSearcher

ENABLE_AGENT = os.getenv("ENABLE_AGENT", "true").lower() in {"1", "true", "yes", "on"}
agent_executor = None
if ENABLE_AGENT:
    from src.agent.graph import agent_executor

# ──────────────────────────────────────────────
# Configurations
# ──────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000/v1")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-local")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
QUERY_RATE_LIMIT = int(os.getenv("QUERY_RATE_LIMIT_PER_MINUTE", "15"))
UPLOAD_RATE_LIMIT = int(os.getenv("UPLOAD_RATE_LIMIT_PER_HOUR", "10"))
UPLOAD_PROCESSING_CONCURRENCY = max(
    1, int(os.getenv("UPLOAD_PROCESSING_CONCURRENCY", "1"))
)
UPLOAD_QUEUE_SIZE = max(0, int(os.getenv("UPLOAD_QUEUE_SIZE", "4")))
ALLOWED_UPLOAD_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
}
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

UPLOAD_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Global Singletons
# ──────────────────────────────────────────────
embedder: Optional[Embedder] = None
vector_store: Optional[VectorStore] = None
mkac_vector_store: Optional[VectorStore] = None
doc_parser: Optional[DocumentParser] = None
rag_pipeline: Optional[RAGPipeline] = None
web_searcher: Optional[WebSearcher] = None
rate_limit_events: Dict[str, Deque[float]] = defaultdict(deque)
rate_limit_lock = asyncio.Lock()
upload_processing_semaphore = asyncio.Semaphore(UPLOAD_PROCESSING_CONCURRENCY)
upload_admission_lock = asyncio.Lock()
upload_active = 0
upload_waiting = 0


def normalize_session_id(session_id: str) -> str:
    """Validate and normalize a public session identifier."""
    try:
        return str(uuid.UUID(session_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid session ID.") from exc


def session_upload_dir(session_id: str) -> Path:
    """Return a filesystem-safe upload directory for a UUID session."""
    return UPLOAD_DIR / normalize_session_id(session_id)


def safe_upload_filename(filename: Optional[str]) -> str:
    """Reject path traversal and unsupported public uploads."""
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    safe_name = Path(filename).name
    if safe_name != filename or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if Path(safe_name).suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed extensions: {allowed}",
        )
    return safe_name


def cleanup_failed_upload(file_path: Path, page_dir: Path) -> None:
    """Remove partial upload artifacts after parsing or indexing fails."""
    if file_path.exists():
        file_path.unlink()
    if page_dir.exists():
        shutil.rmtree(page_dir)


def client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


async def enforce_rate_limit(
    request: Request,
    bucket: str,
    limit: int,
    window_seconds: int,
) -> None:
    """Apply a small in-memory limit suitable for this single API process."""
    if limit <= 0:
        return

    now = time.monotonic()
    key = f"{bucket}:{client_ip(request)}"
    async with rate_limit_lock:
        events = rate_limit_events[key]
        while events and now - events[0] >= window_seconds:
            events.popleft()
        if len(events) >= limit:
            retry_after = max(1, int(window_seconds - (now - events[0])))
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later.",
                headers={"Retry-After": str(retry_after)},
            )
        events.append(now)


@asynccontextmanager
async def upload_processing_slot():
    """Bound expensive parsing/indexing work and reject an overloaded queue."""
    global upload_active, upload_waiting

    async with upload_admission_lock:
        capacity = UPLOAD_PROCESSING_CONCURRENCY + UPLOAD_QUEUE_SIZE
        if upload_active + upload_waiting >= capacity:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The document processing queue is full. "
                    "Please retry in a few minutes."
                ),
                headers={"Retry-After": "30"},
            )
        upload_waiting += 1

    acquired = False
    try:
        await upload_processing_semaphore.acquire()
        acquired = True
        async with upload_admission_lock:
            upload_waiting -= 1
            upload_active += 1
        yield
    except BaseException:
        if not acquired:
            async with upload_admission_lock:
                upload_waiting -= 1
        raise
    finally:
        if acquired:
            async with upload_admission_lock:
                upload_active -= 1
            upload_processing_semaphore.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Khởi tạo các mô hình và kết nối database khi ứng dụng bắt đầu.
    """
    global embedder, vector_store, mkac_vector_store, doc_parser, rag_pipeline
    global web_searcher

    logger.info("🚀 Starting VLLM-PD API Gateway on Machine 2...")
    
    # Khởi tạo Vector DB trước
    vector_store = VectorStore(host=QDRANT_HOST, port=QDRANT_PORT)
    mkac_vector_store = VectorStore(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=os.getenv("MKAC_COLLECTION_NAME", "mkac_knowledge"),
    )
    
    # Khởi tạo local embedder (BGE-M3)
    embedder = Embedder()
    
    # Khởi tạo bộ phân tích tài liệu Docling
    doc_parser = DocumentParser()
    web_searcher = WebSearcher()
    
    # Khởi tạo RAG Pipeline kết nối với LiteLLM
    rag_pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        mkac_vector_store=mkac_vector_store,
        web_searcher=web_searcher,
    )

    logger.info("✅ VLLM-PD API Gateway is fully operational.")
    yield
    logger.info("Shutdown completed.")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="VLLM-PD API Gateway",
    description="RAG & Agent Server cho Máy 2",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────
class SessionResponse(BaseModel):
    session_id: str
    message: str


class QueryRequest(BaseModel):
    session_id: str
    question: str
    stream: bool = True
    model: Literal["auto", "local", "mimo", "openai", "grok"] = "auto"
    mode: Literal["mkac", "research"] = "mkac"


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    model: str
    mode: str
    answer_scope: str


class SessionInfoResponse(BaseModel):
    session_id: str
    num_chunks: int
    files: List[str]
    num_files: int


class AgentRequest(BaseModel):
    session_id: str
    task: str


class AgentResponse(BaseModel):
    session_id: str
    status: str
    output: str
    steps: List[Dict[str, Any]]


def message_text(message: Any) -> str:
    """Normalize LangChain message content into client-friendly text."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
                return parsed["content"]
        except json.JSONDecodeError:
            pass
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


# ──────────────────────────────────────────────
# RAG Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    mkac_info = (
        mkac_vector_store.get_session_info("mkac")
        if mkac_vector_store is not None
        else None
    )
    return {
        "status": "healthy",
        "qdrant_host": QDRANT_HOST,
        "qdrant_port": QDRANT_PORT,
        "mkac_documents": (mkac_info or {}).get("num_files", 0),
        "mkac_chunks": (mkac_info or {}).get("num_chunks", 0),
        "mkac_web_search": bool(web_searcher and web_searcher.enabled),
        "document_processing": {
            "active": upload_active,
            "waiting": upload_waiting,
            "concurrency": UPLOAD_PROCESSING_CONCURRENCY,
            "queue_size": UPLOAD_QUEUE_SIZE,
            "embedding_device": getattr(embedder, "device", None),
            "embedding_dtype": getattr(embedder, "dtype", None),
            "ocr_device": getattr(doc_parser, "ocr_device", None),
        },
    }


@app.get("/knowledge/mkac/status")
async def mkac_knowledge_status():
    """Return the shared MKAC knowledge-base indexing status."""
    info = (
        mkac_vector_store.get_session_info("mkac")
        if mkac_vector_store is not None
        else None
    )
    return {
        "ready": bool(info),
        "collection": os.getenv("MKAC_COLLECTION_NAME", "mkac_knowledge"),
        "num_documents": (info or {}).get("num_files", 0),
        "num_chunks": (info or {}).get("num_chunks", 0),
        "files": sorted((info or {}).get("files", [])),
    }


@app.get("/models")
async def list_models():
    """Danh sách model người dùng có thể chọn trên frontend."""
    return {
        "default": "auto",
        "models": [
            {
                "id": "auto",
                "name": "Tự động",
                "description": "Ưu tiên MiMo Pro, fallback sang OpenAI rồi Gemma4 local.",
            },
            {
                "id": "local",
                "name": "Gemma4 Local",
                "description": "Chạy trên Máy 1 cho tài liệu text; session có ảnh tự chuyển sang OpenAI Vision.",
            },
            {
                "id": "mimo",
                "name": "MiMo 2.5 Pro",
                "description": "Phù hợp tổng hợp tài liệu text; session có ảnh tự chuyển sang OpenAI Vision.",
            },
            {
                "id": "openai",
                "name": "OpenAI GPT-5.4 mini",
                "description": "Dùng GPT-5.4 mini cho coding, tool calling và truy vấn khó.",
            },
            {
                "id": "grok",
                "name": "Grok 4.20 Reasoning",
                "description": "Suy luận chuyên sâu qua Azure; session có ảnh tự chuyển sang OpenAI Vision.",
            },
        ],
    }


@app.post("/sessions", response_model=SessionResponse)
async def create_session():
    """Tạo session mới."""
    session_id = str(uuid.uuid4())
    vector_store.create_session(session_id)
    return SessionResponse(
        session_id=session_id,
        message="Session created successfully in Qdrant context.",
    )


@app.get("/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str):
    """Lấy thông tin chi tiết về session (số lượng files, chunks)."""
    session_id = normalize_session_id(session_id)
    info = vector_store.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found or empty.")
    return SessionInfoResponse(**info)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Xóa session và mọi tài liệu liên quan."""
    session_id = normalize_session_id(session_id)
    session_dir = session_upload_dir(session_id)
    vector_store.delete_session(session_id)
    
    # Xóa file vật lý lưu trên server
    if session_dir.exists():
        shutil.rmtree(session_dir)
        
    return {"message": f"Session {session_id} and its files have been deleted."}


@app.post("/sessions/{session_id}/upload")
async def upload_document(
    session_id: str,
    request: Request,
    file: UploadFile = File(...),
):
    """
    Tải tài liệu lên và lưu trữ dạng vector trong Qdrant.
    Được nâng cấp sử dụng Docling.
    """
    await enforce_rate_limit(request, "upload", UPLOAD_RATE_LIMIT, 3600)
    session_id = normalize_session_id(session_id)
    filename = safe_upload_filename(file.filename)
    session_dir = session_upload_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / filename
    page_dir = session_dir / "_pages" / file_path.stem
    uploaded_bytes = 0

    try:
        with file_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                uploaded_bytes += len(chunk)
                if uploaded_bytes > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB upload limit.",
                    )
                destination.write(chunk)

        logger.info(f"Uploaded file '{filename}' to session '{session_id}'")

        async with upload_processing_slot():
            chunks = await asyncio.to_thread(
                doc_parser.process_file,
                file_path,
                image_output_dir=page_dir,
            )
            if not chunks:
                raise HTTPException(
                    status_code=422,
                    detail="Could not extract any content from the file.",
                )

            texts = [c.text for c in chunks]
            embeddings = await asyncio.to_thread(embedder.embed_documents, texts)

            await asyncio.to_thread(vector_store.remove_file, session_id, filename)
            await asyncio.to_thread(
                vector_store.add_chunks,
                session_id,
                chunks,
                embeddings,
            )

        return {
            "filename": filename,
            "num_chunks": len(chunks),
            "file_size_kb": round(uploaded_bytes / 1024, 1),
            "message": f"Successfully parsed and indexed {len(chunks)} chunks into Qdrant."
        }

    except DocumentLimitError as e:
        cleanup_failed_upload(file_path, page_dir)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except DocumentProcessingTimeout as e:
        cleanup_failed_upload(file_path, page_dir)
        raise HTTPException(status_code=408, detail=str(e)) from e
    except HTTPException:
        cleanup_failed_upload(file_path, page_dir)
        raise
    except Exception as e:
        logger.error(f"Error processing upload for file '{filename}': {e}", exc_info=True)
        cleanup_failed_upload(file_path, page_dir)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest, request: Request):
    """
    Hỏi đáp dựa trên tài liệu (non-streaming).
    """
    await enforce_rate_limit(request, "query", QUERY_RATE_LIMIT, 60)
    normalize_session_id(req.session_id)

    try:
        answer, results, routed_model, answer_scope = await rag_pipeline.query(
            session_id=req.session_id,
            question=req.question,
            model=req.model,
            mode=req.mode,
        )
        return QueryResponse(
            answer=answer,
            sources=rag_pipeline.format_sources(results),
            session_id=req.session_id,
            model=routed_model,
            mode=req.mode,
            answer_scope=answer_scope,
        )
    except Exception as e:
        logger.error(f"RAG query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    """
    Hỏi đáp dựa trên tài liệu dạng streaming SSE.
    """
    await enforce_rate_limit(request, "query", QUERY_RATE_LIMIT, 60)
    normalize_session_id(req.session_id)

    async def event_generator():
        import json
        try:
            token_stream, results, routed_model, answer_scope = (
                await rag_pipeline.query_stream(
                session_id=req.session_id,
                question=req.question,
                model=req.model,
                mode=req.mode,
            )
            )
            
            # Gửi nguồn trích dẫn (sources) trước
            sources = rag_pipeline.format_sources(results)
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'model': routed_model, 'mode': req.mode, 'answer_scope': answer_scope})}\n\n"

            # Stream từng token câu trả lời
            async for token in token_stream:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.error(f"Stream generation error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.delete("/sessions/{session_id}/files/{filename}")
async def remove_file(session_id: str, filename: str):
    """Xóa file khỏi session và database."""
    session_id = normalize_session_id(session_id)
    safe_filename = safe_upload_filename(filename)
    removed_chunks = vector_store.remove_file(session_id, safe_filename)

    # Xóa file vật lý
    file_path = session_upload_dir(session_id) / safe_filename
    if file_path.exists():
        file_path.unlink()
    page_dir = session_upload_dir(session_id) / "_pages" / Path(safe_filename).stem
    if page_dir.exists():
        shutil.rmtree(page_dir)

    return {
        "message": f"Successfully removed file '{safe_filename}' from session '{session_id}'",
        "chunks_removed": removed_chunks
    }


# ──────────────────────────────────────────────
# Agent Endpoints
# ──────────────────────────────────────────────

@app.post("/agent", response_model=AgentResponse)
async def run_agent(
    req: AgentRequest,
    x_agent_api_key: Optional[str] = Header(default=None),
):
    """
    Thực thi Coding Agent thông qua LangGraph Agent Executor.
    Trả về toàn bộ nhật ký suy luận và kết quả sửa code.
    """
    if not ENABLE_AGENT or agent_executor is None:
        raise HTTPException(status_code=404, detail="Coding Agent is disabled.")

    logger.info(f"Triggering LangGraph agent for task: '{req.task[:100]}'")

    if AGENT_API_KEY and (
        not x_agent_api_key
        or not secrets.compare_digest(x_agent_api_key, AGENT_API_KEY)
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing agent API key.")
    
    try:
        # Chạy đồ thị LangGraph với đầu vào là câu hỏi/tác vụ
        inputs = {"messages": [("user", req.task)]}
        result = await agent_executor.ainvoke(inputs)
        
        # Lấy tin nhắn cuối cùng (thường là câu trả lời hoặc báo cáo kết quả của Agent)
        final_message = result["messages"][-1]
        output_text = message_text(final_message)
        
        # Biên soạn các bước chạy
        steps = []
        for msg in result["messages"]:
            role = "assistant"
            if hasattr(msg, "type"):
                role = msg.type
            elif hasattr(msg, "role"):
                role = msg.role
                
            steps.append({
                "role": role,
                "content": message_text(msg),
                "tool_calls": getattr(msg, "tool_calls", None)
            })

        return AgentResponse(
            session_id=req.session_id,
            status="completed",
            output=output_text,
            steps=steps
        )
        
    except Exception as e:
        logger.error(f"Agent execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    # Khởi chạy trên cổng 8001 của Máy 2
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8001, reload=False)
