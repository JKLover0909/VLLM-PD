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

from src.rag.parser import DocumentParser
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.rag_pipeline import RAGPipeline
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
doc_parser: Optional[DocumentParser] = None
rag_pipeline: Optional[RAGPipeline] = None
rate_limit_events: Dict[str, Deque[float]] = defaultdict(deque)
rate_limit_lock = asyncio.Lock()


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
async def lifespan(app: FastAPI):
    """
    Khởi tạo các mô hình và kết nối database khi ứng dụng bắt đầu.
    """
    global embedder, vector_store, doc_parser, rag_pipeline

    logger.info("🚀 Starting VLLM-PD API Gateway on Machine 2...")
    
    # Khởi tạo Vector DB trước
    vector_store = VectorStore(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Khởi tạo local embedder (BGE-M3)
    embedder = Embedder()
    
    # Khởi tạo bộ phân tích tài liệu Docling
    doc_parser = DocumentParser()
    
    # Khởi tạo RAG Pipeline kết nối với LiteLLM
    rag_pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store
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
    model: Literal["auto", "local", "mimo", "openai"] = "auto"
    mode: Literal["chat", "research"] = "chat"


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    model: str
    mode: str


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
    return {
        "status": "healthy",
        "qdrant_host": QDRANT_HOST,
        "qdrant_port": QDRANT_PORT,
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
                "description": "Ưu tiên Gemma4 local, fallback sang Mimo Pro rồi OpenAI.",
            },
            {
                "id": "local",
                "name": "Gemma4 Local",
                "description": "Chạy trên Máy 1, dữ liệu không gửi tới cloud provider.",
            },
            {
                "id": "mimo",
                "name": "MiMo 2.5 Pro",
                "description": "Phù hợp tổng hợp và nghiên cứu tài liệu.",
            },
            {
                "id": "openai",
                "name": "OpenAI",
                "description": "Dùng GPT-4o mini cho truy vấn khó và phản hồi ổn định.",
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

        # Sử dụng Docling trích xuất cấu trúc văn bản
        chunks = await asyncio.to_thread(doc_parser.process_file, file_path)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any content from the file."
            )

        # Chuyển đổi văn bản sang vector embeddings (BGE-M3)
        texts = [c.text for c in chunks]
        embeddings = await asyncio.to_thread(embedder.embed_documents, texts)

        # Lưu trữ vào Qdrant với payload chứa thông tin session
        await asyncio.to_thread(vector_store.remove_file, session_id, filename)
        await asyncio.to_thread(vector_store.add_chunks, session_id, chunks, embeddings)

        return {
            "filename": filename,
            "num_chunks": len(chunks),
            "file_size_kb": round(uploaded_bytes / 1024, 1),
            "message": f"Successfully parsed and indexed {len(chunks)} chunks into Qdrant."
        }

    except HTTPException:
        if file_path.exists():
            file_path.unlink()
        raise
    except Exception as e:
        logger.error(f"Error processing upload for file '{filename}': {e}", exc_info=True)
        # Cleanup file lỗi
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest, request: Request):
    """
    Hỏi đáp dựa trên tài liệu (non-streaming).
    """
    await enforce_rate_limit(request, "query", QUERY_RATE_LIMIT, 60)
    normalize_session_id(req.session_id)

    try:
        answer, results, routed_model = await rag_pipeline.query(
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
            token_stream, results, routed_model = await rag_pipeline.query_stream(
                session_id=req.session_id,
                question=req.question,
                model=req.model,
                mode=req.mode,
            )
            
            # Gửi nguồn trích dẫn (sources) trước
            sources = rag_pipeline.format_sources(results)
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'meta', 'model': routed_model, 'mode': req.mode})}\n\n"

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
