"""
src/api/main.py
---------------
FastAPI API Gateway chính cho Máy 2.
Quản lý các REST API phục vụ cho RAG (Upload, Index, Query) và AI Agent (LangGraph Execution).
"""

import asyncio
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str


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
    info = vector_store.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found or empty.")
    return SessionInfoResponse(**info)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Xóa session và mọi tài liệu liên quan."""
    vector_store.delete_session(session_id)
    
    # Xóa file vật lý lưu trên server
    session_dir = UPLOAD_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
        
    return {"message": f"Session {session_id} and its files have been deleted."}


@app.post("/sessions/{session_id}/upload")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    """
    Tải tài liệu lên và lưu trữ dạng vector trong Qdrant.
    Được nâng cấp sử dụng Docling.
    """
    # Đảm bảo thư mục lưu trữ tồn tại
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    file_path = session_dir / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    logger.info(f"Uploaded file '{file.filename}' to session '{session_id}'")

    try:
        # Sử dụng Docling trích xuất cấu trúc văn bản
        chunks = doc_parser.process_file(file_path)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any content from the file."
            )

        # Chuyển đổi văn bản sang vector embeddings (BGE-M3)
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_documents(texts)

        # Lưu trữ vào Qdrant với payload chứa thông tin session
        vector_store.add_chunks(session_id, chunks, embeddings)

        return {
            "filename": file.filename,
            "num_chunks": len(chunks),
            "file_size_kb": round(len(content) / 1024, 1),
            "message": f"Successfully parsed and indexed {len(chunks)} chunks into Qdrant."
        }

    except Exception as e:
        logger.error(f"Error processing upload for file '{file.filename}': {e}", exc_info=True)
        # Cleanup file lỗi
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """
    Hỏi đáp dựa trên tài liệu (non-streaming).
    """
    try:
        answer, results = await rag_pipeline.query(
            session_id=req.session_id,
            question=req.question
        )
        return QueryResponse(
            answer=answer,
            sources=rag_pipeline.format_sources(results),
            session_id=req.session_id
        )
    except Exception as e:
        logger.error(f"RAG query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    Hỏi đáp dựa trên tài liệu dạng streaming SSE.
    """
    async def event_generator():
        import json
        try:
            token_stream, results = await rag_pipeline.query_stream(
                session_id=req.session_id,
                question=req.question
            )
            
            # Gửi nguồn trích dẫn (sources) trước
            sources = rag_pipeline.format_sources(results)
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

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
    removed_chunks = vector_store.remove_file(session_id, filename)

    # Xóa file vật lý
    file_path = UPLOAD_DIR / session_id / filename
    if file_path.exists():
        file_path.unlink()

    return {
        "message": f"Successfully removed file '{filename}' from session '{session_id}'",
        "chunks_removed": removed_chunks
    }


# ──────────────────────────────────────────────
# Agent Endpoints
# ──────────────────────────────────────────────

@app.post("/agent", response_model=AgentResponse)
async def run_agent(req: AgentRequest):
    """
    Thực thi Coding Agent thông qua LangGraph Agent Executor.
    Trả về toàn bộ nhật ký suy luận và kết quả sửa code.
    """
    logger.info(f"Triggering LangGraph agent for task: '{req.task[:100]}'")
    
    try:
        # Chạy đồ thị LangGraph với đầu vào là câu hỏi/tác vụ
        inputs = {"messages": [("user", req.task)]}
        result = await agent_executor.ainvoke(inputs)
        
        # Lấy tin nhắn cuối cùng (thường là câu trả lời hoặc báo cáo kết quả của Agent)
        final_message = result["messages"][-1]
        output_text = final_message.content if hasattr(final_message, "content") else str(final_message)
        
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
                "content": msg.content if hasattr(msg, "content") else str(msg),
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


if __name__ == "__main__":
    import uvicorn
    # Khởi chạy trên cổng 8001 của Máy 2
    uvicorn.run("src.api.main.app", host="0.0.0.0", port=8001, reload=False)
