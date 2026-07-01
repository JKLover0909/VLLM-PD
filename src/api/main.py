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
import re
import secrets
import shutil
import time
import unicodedata
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
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
from src.auth.employee_directory import EmployeeDirectory
from src.integrations.mes_database import MesDatabase
from src.integrations.mes_query_service import MesQueryService
from src.integrations.gmail_sender import (
    GmailSender,
    GmailSenderError,
    parse_email_send_command,
)
from src.i18n.translation import TranslationError, TranslationService

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
RESEARCH_DEMO_DIR = Path(os.getenv("RESEARCH_DEMO_DIR", "./documents/Research"))
RESEARCH_DEMO_SESSION_ID = os.getenv(
    "RESEARCH_DEMO_SESSION_ID",
    "00000000-0000-4000-8000-000000000001",
)
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
PREVIEW_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
EMPLOYEE_DIRECTORY_DB_PATH = Path(
    os.getenv("EMPLOYEE_DIRECTORY_DB_PATH", "data/employee_directory.sqlite")
)
MKAC_PAGE_IMAGE_DIR = Path(os.getenv("MKAC_PAGE_IMAGE_DIR", "mkac_processed/pages"))

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
mes_query_service: Optional[MesQueryService] = None
web_searcher: Optional[WebSearcher] = None
employee_directory = EmployeeDirectory(EMPLOYEE_DIRECTORY_DB_PATH)
mes_database = MesDatabase.from_env()
gmail_sender = GmailSender.from_env()
translation_service = TranslationService.from_env()
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


def research_demo_source_files() -> List[str]:
    """List supported documents prepared for the built-in research demo."""
    if not RESEARCH_DEMO_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in RESEARCH_DEMO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_UPLOAD_EXTENSIONS
    )


def path_is_inside(path: Path, root: Path) -> bool:
    """Return whether path is equal to or nested inside root after resolving."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


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
    global mes_query_service
    global web_searcher

    logger.info("🚀 Starting Meibook API Gateway on Machine 2...")
    
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
        mes_database=mes_database,
    )
    mes_query_service = MesQueryService(
        mes_database=mes_database,
        mes_sql_agent=rag_pipeline.mes_sql_agent,
    )

    logger.info("✅ Meibook API Gateway is fully operational.")
    yield
    logger.info("Shutdown completed.")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Meibook API Gateway",
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
    model: Literal["auto", "local", "openai", "grok"] = "auto"
    mode: Literal["mkac", "mes", "research"] = "mkac"
    ui_language: Literal["vi", "ja"] = "vi"
    employee_id: Optional[str] = None
    conversation_context: List[Dict[str, Any]] = Field(default_factory=list)


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


class ResearchDemoResponse(BaseModel):
    enabled: bool
    ready: bool
    session_id: str
    num_chunks: int
    num_files: int
    files: List[str]
    source_files: List[str]


class EmployeeAuthRequest(BaseModel):
    employee_id: str


class EmployeeResponse(BaseModel):
    id: str
    name: str
    gender: str = ""
    position: str = ""
    department: str = ""
    greeting: str = ""
    department_size: int = 0
    department_heads: List[str] = Field(default_factory=list)
    department_deputies: List[str] = Field(default_factory=list)


class EmployeeAuthResponse(BaseModel):
    employee: EmployeeResponse


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


def verify_mkac_employee(employee_id: Optional[str]) -> EmployeeResponse:
    """Return the employee record or reject MKAC access."""
    employee = employee_directory.profile(employee_id or "")
    if not employee:
        raise HTTPException(status_code=403, detail="Mã nhân viên không hợp lệ.")
    return EmployeeResponse(
        id=employee["id"],
        name=employee["name"],
        gender=employee.get("gender", ""),
        position=employee.get("position", ""),
        department=employee.get("department", ""),
        greeting=employee.get("greeting", ""),
        department_size=employee.get("department_size", 0),
        department_heads=employee.get("department_heads", []),
        department_deputies=employee.get("department_deputies", []),
    )


def authorize_query(req: QueryRequest) -> Optional[EmployeeResponse]:
    if req.mode not in {"mkac", "mes"}:
        return None
    return verify_mkac_employee(req.employee_id)


def employee_context_for_query(
    req: QueryRequest,
    employee: Optional[EmployeeResponse],
) -> Optional[Dict[str, Any]]:
    if employee is None:
        return None

    context = employee.model_dump()
    context["company_name"] = "Meiko Automation"
    context["company_legal_name"] = "Công ty Cổ phần Meiko Automation"
    department_context = employee_directory.department_context_for_question(
        req.question,
        current_department=employee.department,
    )
    if department_context:
        context["queried_departments"] = department_context
    people_context = employee_directory.people_context_for_question(req.question)
    if people_context:
        context["queried_people"] = people_context
    return context


async def localize_query_request(req: QueryRequest) -> QueryRequest:
    """Translate Japanese UI questions into Vietnamese for the core backend."""
    if req.ui_language != "ja" or translation_service is None:
        return req
    try:
        translated = await translation_service.translate_query(
            req.question,
            ui_language=req.ui_language,
            mode=req.mode,
        )
    except TranslationError as exc:
        logger.warning("Cannot translate query for UI; using original question: %s", exc)
        return req
    if translated.backend_question == req.question:
        return req
    logger.info(
        "Translated UI query language=%s mode=%s original=%r backend=%r",
        req.ui_language,
        req.mode,
        translated.original_question,
        translated.backend_question,
    )
    return req.model_copy(update={"question": translated.backend_question})


async def translate_answer_for_ui(answer: str, req: QueryRequest) -> str:
    """Translate Vietnamese backend answers back to the selected UI language."""
    if req.ui_language != "ja" or translation_service is None:
        return answer
    try:
        return await translation_service.translate_answer(
            answer,
            ui_language=req.ui_language,
            mode=req.mode,
        )
    except TranslationError as exc:
        logger.warning("Cannot translate answer for UI; returning original answer: %s", exc)
        return answer


async def translate_sources_for_ui(
    sources: List[Dict[str, Any]],
    req: QueryRequest,
) -> List[Dict[str, Any]]:
    """Translate short citation previews for non-Vietnamese UI languages."""
    if req.ui_language != "ja" or translation_service is None:
        return sources

    translated_sources: List[Dict[str, Any]] = []
    for source in sources:
        translated_source = dict(source)
        preview = translated_source.get("preview")
        if isinstance(preview, str) and preview.strip():
            try:
                translated_source["preview"] = await translation_service.translate_ui_text(
                    preview,
                    ui_language=req.ui_language,
                    mode=req.mode,
                    purpose="source preview",
                )
            except TranslationError as exc:
                logger.warning("Cannot translate source preview for UI: %s", exc)
        translated_sources.append(translated_source)
    return translated_sources


def ensure_query_services_ready() -> None:
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not ready.")
    if mes_query_service is None:
        raise HTTPException(status_code=503, detail="MES query service is not ready.")


async def route_query(
    req: QueryRequest,
    *,
    question: Optional[str] = None,
    current_user_context: Optional[Dict[str, Any]] = None,
) -> tuple[str, list, str, str]:
    """Route by mode so MES never falls through to document RAG."""
    ensure_query_services_ready()
    routed_question = question or req.question
    if req.mode == "mes":
        logger.info("Routing query to MES service.")
        return await mes_query_service.query(
            question=routed_question,
            model=req.model,
        )
    if req.mode in {"mkac", "research"}:
        logger.info("Routing query to %s RAG service.", req.mode)
        return await rag_pipeline.query(
            session_id=req.session_id,
            question=routed_question,
            model=req.model,
            mode=req.mode,
            current_user=current_user_context,
        )
    raise HTTPException(status_code=400, detail=f"Unsupported query mode: {req.mode}")


async def route_query_stream(
    req: QueryRequest,
    *,
    current_user_context: Optional[Dict[str, Any]] = None,
):
    """Streaming variant of route_query with explicit mode separation."""
    ensure_query_services_ready()
    if req.mode == "mes":
        logger.info("Routing streaming query to MES service.")
        return await mes_query_service.query_stream(
            question=req.question,
            model=req.model,
        )
    if req.mode in {"mkac", "research"}:
        logger.info("Routing streaming query to %s RAG service.", req.mode)
        return await rag_pipeline.query_stream(
            session_id=req.session_id,
            question=req.question,
            model=req.model,
            mode=req.mode,
            current_user=current_user_context,
        )
    raise HTTPException(status_code=400, detail=f"Unsupported query mode: {req.mode}")


def build_email_body(
    *,
    original_question: str,
    data_question: str,
    answer: str,
    answer_scope: str,
) -> str:
    return (
        "Xin chào,\n\n"
        "Meibook gửi bạn thông tin theo yêu cầu:\n\n"
        f"{answer.strip()}\n\n"
        "---\n"
        f"Yêu cầu gốc: {original_question.strip()}\n"
        f"Câu hỏi dữ liệu: {data_question.strip()}\n"
        f"Nguồn trả lời: {answer_scope}\n"
    )


def _normalize_reference_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"\s+", " ", normalized).strip()


def is_context_reference(value: str) -> bool:
    normalized = _normalize_reference_text(value)
    reference_markers = (
        "thong tin nay",
        "noi dung nay",
        "ket qua nay",
        "cau tra loi nay",
        "phan tren",
        "o tren",
        "vua roi",
        "ben tren",
    )
    return any(marker in normalized for marker in reference_markers)


def latest_assistant_context(
    conversation_context: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for item in reversed(conversation_context or []):
        if item.get("role") != "assistant":
            continue
        content = str(item.get("content") or "").strip()
        if content:
            return {
                "content": content,
                "answer_scope": str(item.get("answer_scope") or "conversation_context"),
                "model": str(item.get("model") or ""),
            }
    return None


async def handle_email_send_query(
    req: QueryRequest,
    current_user_context: Optional[Dict[str, Any]],
) -> Optional[QueryResponse]:
    command = parse_email_send_command(req.question)
    if command is None:
        return None
    if gmail_sender is None or not gmail_sender.available:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gmail send chưa sẵn sàng. Hãy kiểm tra GMAIL_SEND_ENABLED, "
                "GMAIL_CREDENTIALS_PATH và token OAuth."
            ),
        )

    if is_context_reference(command.data_question):
        previous_answer = latest_assistant_context(req.conversation_context)
        if previous_answer is None:
            raise GmailSenderError(
                "Chưa có nội dung trước đó để gửi. Hãy hỏi lấy kết quả trước, "
                "hoặc viết rõ nội dung cần gửi trong câu lệnh email."
            )

        body = build_email_body(
            original_question=req.question,
            data_question="Nội dung từ câu trả lời gần nhất trong cuộc hội thoại",
            answer=previous_answer["content"],
            answer_scope=previous_answer["answer_scope"],
        )
        send_result = await asyncio.to_thread(
            gmail_sender.send_email,
            command.to_email,
            command.subject,
            body,
        )
        logger.info(
            "Sent Meibook contextual email action to=%s message_id=%s subject=%s",
            send_result.to_email,
            send_result.message_id,
            send_result.subject,
        )
        status_answer = (
            f"Đã gửi email tới {send_result.to_email} với tiêu đề "
            f"\"{send_result.subject}\".\n\n"
            f"Nội dung chính:\n{previous_answer['content']}"
        )
        return QueryResponse(
            answer=status_answer,
            sources=[],
            session_id=req.session_id,
            model=previous_answer["model"] or req.model,
            mode=req.mode,
            answer_scope="email_action",
        )

    answer, results, routed_model, answer_scope = await route_query(
        req,
        question=command.data_question,
        current_user_context=current_user_context,
    )
    body = build_email_body(
        original_question=req.question,
        data_question=command.data_question,
        answer=answer,
        answer_scope=answer_scope,
    )
    send_result = await asyncio.to_thread(
        gmail_sender.send_email,
        command.to_email,
        command.subject,
        body,
    )
    logger.info(
        "Sent Meibook email action to=%s message_id=%s subject=%s",
        send_result.to_email,
        send_result.message_id,
        send_result.subject,
    )
    status_answer = (
        f"Đã gửi email tới {send_result.to_email} với tiêu đề "
        f"\"{send_result.subject}\".\n\n"
        f"Nội dung chính:\n{answer}"
    )
    return QueryResponse(
        answer=status_answer,
        sources=rag_pipeline.format_sources(results),
        session_id=req.session_id,
        model=routed_model,
        mode=req.mode,
        answer_scope="email_action",
    )


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
        "employee_directory": {
            "db_path": str(EMPLOYEE_DIRECTORY_DB_PATH),
            "employees": employee_directory.count(),
        },
        "mes_database": (
            {
                **mes_database.status(),
                "sql_agent_available": bool(
                    mes_query_service
                    and mes_query_service.mes_sql_agent
                    and mes_query_service.mes_sql_agent.available
                ),
            }
            if mes_database is not None
            else {"available": False, "enabled": False}
        ),
        "gmail_send": (
            gmail_sender.status()
            if gmail_sender is not None
            else {"enabled": False, "available": False}
        ),
        "translation": {
            "enabled": translation_service is not None,
            "model": getattr(translation_service, "model", ""),
        },
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


@app.post("/auth/employee", response_model=EmployeeAuthResponse)
async def authenticate_employee(req: EmployeeAuthRequest):
    """Check whether a MKAC employee ID exists in the local directory."""
    employee = verify_mkac_employee(req.employee_id)
    return EmployeeAuthResponse(employee=employee)


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


@app.get("/research/demo", response_model=ResearchDemoResponse)
async def research_demo_status():
    """Return the pre-indexed research demo session, if it is available."""
    session_id = normalize_session_id(RESEARCH_DEMO_SESSION_ID)
    source_files = research_demo_source_files()
    info = vector_store.get_session_info(session_id) if vector_store else None
    indexed_files = sorted((info or {}).get("files", []))
    return ResearchDemoResponse(
        enabled=bool(source_files),
        ready=bool(info),
        session_id=session_id,
        num_chunks=(info or {}).get("num_chunks", 0),
        num_files=(info or {}).get("num_files", 0),
        files=indexed_files,
        source_files=source_files,
    )


@app.get("/sources/preview")
async def source_page_preview(
    session_id: str,
    mode: Literal["mkac", "research"],
    file: str,
    page: int,
    language: Literal["vi", "ja"] = "vi",
):
    """Return a page image preview for an indexed citation."""
    def preview_error(status_code: int, vi_detail: str, ja_detail: str) -> None:
        raise HTTPException(
            status_code=status_code,
            detail=ja_detail if language == "ja" else vi_detail,
        )

    filename = Path(file).name
    if filename != file or not filename:
        preview_error(400, "Invalid source filename.", "参照元ファイル名が正しくありません。")
    if page <= 0:
        preview_error(400, "Invalid source page.", "参照元ページ番号が正しくありません。")

    if mode == "mkac":
        store = mkac_vector_store
        lookup_session_id = "mkac"
    else:
        store = vector_store
        lookup_session_id = normalize_session_id(session_id)

    if store is None:
        preview_error(503, "Vector store is not ready.", "ベクトルデータベースはまだ準備できていません。")

    image_path = store.get_page_image_path(lookup_session_id, filename, page)
    if not image_path:
        preview_error(404, "Preview image not found.", "プレビュー画像が見つかりません。")

    resolved_path = Path(image_path).resolve()
    allowed_roots = [UPLOAD_DIR.resolve(), MKAC_PAGE_IMAGE_DIR.resolve()]
    if not any(path_is_inside(resolved_path, root) for root in allowed_roots):
        preview_error(403, "Preview path is not allowed.", "このプレビュー画像の参照は許可されていません。")
    if not resolved_path.is_file() or resolved_path.suffix.lower() not in PREVIEW_IMAGE_EXTENSIONS:
        preview_error(404, "Preview image not found.", "プレビュー画像が見つかりません。")

    media_type = "image/jpeg" if resolved_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(resolved_path, media_type=media_type)


@app.get("/models")
async def list_models(language: Literal["vi", "ja"] = "vi"):
    """Danh sách model người dùng có thể chọn trên frontend."""
    model_text = {
        "vi": {
            "local": {
                "name": "Qwen Local Model",
                "description": "Chạy Qwen nội bộ/local cho hỏi đáp dạng text.",
            },
        },
        "ja": {
            "local": {
                "name": "Qwenローカルモデル",
                "description": "テキストQ&A向けに社内/ローカルQwenモデルを使用します。",
            },
        },
    }
    text = model_text.get(language, model_text["vi"])
    return {
        "default": "auto",
        "models": [
            {
                "id": "auto",
                **text["local"],
            },
        ],
    }


# Cache dữ liệu quick answers khi load lần đầu
_quick_answers_cache: Optional[dict] = None
QUICK_ANSWERS_PATH = Path(__file__).resolve().parents[2] / "config" / "quick_answers.json"


def _load_quick_answers() -> dict:
    """Đọc file config/quick_answers.json và cache kết quả."""
    global _quick_answers_cache
    if _quick_answers_cache is not None:
        return _quick_answers_cache
    try:
        with QUICK_ANSWERS_PATH.open("r", encoding="utf-8") as f:
            _quick_answers_cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Cannot load quick answers config: {e}")
        _quick_answers_cache = {}
    return _quick_answers_cache


@app.get("/quick-answers")
async def quick_answers(mode: str = "mkac", language: Literal["vi", "ja"] = "vi"):
    """Trả về danh sách câu hỏi gợi ý theo chế độ."""
    data = _load_quick_answers()
    items = data.get(mode, [])
    suggestions = []
    for item in items:
        question = item.get("question", "")
        answer = item.get("answer", "")
        if language == "ja":
            question = item.get("question_ja", "")
            answer = item.get("answer_ja", "")
        if not question or not answer:
            continue
        suggestions.append(
            {
                "question": question,
                "keywords": item.get("keywords", []),
                "answer": answer,
            }
        )
    return {
        "mode": mode,
        "language": language,
        "short_answer_threshold": data.get("short_answer_threshold", 300),
        "max_suggestions": data.get("max_suggestions", 3),
        "suggestions": suggestions,
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
    current_employee = authorize_query(req)

    try:
        localized_req = await localize_query_request(req)
        current_user_context = employee_context_for_query(
            localized_req,
            current_employee,
        )
        email_response = await handle_email_send_query(localized_req, current_user_context)
        if email_response is not None:
            translated_email_answer = await translate_answer_for_ui(
                email_response.answer,
                req,
            )
            translated_email_sources = await translate_sources_for_ui(
                email_response.sources,
                req,
            )
            if translated_email_answer != email_response.answer:
                return email_response.model_copy(
                    update={
                        "answer": translated_email_answer,
                        "sources": translated_email_sources,
                    }
                )
            return email_response.model_copy(
                update={"sources": translated_email_sources}
            )

        answer, results, routed_model, answer_scope = await route_query(
            localized_req,
            current_user_context=current_user_context,
        )
        answer = await translate_answer_for_ui(answer, req)
        sources = await translate_sources_for_ui(
            rag_pipeline.format_sources(results),
            req,
        )
        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=req.session_id,
            model=routed_model,
            mode=req.mode,
            answer_scope=answer_scope,
        )
    except GmailSenderError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except TranslationError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
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
    current_employee = authorize_query(req)
    try:
        localized_req = await localize_query_request(req)
        current_user_context = employee_context_for_query(
            localized_req,
            current_employee,
        )
    except TranslationError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    async def event_generator():
        import json
        try:
            email_response = await handle_email_send_query(
                localized_req,
                current_user_context,
            )
            if email_response is not None:
                translated_email_answer = await translate_answer_for_ui(
                    email_response.answer,
                    req,
                )
                translated_email_sources = await translate_sources_for_ui(
                    email_response.sources,
                    req,
                )
                yield f"data: {json.dumps({'type': 'sources', 'sources': translated_email_sources})}\n\n"
                yield f"data: {json.dumps({'type': 'meta', 'model': email_response.model, 'mode': email_response.mode, 'answer_scope': email_response.answer_scope})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': translated_email_answer})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            if req.ui_language == "ja":
                answer, results, routed_model, answer_scope = await route_query(
                    localized_req,
                    current_user_context=current_user_context,
                )
                translated_answer = await translate_answer_for_ui(answer, req)
                sources = await translate_sources_for_ui(
                    rag_pipeline.format_sources(results),
                    req,
                )
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                yield f"data: {json.dumps({'type': 'meta', 'model': routed_model, 'mode': req.mode, 'answer_scope': answer_scope})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': translated_answer})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            token_stream, results, routed_model, answer_scope = (
                await route_query_stream(
                    localized_req,
                    current_user_context=current_user_context,
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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve the MKAC logo as the browser tab icon."""
    favicon_path = FRONTEND_DIST / "mkac-logo.png"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="Favicon not found.")
    return FileResponse(favicon_path, media_type="image/png")


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    # Khởi chạy trên cổng 8001 của Máy 2
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8001, reload=False)
