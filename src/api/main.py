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
import uuid
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
from src.auth.employee_intent import normalize_text
from src.integrations.mes_database import MesDatabase
from src.integrations.mes_query_service import MesQueryService
from src.integrations.gmail_sender import (
    GmailSender,
    GmailSenderError,
    parse_email_send_command,
    try_parse_email_send_command,
)
from src.i18n.translation import TranslationError, TranslationService

from src.api import config
from src.api.config import (
    AGENT_API_KEY,
    EMPLOYEE_DIRECTORY_DB_PATH,
    FRONTEND_DIST,
    LOG_LEVEL,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    MES_QUERY_CACHE_TTL_SECONDS,
    MIN_QUERY_RESPONSE_SECONDS,
    MKAC_PAGE_IMAGE_DIR,
    PREVIEW_IMAGE_EXTENSIONS,
    QDRANT_HOST,
    QDRANT_PORT,
    QUERY_RATE_LIMIT,
    QUERY_RESPONSE_CACHE_SIZE,
    QUERY_RESPONSE_CACHE_TTL_SECONDS,
    RESEARCH_DEMO_SESSION_ID,
    UPLOAD_DIR,
    UPLOAD_PROCESSING_CONCURRENCY,
    UPLOAD_QUEUE_SIZE,
    UPLOAD_RATE_LIMIT,
)
from src.api.schemas import (
    AgentRequest,
    AgentResponse,
    EmployeeAuthRequest,
    EmployeeAuthResponse,
    EmployeeResponse,
    QueryRequest,
    QueryResponse,
    ResearchDemoResponse,
    SessionInfoResponse,
    SessionResponse,
)
from src.api.sse import (
    query_processing_status_key,
    sse_event,
    sse_status,
)
from src.api.helpers import (
    build_direct_email_body,
    build_email_body,
    cleanup_failed_upload,
    client_ip,
    is_context_reference,
    latest_assistant_context,
    message_text,
    normalize_prepared_question,
    normalize_session_id,
    path_is_inside,
    query_cache_key,
    research_demo_source_files,
    safe_upload_filename,
    session_upload_dir,
)

ENABLE_AGENT = config.ENABLE_AGENT
agent_executor = None
if ENABLE_AGENT:
    from src.agent.graph import agent_executor

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
query_response_cache: "OrderedDict[str, tuple[float, QueryResponse]]" = OrderedDict()
query_response_cache_lock = asyncio.Lock()

EMPLOYEE_CONTEXT_REFERENCE_MARKERS = (
    "anh nay",
    "chi nay",
    "nguoi nay",
    "ong nay",
    "ba nay",
    "co nay",
    "chu nay",
    "bac nay",
    "anh do",
    "chi do",
    "nguoi do",
    "ong do",
    "ba do",
    "anh ay",
    "chi ay",
    "ong ay",
    "ba ay",
    "nguoi ay",
    "nguoi vua roi",
    "nhan su nay",
    "nhan vien nay",
    "this person",
    "that person",
    "this employee",
    "that employee",
    "この人",
    "その人",
    "この方",
    "その方",
    "彼",
    "彼女",
)

# ── Observability: đếm request + đo độ trễ theo route, tỉ lệ cache hit ──
# Nhẹ, in-memory, per-process (khi scale nhiều worker cần gom qua Redis/Prometheus).
query_metrics: Dict[str, Any] = {
    "total": 0,
    "cache_hits": 0,
    "errors": 0,
    "by_scope": defaultdict(int),
    "by_mode": defaultdict(int),
    "latency_ms": deque(maxlen=500),  # mẫu để tính p50/p95
}
query_metrics_lock = asyncio.Lock()
upload_processing_semaphore = asyncio.Semaphore(UPLOAD_PROCESSING_CONCURRENCY)
upload_admission_lock = asyncio.Lock()
upload_active = 0
upload_waiting = 0


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
# Query response cache (giữ trạng thái toàn cục nên ở lại main.py)
# ──────────────────────────────────────────────
async def get_cached_query_response(cache_key: Optional[str]) -> Optional[QueryResponse]:
    if not cache_key:
        return None
    now = time.monotonic()
    async with query_response_cache_lock:
        cached = query_response_cache.get(cache_key)
        if cached is None:
            return None
        expiry_at, response = cached
        if now > expiry_at:
            query_response_cache.pop(cache_key, None)
            return None
        query_response_cache.move_to_end(cache_key)
        return response


async def set_cached_query_response(
    cache_key: Optional[str],
    response: QueryResponse,
) -> None:
    if not cache_key:
        return
    # MES snapshot tĩnh → giữ cache lâu hơn; các mode khác dùng TTL mặc định.
    ttl = (
        MES_QUERY_CACHE_TTL_SECONDS
        if response.mode == "mes"
        else QUERY_RESPONSE_CACHE_TTL_SECONDS
    )
    async with query_response_cache_lock:
        query_response_cache[cache_key] = (time.monotonic() + ttl, response)
        query_response_cache.move_to_end(cache_key)
        while len(query_response_cache) > QUERY_RESPONSE_CACHE_SIZE:
            query_response_cache.popitem(last=False)


def build_query_cache_key(req: QueryRequest) -> Optional[str]:
    """query_cache_key có gắn phiên bản snapshot MES để re-import tự vô hiệu."""
    if question_uses_employee_context_reference(req.question):
        return None
    snapshot_version = ""
    if req.mode == "mes" and mes_database is not None:
        try:
            snapshot_version = mes_database.snapshot_version()
        except Exception:  # pragma: no cover - phòng lỗi đọc metadata
            snapshot_version = ""
    return query_cache_key(req, snapshot_version=snapshot_version)


def question_uses_employee_context_reference(question: str) -> bool:
    normalized = normalize_text(question)
    if any(marker in normalized for marker in EMPLOYEE_CONTEXT_REFERENCE_MARKERS):
        return True
    return any(marker in (question or "") for marker in ("この人", "その人", "この方", "その方", "彼", "彼女"))


def latest_referenced_employee(
    conversation_context: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Find the latest single employee mentioned in recent chat context."""
    for role in ("assistant", "user"):
        for item in reversed(conversation_context or []):
            if item.get("role") != role:
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            people = employee_directory.people_context_for_text(content)
            if len(people) == 1:
                return people[0]
    return None


def resolve_employee_context_question(req: QueryRequest, question: str) -> str:
    """Rewrite pronoun-based HR follow-ups to the latest discussed employee."""
    if req.mode != "mkac" or not question_uses_employee_context_reference(question):
        return question
    if employee_directory.people_context_for_text(question):
        return question
    person = latest_referenced_employee(req.conversation_context)
    if not person:
        return question

    name = str(person.get("name") or "").strip()
    employee_id = str(person.get("id") or "").strip()
    if not name:
        return question

    replacement = f"{name} (mã nhân viên {employee_id})" if employee_id else name
    rewritten = re.sub(
        r"\b(?:anh|chị|chi|người|nguoi|ông|ong|bà|ba|cô|co|chú|chu|bác|bac)\s+"
        r"(?:này|nay|đó|do|ấy|ay|vừa rồi|vua roi)\b",
        replacement,
        question,
        count=1,
        flags=re.IGNORECASE,
    )
    for marker in ("この人", "その人", "この方", "その方", "彼", "彼女"):
        if marker in rewritten:
            rewritten = rewritten.replace(marker, replacement, 1)
            break
    if rewritten == question:
        rewritten = f"{replacement}: {question}"
    logger.info("Resolved employee context question: %r -> %r", question, rewritten)
    return rewritten


def question_uses_mes_lot_context_reference(question: str) -> bool:
    normalized = normalize_text(question)
    return any(
        marker in normalized
        for marker in (
            "lot do",
            "lot nay",
            "lo do",
            "lo nay",
            "that lot",
            "this lot",
        )
    ) or any(marker in (question or "") for marker in ("そのロット", "このロット"))


def latest_lot_id_in_context(conversation_context: list[dict[str, Any]]) -> str:
    for item in reversed(conversation_context or []):
        content = str(item.get("content") or "")
        match = MesDatabase.LOT_PATTERN.search(content)
        if match:
            return match.group(0)
    return ""


PRODUCT_CONTEXT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+|\d{4}-\d{4})(?![A-Za-z0-9])"
)


def product_ids_in_text(text: str) -> list[str]:
    products: list[str] = []
    for match in PRODUCT_CONTEXT_PATTERN.finditer(text or ""):
        candidate = match.group(0)
        if MesDatabase.LOT_PATTERN.fullmatch(candidate):
            continue
        if candidate not in products:
            products.append(candidate)
    return products


def latest_product_id_in_context(conversation_context: list[dict[str, Any]]) -> str:
    for item in reversed(conversation_context or []):
        content = str(item.get("content") or "")
        products = product_ids_in_text(content)
        if products:
            return products[0]
    return ""


def latest_mes_context_kind(conversation_context: list[dict[str, Any]]) -> str:
    for item in reversed(conversation_context or []):
        content = str(item.get("content") or "")
        normalized = normalize_text(content)
        if product_ids_in_text(content) and any(
            marker in normalized
            for marker in ("ma hang", "san pham", "product", "品番", "製品")
        ):
            return "product"
        if MesDatabase.LOT_PATTERN.search(content):
            return "lot"
    return ""


def question_uses_mes_product_compare_context(question: str) -> bool:
    normalized = normalize_text(question)
    return any(
        marker in normalized
        for marker in ("so voi", "so sanh", "compare", "comparison", "thi sao")
    ) or any(marker in (question or "") for marker in ("比較", "比べ", "それと比べ"))


def mes_rank_position_from_question(question: str) -> int | None:
    normalized = normalize_text(question)
    match = re.search(r"\b(?:dung\s+)?thu\s+(\d+)\b|\b(\d+)(?:st|nd|rd|th)\b", normalized)
    if match:
        for group in match.groups():
            if group:
                return int(group)
    word_match = re.search(
        r"\b(?:dung\s+)?thu\s+(hai|ba|bon|tu|nam|sau|bay|tam|chin|muoi)\b|\b(second|third|fourth|fifth)\b",
        normalized,
    )
    if word_match:
        ranks = {
            "hai": 2,
            "second": 2,
            "ba": 3,
            "third": 3,
            "bon": 4,
            "tu": 4,
            "fourth": 4,
            "nam": 5,
            "fifth": 5,
            "sau": 6,
            "bay": 7,
            "tam": 8,
            "chin": 9,
            "muoi": 10,
        }
        for group in word_match.groups():
            if group:
                return ranks.get(group)
    jp_match = re.search(r"(\d+)\s*番目|第\s*(\d+)", question or "")
    if jp_match:
        for group in jp_match.groups():
            if group:
                return int(group)
    if re.search(r"(二番目|2番目|第2)", question or ""):
        return 2
    return None


def resolve_mes_context_question(req: QueryRequest, question: str) -> str:
    """Rewrite short MES follow-ups using recent structured chat context."""
    if req.mode != "mes":
        return question

    current_products = product_ids_in_text(question)
    if (
        question_uses_mes_product_compare_context(question)
        and len(current_products) == 1
    ):
        previous_product = latest_product_id_in_context(req.conversation_context)
        if previous_product and previous_product not in current_products:
            rewritten = (
                f"So sánh tổng lỗi giữa sản phẩm {previous_product} "
                f"và {current_products[0]}"
            )
            logger.info("Resolved MES product comparison: %r -> %r", question, rewritten)
            return rewritten

    rank = mes_rank_position_from_question(question)
    if rank and latest_mes_context_kind(req.conversation_context) == "product":
        rewritten = f"Mã hàng có tổng lỗi đứng thứ {rank} là gì?"
        logger.info("Resolved MES rank follow-up: %r -> %r", question, rewritten)
        return rewritten

    if not question_uses_mes_lot_context_reference(question):
        return question
    if MesDatabase.LOT_PATTERN.search(question or ""):
        return question
    lot_id = latest_lot_id_in_context(req.conversation_context)
    if not lot_id:
        return question
    rewritten = f"{question.strip()} Lot {lot_id}"
    logger.info("Resolved MES context question: %r -> %r", question, rewritten)
    return rewritten


async def record_query_metric(
    *,
    mode: str,
    ui_language: str,
    answer_scope: str,
    cache_hit: bool,
    latency_ms: float,
    error: bool = False,
) -> None:
    """Ghi một dòng log có cấu trúc + cập nhật bộ đếm cho /metrics."""
    logger.info(
        "query_complete mode=%s lang=%s scope=%s cache_hit=%s error=%s "
        "latency_ms=%d",
        mode,
        ui_language,
        answer_scope,
        cache_hit,
        error,
        int(latency_ms),
    )
    async with query_metrics_lock:
        query_metrics["total"] += 1
        if cache_hit:
            query_metrics["cache_hits"] += 1
        if error:
            query_metrics["errors"] += 1
        query_metrics["by_scope"][answer_scope] += 1
        query_metrics["by_mode"][mode] += 1
        query_metrics["latency_ms"].append(latency_ms)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[rank], 1)


async def wait_for_min_query_latency(started_at: float) -> None:
    """Keep very fast prepared/cache answers feeling like a normal request."""
    if MIN_QUERY_RESPONSE_SECONDS <= 0:
        return
    remaining = MIN_QUERY_RESPONSE_SECONDS - (time.monotonic() - started_at)
    if remaining > 0:
        await asyncio.sleep(remaining)


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


def employee_directory_query_response(
    req: QueryRequest,
    employee: Optional[EmployeeResponse],
    *,
    question: Optional[str] = None,
) -> Optional[QueryResponse]:
    """Answer structured employee-directory questions directly from SQLite."""
    if req.mode != "mkac" or try_parse_email_send_command(req.question) is not None:
        return None

    lookup_question = question or req.question
    answer = employee_directory.structured_answer_for_question(
        lookup_question,
        current_department=employee.department if employee else "",
        language=req.ui_language,
        conversation_context=req.conversation_context,
    )
    if not answer:
        return None

    return QueryResponse(
        answer=answer,
        sources=[],
        session_id=req.session_id,
        model="auto-model",
        mode=req.mode,
        answer_scope="mkac",
    )


async def localize_query_request(req: QueryRequest) -> QueryRequest:
    """Translate Japanese UI questions into Vietnamese for the core backend."""
    if req.ui_language != "ja" or translation_service is None:
        return req
    # MES có bộ rule deterministic đọc được các marker Nhật cơ bản. Dịch câu hỏi
    # trước khi route dễ làm méo mã Lot/mã hàng/tên lỗi, nên để nguyên câu gốc.
    if req.mode == "mes":
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

    async def translate_preview(preview: str) -> str:
        try:
            return await translation_service.translate_ui_text(
                preview,
                ui_language=req.ui_language,
                mode=req.mode,
                purpose="source preview",
            )
        except TranslationError as exc:
            logger.warning("Cannot translate source preview for UI: %s", exc)
            return preview

    # Dịch các preview song song thay vì tuần tự để cắt độ trễ khi có nhiều
    # nguồn (thường gặp ở chế độ research). Nguồn không có preview giữ nguyên.
    tasks: Dict[int, "asyncio.Task[str]"] = {}
    for index, source in enumerate(sources):
        preview = source.get("preview")
        if isinstance(preview, str) and preview.strip():
            tasks[index] = asyncio.create_task(translate_preview(preview))
    if tasks:
        await asyncio.gather(*tasks.values())

    translated_sources: List[Dict[str, Any]] = []
    for index, source in enumerate(sources):
        translated_source = dict(source)
        if index in tasks:
            translated_source["preview"] = tasks[index].result()
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
        routed_question = resolve_mes_context_question(req, routed_question)
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
            conversation_context=req.conversation_context,
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
        routed_question = resolve_mes_context_question(req, req.question)
        logger.info("Routing streaming query to MES service.")
        return await mes_query_service.query_stream(
            question=routed_question,
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
            conversation_context=req.conversation_context,
        )
    raise HTTPException(status_code=400, detail=f"Unsupported query mode: {req.mode}")


async def handle_email_send_query(
    req: QueryRequest,
    current_user_context: Optional[Dict[str, Any]],
) -> Optional[QueryResponse]:
    try:
        command = parse_email_send_command(req.question)
    except GmailSenderError as exc:
        normalized = normalize_text(req.question)
        if any(
            marker in normalized
            for marker in (
                "gui email",
                "gui mail",
                "send email",
                "email cho",
                "mail cho",
                "cho email",
            )
        ):
            raise exc
        logger.info("Ignoring ambiguous email parse miss for non-email query.")
        return None
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

    if command.has_explicit_body:
        body = build_direct_email_body(
            original_question=req.question,
            body=command.explicit_body,
        )
        send_result = await asyncio.to_thread(
            gmail_sender.send_email,
            command.to_email,
            command.subject,
            body,
        )
        logger.info(
            "Sent Meibook explicit email action to=%s message_id=%s subject=%s",
            send_result.to_email,
            send_result.message_id,
            send_result.subject,
        )
        status_answer = (
            f"Đã gửi email tới {send_result.to_email} với tiêu đề "
            f"\"{send_result.subject}\".\n\n"
            f"Nội dung chính:\n{command.explicit_body}"
        )
        return QueryResponse(
            answer=status_answer,
            sources=[],
            session_id=req.session_id,
            model=req.model,
            mode=req.mode,
            answer_scope="email_action",
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

@app.get("/metrics")
async def metrics():
    """Số liệu hiệu năng hỏi đáp (in-memory, per-process)."""
    async with query_metrics_lock:
        samples = list(query_metrics["latency_ms"])
        total = query_metrics["total"]
        cache_hits = query_metrics["cache_hits"]
        payload = {
            "total_queries": total,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / total, 3) if total else 0.0,
            "errors": query_metrics["errors"],
            "latency_ms": {
                "count": len(samples),
                "p50": _percentile(samples, 50),
                "p95": _percentile(samples, 95),
                "avg": round(sum(samples) / len(samples), 1) if samples else 0.0,
            },
            "by_mode": dict(query_metrics["by_mode"]),
            "by_scope": dict(query_metrics["by_scope"]),
        }
    return payload


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
                "name": "Local Model",
                "description": "Chạy model nội bộ/local cho hỏi đáp dạng text.",
            },
        },
        "ja": {
            "local": {
                "name": "ローカルモデル",
                "description": "テキストQ&A向けに社内/ローカルモデルを使用します。",
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


def safety_guard_response(req: QueryRequest) -> Optional[QueryResponse]:
    """Block clearly unsafe cross-system requests before retrieval/model calls."""
    normalized = normalize_text(req.question)
    original = req.question or ""
    blocked = False
    if any(marker in normalized for marker in (" doc file env", " file env", ".env", "dotenv")):
        blocked = True
    if ".env" in original:
        blocked = True
    if any(
        marker in original
        for marker in (
            "環境変数",
            "システムファイル",
            "制限を無視",
            "役割を無視",
            "修正",
            "変更",
            "更新",
        )
    ):
        blocked = True
    if "無視" in original and ("制限" in original or "役割" in original):
        blocked = True
    if any(
        marker in normalized
        for marker in (
            "drop table",
            "update tat ca",
            "sua lai thong tin",
            "bo qua vai tro",
            "bo qua moi gioi han",
            "ignore previous",
            "ignore all",
        )
    ):
        blocked = True
    if re.search(r"\bdan\b", normalized):
        blocked = True
    if not blocked:
        return None

    answer = (
        "このリクエストは安全上の理由で実行できません。Meibookは許可されたMKAC文書、"
        "人事ディレクトリ、MESスナップショットの範囲内でのみ回答します。"
        if req.ui_language == "ja"
        else (
            "Không thể thực hiện yêu cầu này vì lý do an toàn. Meibook chỉ trả lời "
            "trong phạm vi tài liệu MKAC, danh bạ nhân sự và MES snapshot được phép."
        )
    )
    return QueryResponse(
        answer=answer,
        sources=[],
        session_id=req.session_id,
        model="auto-model",
        mode=req.mode,
        answer_scope="guardrail",
    )


def prepared_query_response(req: QueryRequest) -> Optional[QueryResponse]:
    """Return curated answers for known MKAC demo questions without calling LLM."""
    if req.mode != "mkac" or try_parse_email_send_command(req.question) is not None:
        return None

    data = _load_quick_answers()
    question_key = normalize_prepared_question(req.question)
    for item in data.get("mkac", []):
        answer = item.get("answer_ja" if req.ui_language == "ja" else "answer", "")
        if not answer:
            continue
        question_field = "question_ja" if req.ui_language == "ja" else "question"
        aliases_field = "aliases_ja" if req.ui_language == "ja" else "aliases"
        candidates = [item.get(question_field, "")]
        aliases = item.get(aliases_field, [])
        if isinstance(aliases, list):
            candidates.extend(str(alias) for alias in aliases)
        if any(
            candidate and normalize_prepared_question(candidate) == question_key
            for candidate in candidates
        ):
            return QueryResponse(
                answer=answer,
                sources=item.get("sources", []),
                session_id=req.session_id,
                model="auto-model",
                mode=req.mode,
                answer_scope="mkac",
            )
    return None


@app.get("/quick-answers")
async def quick_answers(mode: str = "mkac", language: Literal["vi", "ja"] = "vi"):
    """Trả về danh sách câu hỏi gợi ý theo chế độ."""
    data = _load_quick_answers()
    items = data.get(mode, [])
    suggestions = []
    for item in items:
        if item.get("hidden"):
            continue
        is_live = bool(item.get("live"))
        question = item.get("question", "")
        answer = item.get("answer", "")
        if language == "ja":
            question = item.get("question_ja", "")
            answer = item.get("answer_ja", "")
        # Câu hỏi "live" chỉ cần question; đáp án lấy từ pipeline thật khi bấm.
        # Câu tĩnh phải có sẵn cả câu hỏi lẫn đáp án đóng hộp.
        if not question:
            continue
        if not is_live and not answer:
            continue
        suggestions.append(
            {
                "question": question,
                "keywords": item.get("keywords", []),
                "answer": answer,
                "live": is_live,
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
    request_started_at = time.monotonic()
    await enforce_rate_limit(request, "query", QUERY_RATE_LIMIT, 60)
    normalize_session_id(req.session_id)
    current_employee = await asyncio.to_thread(authorize_query, req)
    cache_key = build_query_cache_key(req)
    cached_response = await get_cached_query_response(cache_key)
    if cached_response is not None:
        logger.info("Query response cache hit mode=%s language=%s", req.mode, req.ui_language)
        await record_query_metric(
            mode=req.mode,
            ui_language=req.ui_language,
            answer_scope=cached_response.answer_scope,
            cache_hit=True,
            latency_ms=(time.monotonic() - request_started_at) * 1000,
        )
        await wait_for_min_query_latency(request_started_at)
        return cached_response.model_copy(update={"session_id": req.session_id})

    try:
        guard_response = safety_guard_response(req)
        if guard_response is not None:
            await wait_for_min_query_latency(request_started_at)
            return guard_response

        contextual_question = resolve_employee_context_question(req, req.question)
        directory_response = await asyncio.to_thread(
            employee_directory_query_response,
            req,
            current_employee,
            question=contextual_question,
        )
        if directory_response is not None:
            logger.info(
                "Employee directory answer hit mode=%s language=%s",
                req.mode,
                req.ui_language,
            )
            await set_cached_query_response(cache_key, directory_response)
            await wait_for_min_query_latency(request_started_at)
            return directory_response

        prepared_response = prepared_query_response(req)
        if prepared_response is not None:
            logger.info("Prepared query answer hit mode=%s language=%s", req.mode, req.ui_language)
            await set_cached_query_response(cache_key, prepared_response)
            await wait_for_min_query_latency(request_started_at)
            return prepared_response

        localized_req = await localize_query_request(req)
        localized_contextual_question = resolve_employee_context_question(
            localized_req,
            localized_req.question,
        )
        if localized_contextual_question != localized_req.question:
            localized_req = localized_req.model_copy(
                update={"question": localized_contextual_question}
            )
        current_user_context = await asyncio.to_thread(
            employee_context_for_query,
            localized_req,
            current_employee,
        )
        directory_response = await asyncio.to_thread(
            employee_directory_query_response,
            req,
            current_employee,
            question=localized_contextual_question,
        )
        if directory_response is not None:
            logger.info(
                "Localized employee directory answer hit mode=%s language=%s",
                req.mode,
                req.ui_language,
            )
            await set_cached_query_response(cache_key, directory_response)
            await wait_for_min_query_latency(request_started_at)
            return directory_response

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
        response = QueryResponse(
            answer=answer,
            sources=sources,
            session_id=req.session_id,
            model=routed_model,
            mode=req.mode,
            answer_scope=answer_scope,
        )
        await set_cached_query_response(cache_key, response)
        await record_query_metric(
            mode=req.mode,
            ui_language=req.ui_language,
            answer_scope=answer_scope,
            cache_hit=False,
            latency_ms=(time.monotonic() - request_started_at) * 1000,
        )
        await wait_for_min_query_latency(request_started_at)
        return response
    except GmailSenderError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except TranslationError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.error(f"RAG query error: {e}", exc_info=True)
        await record_query_metric(
            mode=req.mode,
            ui_language=req.ui_language,
            answer_scope="error",
            cache_hit=False,
            latency_ms=(time.monotonic() - request_started_at) * 1000,
            error=True,
        )
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    """
    Hỏi đáp dựa trên tài liệu dạng streaming SSE.
    """
    request_started_at = time.monotonic()
    await enforce_rate_limit(request, "query", QUERY_RATE_LIMIT, 60)
    normalize_session_id(req.session_id)
    current_employee = await asyncio.to_thread(authorize_query, req)
    cache_key = build_query_cache_key(req)
    cached_response = await get_cached_query_response(cache_key)
    if cached_response is not None:
        logger.info("Streaming query response cache hit mode=%s language=%s", req.mode, req.ui_language)

        async def cached_event_generator():
            import json

            response = cached_response.model_copy(update={"session_id": req.session_id})
            yield sse_status(req, "received")
            yield sse_status(req, "cache")
            await record_query_metric(
                mode=req.mode,
                ui_language=req.ui_language,
                answer_scope=response.answer_scope,
                cache_hit=True,
                latency_ms=(time.monotonic() - request_started_at) * 1000,
            )
            await wait_for_min_query_latency(request_started_at)
            yield sse_status(req, "finalizing")
            yield sse_event({"type": "sources", "sources": response.sources})
            yield sse_event({"type": "meta", "model": response.model, "mode": response.mode, "answer_scope": response.answer_scope})
            yield sse_event({"type": "token", "content": response.answer})
            yield sse_event({"type": "done"})

        return StreamingResponse(
            cached_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    prepared_response = prepared_query_response(req)
    if prepared_response is not None:
        logger.info("Streaming prepared query answer hit mode=%s language=%s", req.mode, req.ui_language)

        async def prepared_event_generator():
            import json

            yield sse_status(req, "received")
            yield sse_status(req, "quick_answer")
            await set_cached_query_response(cache_key, prepared_response)
            await wait_for_min_query_latency(request_started_at)
            yield sse_status(req, "finalizing")
            yield sse_event({"type": "sources", "sources": prepared_response.sources})
            yield sse_event({"type": "meta", "model": prepared_response.model, "mode": prepared_response.mode, "answer_scope": prepared_response.answer_scope})
            yield sse_event({"type": "token", "content": prepared_response.answer})
            yield sse_event({"type": "done"})

        return StreamingResponse(
            prepared_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    try:
        guard_response = safety_guard_response(req)
        if guard_response is not None:
            async def guard_event_generator():
                yield sse_status(req, "received")
                await wait_for_min_query_latency(request_started_at)
                yield sse_event({"type": "sources", "sources": []})
                yield sse_event({"type": "meta", "model": guard_response.model, "mode": guard_response.mode, "answer_scope": guard_response.answer_scope})
                yield sse_event({"type": "token", "content": guard_response.answer})
                yield sse_event({"type": "done"})

            return StreamingResponse(
                guard_event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        contextual_question = resolve_employee_context_question(req, req.question)
        directory_response = await asyncio.to_thread(
            employee_directory_query_response,
            req,
            current_employee,
            question=contextual_question,
        )
        if directory_response is not None:
            logger.info(
                "Streaming employee directory answer hit mode=%s language=%s",
                req.mode,
                req.ui_language,
            )

            async def directory_event_generator():
                import json

                yield sse_status(req, "received")
                yield sse_status(req, "hr_directory")
                await set_cached_query_response(cache_key, directory_response)
                await wait_for_min_query_latency(request_started_at)
                yield sse_status(req, "finalizing")
                yield sse_event({"type": "sources", "sources": directory_response.sources})
                yield sse_event({"type": "meta", "model": directory_response.model, "mode": directory_response.mode, "answer_scope": directory_response.answer_scope})
                yield sse_event({"type": "token", "content": directory_response.answer})
                yield sse_event({"type": "done"})

            return StreamingResponse(
                directory_event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        localized_req = await localize_query_request(req)
        localized_contextual_question = resolve_employee_context_question(
            localized_req,
            localized_req.question,
        )
        if localized_contextual_question != localized_req.question:
            localized_req = localized_req.model_copy(
                update={"question": localized_contextual_question}
            )
        current_user_context = await asyncio.to_thread(
            employee_context_for_query,
            localized_req,
            current_employee,
        )
        directory_response = await asyncio.to_thread(
            employee_directory_query_response,
            req,
            current_employee,
            question=localized_contextual_question,
        )
        if directory_response is not None:
            logger.info(
                "Streaming localized employee directory answer hit mode=%s language=%s",
                req.mode,
                req.ui_language,
            )

            async def localized_directory_event_generator():
                import json

                yield sse_status(req, "received")
                yield sse_status(req, "hr_directory")
                await set_cached_query_response(cache_key, directory_response)
                await wait_for_min_query_latency(request_started_at)
                yield sse_status(req, "finalizing")
                yield sse_event({"type": "sources", "sources": directory_response.sources})
                yield sse_event({"type": "meta", "model": directory_response.model, "mode": directory_response.mode, "answer_scope": directory_response.answer_scope})
                yield sse_event({"type": "token", "content": directory_response.answer})
                yield sse_event({"type": "done"})

            return StreamingResponse(
                localized_directory_event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
    except TranslationError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    async def event_generator():
        import json
        try:
            yield sse_status(req, "received")
            yield sse_status(req, "routing")
            if try_parse_email_send_command(localized_req.question) is not None:
                yield sse_status(req, "email")
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
                yield sse_status(req, "finalizing")
                yield sse_event({"type": "sources", "sources": translated_email_sources})
                yield sse_event({"type": "meta", "model": email_response.model, "mode": email_response.mode, "answer_scope": email_response.answer_scope})
                yield sse_event({"type": "token", "content": translated_email_answer})
                yield sse_event({"type": "done"})
                return

            if req.ui_language == "ja":
                yield sse_status(req, query_processing_status_key(localized_req))
                answer, results, routed_model, answer_scope = await route_query(
                    localized_req,
                    current_user_context=current_user_context,
                )
                yield sse_status(req, "translation")
                translated_answer = await translate_answer_for_ui(answer, req)
                sources = await translate_sources_for_ui(
                    rag_pipeline.format_sources(results),
                    req,
                )
                await wait_for_min_query_latency(request_started_at)
                yield sse_status(req, "finalizing")
                yield sse_event({"type": "sources", "sources": sources})
                yield sse_event({"type": "meta", "model": routed_model, "mode": req.mode, "answer_scope": answer_scope})
                yield sse_event({"type": "token", "content": translated_answer})
                yield sse_event({"type": "done"})
                await set_cached_query_response(
                    cache_key,
                    QueryResponse(
                        answer=translated_answer,
                        sources=sources,
                        session_id=req.session_id,
                        model=routed_model,
                        mode=req.mode,
                        answer_scope=answer_scope,
                    ),
                )
                await record_query_metric(
                    mode=req.mode,
                    ui_language=req.ui_language,
                    answer_scope=answer_scope,
                    cache_hit=False,
                    latency_ms=(time.monotonic() - request_started_at) * 1000,
                )
                return

            yield sse_status(req, query_processing_status_key(localized_req))
            token_stream, results, routed_model, answer_scope = (
                await route_query_stream(
                    localized_req,
                    current_user_context=current_user_context,
                )
            )
            
            # Gửi nguồn trích dẫn (sources) trước
            sources = rag_pipeline.format_sources(results)
            await wait_for_min_query_latency(request_started_at)
            yield sse_status(req, "finalizing")
            yield sse_event({"type": "sources", "sources": sources})
            yield sse_event({"type": "meta", "model": routed_model, "mode": req.mode, "answer_scope": answer_scope})

            # Stream từng token câu trả lời. Generator phát tuple (kind, text):
            # 'token' = delta để hiển thị dần; 'replace' = bản đã hậu xử lý,
            # chỉ xuất hiện khi cleanup đổi nội dung (ca model local lỗi).
            answer_parts = []
            final_answer = None
            async for kind, text in token_stream:
                if kind == "replace":
                    final_answer = text
                    yield sse_event({"type": "replace", "content": text})
                else:
                    answer_parts.append(text)
                    yield sse_event({"type": "token", "content": text})

            yield sse_event({"type": "done"})
            cached_answer = (
                final_answer if final_answer is not None else "".join(answer_parts)
            )
            await set_cached_query_response(
                cache_key,
                QueryResponse(
                    answer=cached_answer,
                    sources=sources,
                    session_id=req.session_id,
                    model=routed_model,
                    mode=req.mode,
                    answer_scope=answer_scope,
                ),
            )
            await record_query_metric(
                mode=req.mode,
                ui_language=req.ui_language,
                answer_scope=answer_scope,
                cache_hit=False,
                latency_ms=(time.monotonic() - request_started_at) * 1000,
            )
        except Exception as e:
            logger.error(f"Stream generation error: {e}", exc_info=True)
            await record_query_metric(
                mode=req.mode,
                ui_language=req.ui_language,
                answer_scope="error",
                cache_hit=False,
                latency_ms=(time.monotonic() - request_started_at) * 1000,
                error=True,
            )
            yield sse_event({"type": "error", "message": str(e)})

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
