"""
src/api/main.py
---------------
FastAPI API Gateway chính cho Máy 2.
Quản lý các REST API phục vụ cho RAG (Upload, Index, Query) và AI Agent (LangGraph Execution).
"""

import asyncio
import inspect
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
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load biến môi trường từ .env
load_dotenv()

from src.actions.artifact_store import ArtifactStore, StoredArtifact
from src.actions.calendar_action import (
    CalendarActionError,
    CalendarActionService,
)
from src.actions.report_agent import (
    HrExecutiveReportAgent,
    MesReportAgent,
    MesWmsReportAgent,
    render_html,
)
from src.actions.report_intent import ReportCapability, report_capability_for_mode
from src.rag.parser import (
    DocumentLimitError,
    DocumentParser,
    DocumentProcessingTimeout,
)
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.rag_pipeline import RAGPipeline
from src.rag.web_search import WebSearcher
from src.rag.media_paths import resolve_processed_image_path
from src.auth.employee_directory import EmployeeDirectory
from src.auth.employee_intent import normalize_text
from src.integrations.mes_database import MesDatabase
from src.integrations.mes_query_service import (
    MesQueryOutcome,
    MesQueryService,
    MesQueryStreamOutcome,
)
from src.integrations.mes_wms_database import MesWmsDatabase
from src.integrations.gmail_sender import (
    EmailDraft,
    EmailDraftStore,
    GmailSender,
    GmailSenderError,
    is_email_cancel_request,
    is_email_confirm_request,
    parse_email_send_command,
    try_parse_email_send_command,
)
from src.i18n.translation import TranslationError, TranslationService

from src.api import config
from src.api.config import (
    AGENT_API_KEY,
    DOCJP_COLLECTION_NAME,
    DOCJP_PAGE_IMAGE_DIR,
    DOCJP_SESSION_ID,
    EMPLOYEE_DIRECTORY_DB_PATH,
    FRONTEND_DIST,
    LOG_LEVEL,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    MES_QUERY_CACHE_TTL_SECONDS,
    MIN_QUERY_RESPONSE_SECONDS,
    REPORT_STEP_PACING_SECONDS,
    WMS_VERIFICATION_STEP_PACING_SECONDS,
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
from src.api.research_topics import (
    load_research_topic_config,
    validate_research_topic,
)
from src.api.research_cached_answers import (
    research_cached_answer_metadata,
    research_cached_query_response,
)
from src.api.schemas import (
    AgentRequest,
    AgentResponse,
    EmployeeAuthRequest,
    EmployeeAuthResponse,
    EmployeeResponse,
    QueryRequest,
    QueryResponse,
    WmsAnswerMetadata,
    ResearchDemoResponse,
    ResearchTopic,
    ResearchTopicsResponse,
    SessionInfoResponse,
    SessionResponse,
)
from src.api.sse import (
    query_processing_status_key,
    sse_agent_plan,
    sse_event,
    sse_status,
    sse_tool_result,
    sse_tool_start,
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
docjp_vector_store: Optional[VectorStore] = None
doc_parser: Optional[DocumentParser] = None
rag_pipeline: Optional[RAGPipeline] = None
mes_query_service: Optional[MesQueryService] = None
mes_report_agent: Optional[MesReportAgent] = None
mes_wms_report_agent: Optional[MesWmsReportAgent] = None
hr_report_agent: Optional[HrExecutiveReportAgent] = None
calendar_action_service: Optional[CalendarActionService] = None
artifact_store = ArtifactStore()
web_searcher: Optional[WebSearcher] = None
employee_directory = EmployeeDirectory(EMPLOYEE_DIRECTORY_DB_PATH)
mes_database = MesDatabase.from_env()
mes_wms_database = MesWmsDatabase.from_env()
gmail_sender = GmailSender.from_env()
email_draft_store = EmailDraftStore()
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
    "wms_verification": {
        "by_source_kind": defaultdict(int),
        "by_outcome": defaultdict(int),
        "duration_ms": deque(maxlen=500),
        "snapshot_validation_ms": deque(maxlen=500),
        "answer_validation_ms": deque(maxlen=500),
        "presentation_pacing_ms": deque(maxlen=500),
    },
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
    global embedder, vector_store, mkac_vector_store, docjp_vector_store
    global doc_parser, rag_pipeline
    global mes_query_service, mes_report_agent, mes_wms_report_agent, hr_report_agent
    global calendar_action_service
    global web_searcher

    logger.info("🚀 Starting Meibook API Gateway on Machine 2...")

    # Khởi tạo Vector DB trước
    vector_store = VectorStore(host=QDRANT_HOST, port=QDRANT_PORT)
    mkac_vector_store = VectorStore(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=os.getenv("MKAC_COLLECTION_NAME", "mkac_knowledge"),
    )
    docjp_vector_store = VectorStore(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=DOCJP_COLLECTION_NAME,
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
        docjp_vector_store=docjp_vector_store,
        web_searcher=web_searcher,
        mes_database=mes_database,
        mes_wms_database=mes_wms_database,
    )
    mes_query_service = MesQueryService(
        mes_database=mes_database,
        mes_sql_agent=rag_pipeline.mes_sql_agent,
        mes_wms_database=mes_wms_database,
    )
    mes_report_agent = MesReportAgent(rag_pipeline.mes_sql_agent)
    mes_wms_report_agent = MesWmsReportAgent(mes_wms_database)
    hr_report_agent = HrExecutiveReportAgent(employee_directory)
    calendar_action_service = CalendarActionService(
        tool_runner=rag_pipeline.run_calendar_tool,
        planner=rag_pipeline.plan_calendar_event,
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
    # Shared static corpora can use a longer mode-specific TTL. Uploaded
    # Research sessions never reach this function because their cache key is
    # disabled in query_cache_key().
    if response.mode == "mes":
        ttl = MES_QUERY_CACHE_TTL_SECONDS
    elif response.mode == "research":
        ttl = config.RESEARCH_QUERY_CACHE_TTL_SECONDS
    else:
        ttl = QUERY_RESPONSE_CACHE_TTL_SECONDS
    async with query_response_cache_lock:
        query_response_cache[cache_key] = (time.monotonic() + ttl, response)
        query_response_cache.move_to_end(cache_key)
        while len(query_response_cache) > QUERY_RESPONSE_CACHE_SIZE:
            query_response_cache.popitem(last=False)


def build_query_cache_key(req: QueryRequest) -> Optional[str]:
    """Attach only the snapshot version relevant to the requested domain."""
    if calendar_action_service and calendar_action_service.is_action_request(req.question):
        return None
    if question_uses_employee_context_reference(req.question):
        return None
    # WMS answers carry contract/availability metadata that must not survive a
    # rebuild or feature-flag change inside the long MES-style response cache.
    if req.mode == "wms" or MesWmsDatabase.is_wms_question(req.question):
        return None
    snapshot_version = ""
    if req.mode == "mes" and mes_database is not None:
        try:
            snapshot_version = mes_database.snapshot_version()
        except Exception:  # pragma: no cover - defensive metadata read
            snapshot_version = ""
    return query_cache_key(req, snapshot_version=snapshot_version)


def normalize_research_request(req: QueryRequest) -> QueryRequest:
    """Resolve the Research corpus explicitly while preserving old clients."""
    if req.mode != "research":
        return req

    validated_topic = validate_research_topic(req.research_topic)
    scope = req.research_scope or ("topic" if validated_topic else "upload")
    if scope == "topic":
        if not validated_topic:
            raise HTTPException(
                status_code=400,
                detail="A valid research_topic is required for topic research.",
            )
        return req.model_copy(
            update={"research_scope": "topic", "research_topic": validated_topic}
        )
    return req.model_copy(update={"research_scope": "upload", "research_topic": None})


def question_uses_employee_context_reference(question: str) -> bool:
    normalized = normalize_text(question)
    if any(
        re.search(rf"\b{re.escape(marker)}\b", normalized)
        for marker in EMPLOYEE_CONTEXT_REFERENCE_MARKERS
    ):
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


async def record_wms_verification_metric(
    *,
    source_kind: str,
    outcome: str,
    duration_ms: float,
    snapshot_validation_ms: float,
    answer_validation_ms: float,
    presentation_pacing_ms: float,
) -> None:
    """Record aggregate WMS timing without request content."""
    async with query_metrics_lock:
        wms_metrics = query_metrics["wms_verification"]
        wms_metrics["by_source_kind"][source_kind] += 1
        wms_metrics["by_outcome"][outcome] += 1
        wms_metrics["duration_ms"].append(duration_ms)
        wms_metrics["snapshot_validation_ms"].append(snapshot_validation_ms)
        wms_metrics["answer_validation_ms"].append(answer_validation_ms)
        wms_metrics.setdefault("presentation_pacing_ms", deque(maxlen=500)).append(
            presentation_pacing_ms
        )


async def pace_wms_presentation(multiplier: float = 1.0) -> float:
    """Pause an already-safe WMS presentation milestone and report its duration."""
    started_at = time.monotonic()
    await pace_wms_verification_step(multiplier)
    return (time.monotonic() - started_at) * 1000


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


async def pace_report_step(multiplier: float = 1.0) -> None:
    """Giãn nhịp phát SSE của report cho khớp cảm giác agent đang suy luận.

    Report deterministic trả kết quả gần như tức thời; nếu phát hết event trong
    vài chục ms thì timeline và thẻ báo cáo bật ra cùng lúc. Hàm này chỉ delay
    thời điểm phát, không sinh/sửa dữ liệu báo cáo.
    """
    if REPORT_STEP_PACING_SECONDS <= 0:
        return
    await asyncio.sleep(REPORT_STEP_PACING_SECONDS * max(0.0, multiplier))


async def pace_wms_verification_step(multiplier: float = 1.0) -> None:
    """Optionally space out already-completed WMS verification milestones."""
    if WMS_VERIFICATION_STEP_PACING_SECONDS <= 0:
        return
    await asyncio.sleep(
        WMS_VERIFICATION_STEP_PACING_SECONDS * max(0.0, multiplier)
    )


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
        raise HTTPException(
            status_code=403,
            detail={
                "code": "INVALID_EMPLOYEE_ID",
                "message": "Mã nhân viên không hợp lệ.",
            },
        )
    return EmployeeResponse(
        id=employee["id"],
        name=employee["name"],
        company_email=employee.get("company_email", ""),
        gender=employee.get("gender", ""),
        position=employee.get("position", ""),
        department=employee.get("department", ""),
        greeting=employee.get("greeting", ""),
        department_size=employee.get("department_size", 0),
        department_heads=employee.get("department_heads", []),
        department_deputies=employee.get("department_deputies", []),
    )


def authorize_query(req: QueryRequest) -> Optional[EmployeeResponse]:
    if req.mode not in {"mkac", "mes", "wms"}:
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
    """Translate Japanese UI questions into Vietnamese for Vietnamese-first routes."""
    if req.ui_language != "ja" or translation_service is None:
        return req
    # Action/report detectors đọc trực tiếp marker Nhật; dịch trước có thể làm
    # lẫn từ "email" hoặc mất audience/domain marker rồi route sai. MES có bộ
    # rule Nhật riêng, Research dùng kho DocJP tiếng Nhật nên cũng giữ nguyên.
    if (
        req.mode in {"mes", "wms", "research"}
        or report_capability_for_mode(req.question, req.mode).is_report
        or is_email_confirm_request(req.question)
        or is_email_cancel_request(req.question)
        or try_parse_email_send_command(req.question) is not None
    ):
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


async def translate_answer_for_ui(
    answer: str,
    req: QueryRequest,
    *,
    answer_scope: str = "",
) -> str:
    """Translate Vietnamese backend answers back to the selected UI language."""
    if req.ui_language != "ja" or translation_service is None:
        return answer
    if answer_scope == "wms_database":
        return answer
    if req.mode == "research" and re.search(r"[\u3040-\u30ff\u3400-\u9fff]", answer or ""):
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

    # DocJP index là tiếng Nhật gốc, không cần dịch lại
    if req.mode == "research":
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


def _mes_method_kwargs(method: Any, req: QueryRequest, question: str) -> Dict[str, Any]:
    """Call legacy tuple services without masking TypeError raised inside them."""
    parameters = inspect.signature(method).parameters
    kwargs: Dict[str, Any] = {"question": question, "model": req.model}
    if "language" in parameters:
        kwargs["language"] = req.ui_language
    return kwargs


def safe_wms_metadata(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate and allowlist WMS metadata before any REST/SSE serialization."""
    if not payload:
        return None
    return WmsAnswerMetadata.model_validate(payload).model_dump()


def safe_wms_metadata_model(
    payload: Optional[Dict[str, Any]],
) -> Optional[WmsAnswerMetadata]:
    """Build the response model through the same allowlisted metadata gate."""
    safe_payload = safe_wms_metadata(payload)
    return WmsAnswerMetadata.model_validate(safe_payload) if safe_payload else None


def wms_suppressed_outcome(
    req: QueryRequest,
    *,
    intent: str,
    reason_code: str,
) -> MesQueryOutcome:
    """Return a public, data-free WMS refusal when validation cannot complete."""
    answer = (
        "WMSスナップショットまたは回答条件を検証できないため、在庫情報は表示しません。"
        if req.ui_language == "ja"
        else "Không thể xác minh WMS snapshot hoặc điều kiện trả lời nên tôi không hiển thị số liệu tồn kho."
    )
    return MesQueryOutcome(
        answer=answer,
        results=[],
        routed_model="wms-validation",
        answer_scope="wms_database",
        wms_metadata={
            "intent": intent,
            "domain": "SUPPRESSED",
            "status": "SUPPRESSED",
            "reason_codes": [reason_code],
            "source_as_of_state": "UNAVAILABLE",
            "source_timezone": "unverified",
        },
    )


def validated_wms_metadata_or_none(
    req: QueryRequest,
    outcome: MesQueryOutcome,
) -> tuple[MesQueryOutcome, Optional[Dict[str, Any]]]:
    """Never permit invalid WMS metadata to reach the stream or REST payload."""
    try:
        metadata = safe_wms_metadata(outcome.wms_metadata)
    except Exception:
        suppressed = wms_suppressed_outcome(
            req,
            intent="wms_metadata_validation_suppressed",
            reason_code="WMS_METADATA_VALIDATION_FAILED",
        )
        return suppressed, safe_wms_metadata(suppressed.wms_metadata)
    if metadata is None:
        suppressed = wms_suppressed_outcome(
            req,
            intent="wms_metadata_validation_suppressed",
            reason_code="WMS_METADATA_VALIDATION_FAILED",
        )
        return suppressed, safe_wms_metadata(suppressed.wms_metadata)
    return outcome, metadata


def wms_verification_copy(language: str) -> Dict[str, str]:
    if language == "ja":
        return {
            "title": "WMS検証ステップ",
            "snapshot": "WMSスナップショットと契約を検証",
            "answer": "回答範囲と根拠を確認",
            "snapshot_done": "スナップショットとデータ契約を確認しました",
            "snapshot_suppressed": "スナップショットが在庫情報の表示を許可していません",
            "prepared_done": "準備済みの内容が許可リストと一致しました",
            "query_done": "決定的なWMS照会の範囲を確認しました",
        }
    return {
        "title": "Các bước kiểm chứng WMS",
        "snapshot": "Kiểm tra snapshot và data contract WMS",
        "answer": "Kiểm tra phạm vi và căn cứ trả lời",
        "snapshot_done": "Đã xác minh snapshot và data contract WMS",
        "snapshot_suppressed": "Snapshot không cho phép hiển thị số liệu tồn kho",
        "prepared_done": "Nội dung chuẩn bị đã khớp allowlist",
        "query_done": "Đã xác minh phạm vi truy vấn WMS tất định",
    }


async def _wms_verification_event_generator(
    req: QueryRequest,
    *,
    request_started_at: float,
):
    """Run an isolated, data-backed WMS validation workflow over SSE.

    The events expose only completed deterministic milestones. They never expose
    raw rows, SQL, user/session identifiers, prompts, or model reasoning.
    """
    copy = wms_verification_copy(req.ui_language)
    presentation_pacing_ms = 0.0
    yield sse_status(req, "received")
    presentation_pacing_ms += await pace_wms_presentation(0.45)
    yield sse_status(req, "routing")
    presentation_pacing_ms += await pace_wms_presentation(0.45)
    yield sse_status(req, "wms")
    presentation_pacing_ms += await pace_wms_presentation(0.45)
    yield sse_agent_plan(
        title=copy["title"],
        workflow="wms_verification",
        steps=[
            {"id": "wms_snapshot", "title": copy["snapshot"]},
            {"id": "wms_answer", "title": copy["answer"]},
        ],
    )
    yield sse_tool_start(
        step_id="wms_snapshot",
        tool="validate_wms_snapshot",
        title=copy["snapshot"],
    )
    snapshot_validation_started_at = time.monotonic()
    try:
        snapshot_outcome = await route_query_outcome(req)
    except asyncio.CancelledError:
        logger.info("WMS verification cancelled during snapshot validation.")
        raise
    except Exception:
        snapshot_outcome = wms_suppressed_outcome(
            req,
            intent="wms_snapshot_validation_suppressed",
            reason_code="WMS_SNAPSHOT_QUERY_ERROR",
        )
    snapshot_outcome, metadata = validated_wms_metadata_or_none(
        req,
        snapshot_outcome,
    )
    status = str((metadata or {}).get("status") or "SUPPRESSED")
    snapshot_validation_ms = (time.monotonic() - snapshot_validation_started_at) * 1000
    presentation_pacing_ms += await pace_wms_presentation()
    yield sse_tool_result(
        step_id="wms_snapshot",
        status="done" if status in {"AVAILABLE", "PARTIAL"} else "error",
        summary=(
            copy["snapshot_done"]
            if status in {"AVAILABLE", "PARTIAL"}
            else copy["snapshot_suppressed"]
        ),
    )

    yield sse_tool_start(
        step_id="wms_answer",
        tool="validate_wms_answer_scope",
        title=copy["answer"],
    )
    answer_validation_started_at = time.monotonic()
    prepared_outcome = resolve_wms_prepared_response(req)
    if req.quick_answer_id and prepared_outcome is None:
        final_outcome = wms_prepared_validation_failure(req)
        final_summary = copy["snapshot_suppressed"]
        source_kind = "prepared"
    elif status not in {"AVAILABLE", "PARTIAL"}:
        final_outcome = snapshot_outcome
        final_summary = copy["snapshot_suppressed"]
        source_kind = "snapshot"
    elif prepared_outcome is not None:
        try:
            prepared_metadata = safe_wms_metadata(prepared_outcome.wms_metadata) or {}
        except Exception:
            prepared_metadata = {}
        contract_matches = all(
            prepared_metadata.get(key) == (metadata or {}).get(key)
            for key in (
                "contract_version",
                "data_contract_version",
                "semantic_contract_version",
                "semantic_epoch",
            )
        )
        if not contract_matches:
            final_outcome = wms_prepared_validation_failure(req)
            final_summary = copy["snapshot_suppressed"]
        else:
            final_outcome = MesQueryOutcome(
                answer=prepared_outcome.answer,
                results=[],
                routed_model=prepared_outcome.routed_model,
                answer_scope="wms_database",
                wms_metadata=metadata,
            )
            final_summary = copy["prepared_done"]
        source_kind = "prepared"
        logger.info(
            "wms_prepared_validation matched=%s contract_matches=%s",
            True,
            contract_matches,
        )
    else:
        final_outcome = snapshot_outcome
        final_summary = copy["query_done"]
        source_kind = "snapshot"

    final_outcome, final_metadata = validated_wms_metadata_or_none(
        req,
        final_outcome,
    )
    answer_validation_ms = (time.monotonic() - answer_validation_started_at) * 1000
    presentation_pacing_ms += await pace_wms_presentation()
    yield sse_tool_result(
        step_id="wms_answer",
        status=(
            "done"
            if str((final_metadata or {}).get("status")) in {"AVAILABLE", "PARTIAL"}
            else "error"
        ),
        summary=final_summary,
    )
    yield sse_status(req, "finalizing")
    await wait_for_min_query_latency(request_started_at)
    yield sse_event({"type": "sources", "sources": []})
    yield sse_event(
        {
            "type": "meta",
            "model": final_outcome.routed_model,
            "mode": "wms",
            "answer_scope": final_outcome.answer_scope,
            "wms_metadata": final_metadata,
            "workflow": "wms_verification",
            "source_kind": source_kind,
            "cache": False,
        }
    )
    yield sse_event({"type": "token", "content": final_outcome.answer})
    yield sse_event({"type": "agent_done"})
    yield sse_event({"type": "done"})
    verification_outcome = (
        "verified"
        if str((final_metadata or {}).get("status")) in {"AVAILABLE", "PARTIAL"}
        else "suppressed"
    )
    duration_ms = (time.monotonic() - request_started_at) * 1000
    logger.info(
        "wms_verification_complete source_kind=%s outcome=%s duration_ms=%d",
        source_kind,
        verification_outcome,
        int(duration_ms),
    )
    await record_wms_verification_metric(
        source_kind=source_kind,
        outcome=verification_outcome,
        duration_ms=duration_ms,
        snapshot_validation_ms=snapshot_validation_ms,
        answer_validation_ms=answer_validation_ms,
        presentation_pacing_ms=presentation_pacing_ms,
    )
    await record_query_metric(
        mode=req.mode,
        ui_language=req.ui_language,
        answer_scope=final_outcome.answer_scope,
        cache_hit=False,
        latency_ms=duration_ms,
    )


async def wms_verification_event_generator(
    req: QueryRequest,
    *,
    request_started_at: float,
):
    """Stop cleanly if the SSE task is cancelled before final payloads."""
    try:
        async for event in _wms_verification_event_generator(
            req,
            request_started_at=request_started_at,
        ):
            yield event
    except asyncio.CancelledError:
        logger.info("WMS verification stream cancelled before completion.")
        raise


async def route_query_outcome(
    req: QueryRequest,
    *,
    question: Optional[str] = None,
    current_user_context: Optional[Dict[str, Any]] = None,
) -> MesQueryOutcome:
    """Route by mode and preserve additive WMS metadata."""
    ensure_query_services_ready()
    routed_question = question or req.question
    if req.mode == "wms":
        logger.info("Routing query to isolated WMS service.")
        wms_method = getattr(mes_query_service, "query_wms_outcome", None)
        if not callable(wms_method):
            raise HTTPException(
                status_code=503,
                detail="WMS query service is not available.",
            )
        return await wms_method(
            question=routed_question,
            model=req.model,
            language=req.ui_language,
        )
    if req.mode == "mes":
        routed_question = resolve_mes_context_question(req, routed_question)
        logger.info("Routing query to MES service.")
        if hasattr(mes_query_service, "query_outcome"):
            return await mes_query_service.query_outcome(
                question=routed_question,
                model=req.model,
                language=req.ui_language,
            )
        answer, results, model, scope = await mes_query_service.query(
            **_mes_method_kwargs(mes_query_service.query, req, routed_question)
        )
        return MesQueryOutcome(answer, results, model, scope)
    if req.mode in {"mkac", "research"}:
        logger.info("Routing query to %s RAG service.", req.mode)
        answer, results, model, scope = await rag_pipeline.query(
            session_id=req.session_id,
            question=routed_question,
            model=req.model,
            mode=req.mode,
            current_user=current_user_context,
            conversation_context=req.conversation_context,
            research_topic=validate_research_topic(req.research_topic),
            research_scope=req.research_scope,
        )
        return MesQueryOutcome(answer, results, model, scope)
    raise HTTPException(status_code=400, detail=f"Unsupported query mode: {req.mode}")


async def route_query(
    req: QueryRequest,
    *,
    question: Optional[str] = None,
    current_user_context: Optional[Dict[str, Any]] = None,
) -> tuple[str, list, str, str]:
    return (
        await route_query_outcome(
            req,
            question=question,
            current_user_context=current_user_context,
        )
    ).as_tuple()


async def route_query_stream_outcome(
    req: QueryRequest,
    *,
    current_user_context: Optional[Dict[str, Any]] = None,
) -> MesQueryStreamOutcome:
    """Streaming route preserving additive WMS metadata."""
    ensure_query_services_ready()
    if req.mode == "wms":
        logger.info("Routing streaming query to isolated WMS service.")
        wms_method = getattr(
            mes_query_service,
            "query_wms_stream_outcome",
            None,
        )
        if not callable(wms_method):
            raise HTTPException(
                status_code=503,
                detail="WMS streaming query service is not available.",
            )
        return await wms_method(
            question=req.question,
            model=req.model,
            language=req.ui_language,
        )
    if req.mode == "mes":
        routed_question = resolve_mes_context_question(req, req.question)
        logger.info("Routing streaming query to MES service.")
        if hasattr(mes_query_service, "query_stream_outcome"):
            return await mes_query_service.query_stream_outcome(
                question=routed_question,
                model=req.model,
                language=req.ui_language,
            )
        stream, results, model, scope = await mes_query_service.query_stream(
            **_mes_method_kwargs(
                mes_query_service.query_stream,
                req,
                routed_question,
            )
        )
        return MesQueryStreamOutcome(stream, results, model, scope)
    if req.mode in {"mkac", "research"}:
        logger.info("Routing streaming query to %s RAG service.", req.mode)
        stream, results, model, scope = await rag_pipeline.query_stream(
            session_id=req.session_id,
            question=req.question,
            model=req.model,
            mode=req.mode,
            current_user=current_user_context,
            conversation_context=req.conversation_context,
            research_topic=validate_research_topic(req.research_topic),
            research_scope=req.research_scope,
        )
        return MesQueryStreamOutcome(stream, results, model, scope)
    raise HTTPException(status_code=400, detail=f"Unsupported query mode: {req.mode}")


async def route_query_stream(
    req: QueryRequest,
    *,
    current_user_context: Optional[Dict[str, Any]] = None,
):
    return (
        await route_query_stream_outcome(
            req, current_user_context=current_user_context
        )
    ).as_tuple()


async def store_report_artifact(
    report: Dict[str, Any],
    req: Optional[QueryRequest] = None,
) -> None:
    """Persist full HTML while keeping SSE payloads metadata-only."""
    report_id = str(report["id"])
    html_content = report.get("html_content")
    has_inline_html = isinstance(html_content, str) and bool(html_content.strip())
    content = html_content if has_inline_html else render_html(report)
    report_type = str(report.get("report_type") or "mes_report")
    filename_prefix = (
        "wms-report" if report_type == "wms_executive_report"
        else "hr-report" if report_type == "hr_executive_report"
        else "mes-report"
    )
    session_id = req.session_id if req else ""
    employee_id = req.employee_id or "" if req else ""
    await artifact_store.put(
        StoredArtifact(
            id=report_id,
            kind="report_html",
            content=content,
            media_type="text/html; charset=utf-8",
            filename=f"{filename_prefix}-{report_id[:8]}.html",
            meta={
                "session_id": session_id,
                "employee_id": employee_id,
                "report_type": report_type,
                "title": report["title"],
            },
            session_id=session_id,
            employee_id=employee_id,
        )
    )


def report_artifact_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return the small allowlisted report card contract sent over SSE."""
    allowed = (
        "id",
        "report_type",
        "title",
        "generated_at",
        "period_label",
        "kpis",
        "charts",
        "matrices",
        "observations",
        "governance",
        "limitations",
        "sections",
    )
    payload = {key: report[key] for key in allowed if key in report}
    if "charts" in payload:
        payload["charts"] = [
            {key: value for key, value in chart.items() if key != "svg"}
            for chart in payload["charts"]
        ]
    payload["download_url"] = f"/reports/{report['id']}"
    return payload


async def handle_calendar_action_query(
    req: QueryRequest,
    current_employee: Optional[EmployeeResponse],
) -> Optional[QueryResponse]:
    if req.mode != "mkac" or calendar_action_service is None:
        return None
    if not calendar_action_service.is_action_request(req.question):
        return None
    if current_employee is None:
        raise CalendarActionError("Cần đăng nhập nhân viên để sử dụng Calendar.")
    try:
        result = await calendar_action_service.handle(
            session_id=req.session_id,
            question=req.question,
            employee=current_employee,
        )
    except CalendarActionError as exc:
        return QueryResponse(
            answer=str(exc),
            sources=[],
            session_id=req.session_id,
            model="calendar-agent",
            mode=req.mode,
            answer_scope="calendar_error",
        )
    if result is None:
        return None
    return QueryResponse(
        answer=result.answer,
        sources=[],
        session_id=req.session_id,
        model="calendar-agent",
        mode=req.mode,
        answer_scope=(
            "calendar_event"
            if result.kind == "created"
            else "calendar"
            if result.kind == "availability"
            else "calendar_error"
            if result.kind in {"conflict", "missing"}
            else "calendar_draft"
        ),
    )


def report_refusal_response(
    req: QueryRequest,
    capability: ReportCapability,
) -> QueryResponse:
    """Từ chối report ngoài capability như một câu trả lời có chủ đích."""
    if capability.shape == "mode_mismatch":
        expected_mode = {
            "hr": "HCNS",
            "mes": "MES",
            "wms": "WMS",
        }.get(capability.domain, "phù hợp")
        answer = (
            f"このレポートは{expected_mode}モードでのみ作成できます。"
            f"現在の{req.mode.upper()}モードでは別領域のレポートを実行しません。"
            if req.ui_language == "ja"
            else (
                f"Báo cáo này chỉ được tạo trong chế độ {expected_mode}. "
                f"Hệ thống không chạy báo cáo khác lĩnh vực trong tab "
                f"{req.mode.upper()}; vui lòng chuyển đúng chế độ rồi thử lại."
            )
        )
    elif not capability.domain:
        answer = (
            "レポートの対象領域が不明です。人事（HR）、MES品質・エラー、"
            "またはWMS工程在庫のいずれかを明示してください。\n\n"
            "例: 「WMS工程在庫の概要レポートを作成」「MESエラーレポートを作成」"
            if req.ui_language == "ja"
            else (
                "Yêu cầu chưa nêu rõ lĩnh vực báo cáo nên tôi không tự chọn nguồn "
                "dữ liệu để tránh trả sai. Bạn hãy nêu kèm lĩnh vực:\n\n"
                "- Nhân sự: \"Báo cáo tổng quan nhân sự\"\n"
                "- Chất lượng MES: \"Báo cáo tổng hợp lỗi MES\"\n"
                "- Tồn kho WMS: \"Báo cáo tổng quan tồn kho WMS\""
            )
        )
    elif capability.domain == "hr":
        answer = (
            "現在のHR Reportでは、現行人事ディレクトリの組織概要のみ対応しています。"
            "期間別、比較、給与・個人KPI・採用・勤怠などのレポートはサポートされていません。"
            if req.ui_language == "ja"
            else (
                "HR Report hiện chỉ hỗ trợ tổng quan danh bạ nhân sự hiện tại. "
                "Hệ thống chưa hỗ trợ báo cáo theo kỳ, so sánh, chi phí lương, "
                "KPI cá nhân, tuyển dụng hay chấm công."
            )
        )
    elif capability.domain == "wms":
        answer = (
            "現在のWMS contract v4では、現行残高（current balance）の概要レポートのみ対応しています。"
            "KPI集計、期間比較、またはカスタムフィルター付きのレポートはサポートされていません。"
            if req.ui_language == "ja"
            else (
                "WMS contract v4 hiện chỉ hỗ trợ báo cáo tổng quan current balance. "
                "Hệ thống chưa hỗ trợ báo cáo KPI, so sánh theo kỳ hoặc bộ lọc tùy biến."
            )
        )
    elif req.ui_language == "ja":
        answer = (
            "現在のReport Agentでは、この形式のレポートを正確に作成できません。"
            "対応しているのは、1期間のMESエラー標準集計レポート、またはTop Nエラー種類"
            "レポートのみです。誤ったレポートを返さないため、標準テンプレートへの置き換えは"
            "行いません。"
        )
    else:
        answer = (
            "Report Agent hiện chưa thể tạo chính xác dạng báo cáo này. Hệ thống tạm "
            "thời chỉ hỗ trợ báo cáo tổng hợp lỗi MES chuẩn hoặc báo cáo Top N loại lỗi "
            "trong một kỳ. Tôi không tự đổi yêu cầu sang mẫu mặc định để tránh trả sai."
        )
    if (
        capability.reason
        and req.ui_language != "ja"
        and capability.shape != "mode_mismatch"
        and capability.domain not in {"", "hr", "wms"}
    ):
        answer += f"\n\nLý do: {capability.reason}"
    return QueryResponse(
        answer=answer,
        sources=[],
        session_id=req.session_id,
        model="report-agent",
        mode=req.mode,
        answer_scope="mes_report_unsupported",
    )


async def handle_report_query(req: QueryRequest) -> Optional[QueryResponse]:
    """Non-streaming report path với capability gate fail-closed."""
    capability = report_capability_for_mode(req.question, req.mode)
    if not capability.is_report:
        return None
    if not capability.supported:
        return report_refusal_response(req, capability)
    if capability.shape == "hr_executive":
        if hr_report_agent is None or not hr_report_agent.available:
            raise HTTPException(
                status_code=503,
                detail="HR Report Agent chưa sẵn sàng vì danh bạ nhân sự chưa khả dụng.",
            )
        report, summary = await hr_report_agent.build_report(
            req.question,
            language=req.ui_language,
        )
        await store_report_artifact(report, req)
        return QueryResponse(
            answer=summary,
            sources=[],
            session_id=req.session_id,
            model="report-agent",
            mode=req.mode,
            answer_scope="hr_executive_report",
            artifact=report_artifact_payload(report),
        )
    if capability.shape == "wms_executive":
        if mes_wms_report_agent is None or not mes_wms_report_agent.available:
            raise HTTPException(
                status_code=503,
                detail="WMS Report Agent chưa sẵn sàng vì WMS snapshot chưa khả dụng.",
            )
        report, summary = await mes_wms_report_agent.generate_report(
            req.question,
            language=req.ui_language,
        )
        await store_report_artifact(report, req)
        return QueryResponse(
            answer=summary,
            sources=[],
            session_id=req.session_id,
            model="report-agent",
            mode=req.mode,
            answer_scope="wms_executive_report",
            artifact=report_artifact_payload(report),
        )
    if mes_report_agent is None or not mes_report_agent.available:
        raise HTTPException(
            status_code=503,
            detail="Report Agent chưa sẵn sàng vì MES snapshot/SQL Agent chưa khả dụng.",
        )
    report, summary = await mes_report_agent.build_report(
        req.question, language=req.ui_language
    )
    await store_report_artifact(report, req)
    return QueryResponse(
        answer=f"{summary}\n\n{report['markdown']}",
        sources=[],
        session_id=req.session_id,
        model="report-agent",
        mode=req.mode,
        answer_scope="mes_report",
        artifact=report_artifact_payload(report),
    )


async def handle_email_send_query(
    req: QueryRequest,
    current_user_context: Optional[Dict[str, Any]],
) -> Optional[QueryResponse]:
    # Research là luồng hỏi đáp tài liệu DocJP; nhiều tài liệu có từ "mail/email"
    # nhưng không phải lệnh gửi Gmail thật. Chỉ xử lý Gmail action ở các mode
    # vận hành chính như HR/MES để tránh bắt nhầm intent trong bộ tài liệu.
    if req.mode == "research":
        return None

    japanese = req.ui_language == "ja"

    if is_email_confirm_request(req.question):
        if gmail_sender is None or not gmail_sender.available:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Gmail send chưa sẵn sàng. Hãy kiểm tra GMAIL_SEND_ENABLED, "
                    "GMAIL_CREDENTIALS_PATH và token OAuth."
                ),
            )
        draft, claimed = await email_draft_store.claim_for_send(
            req.session_id,
            req.employee_id or "",
        )
        if draft is None:
            answer = (
                "送信待ちのメール下書きがありません。先に「...へメールを送信」と指示してください。"
                if japanese
                else (
                    "Chưa có bản nháp email nào đang chờ gửi. Bạn hãy nhập câu lệnh "
                    "gửi email trước, ví dụ: gửi email báo cáo này cho a@mkac.vn ..."
                )
            )
            return QueryResponse(
                answer=answer,
                sources=[],
                session_id=req.session_id,
                model=req.model,
                mode=req.mode,
                answer_scope="email_action",
            )
        if draft.employee_id != (req.employee_id or ""):
            raise HTTPException(status_code=403, detail="Bản nháp email không thuộc về phiên làm việc này.")
        if draft.status == "sent":
            answer = (
                f"メールは送信済みです（{draft.to_email}）。"
                if japanese
                else f"Email này đã được gửi trước đó tới {draft.to_email}."
            )
            return QueryResponse(
                answer=answer,
                sources=[],
                session_id=req.session_id,
                model=req.model,
                mode=req.mode,
                answer_scope="email_action",
            )
        if not claimed:
            answer = (
                "メール送信を処理中です。重複送信は行いません。"
                if japanese
                else "Email đang được gửi. Hệ thống sẽ không gửi trùng."
            )
            return QueryResponse(
                answer=answer,
                sources=[],
                session_id=req.session_id,
                model=req.model,
                mode=req.mode,
                answer_scope="email_action",
            )

        attachments = []
        if draft.artifact_id:
            artifact = await artifact_store.get(draft.artifact_id)
            if artifact is None:
                await email_draft_store.update_status(req.session_id, status="pending")
                raise GmailSenderError(
                    "Báo cáo HTML gắn kèm đã hết hạn hoặc không tồn tại. Vui lòng tạo lại báo cáo."
                )
            if (
                artifact.session_id != req.session_id
                or artifact.employee_id != (req.employee_id or "")
            ):
                await email_draft_store.update_status(req.session_id, status="pending")
                raise HTTPException(status_code=403, detail="Báo cáo không thuộc quyền sở hữu của người dùng hiện tại.")
            attachments.append(
                {
                    "filename": artifact.filename,
                    "content": artifact.content,
                    "media_type": artifact.media_type,
                }
            )

        try:
            send_result = await asyncio.to_thread(
                gmail_sender.send_email,
                draft.to_email,
                draft.subject,
                draft.body_text,
                attachments=attachments,
            )
        except Exception:
            await email_draft_store.update_status(req.session_id, status="pending")
            raise
        await email_draft_store.update_status(
            req.session_id,
            status="sent",
            message_id=send_result.message_id,
        )
        attachment_note = (
            f" (添付: {draft.filename})" if draft.filename and japanese
            else f" (đã đính kèm file {draft.filename})" if draft.filename
            else ""
        )
        answer = (
            f"メールを {send_result.to_email} へ送信しました{attachment_note}。"
            if japanese
            else f"Đã gửi email tới {send_result.to_email} với tiêu đề \"{send_result.subject}\"{attachment_note}."
        )
        return QueryResponse(
            answer=answer,
            sources=[],
            session_id=req.session_id,
            model=req.model,
            mode=req.mode,
            answer_scope="email_action",
        )

    if is_email_cancel_request(req.question):
        draft = await email_draft_store.discard(req.session_id)
        answer = (
            "メール下書きをキャンセルしました。"
            if japanese
            else "Đã hủy bản nháp email."
        )
        return QueryResponse(
            answer=answer,
            sources=[],
            session_id=req.session_id,
            model=req.model,
            mode=req.mode,
            answer_scope="email_action",
        )

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

    artifact_id = ""
    filename = ""
    media_type = ""
    previous_context = latest_assistant_context(req.conversation_context)

    if is_context_reference(command.data_question):
        if previous_context is None:
            raise GmailSenderError(
                "Chưa có nội dung trước đó để gửi. Hãy hỏi lấy kết quả hoặc lập báo cáo trước."
            )
        target_artifact_id = str(previous_context.get("artifact_id") or "")
        target_artifact_type = str(previous_context.get("artifact_type") or "")
        if target_artifact_type in {
            "mes_report",
            "wms_executive_report",
            "hr_executive_report",
        } and not target_artifact_id:
            raise GmailSenderError(
                "Báo cáo HTML đã hết ngữ cảnh hoặc không có mã artifact. Vui lòng tạo lại báo cáo."
            )
        if target_artifact_id:
            artifact = await artifact_store.get(target_artifact_id)
            if artifact is None:
                raise GmailSenderError(
                    "Báo cáo HTML đã hết hạn hoặc không tồn tại. Vui lòng tạo lại báo cáo."
                )
            if (
                artifact.session_id != req.session_id
                or artifact.employee_id != (req.employee_id or "")
            ):
                raise HTTPException(status_code=403, detail="Báo cáo không thuộc về phiên hiện tại.")
            artifact_id = artifact.id
            filename = artifact.filename
            media_type = artifact.media_type
            subject = f"Báo cáo Meibook - {artifact.meta.get('title', 'Executive Report')}"
            body_text = build_direct_email_body(
                original_question=req.question,
                body=f"Đính kèm báo cáo HTML {artifact.meta.get('title', '')}."
            )
        else:
            body_text = build_email_body(
                original_question=req.question,
                data_question=command.data_question,
                answer=previous_context["content"],
                answer_scope=previous_context["answer_scope"],
            )
            subject = command.subject
    elif command.has_explicit_body:
        subject = command.subject
        body_text = build_direct_email_body(
            original_question=req.question,
            body=command.explicit_body,
        )
    else:
        answer, results, routed_model, answer_scope = await route_query(
            req,
            question=command.data_question,
            current_user_context=current_user_context,
        )
        subject = command.subject
        body_text = build_email_body(
            original_question=req.question,
            data_question=command.data_question,
            answer=answer,
            answer_scope=answer_scope,
        )

    draft_id = str(uuid.uuid4())
    draft = EmailDraft(
        id=draft_id,
        session_id=req.session_id,
        employee_id=req.employee_id or "",
        to_email=command.to_email,
        subject=subject,
        body_text=body_text,
        artifact_id=artifact_id,
        filename=filename,
        media_type=media_type,
        status="pending",
    )
    await email_draft_store.put(draft)

    attachment_line = f"\n- File đính kèm: {filename}" if filename else ""
    if japanese:
        answer = (
            f"**メール下書きを作成しました**\n\n"
            f"- 送信先: `{command.to_email}`\n"
            f"- 件名: {subject}\n"
            f"{f'- 添付: {filename}' if filename else ''}\n"
            f"送信するには **「送信を確定」** と入力し、中止するには **「送信をキャンセル」** と入力してください。"
        )
    else:
        answer = (
            f"**Đã chuẩn bị bản nháp email**\n\n"
            f"- Người nhận: `{command.to_email}`\n"
            f"- Tiêu đề: {subject}"
            f"{attachment_line}\n\n"
            f"Nhập **\"Xác nhận gửi email\"** để gửi đi, hoặc **\"Hủy gửi email\"** để hủy."
        )
    return QueryResponse(
        answer=answer,
        sources=[],
        session_id=req.session_id,
        model=req.model,
        mode=req.mode,
        answer_scope="email_action",
    )


# ──────────────────────────────────────────────
# Report Agent artifacts
# ──────────────────────────────────────────────

@app.get("/reports/{report_id}")
async def download_report(report_id: str):
    """Tải artifact HTML của Report Agent; artifact hết hạn sau vài giờ/restart."""
    try:
        normalized_id = str(uuid.UUID(report_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid report ID.") from exc
    artifact = await artifact_store.get(normalized_id)
    if artifact is None or artifact.kind != "report_html":
        raise HTTPException(status_code=404, detail="Report not found or expired.")
    return Response(
        content=artifact.content,
        media_type=artifact.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{artifact.filename}"',
            "Cache-Control": "private, max-age=300",
            "X-Content-Type-Options": "nosniff",
        },
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
            "wms_verification": {
                "by_source_kind": dict(
                    query_metrics["wms_verification"]["by_source_kind"]
                ),
                "by_outcome": dict(query_metrics["wms_verification"]["by_outcome"]),
                "duration_ms": {
                    "count": len(query_metrics["wms_verification"]["duration_ms"]),
                    "p50": _percentile(
                        list(query_metrics["wms_verification"]["duration_ms"]),
                        50,
                    ),
                    "p95": _percentile(
                        list(query_metrics["wms_verification"]["duration_ms"]),
                        95,
                    ),
                },
                "snapshot_validation_ms": {
                    "count": len(query_metrics["wms_verification"]["snapshot_validation_ms"]),
                    "p50": _percentile(
                        list(query_metrics["wms_verification"]["snapshot_validation_ms"]),
                        50,
                    ),
                    "p95": _percentile(
                        list(query_metrics["wms_verification"]["snapshot_validation_ms"]),
                        95,
                    ),
                },
                "answer_validation_ms": {
                    "count": len(query_metrics["wms_verification"]["answer_validation_ms"]),
                    "p50": _percentile(
                        list(query_metrics["wms_verification"]["answer_validation_ms"]),
                        50,
                    ),
                    "p95": _percentile(
                        list(query_metrics["wms_verification"]["answer_validation_ms"]),
                        95,
                    ),
                },
                "presentation_pacing_ms": {
                    "count": len(
                        query_metrics["wms_verification"]["presentation_pacing_ms"]
                    ),
                    "p50": _percentile(
                        list(
                            query_metrics["wms_verification"][
                                "presentation_pacing_ms"
                            ]
                        ),
                        50,
                    ),
                    "p95": _percentile(
                        list(
                            query_metrics["wms_verification"][
                                "presentation_pacing_ms"
                            ]
                        ),
                        95,
                    ),
                },
            },
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
        "mes_wms_database": (
            mes_wms_database.status()
            if mes_wms_database is not None
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


@app.get("/research/topics", response_model=ResearchTopicsResponse)
async def research_topics_status():
    """Return the predefined research topic groups with index statistics."""
    registry = load_research_topic_config()
    if docjp_vector_store is None or not registry["topics"]:
        return ResearchTopicsResponse(
            ready=False,
            collection=registry["collection"],
            session_id=registry["session_id"],
            default_topic=registry["default_topic"],
            allow_all=registry["allow_all"],
            topics=[],
        )

    def build_topics() -> list[ResearchTopic]:
        topics: list[ResearchTopic] = []
        for item in registry["topics"]:
            category = item.get("category")
            info = docjp_vector_store.get_session_info(
                DOCJP_SESSION_ID,
                metadata_filters={"category": category} if category else None,
            )
            topics.append(
                ResearchTopic(
                    id=item["id"],
                    category=category,
                    label_vi=item.get("label_vi", item["id"]),
                    label_ja=item.get("label_ja", item["id"]),
                    short_label_vi=item.get("short_label_vi", ""),
                    short_label_ja=item.get("short_label_ja", ""),
                    description_vi=item.get("description_vi", ""),
                    description_ja=item.get("description_ja", ""),
                    icon=item.get("icon", "file_text"),
                    accent=item.get("accent", "neutral"),
                    ready=bool(info),
                    num_files=(info or {}).get("num_files", 0),
                    num_chunks=(info or {}).get("num_chunks", 0),
                    files=(info or {}).get("files", []),
                    quick_prompts_vi=item.get("quick_prompts_vi", []),
                    quick_prompts_ja=item.get("quick_prompts_ja", []),
                )
            )
        return topics

    # get_session_info scroll đồng bộ qua Qdrant — đẩy sang thread để không
    # chặn event loop khi collection lớn.
    topics = await asyncio.to_thread(build_topics)
    return ResearchTopicsResponse(
        ready=any(topic.ready for topic in topics),
        collection=registry["collection"],
        session_id=registry["session_id"],
        default_topic=registry["default_topic"],
        allow_all=registry["allow_all"],
        topics=topics,
    )


@app.get("/sources/preview")
async def source_page_preview(
    session_id: str,
    mode: Literal["mkac", "research"],
    file: str,
    page: int,
    language: Literal["vi", "ja"] = "vi",
    source_scope: Optional[Literal["topic", "upload"]] = None,
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
    elif source_scope == "upload":
        store = vector_store
        lookup_session_id = normalize_session_id(session_id)
    else:
        store = docjp_vector_store
        lookup_session_id = DOCJP_SESSION_ID

    if store is None:
        preview_error(503, "Vector store is not ready.", "ベクトルデータベースはまだ準備できていません。")

    image_path_str = store.get_page_image_path(lookup_session_id, filename, page)
    if not image_path_str:
        preview_error(404, "Preview image not found.", "プレビュー画像が見つかりません。")

    # Re-root path đã lưu (tuyệt đối lúc index) về CWD hiện tại. Dùng chung
    # helper với format_sources để has_page_preview và endpoint luôn khớp.
    resolved = resolve_processed_image_path(image_path_str)
    resolved_path = resolved.resolve() if resolved else None

    allowed_roots = [
        UPLOAD_DIR.resolve(),
        MKAC_PAGE_IMAGE_DIR.resolve(),
        DOCJP_PAGE_IMAGE_DIR.resolve(),
    ]
    if resolved_path is None or not any(
        path_is_inside(resolved_path, root) for root in allowed_roots
    ):
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
    if re.search(r"\bdan\b", normalized) and any(
        marker in normalized
        for marker in (
            "do anything now",
            "jailbreak",
            "act as dan",
            "as dan",
            "che do dan",
            "vai tro dan",
            "dong vai dan",
        )
    ):
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


def _wms_prepared_candidates(item: dict[str, Any], language: str) -> list[str]:
    question_field = "question_ja" if language == "ja" else "question"
    aliases_field = "aliases_ja" if language == "ja" else "aliases"
    candidates = [str(item.get(question_field) or "")]
    aliases = item.get(aliases_field, [])
    if isinstance(aliases, list):
        candidates.extend(str(alias) for alias in aliases)
    return [candidate for candidate in candidates if candidate]


def resolve_wms_prepared_response(req: QueryRequest) -> Optional[MesQueryOutcome]:
    """Resolve a server-prepared WMS answer only after API authorization.

    ``quick_answer_id`` is an untrusted browser hint. An entry is accepted only
    when it is explicitly server-prepared, has a revision/provenance declaration,
    and its canonical localized question still matches the request. The same
    canonical-question match supports older clients that do not send the ID.
    """
    if req.mode != "wms":
        return None

    question_key = normalize_prepared_question(req.question)
    for item in _load_quick_answers().get("wms", []):
        if item.get("hidden") or item.get("execution") != "server_prepared":
            continue
        item_id = str(item.get("id") or "")
        if req.quick_answer_id and req.quick_answer_id != item_id:
            continue
        if not item_id or not str(item.get("revision") or "").strip():
            continue
        provenance = item.get("provenance")
        if not isinstance(provenance, dict):
            continue
        if not all(
            str(provenance.get(field) or "").strip()
            for field in ("data_contract_version", "semantic_contract_version")
        ):
            continue
        candidates = _wms_prepared_candidates(item, req.ui_language)
        if not any(
            normalize_prepared_question(candidate) == question_key
            for candidate in candidates
        ):
            continue
        answer_field = "answer_ja" if req.ui_language == "ja" else "answer"
        answer = str(item.get(answer_field) or "").strip()
        if not answer:
            continue
        return MesQueryOutcome(
            answer=answer,
            results=[],
            routed_model="wms-prepared",
            answer_scope="wms_database",
            wms_metadata={
                "contract_version": str(provenance.get("contract_version") or ""),
                "data_contract_version": str(
                    provenance.get("data_contract_version") or ""
                ),
                "semantic_contract_version": str(
                    provenance.get("semantic_contract_version") or ""
                ),
                "intent": "wms_server_prepared",
                "domain": "CURRENT_BALANCE",
                "status": "PARTIAL",
                "reason_codes": ["UOM_MASTER_UNAVAILABLE"],
                "imported_at": "",
                "source_as_of": "",
                "source_as_of_state": "UNAVAILABLE",
                "source_as_of_basis": "",
                "source_timezone": "unverified",
                "semantic_epoch": str(provenance.get("semantic_epoch") or ""),
                "dataset_evidence": [],
                "grain": "process_id,item_code",
                "pagination": None,
            },
        )
    return None


def wms_prepared_validation_failure(req: QueryRequest) -> Optional[MesQueryOutcome]:
    """Fail closed when a caller supplies an invalid WMS prepared-answer ID."""
    if req.mode != "wms" or not req.quick_answer_id:
        return None
    answer = (
        "準備済みのWMS回答を検証できないため、在庫情報は表示しません。"
        if req.ui_language == "ja"
        else (
            "Không thể xác minh gợi ý WMS đã chuẩn bị nên tôi không hiển thị "
            "thông tin tồn kho. Vui lòng chọn lại gợi ý hoặc đặt câu hỏi WMS cụ thể."
        )
    )
    return MesQueryOutcome(
        answer=answer,
        results=[],
        routed_model="wms-prepared",
        answer_scope="wms_database",
        wms_metadata={
            "intent": "wms_prepared_validation_suppressed",
            "domain": "SUPPRESSED",
            "status": "SUPPRESSED",
            "reason_codes": ["WMS_PREPARED_VALIDATION_FAILED"],
            "source_as_of_state": "UNAVAILABLE",
            "source_timezone": "unverified",
        },
    )


@app.get("/quick-answers")
async def quick_answers(mode: str = "mkac", language: Literal["vi", "ja"] = "vi"):
    """Trả về danh sách câu hỏi gợi ý theo chế độ."""
    data = _load_quick_answers()
    items = data.get(mode, [])
    suggestions = []
    for item in items:
        if item.get("hidden"):
            continue
        execution = str(item.get("execution") or "")
        is_live = bool(item.get("live"))
        question = item.get("question", "")
        answer = item.get("answer", "")
        if language == "ja":
            question = item.get("question_ja", "")
            answer = item.get("answer_ja", "")
        if not question:
            continue
        if mode == "wms":
            if execution == "server_prepared":
                item_id = str(item.get("id") or "")
                if not item_id:
                    continue
                suggestions.append(
                    {
                        "id": item_id,
                        "question": question,
                        "keywords": item.get("keywords", []),
                        "execution": execution,
                        "live": False,
                    }
                )
                continue
            if execution != "query" and not is_live:
                continue
            suggestions.append(
                {
                    "id": str(item.get("id") or ""),
                    "question": question,
                    "keywords": item.get("keywords", []),
                    "execution": "query",
                    "live": True,
                }
            )
            continue
        # Câu hỏi "live" chỉ cần question; đáp án lấy từ pipeline thật khi bấm.
        # Câu tĩnh phải có sẵn cả câu hỏi lẫn đáp án đóng hộp.
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


@app.post(
    "/query",
    response_model=QueryResponse,
    response_model_exclude_none=True,
)
async def query_documents(req: QueryRequest, request: Request):
    """
    Hỏi đáp dựa trên tài liệu (non-streaming).
    """
    request_started_at = time.monotonic()
    await enforce_rate_limit(request, "query", QUERY_RATE_LIMIT, 60)
    normalize_session_id(req.session_id)
    req = normalize_research_request(req)
    current_employee = await asyncio.to_thread(authorize_query, req)
    if req.mode == "wms" and (
        req.quick_answer_id or resolve_wms_prepared_response(req) is not None
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "WMS_STREAM_REQUIRED: server-prepared WMS answers require "
                "/query/stream."
            ),
        )
    research_cached_response = research_cached_query_response(req)
    if research_cached_response is not None:
        logger.info(
            "Research cached answer hit topic=%s language=%s",
            req.research_topic,
            req.ui_language,
        )
        await record_query_metric(
            mode=req.mode,
            ui_language=req.ui_language,
            answer_scope=research_cached_response.answer_scope,
            cache_hit=True,
            latency_ms=(time.monotonic() - request_started_at) * 1000,
        )
        await wait_for_min_query_latency(request_started_at)
        return research_cached_response.model_copy(
            update={"session_id": req.session_id}
        )

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

        prepared_response = prepared_query_response(req)
        if prepared_response is not None:
            logger.info(
                "Prepared REST query answer hit mode=%s language=%s",
                req.mode,
                req.ui_language,
            )
            await set_cached_query_response(cache_key, prepared_response)
            await wait_for_min_query_latency(request_started_at)
            return prepared_response

        localized_req = await localize_query_request(req)
        current_user_context = await asyncio.to_thread(
            employee_context_for_query,
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

        calendar_response = await handle_calendar_action_query(
            localized_req,
            current_employee,
        )
        if calendar_response is not None:
            translated_calendar_answer = await translate_answer_for_ui(
                calendar_response.answer,
                req,
                answer_scope=calendar_response.answer_scope,
            )
            await wait_for_min_query_latency(request_started_at)
            return calendar_response.model_copy(
                update={"answer": translated_calendar_answer}
            )

        report_response = await handle_report_query(localized_req)
        if report_response is not None:
            # Report agents own localization and receive the requested UI language.
            # Translating their summary again adds latency and can alter numbers or
            # technical identifiers in an already-localized deterministic artifact.
            await wait_for_min_query_latency(request_started_at)
            return report_response

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

        query_outcome = await route_query_outcome(
            localized_req,
            current_user_context=current_user_context,
        )
        if req.mode == "wms":
            query_outcome, wms_metadata = validated_wms_metadata_or_none(
                req,
                query_outcome,
            )
        else:
            wms_metadata = safe_wms_metadata_model(query_outcome.wms_metadata)
        answer = query_outcome.answer
        results = query_outcome.results
        routed_model = query_outcome.routed_model
        answer_scope = query_outcome.answer_scope
        answer = await translate_answer_for_ui(
            answer,
            req,
            answer_scope=answer_scope,
        )
        sources = await translate_sources_for_ui(
            rag_pipeline.format_sources(
                results,
                research_scope=req.research_scope if req.mode == "research" else None,
            ),
            req,
        )
        response = QueryResponse(
            answer=answer,
            sources=sources,
            session_id=req.session_id,
            model=routed_model,
            mode=req.mode,
            answer_scope=answer_scope,
            wms_metadata=(
                WmsAnswerMetadata.model_validate(wms_metadata)
                if isinstance(wms_metadata, dict)
                else wms_metadata
            ),
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
    except HTTPException:
        raise
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
    req = normalize_research_request(req)
    current_employee = await asyncio.to_thread(authorize_query, req)
    research_cached_response = research_cached_query_response(req)
    if research_cached_response is not None:
        logger.info(
            "Streaming research cached answer hit topic=%s language=%s",
            req.research_topic,
            req.ui_language,
        )

        async def research_cached_event_generator():
            response = research_cached_response.model_copy(
                update={"session_id": req.session_id}
            )
            cached_metadata = research_cached_answer_metadata(req)
            yield sse_status(req, "received")
            yield sse_status(req, "research_cache")
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
            yield sse_event(
                {
                    "type": "citations",
                    "citations": cached_metadata.get("citations", []),
                }
            )
            yield sse_event(
                {
                    "type": "meta",
                    "model": response.model,
                    "mode": response.mode,
                    "answer_scope": response.answer_scope,
                    "research_scope": req.research_scope,
                    "cache": "research_static",
                    "cache_id": cached_metadata.get("id", ""),
                }
            )
            yield sse_event({"type": "token", "content": response.answer})
            yield sse_event({"type": "done"})

        return StreamingResponse(
            research_cached_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    cache_key = build_query_cache_key(req)
    cached_response = await get_cached_query_response(cache_key)
    if cached_response is not None:
        logger.info("Streaming query response cache hit mode=%s language=%s", req.mode, req.ui_language)

        async def cached_event_generator():
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
            yield sse_event({"type": "meta", "model": response.model, "mode": response.mode, "answer_scope": response.answer_scope, "research_scope": req.research_scope})
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

        if (
            req.mode == "wms"
            and not report_capability_for_mode(req.question, req.mode).is_report
        ):
            return StreamingResponse(
                wms_verification_event_generator(
                    req,
                    request_started_at=request_started_at,
                ),
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
        try:
            yield sse_status(req, "received")
            yield sse_status(req, "routing")
            if (
                localized_req.mode != "research"
                and (
                    is_email_confirm_request(localized_req.question)
                    or is_email_cancel_request(localized_req.question)
                    or try_parse_email_send_command(localized_req.question) is not None
                )
            ):
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

            calendar_response = await handle_calendar_action_query(
                localized_req,
                current_employee,
            )
            if calendar_response is not None:
                yield sse_status(req, "calendar")
                translated_calendar_answer = await translate_answer_for_ui(
                    calendar_response.answer,
                    req,
                )
                await wait_for_min_query_latency(request_started_at)
                yield sse_status(req, "finalizing")
                yield sse_event({"type": "sources", "sources": []})
                yield sse_event(
                    {
                        "type": "meta",
                        "model": calendar_response.model,
                        "mode": calendar_response.mode,
                        "answer_scope": calendar_response.answer_scope,
                    }
                )
                yield sse_event(
                    {"type": "token", "content": translated_calendar_answer}
                )
                yield sse_event({"type": "done"})
                await record_query_metric(
                    mode=req.mode,
                    ui_language=req.ui_language,
                    answer_scope=calendar_response.answer_scope,
                    cache_hit=False,
                    latency_ms=(time.monotonic() - request_started_at) * 1000,
                )
                return

            report_capability_result = report_capability_for_mode(
                localized_req.question,
                localized_req.mode,
            )
            if report_capability_result.is_report:
                if not report_capability_result.supported:
                    refusal = report_refusal_response(
                        localized_req,
                        report_capability_result,
                    )
                    yield sse_event({"type": "sources", "sources": []})
                    yield sse_event(
                        {
                            "type": "meta",
                            "model": refusal.model,
                            "mode": refusal.mode,
                            "answer_scope": refusal.answer_scope,
                        }
                    )
                    yield sse_event({"type": "token", "content": refusal.answer})
                    yield sse_event({"type": "done"})
                    return
                if report_capability_result.shape == "hr_executive":
                    if localized_req.mode != "mkac":
                        yield sse_event({"type": "done"})
                        return
                    if hr_report_agent is None or not hr_report_agent.available:
                        raise RuntimeError("HR Report Agent chưa sẵn sàng.")
                    yield sse_status(req, "report")
                    yield sse_event(
                        {
                            "type": "agent_plan",
                            "title": "Báo cáo Tổng quan Nhân sự Cấp Điều hành",
                            "period_label": "Danh bạ nhân sự hiện tại",
                            "steps": [
                                {"id": "hr_aggregate", "title": "Tổng hợp headcount theo phòng ban"},
                                {"id": "hr_artifact", "title": "Dựng báo cáo HTML"},
                            ],
                        }
                    )
                    await pace_report_step(0.6)
                    yield sse_event(
                        {
                            "type": "tool_start",
                            "step_id": "hr_aggregate",
                            "tool": "query_hr_directory",
                            "title": "Tổng hợp headcount theo phòng ban",
                        }
                    )
                    report, summary = await hr_report_agent.build_report(
                        localized_req.question,
                        language=req.ui_language,
                    )
                    await pace_report_step()
                    yield sse_event(
                        {
                            "type": "tool_result",
                            "step_id": "hr_aggregate",
                            "status": "done",
                            "summary": "Đã tổng hợp headcount theo phòng ban",
                        }
                    )
                    await pace_report_step(0.4)
                    yield sse_event(
                        {
                            "type": "tool_start",
                            "step_id": "hr_artifact",
                            "tool": "render_hr_report",
                            "title": "Dựng báo cáo HTML",
                        }
                    )
                    await store_report_artifact(report, req)
                    await pace_report_step()
                    yield sse_event(
                        {
                            "type": "tool_result",
                            "step_id": "hr_artifact",
                            "status": "done",
                            "summary": "Báo cáo HTML đã sẵn sàng",
                        }
                    )
                    await wait_for_min_query_latency(request_started_at)
                    await pace_report_step(0.7)
                    yield sse_event(
                        {
                            "type": "artifact",
                            "artifact_type": "hr_executive_report",
                            "artifact": report_artifact_payload(report),
                        }
                    )
                    yield sse_event({"type": "sources", "sources": []})
                    yield sse_event(
                        {
                            "type": "meta",
                            "model": "report-agent",
                            "mode": req.mode,
                            "answer_scope": "hr_executive_report",
                        }
                    )
                    yield sse_event({"type": "token", "content": summary})
                    yield sse_event({"type": "agent_done"})
                    yield sse_event({"type": "done"})
                    await record_query_metric(
                        mode=req.mode,
                        ui_language=req.ui_language,
                        answer_scope="hr_executive_report",
                        cache_hit=False,
                        latency_ms=(time.monotonic() - request_started_at) * 1000,
                    )
                    return
                if report_capability_result.shape == "wms_executive":
                    if mes_wms_report_agent is None or not mes_wms_report_agent.available:
                        raise RuntimeError("WMS Report Agent chưa sẵn sàng.")
                    yield sse_status(req, "report")
                    yield sse_event(
                        {
                            "type": "agent_plan",
                            "title": (
                                "WMS工程倉庫 在庫エグゼクティブレポート"
                                if req.ui_language == "ja"
                                else "Báo cáo Tồn kho WMS Cấp Điều hành"
                            ),
                            "period_label": "Current balance snapshot",
                            "steps": [
                                {
                                    "id": "wms_current_balance",
                                    "title": (
                                        "WMS current balanceを検証"
                                        if req.ui_language == "ja"
                                        else "Kiểm tra current balance WMS"
                                    ),
                                },
                                {
                                    "id": "wms_artifact",
                                    "title": (
                                        "HTMLレポートを作成"
                                        if req.ui_language == "ja"
                                        else "Dựng báo cáo HTML"
                                    ),
                                },
                            ],
                        }
                    )
                    await pace_report_step(0.6)
                    yield sse_event(
                        {
                            "type": "tool_start",
                            "step_id": "wms_current_balance",
                            "tool": "query_wms",
                            "title": (
                                "WMS current balanceを検証"
                                if req.ui_language == "ja"
                                else "Kiểm tra current balance WMS"
                            ),
                        }
                    )
                    report, summary = await mes_wms_report_agent.generate_report(
                        localized_req.question,
                        language=req.ui_language,
                    )
                    await pace_report_step()
                    yield sse_event(
                        {
                            "type": "tool_result",
                            "step_id": "wms_current_balance",
                            "status": "done",
                            "summary": (
                                "Contract v4 compatible"
                                if req.ui_language == "ja"
                                else "Snapshot tương thích contract v4"
                            ),
                        }
                    )
                    await pace_report_step(0.4)
                    yield sse_event(
                        {
                            "type": "tool_start",
                            "step_id": "wms_artifact",
                            "tool": "render_wms_report",
                            "title": (
                                "HTMLレポートを作成"
                                if req.ui_language == "ja"
                                else "Dựng báo cáo HTML"
                            ),
                        }
                    )
                    await store_report_artifact(report, req)
                    await pace_report_step()
                    yield sse_event(
                        {
                            "type": "tool_result",
                            "step_id": "wms_artifact",
                            "status": "done",
                            "summary": (
                                "HTMLレポートの準備が完了"
                                if req.ui_language == "ja"
                                else "Báo cáo HTML đã sẵn sàng"
                            ),
                        }
                    )
                    await wait_for_min_query_latency(request_started_at)
                    await pace_report_step(0.7)
                    artifact_payload = report_artifact_payload(report)
                    yield sse_event(
                        {
                            "type": "artifact",
                            "artifact_type": "wms_executive_report",
                            "artifact": artifact_payload,
                        }
                    )
                    yield sse_event({"type": "sources", "sources": []})
                    yield sse_event(
                        {
                            "type": "meta",
                            "model": "report-agent",
                            "mode": req.mode,
                            "answer_scope": "wms_executive_report",
                        }
                    )
                    yield sse_event({"type": "token", "content": summary})
                    yield sse_event({"type": "agent_done"})
                    yield sse_event({"type": "done"})
                    await record_query_metric(
                        mode=req.mode,
                        ui_language=req.ui_language,
                        answer_scope="wms_executive_report",
                        cache_hit=False,
                        latency_ms=(time.monotonic() - request_started_at) * 1000,
                    )
                    return
                if mes_report_agent is None or not mes_report_agent.available:
                    raise RuntimeError(
                        "Report Agent chưa sẵn sàng vì MES snapshot/SQL Agent "
                        "chưa khả dụng."
                    )
                yield sse_status(req, "report")
                async for agent_event in mes_report_agent.run(
                    localized_req.question, language=req.ui_language
                ):
                    event_kind = agent_event["event"]
                    if event_kind == "plan":
                        yield sse_event(
                            {
                                "type": "agent_plan",
                                "title": agent_event["title"],
                                "period_label": agent_event["period_label"],
                                "steps": agent_event["steps"],
                            }
                        )
                    elif event_kind == "step_start":
                        await pace_report_step(0.4)
                        yield sse_event(
                            {
                                "type": "tool_start",
                                "step_id": agent_event["step_id"],
                                "tool": "query_mes",
                                "title": agent_event["title"],
                            }
                        )
                    elif event_kind == "step_result":
                        await pace_report_step()
                        yield sse_event(
                            {
                                "type": "tool_result",
                                "step_id": agent_event["step_id"],
                                "status": agent_event["status"],
                                "summary": agent_event["summary"],
                            }
                        )
                    elif event_kind == "report":
                        report = agent_event["report"]
                        await store_report_artifact(report, req)
                        # Đừng để báo cáo deterministic chớp lên tức thì: chờ
                        # tới ngưỡng latency chung (~2s), nhưng chỉ phần thời gian
                        # còn thiếu nên truy vấn chậm không bị cộng thêm delay.
                        await wait_for_min_query_latency(request_started_at)
                        await pace_report_step(0.7)
                        artifact_payload = report_artifact_payload(report)
                        yield sse_event(
                            {
                                "type": "artifact",
                                "artifact_type": "mes_report",
                                "artifact": artifact_payload,
                            }
                        )
                        translated_summary = await translate_answer_for_ui(
                            agent_event["summary_text"],
                            req,
                        )
                        yield sse_event(
                            {
                                "type": "meta",
                                "model": "report-agent",
                                "mode": req.mode,
                                "answer_scope": "mes_report",
                            }
                        )
                        yield sse_event(
                            {"type": "token", "content": translated_summary}
                        )
                yield sse_event({"type": "agent_done"})
                yield sse_event({"type": "done"})
                await record_query_metric(
                    mode=req.mode,
                    ui_language=req.ui_language,
                    answer_scope="mes_report",
                    cache_hit=False,
                    latency_ms=(time.monotonic() - request_started_at) * 1000,
                )
                return

            if req.ui_language == "ja" and req.mode != "research":
                yield sse_status(req, query_processing_status_key(localized_req))
                query_outcome = await route_query_outcome(
                    localized_req,
                    current_user_context=current_user_context,
                )
                answer = query_outcome.answer
                results = query_outcome.results
                routed_model = query_outcome.routed_model
                answer_scope = query_outcome.answer_scope
                yield sse_status(req, "translation")
                translated_answer = await translate_answer_for_ui(
                    answer,
                    req,
                    answer_scope=answer_scope,
                )
                sources = await translate_sources_for_ui(
                    rag_pipeline.format_sources(results),
                    req,
                )
                await wait_for_min_query_latency(request_started_at)
                yield sse_status(req, "finalizing")
                yield sse_event({"type": "sources", "sources": sources})
                meta_event = {
                    "type": "meta",
                    "model": routed_model,
                    "mode": req.mode,
                    "answer_scope": answer_scope,
                }
                if query_outcome.wms_metadata:
                    meta_event["wms_metadata"] = safe_wms_metadata(
                        query_outcome.wms_metadata
                    )
                yield sse_event(meta_event)
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
                        wms_metadata=(
                            WmsAnswerMetadata(**safe_wms_metadata(query_outcome.wms_metadata))
                            if query_outcome.wms_metadata
                            else None
                        ),
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
            stream_outcome = await route_query_stream_outcome(
                localized_req,
                current_user_context=current_user_context,
            )
            token_stream = stream_outcome.token_stream
            results = stream_outcome.results
            routed_model = stream_outcome.routed_model
            answer_scope = stream_outcome.answer_scope
            
            # Gửi nguồn trích dẫn (sources) trước
            sources = rag_pipeline.format_sources(
                results,
                research_scope=req.research_scope if req.mode == "research" else None,
            )
            await wait_for_min_query_latency(request_started_at)
            yield sse_status(req, "finalizing")
            yield sse_event({"type": "sources", "sources": sources})
            meta_event = {
                "type": "meta",
                "model": routed_model,
                "mode": req.mode,
                "answer_scope": answer_scope,
                "research_scope": req.research_scope,
            }
            if stream_outcome.wms_metadata:
                meta_event["wms_metadata"] = safe_wms_metadata(
                    stream_outcome.wms_metadata
                )
            yield sse_event(meta_event)

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
                    wms_metadata=(
                        WmsAnswerMetadata(**safe_wms_metadata(stream_outcome.wms_metadata))
                        if stream_outcome.wms_metadata
                        else None
                    ),
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
