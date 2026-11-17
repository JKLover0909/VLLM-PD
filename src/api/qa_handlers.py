"""REST/SSE handlers for ``mode="qa"`` (Agent QMS 5M1E).

QA is dispatched early from ``main.py`` — right after rate limit, session
validation, employee gate and safety guard — so it never touches response
caches, translation, email/calendar/report interception, RAG or MES. Both
transports share ``QaAgentService.ask()``; this module only shapes output.

``record_metric`` is injected (``main.record_query_metric``) to avoid a
circular import of ``main``.

Follow-up questions (agent API 1.1): the agent is stateless, so the
``context``/``history`` it returns are kept here per (session, employee) and
sent with the next question. They never reach the browser. A ``done`` job
replaces the stored state, a failed/lost job keeps it, a rejected context
(``QA_CONTEXT_RESET``), a guardrail answer or ``qa_reset_context`` clears it.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import QaAnswerMetadata, QueryRequest, QueryResponse
from src.api.query_guard import guard_query_response
from src.api.sse import sse_event, sse_status
from src.integrations.qa_agent_service import (
    QA_ANSWER_SCOPE,
    QA_MODEL_LABEL,
    QaAgentError,
    QaAgentResult,
    QaAgentService,
    qa_error_detail,
    validate_qa_question,
)

logger = logging.getLogger(__name__)

RecordMetric = Callable[..., Awaitable[None]]

# SSE comment frames keep proxies from closing an idle stream while the agent
# works; the frontend parser ignores lines that do not start with ``data: ``.
QA_HEARTBEAT_SECONDS = 10.0
_PROGRESS_STATUS_KEYS = {"queued": "qa_queued", "running": "qa_running"}
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

QA_CONTEXT_TTL_SECONDS = 60 * 60
QA_CONTEXT_MAX_THREADS = 500

ThreadKey = tuple[str, str]


@dataclass
class _QaThread:
    context: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    touched_at: float = 0.0


class QaContextStore:
    """In-memory follow-up state, LRU- and TTL-bounded.

    Process-local by design: the app runs one uvicorn worker, and losing the
    state on restart only turns the next question into a fresh thread. All
    methods are synchronous, so no await can interleave a load and a save.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = QA_CONTEXT_TTL_SECONDS,
        max_threads: int = QA_CONTEXT_MAX_THREADS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_threads = max(1, max_threads)
        self._clock = clock
        self._threads: OrderedDict[ThreadKey, _QaThread] = OrderedDict()

    def __len__(self) -> int:
        return len(self._threads)

    @staticmethod
    def key(req: QueryRequest) -> ThreadKey:
        # ``authorize_query`` has already verified the employee for QA, so a
        # thread cannot be continued from another account's session.
        try:
            session = str(uuid.UUID(req.session_id))
        except ValueError:
            session = req.session_id
        return session, (req.employee_id or "").strip()

    def _expire(self) -> None:
        cutoff = self._clock() - self._ttl
        for key in [k for k, t in self._threads.items() if t.touched_at <= cutoff]:
            del self._threads[key]

    def load(self, key: ThreadKey) -> tuple[dict[str, Any], list[dict[str, str]]]:
        self._expire()
        thread = self._threads.get(key)
        if thread is None:
            return {}, []
        self._threads.move_to_end(key)
        return dict(thread.context), list(thread.history)

    def save(self, key: ThreadKey, result: QaAgentResult) -> None:
        self._expire()
        self._threads[key] = _QaThread(
            context=result.context or {},
            history=result.history or [],
            touched_at=self._clock(),
        )
        self._threads.move_to_end(key)
        while len(self._threads) > self._max_threads:
            self._threads.popitem(last=False)

    def reset(self, key: ThreadKey) -> None:
        self._threads.pop(key, None)

    def clear(self) -> None:
        self._threads.clear()


qa_context_store = QaContextStore()


def _start_thread(
    req: QueryRequest, store: QaContextStore
) -> tuple[ThreadKey, dict[str, Any], list[dict[str, str]]]:
    key = store.key(req)
    if req.qa_reset_context:
        store.reset(key)
    context, history = store.load(key)
    return key, context, history


def _finish_thread(
    store: QaContextStore,
    key: ThreadKey,
    result: QaAgentResult,
    response: QueryResponse,
) -> None:
    if response.answer_scope == QA_ANSWER_SCOPE:
        store.save(key, result)
    else:
        # A guardrail replaced the answer: do not build on it.
        store.reset(key)


def _thread_error(store: QaContextStore, key: ThreadKey, error: QaAgentError) -> None:
    # Other failures keep the last good state so the user can simply retry.
    if error.code == "QA_CONTEXT_RESET":
        store.reset(key)


def _http_error(error: QaAgentError, ui_language: str) -> HTTPException:
    headers = (
        {"Retry-After": str(error.retry_after)}
        if error.retry_after is not None
        else None
    )
    return HTTPException(
        status_code=error.http_status,
        detail=qa_error_detail(error, ui_language),
        headers=headers,
    )


def qa_precheck(
    req: QueryRequest, service: Optional[QaAgentService]
) -> str:
    """Fail before any outbound call or stream; return the trimmed question."""
    if service is None or not service.enabled:
        raise _http_error(QaAgentError("QA_DISABLED"), req.ui_language)
    if not service.configured:
        raise _http_error(QaAgentError("QA_MISCONFIGURED"), req.ui_language)
    try:
        return validate_qa_question(req.question)
    except QaAgentError as exc:
        raise _http_error(exc, req.ui_language) from None


def _metadata(result: QaAgentResult) -> QaAnswerMetadata:
    return QaAnswerMetadata.model_validate(result.metadata())


def qa_query_response(
    req: QueryRequest, result: QaAgentResult
) -> QueryResponse:
    return guard_query_response(
        QueryResponse(
            answer=result.answer,
            sources=[],
            session_id=req.session_id,
            model=QA_MODEL_LABEL,
            mode="qa",
            answer_scope=QA_ANSWER_SCOPE,
            qa_metadata=_metadata(result),
        ),
        req.ui_language,
    )


async def _record(
    record_metric: RecordMetric,
    req: QueryRequest,
    started_at: float,
    *,
    error: bool,
) -> None:
    try:
        await record_metric(
            mode="qa",
            ui_language=req.ui_language,
            # Failures use the shared "error" scope, as in the generic routes.
            answer_scope="error" if error else QA_ANSWER_SCOPE,
            cache_hit=False,
            latency_ms=(time.monotonic() - started_at) * 1000,
            error=error,
        )
    except Exception:  # pragma: no cover - metrics must not break answers
        logger.warning("QA metric recording failed.")


async def handle_qa_query(
    req: QueryRequest,
    service: Optional[QaAgentService],
    *,
    record_metric: RecordMetric,
    started_at: float,
    store: Optional[QaContextStore] = None,
) -> QueryResponse:
    """Non-streaming ``/query`` for QA."""
    question = qa_precheck(req, service)
    store = qa_context_store if store is None else store
    thread_key, context, history = _start_thread(req, store)
    try:
        result = await service.ask(question, context=context, history=history)
    except QaAgentError as exc:
        _thread_error(store, thread_key, exc)
        await _record(record_metric, req, started_at, error=True)
        raise _http_error(exc, req.ui_language) from None
    await _record(record_metric, req, started_at, error=False)
    response = qa_query_response(req, result)
    _finish_thread(store, thread_key, result, response)
    return response


async def _qa_event_generator(
    req: QueryRequest,
    service: QaAgentService,
    question: str,
    *,
    record_metric: RecordMetric,
    started_at: float,
    store: QaContextStore,
) -> AsyncIterator[str]:
    progress: asyncio.Queue[str] = asyncio.Queue()

    async def on_progress(status: str) -> None:
        progress.put_nowait(status)

    yield sse_status(req, "received")
    yield sse_status(req, "qa")
    thread_key, context, history = _start_thread(req, store)
    task = asyncio.create_task(
        service.ask(
            question, context=context, history=history, on_progress=on_progress
        )
    )
    last_key = ""
    last_write = time.monotonic()
    try:
        while not task.done():
            getter = asyncio.ensure_future(progress.get())
            done, _ = await asyncio.wait(
                {task, getter},
                timeout=QA_HEARTBEAT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if getter in done:
                key = _PROGRESS_STATUS_KEYS.get(getter.result(), "")
                if key and key != last_key:
                    last_key = key
                    last_write = time.monotonic()
                    yield sse_status(req, key)
            else:
                getter.cancel()
            # Progress callbacks arrive every poll, so the heartbeat is driven
            # by time since the last write, not by wait() timing out.
            if time.monotonic() - last_write >= QA_HEARTBEAT_SECONDS:
                last_write = time.monotonic()
                yield ": keepalive\n\n"

        try:
            result = task.result()
        except QaAgentError as exc:
            _thread_error(store, thread_key, exc)
            await _record(record_metric, req, started_at, error=True)
            detail = qa_error_detail(exc, req.ui_language)
            yield sse_event({"type": "error", **detail})
            return

        await _record(record_metric, req, started_at, error=False)
        response = qa_query_response(req, result)
        # Saved before the first frame: the job is done upstream even if the
        # client disconnects while the answer is being written.
        _finish_thread(store, thread_key, result, response)
        yield sse_event({"type": "sources", "sources": []})
        yield sse_event(
            {
                "type": "meta",
                "model": response.model,
                "mode": response.mode,
                "answer_scope": response.answer_scope,
                **({"qa_metadata": response.qa_metadata.model_dump()}
                   if response.qa_metadata is not None else {}),
            }
        )
        # ``replace`` sets the full answer at once; a long ``token`` would be
        # drip-split by the UI into fake typing.
        yield sse_event({"type": "replace", "content": response.answer})
        yield sse_event({"type": "done"})
    finally:
        if not task.done():
            # Client disconnected: stop waiting. The upstream job cannot be
            # cancelled and may keep running remotely.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def handle_qa_stream(
    req: QueryRequest,
    service: Optional[QaAgentService],
    *,
    record_metric: RecordMetric,
    started_at: float,
    store: Optional[QaContextStore] = None,
) -> StreamingResponse:
    """Streaming ``/query/stream`` for QA. Precheck errors stay HTTP errors."""
    question = qa_precheck(req, service)
    return StreamingResponse(
        _qa_event_generator(
            req,
            service,
            question,
            record_metric=record_metric,
            started_at=started_at,
            store=qa_context_store if store is None else store,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def guard_stream_response(
    req: QueryRequest, guard_response: QueryResponse
) -> StreamingResponse:
    """Stream a safety-guard refusal shaped like a QA answer."""

    async def generator() -> AsyncIterator[str]:
        yield sse_status(req, "received")
        yield sse_event({"type": "sources", "sources": []})
        payload: dict[str, Any] = {
            "type": "meta",
            "model": guard_response.model,
            "mode": guard_response.mode,
            "answer_scope": guard_response.answer_scope,
        }
        yield sse_event(payload)
        yield sse_event({"type": "replace", "content": guard_response.answer})
        yield sse_event({"type": "done"})

    return StreamingResponse(
        generator(), media_type="text/event-stream", headers=_SSE_HEADERS
    )
