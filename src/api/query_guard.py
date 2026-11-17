"""Final disclosure boundary shared by REST and every query SSE branch."""

from __future__ import annotations

import asyncio
import codecs
import json
import re
from functools import wraps

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from src.api.prompt_guard import has_output_disclosure, payload_has_disclosure, refusal_text
from src.api.schemas import QueryRequest, QueryResponse
from src.api.sse import sse_event

MAX_BUFFER_CHARS = 1_000_000
MAX_ANSWER_CHARS = 64_000
HEARTBEAT_SECONDS = 10.0
_HEARTBEAT = object()
_PROGRESS_TYPES = {"status", "agent_plan", "tool_start", "tool_result"}


def clean_answer(text: str) -> str:
    # Never expose a reasoning block, including an unterminated final block.
    text = re.sub(r"<think\b[^>]*>[\s\S]*?(?:</think\s*>|$)", "", text, flags=re.I)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1]
    return text.strip()


def guardrail_response(req: QueryRequest) -> QueryResponse:
    return QueryResponse(
        answer=refusal_text(req.ui_language), sources=[], session_id=req.session_id,
        model="auto-model", mode=req.mode, answer_scope="guardrail",
    )


def guard_query_response(response: QueryResponse, language: str = "vi") -> QueryResponse:
    if (response.answer in {refusal_text("vi"), refusal_text("ja")}
            or payload_has_disclosure(response.model_dump(exclude_none=True))):
        return QueryResponse(
            answer=refusal_text(language), sources=[], session_id=response.session_id,
            model="auto-model", mode=response.mode, answer_scope="guardrail",
        )
    answer = clean_answer(response.answer)
    if answer != response.answer:
        return response.model_copy(update={"answer": answer})
    return response


def _refusal_events(req: QueryRequest):
    response = guardrail_response(req)
    return [
        {"type": "sources", "sources": []},
        {"type": "meta", "model": response.model, "mode": req.mode, "answer_scope": "guardrail"},
        {"type": "replace" if req.mode == "qa" else "token", "content": response.answer},
        {"type": "done"},
    ]


def _stream_error(req: QueryRequest) -> dict:
    return {
        "type": "error",
        "message": "回答を完了できませんでした。再度お試しください。" if req.ui_language == "ja" else "Không thể hoàn tất câu trả lời. Vui lòng thử lại.",
    }


async def _with_heartbeats(body_iterator):
    """Keep idle streams alive without cancelling a pending model operation."""
    iterator = body_iterator.__aiter__()
    pending = None
    try:
        while True:
            pending = asyncio.ensure_future(iterator.__anext__())
            while True:
                done, _ = await asyncio.wait({pending}, timeout=HEARTBEAT_SECONDS)
                if done:
                    break
                yield _HEARTBEAT
            try:
                yield pending.result()
            except StopAsyncIteration:
                return
    finally:
        if pending is not None:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()


async def guarded_sse(body_iterator, req: QueryRequest):
    """Buffer answer-bearing events until done; never retract leaked wire bytes.

    Progress remains live. Bound both decoded frame storage and final answer.
    Refuse incomplete/malformed streams and close the producer on cancellation.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    pending = ""
    events = []
    answer = ""
    blocked = False
    buffered_chars = 0
    terminal = False
    answer_seen = False
    answer_kind = "token"

    chunks = _with_heartbeats(body_iterator)
    try:
        async for chunk in chunks:
            if terminal:
                continue
            if chunk is _HEARTBEAT:
                yield ": keepalive\n\n"
                continue
            pending += decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
            if len(pending) + buffered_chars > MAX_BUFFER_CHARS:
                for event in _refusal_events(req):
                    yield sse_event(event)
                return
            # Application emitters use LF; accept standard CRLF frames as well.
            pending = pending.replace("\r\n", "\n")
            while "\n\n" in pending:
                frame, pending = pending.split("\n\n", 1)
                if terminal:
                    continue
                lines = [
                    line[5:].lstrip(" ")
                    for line in frame.splitlines()
                    if line.startswith("data:")
                ]
                if not lines:
                    if frame.startswith(":"):
                        yield ": keepalive\n\n"
                    continue
                event = json.loads("\n".join(lines))
                if not isinstance(event, dict):
                    raise ValueError("Invalid event")
                kind = event.get("type")
                if kind in _PROGRESS_TYPES:
                    if payload_has_disclosure(event):
                        blocked = True
                    else:
                        yield sse_event(event)
                    continue
                if kind == "error":
                    # Error contracts from QA are allowlisted upstream. Generic
                    # exceptions must not echo SQL, tool arguments or prompts.
                    safe_qa_error = (
                        req.mode == "qa" and not payload_has_disclosure(event)
                    )
                    yield sse_event(event if safe_qa_error else _stream_error(req))
                    terminal = True
                    return
                if kind == "done":
                    terminal = True
                    blocked = (
                        blocked
                        or has_output_disclosure(answer)
                        or answer in {refusal_text("vi"), refusal_text("ja")}
                    )
                    if blocked:
                        for safe_event in _refusal_events(req):
                            yield sse_event(safe_event)
                    else:
                        emitted_answer = False
                        for saved in events:
                            if saved.get("type") == "_answer":
                                if not emitted_answer:
                                    yield sse_event(
                                        {
                                            "type": answer_kind,
                                            "content": clean_answer(answer),
                                        }
                                    )
                                    emitted_answer = True
                            else:
                                yield sse_event(saved)
                        yield sse_event({"type": "done"})
                    # Let producers finish post-done metrics/cache writes, but
                    # never emit anything after a terminal event.
                    continue
                buffered_chars += len(frame)
                if kind in {"token", "replace"}:
                    text = event.get("content", "")
                    if not isinstance(text, str):
                        raise ValueError("Invalid answer")
                    answer = text if kind == "replace" else answer + text
                    if kind == "replace":
                        answer_kind = "replace"
                    blocked = blocked or has_output_disclosure(answer)
                    if not answer_seen:
                        events.append({"type": "_answer"})
                        answer_seen = True
                    if len(answer) > MAX_ANSWER_CHARS:
                        for safe_event in _refusal_events(req):
                            yield sse_event(safe_event)
                        return
                else:
                    blocked = blocked or payload_has_disclosure(event)
                    events.append(event)
                if buffered_chars > MAX_BUFFER_CHARS:
                    for safe_event in _refusal_events(req):
                        yield sse_event(safe_event)
                    return
        if not terminal:
            yield sse_event(_stream_error(req))
    except (ValueError, TypeError, UnicodeError):
        if not terminal:
            yield sse_event(_stream_error(req))
    except Exception:
        if not terminal:
            yield sse_event(_stream_error(req))
    finally:
        await chunks.aclose()


def guarded_query(handler):
    """Wrap route results, including early returns, without changing schemas."""
    @wraps(handler)
    async def wrapped(req: QueryRequest, request):
        try:
            result = await handler(req, request)
        except HTTPException as exc:
            if payload_has_disclosure([exc.detail, exc.headers]):
                raise HTTPException(
                    status_code=exc.status_code,
                    detail=_stream_error(req)["message"],
                ) from None
            raise
        if isinstance(result, QueryResponse):
            return guard_query_response(result, req.ui_language)
        if isinstance(result, StreamingResponse):
            result.body_iterator = guarded_sse(result.body_iterator, req)
        return result
    return wrapped
