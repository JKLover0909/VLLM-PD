"""Proxy from meibook to this repo's agent API v1 for the MSM chat mode.

The browser never sees the MSM admin credential: this module logs in once,
caches the JWT, and re-logs in when the agent answers 401.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Mapping

import httpx
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

LOGIN_PATH = "/agent/login"


class MsmAgentUnavailable(RuntimeError):
    """The MSM agent is not configured or refused the admin login."""


@dataclass(frozen=True)
class MsmAgentConfig:
    base_url: str
    username: str
    password: str = field(repr=False)
    timeout_s: float


def load_config(env: Mapping[str, str] | None = None) -> MsmAgentConfig:
    """Read the agent connection settings; fail closed when any is missing."""
    source = os.environ if env is None else env
    missing = [
        name
        for name in ("MSM_AGENT_BASE", "MSM_AGENT_USER", "MSM_AGENT_PASS")
        if not source.get(name)
    ]
    if missing:
        raise MsmAgentUnavailable(
            "MSM agent is not configured: missing " + ", ".join(missing)
        )
    try:
        timeout_s = float(source.get("MSM_AGENT_TIMEOUT_S") or 90)
    except ValueError:
        timeout_s = 90.0
    return MsmAgentConfig(
        base_url=source["MSM_AGENT_BASE"].rstrip("/"),
        username=source["MSM_AGENT_USER"],
        password=source["MSM_AGENT_PASS"],
        timeout_s=timeout_s,
    )


class MsmTokenCache:
    """One admin JWT shared by every request of this process."""

    def __init__(
        self,
        config: MsmAgentConfig,
        client: httpx.AsyncClient,
        login_path: str = LOGIN_PATH,
    ) -> None:
        self._config = config
        self._client = client
        self._login_path = login_path
        self._token = ""
        self._lock = asyncio.Lock()

    @property
    def config(self) -> MsmAgentConfig:
        return self._config

    def invalidate(self) -> None:
        self._token = ""

    async def token(self) -> str:
        if self._token:
            return self._token
        async with self._lock:
            if self._token:
                return self._token
            token = await self._login()
            self._token = token
            return token

    async def _login(self) -> str:
        try:
            response = await self._client.post(
                self._config.base_url + self._login_path,
                json={
                    "username": self._config.username,
                    "password": self._config.password,
                },
                timeout=self._config.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise MsmAgentUnavailable("MSM agent login failed") from exc
        if response.status_code != 200:
            raise MsmAgentUnavailable(
                f"MSM agent login failed: HTTP {response.status_code}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise MsmAgentUnavailable("MSM agent login returned invalid JSON") from exc
        token = (body or {}).get("token") or ""
        if not token:
            raise MsmAgentUnavailable("MSM agent login returned no token")
        return token


def to_meibook_event(event: dict) -> dict | None:
    """Translate one agent SSE event into the vocabulary meibook renders."""
    kind = event.get("type")
    if kind == "answer":
        return {"type": "token", "content": event["text"]}
    if kind == "done":
        return {"type": "done", "result": event.get("result", {})}
    if kind == "error":
        return {
            "type": "error",
            "code": event.get("code", "INTERNAL"),
            "message": event.get("message", ""),
        }
    if event.get("label"):
        return {"type": "meta", "label": event["label"]}
    return None


class MsmSessions:
    """One agent thread per meibook session, oldest binding evicted first."""

    def __init__(self, max_sessions: int = 500) -> None:
        self._threads: dict[str, str] = {}
        self._max_sessions = max_sessions

    def thread(self, session_id: str) -> str | None:
        return self._threads.get(session_id)

    def bind(self, session_id: str, thread_id: str) -> None:
        self._threads[session_id] = thread_id
        while len(self._threads) > self._max_sessions:
            del self._threads[next(iter(self._threads))]


SESSIONS = MsmSessions()
_config: MsmAgentConfig | None = None
_http: httpx.AsyncClient | None = None
_cache: MsmTokenCache | None = None
_sessions_lock = asyncio.Lock()


def _state() -> tuple[MsmAgentConfig, httpx.AsyncClient, MsmTokenCache]:
    """Resolve config and clients at request time, creating them on first use."""
    global _config, _http, _cache
    if _config is None:
        _config = load_config()
    if _http is None:
        _http = httpx.AsyncClient(timeout=_config.timeout_s)
    if _cache is None or _cache.config is not _config:
        _cache = MsmTokenCache(_config, _http)
    return _config, _http, _cache


class MsmQueryIn(BaseModel):
    session_id: str
    question: str
    employee_id: str


class MsmActionIn(BaseModel):
    session_id: str
    employee_id: str


async def _send(
    cfg: MsmAgentConfig,
    client: httpx.AsyncClient,
    cache: MsmTokenCache,
    path: str,
    payload: dict,
) -> httpx.Response:
    return await client.post(
        cfg.base_url + path,
        json=payload,
        headers={"Authorization": f"Bearer {await cache.token()}"},
        timeout=cfg.timeout_s,
    )


async def _post(
    cfg: MsmAgentConfig,
    client: httpx.AsyncClient,
    cache: MsmTokenCache,
    path: str,
    payload: dict,
) -> httpx.Response:
    """POST once; on 401 re-login and retry exactly once."""
    response = await _send(cfg, client, cache, path, payload)
    if response.status_code == 401:
        cache.invalidate()
        response = await _send(cfg, client, cache, path, payload)
    return response


async def _open_stream(
    cfg: MsmAgentConfig,
    client: httpx.AsyncClient,
    cache: MsmTokenCache,
    path: str,
    payload: dict,
) -> httpx.Response:
    """Open the SSE body; on 401 re-login and retry exactly once."""
    response = None
    for _ in range(2):
        request = client.build_request(
            "POST",
            cfg.base_url + path,
            json=payload,
            headers={"Authorization": f"Bearer {await cache.token()}"},
        )
        response = await client.send(request, stream=True)
        if response.status_code != 401:
            return response
        await response.aclose()
        cache.invalidate()
    return response


async def _thread_for(
    cfg: MsmAgentConfig,
    client: httpx.AsyncClient,
    cache: MsmTokenCache,
    session_id: str,
) -> str:
    thread_id = SESSIONS.thread(session_id)
    if thread_id:
        return thread_id
    async with _sessions_lock:
        thread_id = SESSIONS.thread(session_id)
        if thread_id:
            return thread_id
        response = await _post(cfg, client, cache, "/agent/v1/threads", {"title": "MSM"})
        if response.status_code != 200:
            raise MsmAgentUnavailable(
                f"MSM agent refused to create a thread: HTTP {response.status_code}"
            )
        try:
            thread_id = (response.json() or {}).get("thread_id") or ""
        except ValueError as exc:
            raise MsmAgentUnavailable("MSM agent returned invalid thread JSON") from exc
        if not thread_id:
            raise MsmAgentUnavailable("MSM agent returned no thread id")
        SESSIONS.bind(session_id, thread_id)
        return thread_id


def _sse(event: dict) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


def _error_event(code: str, message: str) -> str:
    return _sse({"type": "error", "code": code, "message": message})


async def _stream(
    cfg: MsmAgentConfig,
    client: httpx.AsyncClient,
    cache: MsmTokenCache,
    body: MsmQueryIn,
) -> AsyncIterator[str]:
    try:
        thread_id = await _thread_for(cfg, client, cache, body.session_id)
        response = await _open_stream(
            cfg,
            client,
            cache,
            f"/agent/v1/threads/{thread_id}/messages/stream",
            {"message": body.question},
        )
    except MsmAgentUnavailable:
        yield _error_event("INTERNAL", "Không kết nối được agent MSM.")
        return
    except httpx.HTTPError:
        yield _error_event("INTERNAL", "Không kết nối được agent MSM.")
        return
    try:
        if response.status_code == 401:
            yield _error_event("MSM_AUTH", "Phiên đăng nhập MSM đã hết hạn.")
            return
        if response.status_code != 200:
            code = "LLM_BUSY" if response.status_code in (409, 503) else "INTERNAL"
            yield _error_event(code, f"Agent MSM trả lỗi HTTP {response.status_code}.")
            return
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[len("data:"):].strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            mapped = to_meibook_event(event)
            if mapped is not None:
                yield _sse(mapped)
    except httpx.HTTPError:
        yield _error_event("INTERNAL", "Agent MSM ngắt kết nối giữa chừng.")
    finally:
        await response.aclose()


router = APIRouter(prefix="/msm", tags=["msm"])


@router.post("/query/stream")
async def query_stream(body: MsmQueryIn) -> StreamingResponse:
    try:
        cfg, client, cache = _state()
    except MsmAgentUnavailable:
        raise HTTPException(503, "MSM_AGENT_NOT_CONFIGURED")
    return StreamingResponse(
        _stream(cfg, client, cache, body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _forward_action(action_id: str, verb: str, body: MsmActionIn) -> Response:
    try:
        cfg, client, cache = _state()
    except MsmAgentUnavailable:
        raise HTTPException(503, "MSM_AGENT_NOT_CONFIGURED")
    thread_id = SESSIONS.thread(body.session_id)
    if not thread_id:
        raise HTTPException(404, "SESSION_NOT_FOUND")
    try:
        response = await _post(
            cfg,
            client,
            cache,
            f"/agent/v1/actions/{action_id}/{verb}",
            {"thread_id": thread_id},
        )
    except MsmAgentUnavailable:
        raise HTTPException(503, "MSM_AGENT_UNAVAILABLE")
    except httpx.HTTPError:
        raise HTTPException(502, "MSM_AGENT_UNAVAILABLE")
    if response.status_code == 401:
        raise HTTPException(503, "MSM_AGENT_UNAVAILABLE")
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type="application/json",
    )


@router.post("/actions/{action_id}/confirm")
async def confirm_action(action_id: str, body: MsmActionIn) -> Response:
    return await _forward_action(action_id, "confirm", body)


@router.post("/actions/{action_id}/cancel")
async def cancel_action(action_id: str, body: MsmActionIn) -> Response:
    return await _forward_action(action_id, "cancel", body)
