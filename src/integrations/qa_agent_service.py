"""Async adapter for the external Agent QMS 5M1E job API (``mode="qa"``).

Meibook is only the UI/gateway here: the remote agent owns QMS access and the
answer. Contract (see ``Markdowns/QA_AGENT.md``):

* ``POST /v1/jobs`` ``{question, client_ref[, context, history]}`` -> 202
  ``{job_id, status}``. Sending ``context``/``history`` (API 1.1) selects the
  follow-up mode; the agent then returns updated ``context``/``history`` plus
  ``clarification`` and ``effective_question`` in ``result``.
* ``GET /v1/jobs/{job_id}`` -> ``queued|running|done|error``.
* ``GET /health`` is liveness only (no key, no LLM/DB check).

Design constraints:
* Endpoint and key come only from env, never from user input. Redirects are
  disabled, env proxies are ignored and ``poll_url`` from the response is never
  followed, so the API key cannot be forwarded to another host.
* The job is submitted exactly once. A POST that times out or returns 5xx has
  an unknown outcome and is reported as such, never resubmitted; there is no
  upstream idempotency or cancel.
* One monotonic deadline covers submit + queue + running + retries. Each HTTP
  call is bounded by ``min(request_timeout, remaining)``.
* Only the question, a random opaque ``client_ref`` and the agent's own
  ``context``/``history`` from the previous answer leave Meibook: never the
  browser's chat history, employee data or session ID. The caller keeps that
  state server-side; it is bounded here in both directions.
* Responses are size-bounded, validated and allowlisted; ``sqls`` and any
  field outside the contract are dropped.
* Logs carry only error codes, the opaque reference, status and latency —
  never the question, answer, citations, headers or the key.
* Disabled means zero outbound requests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import quote, urlsplit

import httpx

log = logging.getLogger(__name__)

QA_MAX_QUESTION_CHARS = 2000
QA_PROVIDER = "qms_5m1e_agent"
QA_MODEL_LABEL = "qms-agent"
QA_ANSWER_SCOPE = "qa_agent"

MAX_RESPONSE_BYTES = 512 * 1024
MAX_ANSWER_CHARS = 60_000
MAX_CITATIONS = 20
MAX_VIEWS_PER_CITATION = 10
MAX_KEYS_PER_CITATION = 10
MAX_VALUES_PER_KEY = 20
MAX_CHUNKS_PER_CITATION = 20
MAX_IDENTIFIER_CHARS = 120
MAX_VALUE_CHARS = 200
MAX_GET_RETRIES = 2
# API 1.1 request limits: history <= 20 items, content <= 4000 chars. The
# context object is agent-owned and opaque; only its size is bounded.
MAX_HISTORY_ITEMS = 20
MAX_HISTORY_CONTENT_CHARS = 4000
MAX_CONTEXT_KEYS = 50
MAX_CONTEXT_BYTES = 16 * 1024

_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.$#:-]{1,%d}$" % MAX_IDENTIFIER_CHARS)
_JOB_STATUSES = {"queued", "running", "done", "error"}
_CITATION_TYPES = {"sql", "row", "semantic"}

STATE_DISABLED = "DISABLED"
STATE_MISCONFIGURED = "MISCONFIGURED"
STATE_UNKNOWN = "UNKNOWN"
STATE_ONLINE = "ONLINE"
STATE_OFFLINE = "OFFLINE"


# code -> (HTTP status before streaming, VI message, JA message)
QA_ERRORS: dict[str, tuple[int, str, str]] = {
    "QA_DISABLED": (
        503,
        "Chế độ QA chưa được bật trên hệ thống.",
        "QAモードはこのシステムで有効になっていません。",
    ),
    "QA_MISCONFIGURED": (
        503,
        "Kết nối Agent QA chưa được cấu hình đúng. Vui lòng liên hệ quản trị viên.",
        "QAエージェントの接続設定が正しくありません。管理者に連絡してください。",
    ),
    "QA_OFFLINE": (
        503,
        "Không kết nối được Agent QA. Vui lòng thử lại sau.",
        "QAエージェントに接続できません。しばらくしてから再試行してください。",
    ),
    "QA_UPSTREAM_AUTH": (
        503,
        "Agent QA từ chối kết nối của Meibook (lỗi cấu hình). Vui lòng liên hệ quản trị viên.",
        "QAエージェントがMeibookの接続を拒否しました（設定エラー）。管理者に連絡してください。",
    ),
    "QA_BUSY": (
        503,
        "Agent QA đang bận. Vui lòng gửi lại câu hỏi sau ít phút.",
        "QAエージェントが混雑しています。少し時間をおいて再送信してください。",
    ),
    "QA_SUBMIT_UNKNOWN": (
        502,
        "Không xác định được Agent QA đã nhận câu hỏi hay chưa. Vui lòng gửi lại nếu cần.",
        "QAエージェントが質問を受け付けたか確認できません。必要に応じて再送信してください。",
    ),
    "QA_JOB_LOST": (
        502,
        "Agent QA không còn giữ yêu cầu này (có thể đã khởi động lại). Vui lòng gửi lại câu hỏi.",
        "QAエージェントにこのリクエストが残っていません（再起動の可能性）。質問を再送信してください。",
    ),
    "QA_AGENT_ERROR": (
        502,
        "Agent QA không xử lý được câu hỏi này.",
        "QAエージェントはこの質問を処理できませんでした。",
    ),
    "QA_BAD_RESPONSE": (
        502,
        "Agent QA trả về dữ liệu không hợp lệ.",
        "QAエージェントから不正なデータが返されました。",
    ),
    "QA_TIMEOUT": (
        504,
        "Agent QA chưa trả lời trong thời gian cho phép. Meibook đã dừng chờ; yêu cầu phía Agent có thể vẫn đang chạy.",
        "QAエージェントが制限時間内に応答しませんでした。Meibookは待機を終了しましたが、エージェント側の処理は継続している可能性があります。",
    ),
    "QA_INVALID_QUESTION": (
        422,
        f"Câu hỏi QA phải có từ 1 đến {QA_MAX_QUESTION_CHARS} ký tự.",
        f"QAの質問は1〜{QA_MAX_QUESTION_CHARS}文字で入力してください。",
    ),
    "QA_CONTEXT_RESET": (
        422,
        "Ngữ cảnh hỏi nối tiếp không còn hợp lệ nên đã được đặt lại. Vui lòng gửi lại câu hỏi kèm đủ mã máy hoặc mã phiếu và khoảng thời gian.",
        "続けて質問するための会話コンテキストが無効になったためリセットしました。設備コードまたはチェックシートコードと期間を含めて再送信してください。",
    ),
    "QA_GUEST_NOT_ALLOWED": (
        403,
        "Chế độ QA yêu cầu đăng nhập bằng mã nhân viên, không hỗ trợ tài khoản khách.",
        "QAモードは社員コードでのログインが必要です。ゲストは利用できません。",
    ),
}


class QaAgentError(RuntimeError):
    """Typed, user-safe failure. ``message`` never contains upstream bodies."""

    def __init__(self, code: str, *, retry_after: Optional[int] = None) -> None:
        if code not in QA_ERRORS:
            code = "QA_BAD_RESPONSE"
        super().__init__(code)
        self.code = code
        self.retry_after = retry_after

    @property
    def http_status(self) -> int:
        return QA_ERRORS[self.code][0]

    def message(self, ui_language: str = "vi") -> str:
        _, vi, ja = QA_ERRORS[self.code]
        # Like the gateway rate limit (429), state a known wait in the message.
        if self.retry_after is not None:
            if ui_language == "ja":
                return f"{ja}（{self.retry_after}秒後に再試行できます）"
            return f"{vi} (Có thể thử lại sau {self.retry_after} giây.)"
        return ja if ui_language == "ja" else vi


def qa_error_detail(error: QaAgentError, ui_language: str = "vi") -> dict[str, Any]:
    detail: dict[str, Any] = {"code": error.code, "message": error.message(ui_language)}
    if error.retry_after is not None:
        detail["retry_after"] = error.retry_after
    return detail


def validate_qa_question(question: str) -> str:
    """Trim and enforce 1..2000 Unicode code points (Python ``len``)."""
    cleaned = (question or "").strip()
    if not cleaned or len(cleaned) > QA_MAX_QUESTION_CHARS:
        raise QaAgentError("QA_INVALID_QUESTION")
    return cleaned


@dataclass(frozen=True)
class QaAgentResult:
    answer: str
    client_ref: str
    elapsed_ms: Optional[float]
    created_at: str
    started_at: str
    finished_at: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    citations_truncated: bool = False
    clarification: bool = False
    effective_question: str = ""
    # Whether a non-empty context/history from an earlier answer was sent.
    context_applied: bool = False
    # Follow-up state to send with the next question. ``None`` when the agent
    # answered in one-shot mode or returned an invalid value. Never exposed
    # through ``metadata()``.
    context: Optional[dict[str, Any]] = None
    history: Optional[list[dict[str, str]]] = None

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": QA_PROVIDER,
            "answer_language": "vi",
            "context_supported": self.context is not None,
            "context_applied": self.context_applied,
            "clarification": self.clarification,
            "effective_question": self.effective_question,
            "client_ref": self.client_ref,
            "elapsed_ms": self.elapsed_ms,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "citations": self.citations,
            "citations_truncated": self.citations_truncated,
        }


ProgressCallback = Callable[[str], Awaitable[None]]


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _valid_base_url(raw: str) -> Optional[str]:
    try:
        parts = urlsplit(raw)
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return None
    if parts.username or parts.password or parts.query or parts.fragment:
        return None
    return raw.rstrip("/")


def _short_text(value: Any, limit: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > limit:
        return None
    return text


def _identifier(value: Any) -> Optional[str]:
    text = _short_text(value, MAX_IDENTIFIER_CHARS)
    if text is None or not _IDENTIFIER_PATTERN.fullmatch(text):
        return None
    return text


def _scalar(value: Any) -> Optional[int | float | str]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return _short_text(value, MAX_VALUE_CHARS)


def _key_or_id(value: Any) -> Optional[int | str]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return _short_text(value, MAX_VALUE_CHARS)


def _timestamp(value: Any) -> str:
    return _short_text(value, 64) or ""


def sanitize_context(raw: Any) -> Optional[dict[str, Any]]:
    """Bound the agent-owned ``context`` object; ``None`` if absent/invalid.

    Keys are interpreted only by the agent (it ignores unknown ones), so the
    value is kept opaque and returned as a detached JSON copy.
    """
    if not isinstance(raw, dict) or len(raw) > MAX_CONTEXT_KEYS:
        return None
    try:
        encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return None
    if len(encoded.encode("utf-8")) > MAX_CONTEXT_BYTES:
        return None
    return json.loads(encoded)


def sanitize_history(raw: Any) -> Optional[list[dict[str, str]]]:
    """Keep the last 20 ``{role, content}`` turns; ``None`` if malformed.

    Content is cut to the request limit so a long answer the agent echoed
    back cannot make the next submit fail validation.
    """
    if not isinstance(raw, list):
        return None
    turns: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            return None
        content = item.get("content")
        if not isinstance(content, str):
            return None
        content = content.strip()[:MAX_HISTORY_CONTENT_CHARS]
        if content:
            turns.append({"role": item["role"], "content": content})
    return turns[-MAX_HISTORY_ITEMS:]


def sanitize_citations(raw: Any) -> tuple[list[dict[str, Any]], bool]:
    """Allowlist and cap citations. Returns ``(citations, truncated)``.

    Invalid entries are dropped and reported as truncated; they never turn the
    answer into a verified one.
    """
    if raw is None:
        return [], False
    if not isinstance(raw, list):
        return [], True
    truncated = len(raw) > MAX_CITATIONS
    cleaned: list[dict[str, Any]] = []
    for item in raw[:MAX_CITATIONS]:
        if not isinstance(item, dict) or item.get("type") not in _CITATION_TYPES:
            truncated = True
            continue
        citation: dict[str, Any] = {"type": item["type"]}

        views_raw = item.get("views") or []
        views = [v for v in (_identifier(x) for x in views_raw) if v] if isinstance(views_raw, list) else []
        if not isinstance(views_raw, list) or len(views_raw) > MAX_VIEWS_PER_CITATION or len(views) != len(views_raw):
            truncated = True
        citation["views"] = views[:MAX_VIEWS_PER_CITATION]

        keys_raw = item.get("keys") or {}
        keys: dict[str, list[int | float | str]] = {}
        if isinstance(keys_raw, dict):
            if len(keys_raw) > MAX_KEYS_PER_CITATION:
                truncated = True
            for name, values in list(keys_raw.items())[:MAX_KEYS_PER_CITATION]:
                key_name = _identifier(name)
                if key_name is None or not isinstance(values, list):
                    truncated = True
                    continue
                kept = [v for v in (_scalar(x) for x in values[:MAX_VALUES_PER_KEY]) if v is not None]
                if len(values) > MAX_VALUES_PER_KEY or len(kept) != min(len(values), MAX_VALUES_PER_KEY):
                    truncated = True
                keys[key_name] = kept
        else:
            truncated = True
        citation["keys"] = keys

        if item["type"] == "semantic":
            ik_raw = item.get("inspection_keys") or []
            if isinstance(ik_raw, list):
                kept_ik = [v for v in (_key_or_id(x) for x in ik_raw[:MAX_VALUES_PER_KEY]) if v is not None]
                if len(ik_raw) > MAX_VALUES_PER_KEY or len(kept_ik) != min(len(ik_raw), MAX_VALUES_PER_KEY):
                    truncated = True
            else:
                kept_ik = []
                truncated = True
            citation["inspection_keys"] = kept_ik

            chunks_raw = item.get("chunks") or []
            chunks: list[dict[str, Any]] = []
            if isinstance(chunks_raw, list):
                if len(chunks_raw) > MAX_CHUNKS_PER_CITATION:
                    truncated = True
                for chunk in chunks_raw[:MAX_CHUNKS_PER_CITATION]:
                    if not isinstance(chunk, dict):
                        truncated = True
                        continue
                    chunks.append(
                        {
                            "id": _key_or_id(chunk.get("id")),
                            "source_view": _identifier(chunk.get("source_view")) or "",
                            "source_key": _key_or_id(chunk.get("source_key")),
                            "inspection_key": _key_or_id(chunk.get("inspection_key")),
                        }
                    )
            else:
                truncated = True
            citation["chunks"] = chunks
        cleaned.append(citation)
    return cleaned, truncated


class QaAgentService:
    """Gateway to the Agent QMS 5M1E job API with a bounded in-flight slot."""

    def __init__(
        self,
        *,
        enabled: bool,
        base_url: str = "",
        api_key: str = "",
        request_timeout_seconds: float = 5.0,
        total_timeout_seconds: float = 180.0,
        poll_interval_seconds: float = 1.5,
        max_inflight: int = 2,
        health_ttl_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._enabled = enabled
        self._base_url = _valid_base_url(base_url) if base_url else None
        self._api_key = api_key
        self._configured = bool(self._base_url and api_key)
        self._request_timeout = request_timeout_seconds
        self._total_timeout = total_timeout_seconds
        self._poll_interval = poll_interval_seconds
        self._max_inflight = max(1, max_inflight)
        self._inflight = 0
        self._health_ttl = health_ttl_seconds
        self._transport = transport
        self._clock = clock
        self._sleep = sleep
        self._client: httpx.AsyncClient | None = None
        self._last_health_ok: Optional[bool] = None
        self._last_health_at: Optional[float] = None

    @classmethod
    def from_env(cls) -> "QaAgentService":
        """Zero I/O. Always returns a service so ``/health`` can report state."""
        return cls(
            enabled=_env_bool("QA_AGENT_ENABLED", False),
            base_url=_env_str("QA_AGENT_URL"),
            api_key=_env_str("QA_AGENT_API_KEY"),
            request_timeout_seconds=_env_float(
                "QA_AGENT_REQUEST_TIMEOUT_SECONDS", 5.0, minimum=1.0, maximum=30.0
            ),
            total_timeout_seconds=_env_float(
                "QA_AGENT_TIMEOUT_SECONDS", 180.0, minimum=10.0, maximum=600.0
            ),
            poll_interval_seconds=_env_float(
                "QA_AGENT_POLL_INTERVAL_SECONDS", 1.5, minimum=0.5, maximum=10.0
            ),
            max_inflight=_env_int("QA_AGENT_MAX_INFLIGHT", 2, minimum=1, maximum=20),
            health_ttl_seconds=_env_float(
                "QA_AGENT_HEALTH_TTL_SECONDS", 30.0, minimum=5.0, maximum=600.0
            ),
        )

    # ------------------------------------------------------------------ state
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def configured(self) -> bool:
        return self._configured

    @property
    def health_interval_seconds(self) -> float:
        """Probe period; half the TTL so a healthy snapshot never goes stale."""
        return self._health_ttl / 2

    @property
    def state(self) -> str:
        if not self._enabled:
            return STATE_DISABLED
        if not self._configured:
            return STATE_MISCONFIGURED
        if self._last_health_at is None or (self._clock() - self._last_health_at) >= self._health_ttl:
            return STATE_UNKNOWN
        return STATE_ONLINE if self._last_health_ok else STATE_OFFLINE

    def status(self) -> dict[str, Any]:
        """Zero-I/O allowlisted status for ``/health``. Never exposes URL/key."""
        return {
            "enabled": self._enabled,
            "configured": self._enabled and self._configured,
            "state": self.state,
            "answer_language": "vi",
            "supports_history": True,
            "max_question_chars": QA_MAX_QUESTION_CHARS,
        }

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._request_timeout,
                follow_redirects=False,
                trust_env=False,
                transport=self._transport,
            )
        return self._client

    async def refresh_health(self) -> None:
        """Background liveness probe (``GET /health`` only, no key, no job)."""
        if not (self._enabled and self._configured):
            return
        ok = False
        try:
            response = await self._http().get(
                f"{self._base_url}/health", timeout=min(self._request_timeout, 3.0)
            )
            ok = response.status_code == 200
        except httpx.HTTPError as exc:
            log.warning("qa_health_failed error=%s", exc.__class__.__name__)
        self._last_health_ok = ok
        self._last_health_at = self._clock()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------- jobs
    def _remaining(self, deadline: float) -> float:
        return deadline - self._clock()

    def _call_timeout(self, deadline: float) -> float:
        remaining = self._remaining(deadline)
        if remaining <= 0:
            raise QaAgentError("QA_TIMEOUT")
        return max(0.1, min(self._request_timeout, remaining))

    async def _read_json(self, response: httpx.Response) -> Any:
        total = 0
        chunks: list[bytes] = []
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                raise QaAgentError("QA_BAD_RESPONSE")
            chunks.append(chunk)
        try:
            return json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise QaAgentError("QA_BAD_RESPONSE") from None

    @staticmethod
    def _retry_after(response: httpx.Response) -> Optional[int]:
        raw = response.headers.get("Retry-After", "").strip()
        if raw.isdigit():
            return max(1, min(int(raw), 600))
        return None

    async def _submit(
        self,
        question: str,
        client_ref: str,
        deadline: float,
        *,
        context: Optional[dict[str, Any]] = None,
        history: Optional[list[dict[str, str]]] = None,
    ) -> str:
        url = f"{self._base_url}/v1/jobs"
        headers = {"X-API-Key": self._api_key, "Accept": "application/json"}
        body: dict[str, Any] = {"question": question, "client_ref": client_ref}
        if context is not None:
            body["context"] = context
        if history is not None:
            body["history"] = history
        try:
            async with self._http().stream(
                "POST",
                url,
                json=body,
                headers=headers,
                timeout=self._call_timeout(deadline),
            ) as response:
                status = response.status_code
                if status == 202:
                    payload = await self._read_json(response)
                elif status in (401, 403):
                    raise QaAgentError("QA_UPSTREAM_AUTH")
                elif status == 503:
                    raise QaAgentError("QA_BUSY", retry_after=self._retry_after(response))
                elif status == 422:
                    # The question is validated locally, so a 422 on a
                    # follow-up means the carried state was rejected.
                    raise QaAgentError(
                        "QA_CONTEXT_RESET" if (context or history) else "QA_INVALID_QUESTION"
                    )
                elif status >= 500:
                    raise QaAgentError("QA_SUBMIT_UNKNOWN")
                else:
                    raise QaAgentError("QA_BAD_RESPONSE")
        except httpx.ConnectError:
            # The connection was never established, so no job exists.
            raise QaAgentError("QA_OFFLINE") from None
        except httpx.TimeoutException:
            raise QaAgentError("QA_SUBMIT_UNKNOWN") from None
        except httpx.HTTPError:
            raise QaAgentError("QA_SUBMIT_UNKNOWN") from None

        if not isinstance(payload, dict):
            raise QaAgentError("QA_BAD_RESPONSE")
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not _JOB_ID_PATTERN.fullmatch(job_id):
            raise QaAgentError("QA_BAD_RESPONSE")
        if payload.get("status") not in _JOB_STATUSES:
            raise QaAgentError("QA_BAD_RESPONSE")
        return job_id

    async def _poll_once(self, job_id: str, deadline: float) -> dict[str, Any]:
        url = f"{self._base_url}/v1/jobs/{quote(job_id, safe='')}"
        headers = {"X-API-Key": self._api_key, "Accept": "application/json"}
        async with self._http().stream(
            "GET", url, headers=headers, timeout=self._call_timeout(deadline)
        ) as response:
            status = response.status_code
            if status == 200:
                payload = await self._read_json(response)
            elif status in (401, 403):
                raise QaAgentError("QA_UPSTREAM_AUTH")
            elif status == 404:
                raise QaAgentError("QA_JOB_LOST")
            elif status >= 500:
                raise _RetryablePollError()
            else:
                raise QaAgentError("QA_BAD_RESPONSE")
        if not isinstance(payload, dict):
            raise QaAgentError("QA_BAD_RESPONSE")
        if payload.get("job_id") != job_id or payload.get("status") not in _JOB_STATUSES:
            raise QaAgentError("QA_BAD_RESPONSE")
        return payload

    def _build_result(
        self, payload: dict[str, Any], client_ref: str, *, context_applied: bool = False
    ) -> QaAgentResult:
        returned_ref = payload.get("client_ref")
        if returned_ref not in (None, client_ref):
            raise QaAgentError("QA_BAD_RESPONSE")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise QaAgentError("QA_BAD_RESPONSE")
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip() or len(answer) > MAX_ANSWER_CHARS:
            raise QaAgentError("QA_BAD_RESPONSE")
        elapsed = result.get("elapsed_ms")
        elapsed_ms: Optional[float] = None
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed >= 0:
            elapsed_ms = float(elapsed)
        citations, truncated = sanitize_citations(result.get("citations"))
        return QaAgentResult(
            answer=answer.strip(),
            client_ref=client_ref,
            elapsed_ms=elapsed_ms,
            created_at=_timestamp(payload.get("created_at")),
            started_at=_timestamp(payload.get("started_at")),
            finished_at=_timestamp(payload.get("finished_at")),
            citations=citations,
            citations_truncated=truncated,
            clarification=result.get("clarification") is True,
            effective_question=_short_text(
                result.get("effective_question"), MAX_HISTORY_CONTENT_CHARS
            ) or "",
            context_applied=context_applied,
            context=sanitize_context(result.get("context")),
            history=sanitize_history(result.get("history")),
        )

    async def ask(
        self,
        question: str,
        *,
        context: Optional[dict[str, Any]] = None,
        history: Optional[list[dict[str, str]]] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> QaAgentResult:
        """Submit one job and wait for its terminal state.

        Passing ``context``/``history`` (``{}``/``[]`` for the first question)
        selects the API 1.1 follow-up mode; the result then carries the state
        for the next question. ``None`` for both keeps the one-shot request.

        ``on_progress`` receives ``"queued"``/``"running"`` on each status
        change and ``"waiting"`` on every poll, so SSE can send heartbeats.
        """
        if not self._enabled:
            raise QaAgentError("QA_DISABLED")
        if not self._configured:
            raise QaAgentError("QA_MISCONFIGURED")
        cleaned = validate_qa_question(question)
        follow_up = context is not None or history is not None
        sent_context = (sanitize_context(context) or {}) if follow_up else None
        sent_history = (sanitize_history(history) or []) if follow_up else None
        context_applied = bool(sent_context or sent_history)
        if self._inflight >= self._max_inflight:
            # No local queue: the agent itself only runs 2 workers.
            raise QaAgentError("QA_BUSY")

        self._inflight += 1
        started = self._clock()
        deadline = started + self._total_timeout
        client_ref = f"meibook-{secrets.token_hex(8)}"
        outcome = "error"
        try:
            job_id = await self._submit(
                cleaned, client_ref, deadline, context=sent_context, history=sent_history
            )
            last_status = ""
            failures = 0
            while True:
                try:
                    payload = await self._poll_once(job_id, deadline)
                    failures = 0
                except (_RetryablePollError, httpx.TimeoutException, httpx.TransportError):
                    failures += 1
                    if failures > MAX_GET_RETRIES:
                        raise QaAgentError("QA_OFFLINE") from None
                    payload = None

                if payload is not None:
                    status = payload["status"]
                    if status == "done":
                        result = self._build_result(
                            payload, client_ref, context_applied=context_applied
                        )
                        outcome = "done"
                        return result
                    if status == "error":
                        outcome = "agent_error"
                        raise QaAgentError("QA_AGENT_ERROR")
                    if status != last_status and on_progress is not None:
                        await on_progress(status)
                    last_status = status

                if on_progress is not None:
                    await on_progress("waiting")
                remaining = self._remaining(deadline)
                if remaining <= 0:
                    raise QaAgentError("QA_TIMEOUT")
                await self._sleep(min(self._poll_interval, remaining))
        except QaAgentError as exc:
            outcome = exc.code
            raise
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            self._inflight -= 1
            log.info(
                "qa_job_complete ref=%s outcome=%s latency_ms=%d",
                client_ref,
                outcome,
                int((self._clock() - started) * 1000),
            )


class _RetryablePollError(Exception):
    """Internal: 5xx on GET, retried within the deadline."""
