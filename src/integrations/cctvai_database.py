"""Deterministic read-only queries over the CCTVAI PostgreSQL reporting replica.

This module has NO SQLite dependency. Every template is a constant; user text
never reaches SQL except through bound parameters.

Connection design
-----------------
No pool. Each query opens a short-lived connection:
* psycopg is imported lazily so the module loads even without the driver
  installed (Phase 7 adds it to requirements.txt, but code and tests can run
  before the image is rebuilt).
* The role already has ``default_transaction_read_only = on`` and
  ``statement_timeout = 30s`` server-side. We add a tighter client-side
  statement timeout (5 s default) so a network hiccup does not block the
  response.
* ``search_path`` is forced in the connection options so even a planner
  mistake cannot bypass the ``cctvai.`` prefix.

Circuit breaker
---------------
``available`` is a cached bool read by ``/health``; it must NEVER do I/O.
Actual health probes run in ``refresh_health()``, called only from the
lifespan background task on a TTL (``CCTVAI_HEALTH_TTL_SECONDS``). Per-query
``query()`` does NOT call ``refresh_health()`` — it only reads the cached
``self._health`` snapshot, which can be up to one TTL period stale. This is
why ``_build_result`` special-cases ``cctvai_data_overview``: that intent's
own SQL already computes a live ``latest_event_at``, which can differ from
the cached health snapshot if a new event lands between probes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from . import cctvai_contract as contract

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy driver import — lets Phases 1-5 run before the image is rebuilt.
# ---------------------------------------------------------------------------
import importlib.util as _importlib_util

_DRIVER_AVAILABLE: bool = _importlib_util.find_spec("psycopg") is not None


# ---------------------------------------------------------------------------
# SQL templates — ALL queries
# ---------------------------------------------------------------------------
# Every template:
#   * qualifies tables as cctvai.<name>
#   * uses LEFT JOIN + AND NOT <alias>.del_flag INSIDE the ON clause
#   * uses count(e.id), never count(*)
#   * converts epoch-ms with to_timestamp(col / 1000.0)
#   * carries a LIMIT
#
# Verified against the replica on 2026-09-03; see handoff §7.

_SQL_EVENTS_BY_SEVERITY = """
SELECT
    COALESCE(vt.severity, '(không rõ)') AS severity,
    count(e.id) AS total
FROM cctvai.event_snapshots e
LEFT JOIN cctvai.cctvai_violation_types vt
       ON LOWER(vt.violation_code) = LOWER(e.violation_type)
      AND NOT vt.del_flag
WHERE to_timestamp(e.detected_time / 1000.0) >= now() - %s::interval
GROUP BY 1
ORDER BY total DESC
LIMIT %s
"""

_SQL_EVENTS_BY_VIOLATION_TYPE = """
SELECT
    COALESCE(vt.violation_name, e.violation_type || ' (không có trong danh mục)') AS violation_name,
    COALESCE(vt.severity, '(không rõ)') AS severity,
    count(e.id) AS total
FROM cctvai.event_snapshots e
LEFT JOIN cctvai.cctvai_violation_types vt
       ON LOWER(vt.violation_code) = LOWER(e.violation_type)
      AND NOT vt.del_flag
WHERE to_timestamp(e.detected_time / 1000.0) >= now() - %s::interval
GROUP BY 1, 2
ORDER BY total DESC
LIMIT %s
"""

_SQL_RECENT_EVENTS = """
SELECT
    COALESCE(c.camera_name, e.camera_id || ' (không còn trong danh mục)') AS camera_name,
    l.line_name,
    vt.violation_name,
    vt.severity,
    to_timestamp(e.detected_time / 1000.0) AS detected_at,
    e.record_status,
    e.confidence_rate
FROM cctvai.event_snapshots e
LEFT JOIN cctvai.cctvai_cameras c
       ON LOWER(c.camera_id) = LOWER(e.camera_id)
      AND NOT c.del_flag
LEFT JOIN cctvai.cctvai_lines l
       ON l.id = c.line_ref_id
      AND NOT l.del_flag
LEFT JOIN cctvai.cctvai_violation_types vt
       ON LOWER(vt.violation_code) = LOWER(e.violation_type)
      AND NOT vt.del_flag
ORDER BY e.detected_time DESC
LIMIT %s
"""

_SQL_ACTIVE_CAMERAS = """
SELECT
    c.camera_id,
    c.camera_name,
    l.line_name,
    c.protocol,
    c.is_active,
    c.description
FROM cctvai.cctvai_cameras c
LEFT JOIN cctvai.cctvai_lines l
       ON l.id = c.line_ref_id
      AND NOT l.del_flag
WHERE NOT c.del_flag
ORDER BY l.line_name, c.camera_name
LIMIT %s
"""

# Uses EXISTS — immune to fan-out, safe even without del_flag on the join side.
_SQL_ORPHAN_EVENTS = """
SELECT
    e.camera_id,
    count(e.id) AS events
FROM cctvai.event_snapshots e
WHERE NOT EXISTS (
    SELECT 1 FROM cctvai.cctvai_cameras c
    WHERE LOWER(c.camera_id) = LOWER(e.camera_id)
      AND NOT c.del_flag
)
GROUP BY 1
ORDER BY events DESC
LIMIT %s
"""

_SQL_EVENTS_BY_CAMERA_LINE = """
SELECT
    COALESCE(c.camera_name, e.camera_id || ' (không còn trong danh mục)') AS camera_name,
    e.camera_id,
    COALESCE(l.line_name, '(không rõ tuyến)') AS line_name,
    count(e.id) AS total
FROM cctvai.event_snapshots e
LEFT JOIN cctvai.cctvai_cameras c
       ON LOWER(c.camera_id) = LOWER(e.camera_id)
      AND NOT c.del_flag
LEFT JOIN cctvai.cctvai_lines l
       ON l.id = c.line_ref_id
      AND NOT l.del_flag
WHERE to_timestamp(e.detected_time / 1000.0) >= now() - %s::interval
GROUP BY 1, 2, 3
ORDER BY total DESC
LIMIT %s
"""

_SQL_DATA_OVERVIEW = """
SELECT
    (SELECT count(e.id) FROM cctvai.event_snapshots e) AS total_events,
    (SELECT count(DISTINCT c.camera_id) FROM cctvai.cctvai_cameras c
       WHERE NOT c.del_flag AND c.is_active) AS active_cameras,
    (SELECT count(DISTINCT vt.violation_code) FROM cctvai.cctvai_violation_types vt WHERE NOT vt.del_flag) AS active_violation_types,
    (SELECT count(DISTINCT l.line_id) FROM cctvai.cctvai_lines l WHERE NOT l.del_flag) AS active_lines,
    (SELECT to_timestamp(max(e.detected_time) / 1000.0)::text FROM cctvai.event_snapshots e) AS latest_event_at,
    (SELECT to_timestamp(min(e.detected_time) / 1000.0)::text FROM cctvai.event_snapshots e) AS oldest_event_at
"""

# Probe query — used by health refresh; minimal impact.
_SQL_HEALTH_PROBE = """
SELECT
    (SELECT count(e.id) FROM cctvai.event_snapshots e) AS event_count,
    (SELECT to_timestamp(max(e.detected_time) / 1000.0)::text
       FROM cctvai.event_snapshots e) AS latest_event_at,
    (SELECT count(c.camera_id) FROM cctvai.cctvai_cameras c
       WHERE NOT c.del_flag AND c.is_active) AS active_camera_count
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CctvaiDatabaseResult:
    intent: str
    rows: list[dict[str, Any]]
    fallback_answer: str
    status: str = "PARTIAL"
    reason_codes: tuple[str, ...] = ()
    domain: str = contract.DOMAIN
    latest_event_at: str = ""
    replica_lag_state: str = contract.REPLICA_LAG_STATE_UNVERIFIED
    grain: str = ""
    schema_version: str = contract.SCHEMA_VERSION
    data_contract_version: str = contract.DATA_CONTRACT_VERSION
    semantic_contract_version: str = contract.SEMANTIC_CONTRACT_VERSION
    source_system: str = contract.SOURCE_SYSTEM

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "domain": self.domain,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "latest_event_at": self.latest_event_at,
            "replica_lag_state": self.replica_lag_state,
            "grain": self.grain,
            "schema_version": self.schema_version,
            "data_contract_version": self.data_contract_version,
            "semantic_contract_version": self.semantic_contract_version,
            "source_system": self.source_system,
        }


# ---------------------------------------------------------------------------
# Health state (shared mutable, only written by refresh_health)
# ---------------------------------------------------------------------------

@dataclass
class _HealthState:
    available: bool = False
    state: str = contract.STATE_DISABLED
    reason_codes: list[str] = field(default_factory=list)
    event_count: int = 0
    active_camera_count: int = 0
    latest_event_at: str = ""
    # circuit breaker
    consecutive_failures: int = 0
    breaker_open_until: float = 0.0
    last_refresh: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CctvaiDatabase:
    """Deterministic read-only gateway to the CCTVAI reporting replica."""

    DEFAULT_INTERVAL = "7 days"
    DEFAULT_LIMIT = 50
    # Circuit breaker thresholds — overridable via env for testing.
    BREAKER_THRESHOLD = 3
    BREAKER_COOLDOWN = 60.0
    HEALTH_TTL = 30.0

    def __init__(
        self,
        *,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        connect_timeout: int = 3,
        statement_timeout_ms: int = 5000,
        health_ttl: float = 30.0,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 60.0,
        max_rows: int = 50,
    ):
        self._host = host
        self._port = port
        self._dbname = dbname
        self._user = user
        self._password = password
        self._connect_timeout = connect_timeout
        self._statement_timeout_ms = statement_timeout_ms
        self._max_rows = max(1, min(int(max_rows), 200))
        self.HEALTH_TTL = health_ttl
        self.BREAKER_THRESHOLD = breaker_threshold
        self.BREAKER_COOLDOWN = breaker_cooldown
        self._health = _HealthState()
        if not _DRIVER_AVAILABLE:
            self._health.state = contract.STATE_DRIVER_MISSING
            self._health.reason_codes = [contract.REASON_DRIVER_MISSING]

    @classmethod
    def from_env(cls) -> "CctvaiDatabase | None":
        """Return None when CCTVAI_DATABASE_ENABLED is not truthy.

        Only reads env; never opens a connection. Safe to call at import time.
        """
        enabled = os.getenv("CCTVAI_DATABASE_ENABLED", "false").lower() in {
            "1", "true", "yes", "on",
        }
        if not enabled:
            return None
        return cls(
            host=os.getenv("CCTVAI_DB_HOST", ""),
            port=int(os.getenv("CCTVAI_DB_PORT", "55434")),
            dbname=os.getenv("CCTVAI_DB_NAME", "cctvai"),
            user=os.getenv("CCTVAI_DB_USER", "cctvai_llm_ro"),
            password=os.getenv("CCTVAI_DB_PASSWORD", ""),
            connect_timeout=int(os.getenv("CCTVAI_DB_CONNECT_TIMEOUT", "3")),
            statement_timeout_ms=int(
                os.getenv("CCTVAI_DB_STATEMENT_TIMEOUT_MS", "5000")
            ),
            health_ttl=float(os.getenv("CCTVAI_HEALTH_TTL_SECONDS", "30")),
            breaker_threshold=int(
                os.getenv("CCTVAI_HEALTH_BREAKER_THRESHOLD", "3")
            ),
            breaker_cooldown=float(
                os.getenv("CCTVAI_HEALTH_BREAKER_COOLDOWN_SECONDS", "60")
            ),
            max_rows=int(os.getenv("CCTVAI_MAX_ROWS", "50")),
        )

    # ------------------------------------------------------------------
    # Public read-only properties (zero I/O)
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Cached availability. Never does I/O — safe for /health."""
        return self._health.available

    @property
    def driver_available(self) -> bool:
        return _DRIVER_AVAILABLE

    def status(self) -> dict[str, Any]:
        """Return health payload for /health. Zero I/O."""
        h = self._health
        payload: dict[str, Any] = {
            "enabled": True,
            "state": h.state,
            "available": h.available,
            "driver_available": _DRIVER_AVAILABLE,
            "reason_codes": list(h.reason_codes),
        }
        if h.available:
            payload["event_count"] = h.event_count
            payload["active_camera_count"] = h.active_camera_count
            payload["latest_event_at"] = h.latest_event_at
            payload.update(contract.contract_versions())
        return payload

    # ------------------------------------------------------------------
    # Health refresh (does real I/O — call from lifespan / bg task only)
    # ------------------------------------------------------------------

    async def refresh_health(self) -> None:
        """Probe the replica and update the cached health state.

        Should be called from an asyncio background task, not from the hot
        request path directly (wrap with ``asyncio.wait_for`` in the caller).
        """
        if not _DRIVER_AVAILABLE:
            self._health.state = contract.STATE_DRIVER_MISSING
            self._health.available = False
            return

        h = self._health
        now = time.monotonic()

        # Breaker: skip probing while open
        if h.breaker_open_until and now < h.breaker_open_until:
            return

        try:
            row = await asyncio.to_thread(self._probe_sync)
        except Exception as exc:
            h.consecutive_failures += 1
            log.warning(
                "CCTVAI health probe failed (%d): %s",
                h.consecutive_failures,
                exc,
            )
            if h.consecutive_failures >= self.BREAKER_THRESHOLD:
                h.breaker_open_until = now + self.BREAKER_COOLDOWN
                log.warning(
                    "CCTVAI circuit breaker opened for %.0fs", self.BREAKER_COOLDOWN
                )
            h.available = False
            h.state = contract.STATE_UNAVAILABLE
            h.reason_codes = [contract.REASON_REPLICA_UNAVAILABLE]
            h.last_refresh = now
            return

        h.consecutive_failures = 0
        h.breaker_open_until = 0.0
        h.event_count = int(row.get("event_count") or 0)
        h.latest_event_at = str(row.get("latest_event_at") or "")
        # Live count, not the 2026-09-03 baseline: /health is what the UI
        # and the demo dashboard read, and a hard-coded 15 would keep claiming
        # 15 after a camera is added or deactivated.
        h.active_camera_count = int(
            row.get("active_camera_count") or contract.BASELINE_ACTIVE_CAMERA_COUNT
        )
        h.available = True
        h.state = contract.STATE_READY
        h.reason_codes = [contract.REASON_REPLICA_LAG_UNVERIFIED]
        h.last_refresh = now

        # Tripwire: if total event count exceeds baseline by >50%, a fan-out
        # slipped past validation somewhere — surface it in the health payload.
        if h.event_count > contract.BASELINE_EVENT_COUNT * 1.5:
            h.reason_codes.append(contract.REASON_FANOUT_SUSPECTED)
            log.error(
                "CCTVAI health: event_count=%d exceeds baseline=%d × 1.5 "
                "— possible fan-out from an unverified query",
                h.event_count,
                contract.BASELINE_EVENT_COUNT,
            )

    def _probe_sync(self) -> dict[str, Any]:
        """Blocking probe — run in a thread via asyncio.to_thread."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_SQL_HEALTH_PROBE)
                cols = [desc[0] for desc in cur.description]
                row = cur.fetchone()
                return dict(zip(cols, row)) if row else {}

    # ------------------------------------------------------------------
    # Intent router
    # ------------------------------------------------------------------

    def route_question(self, question: str) -> str:
        """Classify a question into one deterministic intent constant.

        Uses keyword matching on a normalised copy of the question, same
        pattern as ``MesWmsDatabase.is_wms_question``. Falls back to
        ``INTENT_SCOPE_CLARIFICATION`` when nothing matches (the caller can
        then try the SQL agent).
        """
        q = question.lower()

        # Suppressed intents — fail closed before anything else
        if any(kw in q for kw in (
            "mật khẩu", "password", "username", "url camera", "rtsp", "endpoint",
            "パスワード", "ユーザー名",
        )):
            return contract.INTENT_CREDENTIALS_SUPPRESSED

        if any(kw in q for kw in (
            "người phụ trách", "phụ trách", "tên người", "email", "nhân viên tuyến",
            "担当者", "担当",
        )):
            return contract.INTENT_RESPONSIBLE_SUPPRESSED

        if any(kw in q for kw in (
            "thời gian diễn ra", "đang diễn ra", "kéo dài bao lâu", "duration",
            "chưa kết thúc", "đang mở",
            "継続時間", "経過時間",
        )):
            return contract.INTENT_OPEN_DURATION_SUPPRESSED

        # Deterministic intents
        if any(kw in q for kw in (
            "mồ côi", "không có trong master", "camera lạ", "camera không rõ",
            "orphan",
        )):
            return contract.INTENT_ORPHAN_EVENTS

        if any(kw in q for kw in (
            "tổng quan", "overview", "bao nhiêu sự kiện tổng", "dữ liệu như thế nào",
            "tổng cộng", "toàn bộ dữ liệu", "snapshot", "quy mô",
        )):
            return contract.INTENT_DATA_OVERVIEW

        if any(kw in q for kw in (
            "camera", "danh sách camera", "camera nào", "camera đang hoạt động",
            "camera active",
            "カメラ",
        )):
            # Camera-specific event breakdown vs camera list
            if any(kw in q for kw in ("sự kiện", "vi phạm", "event")):
                return contract.INTENT_EVENTS_BY_CAMERA_LINE
            return contract.INTENT_ACTIVE_CAMERAS

        if any(kw in q for kw in (
            "tuyến", "line", "khu vực",
            "ライン", "エリア",
        )):
            return contract.INTENT_EVENTS_BY_CAMERA_LINE

        if any(kw in q for kw in (
            "mức độ", "severity", "nghiêm trọng", "alert", "normal",
            "深刻度", "重要度",
        )):
            return contract.INTENT_EVENTS_BY_SEVERITY

        if any(kw in q for kw in (
            "loại vi phạm", "violation", "vi phạm nào", "mã vi phạm",
            "违反", "違反",
        )):
            return contract.INTENT_EVENTS_BY_VIOLATION_TYPE

        if any(kw in q for kw in (
            "gần nhất", "mới nhất", "recent", "vừa xảy ra", "sự kiện gần đây",
            "最新", "最近",
        )):
            return contract.INTENT_RECENT_EVENTS

        if any(kw in q for kw in (
            "sự kiện", "event", "vi phạm",
            "イベント", "違反",
        )):
            return contract.INTENT_EVENTS_BY_VIOLATION_TYPE

        return contract.INTENT_SCOPE_CLARIFICATION

    # ------------------------------------------------------------------
    # Public query entry point
    # ------------------------------------------------------------------

    async def query(
        self,
        question: str,
        *,
        language: str = "vi",
    ) -> CctvaiDatabaseResult:
        """Route and execute. Never raises; returns a fail-closed result."""
        if not _DRIVER_AVAILABLE:
            return self._unavailable_result(
                contract.INTENT_SCOPE_CLARIFICATION,
                contract.REASON_DRIVER_MISSING,
                language,
            )
        if not self.available:
            return self._unavailable_result(
                contract.INTENT_SCOPE_CLARIFICATION,
                contract.REASON_REPLICA_UNAVAILABLE,
                language,
            )

        intent = self.route_question(question)

        # Suppressed — no DB access needed
        suppressed_reason = contract.SUPPRESSED_INTENT_REASONS.get(intent)
        if suppressed_reason:
            return self._suppressed_result(intent, suppressed_reason, language)

        # Scope clarification — let caller try SQL agent
        if intent == contract.INTENT_SCOPE_CLARIFICATION:
            return self._scope_clarification_result(language)

        try:
            rows = await asyncio.to_thread(self._execute_intent, intent)
        except Exception as exc:
            log.warning("CCTVAI query error for intent %s: %s", intent, exc)
            return self._error_result(intent, language)

        return self._build_result(intent, rows, language)

    # ------------------------------------------------------------------
    # Synchronous execution (runs in thread)
    # ------------------------------------------------------------------

    def _execute_intent(self, intent: str) -> list[dict[str, Any]]:
        interval = self.DEFAULT_INTERVAL
        limit = self._max_rows
        with self._connect() as conn:
            with conn.cursor() as cur:
                if intent == contract.INTENT_EVENTS_BY_SEVERITY:
                    cur.execute(_SQL_EVENTS_BY_SEVERITY, (interval, limit))
                elif intent == contract.INTENT_EVENTS_BY_VIOLATION_TYPE:
                    cur.execute(_SQL_EVENTS_BY_VIOLATION_TYPE, (interval, limit))
                elif intent == contract.INTENT_RECENT_EVENTS:
                    cur.execute(_SQL_RECENT_EVENTS, (limit,))
                elif intent == contract.INTENT_ACTIVE_CAMERAS:
                    cur.execute(_SQL_ACTIVE_CAMERAS, (limit,))
                elif intent == contract.INTENT_ORPHAN_EVENTS:
                    cur.execute(_SQL_ORPHAN_EVENTS, (limit,))
                elif intent == contract.INTENT_EVENTS_BY_CAMERA_LINE:
                    cur.execute(_SQL_EVENTS_BY_CAMERA_LINE, (interval, limit))
                elif intent == contract.INTENT_DATA_OVERVIEW:
                    cur.execute(_SQL_DATA_OVERVIEW)
                else:
                    raise ValueError(f"Unknown intent: {intent}")
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def execute_ad_hoc(self, sql: str) -> list[dict[str, Any]]:
        """Execute an already-validated, read-only ad-hoc SQL statement.

        Callers MUST validate ``sql`` via ``CctvaiSqlAgent.validate_sql()``
        first — this method performs no safety checks of its own, only runs
        exactly the text it is given through the same short-lived, read-only
        connection settings (forced ``search_path``, statement timeout) as
        the deterministic templates in ``_execute_intent``.

        Called with no ``params`` argument (unlike the deterministic
        templates, which bind ``%s`` placeholders): the SQL agent renders a
        complete literal SQL string via sqlglot, not a parametrized template.
        psycopg only attempts ``%``-style placeholder substitution when a
        non-``None`` ``params`` argument is passed, so a literal ``%`` in the
        text (e.g. inside a string literal) is not misinterpreted here.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _connect(self):
        """Open a short-lived read-only connection with safe options."""
        if not _DRIVER_AVAILABLE:
            raise RuntimeError("psycopg driver not installed")
        import psycopg  # noqa: PLC0415

        return psycopg.connect(
            host=self._host,
            port=self._port,
            dbname=self._dbname,
            user=self._user,
            password=self._password,
            connect_timeout=self._connect_timeout,
            options=(
                f"-c search_path=cctvai,public "
                f"-c statement_timeout={self._statement_timeout_ms}"
            ),
        )

    # ------------------------------------------------------------------
    # Result constructors
    # ------------------------------------------------------------------

    def _build_result(
        self,
        intent: str,
        rows: list[dict[str, Any]],
        language: str,
    ) -> CctvaiDatabaseResult:
        grain = contract.INTENT_GRAINS.get(intent, "")
        answer = self._format_rows(intent, rows, language)
        latest = self._health.latest_event_at
        if (
            intent == contract.INTENT_DATA_OVERVIEW
            and rows
            and rows[0].get("latest_event_at")
        ):
            # cctvai_data_overview đã tự tính latest_event_at LIVE trong câu
            # query của chính nó (khác _health.latest_event_at là snapshot
            # cache, có thể cũ tới 1 chu kỳ TTL). Ưu tiên giá trị live để
            # không hiện 2 mốc "sự kiện mới nhất" khác nhau trong cùng 1 tin
            # nhắn (giá trị trong thân câu trả lời và trong disclaimer).
            latest = rows[0]["latest_event_at"]
        answer = self._with_freshness(answer, latest_event_at=latest, language=language)
        reason_codes = (contract.REASON_REPLICA_LAG_UNVERIFIED,)
        return CctvaiDatabaseResult(
            intent=intent,
            rows=rows,
            fallback_answer=answer,
            status="PARTIAL",
            reason_codes=reason_codes,
            grain=grain,
            latest_event_at=latest,
        )

    def _unavailable_result(
        self, intent: str, reason: str, language: str
    ) -> CctvaiDatabaseResult:
        if language == "ja":
            answer = "CCTVAIのレポートデータは現在利用できません。"
        else:
            answer = "Dữ liệu báo cáo CCTVAI hiện không khả dụng."
        return CctvaiDatabaseResult(
            intent=intent,
            rows=[],
            fallback_answer=answer,
            status="PARTIAL",
            reason_codes=(reason,),
        )

    def _suppressed_result(
        self, intent: str, reason: str, language: str
    ) -> CctvaiDatabaseResult:
        messages = {
            contract.REASON_CAMERA_CREDENTIALS_UNREADABLE: (
                "Thông tin đăng nhập camera (URL, tài khoản, mật khẩu) không nằm "
                "trong quyền đọc của hệ thống báo cáo. "
                "Liên hệ team ICT nếu cần thông tin này.",
                "カメラの認証情報（URL・アカウント・パスワード）はレポート用の読み取り権限に含まれていません。"
                "必要な場合はICTチームにお問い合わせください。",
            ),
            contract.REASON_RESPONSIBLE_IDENTITY_UNAVAILABLE: (
                "Dữ liệu báo cáo chỉ lưu mã số người phụ trách, không có "
                "tên hay email nên không thể tra ra danh tính cụ thể.",
                "レポートデータには担当者IDのみが保存されており、氏名やメールは"
                "含まれないため、担当者を特定することはできません。",
            ),
            contract.REASON_OPEN_EVENT_DURATION_UNAVAILABLE: (
                "Không tính được thời lượng của sự kiện đang diễn ra vì "
                "chưa có thời điểm kết thúc. Chỉ tính được thời lượng của "
                "các sự kiện đã kết thúc.",
                "進行中のイベントは終了時刻が未記録のため継続時間を計算できません。"
                "完了済みイベントの継続時間のみ計算可能です。",
            ),
        }
        vi, ja = messages.get(reason, ("Không hỗ trợ yêu cầu này.", "この要求には対応していません。"))
        answer = ja if language == "ja" else vi
        return CctvaiDatabaseResult(
            intent=intent,
            rows=[],
            fallback_answer=answer,
            status="SUPPRESSED",
            reason_codes=(reason,),
        )

    def _scope_clarification_result(self, language: str) -> CctvaiDatabaseResult:
        if language == "ja":
            answer = (
                "ご質問がCCTVAIレポートデータで回答できる範囲か判断できませんでした。"
                "違反イベントの集計、稼働中カメラの一覧、ライン別の集計、"
                "データ概要のいずれかでしょうか？"
            )
        else:
            answer = (
                "Câu hỏi này chưa khớp với các nội dung tôi tra cứu được. "
                "Bạn muốn xem thống kê vi phạm theo loại/camera/tuyến, danh "
                "sách camera đang hoạt động, hay tổng quan dữ liệu CCTVAI?"
            )
        return CctvaiDatabaseResult(
            intent=contract.INTENT_SCOPE_CLARIFICATION,
            rows=[],
            fallback_answer=answer,
            status="PARTIAL",
            reason_codes=(contract.REASON_SCOPE_UNSUPPORTED,),
        )

    def _error_result(self, intent: str, language: str) -> CctvaiDatabaseResult:
        if language == "ja":
            answer = "CCTVAIデータの照会中にエラーが発生しました。しばらくしてから再試行してください。"
        else:
            answer = (
                "Có lỗi khi truy vấn dữ liệu báo cáo CCTVAI. "
                "Vui lòng thử lại sau ít phút."
            )
        return CctvaiDatabaseResult(
            intent=intent,
            rows=[],
            fallback_answer=answer,
            status="PARTIAL",
            reason_codes=(contract.REASON_REPLICA_QUERY_ERROR,),
        )

    # ------------------------------------------------------------------
    # Row formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_rows(
        intent: str, rows: list[dict[str, Any]], language: str
    ) -> str:
        if not rows:
            return (
                "データがありません。" if language == "ja"
                else "Không có dữ liệu phù hợp trong replica."
            )

        header = {
            contract.INTENT_EVENTS_BY_SEVERITY: (
                "Sự kiện theo mức độ nghiêm trọng (7 ngày gần nhất):",
                "重要度別イベント（直近7日間）:",
            ),
            contract.INTENT_EVENTS_BY_VIOLATION_TYPE: (
                "Sự kiện theo loại vi phạm (7 ngày gần nhất):",
                "違反種別イベント（直近7日間）:",
            ),
            contract.INTENT_RECENT_EVENTS: (
                "Các sự kiện gần nhất:",
                "最近のイベント:",
            ),
            contract.INTENT_ACTIVE_CAMERAS: (
                "Danh sách camera đang hoạt động:",
                "稼働中カメラ一覧:",
            ),
            contract.INTENT_ORPHAN_EVENTS: (
                "Sự kiện của camera không còn trong danh mục:",
                "台帳外カメラのイベント:",
            ),
            contract.INTENT_EVENTS_BY_CAMERA_LINE: (
                "Sự kiện theo camera/tuyến (7 ngày gần nhất):",
                "カメラ／ライン別イベント（直近7日間）:",
            ),
            contract.INTENT_DATA_OVERVIEW: (
                "Tổng quan dữ liệu CCTVAI:",
                "CCTVAIデータ概要:",
            ),
        }
        vi_header, ja_header = header.get(
            intent, ("Kết quả:", "結果:")
        )
        title = ja_header if language == "ja" else vi_header
        lines = [title]
        shown = rows[: contract.DISPLAY_ROW_LIMIT]
        for row in shown:
            parts = []
            for key, value in row.items():
                if value is None or value == "":
                    continue
                label = contract.display_label(key, language)
                rendered = contract.format_display_value(key, value, language)
                parts.append(f"{label}: {rendered}")
            lines.append("- " + ", ".join(parts) if parts else "- (trống)")
        hidden = len(rows) - len(shown)
        if hidden > 0:
            lines.append(
                f"…và {hidden} dòng nữa (xem báo cáo để có danh sách đầy đủ)."
                if language != "ja"
                else f"…他{hidden}件（全件はレポートをご確認ください）。"
            )
        return "\n".join(lines)

    @staticmethod
    def _with_freshness(
        answer: str,
        *,
        latest_event_at: str,
        language: str,
    ) -> str:
        # Same trimmed form as the answer body — an answer showing
        # "2026-09-04 10:26" in one line and "2026-09-04 10:26:32.122+00" in
        # the next looks like two different numbers to a reader.
        latest = contract.trim_timestamp(latest_event_at) if latest_event_at else ""
        if language == "ja":
            note = (
                f"\n\nデータはCCTVAIレポートレプリカから取得（非同期レプリケーション、遅延は未確認）。"
                f"最新イベント: {latest or '未確認'}。"
            )
        else:
            note = (
                f"\n\nNguồn: replica báo cáo CCTVAI — không phải dữ liệu thời "
                f"gian thực, độ trễ chưa xác minh. "
                f"Sự kiện mới nhất: {latest or 'chưa xác định'}."
            )
        return answer + note
