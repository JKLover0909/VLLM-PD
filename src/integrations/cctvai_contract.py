"""Shared CCTVAI reporting-replica schema, capability, and safety contract.

Single source of truth for what the CCTVAI PostgreSQL read replica exposes and
what may never be inferred from it. Both ``cctvai_database`` (deterministic
templates) and ``cctvai_sql_agent`` (LLM planner) import from here so the two
paths cannot drift apart.

Every fact below was verified with live queries against the replica on
2026-09-03; see ``Markdowns/cctvai_llm_report_replica_handoff.md``.
"""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "1"
DATA_CONTRACT_VERSION = "cctvai-replica-v1"
SEMANTIC_CONTRACT_VERSION = "cctvai-report-v1"
SOURCE_SCHEMA = "cctvai"
SOURCE_SYSTEM = "CCTVAI_REPORTING_REPLICA"
DOMAIN = "cctvai"

# The replica syncs asynchronously. Client code cannot measure the lag, so the
# only honest state is "unverified" — never claim realtime.
REPLICA_LAG_STATE_UNVERIFIED = "REPLICA_LAG_UNVERIFIED"

REASON_CCTVAI_DISABLED = "CCTVAI_DISABLED"
REASON_DRIVER_MISSING = "CCTVAI_DRIVER_MISSING"
REASON_REPLICA_UNAVAILABLE = "CCTVAI_REPLICA_UNAVAILABLE"
REASON_REPLICA_INCOMPATIBLE = "CCTVAI_REPLICA_INCOMPATIBLE"
REASON_REPLICA_QUERY_ERROR = "CCTVAI_REPLICA_QUERY_ERROR"
REASON_REPLICA_LAG_UNVERIFIED = "CCTVAI_REPLICA_LAG_UNVERIFIED"
REASON_ORPHAN_EVENTS_PRESENT = "CCTVAI_ORPHAN_EVENTS_PRESENT"
REASON_OPEN_EVENT_DURATION_UNAVAILABLE = "CCTVAI_OPEN_EVENT_DURATION_UNAVAILABLE"
REASON_RESPONSIBLE_IDENTITY_UNAVAILABLE = "CCTVAI_RESPONSIBLE_IDENTITY_UNAVAILABLE"
REASON_CAMERA_CREDENTIALS_UNREADABLE = "CCTVAI_CAMERA_CREDENTIALS_UNREADABLE"
REASON_SCOPE_UNSUPPORTED = "CCTVAI_SCOPE_UNSUPPORTED"
REASON_SQL_AGENT_UNVERIFIED = "CCTVAI_SQL_AGENT_ANSWER_UNVERIFIED"
# An INNER JOIN from event_snapshots to a master silently drops the 26 events
# whose camera is not in the master. Legal, but the answer must say so.
REASON_INNER_JOIN_DROPS_ORPHANS = "CCTVAI_INNER_JOIN_DROPS_ORPHAN_EVENTS"
# Row count exceeded the measured baseline by more than tolerance — the shape
# check let a fan-out through.
REASON_FANOUT_SUSPECTED = "CCTVAI_ROW_COUNT_FANOUT_SUSPECTED"

# Column-level GRANTs, verified via information_schema.column_privileges.
# `SELECT *` on cctvai_cameras fails with "permission denied for table" — so the
# planner and the templates must both stay inside these sets. Checking here
# fails closed before the query leaves the process.
READABLE_COLUMNS: dict[str, frozenset[str]] = {
    "cctvai_lines": frozenset(
        {
            "id",
            "line_id",
            "line_name",
            "description",
            "is_active",
            "del_flag",
            "create_date",
            "edit_date",
        }
    ),
    # 11 of 19 columns. The 8 missing ones are physical-camera device logins.
    "cctvai_cameras": frozenset(
        {
            "id",
            "camera_id",
            "camera_name",
            "line_ref_id",
            "protocol",
            "description",
            "is_active",
            "del_flag",
            "create_date",
            "edit_date",
            "camera_name_translations",
        }
    ),
    "cctvai_violation_types": frozenset(
        {
            "id",
            "violation_code",
            "violation_name",
            "severity",
            "description",
            "del_flag",
            "create_date",
            "edit_date",
        }
    ),
    "cctvai_camera_violation_mappings": frozenset(
        {"camera_ref_id", "violation_ref_id", "create_date", "edit_date"}
    ),
    "cctvai_line_responsibles": frozenset(
        {"line_ref_id", "user_id", "create_date", "edit_date"}
    ),
    "event_snapshots": frozenset(
        {
            "id",
            "camera_id",
            "violation_type",
            "detected_time",
            "end_time",
            "thumbnail_path",
            "thumbnail_metadata",
            "video_path",
            "confidence_rate",
            "created_at",
            "record_status",
            "updated_at",
        }
    ),
}

# Deliberately unreadable: physical camera device credentials, some stored in
# plaintext upstream. Never selected, never named in an answer.
BLOCKED_CAMERA_COLUMNS = frozenset(
    {
        "username",
        "password",
        "endpoint",
        "main_cam_url",
        "sub_cam_url",
        "speaker_url",
        "speaker_topic",
        "light_topic",
    }
)

# Internal queues and the migration bookkeeping table. Readable, but out of
# reporting scope — keeping them off the allowlist keeps the planner focused.
OUT_OF_SCOPE_TABLES = frozenset(
    {
        "cctvai_notification_outbox",
        "cctvai_speaker_outbox",
        "schema_migrations",
    }
)

REPORTABLE_TABLES = frozenset(READABLE_COLUMNS)

# THE trap of this schema. `camera_id` and `violation_code` are NOT unique
# across these tables: cam008 has 4 rows (1 active, 3 deleted), and
# violation_code 'glove_not_changed' has 2 rows with different severity. Only
# `del_flag = FALSE` makes the code a unique key. Forget it in a JOIN and the
# row count inflates by +44% (11,986 instead of 8,305) while the query still
# succeeds silently.
SOFT_DELETE_TABLES = frozenset(
    {"cctvai_cameras", "cctvai_violation_types", "cctvai_lines"}
)

# event_snapshots has no hard FK to the masters. Verified: 26 events match no
# active camera, 3 carry an unknown violation_type. INNER JOIN drops them
# silently, so reporting must use LEFT JOIN + COALESCE and count(<events>.id).
EVENT_TABLE = "event_snapshots"
EVENT_COUNT_COLUMN = "id"

# detected_time / end_time are epoch MILLISECONDS, not timestamps.
EPOCH_MILLISECOND_COLUMNS = frozenset({"detected_time", "end_time"})
EPOCH_MS_DIVISOR = "1000.0"

# Baseline row counts measured 2026-09-03. Used as a fan-out tripwire: a total
# event count above this by more than the tolerance means a JOIN duplicated
# rows rather than that new events arrived.
BASELINE_EVENT_COUNT = 8305
BASELINE_ACTIVE_CAMERA_COUNT = 15
BASELINE_ACTIVE_VIOLATION_TYPE_COUNT = 15
BASELINE_ACTIVE_LINE_COUNT = 2
BASELINE_ORPHAN_EVENT_COUNT = 26
BASELINE_OPEN_EVENT_COUNT = 166

# Questions that must always fail closed. The replica genuinely cannot answer
# these, and guessing would be worse than refusing.
SUPPRESSED_CAPABILITIES: dict[str, str] = {
    "CAMERA_CREDENTIALS": REASON_CAMERA_CREDENTIALS_UNREADABLE,
    "RESPONSIBLE_IDENTITY": REASON_RESPONSIBLE_IDENTITY_UNAVAILABLE,
    "OPEN_EVENT_DURATION": REASON_OPEN_EVENT_DURATION_UNAVAILABLE,
}

BASE_CAPABILITY_STATUSES: dict[str, tuple[str, str]] = {
    "EVENTS_BY_SEVERITY": ("PARTIAL", REASON_REPLICA_LAG_UNVERIFIED),
    "EVENTS_BY_VIOLATION_TYPE": ("PARTIAL", REASON_REPLICA_LAG_UNVERIFIED),
    "EVENTS_BY_CAMERA_LINE": ("PARTIAL", REASON_REPLICA_LAG_UNVERIFIED),
    "RECENT_EVENTS": ("PARTIAL", REASON_REPLICA_LAG_UNVERIFIED),
    "ACTIVE_CAMERAS": ("PARTIAL", REASON_REPLICA_LAG_UNVERIFIED),
    "ORPHAN_EVENTS": ("PARTIAL", REASON_ORPHAN_EVENTS_PRESENT),
    "DATA_OVERVIEW": ("PARTIAL", REASON_REPLICA_LAG_UNVERIFIED),
    **{
        capability: ("SUPPRESSED", reason)
        for capability, reason in SUPPRESSED_CAPABILITIES.items()
    },
}

VALID_CAPABILITY_STATUSES = frozenset({"AVAILABLE", "PARTIAL", "SUPPRESSED"})

STATE_DISABLED = "DISABLED"
STATE_DRIVER_MISSING = "DRIVER_MISSING"
STATE_UNAVAILABLE = "UNAVAILABLE"
STATE_INCOMPATIBLE = "INCOMPATIBLE"
STATE_QUERY_ERROR = "QUERY_ERROR"
STATE_READY = "READY"

VALID_STATES = frozenset(
    {
        STATE_DISABLED,
        STATE_DRIVER_MISSING,
        STATE_UNAVAILABLE,
        STATE_INCOMPATIBLE,
        STATE_QUERY_ERROR,
        STATE_READY,
    }
)

# Deterministic intents. Each maps to one constant SQL template in
# cctvai_database; user text never reaches SQL except through bound params.
INTENT_EVENTS_BY_SEVERITY = "cctvai_events_by_severity"
INTENT_EVENTS_BY_VIOLATION_TYPE = "cctvai_events_by_violation_type"
INTENT_EVENTS_BY_CAMERA_LINE = "cctvai_events_by_camera_line"
INTENT_RECENT_EVENTS = "cctvai_recent_events"
INTENT_ACTIVE_CAMERAS = "cctvai_active_cameras"
INTENT_ORPHAN_EVENTS = "cctvai_orphan_events"
INTENT_DATA_OVERVIEW = "cctvai_data_overview"
INTENT_SCOPE_CLARIFICATION = "cctvai_scope_clarification"
INTENT_CREDENTIALS_SUPPRESSED = "cctvai_camera_credentials_suppressed"
INTENT_RESPONSIBLE_SUPPRESSED = "cctvai_responsible_identity_suppressed"
INTENT_OPEN_DURATION_SUPPRESSED = "cctvai_open_event_duration_suppressed"

DETERMINISTIC_INTENTS = (
    INTENT_EVENTS_BY_SEVERITY,
    INTENT_EVENTS_BY_VIOLATION_TYPE,
    INTENT_EVENTS_BY_CAMERA_LINE,
    INTENT_RECENT_EVENTS,
    INTENT_ACTIVE_CAMERAS,
    INTENT_ORPHAN_EVENTS,
    INTENT_DATA_OVERVIEW,
)

SUPPRESSED_INTENT_REASONS: dict[str, str] = {
    INTENT_CREDENTIALS_SUPPRESSED: REASON_CAMERA_CREDENTIALS_UNREADABLE,
    INTENT_RESPONSIBLE_SUPPRESSED: REASON_RESPONSIBLE_IDENTITY_UNAVAILABLE,
    INTENT_OPEN_DURATION_SUPPRESSED: REASON_OPEN_EVENT_DURATION_UNAVAILABLE,
}

INTENT_CAPABILITIES: dict[str, str] = {
    INTENT_EVENTS_BY_SEVERITY: "EVENTS_BY_SEVERITY",
    INTENT_EVENTS_BY_VIOLATION_TYPE: "EVENTS_BY_VIOLATION_TYPE",
    INTENT_EVENTS_BY_CAMERA_LINE: "EVENTS_BY_CAMERA_LINE",
    INTENT_RECENT_EVENTS: "RECENT_EVENTS",
    INTENT_ACTIVE_CAMERAS: "ACTIVE_CAMERAS",
    INTENT_ORPHAN_EVENTS: "ORPHAN_EVENTS",
    INTENT_DATA_OVERVIEW: "DATA_OVERVIEW",
    INTENT_CREDENTIALS_SUPPRESSED: "CAMERA_CREDENTIALS",
    INTENT_RESPONSIBLE_SUPPRESSED: "RESPONSIBLE_IDENTITY",
    INTENT_OPEN_DURATION_SUPPRESSED: "OPEN_EVENT_DURATION",
}

# Row grain per intent, so an answer can state what one row means.
INTENT_GRAINS: dict[str, str] = {
    INTENT_EVENTS_BY_SEVERITY: "severity",
    INTENT_EVENTS_BY_VIOLATION_TYPE: "violation_code",
    INTENT_EVENTS_BY_CAMERA_LINE: "camera_id,line_name",
    INTENT_RECENT_EVENTS: "event",
    INTENT_ACTIVE_CAMERAS: "camera_id",
    INTENT_ORPHAN_EVENTS: "camera_id",
    INTENT_DATA_OVERVIEW: "snapshot",
}


def capability_status(capability: str) -> tuple[str, str]:
    """Return ``(status, reason_code)`` for a capability, unknown → suppressed."""
    return BASE_CAPABILITY_STATUSES.get(
        capability, ("SUPPRESSED", REASON_SCOPE_UNSUPPORTED)
    )


def readable_columns(table: str) -> frozenset[str]:
    """Readable columns for ``table``; empty when the table is out of scope."""
    return READABLE_COLUMNS.get(table, frozenset())


def is_column_readable(table: str, column: str) -> bool:
    return column in READABLE_COLUMNS.get(table, frozenset())


def unreadable_columns(table: str, columns: Any) -> list[str]:
    """Sorted columns of ``table`` that the replica role cannot read.

    Used to fail closed before a query is sent rather than relying on the
    server's ``permission denied`` message.
    """
    allowed = READABLE_COLUMNS.get(table)
    if allowed is None:
        return sorted({str(column) for column in columns or ()})
    return sorted({str(column) for column in columns or ()} - allowed)


def requires_soft_delete_filter(table: str) -> bool:
    return table in SOFT_DELETE_TABLES


def epoch_ms_to_timestamp(column: str) -> str:
    """SQL fragment converting an epoch-millisecond column to a timestamp."""
    return f"to_timestamp({column} / {EPOCH_MS_DIVISOR})"


def qualified(table: str) -> str:
    """Schema-qualified name; the role's search_path does not include cctvai."""
    return f"{SOURCE_SCHEMA}.{table}"


def contract_versions() -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "semantic_contract_version": SEMANTIC_CONTRACT_VERSION,
        "source_schema": SOURCE_SCHEMA,
        "source_system": SOURCE_SYSTEM,
    }


# ---------------------------------------------------------------------------
# Display formatting (user-facing answers)
# ---------------------------------------------------------------------------
# Raw SQL column names must never reach the chat UI: an end user asking in
# Vietnamese should not read "active violation types: 15, latest event at:
# 2026-09-04 10:26:32.122+00". Both answer paths (deterministic templates in
# cctvai_database and the LLM SQL agent's template fallback) render through
# these labels so the two cannot drift apart.

DISPLAY_LABELS: dict[str, tuple[str, str]] = {
    # Overview aggregates
    "total_events": ("Tổng sự kiện", "累計イベント数"),
    "active_cameras": ("Camera đang hoạt động", "稼働中カメラ"),
    "active_violation_types": ("Loại vi phạm đang dùng", "有効な違反種別"),
    "active_lines": ("Tuyến đang hoạt động", "稼働中ライン"),
    "latest_event_at": ("Sự kiện mới nhất", "最新イベント"),
    "oldest_event_at": ("Sự kiện cũ nhất", "最古イベント"),
    # Counts
    "total": ("Số sự kiện", "イベント数"),
    "events": ("Số sự kiện", "イベント数"),
    "event_count": ("Số sự kiện", "イベント数"),
    # Master data
    "camera_id": ("Mã camera", "カメラID"),
    "camera_name": ("Camera", "カメラ"),
    "line_name": ("Tuyến", "ライン"),
    "protocol": ("Giao thức", "プロトコル"),
    "is_active": ("Đang hoạt động", "稼働中"),
    "description": ("Ghi chú", "備考"),
    # Violations
    "violation_name": ("Loại vi phạm", "違反種別"),
    "violation_type": ("Mã vi phạm", "違反コード"),
    "violation_code": ("Mã vi phạm", "違反コード"),
    "severity": ("Mức độ", "重要度"),
    # Event detail
    "detected_at": ("Thời điểm", "検知時刻"),
    "record_status": ("Trạng thái ghi hình", "録画状態"),
    "confidence_rate": ("Độ tin cậy", "信頼度"),
}

# Columns holding a 0..1 ratio that reads better as a percentage.
RATIO_COLUMNS = frozenset({"confidence_rate"})

# Enum-ish values stored in English. Verified live on the replica: severity is
# only alert/normal, record_status only recorded/recording. Unknown values pass
# through untranslated rather than being guessed at.
DISPLAY_VALUES: dict[str, dict[str, tuple[str, str]]] = {
    "severity": {
        "alert": ("Cảnh báo", "警告"),
        "normal": ("Thường", "通常"),
    },
    "record_status": {
        "recorded": ("Đã ghi", "録画済み"),
        "recording": ("Đang ghi", "録画中"),
    },
}

# Rows shown inline in the chat answer. The SQL LIMIT stays higher (CCTVAI_MAX_ROWS)
# so counts remain correct; this only caps how much is pasted into one message —
# 50 recent events rendered inline is an 11k-character wall of text.
DISPLAY_ROW_LIMIT = 10

_TIMESTAMP_PATTERN = None  # lazily compiled in trim_timestamp


def display_label(column: str, language: str = "vi") -> str:
    """Human label for a result column; falls back to a de-underscored name."""
    labels = DISPLAY_LABELS.get(column)
    if labels:
        return labels[1] if language == "ja" else labels[0]
    return column.replace("_", " ")


def trim_timestamp(text: str) -> str:
    """Cut Postgres/psycopg timestamps down to minutes for display.

    ``2026-09-04 10:26:32.122000+00:00`` and ``2026-09-04 10:26:32.122+00``
    both become ``2026-09-04 10:26``; anything else is returned unchanged.
    """
    global _TIMESTAMP_PATTERN
    if _TIMESTAMP_PATTERN is None:
        import re  # noqa: PLC0415 - keep module import-time cost at zero

        _TIMESTAMP_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})")
    match = _TIMESTAMP_PATTERN.match(text)
    if not match:
        return text
    return f"{match.group(1)} {match.group(2)}"


def format_display_value(column: str, value: Any, language: str = "vi") -> str:
    """Render one cell for a chat answer: no raw bools, floats or timestamps."""
    if isinstance(value, bool):
        if language == "ja":
            return "はい" if value else "いいえ"
        return "Có" if value else "Không"
    translations = DISPLAY_VALUES.get(column)
    if translations and isinstance(value, str):
        pair = translations.get(value.strip().lower())
        if pair:
            return pair[1] if language == "ja" else pair[0]
    if column in RATIO_COLUMNS and isinstance(value, (int, float)):
        return f"{float(value) * 100:.0f}%"
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}".replace(",", ".")
        return f"{value:.2f}"
    return trim_timestamp(str(value))
