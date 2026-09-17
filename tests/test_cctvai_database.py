"""Tests for CctvaiDatabase — fake connection, no real Postgres needed.

Pattern mirrors tests/test_wms_sql_agent.py: inject a fake connection
factory so every test is self-contained and deterministic.

Key things being tested:
* Intent classification for all deterministic intents
* Suppressed-intent short-circuits (no DB access, correct reason code)
* SQL template correctness: del_flag in ON, count(e.id) not count(*),
  to_timestamp(col / 1000.0), LIMIT present
* Circuit breaker: 3 consecutive failures → available=False for cooldown
* status() payload never exposes host / user / password / port string
* unavailable path when driver is missing
"""

from __future__ import annotations

import asyncio
import re
import time
from unittest.mock import MagicMock, patch

import pytest

from src.integrations import cctvai_contract as contract
from src.integrations.cctvai_database import (
    CctvaiDatabase,
    CctvaiDatabaseResult,
    _SQL_ACTIVE_CAMERAS,
    _SQL_DATA_OVERVIEW,
    _SQL_EVENTS_BY_CAMERA_LINE,
    _SQL_EVENTS_BY_SEVERITY,
    _SQL_EVENTS_BY_VIOLATION_TYPE,
    _SQL_HEALTH_PROBE,
    _SQL_ORPHAN_EVENTS,
    _SQL_RECENT_EVENTS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENSITIVE_STRINGS = ("55434", "cctvai_llm_ro", "fake-pass", "password", "192.86")


def _make_db(**kwargs) -> CctvaiDatabase:
    defaults = dict(
        host="fake-host",
        port=55434,
        dbname="cctvai",
        user="cctvai_llm_ro",
        password="fake-pass",
        health_ttl=30.0,
        breaker_threshold=3,
        breaker_cooldown=60.0,
    )
    defaults.update(kwargs)
    return CctvaiDatabase(**defaults)


def _probe_row(event_count: int = 8305, latest: str = "2024-07-23 06:00:00+00") -> dict:
    return {"event_count": event_count, "latest_event_at": latest}


def _fake_connect(rows: list[dict], columns: list[str] | None = None):
    """Return a context-manager fake that yields rows for fetchall()."""
    if columns is None and rows:
        columns = list(rows[0].keys())
    elif columns is None:
        columns = []

    row_tuples = [tuple(r[c] for c in columns) for r in rows]

    cursor = MagicMock()
    cursor.__enter__ = lambda s: s
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.description = [(c,) for c in columns]
    cursor.fetchall.return_value = row_tuples
    cursor.fetchone.return_value = row_tuples[0] if row_tuples else None

    conn = MagicMock()
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor

    return conn


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


def test_from_env_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("CCTVAI_DATABASE_ENABLED", raising=False)
    assert CctvaiDatabase.from_env() is None
    monkeypatch.setenv("CCTVAI_DATABASE_ENABLED", "false")
    assert CctvaiDatabase.from_env() is None


def test_from_env_returns_instance_when_enabled(monkeypatch):
    monkeypatch.setenv("CCTVAI_DATABASE_ENABLED", "true")
    monkeypatch.setenv("CCTVAI_DB_HOST", "db.example.com")
    monkeypatch.setenv("CCTVAI_DB_PASSWORD", "secret")
    db = CctvaiDatabase.from_env()
    assert db is not None
    payload = db.status()
    dumped = str(payload)
    for s in ("secret", "db.example.com"):
        assert s not in dumped


# ---------------------------------------------------------------------------
# status() safety
# ---------------------------------------------------------------------------


def test_status_never_exposes_credentials():
    db = _make_db()
    payload = db.status()
    dumped = str(payload)
    for s in _SENSITIVE_STRINGS:
        assert s not in dumped, f"Sensitive string {s!r} found in status()"


def test_status_disabled_state_when_driver_missing():
    with patch("src.integrations.cctvai_database._DRIVER_AVAILABLE", False):
        db2 = _make_db()
        assert db2.status()["state"] == contract.STATE_DRIVER_MISSING
        assert contract.REASON_DRIVER_MISSING in db2.status()["reason_codes"]


def test_status_ready_after_successful_probe():
    db = _make_db()
    db._health.available = True
    db._health.state = contract.STATE_READY
    db._health.event_count = 8305
    db._health.active_camera_count = 15
    db._health.latest_event_at = "2026-09-03"
    db._health.reason_codes = [contract.REASON_REPLICA_LAG_UNVERIFIED]
    payload = db.status()
    assert payload["state"] == contract.STATE_READY
    assert payload["available"] is True
    assert payload["event_count"] == 8305


# ---------------------------------------------------------------------------
# refresh_health circuit breaker
# ---------------------------------------------------------------------------


def test_refresh_health_success_sets_available():
    db = _make_db()
    with (
        patch("src.integrations.cctvai_database._DRIVER_AVAILABLE", True),
        patch.object(db, "_probe_sync", return_value=_probe_row()),
    ):
        asyncio.run(db.refresh_health())
    assert db.available is True
    assert db._health.state == contract.STATE_READY
    assert db._health.event_count == 8305
    assert db._health.consecutive_failures == 0


def test_refresh_health_three_failures_open_breaker():
    db = _make_db()
    with (
        patch("src.integrations.cctvai_database._DRIVER_AVAILABLE", True),
        patch.object(
            db, "_probe_sync", side_effect=Exception("connection refused")
        ),
    ):
        for _ in range(3):
            asyncio.run(db.refresh_health())
    assert db.available is False
    assert db._health.breaker_open_until > time.monotonic()
    assert db._health.state == contract.STATE_UNAVAILABLE


def test_refresh_health_skips_when_breaker_open():
    db = _make_db()
    db._health.breaker_open_until = time.monotonic() + 9999.0
    db._health.available = False

    probe = MagicMock(return_value=_probe_row())
    with (
        patch("src.integrations.cctvai_database._DRIVER_AVAILABLE", True),
        patch.object(db, "_probe_sync", probe),
    ):
        asyncio.run(db.refresh_health())
    probe.assert_not_called()
    assert db.available is False


def test_refresh_health_resets_breaker_on_success():
    db = _make_db()
    db._health.consecutive_failures = 2
    db._health.breaker_open_until = 0.0  # breaker not yet open
    with (
        patch("src.integrations.cctvai_database._DRIVER_AVAILABLE", True),
        patch.object(db, "_probe_sync", return_value=_probe_row()),
    ):
        asyncio.run(db.refresh_health())
    assert db._health.consecutive_failures == 0
    assert db.available is True


def test_refresh_health_flags_fanout_when_count_too_high():
    db = _make_db()
    with (
        patch("src.integrations.cctvai_database._DRIVER_AVAILABLE", True),
        patch.object(
            db, "_probe_sync", return_value=_probe_row(event_count=99999)
        ),
    ):
        asyncio.run(db.refresh_health())
    assert contract.REASON_FANOUT_SUSPECTED in db._health.reason_codes


# ---------------------------------------------------------------------------
# Intent router
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question, expected_intent",
    [
        ("mức độ nghiêm trọng", contract.INTENT_EVENTS_BY_SEVERITY),
        ("loại vi phạm nào nhiều nhất", contract.INTENT_EVENTS_BY_VIOLATION_TYPE),
        ("20 sự kiện gần nhất", contract.INTENT_RECENT_EVENTS),
        ("danh sách camera đang hoạt động", contract.INTENT_ACTIVE_CAMERAS),
        ("sự kiện của camera không có trong master", contract.INTENT_ORPHAN_EVENTS),
        ("sự kiện camera theo tuyến", contract.INTENT_EVENTS_BY_CAMERA_LINE),
        ("tổng quan dữ liệu", contract.INTENT_DATA_OVERVIEW),
        ("thông tin mật khẩu camera", contract.INTENT_CREDENTIALS_SUPPRESSED),
        ("người phụ trách tuyến A", contract.INTENT_RESPONSIBLE_SUPPRESSED),
        ("sự kiện đang diễn ra kéo dài bao lâu", contract.INTENT_OPEN_DURATION_SUPPRESSED),
    ],
)
def test_route_question(question, expected_intent):
    db = _make_db()
    assert db.route_question(question) == expected_intent


def test_route_unknown_question_returns_scope_clarification():
    db = _make_db()
    assert db.route_question("thời tiết hôm nay thế nào") == contract.INTENT_SCOPE_CLARIFICATION


# ---------------------------------------------------------------------------
# SQL template safety checks
# ---------------------------------------------------------------------------

SOFT_DELETE_TEMPLATES = [
    ("_SQL_EVENTS_BY_SEVERITY", _SQL_EVENTS_BY_SEVERITY, ["cctvai_violation_types"]),
    ("_SQL_EVENTS_BY_VIOLATION_TYPE", _SQL_EVENTS_BY_VIOLATION_TYPE, ["cctvai_violation_types"]),
    ("_SQL_RECENT_EVENTS", _SQL_RECENT_EVENTS, ["cctvai_cameras", "cctvai_lines", "cctvai_violation_types"]),
    ("_SQL_ACTIVE_CAMERAS", _SQL_ACTIVE_CAMERAS, ["cctvai_lines"]),
    ("_SQL_EVENTS_BY_CAMERA_LINE", _SQL_EVENTS_BY_CAMERA_LINE, ["cctvai_cameras", "cctvai_lines"]),
]


@pytest.mark.parametrize("name,sql,tables", SOFT_DELETE_TEMPLATES)
def test_template_has_del_flag_in_on_for_each_soft_delete_table(name, sql, tables):
    sql_upper = sql.upper()
    # Every JOIN to a soft-delete table must have AND NOT ...del_flag before the
    # next JOIN or WHERE keyword. Use a simple structural check.
    for table in tables:
        assert table.upper() in sql_upper, f"{name}: {table} not in SQL"
        # Find the ON clause following this table's JOIN
        join_idx = sql_upper.find(table.upper())
        on_idx = sql_upper.find("ON", join_idx)
        next_join_or_where = min(
            (sql_upper.find(kw, on_idx + 1) for kw in ("JOIN", "WHERE", "ORDER", "GROUP", "LIMIT")
             if sql_upper.find(kw, on_idx + 1) > 0),
            default=len(sql_upper),
        )
        on_clause = sql_upper[on_idx:next_join_or_where]
        assert "DEL_FLAG" in on_clause, (
            f"{name}: {table} JOIN is missing del_flag in ON clause. "
            f"ON clause: {on_clause[:200]}"
        )
        assert "NOT" in on_clause, f"{name}: {table} del_flag not negated in ON"


def test_all_templates_use_count_e_id_not_count_star():
    templates = [
        _SQL_EVENTS_BY_SEVERITY, _SQL_EVENTS_BY_VIOLATION_TYPE,
        _SQL_EVENTS_BY_CAMERA_LINE, _SQL_ORPHAN_EVENTS,
    ]
    for sql in templates:
        assert "count(*)" not in sql.lower(), "count(*) found in template"
        assert "count(e.id)" in sql.lower() or "count(e.id)" in sql, (
            f"count(e.id) not found in template:\n{sql[:200]}"
        )


def test_epoch_templates_divide_by_1000():
    epoch_templates = [
        _SQL_EVENTS_BY_SEVERITY,
        _SQL_EVENTS_BY_VIOLATION_TYPE,
        _SQL_RECENT_EVENTS,
        _SQL_EVENTS_BY_CAMERA_LINE,
        _SQL_HEALTH_PROBE,
    ]
    for sql in epoch_templates:
        assert "1000.0" in sql, f"epoch /1000.0 not found in template:\n{sql[:200]}"


def test_all_templates_have_limit():
    no_limit_ok = {_SQL_DATA_OVERVIEW}
    templates = [
        _SQL_EVENTS_BY_SEVERITY,
        _SQL_EVENTS_BY_VIOLATION_TYPE,
        _SQL_RECENT_EVENTS,
        _SQL_ACTIVE_CAMERAS,
        _SQL_ORPHAN_EVENTS,
        _SQL_EVENTS_BY_CAMERA_LINE,
    ]
    for sql in templates:
        if sql in no_limit_ok:
            continue
        assert re.search(r"\bLIMIT\b", sql, re.IGNORECASE), (
            f"LIMIT not found:\n{sql[:200]}"
        )


def test_all_templates_prefix_cctvai():
    all_templates = [
        _SQL_EVENTS_BY_SEVERITY, _SQL_EVENTS_BY_VIOLATION_TYPE,
        _SQL_RECENT_EVENTS, _SQL_ACTIVE_CAMERAS, _SQL_ORPHAN_EVENTS,
        _SQL_EVENTS_BY_CAMERA_LINE, _SQL_DATA_OVERVIEW, _SQL_HEALTH_PROBE,
    ]
    for sql in all_templates:
        for table in contract.REPORTABLE_TABLES:
            if table in sql.lower():
                assert f"cctvai.{table}" in sql.lower(), (
                    f"{table} in template but without cctvai. prefix:\n{sql[:200]}"
                )


def test_template_columns_are_readable():
    """No template may select a blocked camera column."""
    all_templates = [
        _SQL_EVENTS_BY_SEVERITY, _SQL_EVENTS_BY_VIOLATION_TYPE,
        _SQL_RECENT_EVENTS, _SQL_ACTIVE_CAMERAS, _SQL_ORPHAN_EVENTS,
        _SQL_EVENTS_BY_CAMERA_LINE, _SQL_DATA_OVERVIEW,
    ]
    for sql in all_templates:
        for blocked in contract.BLOCKED_CAMERA_COLUMNS:
            # Allow it only in comments or strings; a raw column reference is forbidden
            assert not re.search(rf"\b{blocked}\b", sql), (
                f"Blocked column {blocked!r} found in template:\n{sql[:200]}"
            )


# ---------------------------------------------------------------------------
# query() integration with fake _connect
# ---------------------------------------------------------------------------


def test_query_events_by_severity_returns_result():
    db = _make_db()
    db._health.available = True
    db._health.state = contract.STATE_READY
    rows_data = [{"severity": "alert", "total": 5000}, {"severity": "normal", "total": 3000}]

    with (
        patch("src.integrations.cctvai_database._DRIVER_AVAILABLE", True),
        patch.object(db, "_connect", return_value=_fake_connect(rows_data)),
    ):
        result = asyncio.run(db.query("7 ngày mức độ nghiêm trọng"))

    assert isinstance(result, CctvaiDatabaseResult)
    assert result.intent == contract.INTENT_EVENTS_BY_SEVERITY
    assert len(result.rows) == 2
    assert result.status == "PARTIAL"
    assert contract.REASON_REPLICA_LAG_UNVERIFIED in result.reason_codes
    assert "replica" in result.fallback_answer.lower()


def test_query_suppressed_returns_no_rows():
    db = _make_db()
    db._health.available = True
    with patch("src.integrations.cctvai_database._DRIVER_AVAILABLE", True):
        result = asyncio.run(db.query("mật khẩu camera là gì"))
    assert result.intent == contract.INTENT_CREDENTIALS_SUPPRESSED
    assert result.status == "SUPPRESSED"
    assert result.rows == []
    assert contract.REASON_CAMERA_CREDENTIALS_UNREADABLE in result.reason_codes


def test_query_scope_clarification_returns_partial():
    db = _make_db()
    db._health.available = True
    result = asyncio.run(db.query("thời tiết hôm nay như thế nào"))
    assert result.intent == contract.INTENT_SCOPE_CLARIFICATION
    assert result.status == "PARTIAL"


def test_query_returns_unavailable_when_not_available():
    db = _make_db()
    db._health.available = False
    db._health.state = contract.STATE_UNAVAILABLE
    with patch("src.integrations.cctvai_database._DRIVER_AVAILABLE", True):
        result = asyncio.run(db.query("sự kiện gần nhất"))
    assert result.status == "PARTIAL"
    assert contract.REASON_REPLICA_UNAVAILABLE in result.reason_codes


def test_query_returns_unavailable_when_driver_missing():
    with patch("src.integrations.cctvai_database._DRIVER_AVAILABLE", False):
        db = _make_db()
        result = asyncio.run(db.query("sự kiện gần nhất"))
    assert contract.REASON_DRIVER_MISSING in result.reason_codes


def test_query_returns_error_result_on_exception():
    db = _make_db()
    db._health.available = True
    db._health.state = contract.STATE_READY
    with (
        patch("src.integrations.cctvai_database._DRIVER_AVAILABLE", True),
        patch.object(
            db, "_execute_intent", side_effect=Exception("network timeout")
        ),
    ):
        result = asyncio.run(db.query("sự kiện gần nhất"))
    assert contract.REASON_REPLICA_QUERY_ERROR in result.reason_codes


# ---------------------------------------------------------------------------
# metadata_payload
# ---------------------------------------------------------------------------


def test_metadata_payload_never_exposes_credentials():
    result = CctvaiDatabaseResult(
        intent=contract.INTENT_EVENTS_BY_SEVERITY,
        rows=[],
        fallback_answer="ok",
    )
    payload = result.metadata_payload()
    dumped = str(payload)
    for s in _SENSITIVE_STRINGS:
        assert s not in dumped
    assert "intent" in payload
    assert "status" in payload
    assert "reason_codes" in payload
    assert "source_system" in payload
    assert payload["source_system"] == contract.SOURCE_SYSTEM


def test_metadata_payload_has_contract_versions():
    result = CctvaiDatabaseResult(
        intent=contract.INTENT_DATA_OVERVIEW,
        rows=[],
        fallback_answer="ok",
    )
    payload = result.metadata_payload()
    assert payload["schema_version"] == contract.SCHEMA_VERSION
    assert payload["data_contract_version"] == contract.DATA_CONTRACT_VERSION
    assert payload["semantic_contract_version"] == contract.SEMANTIC_CONTRACT_VERSION


# ---------------------------------------------------------------------------
# freshness note
# ---------------------------------------------------------------------------


def test_with_freshness_appends_replica_note():
    db = _make_db()
    result = db._with_freshness("Có 10 sự kiện.", latest_event_at="2026-09-03", language="vi")
    assert "replica" in result.lower()
    # The disclosure must always deny realtime and flag unverified lag; the
    # exact wording is user-facing copy and may be reworded.
    assert "không phải dữ liệu thời gian thực" in result
    assert "chưa xác minh" in result
    assert "2026-09-03" in result


def test_with_freshness_japanese():
    db = _make_db()
    result = db._with_freshness("10件のイベント。", latest_event_at="2026-09-03", language="ja")
    assert "レプリカ" in result
    assert "2026-09-03" in result


# ---------------------------------------------------------------------------
# _format_rows
# ---------------------------------------------------------------------------


def test_format_rows_empty_returns_no_data_message():
    answer = CctvaiDatabase._format_rows(contract.INTENT_RECENT_EVENTS, [], "vi")
    assert "không" in answer.lower() or "data" in answer.lower() or "dữ liệu" in answer.lower()


def test_format_rows_returns_bullet_list():
    rows = [{"camera_id": "cam001", "total": 500}]
    answer = CctvaiDatabase._format_rows(contract.INTENT_EVENTS_BY_CAMERA_LINE, rows, "vi")
    assert "cam001" in answer
    assert "500" in answer
    assert "-" in answer  # bullet format
