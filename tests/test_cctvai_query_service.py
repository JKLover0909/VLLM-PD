"""Tests for the CCTVAI SQL-agent fallback wiring in CctvaiQueryService.

Uses the REAL CctvaiSqlAgent (pointed at the real semantic model) so plan
validation is genuinely exercised end-to-end through the query service, not
just at the unit level in test_cctvai_sql_agent.py. Only the LLM calls
(openai_client) and the Postgres connection (CctvaiDatabase.execute_ad_hoc)
are faked — no network, no real replica.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from src.integrations import cctvai_contract as contract
from src.integrations.cctvai_query_service import CctvaiQueryService
from src.integrations.cctvai_sql_agent import CctvaiSqlAgent

SEMANTIC_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "cctvai_semantic_model.json"
)

ACTIVE_CAMERAS_SQL = (
    "SELECT c.camera_id, c.camera_name, l.line_name, c.protocol, c.is_active "
    "FROM cctvai.cctvai_cameras c "
    "LEFT JOIN cctvai.cctvai_lines l ON l.id = c.line_ref_id AND NOT l.del_flag "
    "WHERE NOT c.del_flag ORDER BY l.line_name, c.camera_name LIMIT 15"
)

ACTIVE_CAMERAS_ROWS = [
    {
        "camera_id": "cam001",
        "camera_name": "Cam cổng 1",
        "line_name": "Line A",
        "protocol": "rtsp",
        "is_active": True,
    }
]

DECLINE_PLAN_JSON = json.dumps(
    {"can_answer": False, "reason": "không thể trả lời với dữ liệu hiện có"}
)
UNSAFE_PLAN_JSON = json.dumps(
    {"can_answer": True, "sql": "SELECT * FROM cctvai.cctvai_cameras", "reason": "bad"}
)


def valid_plan_json(sql: str = ACTIVE_CAMERAS_SQL) -> str:
    return json.dumps({"can_answer": True, "sql": sql, "reason": "ok"})


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _ScriptedOpenAIClient:
    """Fake AsyncOpenAI-shaped client returning canned responses in order."""

    def __init__(self, responses: list[str | Exception]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Scripted client ran out of canned responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


@dataclass
class _FakeDbResult:
    intent: str
    fallback_answer: str

    def metadata_payload(self) -> dict:
        return {"intent": self.intent, "status": "PARTIAL"}


class _FakeCctvaiDatabase:
    def __init__(
        self,
        *,
        available: bool = True,
        intent: str = "cctvai_scope_clarification",
        fallback_answer: str = "Xin lỗi, tôi chưa hiểu câu hỏi này.",
        ad_hoc_rows: list[dict] | None = None,
        ad_hoc_error: Exception | None = None,
        latest_event_at: str = "2026-09-04 09:00:00+00",
    ):
        self.available = available
        self._intent = intent
        self._fallback_answer = fallback_answer
        self._ad_hoc_rows = ad_hoc_rows if ad_hoc_rows is not None else []
        self._ad_hoc_error = ad_hoc_error
        self._latest_event_at = latest_event_at
        self.execute_calls: list[str] = []

    async def query(self, question: str, *, language: str = "vi"):
        return _FakeDbResult(intent=self._intent, fallback_answer=self._fallback_answer)

    def execute_ad_hoc(self, sql: str) -> list[dict]:
        self.execute_calls.append(sql)
        if self._ad_hoc_error is not None:
            raise self._ad_hoc_error
        return self._ad_hoc_rows

    def status(self) -> dict:
        return {"latest_event_at": self._latest_event_at}


class _UnavailableSqlAgent:
    available = False


def _real_sql_agent() -> CctvaiSqlAgent:
    return CctvaiSqlAgent(SEMANTIC_MODEL_PATH, max_rows=50)


# ----------------------------------------------------------------------
# Routing scope: only cctvai_scope_clarification triggers the SQL agent
# ----------------------------------------------------------------------


def test_sql_agent_not_invoked_for_matched_deterministic_intent():
    database = _FakeCctvaiDatabase(
        intent="cctvai_active_cameras",
        fallback_answer="Danh sách camera đang hoạt động: ...",
    )
    client = _ScriptedOpenAIClient([])
    service = CctvaiQueryService(
        cctvai_database=database,
        cctvai_sql_agent=_real_sql_agent(),
        openai_client=client,
    )

    outcome = asyncio.run(
        service.query_cctvai_outcome("Camera nào đang hoạt động?", "openai", "vi")
    )

    assert outcome.answer == "Danh sách camera đang hoạt động: ..."
    assert client.calls == []
    assert database.execute_calls == []


def test_sql_agent_skipped_when_agent_unavailable():
    database = _FakeCctvaiDatabase(intent="cctvai_scope_clarification")
    client = _ScriptedOpenAIClient([])
    service = CctvaiQueryService(
        cctvai_database=database,
        cctvai_sql_agent=_UnavailableSqlAgent(),
        openai_client=client,
    )

    outcome = asyncio.run(
        service.query_cctvai_outcome("Câu hỏi lạ ngoài 7 mẫu?", "openai", "vi")
    )

    assert outcome.answer == database._fallback_answer
    assert client.calls == []


def test_sql_agent_skipped_when_database_unavailable():
    database = _FakeCctvaiDatabase(
        available=False, intent="cctvai_scope_clarification"
    )
    client = _ScriptedOpenAIClient([])
    service = CctvaiQueryService(
        cctvai_database=database,
        cctvai_sql_agent=_real_sql_agent(),
        openai_client=client,
    )

    outcome = asyncio.run(
        service.query_cctvai_outcome("Câu hỏi lạ ngoài 7 mẫu?", "openai", "vi")
    )

    assert outcome.answer == database._fallback_answer
    assert client.calls == []


# ----------------------------------------------------------------------
# Success path
# ----------------------------------------------------------------------


def test_sql_agent_answers_when_deterministic_intent_is_scope_clarification():
    database = _FakeCctvaiDatabase(
        intent="cctvai_scope_clarification",
        ad_hoc_rows=ACTIVE_CAMERAS_ROWS,
    )
    client = _ScriptedOpenAIClient(
        [
            valid_plan_json(),
            "Camera cam001 (Cam cổng 1) trên Line A đang hoạt động qua giao thức rtsp.",
        ]
    )
    service = CctvaiQueryService(
        cctvai_database=database,
        cctvai_sql_agent=_real_sql_agent(),
        openai_client=client,
    )

    outcome = asyncio.run(
        service.query_cctvai_outcome(
            "Camera cam001 đang dùng giao thức gì?", "openai", "vi"
        )
    )

    assert "cam001" in outcome.answer
    assert len(client.calls) == 2
    assert len(database.execute_calls) == 1
    assert "cctvai_cameras" in database.execute_calls[0]
    assert outcome.answer_scope == "cctvai_database"
    assert outcome.cctvai_metadata["intent"] == "cctvai_sql_agent"
    assert outcome.cctvai_metadata["status"] == "PARTIAL"
    assert (
        contract.REASON_SQL_AGENT_UNVERIFIED
        in outcome.cctvai_metadata["reason_codes"]
    )
    assert outcome.cctvai_metadata["latest_event_at"] == "2026-09-04 09:00:00+00"


def test_sql_agent_unwraps_json_object_leaked_by_model():
    """Model đôi khi tự bọc câu trả lời trong JSON (vd. {"tra_loi": "..."})
    thay vì trả text thuần theo đúng yêu cầu prompt. Người dùng không được
    thấy JSON thô — service phải giải nén và chỉ hiển thị nội dung thật."""
    database = _FakeCctvaiDatabase(
        intent="cctvai_scope_clarification",
        ad_hoc_rows=ACTIVE_CAMERAS_ROWS,
    )
    leaked_json_answer = (
        '{"tra_loi":"Camera cam001 (Cam cổng 1) trên Line A đang hoạt động '
        'qua giao thức rtsp."}'
    )
    client = _ScriptedOpenAIClient([valid_plan_json(), leaked_json_answer])
    service = CctvaiQueryService(
        cctvai_database=database,
        cctvai_sql_agent=_real_sql_agent(),
        openai_client=client,
    )

    outcome = asyncio.run(
        service.query_cctvai_outcome(
            "Camera cam001 đang dùng giao thức gì?", "openai", "vi"
        )
    )

    assert '{"tra_loi"' not in outcome.answer
    assert "cam001" in outcome.answer
    assert outcome.answer == (
        "Camera cam001 (Cam cổng 1) trên Line A đang hoạt động qua giao "
        "thức rtsp."
    )


def test_sql_agent_previous_error_is_fed_back_to_planner_on_retry():
    database = _FakeCctvaiDatabase(
        intent="cctvai_scope_clarification",
        ad_hoc_rows=ACTIVE_CAMERAS_ROWS,
    )
    client = _ScriptedOpenAIClient(
        [
            UNSAFE_PLAN_JSON,
            valid_plan_json(),
            "Camera cam001 (Cam cổng 1) trên Line A đang hoạt động.",
        ]
    )
    service = CctvaiQueryService(
        cctvai_database=database,
        cctvai_sql_agent=_real_sql_agent(),
        openai_client=client,
    )

    outcome = asyncio.run(
        service.query_cctvai_outcome("Camera nào đang hoạt động?", "openai", "vi")
    )

    assert "cam001" in outcome.answer
    assert len(client.calls) == 3
    # The second planner call must carry the first attempt's rejection reason.
    second_call_messages = client.calls[1]["messages"]
    assert any(
        "'*'" in message["content"] or "Liệt kê cột" in message["content"]
        for message in second_call_messages
    )
    assert len(database.execute_calls) == 1
    assert "cctvai_cameras" in database.execute_calls[0]


# ----------------------------------------------------------------------
# Failure / give-up paths fall back to the deterministic refusal
# ----------------------------------------------------------------------


def test_sql_agent_falls_back_when_planner_declines():
    database = _FakeCctvaiDatabase(intent="cctvai_scope_clarification")
    client = _ScriptedOpenAIClient([DECLINE_PLAN_JSON])
    service = CctvaiQueryService(
        cctvai_database=database,
        cctvai_sql_agent=_real_sql_agent(),
        openai_client=client,
    )

    outcome = asyncio.run(
        service.query_cctvai_outcome("Câu hỏi không thể trả lời?", "openai", "vi")
    )

    assert outcome.answer == database._fallback_answer
    assert len(client.calls) == 1
    assert database.execute_calls == []


def test_sql_agent_gives_up_after_max_attempts_all_rejected(monkeypatch):
    monkeypatch.setenv("CCTVAI_SQL_AGENT_MAX_ATTEMPTS", "2")
    database = _FakeCctvaiDatabase(intent="cctvai_scope_clarification")
    client = _ScriptedOpenAIClient([UNSAFE_PLAN_JSON, UNSAFE_PLAN_JSON])
    service = CctvaiQueryService(
        cctvai_database=database,
        cctvai_sql_agent=_real_sql_agent(),
        openai_client=client,
    )

    outcome = asyncio.run(service.query_cctvai_outcome("Câu hỏi lạ?", "openai", "vi"))

    assert outcome.answer == database._fallback_answer
    assert len(client.calls) == 2
    assert database.execute_calls == []


def test_sql_agent_gives_up_immediately_on_execution_error():
    """A DB execution failure is not retried — a different plan will not fix it."""
    database = _FakeCctvaiDatabase(
        intent="cctvai_scope_clarification",
        ad_hoc_error=RuntimeError("connection reset by peer"),
    )
    client = _ScriptedOpenAIClient([valid_plan_json()])
    service = CctvaiQueryService(
        cctvai_database=database,
        cctvai_sql_agent=_real_sql_agent(),
        openai_client=client,
    )

    outcome = asyncio.run(
        service.query_cctvai_outcome("Camera nào đang hoạt động?", "openai", "vi")
    )

    assert outcome.answer == database._fallback_answer
    assert len(client.calls) == 1
    assert len(database.execute_calls) == 1
    assert "cctvai_cameras" in database.execute_calls[0]


# ----------------------------------------------------------------------
# Answer shaping: empty result and verification-failed candidates
# ----------------------------------------------------------------------


def test_sql_agent_uses_template_for_empty_result_without_second_llm_call():
    database = _FakeCctvaiDatabase(
        intent="cctvai_scope_clarification",
        ad_hoc_rows=[],
    )
    client = _ScriptedOpenAIClient([valid_plan_json()])
    service = CctvaiQueryService(
        cctvai_database=database,
        cctvai_sql_agent=_real_sql_agent(),
        openai_client=client,
    )

    outcome = asyncio.run(
        service.query_cctvai_outcome("Camera nào đang hoạt động?", "openai", "vi")
    )

    assert "không có dữ liệu" in outcome.answer.lower()
    assert len(client.calls) == 1  # planner only — no answer-shaping call needed


def test_sql_agent_falls_back_to_template_when_answer_fails_verification():
    database = _FakeCctvaiDatabase(
        intent="cctvai_scope_clarification",
        ad_hoc_rows=ACTIVE_CAMERAS_ROWS,
    )
    client = _ScriptedOpenAIClient(
        [
            valid_plan_json(),
            "SELECT camera_name FROM cctvai.cctvai_cameras -- leaked SQL",
        ]
    )
    service = CctvaiQueryService(
        cctvai_database=database,
        cctvai_sql_agent=_real_sql_agent(),
        openai_client=client,
    )

    outcome = asyncio.run(
        service.query_cctvai_outcome("Camera nào đang hoạt động?", "openai", "vi")
    )

    assert "select" not in outcome.answer.lower()
    assert "cam001" in outcome.answer
    assert "chưa được kiểm chứng" in outcome.answer
