import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.integrations.mes_query_service import MesQueryService
from src.integrations.mes_wms_database import MesWmsDatabaseResult
from src.integrations.wms_sql_agent import (
    WmsSqlAgent,
    WmsSqlAgentError,
    WmsSqlQueryResult,
)


@pytest.fixture
def wms_sql_agent(tmp_path: Path) -> WmsSqlAgent:
    db_path = tmp_path / "mes_wms.sqlite"
    schema_path = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"
    semantic_path = (
        Path(__file__).parents[1] / "config" / "wms_semantic_model.json"
    )
    with sqlite3.connect(db_path) as connection:
        connection.executescript(schema_path.read_text(encoding="utf-8"))
        connection.executemany(
            """
            INSERT INTO wms_processes (
                process_pk, source_id, process_id, process_name, status
            ) VALUES (?, ?, ?, ?, '1')
            """,
            [
                (1, 1, "PROC-A", "Công đoạn A"),
                (2, 2, "PROC-B", "Công đoạn B"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO wms_current_balances (
                balance_pk, source_id, item_code, quantity_decimal,
                quantity_valid, time_update, process_id, process_pk
            ) VALUES (?, ?, ?, ?, 1, ?, ?, ?)
            """,
            [
                (1, 1, "ITEM-A", "12.5", "2026-07-27 10:00:00", "PROC-A", 1),
                (2, 2, "ITEM-B", "7", "2026-07-27 11:00:00", "PROC-A", 1),
                (3, 3, "ITEM-C", "100", "2026-07-27 12:00:00", "PROC-B", 2),
            ],
        )
        connection.execute(
            """
            INSERT INTO wms_legacy_archive_records (
                source_id, archive_id, archive_date, item_code, item_lot_id,
                process_id, process_pk, quantity_decimal, quantity_valid
            ) VALUES (10, 'ARCH-1', '2025-12-20', 'ITEM-A', 'LOT-A',
                      'PROC-A', 1, '8.5', 1)
            """
        )
        connection.execute(
            "INSERT INTO schema_metadata (key, value) VALUES "
            "('imported_at', '2026-07-28T12:00:00+00:00')"
        )
    return WmsSqlAgent(
        db_path,
        semantic_path,
        max_rows=20,
        timeout_seconds=1,
    )


def test_executes_process_ranking_by_distinct_item_count(wms_sql_agent):
    result = wms_sql_agent.execute(
        """
        SELECT process_id, process_name,
               COUNT(DISTINCT item_code) AS distinct_item_count
        FROM v_wms_current_balance_by_process_item
        GROUP BY process_id, process_name
        ORDER BY distinct_item_count DESC, process_id
        LIMIT 10
        """
    )

    assert result.domain == "CURRENT_BALANCE"
    assert result.rows == [
        {
            "process_id": "PROC-A",
            "process_name": "Công đoạn A",
            "distinct_item_count": 2,
        },
        {
            "process_id": "PROC-B",
            "process_name": "Công đoạn B",
            "distinct_item_count": 1,
        },
    ]
    assert result.imported_at == "2026-07-28T12:00:00+00:00"


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM v_wms_current_balance_by_process_item",
        "SELECT * FROM wms_current_balances",
        "SELECT * FROM v_wms_current_balance_by_process_item; "
        "SELECT * FROM v_wms_current_quality",
        "ATTACH DATABASE '/tmp/other.db' AS other",
        "PRAGMA table_info(v_wms_current_balance_by_process_item)",
    ],
)
def test_rejects_unsafe_or_private_sql(wms_sql_agent, sql):
    with pytest.raises(WmsSqlAgentError):
        wms_sql_agent.validate_sql(sql)


def test_allows_cross_era_read_only_sql(wms_sql_agent):
    safe_sql = wms_sql_agent.validate_sql(
        """
        SELECT current.item_code
        FROM v_wms_current_balance_by_process_item AS current
        JOIN v_wms_legacy_archive_exact_key AS legacy
          ON legacy.item_code = current.item_code
        """
    )

    assert "v_wms_current_balance_by_process_item" in safe_sql
    assert "v_wms_legacy_archive_exact_key" in safe_sql
    assert "LIMIT 20" in safe_sql


def test_adds_default_limit(wms_sql_agent):
    safe_sql = wms_sql_agent.validate_sql(
        "SELECT process_id FROM v_wms_current_balance_by_process_item"
    )

    assert "LIMIT 20" in safe_sql


def test_semantic_model_matches_python_view_allowlist(wms_sql_agent):
    assert set(wms_sql_agent.semantic_model()["views"]) == WmsSqlAgent.ALLOWED_VIEWS


def test_fallback_answer_hides_nulls_and_localizes_known_fields(wms_sql_agent):
    result = WmsSqlQueryResult(
        columns=["process_id", "process_name", "item_count"],
        rows=[
            {
                "process_id": "PROC-A",
                "process_name": None,
                "item_count": 2,
            }
        ],
    )

    answer = wms_sql_agent.fallback_answer(result, language="vi")

    assert "Mã công đoạn: PROC-A" in answer
    assert "Số mã vật tư: 2" in answer
    assert "process_id" not in answer
    assert "process_name" not in answer
    assert "item_count" not in answer
    assert "None" not in answer


def test_fallback_answer_warns_about_unverified_cross_item_quantity(wms_sql_agent):
    result = WmsSqlQueryResult(
        columns=["total_quantity"],
        rows=[{"total_quantity": 125.5}],
    )

    answer = wms_sql_agent.fallback_answer(result, language="vi")

    assert "Tổng thô chưa chuẩn hóa UOM: 125.5" in answer
    assert "không phải tổng nghiệp vụ đã kiểm chứng" in answer


class _ClarifyingWmsDatabase:
    def query_question(self, question, *, language="vi", assume_wms=False):
        del question, assume_wms
        return MesWmsDatabaseResult(
            intent="wms_scope_clarification",
            rows=[],
            imported_at="2026-07-28T12:00:00+00:00",
            source_as_of="2026-07-27 23:49:17",
            fallback_answer=(
                "処理工程または品目コードを指定してください。"
                if language == "ja"
                else "Vui lòng nêu mã công đoạn hoặc mã vật tư cần xem."
            ),
            status="PARTIAL",
            domain="CURRENT_BALANCE",
        )

    def sql_agent_result(self, rows, answer, *, domain, evidence_domains=None):
        del evidence_domains
        return MesWmsDatabaseResult(
            intent="wms_sql_agent_answer",
            rows=rows,
            imported_at="2026-07-28T12:00:00+00:00",
            source_as_of="2026-07-27 23:49:17",
            fallback_answer=answer,
            status="PARTIAL",
            reason_codes=("SQL_AGENT_ANSWER_UNVERIFIED",),
            domain=domain,
            grain="process_id,item_code",
        )


class _FakeWmsSqlAgent:
    available = True

    def planner_messages(self, question, previous_error=""):
        return [{"role": "user", "content": question + previous_error}]

    def parse_plan(self, content):
        assert "SELECT" in content
        return SimpleNamespace(can_answer=True, sql="SELECT safe", reason="")

    def execute(self, sql):
        assert sql == "SELECT safe"
        return WmsSqlQueryResult(
            columns=["process_id", "distinct_item_count"],
            rows=[{"process_id": "PROC-A", "distinct_item_count": 2}],
            imported_at="2026-07-28T12:00:00+00:00",
            domain="CURRENT_BALANCE",
        )

    def answer_messages(self, question, result, *, language="vi"):
        del question, result, language
        return [{"role": "user", "content": "answer"}]

    @staticmethod
    def answer_is_natural(answer):
        return bool(answer)

    @staticmethod
    def answer_matches_result(answer, result):
        return "PROC-A" in answer and "2" in answer and bool(result.rows)

    def fallback_answer(self, result, *, language="vi"):
        del result, language
        return "fallback"


class _QueuedCompletions:
    def __init__(self):
        self.contents = [
            '{"can_answer":true,"sql":"SELECT safe","reason":"ok"}',
            "Theo WMS snapshot, công đoạn PROC-A quản lý nhiều nhất với 2 mã vật tư.",
        ]

    async def create(self, **kwargs):
        del kwargs
        content = self.contents.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _OpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_QueuedCompletions())


def test_scope_clarification_falls_back_to_wms_sql_agent(monkeypatch):
    monkeypatch.setenv("WMS_SQL_AGENT_MODEL", "openai-chat-fallback")
    service = MesQueryService(
        mes_client=SimpleNamespace(),
        mes_database=SimpleNamespace(),
        mes_sql_agent=SimpleNamespace(available=False),
        mes_wms_database=_ClarifyingWmsDatabase(),
        wms_sql_agent=_FakeWmsSqlAgent(),
        openai_client=_OpenAIClient(),
    )

    outcome = asyncio.run(
        service.query_wms_outcome(
            "Công đoạn nào có nhiều mã vật tư đang được quản lý nhất?",
            model="openai",
            language="vi",
        )
    )

    assert "PROC-A" in outcome.answer
    assert "2 mã vật tư" in outcome.answer
    assert outcome.answer_scope == "wms_database"
    assert outcome.wms_metadata["intent"] == "wms_sql_agent_answer"
    assert outcome.wms_metadata["status"] == "PARTIAL"
    assert outcome.wms_metadata["reason_codes"] == [
        "SQL_AGENT_ANSWER_UNVERIFIED"
    ]
