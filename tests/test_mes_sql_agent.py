import sqlite3
from pathlib import Path

import pytest

from src.integrations.mes_sql_agent import (
    MesSqlAgent,
    MesSqlAgentError,
    MesSqlQueryResult,
)


@pytest.fixture
def sql_agent(tmp_path: Path) -> MesSqlAgent:
    db_path = tmp_path / "mes.sqlite"
    schema_path = Path(__file__).parents[1] / "database" / "schema" / "mes.sql"
    semantic_path = (
        Path(__file__).parents[1] / "config" / "mes_semantic_model.json"
    )
    with sqlite3.connect(db_path) as connection:
        connection.executescript(schema_path.read_text(encoding="utf-8"))
        connection.executemany(
            """
            INSERT INTO lots (
                lot_pk, source_id, product_id, lot_id, status, pcs_lot
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (1, 1, "PRODUCT-A", "LOT-A", "1", 100),
                (2, 2, "PRODUCT-B", "LOT-B", "1", 200),
                (3, 3, "Testlot", "LOT-TEST", "1", 300),
            ],
        )
        connection.executemany(
            """
            INSERT INTO error_catalog (
                error_catalog_pk, error_id, error_type, process_id,
                error_name_vi, is_canonical
            ) VALUES (?, ?, ?, ?, ?, 1)
            """,
            [
                (1, "E1", "1", "P1", "Lỗi một"),
                (2, "E2", "1", "P2", "Lỗi hai"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO error_events (
                error_pk, lot_pk, error_catalog_pk, lot_id, process_id,
                error_type, error_id, quantity, error_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, 1, 1, "LOT-A", "P1", "1", "E1", 5, "2026-06-01 08:00:00"),
                (2, 2, 1, "LOT-B", "P1", "1", "E1", 20, "2026-06-02 09:00:00"),
                (3, 2, 2, "LOT-B", "P2", "1", "E2", 10, "2026-06-02 10:00:00"),
                (4, 3, 1, "LOT-TEST", "P1", "1", "E1", 999, "2026-06-03 11:00:00"),
            ],
        )
        connection.execute(
            "INSERT INTO schema_metadata (key, value) VALUES ('imported_at', '2026-06-20')"
        )
    return MesSqlAgent(db_path, semantic_path, max_rows=20, timeout_seconds=1)


def test_parse_json_plan_with_markdown_fence():
    plan = MesSqlAgent.parse_plan(
        '```json\n{"can_answer":true,"sql":"SELECT lot_id FROM '
        'v_lot_error_summary","reason":"ok"}\n```'
    )

    assert plan.can_answer is True
    assert plan.sql.startswith("SELECT")


def test_executes_compound_top_lot_top_errors_query(sql_agent):
    result = sql_agent.execute(
        """
        WITH top_lot AS (
            SELECT lot_id
            FROM v_lot_error_summary
            ORDER BY total_error_qty DESC
            LIMIT 1
        )
        SELECT b.lot_id, b.error_id, b.error_name,
               SUM(b.total_error_qty) AS error_quantity
        FROM v_lot_error_breakdown AS b
        JOIN top_lot AS t ON t.lot_id = b.lot_id
        GROUP BY b.lot_id, b.error_id, b.error_name
        ORDER BY error_quantity DESC
        LIMIT 3
        """
    )

    assert [row["error_id"] for row in result.rows] == ["E1", "E2"]
    assert result.rows[0]["lot_id"] == "LOT-B"
    assert result.rows[0]["error_quantity"] == 20
    assert result.imported_at == "2026-06-20"


def test_sql_agent_views_hide_test_lots(sql_agent):
    result = sql_agent.execute(
        """
        SELECT lot_id, product_id, total_error_qty
        FROM v_lot_error_summary
        ORDER BY total_error_qty DESC
        """
    )

    assert [row["lot_id"] for row in result.rows] == ["LOT-B", "LOT-A"]
    assert all("test" not in row["product_id"].lower() for row in result.rows)


def test_executes_daily_error_aggregation_without_test_lots(sql_agent):
    result = sql_agent.execute(
        """
        SELECT date(error_time) AS error_date,
               SUM(quantity) AS total_error_qty,
               COUNT(*) AS error_record_count
        FROM v_error_details
        WHERE error_time IS NOT NULL
        GROUP BY date(error_time)
        ORDER BY total_error_qty DESC
        LIMIT 5
        """
    )

    assert result.rows[0] == {
        "error_date": "2026-06-02",
        "total_error_qty": 30,
        "error_record_count": 2,
    }
    assert [row["error_date"] for row in result.rows] == [
        "2026-06-02",
        "2026-06-01",
    ]


def test_executes_monthly_error_aggregation(sql_agent):
    result = sql_agent.execute(
        """
        SELECT strftime('%Y-%m', error_time) AS error_month,
               SUM(quantity) AS total_error_qty
        FROM v_error_details
        WHERE error_time >= '2026-06-01'
          AND error_time < date('2026-06-01', '+1 month')
        GROUP BY strftime('%Y-%m', error_time)
        ORDER BY total_error_qty DESC
        """
    )

    assert result.rows == [{"error_month": "2026-06", "total_error_qty": 35}]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM v_error_details",
        "SELECT * FROM lots",
        "SELECT * FROM v_error_details; SELECT * FROM v_lot_error_summary",
        "ATTACH DATABASE '/tmp/other.db' AS other",
        "PRAGMA table_info(v_error_details)",
    ],
)
def test_rejects_unsafe_or_private_sql(sql_agent, sql):
    with pytest.raises(MesSqlAgentError):
        sql_agent.validate_sql(sql)


def test_adds_default_limit(sql_agent):
    safe_sql = sql_agent.validate_sql(
        "SELECT lot_id, total_error_qty FROM v_lot_error_summary"
    )

    assert "LIMIT 20" in safe_sql


def test_fallback_answer_formats_top_error_breakdown():
    result = MesSqlQueryResult(
        columns=["lot_id", "product_id", "error_id", "error_name", "error_quantity"],
        rows=[
            {
                "lot_id": "LOT-B",
                "product_id": "PRODUCT-B",
                "error_id": "E1",
                "error_name": "Lỗi một",
                "error_quantity": 31240,
            },
            {
                "lot_id": "LOT-B",
                "product_id": "PRODUCT-B",
                "error_id": "E2",
                "error_name": None,
                "error_quantity": 7800,
            },
        ],
        imported_at="2026-06-20",
        truncated=False,
    )

    answer = MesSqlAgent.fallback_answer(result)

    assert "Lot LOT-B" in answer
    assert "mã hàng PRODUCT-B" in answer
    assert "E1 - Lỗi một: 31.240" in answer
    assert "E2 - *Lỗi chưa rõ tên*: 7.800" in answer


def test_fallback_answer_formats_time_aggregation():
    result = MesSqlQueryResult(
        columns=["error_date", "total_error_qty"],
        rows=[
            {"error_date": "2026-06-02", "total_error_qty": 30000},
            {"error_date": "2026-06-01", "total_error_qty": 5000},
        ],
        imported_at="2026-06-20",
        truncated=False,
    )

    answer = MesSqlAgent.fallback_answer(result)

    assert "lỗi theo thời gian" in answer
    assert "2026-06-02: 30.000" in answer
    assert "2026-06-01: 5.000" in answer
