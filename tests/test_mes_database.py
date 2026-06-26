import sqlite3
from pathlib import Path

import pytest

from src.integrations.mes_database import MesDatabase


@pytest.fixture
def mes_database(tmp_path: Path) -> MesDatabase:
    db_path = tmp_path / "mes.sqlite"
    schema_path = Path(__file__).parents[1] / "database" / "schema" / "mes.sql"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(schema_path.read_text(encoding="utf-8"))
        connection.executemany(
            """
            INSERT INTO lots (
                lot_pk, source_id, product_id, lot_id, status, pcs_lot, produce_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, 101, "PRODUCT-A", "000001-01-000", "1", 1000, "2026-06-01"),
                (2, 102, "PRODUCT-B", "000002-01-000", "2", 2000, "2026-06-02"),
                (3, 103, "Testlot", "000003-01-000", "2", 3000, "2026-06-03"),
                (4, 104, "m_test_lot", "001208-01-000", "2", 4000, "2026-06-04"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO error_catalog (
                error_catalog_pk, error_id, error_type, process_id,
                error_name, error_name_vi, is_canonical
            ) VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            [
                (1, "E001", "1", "PROC-1", "Short", "Ngắn mạch"),
                (2, "E002", "1", "PROC-2", "Scratch", "Xước"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO error_events (
                error_pk, lot_pk, error_catalog_pk, lot_id, process_id,
                error_type, error_id, quantity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, 1, 1, "000001-01-000", "PROC-1", "1", "E001", 50),
                (2, 1, 2, "000001-01-000", "PROC-2", "1", "E002", 10),
                (3, 2, 1, "000002-01-000", "PROC-1", "1", "E001", 100),
                (4, 3, 1, "000003-01-000", "PROC-1", "1", "E001", 10000),
                (5, 4, 1, "001208-01-000", "PROC-1", "1", "E001", 9000),
            ],
        )
        connection.executemany(
            "INSERT INTO schema_metadata (key, value) VALUES (?, ?)",
            [
                ("imported_at", "2026-06-20T03:52:08+00:00"),
                ("lot_count", "2"),
                ("error_event_count", "3"),
                ("error_catalog_count", "2"),
                ("unmapped_error_name_count", "0"),
            ],
        )
    return MesDatabase(db_path)


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Lot 000001-01-000 đang sản xuất mã hàng nào?", "lot_details"),
        ("Lot 000001-01-000 có những lỗi gì?", "lot_error_breakdown"),
        ("Mã lỗi E001 là lỗi gì?", "error_name"),
        ("Mã hàng PRODUCT-A có tổng bao nhiêu lỗi?", "product_error_summary"),
        ("Mã hàng PRODUCT-A có những lỗi nào?", "product_error_breakdown"),
        ("For product code PRODUCT-A, how many errors are recorded?", "product_error_summary"),
        ("Sản phẩm nào có tổng lỗi cao nhất?", "highest_error_product"),
        ("Những Lot nào có mã lỗi E001?", "lots_for_error"),
        ("Which lots have error code E001?", "lots_for_error"),
        ("Liệt kê các lot", "list_lots"),
        ("Có những lot nào?", "list_lots"),
    ],
)
def test_routes_allowlisted_mes_questions(mes_database, question, intent):
    result = mes_database.query_question(question)

    assert result is not None
    assert result.intent == intent
    assert result.rows
    assert result.imported_at == "2026-06-20T03:52:08+00:00"


def test_list_lots_returns_recent_lots_without_sql_agent(mes_database):
    result = mes_database.query_question("Danh sách các lot hiện có")

    assert result is not None
    assert result.intent == "list_lots"
    assert result.rows[0]["total_lot_count"] == 2
    assert result.rows[0]["items"][0]["lot_id"] == "000002-01-000"
    assert all(
        "test" not in item["product_id"].lower()
        for item in result.rows[0]["items"]
    )
    assert "000002-01-000" in result.fallback_answer
    assert "test" not in result.fallback_answer.lower()
    assert "Testlot" not in result.fallback_answer


def test_highest_lot_requires_explicit_permission(mes_database):
    question = "Theo database, Lot nào có số lượng lỗi nhiều nhất?"

    assert mes_database.query_question(question) is None
    result = mes_database.query_question(question, allow_highest_lot=True)

    assert result is not None
    assert result.intent == "highest_error_lot"
    assert result.rows[0]["lot_id"] == "000002-01-000"
    assert result.rows[0]["total_error_qty"] == 100
    assert "Testlot" not in result.fallback_answer


def test_top_n_highest_error_lots(mes_database):
    result = mes_database.query_question(
        "Liệt kê danh sách 2 lot nhiều lỗi nhất",
        allow_highest_lot=True,
    )

    assert result is not None
    assert result.intent == "highest_error_lot"
    assert [row["lot_id"] for row in result.rows] == [
        "000002-01-000",
        "000001-01-000",
    ]
    assert all("test" not in row["product_id"].lower() for row in result.rows)
    assert "000002-01-000" in result.fallback_answer
    assert "000001-01-000" in result.fallback_answer
    assert "Testlot" not in result.fallback_answer


def test_english_top_n_highest_error_lots_does_not_extract_product_from_production(mes_database):
    result = mes_database.query_question(
        "List the top 2 real production lots by total error quantity and exclude test lots.",
        allow_highest_lot=True,
    )

    assert result is not None
    assert result.intent == "highest_error_lot"
    assert [row["lot_id"] for row in result.rows] == [
        "000002-01-000",
        "000001-01-000",
    ]
    assert "ion" not in result.fallback_answer
    assert "Testlot" not in result.fallback_answer


def test_top_error_codes_does_not_change_highest_lot_limit(mes_database):
    result = mes_database.query_question(
        (
            "For the lot with the highest error quantity, show the product code "
            "and the top three error codes with Vietnamese error names."
        ),
        allow_highest_lot=True,
    )

    assert result is not None
    assert result.intent == "highest_error_lot"
    assert [row["lot_id"] for row in result.rows] == ["000002-01-000"]


def test_unrelated_question_is_not_routed_to_mes(mes_database):
    assert mes_database.query_question("Quy định làm thêm giờ thế nào?") is None


def test_status_reads_snapshot_metadata(mes_database):
    status = mes_database.status()

    assert status["available"] is True
    assert status["lots"] == 2
    assert status["error_events"] == 3
