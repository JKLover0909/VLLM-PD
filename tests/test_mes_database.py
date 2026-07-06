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
                ("lot_count", "4"),
                ("error_event_count", "5"),
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


def test_ambiguous_lot_count_asks_for_scope_instead_of_guessing(mes_database):
    result = mes_database.query_question("Có bao nhiêu lot?")

    assert result is not None
    assert result.intent == "ambiguous_lot_count"
    assert "chưa rõ" in result.fallback_answer.lower()


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


def test_lot_error_record_count_is_not_routed_to_breakdown(mes_database):
    result = mes_database.query_question(
        "Lot 000001-01-000 có bao nhiêu bản ghi lỗi?"
    )

    assert result is not None
    assert result.intent == "lot_error_record_count"
    assert result.rows[0]["error_record_count"] == 2
    assert "2 bản ghi lỗi" in result.fallback_answer


def test_product_lot_count_and_average_are_deterministic(mes_database):
    lot_count = mes_database.query_question(
        "Sản phẩm PRODUCT-A có bao nhiêu lot bị lỗi?"
    )
    average = mes_database.query_question(
        "Trung bình mỗi lot của sản phẩm PRODUCT-A có bao nhiêu lỗi?"
    )

    assert lot_count is not None
    assert lot_count.intent == "product_lot_count"
    assert lot_count.rows[0]["lot_count"] == 1
    assert average is not None
    assert average.intent == "product_average_errors_per_lot"
    assert average.rows[0]["average_error_qty_per_lot"] == 60


def test_error_name_questions_route_by_vietnamese_name(mes_database):
    name = mes_database.query_question(
        'Lỗi "Ngắn mạch" thuộc loại lỗi gì, xảy ra ở process nào?'
    )
    quantity = mes_database.query_question(
        'Số lượng quantity lỗi "Ngắn mạch" trong bản ghi liên quan là bao nhiêu?'
    )
    lots = mes_database.query_question('Lỗi "Ngắn mạch" xuất hiện ở lot nào?')

    assert name is not None
    assert name.intent == "error_name_search"
    assert name.rows[0]["process_id"] == "PROC-1"
    assert quantity is not None
    assert quantity.intent == "error_quantity_by_name"
    assert quantity.rows[0]["top_record"]["quantity"] == 100
    assert lots is not None
    assert lots.intent == "lots_for_error_name"
    assert lots.rows[0]["lot_id"] == "000002-01-000"


def test_lowest_lot_and_typo_question_are_deterministic(mes_database):
    lowest = mes_database.query_question("Lot nào có ít lỗi nhất trong hệ thống?")
    typo = mes_database.query_question(
        "lot nào lỗi nhìu nhất",
        allow_highest_lot=True,
    )

    assert lowest is not None
    assert lowest.intent == "lowest_error_lot"
    assert lowest.rows[0]["lot_id"] == "000001-01-000"
    assert typo is not None
    assert typo.intent == "highest_error_lot"
    assert typo.rows[0]["lot_id"] == "000002-01-000"


def test_unsupported_mes_scope_is_rejected_before_sql_agent(mes_database):
    for question in (
        "Lot 000001-01-000 do công nhân nào sản xuất?",
        "Chi phí sửa lỗi của sản phẩm PRODUCT-A là bao nhiêu?",
        "Dự đoán tháng sau sản phẩm nào sẽ lỗi nhiều nhất?",
        "Sản phẩm nào chưa từng bị lỗi?",
    ):
        result = mes_database.query_question(question)
        assert result is not None
        assert result.intent == "unsupported_mes_scope"


def test_number_only_error_record_count(mes_database):
    result = mes_database.query_question(
        "Chỉ trả lời bằng 1 số duy nhất: tổng số bản ghi lỗi trong hệ thống"
    )

    assert result is not None
    assert result.intent == "count_error_records"
    assert result.fallback_answer == "3"


def test_unrelated_question_is_not_routed_to_mes(mes_database):
    assert mes_database.query_question("Quy định làm thêm giờ thế nào?") is None


def test_status_reads_snapshot_metadata(mes_database):
    status = mes_database.status()

    assert status["available"] is True
    assert status["lots"] == 2
    assert status["raw_lots"] == 4
    assert status["excluded_test_lots"] == 2
    assert status["error_events"] == 3
    assert status["raw_error_events"] == 5
    assert status["excluded_test_error_events"] == 2
