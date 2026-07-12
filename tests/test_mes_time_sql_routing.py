from src.integrations.mes_query_service import MesQueryService


def test_time_sql_routes_monthly_top_lots_to_error_time():
    sql = MesQueryService.time_sql_for_question(
        "Trong tháng 2025-07, 5 lot nào có tổng lỗi cao nhất?"
    )

    assert "v_error_details" in sql
    assert "produce_date" not in sql
    assert "2025-07-01" in sql
    assert "GROUP BY lot_id, product_id" in sql
    assert "LIMIT 5" in sql


def test_time_sql_routes_top_error_codes_in_highest_month():
    sql = MesQueryService.time_sql_for_question(
        "Trong tháng có tổng lỗi nhiều nhất, 3 mã lỗi phổ biến nhất là gì?"
    )

    assert "top_month" in sql
    assert "strftime('%Y-%m', error_time)" in sql
    assert "GROUP BY t.error_month, e.error_id, e.error_name" in sql
    assert "LIMIT 3" in sql


def test_time_sql_routes_daily_total_error_question():
    sql = MesQueryService.time_sql_for_question(
        "Ngày nào có tổng số lỗi nhiều nhất?"
    )

    assert "date(error_time) AS error_date" in sql
    assert "SUM(quantity) AS total_error_qty" in sql
    assert "LIMIT 1" in sql


def test_time_sql_routes_raw_japanese_monthly_top_lots():
    sql = MesQueryService.time_sql_for_question(
        "2025年7月で総エラー数が多い上位5つのLotを教えてください。"
    )

    assert "v_error_details" in sql
    assert "2025-07-01" in sql
    assert "GROUP BY lot_id, product_id" in sql
    assert "LIMIT 5" in sql


def test_time_sql_routes_inclusive_date_range_top_lots():
    sql = MesQueryService.time_sql_for_question(
        "Từ ngày 2026-06-01 đến 2026-06-15, Top 3 Lot nào có tổng lỗi cao nhất?"
    )

    assert "v_error_details" in sql
    assert "error_time >= '2026-06-01'" in sql
    assert "date('2026-06-15', '+1 day')" in sql
    assert "GROUP BY lot_id, product_id" in sql
    assert "LIMIT 3" in sql


def test_time_sql_orders_reversed_date_range():
    sql = MesQueryService.time_sql_for_question(
        "Từ ngày 2026-06-15 đến 2026-06-01, Top 3 Lot nào có tổng lỗi cao nhất?"
    )

    assert "error_time >= '2026-06-01'" in sql
    assert "date('2026-06-15', '+1 day')" in sql


def test_time_sql_does_not_generate_sql_for_invalid_month():
    assert (
        MesQueryService.time_sql_for_question(
            "Top 3 Lot có lỗi nhiều nhất tháng 13/2026"
        )
        == ""
    )
