import asyncio
import sqlite3
from pathlib import Path

import pytest

from src.actions.report_agent import (
    MesReportAgent,
    build_report_plan,
    format_number,
    render_html,
)
from src.actions.report_intent import (
    UnsupportedReportRequest,
    is_mes_report_request,
    is_report_request,
    report_capability,
    report_period_for_question,
    report_top_limit,
)
from src.integrations.mes_sql_agent import (
    MesSqlAgent,
    MesSqlAgentError,
    MesSqlQueryResult,
)


@pytest.fixture
def sql_agent(tmp_path: Path) -> MesSqlAgent:
    db_path = tmp_path / "mes.sqlite"
    schema_path = Path(__file__).parents[1] / "database" / "schema" / "mes.sql"
    semantic_path = Path(__file__).parents[1] / "config" / "mes_semantic_model.json"
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
    return MesSqlAgent(db_path, semantic_path, max_rows=50, timeout_seconds=1)


def test_report_intent_requires_report_and_mes_context():
    assert is_mes_report_request("Lập báo cáo lỗi sản xuất tháng 6/2026") is True
    assert is_mes_report_request("Tạo báo cáo top 5 Lot có lỗi cao nhất") is True
    assert is_mes_report_request("Mã Lot nào có số lỗi nhiều nhất?") is False
    assert is_mes_report_request("Lập báo cáo nhân sự") is False
    assert is_report_request("Lập báo cáo nhân sự") is True


@pytest.mark.parametrize(
    ("question", "status", "shape"),
    [
        ("Lập báo cáo tổng hợp lỗi MES", "supported", "overview"),
        ("Lập báo cáo về 7 lỗi nhiều nhất", "supported", "top_errors"),
        ("Lot nào có tổng lỗi cao nhất?", "not_report", ""),
        ("Lập báo cáo nhân sự quý 2", "unsupported", ""),
        ("Lập báo cáo sản lượng sản xuất theo ca", "unsupported", ""),
        ("Tạo báo cáo chất lượng riêng cho mã hàng PRODUCT-B", "unsupported", ""),
        ("Chỉ tạo báo cáo xu hướng tổng lỗi theo tháng", "unsupported", ""),
        ("Tạo báo cáo các Lot có trên 100 lỗi", "unsupported", ""),
        ("Tạo báo cáo so sánh lỗi tháng 5/2026 và tháng 6/2026", "unsupported", ""),
        ("Generate a production report by shift", "unsupported", ""),
        ("品番別の生産量レポートを作成してください。", "unsupported", ""),
    ],
)
def test_report_capability_fails_closed(question, status, shape):
    capability = report_capability(question)

    assert capability.status == status
    assert capability.shape == shape


@pytest.mark.parametrize(
    "question",
    [
        "Tạo báo cáo so sánh lỗi tháng 5/2026 và tháng 6/2026",
        "Tạo báo cáo chất lượng riêng cho mã hàng PRODUCT-B",
        "Chỉ tạo báo cáo xu hướng tổng lỗi theo tháng, không cần top Lot",
        "Tạo báo cáo các Lot có trên 100 lỗi; với mỗi Lot lấy top 2 lỗi",
        "Lập báo cáo sản lượng sản xuất theo ca",
        "Lập báo cáo lỗi ngày 2026-02-30",
        "Lập báo cáo lỗi tháng 13/2026",
        "Lập báo cáo lỗi 2026-13",
        "2026年13月の生産エラーレポートを作成してください。",
    ],
)
def test_report_plan_rejects_unsupported_requests(question):
    with pytest.raises(UnsupportedReportRequest):
        build_report_plan(question)


def test_report_period_parses_month_day_and_range():
    month = report_period_for_question("Lập báo cáo lỗi tháng 6/2026")
    assert month.kind == "month"
    assert month.start == "2026-06-01"
    assert "+1 month" in month.end_exclusive_sql

    day = report_period_for_question("Tạo báo cáo lỗi ngày 2026-06-02")
    assert day.kind == "day"
    assert day.start == "2026-06-02"

    date_range = report_period_for_question(
        "Lập báo cáo lỗi từ 2026-06-01 đến 2026-06-15"
    )
    assert date_range.kind == "range"
    assert date_range.start == "2026-06-01"
    assert "2026-06-15" in date_range.end_exclusive_sql


def test_report_plan_uses_error_time_and_guarded_views():
    plan = build_report_plan("Lập báo cáo top 3 lỗi tháng 6/2026")

    assert len(plan.steps) == 5
    assert plan.limit == 3
    for step in plan.steps:
        assert "v_error_details" in step.sql
        assert "2026-06-01" in step.sql
        assert "INSERT" not in step.sql.upper()
        assert "UPDATE" not in step.sql.upper()


def test_report_agent_generates_verified_report_and_excludes_test_data(sql_agent):
    agent = MesReportAgent(sql_agent)
    report, summary = asyncio.run(
        agent.build_report("Lập báo cáo top 5 lỗi sản xuất tháng 6/2026")
    )

    kpis = {item["key"]: item["value"] for item in report["kpis"]}
    assert kpis["total_error_qty"] == 35
    assert kpis["lot_count"] == 2
    assert kpis["product_count"] == 2
    assert "999" not in report["markdown"]
    assert "LOT-B" in report["markdown"]
    assert "35" in summary
    assert report["snapshot_imported_at"] == "2026-06-20"
    assert report["observations"]


def test_report_agent_streams_plan_steps_and_artifact(sql_agent):
    async def collect():
        events = []
        async for event in MesReportAgent(sql_agent).run(
            "Lập báo cáo lỗi sản xuất tháng 6/2026"
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())
    assert events[0]["event"] == "plan"
    assert len([item for item in events if item["event"] == "step_start"]) == 5
    assert len([item for item in events if item["event"] == "step_result"]) == 5
    assert events[-1]["event"] == "report"


def test_report_html_escapes_data(sql_agent):
    report, _ = asyncio.run(
        MesReportAgent(sql_agent).build_report(
            "Lập báo cáo lỗi sản xuất tháng 6/2026"
        )
    )
    report["title"] = "Báo cáo <script>alert(1)</script>"
    report["sections"][0]["rows"][0]["lot_id"] = "<img src=x onerror=alert(1)>"
    report["observations"] = ["<script>alert(2)</script>"]
    report["limitations"] = ["<b onclick=alert(3)>unsafe</b>"]

    html = render_html(report)

    assert "<script>alert(1)</script>" not in html
    assert "<script>alert(2)</script>" not in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "<b onclick=alert(3)>unsafe</b>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Generate production error report for 2026-06", True),
        ("2026年6月の生産エラーレポートを作成してください。", True),
        ("Lập báo cáo nhân sự", False),
        ("Lập báo cáo công tác nhân sự", False),
        ("Lập báo cáo marketing quý 2", False),
        ("Mã Lot nào có số lỗi nhiều nhất?", False),
        ("", False),
    ],
)
def test_report_intent_multilingual_and_negative_cases(question, expected):
    assert is_mes_report_request(question) is expected


def test_report_period_defaults_and_reverses_explicit_range():
    whole_snapshot = report_period_for_question("Lập báo cáo tổng hợp lỗi MES")
    assert whole_snapshot.kind == "all"
    assert whole_snapshot.where_clause() == ""

    missing_year = report_period_for_question("Lập báo cáo lỗi tháng 6")
    assert missing_year.kind == "all"
    assert "không rõ năm" in missing_year.notes[0]

    reversed_range = report_period_for_question(
        "Lập báo cáo lỗi từ 2026-06-15 đến 2026-06-01"
    )
    assert reversed_range.start == "2026-06-01"
    assert "2026-06-15" in reversed_range.end_exclusive_sql

    cross_month_range = report_period_for_question(
        "Lập báo cáo lỗi từ 2026-05-31 đến 2026-06-01"
    )
    assert cross_month_range.kind == "range"
    assert cross_month_range.start == "2026-05-31"
    assert "2026-06-01" in cross_month_range.end_exclusive_sql


def test_report_capability_rejects_invalid_explicit_months():
    for question in (
        "Lập báo cáo lỗi tháng 13/2026",
        "Lập báo cáo lỗi tháng 0/2026",
        "Lập báo cáo lỗi 2026-13",
        "2026年13月の生産エラーレポートを作成してください。",
    ):
        capability = report_capability(question)
        assert capability.status == "unsupported"
        assert "Tháng" in capability.reason


def test_report_top_limit_defaults_and_caps_at_twenty():
    assert report_top_limit("Lập báo cáo lỗi MES") == 5
    assert report_top_limit("Lập báo cáo top 3 lỗi MES") == 3
    assert report_top_limit("Lập báo cáo về 7 lỗi nhiều nhất") == 7
    assert report_top_limit("Lập báo cáo 12 loại lỗi phổ biến nhất") == 12
    assert report_top_limit("Lập báo cáo top 100 Lot lỗi MES") == 20


def test_ranked_errors_report_uses_focused_two_step_plan():
    plan = build_report_plan("Lập báo cáo về 7 lỗi nhiều nhất")

    assert plan.limit == 7
    assert [step.id for step in plan.steps] == ["kpi", "top_errors"]
    assert "LIMIT 7" in plan.steps[1].sql
    assert "GROUP BY error_id, error_name" in plan.steps[1].sql


def test_report_plan_changes_trend_granularity_by_period():
    all_plan = build_report_plan("Lập báo cáo tổng hợp lỗi MES")
    month_plan = build_report_plan("Lập báo cáo lỗi tháng 6/2026")

    all_trend = next(step for step in all_plan.steps if step.id == "trend")
    month_trend = next(step for step in month_plan.steps if step.id == "trend")
    assert "strftime('%Y-%m', error_time)" in all_trend.sql
    assert "LIMIT 60" in all_trend.sql
    assert "date(error_time)" in month_trend.sql
    assert "LIMIT 366" in month_trend.sql


def test_report_agent_generates_expected_sections_and_observations(sql_agent):
    report, _ = asyncio.run(
        MesReportAgent(sql_agent).build_report(
            "Lập báo cáo top 5 lỗi sản xuất tháng 6/2026"
        )
    )
    sections = {section["id"]: section for section in report["sections"]}

    assert sections["top_lots"]["rows"] == [
        {"lot_id": "LOT-B", "product_id": "PRODUCT-B", "total_error_qty": 30},
        {"lot_id": "LOT-A", "product_id": "PRODUCT-A", "total_error_qty": 5},
    ]
    assert sections["top_errors"]["rows"] == [
        {"error_id": "E1", "error_name": "Lỗi một", "total_error_qty": 25},
        {"error_id": "E2", "error_name": "Lỗi hai", "total_error_qty": 10},
    ]
    assert sections["top_products"]["rows"] == [
        {"product_id": "PRODUCT-B", "total_error_qty": 30, "lot_count": 1},
        {"product_id": "PRODUCT-A", "total_error_qty": 5, "lot_count": 1},
    ]
    assert sections["trend"]["rows"] == [
        {"error_date": "2026-06-01", "total_error_qty": 5},
        {"error_date": "2026-06-02", "total_error_qty": 30},
    ]
    assert any("LOT-B" in note and "85,70%" in note for note in report["observations"])
    assert any("E1" in note and "71,40%" in note for note in report["observations"])
    assert any("2026-06-02" in note for note in report["observations"])


def test_report_agent_handles_empty_period(sql_agent):
    report, summary = asyncio.run(
        MesReportAgent(sql_agent).build_report(
            "Lập báo cáo lỗi sản xuất tháng 7/2026"
        )
    )

    assert report["kpis"] == []
    assert all(not section["rows"] for section in report["sections"])
    assert "Không có dữ liệu lỗi trong kỳ báo cáo" in report["limitations"][-1]
    assert "Không có dữ liệu lỗi" in summary


def test_format_number_covers_report_value_types():
    assert format_number(1234) == "1.234"
    assert format_number(1234.0) == "1.234"
    assert format_number(71.428) == "71,43"
    assert format_number(None) == "chưa rõ"
    assert format_number("") == "chưa rõ"
    assert format_number(True) == "True"


class ScriptedSqlAgent:
    available = True

    def __init__(self):
        self.calls = 0

    def execute(self, sql):
        self.calls += 1
        if self.calls == 2:
            raise MesSqlAgentError("Top Lot query failed")
        if self.calls == 1:
            return MesSqlQueryResult(
                columns=["total_error_qty", "error_record_count", "lot_count", "product_count"],
                rows=[{"total_error_qty": 10, "error_record_count": 1, "lot_count": 1, "product_count": 1}],
                imported_at="2026-06-20",
                truncated=False,
            )
        return MesSqlQueryResult(
            columns=[], rows=[], imported_at="2026-06-20", truncated=False
        )


def test_report_agent_continues_after_guarded_sql_step_failure():
    async def collect():
        events = []
        async for event in MesReportAgent(ScriptedSqlAgent()).run(
            "Lập báo cáo lỗi sản xuất"
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())
    results = [event for event in events if event["event"] == "step_result"]
    report = events[-1]["report"]

    assert len(results) == 5
    assert results[1]["step_id"] == "top_lots"
    assert results[1]["status"] == "error"
    assert any("Top 5 Lot" in note for note in report["limitations"])
    assert events[-1]["event"] == "report"
