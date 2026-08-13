import asyncio
import sqlite3
from pathlib import Path

import pytest

from src.actions.report_agent import (
    HrExecutiveReportAgent,
    MesReportAgent,
    MesWmsReportAgent,
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
    ("question", "status", "shape", "domain"),
    [
        ("Lập báo cáo tổng hợp lỗi MES", "supported", "overview", "mes"),
        ("Lập báo cáo về 7 lỗi nhiều nhất", "supported", "top_errors", "mes"),
        ("Ban giám đốc cần tổng quan lỗi MES", "supported", "overview", "mes"),
        ("役員向けにMESエラーの全体像を出してください", "supported", "overview", "mes"),
        ("Ban giám đốc cần tổng quan nhân sự", "supported", "hr_executive", "hr"),
        ("社長向けに人事組織の概要を出してください", "supported", "hr_executive", "hr"),
        ("Ban giám đốc cần tình hình chung tồn kho WMS", "supported", "wms_executive", "wms"),
        ("役員向けにWMS工程在庫の概要を出してください", "supported", "wms_executive", "wms"),
        ("Lot nào có tổng lỗi cao nhất?", "not_report", "", ""),
        ("Ban giám đốc gồm những ai?", "not_report", "", ""),
        ("Tình hình Lot A thế nào?", "not_report", "", ""),
        (
            "Lập báo cáo tổng quan tồn kho công đoạn WMS",
            "supported",
            "wms_executive",
            "wms",
        ),
        ("工程在庫レポートを作成してください。", "supported", "wms_executive", "wms"),
        ("Lập báo cáo nhân sự quý 2", "unsupported", "", "hr"),
        ("Lập báo cáo sản lượng sản xuất theo ca", "unsupported", "", "mes"),
        ("Ban giám đốc cần tổng quan tình hình sản xuất", "unsupported", "", "mes"),
        ("Tạo báo cáo chất lượng riêng cho mã hàng PRODUCT-B", "unsupported", "", "mes"),
        ("Chỉ tạo báo cáo xu hướng tổng lỗi theo tháng", "unsupported", "", "mes"),
        ("Tạo báo cáo các Lot có trên 100 lỗi", "unsupported", "", "mes"),
        ("Tạo báo cáo so sánh lỗi tháng 5/2026 và tháng 6/2026", "unsupported", "", "mes"),
        ("Generate a production report by shift", "unsupported", "", "mes"),
        ("品番別の生産量レポートを作成してください。", "unsupported", "", "mes"),
    ],
)
def test_report_capability_fails_closed(question, status, shape, domain):
    capability = report_capability(question)

    assert capability.status == status
    assert capability.shape == shape
    assert capability.domain == domain


@pytest.mark.parametrize(
    ("question", "domain", "shape"),
    [
        # Động từ giao việc liền danh từ.
        ("Cho xem báo cáo WMS", "wms", "wms_executive"),
        ("Xin báo cáo WMS", "wms", "wms_executive"),
        ("Cần báo cáo tồn kho WMS ngay", "wms", "wms_executive"),
        ("Gửi báo cáo lỗi MES", "mes", "overview"),
        # Động từ cách danh từ vài từ.
        ("Cho anh coi báo cáo WMS", "wms", "wms_executive"),
        ("Trình Chủ tịch báo cáo nhân sự", "hr", "hr_executive"),
        ("Xuất file báo cáo tồn kho WMS", "wms", "wms_executive"),
        # Danh từ đứng đầu kiểu khẩu lệnh.
        ("Báo cáo headcount", "hr", "hr_executive"),
        ("Báo cáo tồn kho WMS", "wms", "wms_executive"),
        ("Báo cáo WMS đâu", "wms", "wms_executive"),
        ("Báo cáo tồn kho và tồn công đoạn", "wms", "wms_executive"),
        # Khẩu lệnh tổng quan nói tắt, không có chữ "báo cáo".
        ("Tình hình tồn kho WMS hiện tại", "wms", "wms_executive"),
        ("Tổng quan nhân sự", "hr", "hr_executive"),
        ("Tình hình lỗi MES", "mes", "overview"),
        # Slash/bang command.
        ("/report wms", "wms", "wms_executive"),
        ("!report mes", "mes", "overview"),
        ("/baocao hr", "hr", "hr_executive"),
        # Khẩu lệnh ngắn tiếng Nhật và tiếng Anh.
        ("WMS在庫レポートを出力してください", "wms", "wms_executive"),
        ("人事レポートを見せて", "hr", "hr_executive"),
        ("MESエラー報告書が必要", "mes", "overview"),
        ("Show report WMS inventory", "wms", "wms_executive"),
        ("Export report MES errors", "mes", "overview"),
    ],
)
def test_short_management_commands_open_report(question, domain, shape):
    assert is_report_request(question) is True
    capability = report_capability(question)

    assert capability.status == "supported"
    assert capability.domain == domain
    assert capability.shape == shape


@pytest.mark.parametrize(
    "question",
    [
        # Chữ "báo cáo" ở vai trò mô tả hoặc hỏi thủ tục.
        "Lỗi này đã được báo cáo chưa?",
        "Quy trình báo cáo lỗi là gì?",
        "Mẫu báo cáo nào đang dùng?",
        "Ai báo cáo sự cố này?",
        # Khẩu lệnh nói tắt nhưng thu hẹp về một đối tượng cụ thể.
        "Tình hình Lot 000866-05-000",
        "Tình hình nhân sự phòng Kế toán",
        "Tình hình Lot A thế nào?",
        # Q&A tra cứu thường.
        "Tổng số bản ghi lỗi trong hệ thống là bao nhiêu?",
        "Trong Lot lỗi nhiều nhất, loại nào phổ biến nhất?",
        "Dữ liệu MES snapshot được lấy từ đâu?",
        "Mã hàng 3736-0008 có tổng bao nhiêu lỗi?",
    ],
)
def test_short_command_matcher_keeps_normal_qa(question):
    assert is_report_request(question) is False
    assert report_capability(question).status == "not_report"


def test_report_without_domain_fails_closed_without_defaulting_to_mes():
    capability = report_capability("Tạo báo cáo")

    assert capability.status == "unsupported"
    assert capability.domain == ""
    assert "lĩnh vực" in capability.reason


@pytest.mark.parametrize(
    "question",
    [
        "Lập báo cáo chi phí lương theo phòng ban",
        "Tạo báo cáo KPI cá nhân từng nhân viên",
        "Xuất báo cáo tuyển dụng và nghỉ việc nhân sự",
        "Báo cáo chấm công và tăng ca nhân sự",
        "Lập báo cáo danh sách nhân viên toàn công ty",
        "給与レポートを作成してください",
        "人事評価レポートを出力",
    ],
)
def test_hr_report_rejects_out_of_scope_concepts(question):
    capability = report_capability(question)

    assert capability.status == "unsupported"
    assert capability.domain == "hr"
    assert capability.shape == ""
    assert "aggregate" in capability.reason


@pytest.mark.parametrize(
    "question",
    [
        "Lập báo cáo tuổi hàng tồn kho WMS",
        "Tạo báo cáo xuất nhập tồn WMS",
        "Xuất báo cáo giá trị tồn kho WMS",
        "Lập báo cáo vòng quay tồn kho WMS",
        "在庫年齢レポートを作成",
        "受払レポートを出力",
    ],
)
def test_wms_report_rejects_out_of_scope_concepts(question):
    capability = report_capability(question)

    assert capability.status == "unsupported"
    assert capability.domain == "wms"
    assert capability.shape == ""
    assert "WMS contract v4" in capability.reason


@pytest.mark.parametrize(
    "question",
    [
        "Lập báo cáo tổng tồn kho WMS",
        "Lập báo cáo xu hướng tồn kho WMS tháng 7/2026",
        "Lập báo cáo WMS dưới tồn tối thiểu",
        "Lập báo cáo WIP và bottleneck kho công đoạn WMS",
        "Tạo báo cáo WMS riêng cho công đoạn PROC-A",
        "WMS在庫推移レポートを作成してください。",
        "WMS最低在庫レポートを作成してください。",
    ],
)
def test_wms_report_capability_rejects_unverified_semantics(question):
    capability = report_capability(question)

    assert capability.status == "unsupported"
    assert capability.shape == ""
    assert "WMS contract v4" in capability.reason


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

    assert len(plan.steps) == 6
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
    assert report["charts"][0]["svg"].startswith("<svg")
    assert report["matrices"][0]["row_label"] == "Mã hàng"
    assert report["matrices"][0]["columns"] == ["E1", "E2"]
    assert "Ma trận" in render_html(report)


def test_report_matrix_groups_same_error_id_across_names(sql_agent):
    with sqlite3.connect(sql_agent.db_path) as connection:
        connection.execute(
            """
            INSERT INTO error_catalog (
                error_catalog_pk, error_id, error_type, process_id,
                error_name_vi, is_canonical
            ) VALUES (?, ?, ?, ?, ?, 1)
            """,
            (3, "E1", "1", "P3", "Tên E1 biến thể"),
        )
        connection.execute(
            """
            INSERT INTO error_events (
                error_pk, lot_pk, error_catalog_pk, lot_id, process_id,
                error_type, error_id, quantity, error_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (5, 1, 3, "LOT-A", "P3", "1", "E1", 7, "2026-06-04 08:00:00"),
        )

    report, _ = asyncio.run(
        MesReportAgent(sql_agent).build_report(
            "Lập báo cáo top 5 lỗi sản xuất tháng 6/2026"
        )
    )
    matrix = report["matrices"][0]
    product_a = next(row for row in matrix["rows"] if row["label"] == "PRODUCT-A")

    assert matrix["columns"] == ["E1", "E2"]
    assert product_a["values"] == [12, None]


class FakeEmployeeDirectory:
    @staticmethod
    def count():
        return 10

    @staticmethod
    def department_summaries():
        return [
            {"department": "設計 <A&B>", "size": 6},
            {"department": "Sản xuất", "size": 4},
        ]


def test_hr_executive_report_is_aggregate_only_and_bilingual():
    agent = HrExecutiveReportAgent(FakeEmployeeDirectory())

    report_vi, _ = asyncio.run(
        agent.build_report("Ban giám đốc cần tổng quan nhân sự", language="vi")
    )
    report_ja, summary_ja = asyncio.run(
        agent.build_report("社長向けに人事組織の概要を出してください", language="ja")
    )

    assert report_vi["report_type"] == "hr_executive_report"
    assert [item["value"] for item in report_vi["kpis"]] == [10, 2, 6]
    assert report_vi["matrices"][0]["heatmap"] is False
    assert set(report_vi["sections"][0]["columns"]) == {"department", "size"}
    assert all("name" not in row for row in report_vi["sections"][0]["rows"])
    assert "設計 &lt;A&amp;B&gt;" in report_vi["html_content"]
    assert "個人一覧" in report_ja["governance"][0]
    assert report_ja["charts"][0]["title"] == "部門別従業員数"
    assert report_ja["matrices"][0]["columns"] == ["従業員数", "構成比"]
    assert "チャート" in summary_ja


class FakeCompatibleWmsDatabase:
    available = True

    @staticmethod
    def compatibility():
        return {"compatible": True}

    @staticmethod
    def get_executive_matrix_data(limit_processes):
        assert limit_processes == 8
        return {
            "quality": {
                "source_as_of": "2026-07-27 23:49:17",
                "distinct_item_count": 3,
                "distinct_process_code_count": 2,
                "current_row_count": 3,
                "mapped_process_row_count": 2,
            },
            "processes": [
                {
                    "process_id": "PROC-A",
                    "process_name": "Công đoạn <A&B>",
                    "process_mapped": 1,
                    "distinct_item_count": 2,
                }
            ],
            "items": ["ITEM-<A&1>"],
            "matrix": {"PROC-A": {"ITEM-<A&1>": "12.5"}},
        }


class FakeIncompatibleWmsDatabase(FakeCompatibleWmsDatabase):
    @staticmethod
    def compatibility():
        return {"compatible": False}


def test_wms_report_agent_requires_compatible_snapshot():
    assert MesWmsReportAgent(FakeCompatibleWmsDatabase()).available is True
    assert MesWmsReportAgent(FakeIncompatibleWmsDatabase()).available is False


def test_wms_report_agent_builds_safe_bilingual_artifacts():
    agent = MesWmsReportAgent(FakeCompatibleWmsDatabase())

    report_vi, summary_vi = asyncio.run(
        agent.generate_report("Lập báo cáo tổng quan WMS", language="vi")
    )
    report_ja, summary_ja = asyncio.run(
        agent.generate_report("WMS在庫レポートを作成", language="ja")
    )

    assert report_vi["report_type"] == "wms_executive_report"
    assert [item["key"] for item in report_vi["kpis"]] == [
        "distinct_item_count",
        "distinct_process_count",
        "process_mapping_coverage",
    ]
    assert report_vi["limitations"]
    assert '<html lang="vi">' in report_vi["html_content"]
    assert "Công đoạn &lt;A&amp;B&gt;" in report_vi["html_content"]
    assert "ITEM-&lt;A&amp;1&gt;" in report_vi["html_content"]
    assert "không tính tổng xuyên vật tư" in report_vi["observations"][1]
    assert "tải xuống" in summary_vi

    assert '<html lang="ja">' in report_ja["html_content"]
    assert "WMS工程倉庫" in report_ja["title"]
    assert "異なる資材間の合計は計算していません" in report_ja["observations"][1]
    assert "ダウンロード" in summary_ja


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
    assert len([item for item in events if item["event"] == "step_start"]) == 6
    assert len([item for item in events if item["event"] == "step_result"]) == 6
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


class TruncatedMatrixSqlAgent:
    available = True

    def __init__(self):
        self.calls = 0

    def execute(self, sql):
        self.calls += 1
        if self.calls == 1:
            return MesSqlQueryResult(
                columns=["total_error_qty", "error_record_count", "lot_count", "product_count"],
                rows=[{"total_error_qty": 10, "error_record_count": 1, "lot_count": 1, "product_count": 1}],
                imported_at="2026-06-20",
                truncated=False,
            )
        if self.calls == 6:
            return MesSqlQueryResult(
                columns=["product_id", "error_label", "total_error_qty"],
                rows=[{"product_id": "PRODUCT-A", "error_label": "E1", "total_error_qty": 10}],
                imported_at="2026-06-20",
                truncated=True,
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

    assert len(results) == 6
    assert results[1]["step_id"] == "top_lots"
    assert results[1]["status"] == "error"
    assert any("Top 5 Lot" in note for note in report["limitations"])
    assert events[-1]["event"] == "report"


def test_report_agent_warns_when_matrix_is_truncated():
    report, _ = asyncio.run(
        MesReportAgent(TruncatedMatrixSqlAgent()).build_report(
            "Lập báo cáo lỗi sản xuất"
        )
    )

    assert "Ma trận điều hành bị cắt bớt theo giới hạn dòng truy vấn." in report["limitations"]
