"""Report Agent cho MES: plan tất định → SQL qua guardrail → render Python.

Nguyên tắc chống hallucination: planner KHÔNG dùng LLM. Mỗi báo cáo là một
danh sách bước SQL tất định (theo kỳ báo cáo parse từ câu hỏi), thực thi qua
``MesSqlAgent.execute`` (AST validate, read-only, LIMIT); số liệu đi thẳng từ
SQLite vào renderer Markdown/HTML thuần Python, không qua LLM diễn đạt lại.
Nhận xét (observations) cũng tính bằng code từ chính kết quả truy vấn.
"""

from __future__ import annotations

import html
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator

import asyncio

from src.actions.report_intent import (
    ReportPeriod,
    UnsupportedReportRequest,
    report_capability,
    report_period_for_question,
    report_top_limit,
)
from src.integrations.mes_sql_agent import (
    MesSqlAgent,
    MesSqlAgentError,
    MesSqlQueryResult,
)

# Nhãn tiếng Việt cho cột kết quả SQL khi hiển thị bảng trên UI/HTML.
COLUMN_LABELS = {
    "lot_id": "Mã Lot",
    "product_id": "Mã hàng",
    "total_error_qty": "Tổng lỗi",
    "error_id": "Mã lỗi",
    "error_name": "Tên lỗi",
    "lot_count": "Số Lot",
    "product_count": "Số mã hàng",
    "error_record_count": "Số bản ghi lỗi",
    "error_date": "Ngày",
    "error_month": "Tháng",
}

UNKNOWN_ERROR_NAME = "*Lỗi chưa rõ tên*"


def format_number(value: Any) -> str:
    """Định dạng số kiểu Việt Nam (1.240), giữ nguyên giá trị khác."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}".replace(",", ".")
    if isinstance(value, float):
        return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    if value is None or value == "":
        return "chưa rõ"
    return str(value)


@dataclass(frozen=True)
class ReportStep:
    id: str
    kind: str  # kpi | top_lots | top_errors | top_products | trend
    title: str
    sql: str


@dataclass(frozen=True)
class ReportPlan:
    question: str
    period: ReportPeriod
    limit: int
    steps: tuple[ReportStep, ...] = field(default_factory=tuple)

    @property
    def title(self) -> str:
        return f"Báo cáo lỗi sản xuất MES — {self.period.label}"


def build_report_plan(question: str) -> ReportPlan:
    """Câu hỏi đã hỗ trợ → kế hoạch SQL; từ chối mọi semantics ngoài mẫu."""
    capability = report_capability(question)
    if not capability.supported:
        reason = capability.reason or "Yêu cầu không thuộc mẫu báo cáo đã chuẩn bị."
        raise UnsupportedReportRequest(reason)
    period = report_period_for_question(question)
    limit = report_top_limit(question)
    where = period.where_clause()
    if where:
        trend_sql = f"""
            SELECT date(error_time) AS error_date,
                   SUM(quantity) AS total_error_qty
            FROM v_error_details
            {where}
            GROUP BY date(error_time)
            ORDER BY error_date
            LIMIT 366
        """
        trend_title = "Diễn biến lỗi theo ngày"
    else:
        trend_sql = """
            SELECT strftime('%Y-%m', error_time) AS error_month,
                   SUM(quantity) AS total_error_qty
            FROM v_error_details
            WHERE error_time IS NOT NULL
            GROUP BY strftime('%Y-%m', error_time)
            ORDER BY error_month
            LIMIT 60
        """
        trend_title = "Diễn biến lỗi theo tháng"

    if capability.shape == "top_errors":
        return ReportPlan(
            question=question,
            period=period,
            limit=limit,
            steps=(
                ReportStep(
                    id="kpi",
                    kind="kpi",
                    title="Tổng quan lỗi trong kỳ",
                    sql=f"""
                        SELECT SUM(quantity) AS total_error_qty,
                               COUNT(*) AS error_record_count,
                               COUNT(DISTINCT lot_id) AS lot_count,
                               COUNT(DISTINCT product_id) AS product_count
                        FROM v_error_details
                        {where}
                    """,
                ),
                ReportStep(
                    id="top_errors",
                    kind="top_errors",
                    title=f"Top {limit} loại lỗi phổ biến nhất",
                    sql=f"""
                        SELECT error_id, error_name,
                               SUM(quantity) AS total_error_qty
                        FROM v_error_details
                        {where}
                        GROUP BY error_id, error_name
                        ORDER BY total_error_qty DESC, error_id
                        LIMIT {limit}
                    """,
                ),
            ),
        )

    steps = (
        ReportStep(
            id="kpi",
            kind="kpi",
            title="Tổng quan lỗi trong kỳ",
            sql=f"""
                SELECT SUM(quantity) AS total_error_qty,
                       COUNT(*) AS error_record_count,
                       COUNT(DISTINCT lot_id) AS lot_count,
                       COUNT(DISTINCT product_id) AS product_count
                FROM v_error_details
                {where}
            """,
        ),
        ReportStep(
            id="top_lots",
            kind="top_lots",
            title=f"Top {limit} Lot có tổng lỗi cao nhất",
            sql=f"""
                SELECT lot_id, product_id, SUM(quantity) AS total_error_qty
                FROM v_error_details
                {where}
                GROUP BY lot_id, product_id
                ORDER BY total_error_qty DESC, lot_id
                LIMIT {limit}
            """,
        ),
        ReportStep(
            id="top_errors",
            kind="top_errors",
            title=f"Top {limit} loại lỗi phổ biến nhất",
            sql=f"""
                SELECT error_id, error_name, SUM(quantity) AS total_error_qty
                FROM v_error_details
                {where}
                GROUP BY error_id, error_name
                ORDER BY total_error_qty DESC, error_id
                LIMIT {limit}
            """,
        ),
        ReportStep(
            id="top_products",
            kind="top_products",
            title=f"Top {limit} mã hàng có tổng lỗi cao nhất",
            sql=f"""
                SELECT product_id,
                       SUM(quantity) AS total_error_qty,
                       COUNT(DISTINCT lot_id) AS lot_count
                FROM v_error_details
                {where}
                GROUP BY product_id
                ORDER BY total_error_qty DESC, product_id
                LIMIT {limit}
            """,
        ),
        ReportStep(id="trend", kind="trend", title=trend_title, sql=trend_sql),
    )
    return ReportPlan(question=question, period=period, limit=limit, steps=steps)


def _section_rows(result: MesSqlQueryResult) -> list[dict[str, Any]]:
    rows = []
    for row in result.rows:
        cleaned = dict(row)
        if "error_name" in cleaned and not cleaned.get("error_name"):
            cleaned["error_name"] = UNKNOWN_ERROR_NAME
        rows.append(cleaned)
    return rows


def _kpi_items(result: MesSqlQueryResult | None) -> list[dict[str, Any]]:
    if result is None or not result.rows:
        return []
    row = result.rows[0]
    if row.get("total_error_qty") is None:
        return []
    return [
        {"key": "total_error_qty", "label": "Tổng lỗi", "value": row.get("total_error_qty") or 0},
        {"key": "error_record_count", "label": "Số bản ghi lỗi", "value": row.get("error_record_count") or 0},
        {"key": "lot_count", "label": "Số Lot có lỗi", "value": row.get("lot_count") or 0},
        {"key": "product_count", "label": "Số mã hàng có lỗi", "value": row.get("product_count") or 0},
    ]


def _observations(
    kpis: list[dict[str, Any]],
    sections: dict[str, dict[str, Any]],
) -> list[str]:
    """Nhận xét tính bằng code từ số liệu đã kiểm chứng — không dùng LLM."""
    notes: list[str] = []
    total = next(
        (item["value"] for item in kpis if item["key"] == "total_error_qty"),
        0,
    )
    if not total:
        return notes

    top_lots = sections.get("top_lots", {}).get("rows") or []
    if top_lots:
        first = top_lots[0]
        share = round(100 * (first.get("total_error_qty") or 0) / total, 1)
        notes.append(
            f"Lot {first.get('lot_id') or 'chưa rõ'} (mã hàng "
            f"{first.get('product_id') or 'chưa rõ'}) có tổng lỗi cao nhất: "
            f"{format_number(first.get('total_error_qty'))} lỗi, chiếm "
            f"{format_number(share)}% tổng lỗi trong kỳ."
        )

    top_errors = sections.get("top_errors", {}).get("rows") or []
    if top_errors:
        first = top_errors[0]
        share = round(100 * (first.get("total_error_qty") or 0) / total, 1)
        notes.append(
            f"Loại lỗi phổ biến nhất là {first.get('error_id') or 'chưa rõ'} - "
            f"{first.get('error_name') or UNKNOWN_ERROR_NAME}: "
            f"{format_number(first.get('total_error_qty'))} lỗi, chiếm "
            f"{format_number(share)}% tổng lỗi."
        )

    trend_rows = sections.get("trend", {}).get("rows") or []
    if len(trend_rows) > 1:
        time_key = "error_date" if "error_date" in trend_rows[0] else "error_month"
        peak = max(trend_rows, key=lambda row: row.get("total_error_qty") or 0)
        unit = "Ngày" if time_key == "error_date" else "Tháng"
        notes.append(
            f"{unit} có lỗi cao nhất trong kỳ là {peak.get(time_key)} với "
            f"{format_number(peak.get('total_error_qty'))} lỗi."
        )
    return notes


def _limitations(
    plan: ReportPlan,
    imported_at: str,
    sections: dict[str, dict[str, Any]],
    kpis: list[dict[str, Any]],
) -> list[str]:
    notes: list[str] = []
    snapshot_note = "Số liệu lấy từ MES snapshot, không phải dữ liệu realtime."
    if imported_at:
        snapshot_note = (
            f"Số liệu lấy từ MES snapshot (import lúc {imported_at}), "
            "không phải dữ liệu realtime."
        )
    notes.append(snapshot_note)
    notes.extend(plan.period.notes)
    if not kpis:
        notes.append("Không có dữ liệu lỗi trong kỳ báo cáo.")
    truncated_titles = [
        section["title"]
        for section in sections.values()
        if section.get("truncated")
    ]
    if truncated_titles:
        notes.append(
            "Một số bảng bị cắt bớt theo giới hạn dòng: "
            + ", ".join(truncated_titles)
            + "."
        )
    failed_titles = [
        section["title"] for section in sections.values() if section.get("error")
    ]
    if failed_titles:
        notes.append(
            "Không truy vấn được: " + ", ".join(failed_titles) + "."
        )
    return notes


def render_markdown(report: dict[str, Any]) -> str:
    """Render Markdown chỉ dùng heading + bullet (ReactMarkdown không có GFM table)."""
    lines: list[str] = [f"## {report['title']}", ""]
    if report["kpis"]:
        lines.append("**Tổng quan:**")
        for item in report["kpis"]:
            lines.append(f"- {item['label']}: {format_number(item['value'])}")
        lines.append("")
    for section in report["sections"]:
        rows = section.get("rows") or []
        if not rows:
            continue
        lines.append(f"**{section['title']}:**")
        for row in rows:
            parts = [
                f"{COLUMN_LABELS.get(column, column)} {format_number(row.get(column))}"
                for column in section["columns"]
            ]
            lines.append("- " + ", ".join(parts))
        lines.append("")
    if report["observations"]:
        lines.append("**Nhận xét (tính từ số liệu):**")
        lines.extend(f"- {note}" for note in report["observations"])
        lines.append("")
    if report["limitations"]:
        lines.append("**Giới hạn dữ liệu:**")
        lines.extend(f"- {note}" for note in report["limitations"])
    return "\n".join(lines).strip()


def render_html(report: dict[str, Any]) -> str:
    """Render HTML tự chứa (inline CSS) để tải về; mọi giá trị đều escape."""

    def esc(value: Any) -> str:
        return html.escape(str(value))

    kpi_cells = "".join(
        f"<div class='kpi'><span class='kpi-value'>{esc(format_number(item['value']))}</span>"
        f"<span class='kpi-label'>{esc(item['label'])}</span></div>"
        for item in report["kpis"]
    )
    section_blocks = []
    for section in report["sections"]:
        rows = section.get("rows") or []
        if not rows:
            continue
        head = "".join(
            f"<th>{esc(COLUMN_LABELS.get(column, column))}</th>"
            for column in section["columns"]
        )
        body = "".join(
            "<tr>"
            + "".join(
                f"<td>{esc(format_number(row.get(column)))}</td>"
                for column in section["columns"]
            )
            + "</tr>"
            for row in rows
        )
        section_blocks.append(
            f"<h2>{esc(section['title'])}</h2>"
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        )
    notes_html = ""
    if report["observations"]:
        items = "".join(f"<li>{esc(note)}</li>" for note in report["observations"])
        notes_html += f"<h2>Nhận xét (tính từ số liệu)</h2><ul>{items}</ul>"
    if report["limitations"]:
        items = "".join(f"<li>{esc(note)}</li>" for note in report["limitations"])
        notes_html += f"<h2>Giới hạn dữ liệu</h2><ul>{items}</ul>"

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>{esc(report['title'])}</title>
<style>
body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 32px auto; max-width: 860px;
       color: #23272f; line-height: 1.5; }}
h1 {{ font-size: 1.45rem; border-bottom: 2px solid #2f6f4f; padding-bottom: 8px; }}
h2 {{ font-size: 1.05rem; margin-top: 28px; }}
.meta {{ color: #5f6672; font-size: 0.9rem; }}
.kpis {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0; }}
.kpi {{ border: 1px solid #d7dce2; border-radius: 10px; padding: 12px 18px;
        display: flex; flex-direction: column; min-width: 140px; }}
.kpi-value {{ font-size: 1.3rem; font-weight: 700; color: #2f6f4f; }}
.kpi-label {{ font-size: 0.82rem; color: #5f6672; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 20px; }}
th, td {{ border: 1px solid #d7dce2; padding: 7px 10px; text-align: left;
          font-size: 0.92rem; }}
th {{ background: #f0f3f6; }}
tr:nth-child(even) td {{ background: #fafbfc; }}
ul {{ padding-left: 20px; }}
</style>
</head>
<body>
<h1>{esc(report['title'])}</h1>
<p class="meta">Tạo lúc {esc(report['generated_at'])} · Kỳ báo cáo: {esc(report['period_label'])}</p>
<div class="kpis">{kpi_cells}</div>
{''.join(section_blocks)}
{notes_html}
</body>
</html>"""


def build_summary_text(report: dict[str, Any]) -> str:
    """Đoạn tóm tắt ngắn hiển thị trong khung chat (số liệu từ report)."""
    lines = [f"Đã tạo **{report['title']}**."]
    if report["kpis"]:
        kpi_text = "; ".join(
            f"{item['label'].lower()}: {format_number(item['value'])}"
            for item in report["kpis"]
        )
        lines.append(f"Tổng quan — {kpi_text}.")
    for note in report["observations"][:2]:
        lines.append(f"- {note}")
    if not report["kpis"]:
        lines.append("Không có dữ liệu lỗi trong kỳ báo cáo này.")
    else:
        lines.append(
            "Chi tiết từng bảng số liệu và phần nhận xét nằm trong thẻ báo cáo "
            "bên dưới (có thể tải bản HTML)."
        )
    return "\n\n".join(lines)


class MesReportAgent:
    """Thực thi kế hoạch báo cáo qua ``MesSqlAgent`` và phát tiến trình từng bước."""

    def __init__(self, sql_agent: MesSqlAgent | None):
        self.sql_agent = sql_agent

    @property
    def available(self) -> bool:
        return self.sql_agent is not None and self.sql_agent.available

    async def run(self, question: str) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator: plan → step_start/step_result từng bước → report."""
        plan = build_report_plan(question)
        yield {
            "event": "plan",
            "title": plan.title,
            "period_label": plan.period.label,
            "steps": [{"id": step.id, "title": step.title} for step in plan.steps],
        }

        sections: dict[str, dict[str, Any]] = {}
        kpi_result: MesSqlQueryResult | None = None
        imported_at = ""
        for step in plan.steps:
            yield {"event": "step_start", "step_id": step.id, "title": step.title}
            try:
                result = await asyncio.to_thread(self.sql_agent.execute, step.sql)
            except MesSqlAgentError as exc:
                sections[step.id] = {
                    "id": step.id,
                    "title": step.title,
                    "columns": [],
                    "rows": [],
                    "error": str(exc),
                }
                yield {
                    "event": "step_result",
                    "step_id": step.id,
                    "status": "error",
                    "summary": str(exc),
                }
                continue

            imported_at = imported_at or result.imported_at
            if step.kind == "kpi":
                kpi_result = result
                total = (result.rows[0].get("total_error_qty") if result.rows else None) or 0
                summary = f"Tổng lỗi trong kỳ: {format_number(total)}"
            else:
                sections[step.id] = {
                    "id": step.id,
                    "title": step.title,
                    "columns": result.columns,
                    "rows": _section_rows(result),
                    "truncated": result.truncated,
                }
                summary = f"{len(result.rows)} dòng dữ liệu"
            yield {
                "event": "step_result",
                "step_id": step.id,
                "status": "done" if (step.kind == "kpi" or result.rows) else "empty",
                "summary": summary,
            }

        kpis = _kpi_items(kpi_result)
        observations = _observations(kpis, sections)
        limitations = _limitations(plan, imported_at, sections, kpis)
        ordered_sections = [
            sections[step.id] for step in plan.steps if step.id in sections
        ]
        report = {
            "id": str(uuid.uuid4()),
            "title": plan.title,
            "period_label": plan.period.label,
            "generated_at": datetime.now().strftime("%H:%M %d/%m/%Y"),
            "snapshot_imported_at": imported_at,
            "kpis": kpis,
            "sections": ordered_sections,
            "observations": observations,
            "limitations": limitations,
        }
        report["markdown"] = render_markdown(report)
        yield {
            "event": "report",
            "report": report,
            "summary_text": build_summary_text(report),
        }

    async def build_report(self, question: str) -> tuple[dict[str, Any], str]:
        """Chạy toàn bộ plan không cần stream; trả (report, summary_text)."""
        report: dict[str, Any] | None = None
        summary_text = ""
        async for event in self.run(question):
            if event["event"] == "report":
                report = event["report"]
                summary_text = event["summary_text"]
        assert report is not None
        return report, summary_text
