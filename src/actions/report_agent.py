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


def render_bar_chart_svg(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    value_key: str,
    title: str,
    accent: str = "#2f6f4f",
) -> str:
    """SVG bar chart thuần từ số liệu đã kiểm chứng; không phụ thuộc chart lib.

    Mọi nhãn/giá trị đều escape. Trả về chuỗi rỗng khi không có số liệu dương
    để renderer không chèn khối biểu đồ trống vào báo cáo.
    """
    points = [
        (str(row.get(label_key) or "chưa rõ"), float(row.get(value_key) or 0))
        for row in rows
        if (row.get(value_key) or 0) > 0
    ]
    if not points:
        return ""

    label_width = 150
    value_width = 90
    bar_area = 320
    bar_height = 18
    bar_gap = 9
    top = 26
    width = label_width + bar_area + value_width
    height = top + len(points) * (bar_height + bar_gap)
    max_value = max(value for _, value in points) or 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" width="100%" height="auto">',
        f"<title>{html.escape(title)}</title>",
        f'<text x="0" y="14" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="12" font-weight="600" fill="#23272f">{html.escape(title)}</text>',
    ]
    y = top
    for label, value in points:
        bar = max(2, int(bar_area * value / max_value))
        shown = label if len(label) <= 22 else f"{label[:21]}…"
        parts.append(
            f'<text x="0" y="{y + bar_height - 5}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11" fill="#5f6672">{html.escape(shown)}</text>'
        )
        parts.append(
            f'<rect x="{label_width}" y="{y}" width="{bar}" height="{bar_height}" '
            f'rx="3" fill="{accent}"><title>{html.escape(label)}: '
            f'{html.escape(format_number(value))}</title></rect>'
        )
        parts.append(
            f'<text x="{label_width + bar + 6}" y="{y + bar_height - 5}" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="11" font-weight="600" '
            f'fill="#23272f">{html.escape(format_number(value))}</text>'
        )
        y += bar_height + bar_gap
    parts.append("</svg>")
    return "".join(parts)


def build_error_matrix(
    rows: list[dict[str, Any]],
    *,
    row_key: str,
    column_key: str,
    value_key: str,
    row_label: str,
) -> dict[str, Any] | None:
    """Ma trận 2 chiều từ một truy vấn đã group sẵn; không tự suy diễn ô trống.

    Ô không có bản ghi giữ ``None`` (hiển thị "—") để phân biệt với giá trị 0
    thật sự có trong dữ liệu.
    """
    if not rows:
        return None
    columns: list[str] = []
    row_labels: list[str] = []
    cells: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_value = str(row.get(row_key) or "chưa rõ")
        column_value = str(row.get(column_key) or "chưa rõ")
        if row_value not in cells:
            cells[row_value] = {}
            row_labels.append(row_value)
        if column_value not in columns:
            columns.append(column_value)
        cells[row_value][column_value] = row.get(value_key)
    values = [
        value
        for mapping in cells.values()
        for value in mapping.values()
        if isinstance(value, (int, float))
    ]
    return {
        "row_label": row_label,
        "columns": columns,
        "rows": [
            {
                "label": label,
                "values": [cells[label].get(column) for column in columns],
            }
            for label in row_labels
        ],
        "max_value": max(values) if values else 0,
    }


@dataclass(frozen=True)
class ReportStep:
    id: str
    kind: str  # kpi | top_lots | top_errors | top_products | trend | matrix
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
        # Ma trận mã hàng × loại lỗi giới hạn theo cùng ``limit`` để bảng luôn
        # đọc được trong báo cáo điều hành; ô thiếu bản ghi giữ NULL.
        ReportStep(
            id="error_matrix",
            kind="matrix",
            title=f"Ma trận Top {limit} mã hàng × loại lỗi",
            sql=f"""
                WITH top_products AS (
                    SELECT product_id
                    FROM v_error_details
                    {where}
                    GROUP BY product_id
                    ORDER BY SUM(quantity) DESC, product_id
                    LIMIT {limit}
                ),
                top_errors AS (
                    SELECT error_id
                    FROM v_error_details
                    {where}
                    GROUP BY error_id
                    ORDER BY SUM(quantity) DESC, error_id
                    LIMIT {limit}
                )
                SELECT d.product_id,
                       d.error_id,
                       d.error_id AS error_label,
                       SUM(d.quantity) AS total_error_qty
                FROM v_error_details d
                JOIN top_products p ON p.product_id = d.product_id
                JOIN top_errors e ON e.error_id = d.error_id
                {where}
                GROUP BY d.product_id, d.error_id
                ORDER BY d.product_id, total_error_qty DESC
            """,
        ),
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


def _kpi_items(
    result: MesSqlQueryResult | None,
    *,
    language: str = "vi",
) -> list[dict[str, Any]]:
    if result is None or not result.rows:
        return []
    row = result.rows[0]
    if row.get("total_error_qty") is None:
        return []
    labels = (
        ("総エラー数", "エラー記録数", "エラー発生ロット数", "エラー発生品番数")
        if language == "ja"
        else ("Tổng lỗi", "Số bản ghi lỗi", "Số Lot có lỗi", "Số mã hàng có lỗi")
    )
    return [
        {"key": "total_error_qty", "label": labels[0], "value": row.get("total_error_qty") or 0},
        {"key": "error_record_count", "label": labels[1], "value": row.get("error_record_count") or 0},
        {"key": "lot_count", "label": labels[2], "value": row.get("lot_count") or 0},
        {"key": "product_count", "label": labels[3], "value": row.get("product_count") or 0},
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

    language = str(report.get("language") or "vi")
    japanese = language == "ja"
    title_text = str(report.get("title") or "Báo cáo Điều hành")
    as_of_text = str(
        report.get("snapshot_imported_at")
        or report.get("period_label")
        or ("未確認" if japanese else "chưa xác nhận")
    )
    governance = [str(item) for item in report.get("governance", [])]
    limitations = [str(item) for item in report.get("limitations", [])]
    observations = [str(item) for item in report.get("observations", [])]

    kpi_cells = "".join(
        f"<div class='kpi'><span class='kpi-value'>{esc(format_number(item['value']))}</span>"
        f"<span class='kpi-label'>{esc(item['label'])}</span></div>"
        for item in report.get("kpis", [])
    )
    chart_blocks = "".join(
        f"<section class='chart'><h2>{esc(chart.get('title', 'Biểu đồ'))}</h2>"
        f"{chart.get('svg', '')}</section>"
        for chart in report.get("charts", [])
        if chart.get("svg")
    )
    matrix_blocks = []
    for matrix in report.get("matrices", []) or []:
        headers = "".join(f"<th>{esc(column)}</th>" for column in matrix.get("columns", []))
        max_value = matrix.get("max_value") or 0
        use_heat = matrix.get("heatmap") is not False
        body_rows = []
        for row in matrix.get("rows", []):
            cells = []
            for value in row.get("values", []):
                if use_heat and isinstance(value, (int, float)) and max_value:
                    alpha = min(0.85, max(0.12, 0.12 + 0.73 * value / max_value))
                    style = f" style='background: rgba(47, 111, 79, {alpha:.2f}); color: #102018; font-weight: 700;'"
                else:
                    style = ""
                cells.append(f"<td{style}>{esc(format_number(value))}</td>")
            body_rows.append(
                f"<tr><th scope='row'>{esc(row.get('label'))}</th>{''.join(cells)}</tr>"
            )
        matrix_blocks.append(
            f"<section class='matrix'><h2>{esc(matrix.get('title', 'Ma trận'))}</h2>"
            f"<div class='scroll'><table><thead><tr><th>{esc(matrix.get('row_label', ''))}</th>{headers}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table></div></section>"
        )
    section_blocks = []
    for section in report.get("sections", []):
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
    if observations:
        heading = "数値からの所見" if japanese else "Nhận xét (tính từ số liệu)"
        items = "".join(f"<li>{esc(note)}</li>" for note in observations)
        notes_html += f"<h2>{esc(heading)}</h2><ul>{items}</ul>"
    if governance:
        heading = "ガバナンス上の注意" if japanese else "Quy tắc Governance & Data Contract"
        items = "".join(f"<li>{esc(note)}</li>" for note in governance)
        notes_html += f"<div class='governance-box'><b>{esc(heading)}:</b><ul>{items}</ul></div>"
    if limitations:
        heading = "データの制限" if japanese else "Giới hạn dữ liệu"
        items = "".join(f"<li>{esc(note)}</li>" for note in limitations)
        notes_html += f"<h2>{esc(heading)}</h2><ul>{items}</ul>"

    generated_label = "作成日時" if japanese else "Tạo lúc"
    period_label = "対象期間" if japanese else "Kỳ báo cáo"
    print_label = "印刷 / PDF保存" if japanese else "In / Tải PDF"
    return f"""<!DOCTYPE html>
<html lang="{'ja' if japanese else 'vi'}">
<head>
<meta charset="utf-8">
<title>{esc(title_text)}</title>
<style>
@media print {{ @page {{ size: A4 landscape; margin: 12mm; }} body {{ background: #fff; padding: 0; }} .no-print {{ display: none; }} }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px auto; max-width: 1040px; color: #1e293b; line-height: 1.5; padding: 0 16px; }}
.header {{ border-bottom: 2px solid #2f6f4f; padding-bottom: 10px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: flex-end; }}
.header h1 {{ font-size: 1.45rem; color: #0f172a; margin: 0 0 6px 0; }}
.meta {{ font-size: 0.85rem; color: #64748b; margin: 0; }}
.print-btn {{ background: #2f6f4f; color: #fff; border: none; border-radius: 6px; padding: 6px 12px; font-size: 0.85rem; cursor: pointer; }}
.kpis {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0; }}
.kpi {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 16px; background: #f8fafc; min-width: 140px; display: flex; flex-direction: column; }}
.kpi-value {{ font-size: 1.3rem; font-weight: 700; color: #2f6f4f; }}
.kpi-label {{ font-size: 0.8rem; color: #64748b; }}
.chart, .matrix {{ border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin: 18px 0; background: #fff; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 20px; font-size: 0.88rem; }}
th, td {{ border: 1px solid #cbd5e1; padding: 7px 10px; text-align: left; }}
th {{ background: #f1f5f9; color: #334155; }}
tr:nth-child(even) td {{ background: #fafbfc; }}
.governance-box {{ background: #fff7ed; border: 1px solid #ffedd5; border-radius: 8px; padding: 12px 16px; font-size: 0.85rem; color: #9a3412; margin-top: 24px; }}
ul {{ padding-left: 20px; }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>{esc(title_text)}</h1>
    <p class="meta">{esc(generated_label)}: {esc(report.get('generated_at') or '')} · {esc(period_label)}: {esc(as_of_text)}</p>
  </div>
  <button class="print-btn no-print" onclick="window.print()">{esc(print_label)}</button>
</div>
<div class="kpis">{kpi_cells}</div>
{chart_blocks}
{''.join(matrix_blocks)}
{''.join(section_blocks)}
{notes_html}
</body>
</html>"""


def build_summary_text(report: dict[str, Any]) -> str:
    """Đoạn tóm tắt ngắn hiển thị trong khung chat (số liệu từ report)."""
    japanese = report.get("language") == "ja"
    lines = [
        f"**{report['title']}**を作成しました。"
        if japanese
        else f"Đã tạo **{report['title']}**."
    ]
    if report["kpis"]:
        kpi_text = "; ".join(
            f"{item['label']}: {format_number(item['value'])}"
            for item in report["kpis"]
        )
        lines.append(("概要" if japanese else "Tổng quan") + f" — {kpi_text}.")
    for note in report["observations"][:2]:
        lines.append(f"- {note}")
    if not report["kpis"]:
        lines.append(
            "対象期間にエラーデータはありません。"
            if japanese
            else "Không có dữ liệu lỗi trong kỳ báo cáo này."
        )
    else:
        lines.append(
            "各表と所見は下のレポートカードに表示され、HTML版をダウンロードできます。"
            if japanese
            else (
                "Chi tiết từng bảng số liệu và phần nhận xét nằm trong thẻ báo cáo "
                "bên dưới (có thể tải bản HTML)."
            )
        )
    return "\n\n".join(lines)


class MesReportAgent:
    """Thực thi kế hoạch báo cáo qua ``MesSqlAgent`` và phát tiến trình từng bước."""

    def __init__(self, sql_agent: MesSqlAgent | None):
        self.sql_agent = sql_agent

    @property
    def available(self) -> bool:
        return self.sql_agent is not None and self.sql_agent.available

    async def run(
        self, question: str, language: str = "vi"
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator: plan → step_start/step_result từng bước → report."""
        plan = build_report_plan(question)
        yield {
            "event": "plan",
            "title": plan.title,
            "period_label": plan.period.label,
            "steps": [{"id": step.id, "title": step.title} for step in plan.steps],
        }

        sections: dict[str, dict[str, Any]] = {}
        matrices: list[dict[str, Any]] = []
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
                rows = _section_rows(result)
                if step.kind == "matrix":
                    matrix = build_error_matrix(
                        rows,
                        row_key="product_id",
                        column_key="error_label",
                        value_key="total_error_qty",
                        row_label="Mã hàng",
                    )
                    if matrix:
                        matrix["id"] = step.id
                        matrix["title"] = step.title
                        matrix["truncated"] = result.truncated
                        matrices.append(matrix)
                else:
                    sections[step.id] = {
                        "id": step.id,
                        "title": step.title,
                        "columns": result.columns,
                        "rows": rows,
                        "truncated": result.truncated,
                    }
                summary = f"{len(result.rows)} dòng dữ liệu"
            yield {
                "event": "step_result",
                "step_id": step.id,
                "status": "done" if (step.kind == "kpi" or result.rows) else "empty",
                "summary": summary,
            }

        kpis = _kpi_items(kpi_result, language=language)
        observations = _observations(kpis, sections)
        limitations = _limitations(plan, imported_at, sections, kpis)
        if any(matrix.get("truncated") for matrix in matrices):
            limitations.append(
                "Ma trận điều hành bị cắt bớt theo giới hạn dòng truy vấn."
            )
        ordered_sections = [
            sections[step.id] for step in plan.steps if step.id in sections
        ]
        charts = []
        top_errors = sections.get("top_errors", {}).get("rows") or []
        if top_errors:
            chart_rows = [
                {
                    **row,
                    "error_label": f"{row.get('error_id') or 'chưa rõ'} - "
                    f"{row.get('error_name') or UNKNOWN_ERROR_NAME}",
                }
                for row in top_errors[: plan.limit]
            ]
            svg = render_bar_chart_svg(
                chart_rows,
                label_key="error_label",
                value_key="total_error_qty",
                title="Top loại lỗi theo số lượng",
            )
            if svg:
                charts.append({
                    "id": "top_errors_chart",
                    "title": "Top loại lỗi",
                    "label_key": "error_label",
                    "value_key": "total_error_qty",
                    "rows": chart_rows,
                    "svg": svg,
                })
        report = {
            "id": str(uuid.uuid4()),
            "report_type": "mes_report",
            "language": language,
            "title": f"Báo cáo Chất lượng MES Cấp Điều hành — {plan.period.label}" if language == "vi" else f"MES品質エグゼクティブレポート — {plan.period.label}",
            "period_label": plan.period.label,
            "generated_at": datetime.now().strftime("%H:%M %d/%m/%Y"),
            "snapshot_imported_at": imported_at,
            "kpis": kpis,
            "charts": charts,
            "matrices": matrices,
            "sections": ordered_sections,
            "observations": observations,
            "governance": [
                "Số liệu chỉ truy vấn từ MES snapshot theo view read-only đã kiểm chứng.",
                "Chủ đề báo cáo tập trung vào dữ liệu lỗi/chất lượng sản xuất, không tự suy diễn chỉ số vận hành khác.",
            ] if language == "vi" else [
                "検証済みリードオンリービューのMESスナップショットからのみ集計しています。",
                "品質・エラー指標に特化し、未確認の稼働・コスト指標は推測しません。",
            ],
            "limitations": limitations,
        }
        report["markdown"] = render_markdown(report)
        report["html_content"] = render_html(report)
        yield {
            "event": "report",
            "report": report,
            "summary_text": build_summary_text(report),
        }

    async def build_report(
        self, question: str, language: str = "vi"
    ) -> tuple[dict[str, Any], str]:
        """Chạy toàn bộ plan không cần stream; trả (report, summary_text)."""
        report: dict[str, Any] | None = None
        summary_text = ""
        async for event in self.run(question, language=language):
            if event["event"] == "report":
                report = event["report"]
                summary_text = event["summary_text"]
        assert report is not None
        return report, summary_text


def generate_wms_svg_chart(top_processes: list[dict[str, Any]]) -> str:
    """Sinh SVG bar chart từ phân bổ mã vật tư theo công đoạn."""
    if not top_processes:
        return ""
    width = 640
    height = 240
    margin_left = 140
    margin_bottom = 30
    margin_top = 20
    margin_right = 30
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    max_val = max((p.get("distinct_item_count", 0) for p in top_processes), default=1)
    if max_val <= 0:
        max_val = 1

    bar_gap = 8
    num_bars = len(top_processes)
    bar_h = max(12, int((chart_h - (num_bars - 1) * bar_gap) / num_bars))

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%; height:auto; background:#ffffff; font-family:sans-serif; border-radius:8px;">',
        f'<text x="{margin_left}" y="15" font-size="12" font-weight="bold" fill="#2c3e50">'
        f'Phân bổ số lượng mã vật tư theo Top công đoạn (WMS)</text>',
    ]

    y = margin_top
    for proc in top_processes:
        raw_name = str(proc.get("process_name") or proc.get("process_id") or "Chưa rõ")
        shown_name = raw_name if len(raw_name) <= 18 else f"{raw_name[:17]}…"
        val = int(proc.get("distinct_item_count", 0))
        mapped = bool(proc.get("process_mapped"))
        fill_color = "#2f6f4f" if mapped else "#8e99a2"
        bar_w = max(4, int((val / max_val) * chart_w))

        svg_parts.append(
            f'<text x="{margin_left - 8}" y="{y + bar_h - 3}" font-size="11" fill="#475569" '
            f'text-anchor="end">{html.escape(shown_name)}</text>'
        )
        svg_parts.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_w}" height="{bar_h}" rx="3" fill="{fill_color}" />'
        )
        svg_parts.append(
            f'<text x="{margin_left + bar_w + 6}" y="{y + bar_h - 3}" font-size="11" font-weight="bold" fill="#1e293b">'
            f'{val} mã</text>'
        )
        y += bar_h + bar_gap

    svg_parts.append('</svg>')
    return "".join(svg_parts)


class HrExecutiveReportAgent:
    """Báo cáo HR tổng hợp cấp điều hành, không xuất danh sách cá nhân."""

    def __init__(self, employee_directory: Any):
        self.employee_directory = employee_directory

    @property
    def available(self) -> bool:
        return self.employee_directory is not None and self.employee_directory.count() > 0

    async def build_report(
        self,
        question: str,
        language: str = "vi",
    ) -> tuple[dict[str, Any], str]:
        departments = await asyncio.to_thread(
            self.employee_directory.department_summaries
        )
        total = await asyncio.to_thread(self.employee_directory.count)
        top_departments = departments[:8]
        japanese = language == "ja"
        title = "HR組織サマリー エグゼクティブレポート" if japanese else "Báo cáo Tổng quan Nhân sự Cấp Điều hành"
        period_label = "Danh bạ nhân sự hiện tại" if not japanese else "現行の人事ディレクトリ"
        chart_title = "部門別従業員数" if japanese else "Quy mô nhân sự theo phòng ban"
        matrix_title = "部門別従業員数の集計表" if japanese else "Ma trận tổng hợp headcount theo phòng ban"
        matrix_row_label = "部門" if japanese else "Phòng ban"
        matrix_columns = ["従業員数", "構成比"] if japanese else ["Số nhân sự", "Tỷ trọng"]
        chart = render_bar_chart_svg(
            top_departments,
            label_key="department",
            value_key="size",
            title=chart_title,
            accent="#4f6f9f",
        )
        matrix = None
        if top_departments:
            matrix = {
                "id": "department_headcount_matrix",
                "title": matrix_title,
                "row_label": matrix_row_label,
                "columns": matrix_columns,
                "heatmap": False,
                "rows": [
                    {
                        "label": row["department"],
                        "values": [
                            row["size"],
                            round(row["size"] * 100 / total, 1) if total else 0,
                        ],
                    }
                    for row in top_departments
                ],
                "max_value": max((row["size"] for row in top_departments), default=0),
            }
        observations = [
            f"Hệ thống ghi nhận {format_number(total)} nhân sự trong danh bạ."
            if not japanese
            else f"人事ディレクトリに{format_number(total)}名の従業員が記録されています。",
        ]
        if top_departments:
            leader = top_departments[0]
            observations.append(
                f"Phòng ban có quy mô lớn nhất là {leader['department']} với {format_number(leader['size'])} nhân sự."
                if not japanese
                else f"最も規模が大きい部門は{leader['department']}（{format_number(leader['size'])}名）です。"
            )
        governance = [
            "Báo cáo chỉ dùng số liệu tổng hợp theo phòng ban, không xuất danh sách hay hồ sơ cá nhân.",
            "Số liệu phản ánh employee_directory.sqlite hiện tại, không phải HRIS realtime.",
        ] if not japanese else [
            "部門別の集計データのみを使用し、個人一覧や個人プロファイルは出力しません。",
            "現在のemployee_directory.sqliteを反映しており、リアルタイムHRISではありません。",
        ]
        limitations = list(governance)
        report = {
            "id": str(uuid.uuid4()),
            "report_type": "hr_executive_report",
            "language": language,
            "title": title,
            "period_label": period_label,
            "generated_at": datetime.now().strftime("%H:%M %d/%m/%Y"),
            "snapshot_imported_at": "",
            "kpis": [
                {
                    "key": "headcount",
                    "label": "Tổng nhân sự" if not japanese else "総従業員数",
                    "value": total,
                },
                {
                    "key": "department_count",
                    "label": "Số phòng ban" if not japanese else "部門数",
                    "value": len(departments),
                },
                {
                    "key": "largest_department_size",
                    "label": "Phòng ban lớn nhất" if not japanese else "最大部門の人数",
                    "value": top_departments[0]["size"] if top_departments else 0,
                },
            ],
            "charts": [
                {
                    "id": "department_chart",
                    "title": chart_title,
                    "label_key": "department",
                    "value_key": "size",
                    "rows": top_departments,
                    "svg": chart,
                }
            ] if chart else [],
            "matrices": [matrix] if matrix else [],
            "sections": [
                {
                    "id": "department_summary",
                    "title": "Top phòng ban theo headcount" if not japanese else "部門別従業員数トップ",
                    "columns": ["department", "size"],
                    "rows": top_departments,
                }
            ],
            "observations": observations,
            "governance": governance,
            "limitations": limitations,
        }
        report["html_content"] = render_html(report)
        summary = (
            f"Đã tạo **{title}**.\n\n"
            f"Tổng quan — tổng nhân sự: {format_number(total)}; số phòng ban: {format_number(len(departments))}.\n\n"
            "Chi tiết biểu đồ và ma trận tổng hợp nằm trong thẻ báo cáo bên dưới."
            if not japanese
            else f"**{title}**を作成しました。\n\n"
            f"概要 — 総従業員数: {format_number(total)}名; 部門数: {format_number(len(departments))}。\n\n"
            "チャートと集計マトリクスの詳細は以下のレポートカードに表示されます。"
        )
        return report, summary


class MesWmsReportAgent:
    """Dựng báo cáo tồn kho WMS cấp điều hành từ current-balance contract."""

    def __init__(self, mes_wms_db: Any):
        self.mes_wms_db = mes_wms_db

    @property
    def available(self) -> bool:
        if self.mes_wms_db is None or not getattr(
            self.mes_wms_db,
            "available",
            False,
        ):
            return False
        try:
            return bool(self.mes_wms_db.compatibility().get("compatible"))
        except Exception:
            return False

    async def generate_report(
        self,
        _question: str,
        language: str = "vi",
    ) -> tuple[dict[str, Any], str]:
        """Tạo báo cáo WMS deterministic từ snapshot tương thích contract v4."""
        matrix_data = await asyncio.to_thread(
            self.mes_wms_db.get_executive_matrix_data,
            8,
        )

        quality = matrix_data.get("quality") or {}
        top_processes = matrix_data.get("processes") or []
        items = matrix_data.get("items") or []
        matrix = matrix_data.get("matrix") or {}

        svg_chart = generate_wms_svg_chart(top_processes)
        report_id = str(uuid.uuid4())
        as_of = str(quality.get("source_as_of") or "")
        distinct_items = int(quality.get("distinct_item_count") or 0)
        distinct_processes = int(quality.get("distinct_process_code_count") or 0)
        total_rows = int(quality.get("current_row_count") or 0)
        mapped_rows = int(quality.get("mapped_process_row_count") or 0)
        mapping_rate = (mapped_rows / total_rows * 100) if total_rows else 0.0
        japanese = language == "ja"
        labels = (
            {
                "title": "WMS工程倉庫 在庫エグゼクティブレポート",
                "period": f"WMSスナップショット {as_of or '未確認'}",
                "as_of": "データ基準時点",
                "timezone": "タイムゾーン未確認",
                "item_kpi": "記録済み資材コード",
                "process_kpi": "WMS工程コード",
                "mapping_kpi": "工程名マッピング率",
                "chart": "1. 工程別の資材コード分布",
                "matrix": "2. 工程 × 資材コード在庫マトリクス",
                "item_column": "資材コード",
                "governance": "ガバナンス上の注意（WMS contract v4 / Phase 2C）",
                "uom_limit": "単位マスター（UOM）が未確認のため、異なる資材コードの数量は合算していません。",
                "kpi_limit": "推移、使用期限、仕掛品、ボトルネックは現在のスナップショットから判定できません。",
                "observation": f"WMSスナップショットには{format_number(distinct_items)}件の資材コードと{format_number(distinct_processes)}件の工程コードが記録されています。",
                "safe_observation": "資材ごとの数量を保持し、異なる資材間の合計は計算していません（UOM_MASTER_UNAVAILABLE）。",
                "summary_title": "WMS在庫エグゼクティブレポートを作成しました",
                "download_note": "SVGチャートと詳細マトリクスはHTMLレポートからダウンロードできます。",
            }
            if japanese
            else {
                "title": "Báo cáo Tồn kho WMS Cấp Điều hành",
                "period": f"Snapshot WMS {as_of or 'chưa xác nhận'}",
                "as_of": "Mốc dữ liệu snapshot",
                "timezone": "Timezone chưa xác nhận",
                "item_kpi": "Mã vật tư ghi nhận",
                "process_kpi": "Mã công đoạn WMS",
                "mapping_kpi": "Độ phủ ánh xạ tên công đoạn",
                "chart": "1. Phân bổ số lượng mã vật tư theo công đoạn",
                "matrix": "2. Ma trận tồn kho công đoạn × mã vật tư",
                "item_column": "Mã vật tư",
                "governance": "Lưu ý Governance (WMS contract v4 / Phase 2C)",
                "uom_limit": "Không cộng gộp số lượng giữa các mã vật tư do chưa có Master Đơn vị tính (UOM).",
                "kpi_limit": "Chưa có căn cứ đánh giá xu hướng, hạn dùng, WIP hoặc bottleneck trên snapshot hiện tại.",
                "observation": f"WMS snapshot ghi nhận {format_number(distinct_items)} mã vật tư trên {format_number(distinct_processes)} mã công đoạn.",
                "safe_observation": "Giữ nguyên số lượng riêng theo mã vật tư, không tính tổng xuyên vật tư (UOM_MASTER_UNAVAILABLE).",
                "summary_title": "Đã tạo Báo cáo Tồn kho WMS Cấp Điều hành",
                "download_note": "SVG chart và ma trận chi tiết có trong báo cáo HTML tải xuống.",
            }
        )

        matrix_rows_data = []
        for item in items[:15]:
            row_values = []
            for p in top_processes:
                pid = p["process_id"]
                qty = matrix.get(pid, {}).get(item)
                row_values.append(qty)
            matrix_rows_data.append({"label": str(item), "values": row_values})

        charts = [
            {
                "id": "wms_process_distribution",
                "title": labels["chart"],
                "label_key": "process_name",
                "value_key": "distinct_item_count",
                "rows": [
                    {
                        "process_id": str(p.get("process_id") or ""),
                        "process_name": str(p.get("process_name") or p.get("process_id") or ""),
                        "distinct_item_count": int(p.get("distinct_item_count") or 0),
                    }
                    for p in top_processes
                ],
                "svg": svg_chart,
            }
        ] if top_processes else []

        matrices = [
            {
                "id": "wms_process_item_matrix",
                "title": labels["matrix"],
                "row_label": labels["item_column"],
                "heatmap": False,
                "columns": [
                    str(p.get("process_name") or p.get("process_id") or "")
                    for p in top_processes
                ],
                "rows": matrix_rows_data,
                "max_value": 0,
            }
        ] if top_processes and items else []

        sections = [
            {
                "id": "top_processes_table",
                "title": "Top công đoạn theo số mã vật tư" if not japanese else "資材コード数上位の工程",
                "columns": ["process_id", "process_name", "distinct_item_count"],
                "rows": [
                    {
                        "process_id": str(p.get("process_id") or ""),
                        "process_name": str(p.get("process_name") or p.get("process_id") or ""),
                        "distinct_item_count": int(p.get("distinct_item_count") or 0),
                    }
                    for p in top_processes
                ],
            }
        ] if top_processes else []

        governance = [labels["uom_limit"], labels["kpi_limit"]]
        report_dict = {
            "id": report_id,
            "report_type": "wms_executive_report",
            "language": language,
            "title": labels["title"],
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "period_label": labels["period"],
            "kpis": [
                {
                    "key": "distinct_item_count",
                    "label": labels["item_kpi"],
                    "value": distinct_items,
                },
                {
                    "key": "distinct_process_count",
                    "label": labels["process_kpi"],
                    "value": distinct_processes,
                },
                {
                    "key": "process_mapping_coverage",
                    "label": labels["mapping_kpi"],
                    "value": f"{mapping_rate:.1f}%",
                },
            ],
            "charts": charts,
            "matrices": matrices,
            "sections": sections,
            "observations": [
                labels["observation"],
                labels["safe_observation"],
            ],
            "governance": governance,
            "limitations": [
                labels["uom_limit"],
                labels["kpi_limit"],
                (
                    "HTMLレポートは上位8工程、最大15資材コードを表示します。"
                    if japanese
                    else (
                        "Báo cáo HTML hiển thị Top 8 công đoạn và tối đa "
                        "15 mã vật tư."
                    )
                ),
            ],
        }
        report_dict["html_content"] = render_html(report_dict)

        if japanese:
            summary_text = (
                f"**{labels['summary_title']}**（データ基準時点: `{as_of or '未確認'}`）。\n\n"
                f"- **資材コード:** {format_number(distinct_items)}件\n"
                f"- **WMS工程:** {format_number(distinct_processes)}件（名称マッピング率: {mapping_rate:.1f}%）\n"
                "- **データガバナンス:** 単位マスターが未確認のため、異なる資材コードの数量は合算していません。\n\n"
                f"{labels['download_note']}"
            )
        else:
            summary_text = (
                f"**{labels['summary_title']}** (mốc dữ liệu: `{as_of or 'chưa xác nhận'}`).\n\n"
                f"- **Mã vật tư:** {format_number(distinct_items)} mã\n"
                f"- **Công đoạn WMS:** {format_number(distinct_processes)} mã công đoạn (Ánh xạ tên: {mapping_rate:.1f}%)\n"
                "- **Quy tắc an toàn dữ liệu:** Không cộng gộp số lượng giữa các mã vật tư do thiếu Master UOM.\n\n"
                f"{labels['download_note']}"
            )

        return report_dict, summary_text
