"""Định dạng câu trả lời fallback và kiểm chứng đầu ra của model cho MES.

Đây là các "gác cổng" chống model bịa/lộ trường kỹ thuật: kiểm tra câu trả lời có
đủ mã/số liệu bắt buộc không, có tự nhiên không, có khớp kết quả SQL không. Tách
riêng để phần kiểm chứng dễ mở rộng. Giữ nguyên hành vi so với bản gốc.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - chỉ dùng cho type hint
    from src.integrations.mes_client import MesLotError
    from src.integrations.mes_database import MesDatabaseResult
    from src.integrations.mes_sql_agent import MesSqlQueryResult


def format_item_list(items: list[str], *, inline_threshold: int = 3) -> str:
    """Nối các mục thành câu tự nhiên nếu ít, hoặc danh sách markdown nếu nhiều.

    Model hay dồn nhiều Lot/mã hàng nối bằng "; " thành một đoạn văn dài, khó
    đọc. Từ trên `inline_threshold` mục, trình bày mỗi mục trên một dòng gạch
    đầu dòng để cả câu trả lời kiểm chứng và giao diện chat (ReactMarkdown)
    render thành danh sách thật.
    """
    if len(items) <= inline_threshold:
        return "; ".join(items)
    return "\n\n" + "\n".join(f"- {item}" for item in items)


def format_live_api_fallback(lots: list["MesLotError"]) -> str:
    def describe(lot: "MesLotError") -> str:
        quantity = f"{lot.total_error_qty:,}".replace(",", ".")
        return (
            f"Lot {lot.lot_id}, mã hàng {lot.product_id}, "
            f"với tổng cộng {quantity} lỗi"
        )

    if len(lots) == 1:
        return f"{describe(lots[0])} là Lot có số lượng lỗi cao nhất."
    return "Các Lot có số lượng lỗi cao nhất là: " + format_item_list(
        [describe(lot) for lot in lots]
    ) + "."


def live_api_answer_has_required_fields(
    answer: str,
    lots: list["MesLotError"],
) -> bool:
    normalized_quantity = answer.replace(".", "").replace(",", "")
    return bool(answer.strip()) and all(
        lot.lot_id in answer
        and lot.product_id in answer
        and str(lot.total_error_qty) in normalized_quantity
        for lot in lots
    )


def database_answer_has_required_terms(
    answer: str,
    result: "MesDatabaseResult",
) -> bool:
    if not answer.strip():
        return False
    forbidden_fields = (
        "total_error_qty",
        "error_record_count",
        "distinct_error_count",
        "unmapped_error_record_count",
        "lot_count",
        "product_id",
        "lot_id",
        "error_id",
        "process_id",
        "json",
        "sql",
        "filters",
        "filter",
        "chính sách hiển thị",
        "chinh sach hien thi",
        "表示ポリシー",
        "フィルタ",
    )
    if any(field in answer.lower() for field in forbidden_fields):
        return False
    normalized_answer = answer.replace(".", "").replace(",", "")
    return all(
        not term
        or term in answer
        or term.replace(".", "").replace(",", "") in normalized_answer
        for term in result.required_terms
    )


def normalize_sql_answer(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(payload, dict):
            nested_answer = payload.get("answer")
            if isinstance(nested_answer, str):
                return nested_answer.strip()
    return text


def sql_answer_matches_result(answer: str, result: "MesSqlQueryResult") -> bool:
    normalized = answer.lower()
    normalized_numbers = answer.replace(".", "").replace(",", "")
    for row in result.rows[:5]:
        checked_row_value = False
        for key in ("lot_id", "product_id", "error_id"):
            value = row.get(key)
            checked_row_value = checked_row_value or bool(value)
            if value and str(value).lower() not in normalized:
                return False
        error_name = row.get("error_name")
        checked_row_value = checked_row_value or bool(error_name)
        if error_name and str(error_name).strip():
            if str(error_name).lower() not in normalized:
                return False
        elif "error_name" in row and "chưa rõ tên" not in normalized:
            return False
        if checked_row_value:
            continue

        for key, value in row.items():
            if value is None or value == "":
                continue
            key_lower = key.lower()
            if any(
                marker in key_lower
                for marker in ("date", "time", "month", "day", "period")
            ):
                if str(value).lower() not in normalized:
                    return False
            elif isinstance(value, (int, float)) and any(
                marker in key_lower
                for marker in (
                    "total",
                    "qty",
                    "quantity",
                    "count",
                    "sum",
                )
            ):
                compact_value = str(int(value) if float(value).is_integer() else value)
                if compact_value not in normalized_numbers:
                    return False
    return True


def sql_answer_is_natural(answer: str) -> bool:
    if not answer.strip():
        return False
    normalized = answer.lower()
    forbidden = (
        "select ",
        " from ",
        "total_error_qty",
        "error_record_count",
        "distinct_error_count",
        "```sql",
        "{\"answer\"",
    )
    return not any(marker in normalized for marker in forbidden)
