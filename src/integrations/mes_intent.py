"""Nhận diện ý định (intent) và sinh SQL tất định cho câu hỏi MES.

Toàn bộ các hàm ở đây là hàm thuần túy trên chuỗi câu hỏi (chỉ dùng ``re`` và
``unicodedata``), không phụ thuộc trạng thái. Gom chung để khi cần thêm/điều
chỉnh cách nhận diện một loại câu hỏi thì chỉ sửa ở một chỗ. Giữ nguyên hành vi
so với bản gốc.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date


def normalized_text(question: str) -> str:
    normalized = unicodedata.normalize(
        "NFD",
        (question or "").lower().replace("đ", "d"),
    )
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    normalized = re.sub(r"\bnhiu\b", "nhieu", normalized)
    return re.sub(r"\bnhiu\b", "nhieu", normalized)


def is_highest_lot_error_question(question: str) -> bool:
    normalized = unicodedata.normalize(
        "NFD",
        question.lower().replace("đ", "d"),
    )
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    normalized = re.sub(r"\bnhiu\b", "nhieu", normalized)

    has_lot = bool(re.search(r"\b(lot|lots|lo|lo san xuat)\b", normalized))
    has_error = bool(re.search(r"\bng\b", normalized)) or any(
        marker in normalized
        for marker in (
            "loi",
            "error",
            "errors",
            "defect",
            "defects",
            "hang loi",
            "san pham loi",
        )
    )
    is_average = bool(re.search(r"\b(trung binh|average)\b", normalized))
    if is_average:
        return False

    has_maximum = bool(
        re.search(r"\b(nhieu|cao|lon)\b(?:\s+\w+){0,3}\s+nhat\b", normalized)
        or re.search(
            r"\btop\s*(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)?\b",
            normalized,
        )
    ) or any(
        marker in normalized
        for marker in (
            "nhieu nhat",
            "cao nhat",
            "lon nhat",
            "toi da",
            "top 1",
            "top loi",
            "dung dau",
            "max",
            "maximum",
            "most",
            "highest",
            "largest",
            "greatest",
        )
    )
    return has_lot and has_error and has_maximum


def is_compound_mes_question(question: str) -> bool:
    normalized = unicodedata.normalize(
        "NFD",
        question.lower().replace("đ", "d"),
    )
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    normalized = re.sub(r"\bnhiu\b", "nhieu", normalized)
    number_token = (
        r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    )
    return bool(
        re.search(rf"\b{number_token}\s+(?:ma\s+|loai\s+)?loi\b", normalized)
        or re.search(
            rf"\btop\s*{number_token}\s+(?:ma\s+|loai\s+)?loi\b",
            normalized,
        )
        or re.search(
            rf"\btop\s*{number_token}\s+(?:error|defect)\s+(?:code|codes|type|types)\b",
            normalized,
        )
        or re.search(
            rf"\b{number_token}\s+(?:error|defect)\s+(?:code|codes|type|types)\b",
            normalized,
        )
        or any(
            marker in normalized
            for marker in ("cac loi nhieu nhat", "nhung loi nhieu nhat")
        )
        or (
            ("error code" in normalized or "error codes" in normalized)
            and any(marker in normalized for marker in ("top", "highest", "most"))
        )
    )


def is_time_related_mes_question(question: str) -> bool:
    original = (question or "").lower()
    normalized = unicodedata.normalize(
        "NFD",
        original.replace("đ", "d"),
    )
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    has_explicit_date_value = bool(
        re.search(r"\b20\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?\b", original)
    )
    if has_explicit_date_value:
        return True
    return any(
        marker in normalized
        for marker in (
            "ngay",
            "hom nay",
            "hom qua",
            "theo ngay",
            "moi ngay",
            "thang",
            "theo thang",
            "moi thang",
            "nam",
            "tuan",
            "khoang thoi gian",
            "tu ngay",
            "den ngay",
            "gan day",
            "moi nhat",
            "date",
            "day",
            "daily",
            "today",
            "yesterday",
            "month",
            "monthly",
            "year",
            "yearly",
            "week",
            "weekly",
            "between",
            "from",
            "recent",
            "latest",
        )
    ) or any(
        marker in original
        for marker in ("今日", "昨日", "日", "月", "年", "期間", "いつ")
    )


def time_sql_for_question(question: str) -> str:
    original = (question or "").lower()
    normalized = normalized_text(question)
    if not is_time_related_mes_question(question):
        return ""

    range_start, range_end = extract_date_range(question)
    explicit_month = extract_month(question)
    explicit_date = extract_date(question)
    limit = extract_top_limit(normalized)
    has_lot = bool(re.search(r"\b(lot|lots|lo|lo san xuat)\b", normalized))
    has_error = bool(
        re.search(r"\b(ng|loi|error|errors|defect|defects)\b", normalized)
        or any(marker in original for marker in ("エラー", "不良", "欠陥"))
    )
    asks_error_type = any(
        marker in normalized
        for marker in (
            "ma loi",
            "loai loi",
            "loi pho bien",
            "pho bien nhat",
            "error code",
            "error codes",
            "defect code",
            "defect codes",
            "top error",
            "top errors",
        )
    )
    asks_top = has_top_marker(normalized) or any(
        marker in original for marker in ("上位", "最も", "多い", "最大")
    )
    asks_month = "thang" in normalized or "month" in normalized or "月" in original
    asks_day = "ngay" in normalized or "day" in normalized or "日" in original

    if range_start and range_end and has_lot and has_error and asks_top:
        return f"""
            SELECT lot_id, product_id, SUM(quantity) AS total_error_qty
            FROM v_error_details
            WHERE error_time >= '{range_start}'
              AND error_time < date('{range_end}', '+1 day')
            GROUP BY lot_id, product_id
            ORDER BY total_error_qty DESC, lot_id
            LIMIT {limit}
        """

    if range_start and range_end and has_error:
        return f"""
            SELECT date(error_time) AS error_date,
                   SUM(quantity) AS total_error_qty,
                   COUNT(*) AS error_record_count
            FROM v_error_details
            WHERE error_time >= '{range_start}'
              AND error_time < date('{range_end}', '+1 day')
            GROUP BY date(error_time)
            ORDER BY error_date
            LIMIT 366
        """

    if explicit_date and has_lot and has_error and asks_top:
        return f"""
            SELECT lot_id, product_id, SUM(quantity) AS total_error_qty
            FROM v_error_details
            WHERE error_time >= '{explicit_date}'
              AND error_time < date('{explicit_date}', '+1 day')
            GROUP BY lot_id, product_id
            ORDER BY total_error_qty DESC, lot_id
            LIMIT {limit}
        """

    if explicit_month and has_lot and has_error and asks_top:
        month_start = f"{explicit_month}-01"
        return f"""
            SELECT lot_id, product_id, SUM(quantity) AS total_error_qty
            FROM v_error_details
            WHERE error_time >= '{month_start}'
              AND error_time < date('{month_start}', '+1 month')
            GROUP BY lot_id, product_id
            ORDER BY total_error_qty DESC, lot_id
            LIMIT {limit}
        """

    if explicit_month and asks_error_type and has_error:
        month_start = f"{explicit_month}-01"
        return f"""
            SELECT error_id, error_name, SUM(quantity) AS total_error_qty
            FROM v_error_details
            WHERE error_time >= '{month_start}'
              AND error_time < date('{month_start}', '+1 month')
            GROUP BY error_id, error_name
            ORDER BY total_error_qty DESC, error_id
            LIMIT {limit}
        """

    if asks_month and asks_error_type and asks_top and has_error:
        return f"""
            WITH top_month AS (
                SELECT strftime('%Y-%m', error_time) AS error_month
                FROM v_error_details
                WHERE error_time IS NOT NULL
                GROUP BY strftime('%Y-%m', error_time)
                ORDER BY SUM(quantity) DESC
                LIMIT 1
            )
            SELECT t.error_month, e.error_id, e.error_name,
                   SUM(e.quantity) AS total_error_qty
            FROM v_error_details AS e
            JOIN top_month AS t
              ON strftime('%Y-%m', e.error_time) = t.error_month
            GROUP BY t.error_month, e.error_id, e.error_name
            ORDER BY total_error_qty DESC, e.error_id
            LIMIT {limit}
        """

    if asks_month and has_error and asks_top and not has_lot:
        return """
            SELECT strftime('%Y-%m', error_time) AS error_month,
                   SUM(quantity) AS total_error_qty,
                   COUNT(*) AS error_record_count
            FROM v_error_details
            WHERE error_time IS NOT NULL
            GROUP BY strftime('%Y-%m', error_time)
            ORDER BY total_error_qty DESC
            LIMIT 1
        """

    if asks_day and has_error and asks_top and not has_lot:
        return """
            SELECT date(error_time) AS error_date,
                   SUM(quantity) AS total_error_qty,
                   COUNT(*) AS error_record_count
            FROM v_error_details
            WHERE error_time IS NOT NULL
            GROUP BY date(error_time)
            ORDER BY total_error_qty DESC
            LIMIT 1
        """
    return ""


def should_use_sql_agent(question: str) -> bool:
    """Return True only for questions that look like MES data queries."""
    original = question or ""
    normalized = normalized_text(question)
    if not normalized and not original:
        return False
    strong_data_markers = (
        "ma hang",
        "ma loi",
        "error code",
        "defect code",
        "product code",
        "bao nhieu",
        "tong",
        "so luong",
        "thong ke",
        "liet ke",
        "danh sach",
        "top",
        "nhieu nhat",
        "cao nhat",
        "pho bien nhat",
        "count",
        "sum",
        "total",
        "list",
        "rank",
        "compare",
        "comparison",
    )
    if any(marker in normalized for marker in strong_data_markers):
        return True
    if re.search(
        r"(ロット|Lot|品番|製品|合計|総数|件数|一覧|上位|多い|最大|集計)",
        original,
    ):
        return True

    general_markers = (
        "la gi",
        "khai niem",
        "giai thich",
        "quy trinh",
        "huong dan",
        "cach ghi nhan",
        "y nghia",
        "what is",
        "explain",
        "concept",
        "process",
        "procedure",
    )
    if any(marker in normalized for marker in general_markers):
        return False
    if re.search(r"(とは|説明|意味|手順|プロセス|ガイド)", original):
        return False

    structured_markers = (
        "lot",
        "lots",
        "ma hang",
        "ma loi",
        "loi",
        "error",
        "errors",
        "defect",
        "defects",
        "product",
        "product code",
        "process",
        "cong doan",
        "thong ke",
        "tong",
        "so luong",
        "bao nhieu",
        "liet ke",
        "danh sach",
        "top",
        "nhieu nhat",
        "cao nhat",
        "pho bien",
        "count",
        "sum",
        "total",
        "list",
        "rank",
        "compare",
        "comparison",
    )
    if any(marker in normalized for marker in structured_markers):
        return True
    if re.search(r"\b(lo|ng)\b", normalized):
        return True
    return bool(
        re.search(
            r"(ロット|Lot|エラー|不良|欠陥|品番|製品|工程|合計|総数|件数|一覧|上位|多い|最大|集計)",
            original,
        )
    )


def _normalized_date(year: str, month: str, day: str) -> str:
    try:
        parsed = date(int(year), int(month), int(day))
    except ValueError:
        return ""
    return parsed.isoformat()


def extract_date(question: str) -> str:
    original = question or ""
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", original)
    if not match:
        return ""
    return _normalized_date(*match.groups())


def extract_date_range(question: str) -> tuple[str, str]:
    """Resolve two valid explicit dates as an inclusive, ordered range."""
    matches = re.findall(
        r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",
        question or "",
    )
    if len(matches) != 2:
        return "", ""
    dates = [_normalized_date(*value) for value in matches]
    if not all(dates):
        return "", ""
    start, end = sorted(dates)
    return start, end


def _normalized_month(year: str, month: str) -> str:
    month_number = int(month)
    if not 1 <= month_number <= 12:
        return ""
    return f"{year}-{month_number:02d}"


def extract_month(question: str) -> str:
    original = question or ""
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})(?:[-/]\d{1,2})?\b", original)
    if match:
        return _normalized_month(*match.groups())
    normalized = normalized_text(question)
    match = re.search(r"\bthang\s+(\d{1,2})\s*(?:/|nam\s+)?(20\d{2})\b", normalized)
    if match:
        month, year = match.groups()
        return _normalized_month(year, month)
    match = re.search(r"\b(\d{1,2})/(20\d{2})\b", original)
    if match:
        month, year = match.groups()
        return _normalized_month(year, month)
    match = re.search(r"\b(20\d{2})\s+nam\s+thang\s+(\d{1,2})\b", normalized)
    if match:
        return _normalized_month(*match.groups())
    japanese_match = re.search(r"(20\d{2})年\s*(\d{1,2})月", original)
    if japanese_match:
        return _normalized_month(*japanese_match.groups())
    return ""


def extract_top_limit(normalized: str, default: int = 5, maximum: int = 50) -> int:
    match = re.search(r"\btop\s*(\d+)\b", normalized)
    if not match:
        match = re.search(
            r"\b(\d+)\s+(?:lot|lots|lo|ma loi|loai loi|error|errors)\b",
            normalized,
        )
    if not match:
        return default
    return max(1, min(maximum, int(match.group(1))))


def has_top_marker(normalized: str) -> bool:
    return bool(
        re.search(r"\b(nhieu|cao|lon|pho bien)\b(?:\s+\w+){0,3}\s+nhat\b", normalized)
        or re.search(r"\btop\s*\d*\b", normalized)
    ) or any(
        marker in normalized
        for marker in (
            "nhieu nhat",
            "cao nhat",
            "lon nhat",
            "pho bien nhat",
            "dung dau",
            "max",
            "maximum",
            "most",
            "highest",
            "largest",
            "greatest",
        )
    )
