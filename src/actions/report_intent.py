"""Nhận diện yêu cầu "lập báo cáo MES" và trích tham số kỳ báo cáo.

Toàn bộ hàm ở đây thuần túy trên chuỗi câu hỏi (regex + unicodedata thông qua
``mes_intent.normalized_text``), không phụ thuộc trạng thái, để test được ở môi
trường thiếu dependency nặng. Planner deterministic: câu hỏi → ``ReportPeriod``
+ danh sách bước SQL cố định, không để LLM sinh SQL cho báo cáo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from src.integrations.mes_intent import (
    extract_date,
    extract_date_range,
    extract_month,
    extract_top_limit,
    normalized_text,
)

# Marker "báo cáo" phải là yêu cầu tạo/lập/xuất báo cáo — tránh bắt nhầm câu
# hỏi dữ liệu thường có chữ "báo cáo" ở vai trò khác ("lỗi đã được báo cáo").
_REPORT_MARKERS = (
    "lap bao cao",
    "tao bao cao",
    "xuat bao cao",
    "viet bao cao",
    "lam bao cao",
    "bao cao loi",
    "bao cao san xuat",
    "bao cao tinh hinh",
    "bao cao tong hop",
    "bao cao thong ke",
    "create report",
    "generate report",
    "make a report",
    "error report",
    "production report",
)
_REPORT_MARKERS_JA = ("レポート", "報告書")

# Câu báo cáo vẫn phải thuộc ngữ cảnh dữ liệu MES (lỗi/sản xuất/Lot...).
_MES_CONTEXT_MARKERS = (
    "loi",
    "error",
    "errors",
    "defect",
    "defects",
    "ng",
    "san xuat",
    "production",
    "lot",
    "lots",
    "ma hang",
    "mes",
)
_MES_CONTEXT_MARKERS_JA = ("エラー", "不良", "欠陥", "ロット", "生産", "品番")

# Detector rộng chỉ dùng để giữ mọi yêu cầu lập báo cáo khỏi cache/fallback. Nó
# không có nghĩa Report Agent có đủ capability để thực hiện yêu cầu đó.
_REPORT_REQUEST_VERBS = (
    "lap bao cao",
    "tao bao cao",
    "xuat bao cao",
    "viet bao cao",
    "lam bao cao",
    "create report",
    "generate report",
    "make a report",
    "write report",
    "prepare report",
)
_REPORT_NOUNS = (
    "bao cao loi",
    "bao cao san xuat",
    "bao cao tinh hinh",
    "bao cao tong hop",
    "bao cao thong ke",
    "error report",
    "production report",
    "defect report",
    "quality report",
)
_REPORT_REQUEST_MARKERS_JA = (
    "レポートを作成",
    "レポート作成",
    "報告書を作成",
    "報告書作成",
    "レポートを生成",
    "報告書を生成",
    "レポートをまとめ",
    "報告書をまとめ",
)

# Capability tạm thời chỉ bao phủ hai template deterministic. Bất kỳ semantics
# tùy biến nào dưới đây phải bị từ chối thay vì bị bỏ qua rồi trả overview sai.
_DYNAMIC_REPORT_MARKERS = (
    "so sanh",
    "doi chieu",
    "chenh lech",
    "tang giam",
    "tang hoac giam",
    "thay doi",
    "du bao",
    "forecast",
    "compare",
    "comparison",
    "versus",
    " vs ",
    "chi tao",
    "chi hien thi",
    "chi bao cao",
    "khong can",
    "bo phan",
    "loai bo",
    "rieng cho",
    "cho tung",
    "voi moi",
    "moi lot",
    "moi ma hang",
    "theo tung",
    "dong gop",
    "nguong",
    "tren ",
    "duoi ",
    "lon hon",
    "nho hon",
    "ty le",
    "ti le",
    "trung binh",
    "per lot",
    "per product",
    "for each",
    "each lot",
    "each product",
    "only ",
    "without ",
    "exclude ",
    "greater than",
    "less than",
    "threshold",
    "average",
    "ratio",
    "rate",
    "increase",
    "decrease",
    "change",
)
_DYNAMIC_REPORT_MARKERS_JA = (
    "比較",
    "増減",
    "差分",
    "変化",
    "予測",
    "のみ",
    "不要",
    "除外",
    "個別",
    "それぞれ",
    "各ロット",
    "各品番",
    "超える",
    "未満",
    "以上",
    "以下",
    "比率",
    "割合",
    "平均",
)
_UNSUPPORTED_PERIOD_MARKERS = (
    "quy",
    "quarter",
    "tuan",
    "week",
    "weekly",
    "hang tuan",
    "theo tuan",
    "nam nay",
    "nam ngoai",
    "this year",
    "last year",
    "yearly",
    "hang nam",
    "theo nam",
    "hom nay",
    "hom qua",
    "today",
    "yesterday",
    "gan day",
    "recent",
    "latest",
)
_UNSUPPORTED_PERIOD_MARKERS_JA = (
    "四半期",
    "週間",
    "週次",
    "今年",
    "昨年",
    "年次",
    "今日",
    "昨日",
    "最近",
)
_NON_ERROR_REPORT_MARKERS = (
    "san luong",
    "so luong san xuat",
    "production output",
    "output quantity",
    "chi phi",
    "cost",
    "doanh thu",
    "revenue",
    "nhan su",
    "human resource",
    "operator",
    "nguoi van hanh",
    "khach hang",
    "customer",
    "theo ca",
    "ca san xuat",
    "by shift",
)
_NON_ERROR_REPORT_MARKERS_JA = (
    "生産量",
    "製造数",
    "コスト",
    "費用",
    "売上",
    "人事",
    "作業者",
    "顧客",
    "シフト別",
)
_ENTITY_FILTER_PATTERN = re.compile(
    r"\b(?:mã hàng|ma hang|sản phẩm|san pham|product|mã lot|ma lot|lot|"
    r"mã lỗi|ma loi|error|công đoạn|cong doan|process)\s+"
    r"(?=[A-Z0-9._/-]*[A-Z0-9])(?=[A-Z0-9._/-]*[-_./0-9])[A-Z0-9._/-]+\b"
)


@dataclass(frozen=True)
class ReportCapability:
    """Kết quả phân loại fail-closed cho Report Agent deterministic."""

    status: str  # not_report | supported | unsupported
    shape: str = ""  # overview | top_errors
    reason: str = ""

    @property
    def is_report(self) -> bool:
        return self.status != "not_report"

    @property
    def supported(self) -> bool:
        return self.status == "supported"


class UnsupportedReportRequest(ValueError):
    """Yêu cầu report nằm ngoài các template deterministic đã kiểm chứng."""


@dataclass(frozen=True)
class ReportPeriod:
    """Kỳ báo cáo đã resolve thành khoảng lọc ``error_time`` cụ thể."""

    kind: str = "all"  # all | month | day | range
    start: str = ""  # YYYY-MM-DD (inclusive)
    end_exclusive_sql: str = ""  # biểu thức SQLite cho cận trên (exclusive)
    label: str = "toàn bộ dữ liệu snapshot"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_filter(self) -> bool:
        return self.kind != "all"

    def where_clause(self) -> str:
        if not self.has_filter:
            return ""
        return (
            f"WHERE error_time >= '{self.start}' "
            f"AND error_time < {self.end_exclusive_sql}"
        )


def is_report_request(question: str) -> bool:
    """Detector rộng cho yêu cầu tạo report; không khẳng định capability."""
    original = question or ""
    normalized = normalized_text(question)
    if not normalized and not original:
        return False
    if any(marker in normalized for marker in _REPORT_REQUEST_VERBS):
        return True
    if re.search(
        r"\b(?:create|generate|make|write|prepare)\b(?:\s+\w+){0,5}\s+reports?\b",
        normalized,
    ):
        return True
    if any(marker in original for marker in _REPORT_REQUEST_MARKERS_JA):
        return True
    # Các noun marker chỉ được xem là request khi có ngữ cảnh mệnh lệnh/ranking;
    # tránh bắt câu mô tả như "lỗi đã được báo cáo".
    has_report_noun = any(marker in normalized for marker in _REPORT_NOUNS)
    has_command_context = bool(
        re.search(r"\b(?:hay|vui long|please|top|cho toi)\b", normalized)
    ) or normalized.startswith(("bao cao ", "report "))
    return has_report_noun and has_command_context


def is_mes_report_request(question: str) -> bool:
    """True nếu câu hỏi là yêu cầu report MES, dù có thể chưa được hỗ trợ."""
    original = question or ""
    normalized = normalized_text(question)
    if not is_report_request(question):
        return False
    # Marker ngắn "ng" là viết tắt nghiệp vụ cho lỗi/NG. Match theo word
    # boundary để không bắt nhầm các từ không liên quan như "công" hoặc
    # "marketing". Các marker dài vẫn dùng substring để chấp nhận biến thể câu.
    has_mes_context = any(
        marker in normalized
        for marker in _MES_CONTEXT_MARKERS
        if marker != "ng"
    ) or bool(re.search(r"\bng\b", normalized))
    return has_mes_context or any(
        marker in original for marker in _MES_CONTEXT_MARKERS_JA
    )


def _has_normalized_marker(normalized: str, markers: tuple[str, ...]) -> bool:
    """Match marker theo ranh giới từ để tránh ``rate`` khớp ``generate``."""
    return any(
        re.search(rf"\b{re.escape(marker.strip())}\b", normalized)
        for marker in markers
    )


def _has_multiple_periods(question: str) -> bool:
    original = question or ""
    normalized = normalized_text(question)
    date_matches = re.findall(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", original)
    # Hai ngày là một range được hỗ trợ, kể cả khi range đi qua hai tháng.
    if len(date_matches) == 2:
        return False
    if len(date_matches) > 2:
        return True

    month_matches = set(
        re.findall(r"\b20\d{2}[-/]\d{1,2}(?![-/]\d)", original)
    )
    month_matches.update(
        f"{year}-{int(month):02d}"
        for month, year in re.findall(
            r"\bthang\s+(\d{1,2})\s*(?:/|nam\s+)(20\d{2})\b",
            normalized,
        )
    )
    month_matches.update(
        f"{year}-{int(month):02d}"
        for year, month in re.findall(r"(20\d{2})年\s*(\d{1,2})月", original)
    )
    return len(month_matches) > 1


def _has_invalid_explicit_date(question: str) -> bool:
    values = re.findall(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", question or "")
    for year, month, day in values:
        try:
            date(int(year), int(month), int(day))
        except ValueError:
            return True
    return False


def _has_invalid_explicit_month(question: str) -> bool:
    original = question or ""
    normalized = normalized_text(question)
    values = re.findall(
        r"\b(20\d{2})[-/](\d{1,2})(?!\d|[-/]\d)",
        original,
    )
    values.extend(
        (year, month)
        for month, year in re.findall(
            r"\bthang\s+(\d{1,2})\s*(?:/|nam\s+)?(20\d{2})\b",
            normalized,
        )
    )
    values.extend(
        (year, month)
        for month, year in re.findall(r"\b(\d{1,2})/(20\d{2})\b", original)
    )
    values.extend(re.findall(r"(20\d{2})年\s*(\d{1,2})月", original))
    return any(not 1 <= int(month) <= 12 for _, month in values)


def report_capability(question: str) -> ReportCapability:
    """Phân loại report theo allowlist; không chắc chắn thì từ chối."""
    original = question or ""
    normalized = normalized_text(question)
    if not is_report_request(question):
        return ReportCapability(status="not_report")
    if not is_mes_report_request(question):
        return ReportCapability(
            status="unsupported",
            reason="Report Agent hiện chỉ hỗ trợ báo cáo lỗi MES.",
        )
    if any(marker in normalized for marker in _NON_ERROR_REPORT_MARKERS) or any(
        marker in original for marker in _NON_ERROR_REPORT_MARKERS_JA
    ):
        return ReportCapability(
            status="unsupported",
            reason="Loại chỉ số được yêu cầu chưa thuộc báo cáo lỗi MES đã chuẩn bị.",
        )
    if _has_multiple_periods(question):
        return ReportCapability(
            status="unsupported",
            reason="Report Agent chưa hỗ trợ so sánh hoặc tổng hợp nhiều kỳ.",
        )
    if _has_invalid_explicit_date(question):
        return ReportCapability(
            status="unsupported",
            reason="Ngày trong yêu cầu báo cáo không hợp lệ.",
        )
    if _has_invalid_explicit_month(question):
        return ReportCapability(
            status="unsupported",
            reason="Tháng trong yêu cầu báo cáo không hợp lệ.",
        )
    if _has_normalized_marker(normalized, _UNSUPPORTED_PERIOD_MARKERS) or any(
        marker in original for marker in _UNSUPPORTED_PERIOD_MARKERS_JA
    ):
        return ReportCapability(
            status="unsupported",
            reason="Kỳ báo cáo này chưa thuộc các mẫu đã chuẩn bị.",
        )
    if _has_normalized_marker(normalized, _DYNAMIC_REPORT_MARKERS) or any(
        marker in original for marker in _DYNAMIC_REPORT_MARKERS_JA
    ):
        return ReportCapability(
            status="unsupported",
            reason="Yêu cầu cần lập kế hoạch báo cáo tùy biến chưa được hỗ trợ.",
        )
    if _ENTITY_FILTER_PATTERN.search(original):
        return ReportCapability(
            status="unsupported",
            reason="Báo cáo lọc theo đối tượng cụ thể chưa thuộc mẫu đã chuẩn bị.",
        )
    shape = "top_errors" if is_top_errors_report_request(question) else "overview"
    return ReportCapability(status="supported", shape=shape)


def _extract_extended_month(question: str) -> str:
    """Bổ sung các dạng 'tháng 6/2026', '6/2026' mà ``extract_month`` bỏ sót."""
    month = extract_month(question)
    if month:
        return month
    normalized = normalized_text(question)
    match = re.search(r"\bthang\s+(\d{1,2})\s*(?:/|nam\s+)?(20\d{2})\b", normalized)
    if not match:
        match = re.search(r"\b(\d{1,2})/(20\d{2})\b", question or "")
    if match:
        month_value, year = match.groups()
        month_number = int(month_value)
        if 1 <= month_number <= 12:
            return f"{year}-{month_number:02d}"
    return ""


def report_period_for_question(question: str) -> ReportPeriod:
    """Resolve kỳ báo cáo từ câu hỏi; mặc định là toàn bộ snapshot."""
    range_start, range_end = extract_date_range(question)
    if range_start and range_end and range_start != range_end:
        return ReportPeriod(
            kind="range",
            start=range_start,
            end_exclusive_sql=f"date('{range_end}', '+1 day')",
            label=f"từ {range_start} đến {range_end}",
        )

    explicit_date = extract_date(question)
    if explicit_date:
        return ReportPeriod(
            kind="day",
            start=explicit_date,
            end_exclusive_sql=f"date('{explicit_date}', '+1 day')",
            label=f"ngày {explicit_date}",
        )

    explicit_month = _extract_extended_month(question)
    if explicit_month:
        month_start = f"{explicit_month}-01"
        year, month = explicit_month.split("-")
        return ReportPeriod(
            kind="month",
            start=month_start,
            end_exclusive_sql=f"date('{month_start}', '+1 month')",
            label=f"tháng {int(month)}/{year}",
        )

    notes: tuple[str, ...] = ()
    normalized = normalized_text(question)
    if re.search(r"\bthang\s+\d{1,2}\b", normalized):
        notes = (
            "Câu hỏi nêu tháng nhưng không rõ năm nên báo cáo tổng hợp "
            "toàn bộ snapshot.",
        )
    return ReportPeriod(notes=notes)


def is_top_errors_report_request(question: str) -> bool:
    """True khi người dùng chỉ yêu cầu báo cáo Top N loại lỗi.

    Ví dụ ``Lập báo cáo về 7 lỗi nhiều nhất`` phải tập trung vào bảy loại lỗi,
    không nhân cùng limit sang Top Lot/Top mã hàng/trend. Marker ``tổng hợp``
    giữ lại báo cáo overview chuẩn nhiều section.
    """
    normalized = normalized_text(question)
    if any(marker in normalized for marker in ("tong hop", "bao cao tinh hinh")):
        return False
    has_ranked_errors = "bao cao ve" in normalized and bool(
        re.search(
            r"\b(?:top\s*)?\d+\s+(?:ma\s+|loai\s+)?loi\b",
            normalized,
        )
    )
    has_other_entity = bool(
        re.search(r"\b(?:lot|lots|ma hang|san pham|product)\b", normalized)
    )
    return has_ranked_errors and not has_other_entity


def report_top_limit(question: str, *, default: int = 5, maximum: int = 20) -> int:
    """Trích giới hạn Top N cho báo cáo.

    ``mes_intent.extract_top_limit`` đã đọc ``top 7`` và ``7 loại lỗi``. Report
    prompt thường rút gọn thành ``báo cáo về 7 lỗi nhiều nhất`` nên bổ sung mẫu
    ``N lỗi`` ở đây; số vẫn được clamp để tránh artifact quá lớn.
    """
    normalized = normalized_text(question)
    extracted = extract_top_limit(normalized, default=default, maximum=maximum)
    if extracted != default or re.search(r"\btop\s*\d+\b", normalized):
        return extracted
    match = re.search(r"\b(\d+)\s+(?:ma\s+|loai\s+)?loi\b", normalized)
    if not match:
        return extracted
    return max(1, min(maximum, int(match.group(1))))
