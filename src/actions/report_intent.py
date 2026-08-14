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
_WMS_CONTEXT_MARKERS = (
    "ton kho",
    "kho cong doan",
    "wms",
    "warehouse stock",
    "process warehouse",
    "inventory",
)
_WMS_CONTEXT_MARKERS_JA = ("在庫", "倉庫", "保管", "資材", "受払", "入出庫")
_HR_CONTEXT_MARKERS = (
    "nhan su",
    "nhan vien",
    "co cau phong ban",
    "phong ban",
    "headcount",
    "hr",
    "human resource",
    "employee",
    "employees",
    # Cụm phải đủ dài để không khớp substring trong "sản lượng"/"chất lượng".
    "chi phi luong",
    "chi phi nhan cong",
    "quy luong",
    "bang luong",
    "payroll",
    "salary",
    "cham cong",
    "tuyen dung",
    "nghi viec",
)
_HR_CONTEXT_MARKERS_JA = (
    "人事",
    "社員",
    "従業員",
    "部門",
    "部署",
    "組織",
    "給与",
    "人件費",
    "採用",
    "勤怠",
    "離職",
)

_EXECUTIVE_AUDIENCE_MARKERS = (
    "chu tich",
    "ban giam doc",
    "ban lanh dao",
    "tong giam doc",
    "lanh dao",
    "quan ly cap cao",
    "executive",
    "executives",
    "board",
    "president",
    "director",
    "management",
)
_EXECUTIVE_AUDIENCE_MARKERS_JA = ("社長", "会長", "役員", "経営層", "取締役", "管理職")
_EXECUTIVE_OVERVIEW_MARKERS = (
    "tong quan",
    "tinh hinh chung",
    "buc tranh tong the",
    "tong the",
    "executive overview",
    "overview",
    "summary",
    "overall status",
)
_EXECUTIVE_OVERVIEW_MARKERS_JA = ("概要", "全体", "全体像", "状況", "サマリー")
_MES_ERROR_SCOPE_MARKERS = (
    "loi",
    "error",
    "errors",
    "defect",
    "defects",
    "quality",
    "chat luong",
    "mes",
)
_MES_ERROR_SCOPE_MARKERS_JA = ("エラー", "不良", "欠陥", "品質", "MES")

# Phase 2C chỉ cho phép một snapshot overview theo current balance. Các marker
# dưới đây đòi business contract/source chưa có, nên report path phải từ chối
# thay vì âm thầm đổi yêu cầu thành overview mặc định.
_WMS_UNSUPPORTED_REPORT_MARKERS = (
    "tong ton kho",
    "tong so luong ton",
    "xep hang so luong",
    "nhieu nhat",
    "it nhat",
    "minimum stock",
    "min stock",
    "safety stock",
    "ton toi thieu",
    "han su dung",
    "expiry",
    "window time",
    "xu huong",
    "trend",
    "delta",
    "net movement",
    "so sanh",
    "comparison",
    "completed movement",
    "giao dich hoan thanh",
    "nhap xuat",
    "wip",
    "bottleneck",
    "nut that",
    "tuoi hang",
    "tuoi ton kho",
    "ton dong",
    "xuat nhap ton",
    "nhap xuat ton",
    "ton dau ky",
    "ton cuoi ky",
    "vong quay",
    "turnover",
    "gia tri ton",
    "gia tri ton kho",
    "chi phi luu kho",
    "chi phi ton kho",
    "aging",
    "inventory value",
)
_WMS_UNSUPPORTED_REPORT_MARKERS_JA = (
    "総在庫",
    "在庫合計",
    "最低在庫",
    "安全在庫",
    "使用期限",
    "有効期限",
    "保管時間",
    "在庫推移",
    "増減",
    "差分",
    "比較",
    "入出庫",
    "完了取引",
    "仕掛品",
    "WIP",
    "ボトルネック",
    "在庫年齢",
    "滞留",
    "受払",
    "期首在庫",
    "期末在庫",
    "回転率",
    "在庫金額",
    "保管コスト",
)

# HR chỉ có snapshot danh bạ dạng aggregate. Lương, KPI cá nhân, tuyển dụng,
# chấm công và roster đều cần nguồn HRIS chưa kết nối, nên phải từ chối rõ ràng
# thay vì trả tổng quan headcount cho một câu hỏi khác hẳn phạm vi.
_HR_UNSUPPORTED_REPORT_MARKERS = (
    "chi phi luong",
    "chi phi nhan cong",
    "quy luong",
    "bang luong",
    "payroll",
    "salary",
    "compensation",
    "thu nhap",
    "thuong",
    "bonus",
    "kpi ca nhan",
    "kpi tung nguoi",
    "kpi nhan vien",
    "individual kpi",
    "danh gia hieu suat",
    "danh gia nhan vien",
    "performance review",
    "performance appraisal",
    "tuyen dung",
    "recruitment",
    "headcount plan",
    "ke hoach nhan su",
    "nghi viec",
    "attrition",
    "turnover",
    "cham cong",
    "attendance",
    "nghi phep",
    "leave balance",
    "overtime",
    "tang ca",
    "hop dong lao dong",
    "labor contract",
    "ho so ca nhan",
    "ho so nhan vien",
    "danh sach nhan vien",
    "employee list",
    "roster",
    "profile",
    "bao hiem",
    "insurance",
    "thue thu nhap",
)
_HR_UNSUPPORTED_REPORT_MARKERS_JA = (
    "給与",
    "人件費",
    "賞与",
    "個人KPI",
    "人事評価",
    "採用",
    "離職",
    "勤怠",
    "残業",
    "雇用契約",
    "個人情報",
    "従業員名簿",
    "社会保険",
)

# Câu chứa chữ "báo cáo" ở vai trò mô tả/thủ tục, không phải khẩu lệnh tạo báo
# cáo. Phải loại trước khi matcher mở rộng bắt danh từ đứng đầu.
# Khẩu lệnh tổng quan nói tắt: đủ để mở report khi đã có domain rõ và câu không
# phải dạng hỏi/lọc theo đối tượng.
_BARE_OVERVIEW_COMMAND_MARKERS = (
    "tong quan",
    "tinh hinh",
    "buc tranh tong the",
    "tong the",
    "overview",
    "overall status",
    "current status",
)
# Dấu hiệu câu hỏi tra cứu: phải giữ nguyên luồng Q&A thường.
# Khẩu lệnh nói tắt bị thu hẹp về một đối tượng cụ thể vẫn là Q&A/query, không
# phải report tổng quan: "Tình hình Lot 000866-05-000", "Tình hình nhân sự phòng
# Kế toán". Kiểm tra không phân biệt hoa thường vì khẩu lệnh viết tự do.
_NARROWED_SCOPE_PATTERN = re.compile(
    r"\b(?:lot|ma hang|san pham|product|ma loi|error|cong doan|process|"
    r"phong|bo phan|to|line|khach hang|customer)\b\s*\S+",
    flags=re.IGNORECASE,
)
_INTERROGATIVE_MARKERS = (
    "nao",
    "bao nhieu",
    "the nao",
    "sao",
    "gi",
    "co khong",
    "khi nao",
    "o dau",
    "dau",
    "ai",
    "which",
    "what",
    "how",
    "when",
    "where",
    "who",
    "why",
)
_REPORT_NON_REQUEST_MARKERS = (
    "duoc bao cao",
    "da bao cao",
    "bao cao boi",
    "ai bao cao",
    "khi nao bao cao",
    "cach bao cao",
    "quy trinh bao cao",
    "huong dan bao cao",
    "mau bao cao",
    "bieu mau bao cao",
    "bao cao la gi",
    "bao cao cho biet",
    "bao cao noi gi",
    "y nghia bao cao",
)

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
    # Khẩu lệnh ngắn cấp quản lý: động từ giao việc đứng liền "báo cáo".
    "xem bao cao",
    "coi bao cao",
    "gui bao cao",
    "trinh bao cao",
    "chuan bi bao cao",
    "can bao cao",
    "xin bao cao",
    "hien thi bao cao",
    "dua bao cao",
    "show report",
    "display report",
    "export report",
    "send report",
    "share report",
    "need report",
)
# Động từ giao việc thường cách "báo cáo" vài từ: "cho anh coi báo cáo WMS",
# "trình Ban Giám đốc báo cáo nhân sự", "xuất file báo cáo tồn kho".
_REPORT_VERB_NEAR_NOUN_PATTERN = re.compile(
    r"\b(?:lap|tao|xuat|viet|lam|xem|coi|gui|trinh|chuan|can|xin|hien|dua|"
    r"show|display|export|send|share|prepare|need|pull)\b"
    r"(?:\s+\w+){0,4}\s+(?:bao cao|report|reports)\b"
)
# Danh từ đứng đầu kiểu khẩu lệnh: "Báo cáo tồn kho WMS", "Báo cáo WMS đâu",
# "Báo cáo headcount". Chỉ nhận khi câu mở đầu bằng chính danh từ báo cáo.
_REPORT_LEADING_NOUN_PATTERN = re.compile(r"^(?:bao cao|report)\b")
# Cú pháp slash/bang command: "/report wms", "!report mes", "/baocao hr".
_REPORT_SLASH_COMMAND_PATTERN = re.compile(
    r"^\s*[/!](?:report|reports|bao\s*cao|baocao|rp)\b",
    flags=re.IGNORECASE,
)
_REPORT_NOUNS = (
    "bao cao loi",
    "bao cao san xuat",
    "bao cao tinh hinh",
    "bao cao tong hop",
    "bao cao thong ke",
    "bao cao ton kho",
    "bao cao nhan su",
    "bao cao headcount",
    "bao cao chat luong",
    "error report",
    "production report",
    "defect report",
    "quality report",
    "inventory report",
    "headcount report",
    "hr report",
    "wms report",
    "mes report",
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
    # Khẩu lệnh ngắn: 出力/表示/見せて/提出/送って + レポート/報告書.
    "レポートを出力",
    "レポート出力",
    "報告書を出力",
    "レポートを表示",
    "レポート表示",
    "報告書を表示",
    "レポートを見せて",
    "報告書を見せて",
    "レポートをください",
    "報告書をください",
    "レポートを提出",
    "報告書を提出",
    "レポートを送って",
    "報告書を送って",
    "レポートが必要",
    "報告書が必要",
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
    shape: str = ""  # overview | top_errors | hr_executive | wms_executive
    reason: str = ""
    domain: str = ""  # hr | mes | wms

    @property
    def is_report(self) -> bool:
        return self.status != "not_report"

    @property
    def supported(self) -> bool:
        return self.status == "supported"


class UnsupportedReportRequest(ValueError):
    """Yêu cầu report nằm ngoài các template deterministic đã kiểm chứng."""


def report_capability_for_mode(question: str, mode: str) -> ReportCapability:
    """Classify reports and fail closed when the selected mode does not match."""
    capability = report_capability(question)
    expected_mode = {
        "hr": "mkac",
        "mes": "mes",
        "wms": "wms",
    }.get(capability.domain)
    if expected_mode and mode != expected_mode:
        return ReportCapability(
            status="unsupported",
            shape="mode_mismatch",
            domain=capability.domain,
            reason=(
                f"Report domain {capability.domain} requires mode {expected_mode}; "
                f"current mode is {mode}."
            ),
        )
    return capability


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


def _has_domain_context(question: str, normalized: str) -> bool:
    original = question or ""
    has_mes = any(
        marker in normalized for marker in _MES_CONTEXT_MARKERS if marker != "ng"
    ) or bool(re.search(r"\bng\b", normalized)) or any(
        marker in original for marker in _MES_CONTEXT_MARKERS_JA
    )
    has_wms = any(marker in normalized for marker in _WMS_CONTEXT_MARKERS) or any(
        marker in original for marker in _WMS_CONTEXT_MARKERS_JA
    )
    has_hr = any(marker in normalized for marker in _HR_CONTEXT_MARKERS) or any(
        marker in original for marker in _HR_CONTEXT_MARKERS_JA
    )
    return has_mes or has_wms or has_hr


def is_executive_overview_request(question: str) -> bool:
    """Yêu cầu tổng quan cấp điều hành, đủ hẹp để không bắt Q&A thường."""
    original = question or ""
    normalized = normalized_text(question)
    if not normalized and not original:
        return False
    has_audience = any(
        marker in normalized for marker in _EXECUTIVE_AUDIENCE_MARKERS
    ) or any(marker in original for marker in _EXECUTIVE_AUDIENCE_MARKERS_JA)
    has_overview = any(
        marker in normalized for marker in _EXECUTIVE_OVERVIEW_MARKERS
    ) or any(marker in original for marker in _EXECUTIVE_OVERVIEW_MARKERS_JA)
    return has_audience and has_overview and _has_domain_context(question, normalized)


def _is_bare_overview_command(question: str, normalized: str) -> bool:
    """Khẩu lệnh tổng quan nói tắt, không có chữ "báo cáo" và không phải câu hỏi.

    Ví dụ đúng: "Tình hình tồn kho WMS hiện tại", "Tổng quan nhân sự".
    Ví dụ phải loại: "Tình hình Lot A thế nào?", "Lot nào tồn nhiều nhất?".
    """
    original = question or ""
    has_overview = _has_normalized_marker(
        normalized,
        _BARE_OVERVIEW_COMMAND_MARKERS,
    ) or any(marker in original for marker in _EXECUTIVE_OVERVIEW_MARKERS_JA)
    if not has_overview:
        return False
    if not _has_domain_context(question, normalized):
        return False
    # Câu hỏi tra cứu và câu lọc theo một đối tượng cụ thể vẫn là Q&A thường.
    if "?" in original or _has_normalized_marker(
        normalized,
        _INTERROGATIVE_MARKERS,
    ):
        return False
    if _NARROWED_SCOPE_PATTERN.search(normalized):
        return False
    return not _ENTITY_FILTER_PATTERN.search(original)


def is_report_request(question: str) -> bool:
    """Detector rộng cho yêu cầu tạo report; không khẳng định capability."""
    original = question or ""
    normalized = normalized_text(question)
    if not normalized and not original:
        return False
    # Slash/bang command là khẩu lệnh tường minh, xét trước mọi bộ lọc văn phong.
    if _REPORT_SLASH_COMMAND_PATTERN.match(original):
        return True
    # Câu dùng chữ "báo cáo" để mô tả/hỏi thủ tục không phải khẩu lệnh tạo báo cáo.
    if any(marker in normalized for marker in _REPORT_NON_REQUEST_MARKERS):
        return False
    if is_executive_overview_request(question):
        return True
    if any(marker in normalized for marker in _REPORT_REQUEST_VERBS):
        return True
    if re.search(
        r"\b(?:create|generate|make|write|prepare)\b(?:\s+\w+){0,5}\s+reports?\b",
        normalized,
    ):
        return True
    # Động từ giao việc cách "báo cáo" vài từ: "cho anh coi báo cáo WMS".
    if _REPORT_VERB_NEAR_NOUN_PATTERN.search(normalized):
        return True
    if any(marker in original for marker in _REPORT_REQUEST_MARKERS_JA):
        return True
    # Khẩu lệnh nói tắt mở đầu bằng danh từ: "Báo cáo tồn kho WMS", "Báo cáo
    # WMS đâu", "Báo cáo headcount". Cần có thêm nội dung sau chữ "báo cáo" để
    # không bắt câu một từ trơ.
    if _REPORT_LEADING_NOUN_PATTERN.match(normalized) and len(normalized.split()) > 2:
        return True
    # Khẩu lệnh tổng quan nói tắt không có chữ "báo cáo": "Tình hình tồn kho WMS
    # hiện tại", "Tổng quan nhân sự".
    if _is_bare_overview_command(question, normalized):
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


def _has_mes_error_scope(question: str, normalized: str) -> bool:
    original = question or ""
    return any(
        marker in normalized for marker in _MES_ERROR_SCOPE_MARKERS
    ) or bool(re.search(r"\bng\b", normalized)) or any(
        marker in original for marker in _MES_ERROR_SCOPE_MARKERS_JA
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
    is_hr_context = any(
        marker in normalized for marker in _HR_CONTEXT_MARKERS
    ) or any(marker in original for marker in _HR_CONTEXT_MARKERS_JA)
    if is_hr_context:
        if _has_normalized_marker(normalized, _HR_UNSUPPORTED_REPORT_MARKERS) or any(
            marker in original for marker in _HR_UNSUPPORTED_REPORT_MARKERS_JA
        ):
            return ReportCapability(
                status="unsupported",
                domain="hr",
                reason=(
                    "HR Report chỉ có snapshot danh bạ dạng aggregate; lương, KPI "
                    "cá nhân, tuyển dụng, chấm công hoặc hồ sơ nhân viên cần nguồn "
                    "HRIS chưa được kết nối."
                ),
            )
        if (
            _has_multiple_periods(question)
            or _has_invalid_explicit_date(question)
            or _has_invalid_explicit_month(question)
            or _has_normalized_marker(normalized, _UNSUPPORTED_PERIOD_MARKERS)
            or any(marker in original for marker in _UNSUPPORTED_PERIOD_MARKERS_JA)
            or _has_normalized_marker(normalized, _DYNAMIC_REPORT_MARKERS)
            or any(marker in original for marker in _DYNAMIC_REPORT_MARKERS_JA)
        ):
            return ReportCapability(
                status="unsupported",
                domain="hr",
                reason=(
                    "HR Report hiện chỉ hỗ trợ tổng quan danh bạ nhân sự hiện tại; "
                    "báo cáo theo kỳ, so sánh hoặc bộ lọc tùy biến chưa được hỗ trợ."
                ),
            )
        return ReportCapability(status="supported", shape="hr_executive", domain="hr")
    is_wms_context = any(
        marker in normalized for marker in _WMS_CONTEXT_MARKERS
    ) or any(marker in original for marker in _WMS_CONTEXT_MARKERS_JA)
    if is_wms_context:
        if (
            _has_multiple_periods(question)
            or _has_invalid_explicit_date(question)
            or _has_invalid_explicit_month(question)
            or _has_normalized_marker(normalized, _UNSUPPORTED_PERIOD_MARKERS)
            or any(marker in original for marker in _UNSUPPORTED_PERIOD_MARKERS_JA)
            or _has_normalized_marker(normalized, _DYNAMIC_REPORT_MARKERS)
            or any(marker in original for marker in _DYNAMIC_REPORT_MARKERS_JA)
            or _has_normalized_marker(normalized, _WMS_UNSUPPORTED_REPORT_MARKERS)
            or any(
                marker in original
                for marker in _WMS_UNSUPPORTED_REPORT_MARKERS_JA
            )
            or _ENTITY_FILTER_PATTERN.search(original)
        ):
            return ReportCapability(
                status="unsupported",
                domain="wms",
                reason=(
                    "WMS contract v4 hiện chỉ hỗ trợ báo cáo tổng quan current "
                    "balance; yêu cầu KPI, kỳ, so sánh hoặc bộ lọc tùy biến đang "
                    "bị khóa do chưa đủ data contract."
                ),
            )
        return ReportCapability(status="supported", shape="wms_executive", domain="wms")
    if not _has_domain_context(question, normalized):
        # Khẩu lệnh đúng nhưng không nêu lĩnh vực ("Tạo báo cáo"). Không mặc định
        # sang MES, vì đoán sai lĩnh vực nguy hiểm hơn việc hỏi lại một câu.
        return ReportCapability(
            status="unsupported",
            domain="",
            reason=(
                "Yêu cầu chưa nêu lĩnh vực báo cáo. Hãy nói rõ HR/nhân sự, "
                "chất lượng-lỗi MES, hoặc tồn kho WMS."
            ),
        )
    if not is_mes_report_request(question):
        return ReportCapability(
            status="unsupported",
            domain="mes",
            reason="Report Agent hiện chỉ hỗ trợ báo cáo chất lượng/lỗi MES.",
        )
    if any(marker in normalized for marker in _NON_ERROR_REPORT_MARKERS) or any(
        marker in original for marker in _NON_ERROR_REPORT_MARKERS_JA
    ) or not _has_mes_error_scope(question, normalized):
        return ReportCapability(
            status="unsupported",
            domain="mes",
            reason=(
                "MES Report hiện chỉ hỗ trợ tổng quan chất lượng/lỗi; sản lượng, "
                "OEE, ca sản xuất hoặc tình hình sản xuất chung chưa có data contract."
            ),
        )
    if _has_multiple_periods(question):
        return ReportCapability(
            status="unsupported",
            domain="mes",
            reason="Report Agent chưa hỗ trợ so sánh hoặc tổng hợp nhiều kỳ.",
        )
    if _has_invalid_explicit_date(question):
        return ReportCapability(
            status="unsupported",
            domain="mes",
            reason="Ngày trong yêu cầu báo cáo không hợp lệ.",
        )
    if _has_invalid_explicit_month(question):
        return ReportCapability(
            status="unsupported",
            domain="mes",
            reason="Tháng trong yêu cầu báo cáo không hợp lệ.",
        )
    if _has_normalized_marker(normalized, _UNSUPPORTED_PERIOD_MARKERS) or any(
        marker in original for marker in _UNSUPPORTED_PERIOD_MARKERS_JA
    ):
        return ReportCapability(
            status="unsupported",
            domain="mes",
            reason="Kỳ báo cáo này chưa thuộc các mẫu đã chuẩn bị.",
        )
    if _has_normalized_marker(normalized, _DYNAMIC_REPORT_MARKERS) or any(
        marker in original for marker in _DYNAMIC_REPORT_MARKERS_JA
    ):
        return ReportCapability(
            status="unsupported",
            domain="mes",
            reason="Yêu cầu cần lập kế hoạch báo cáo tùy biến chưa được hỗ trợ.",
        )
    if _ENTITY_FILTER_PATTERN.search(original):
        return ReportCapability(
            status="unsupported",
            domain="mes",
            reason="Báo cáo lọc theo đối tượng cụ thể chưa thuộc mẫu đã chuẩn bị.",
        )
    shape = "top_errors" if is_top_errors_report_request(question) else "overview"
    return ReportCapability(status="supported", shape=shape, domain="mes")


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
