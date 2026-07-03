"""Chuẩn hóa văn bản và nhận diện ý định cho câu hỏi danh bạ nhân sự MKAC.

Các hàm ở đây thuần túy trên chuỗi câu hỏi (đã/chưa chuẩn hóa), không truy cập
database. Tách riêng để khi cần thêm/điều chỉnh cách nhận diện một loại câu hỏi
nhân sự thì chỉ sửa ở một chỗ. Giữ nguyên hành vi so với bản gốc.
"""

import re
import unicodedata


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    without_marks = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", without_marks.lower()).strip()


def department_name_in_question(
    question: str,
    normalized_question: str,
    department: str,
    normalized_department: str,
) -> bool:
    if len(normalized_department) <= 2:
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(department)}(?![A-Za-z0-9])",
            question,
        ):
            return True
        return bool(
            re.search(
                rf"\b(?:phong|bo phan|team|department|dept)\s+{re.escape(normalized_department)}\b",
                normalized_question,
            )
            or re.search(
                rf"\b{re.escape(normalized_department)}\s+(?:phong|bo phan|team|department|dept)\b",
                normalized_question,
            )
        )
    if " " in normalized_department:
        return normalized_department in normalized_question
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_department)}(?![a-z0-9])",
            normalized_question,
        )
    )


def question_targets_current_department(normalized_question: str) -> bool:
    markers = {
        "bo phan cua toi",
        "phong ban cua toi",
        "phong cua toi",
        "bo phan cua minh",
        "phong ban cua minh",
        "phong cua minh",
    }
    return any(marker in normalized_question for marker in markers)


def question_requests_department_people(normalized_question: str) -> bool:
    keywords = {
        "bao nhieu nguoi",
        "so nguoi",
        "gom nhung ai",
        "co nhung ai",
        "la nhung ai",
        "liet ke",
        "danh sach",
        "thanh vien",
        "nhan vien",
        "truong phong",
        "pho phong",
        "ai la",
        "members",
        "people",
        "employees",
        "manager",
    }
    return any(keyword in normalized_question for keyword in keywords)


def question_requests_department_count(
    normalized_question: str,
    original_question: str,
) -> bool:
    keywords = {
        "bao nhieu nguoi",
        "bao nhieu nhan su",
        "may nguoi",
        "so nguoi",
        "so luong nguoi",
        "so nhan su",
        "tong so nguoi",
        "tong nhan su",
        "headcount",
        "how many people",
        "how many employees",
    }
    if any(keyword in normalized_question for keyword in keywords):
        return True
    return bool(re.search(r"(何人|人数|社員数|従業員数)", original_question or ""))


def question_requests_company_headcount(
    normalized_question: str,
    original_question: str,
) -> bool:
    person_markers = {
        "bao nhieu thanh vien",
        "bao nhieu nhan vien",
        "bao nhieu nhan su",
        "bao nhieu nguoi",
        "tong so thanh vien",
        "tong so nhan vien",
        "tong so nhan su",
        "tong so nguoi",
        "so luong thanh vien",
        "so luong nhan vien",
        "so luong nhan su",
        "company headcount",
        "how many employees",
        "how many people",
    }
    company_markers = {
        "cong ty",
        "mkac",
        "meiko automation",
        "toan cong ty",
        "company",
    }
    if any(marker in normalized_question for marker in person_markers) and (
        any(marker in normalized_question for marker in company_markers)
        or not any(
            marker in normalized_question
            for marker in ("phong", "phong ban", "bo phan", "department", "dept")
        )
    ):
        return True
    return bool(
        re.search(r"(会社|全社|Meiko|MKAC).*(何人|人数|社員数|従業員数)", original_question or "")
    )


def question_requests_department_catalog(
    normalized_question: str,
    original_question: str,
) -> bool:
    count_markers = {
        "bao nhieu phong ban",
        "bao nhieu bo phan",
        "co bao nhieu phong",
        "co bao nhieu bo phan",
        "tong so phong ban",
        "tong so bo phan",
        "number of departments",
        "how many departments",
    }
    list_markers = {
        "liet ke phong ban",
        "liet ke cac phong ban",
        "danh sach phong ban",
        "co nhung phong ban nao",
        "gom cac phong ban nao",
        "departments list",
        "list departments",
    }
    if any(marker in normalized_question for marker in count_markers | list_markers):
        return True
    return bool(re.search(r"(部門|部署).*(一覧|何個|いくつ|全部|リスト)", original_question or ""))


def question_requests_largest_department(
    normalized_question: str,
    original_question: str,
) -> bool:
    markers = {
        "phong nao dong nguoi nhat",
        "bo phan nao dong nguoi nhat",
        "phong nao nhieu nguoi nhat",
        "bo phan nao nhieu nguoi nhat",
        "phong nao nhieu nhan su nhat",
        "bo phan nao nhieu nhan su nhat",
        "phong ban lon nhat",
        "bo phan lon nhat",
        "largest department",
        "biggest department",
        "most employees",
        "highest headcount",
    }
    if any(marker in normalized_question for marker in markers):
        return True
    return bool(
        re.search(
            r"\b(?:phong|phong ban|bo phan).*(?:nhieu|dong|lon).*(?:nguoi|nhan su)?.*nhat\b",
            normalized_question,
        )
        or re.search(r"(人数|社員|従業員).*(最も|一番|最大|多い)", original_question or "")
    )


def question_requests_department_leadership(
    normalized_question: str,
    original_question: str,
) -> bool:
    markers = {
        "truong phong",
        "pho phong",
        "quan ly",
        "cap tren",
        "manager",
        "leader",
        "head of",
    }
    if any(marker in normalized_question for marker in markers):
        return True
    return bool(re.search(r"(部長|課長|管理者|リーダー|マネージャー)", original_question or ""))


def question_requests_department_roster(
    normalized_question: str,
    original_question: str,
) -> bool:
    markers = {
        "gom nhung ai",
        "co nhung ai",
        "la nhung ai",
        "liet ke nhan su",
        "danh sach nhan su",
        "danh sach nhan vien",
        "thanh vien",
        "members",
        "employees",
        "people in",
        "list people",
    }
    if any(marker in normalized_question for marker in markers):
        return True
    return bool(re.search(r"(メンバー|社員|従業員).*(一覧|誰|リスト)", original_question or ""))


def question_requests_person_identity(normalized_question: str) -> bool:
    keywords = {
        "la ai",
        "ai la",
        "thong tin",
        "lam bo phan nao",
        "lam phong nao",
        "lam o phong nao",
        "lam phong gi",
        "lam o bo phan nao",
        "lam bo phan gi",
        "thuoc bo phan nao",
        "thuoc phong ban nao",
        "chuc danh",
        "chuc vu",
        "vi tri",
        "vai tro",
        "lam vai tro gi",
        "ma nhan vien",
        # Bare "phong ban"/"bo phan" bắt được cách hỏi dùng "của" thay vì
        # "thuộc ... nào"/"làm ... nào" (vd "phòng ban của X là gì?").
        "phong ban",
        "bo phan",
        "who is",
        "which department",
        "position",
        "role",
    }
    return any(keyword in normalized_question for keyword in keywords)
