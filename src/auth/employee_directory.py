"""Tra cứu danh bạ nhân viên MKAC theo mã nhân viên."""

import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Optional, TypedDict


EMPLOYEE_ID_PATTERN = re.compile(r"^\d{6}$")
GUEST_EMPLOYEE_ID = "000000"


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    without_marks = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", without_marks.lower()).strip()


class EmployeeRecord(TypedDict):
    id: str
    name: str
    gender: str
    position: str
    department: str
    greeting: str
    department_size: int
    department_heads: list[str]
    department_deputies: list[str]


class EmployeeDirectory:
    """SQLite-backed employee directory used for the MKAC access gate."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    def lookup(self, employee_id: str) -> Optional[EmployeeRecord]:
        normalized_id = employee_id.strip()
        if not EMPLOYEE_ID_PATTERN.fullmatch(normalized_id):
            return None
        if normalized_id == GUEST_EMPLOYEE_ID:
            return {
                "id": GUEST_EMPLOYEE_ID,
                "name": "Guest",
                "gender": "",
                "position": "",
                "department": "",
                "greeting": "Chào mừng đến với hệ thống Meibook,",
                "department_size": 0,
                "department_heads": [],
                "department_deputies": [],
            }
        if not self.db_path.is_file():
            return None

        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT employee_id, full_name, gender, position, department, greeting
                FROM employees
                WHERE employee_id = ?
                """,
                (normalized_id,),
            ).fetchone()

        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "gender": row[2] or "",
            "position": row[3] or "",
            "department": row[4] or "",
            "greeting": row[5] or "",
            "department_size": 0,
            "department_heads": [],
            "department_deputies": [],
        }

    def profile(self, employee_id: str) -> Optional[EmployeeRecord]:
        employee = self.lookup(employee_id)
        if not employee:
            return None
        department = employee.get("department", "")
        if not department or not self.db_path.is_file():
            return employee

        with sqlite3.connect(self.db_path) as connection:
            department_size = connection.execute(
                """
                SELECT COUNT(*)
                FROM employees
                WHERE department = ?
                """,
                (department,),
            ).fetchone()[0]
            managers = connection.execute(
                """
                SELECT full_name, position
                FROM employees
                WHERE department = ?
                  AND (position LIKE '%Trưởng phòng%' OR position LIKE '%Phó phòng%')
                ORDER BY
                  CASE
                    WHEN position LIKE '%Trưởng phòng%' THEN 0
                    WHEN position LIKE '%Phó phòng%' THEN 1
                    ELSE 2
                  END,
                  employee_id
                """,
                (department,),
            ).fetchall()

        employee["department_size"] = int(department_size or 0)
        employee["department_heads"] = [
            f"{name} ({position})"
            for name, position in managers
            if "Trưởng phòng" in (position or "")
        ]
        employee["department_deputies"] = [
            f"{name} ({position})"
            for name, position in managers
            if "Phó phòng" in (position or "")
        ]
        return employee

    def context_for(self, employee_id: str) -> Optional[dict[str, Any]]:
        employee = self.profile(employee_id)
        if not employee:
            return None
        return dict(employee)

    def department_context_for_question(
        self,
        question: str,
        current_department: str = "",
    ) -> list[dict[str, Any]]:
        """Return roster context for departments explicitly mentioned in a question."""
        if not self._question_requests_department_people(normalize_text(question)):
            return []
        departments = self._mentioned_departments(question, current_department)
        return [self.department_profile(department) for department in departments]

    def department_counts_for_question(
        self,
        question: str,
        current_department: str = "",
    ) -> list[dict[str, Any]]:
        """Return department sizes for direct headcount questions."""
        normalized_question = normalize_text(question)
        if not self._question_requests_department_count(normalized_question, question):
            return []
        departments = self._mentioned_departments(question, current_department)
        return [self.department_profile(department) for department in departments]

    def structured_answer_for_question(
        self,
        question: str,
        current_department: str = "",
        language: str = "vi",
    ) -> Optional[str]:
        """Answer structured HR questions directly from the employee database."""
        if not self.db_path.is_file():
            return None

        normalized_question = normalize_text(question)
        departments = self._mentioned_departments(question, current_department)

        if self._question_requests_company_headcount(normalized_question, question):
            return self._format_company_headcount(language)

        if self._question_requests_department_catalog(normalized_question, question):
            return self._format_department_catalog(language)

        if self._question_requests_largest_department(normalized_question, question):
            return self._format_largest_departments(language)

        if self._question_requests_department_count(normalized_question, question):
            if not departments:
                return None
            profiles = [self.department_profile(department) for department in departments]
            return self._format_department_counts(profiles, language)

        if self._question_requests_department_leadership(normalized_question, question):
            if not departments:
                return None
            profiles = [self.department_profile(department) for department in departments]
            return self._format_department_leadership(profiles, language)

        if self._question_requests_department_roster(normalized_question, question):
            if not departments:
                return None
            profiles = [self.department_profile(department) for department in departments]
            return self._format_department_rosters(profiles, language)

        people = self.people_context_for_question(question)
        if people:
            return self._format_people_profiles(people, language)
        return None

    def people_context_for_question(self, question: str) -> list[dict[str, Any]]:
        """Return employee profiles when a full employee name appears in a question."""
        if not self.db_path.is_file():
            return []

        normalized_question = normalize_text(question)
        if not self._question_requests_person_identity(normalized_question):
            return []

        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT employee_id, full_name
                FROM employees
                ORDER BY LENGTH(full_name) DESC, employee_id
                """
            ).fetchall()

        people: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for employee_id, full_name in rows:
            normalized_name = normalize_text(full_name)
            if not normalized_name or normalized_name not in normalized_question:
                continue
            profile = self.profile(employee_id)
            if profile and profile["id"] not in seen_ids:
                people.append(dict(profile))
                seen_ids.add(profile["id"])
        return people

    def department_profile(self, department: str) -> dict[str, Any]:
        if not self.db_path.is_file():
            return {
                "department": department,
                "size": 0,
                "heads": [],
                "deputies": [],
                "members": [],
            }

        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT employee_id, full_name, position
                FROM employees
                WHERE department = ?
                ORDER BY
                  CASE
                    WHEN position LIKE '%Trưởng phòng%' THEN 0
                    WHEN position LIKE '%Phó phòng%' THEN 1
                    ELSE 2
                  END,
                  employee_id
                """,
                (department,),
            ).fetchall()

        members = [
            {"id": employee_id, "name": name, "position": position or ""}
            for employee_id, name, position in rows
        ]
        return {
            "department": department,
            "size": len(members),
            "heads": [
                f"{member['name']} ({member['position']})"
                for member in members
                if "Trưởng phòng" in member["position"]
            ],
            "deputies": [
                f"{member['name']} ({member['position']})"
                for member in members
                if "Phó phòng" in member["position"]
            ],
            "members": members,
        }

    def department_summaries(self) -> list[dict[str, Any]]:
        if not self.db_path.is_file():
            return []
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT department, COUNT(*) AS size
                FROM employees
                WHERE department IS NOT NULL AND department != ''
                GROUP BY department
                ORDER BY size DESC, department
                """
            ).fetchall()
        return [{"department": department, "size": int(size or 0)} for department, size in rows]

    def _format_company_headcount(self, language: str) -> str:
        total = self.count()
        if language == "ja":
            return f"社員名簿上、Meiko Automationには現在{total}人が登録されています。"
        return f"Theo danh bạ nhân sự, Meiko Automation hiện có {total} thành viên."

    def _format_department_catalog(self, language: str) -> str:
        summaries = sorted(
            self.department_summaries(),
            key=lambda item: normalize_text(item["department"]),
        )
        total = len(summaries)
        if language == "ja":
            names = "、".join(item["department"] for item in summaries)
            return f"Meiko Automationには合計{total}部門があります。部門は{names}です。"
        names = ", ".join(item["department"] for item in summaries)
        return f"Meiko Automation hiện có tổng cộng {total} phòng ban, gồm: {names}."

    def _format_largest_departments(self, language: str) -> Optional[str]:
        summaries = self.department_summaries()
        if not summaries:
            return None
        max_size = max(item["size"] for item in summaries)
        largest = [item for item in summaries if item["size"] == max_size]
        if language == "ja":
            departments = "、".join(
                f"{item['department']}（{item['size']}人）" for item in largest
            )
            return f"人数が最も多い部門は{departments}です。"
        departments = ", ".join(
            f"{item['department']} ({item['size']} người)" for item in largest
        )
        return f"Phòng ban có nhiều nhân sự nhất là {departments}."

    @staticmethod
    def _format_department_counts(
        profiles: list[dict[str, Any]],
        language: str,
    ) -> str:
        if language == "ja":
            if len(profiles) == 1:
                profile = profiles[0]
                return f"{profile['department']}部門には{profile['size']}人が在籍しています。"
            lines = [
                f"- {profile['department']}: {profile['size']}人"
                for profile in profiles
            ]
            return "該当する部門の人数は以下の通りです。\n" + "\n".join(lines)

        if len(profiles) == 1:
            profile = profiles[0]
            return f"Phòng {profile['department']} hiện có {profile['size']} người."
        lines = [
            f"- {profile['department']}: {profile['size']} người"
            for profile in profiles
        ]
        return "Số nhân sự của các phòng ban được hỏi là:\n" + "\n".join(lines)

    @staticmethod
    def _format_department_leadership(
        profiles: list[dict[str, Any]],
        language: str,
    ) -> str:
        lines: list[str] = []
        for profile in profiles:
            leaders = profile["heads"] + profile["deputies"]
            if language == "ja":
                if leaders:
                    lines.append(f"{profile['department']}: " + "、".join(leaders))
                else:
                    lines.append(f"{profile['department']}: 管理者情報は登録されていません。")
            else:
                if leaders:
                    lines.append(f"- {profile['department']}: " + "; ".join(leaders))
                else:
                    lines.append(
                        f"- {profile['department']}: chưa có thông tin trưởng/phó phòng."
                    )
        if language == "ja":
            return "該当部門の管理者情報は以下の通りです。\n" + "\n".join(lines)
        return "Thông tin trưởng/phó phòng là:\n" + "\n".join(lines)

    @staticmethod
    def _format_department_rosters(
        profiles: list[dict[str, Any]],
        language: str,
    ) -> str:
        sections: list[str] = []
        for profile in profiles:
            members = profile["members"]
            if language == "ja":
                lines = [
                    f"{member['name']}（{member['position'] or '未登録'}）"
                    for member in members
                ]
                sections.append(
                    f"{profile['department']}部門（{profile['size']}人）:\n"
                    + "\n".join(lines)
                )
            else:
                lines = [
                    f"- {member['name']} ({member['position'] or 'chưa có chức danh'})"
                    for member in members
                ]
                sections.append(
                    f"Phòng {profile['department']} ({profile['size']} người):\n"
                    + "\n".join(lines)
                )
        return "\n\n".join(sections)

    @staticmethod
    def _format_people_profiles(
        people: list[dict[str, Any]],
        language: str,
    ) -> str:
        if language == "ja":
            lines = [
                (
                    f"- {person['name']}（社員番号: {person['id']}、"
                    f"部門: {person.get('department') or '未登録'}、"
                    f"役職: {person.get('position') or '未登録'}）"
                )
                for person in people
            ]
            return "該当する社員情報は以下の通りです。\n" + "\n".join(lines)
        lines = [
            (
                f"- {person['name']} - mã nhân viên {person['id']}, "
                f"phòng {person.get('department') or 'chưa có'}, "
                f"chức danh {person.get('position') or 'chưa có'}"
            )
            for person in people
        ]
        return "Thông tin nhân sự tìm thấy:\n" + "\n".join(lines)

    def _mentioned_departments(
        self,
        question: str,
        current_department: str = "",
    ) -> list[str]:
        if not self.db_path.is_file():
            return []

        normalized_question = normalize_text(question)
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT department
                FROM employees
                WHERE department IS NOT NULL AND department != ''
                ORDER BY LENGTH(department) DESC, department
                """
            ).fetchall()

        matches: list[str] = []
        for (department,) in rows:
            normalized_department = normalize_text(department)
            if not normalized_department:
                continue
            if self._department_name_in_question(
                question,
                normalized_question,
                department,
                normalized_department,
            ):
                matches.append(department)

        if matches:
            return list(dict.fromkeys(matches))

        if current_department and self._question_targets_current_department(
            normalized_question
        ):
            return [current_department]
        return []

    @staticmethod
    def _department_name_in_question(
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

    @staticmethod
    def _question_targets_current_department(normalized_question: str) -> bool:
        markers = {
            "bo phan cua toi",
            "phong ban cua toi",
            "phong cua toi",
            "bo phan cua minh",
            "phong ban cua minh",
            "phong cua minh",
        }
        return any(marker in normalized_question for marker in markers)

    @staticmethod
    def _question_requests_department_people(normalized_question: str) -> bool:
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

    @staticmethod
    def _question_requests_department_count(
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

    @staticmethod
    def _question_requests_company_headcount(
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

    @staticmethod
    def _question_requests_department_catalog(
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

    @staticmethod
    def _question_requests_largest_department(
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

    @staticmethod
    def _question_requests_department_leadership(
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

    @staticmethod
    def _question_requests_department_roster(
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

    @staticmethod
    def _question_requests_person_identity(normalized_question: str) -> bool:
        keywords = {
            "la ai",
            "ai la",
            "thong tin",
            "lam bo phan nao",
            "lam phong nao",
            "thuoc bo phan nao",
            "thuoc phong ban nao",
            "chuc danh",
            "vi tri",
            "ma nhan vien",
            "who is",
            "which department",
            "position",
        }
        return any(keyword in normalized_question for keyword in keywords)

    def count(self) -> int:
        if not self.db_path.is_file():
            return 0
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM employees").fetchone()
        return int(row[0] or 0)
