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
            return bool(
                re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(department)}(?![A-Za-z0-9])",
                    question,
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
