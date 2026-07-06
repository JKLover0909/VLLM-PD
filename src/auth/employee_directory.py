"""Tra cứu danh bạ nhân viên MKAC theo mã nhân viên.

Lớp ``EmployeeDirectory`` chịu trách nhiệm truy cập SQLite và điều phối câu trả
lời. Các hàm phụ trợ thuần túy đã tách sang module chuyên biệt để dễ phát triển:

* ``employee_intent``  – chuẩn hóa văn bản và nhận diện ý định câu hỏi nhân sự.
* ``employee_answers`` – định dạng câu trả lời VI/JA từ dữ liệu đã truy vấn.

Để giữ nguyên API nội bộ (``self._question_requests_*`` / ``self._format_*``),
lớp gắn lại các hàm module thành ``staticmethod``.
"""

import re
import sqlite3
from pathlib import Path
from typing import Any, Optional, TypedDict

from src.auth import employee_answers, employee_intent
from src.auth.employee_intent import normalize_text

EMPLOYEE_ID_PATTERN = re.compile(r"^\d{6}$")
GUEST_EMPLOYEE_ID = "000000"


class EmployeeRecord(TypedDict):
    id: str
    name: str
    gender: str
    position: str
    department: str
    birth_date: str
    marital_status: str
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
                "birth_date": "",
                "marital_status": "",
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
                SELECT employee_id, full_name, gender, position, department,
                       birth_date, marital_status, greeting
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
            "birth_date": row[5] or "",
            "marital_status": row[6] or "",
            "greeting": row[7] or "",
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
        conversation_context: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Answer structured HR questions directly from the employee database."""
        if not self.db_path.is_file():
            return None

        normalized_question = normalize_text(question)
        departments = self._mentioned_departments(question, current_department)
        # Câu nối tiếp kiểu "liệt kê thành viên phòng này" không nêu tên phòng —
        # lấy phòng từ lượt hội thoại gần nhất có nhắc tới một phòng ban.
        if (
            not departments
            and conversation_context
            and self._question_references_prior_department(normalized_question)
        ):
            departments = self.departments_from_context(conversation_context)

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
        has_japanese_person_marker = bool(
            re.search(
                r"(部署|部門|所属|役職|職位|社員番号|従業員番号|どの|どこ|誰|同じ)",
                question or "",
            )
        )
        if (
            not self._question_requests_person_identity(normalized_question)
            and not has_japanese_person_marker
        ):
            return []
        return self.people_context_for_text(question)

    def people_context_for_text(self, text: str) -> list[dict[str, Any]]:
        """Return employee profiles explicitly mentioned in any text snippet."""
        if not self.db_path.is_file():
            return []

        normalized_text = normalize_text(text)
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
            id_mentioned = bool(
                re.search(rf"(?<!\d){re.escape(employee_id)}(?!\d)", text or "")
            )
            name_mentioned = bool(normalized_name and normalized_name in normalized_text)
            if not (id_mentioned or name_mentioned):
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

    def departments_from_context(
        self,
        conversation_context: list[dict[str, Any]],
    ) -> list[str]:
        """Resolve department references ("phòng này") from recent chat turns.

        Scan newest-first so "phòng này" binds to the department discussed most
        recently, not an older one earlier in the conversation.
        """
        for item in reversed(conversation_context or []):
            content = str(item.get("content") or "")
            if not content.strip():
                continue
            departments = self._mentioned_departments(content)
            if departments:
                return departments
        return []

    def _format_company_headcount(self, language: str) -> str:
        return employee_answers.format_company_headcount(self.count(), language)

    def _format_department_catalog(self, language: str) -> str:
        return employee_answers.format_department_catalog(
            self.department_summaries(),
            language,
        )

    def _format_largest_departments(self, language: str) -> Optional[str]:
        return employee_answers.format_largest_departments(
            self.department_summaries(),
            language,
        )

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

    def count(self) -> int:
        if not self.db_path.is_file():
            return 0
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM employees").fetchone()
        return int(row[0] or 0)

    # ──────────────────────────────────────────────────────────────────
    # Delegators giữ nguyên API nội bộ. Logic thật nằm ở các module
    # employee_intent (nhận diện ý định) và employee_answers (định dạng).
    # ──────────────────────────────────────────────────────────────────

    # employee_intent
    _department_name_in_question = staticmethod(employee_intent.department_name_in_question)
    _question_targets_current_department = staticmethod(
        employee_intent.question_targets_current_department
    )
    _question_requests_department_people = staticmethod(
        employee_intent.question_requests_department_people
    )
    _question_requests_department_count = staticmethod(
        employee_intent.question_requests_department_count
    )
    _question_requests_company_headcount = staticmethod(
        employee_intent.question_requests_company_headcount
    )
    _question_requests_department_catalog = staticmethod(
        employee_intent.question_requests_department_catalog
    )
    _question_requests_largest_department = staticmethod(
        employee_intent.question_requests_largest_department
    )
    _question_requests_department_leadership = staticmethod(
        employee_intent.question_requests_department_leadership
    )
    _question_requests_department_roster = staticmethod(
        employee_intent.question_requests_department_roster
    )
    _question_requests_person_identity = staticmethod(
        employee_intent.question_requests_person_identity
    )
    _question_references_prior_department = staticmethod(
        employee_intent.question_references_prior_department
    )

    # employee_answers
    _format_department_counts = staticmethod(employee_answers.format_department_counts)
    _format_department_leadership = staticmethod(
        employee_answers.format_department_leadership
    )
    _format_department_rosters = staticmethod(employee_answers.format_department_rosters)
    _format_people_profiles = staticmethod(employee_answers.format_people_profiles)
