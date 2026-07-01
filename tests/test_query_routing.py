import sqlite3
from pathlib import Path

import pytest

from src.auth.employee_directory import EmployeeDirectory
from src.integrations.mes_query_service import MesQueryService


@pytest.fixture
def employee_directory(tmp_path: Path) -> EmployeeDirectory:
    db_path = tmp_path / "employee_directory.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE employees (
                employee_id TEXT PRIMARY KEY,
                full_name TEXT,
                gender TEXT,
                position TEXT,
                department TEXT,
                greeting TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO employees (
                employee_id, full_name, gender, position, department, greeting
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("000001", "Nguyễn Minh Tiến", "Nam", "Trưởng phòng", "AGV", ""),
                ("000002", "Trần Đức Hải", "Nam", "Phó phòng", "AGV", ""),
                ("000003", "Vũ Minh Hoàng", "Nam", "Kỹ sư", "AI", ""),
            ],
        )
    return EmployeeDirectory(db_path)


def test_hr_structured_question_uses_employee_database(employee_directory):
    assert (
        employee_directory.structured_answer_for_question(
            "Công ty MKAC có bao nhiêu thành viên?",
            language="vi",
        )
        == "Theo danh bạ nhân sự, Meiko Automation hiện có 3 thành viên."
    )
    assert (
        employee_directory.structured_answer_for_question(
            "Ai là trưởng phòng AGV?",
            language="vi",
        )
        == "Thông tin trưởng/phó phòng là:\n- AGV: Nguyễn Minh Tiến (Trưởng phòng); Trần Đức Hải (Phó phòng)"
    )
    assert (
        employee_directory.structured_answer_for_question(
            "phòng ai có bao nhiêu người?",
            language="vi",
        )
        == "Phòng AI hiện có 1 người."
    )
    assert (
        employee_directory.structured_answer_for_question(
            "Phòng nào có nhiều nhân sự nhất?",
            language="vi",
        )
        == "Phòng ban có nhiều nhân sự nhất là AGV (2 người)."
    )


def test_hr_policy_question_falls_back_to_rag(employee_directory):
    assert (
        employee_directory.structured_answer_for_question(
            "Quy định làm thêm giờ ở MKAC như thế nào?",
            language="vi",
        )
        is None
    )


@pytest.mark.parametrize(
    "question",
    [
        "Liệt kê 5 lot nhiều lỗi nhất",
        "Mã hàng 3736-0008 có tổng bao nhiêu lỗi?",
        "Mã lỗi 0002 là gì?",
    ],
)
def test_mes_data_questions_can_use_sql_agent(question):
    assert MesQueryService.should_use_sql_agent(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "MES là gì?",
        "Quy trình ghi nhận lỗi sản xuất như thế nào?",
    ],
)
def test_mes_general_questions_skip_sql_agent(question):
    assert MesQueryService.should_use_sql_agent(question) is False
