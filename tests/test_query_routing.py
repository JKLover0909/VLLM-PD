import sqlite3
from pathlib import Path

import pytest

from src.auth.employee_directory import EmployeeDirectory
from src.integrations.mes_query_service import MesQueryService
from src.rag.rag_pipeline import RAGPipeline


@pytest.fixture
def employee_directory(tmp_path: Path) -> EmployeeDirectory:
    db_path = tmp_path / "employee_directory.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE employees (
                employee_id TEXT PRIMARY KEY,
                full_name TEXT,
                company_email TEXT,
                gender TEXT,
                position TEXT,
                department TEXT,
                birth_date TEXT,
                marital_status TEXT,
                greeting TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO employees (
                employee_id, full_name, company_email, gender, position,
                department, birth_date, marital_status, greeting
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("000001", "Nguyễn Minh Tiến", "tien.nguyenminh@meiko.vn", "Nam", "Trưởng phòng", "AGV", "03/06/1989", "", ""),
                ("000002", "Trần Đức Hải", "hai.tranduc@meiko.vn", "Nam", "Phó phòng", "AGV", "03/02/1993", "", ""),
                ("000003", "Vũ Minh Hoàng", "hoang.vuminh@meiko.vn", "Nam", "Kỹ sư", "AI", "12/06/2002", "", ""),
                ("000004", "Vũ Minh Đức", "duc.vuminh@meiko.vn", "Nam", "Trưởng phòng", "R&D S", "30/09/1993", "", ""),
                ("000005", "Nguyễn Thị Thu Hương", "huong.nguyenthithu@meiko.vn", "Nữ", "Nhân viên", "Kho", "03/12/1992", "Đã kết hôn", ""),
            ],
        )
    return EmployeeDirectory(db_path)


def test_hr_structured_question_uses_employee_database(employee_directory):
    assert (
        employee_directory.structured_answer_for_question(
            "Công ty MKAC có bao nhiêu thành viên?",
            language="vi",
        )
        == "Theo danh bạ nhân sự, Meiko Automation hiện có 5 thành viên."
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


def test_hr_department_comparison_uses_recent_context(employee_directory):
    answer = employee_directory.structured_answer_for_question(
        "So với phòng AGV thì phòng nào đông hơn?",
        language="vi",
        conversation_context=[
            {
                "role": "assistant",
                "content": "Phòng AI hiện có 1 người.",
            }
        ],
    )

    assert answer is not None
    assert "AGV có 2 người" in answer
    assert "AI có 1 người" in answer
    assert "Phòng đông hơn là AGV" in answer


def test_hr_policy_question_falls_back_to_rag(employee_directory):
    assert (
        employee_directory.structured_answer_for_question(
            "Quy định làm thêm giờ ở MKAC như thế nào?",
            language="vi",
        )
        is None
    )


def test_hr_name_questions_match_lam_o_phong_and_lam_phong_gi(employee_directory):
    assert "R&D S" in (
        employee_directory.structured_answer_for_question(
            "Vũ Minh Đức làm ở phòng nào?",
            language="vi",
        )
        or ""
    )
    assert "Kho" in (
        employee_directory.structured_answer_for_question(
            "Nguyễn Thị Thu Hương làm phòng gì?",
            language="vi",
        )
        or ""
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


@pytest.mark.parametrize(
    "question",
    [
        "Kiểm tra lịch họp tuần này",
        "Xem tôi có lịch sinh nhật nào không?",
        "Kiểm tra xem tôi có rảnh từ 9 giờ đến 10 giờ sáng mai không.",
        "Tìm phòng họp còn trống từ 14 giờ đến 16 giờ chiều mai cho 8 người.",
        "Check my calendar events today",
        "来週の会議予定を確認してください。",
    ],
)
def test_calendar_read_questions_route_to_calendar(question):
    assert RAGPipeline._is_calendar_read_question(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Quy định đăng ký lịch làm việc trong tháng là gì?",
        "Tôi muốn đặt lịch họp ngày mai",
        "Tạo sự kiện Calendar lúc 9 giờ",
        "Cuộc họp được quy định trong nội quy thế nào?",
    ],
)
def test_non_read_calendar_questions_stay_out_of_calendar_tools(question):
    assert RAGPipeline._is_calendar_read_question(question) is False
