"""Company-email generation for the employee directory import."""

import sqlite3

from scripts.import_employee_directory import (
    build_company_email,
    resolve_company_emails,
    write_database,
)


def test_company_email_uses_given_name_then_family_and_middle_names():
    assert (
        build_company_email("000999", "Nguyễn Đình Sơn")
        == "son.nguyendinh@meiko.vn"
    )


def test_company_email_normalizes_vietnamese_diacritics_and_whitespace():
    assert (
        build_company_email("000999", "  Trần   Thị Ánh  ")
        == "anh.tranthi@meiko.vn"
    )


def test_verified_override_resolves_known_collision():
    employees = [
        {"employee_id": "000209", "full_name": "Nguyễn Văn Hùng"},
        {"employee_id": "000107", "full_name": "Nguyên Văn Hùng"},
    ]

    emails = resolve_company_emails(employees)

    assert emails["000107"] == "hung.nguyenvan@meiko.vn"
    assert emails["000209"] == "hung.nguyenvan1@meiko.vn"


def test_unverified_collision_keeps_lower_id_and_nulls_higher_id():
    employees = [
        {"employee_id": "000998", "full_name": "Nguyễn Văn Bình"},
        {"employee_id": "000997", "full_name": "Nguyên Văn Bình"},
    ]

    emails = resolve_company_emails(employees)

    assert emails["000997"] == "binh.nguyenvan@meiko.vn"
    assert emails["000998"] is None


def test_company_email_returns_none_for_empty_name():
    assert build_company_email("000999", "") is None


def test_write_database_persists_verified_collision_override(tmp_path):
    db_path = tmp_path / "employees.sqlite"
    base = {
        "gender": "",
        "position": "",
        "department": "QA",
        "birth_date": "",
        "marital_status": "",
        "greeting": "",
        "source_file": "test.pdf",
        "source_page": 1,
    }
    employees = [
        {**base, "employee_id": "000107", "full_name": "Nguyên Văn Hùng"},
        {**base, "employee_id": "000209", "full_name": "Nguyễn Văn Hùng"},
    ]

    write_database(db_path, employees)

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT employee_id, company_email FROM employees ORDER BY employee_id"
        ).fetchall()
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(employees)")
        }
    assert "company_email" in columns
    assert rows == [
        ("000107", "hung.nguyenvan@meiko.vn"),
        ("000209", "hung.nguyenvan1@meiko.vn"),
    ]
