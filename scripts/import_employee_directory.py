#!/usr/bin/env python3
"""Import MKAC employee IDs and names from PDF files into SQLite."""

import argparse
from collections import Counter
from html import escape
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EMPLOYEE_ID_PATTERN = re.compile(r"^000\d{3}$")
DATE_PATTERN = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
HEADER_VALUES = {
    "STT",
    "ID",
    "HỌ VÀ TÊN",
    "GIỚI TÍNH",
    "NĂM SINH",
    "CHỨC DANH",
    "PHÒNG BAN",
    "TÌNH TRẠNG",
    "HÔN NHÂN",
    "GHI CHÚ",
    "NAM",
    "NỮ",
}
FIXED_EMPLOYEES = [
    {
        "employee_id": "000001",
        "full_name": "Nguyễn Văn Thuận",
        "gender": "Nam",
        "position": "Giám đốc; Phó tổng giám đốc",
        "department": "Ban Giám đốc",
        "greeting": "Chào anh Nguyễn Văn Thuận, Giám đốc Meiko Automation",
        "source_file": "Thông tin cố định",
        "source_page": 0,
    }
]
FIXED_LEADERSHIP = {
    "director": "Nguyễn Văn Thuận",
    "deputy_general_director": "Nguyễn Văn Thuận",
    "general_director": "YUICHIRO NAYA",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--glob", default=None)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def valid_name(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    if normalized.upper() in HEADER_VALUES:
        return False
    if DATE_PATTERN.fullmatch(normalized):
        return False
    if any(char.isdigit() for char in normalized):
        return False
    return True


def extract_pdf(path: Path) -> list[dict]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required. Install project requirements before importing."
        ) from exc

    employees = []
    document = fitz.open(path)
    for page_number, page in enumerate(document, start=1):
        page_extracted = False
        tables = page.find_tables()
        for table in tables.tables:
            data = table.extract()
            if not data:
                continue
            header = [(value or "").replace("\n", " ").strip() for value in data[0]]
            if "ID" not in header or "HỌ VÀ TÊN" not in header:
                continue
            for row in data[1:]:
                values = {
                    header[index]: (row[index] or "").replace("\n", " ").strip()
                    for index in range(min(len(header), len(row)))
                }
                employee_id = values.get("ID", "")
                full_name = values.get("HỌ VÀ TÊN", "")
                if not EMPLOYEE_ID_PATTERN.fullmatch(employee_id):
                    continue
                if not valid_name(full_name):
                    continue
                employees.append(
                    {
                        "employee_id": employee_id,
                        "full_name": full_name,
                        "gender": values.get("GIỚI TÍNH", ""),
                        "position": values.get("CHỨC DANH", ""),
                        "department": normalize_department(values.get("PHÒNG BAN", "")),
                        "greeting": build_greeting(
                            full_name,
                            values.get("GIỚI TÍNH", ""),
                            values.get("CHỨC DANH", ""),
                            normalize_department(values.get("PHÒNG BAN", "")),
                        ),
                        "source_file": path.name,
                        "source_page": page_number,
                    }
                )
                page_extracted = True
        if page_extracted:
            continue

        lines = [line.strip() for line in page.get_text().splitlines() if line.strip()]
        for index, line in enumerate(lines[:-1]):
            if not EMPLOYEE_ID_PATTERN.fullmatch(line):
                continue
            full_name = lines[index + 1].strip()
            if not valid_name(full_name):
                continue
            employees.append(
                {
                    "employee_id": line,
                    "full_name": full_name,
                    "gender": "",
                    "position": "",
                    "department": "",
                    "greeting": "",
                    "source_file": path.name,
                    "source_page": page_number,
                }
            )
    return employees


def normalize_department(value: str) -> str:
    normalized = re.sub(r"\s+", " ", (value or "").strip())
    if normalized.lower() == "gia công":
        return "Gia công"
    return normalized or "Chưa rõ"


def honorific(gender: str) -> str:
    normalized = (gender or "").strip().lower()
    if normalized == "nữ":
        return "chị"
    return "anh"


def build_greeting(
    full_name: str,
    gender: str,
    position: str,
    department: str,
) -> str:
    normalized_position = (position or "").strip()
    if normalized_position not in {"Trưởng phòng", "Phó phòng"}:
        return ""
    suffix = f" {department}" if department else ""
    return f"Chào {honorific(gender)} {full_name}, {normalized_position}{suffix}"


def collect_employees(source_dir: Path, pattern: str) -> list[dict]:
    if not source_dir.is_dir():
        raise SystemExit(f"Employee source directory not found: {source_dir}")

    employees = []
    for path in sorted(source_dir.glob(pattern)):
        extracted = extract_pdf(path)
        print(f"{path.name}: {len(extracted)} employees")
        employees.extend(extracted)
    for fixed in FIXED_EMPLOYEES:
        if not any(item["employee_id"] == fixed["employee_id"] for item in employees):
            employees.append(dict(fixed))
    return employees


def write_summary(summary_path: Path, employees: list[dict], source_pattern: str) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    departments = Counter(
        normalize_department(item.get("department", ""))
        for item in employees
    )
    total = len({item["employee_id"] for item in employees})
    pdf_total = sum(1 for item in employees if item.get("source_page", 0) > 0)
    department_rows = "\n".join(
        f"<tr><td>{escape(department)}</td><td>{count}</td></tr>"
        for department, count in sorted(departments.items())
    )
    summary_path.write_text(
        f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Thông tin nhân sự và lãnh đạo MKAC</title>
</head>
<body>
  <h1>Thông tin nhân sự và lãnh đạo MKAC</h1>
  <p>Tài liệu này là bản tóm tắt nội bộ được tạo từ danh sách nhân sự MKAC và các thông tin lãnh đạo cố định.</p>

  <h2>Thống kê nhân sự</h2>
  <ul>
    <li>Số nhân sự trong file danh sách khám sức khỏe 2026: {pdf_total} người.</li>
    <li>Số nhân sự có mã ID đang được hệ thống ghi nhận: {total} người.</li>
    <li>Số phòng ban/nhóm trong thống kê theo mã ID: {len(departments)}.</li>
    <li>Nguồn chính: {escape(source_pattern)}.</li>
  </ul>

  <h2>Thống kê theo phòng ban</h2>
  <table>
    <thead><tr><th>Phòng ban/nhóm</th><th>Số người</th></tr></thead>
    <tbody>
      {department_rows}
    </tbody>
  </table>

  <h2>Thông tin lãnh đạo cố định</h2>
  <ul>
    <li>Giám đốc hiện tại: {escape(FIXED_LEADERSHIP["director"])}.</li>
    <li>Phó tổng giám đốc: {escape(FIXED_LEADERSHIP["deputy_general_director"])}.</li>
    <li>Tổng giám đốc: {escape(FIXED_LEADERSHIP["general_director"])}.</li>
    <li>Mã nhân viên 000001 là {escape(FIXED_EMPLOYEES[0]["full_name"])}, Giám đốc Meiko Automation.</li>
  </ul>

  <p>Lưu ý: {escape(FIXED_LEADERSHIP["general_director"])} được lưu như thông tin lãnh đạo cố định, chưa có mã nhân viên trong danh sách này nên không cộng vào thống kê theo mã ID nếu chưa có dữ liệu bổ sung.</p>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_database(db_path: Path, employees: list[dict]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    imported_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE IF EXISTS employees")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                gender TEXT,
                position TEXT,
                department TEXT,
                greeting TEXT,
                source_file TEXT,
                source_page INTEGER,
                imported_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS import_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute("DELETE FROM employees")
        connection.executemany(
            """
            INSERT INTO employees (
                employee_id,
                full_name,
                gender,
                position,
                department,
                greeting,
                source_file,
                source_page,
                imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(employee_id) DO UPDATE SET
                full_name = excluded.full_name,
                gender = excluded.gender,
                position = excluded.position,
                department = excluded.department,
                greeting = excluded.greeting,
                source_file = excluded.source_file,
                source_page = excluded.source_page,
                imported_at = excluded.imported_at
            """,
            [
                (
                    item["employee_id"],
                    item["full_name"],
                    item.get("gender", ""),
                    item.get("position", ""),
                    item.get("department", ""),
                    item.get("greeting", ""),
                    item["source_file"],
                    item["source_page"],
                    imported_at,
                )
                for item in employees
            ],
        )
        connection.execute(
            """
            INSERT INTO import_metadata (key, value)
            VALUES ('imported_at', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (imported_at,),
        )
        connection.execute(
            """
            INSERT INTO import_metadata (key, value)
            VALUES ('employee_count', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(len({item["employee_id"] for item in employees})),),
        )


def main() -> int:
    os.chdir(REPO_ROOT)
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    source_dir = args.source or Path(
        os.getenv("EMPLOYEE_DIRECTORY_SOURCE_DIR", "documents/MKAC")
    )
    source_glob = args.glob or os.getenv(
        "EMPLOYEE_DIRECTORY_SOURCE_GLOB",
        "3. DANH SÁCH KHÁM SỨC KHOẺ 2026.pdf",
    )
    db_path = args.db or Path(
        os.getenv("EMPLOYEE_DIRECTORY_DB_PATH", "data/employee_directory.sqlite")
    )
    summary_path = args.summary or Path(
        os.getenv(
            "EMPLOYEE_DIRECTORY_SUMMARY_PATH",
            "documents/MKAC/0. Thong tin nhan su va lanh dao MKAC.html",
        )
    )

    employees = collect_employees(source_dir, source_glob)
    unique_ids = {item["employee_id"] for item in employees}
    duplicates = len(employees) - len(unique_ids)
    print(f"Total extracted: {len(employees)}")
    print(f"Unique IDs:       {len(unique_ids)}")
    if duplicates:
        print(f"Duplicates:       {duplicates} (last value wins)")

    if args.dry_run:
        for item in employees[:20]:
            print(f"- {item['employee_id']} {item['full_name']}")
        if len(employees) > 20:
            print(f"... {len(employees) - 20} more")
        return 0

    write_database(db_path, employees)
    write_summary(summary_path, employees, source_glob)
    print(f"Database:         {db_path}")
    print(f"Summary:          {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
