#!/usr/bin/env python3
"""Create a reproducible, synthetic SQLite dataset for MSM/MSA development."""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = REPO_ROOT / "database" / "schema" / "msm_msa.sql"
DEFAULT_DB = REPO_ROOT / "data" / "msm_mock.sqlite"
DEFAULT_AS_OF = date(2026, 10, 17)

DEPARTMENTS = (
    ("QA", "Quality Assurance"),
    ("MA", "Maintenance"),
    ("PD", "Production"),
    ("PE", "Process Engineering"),
    ("QC", "Quality Control"),
    ("IT", "Information Technology"),
)
MACHINE_TYPES = (
    ("Digital caliper", "Mitutoyo-style", "0-150 mm", "±0.01 mm"),
    ("Micrometer", "Outside micrometer", "0-25 mm", "±0.001 mm"),
    ("Height gauge", "Digital height gauge", "0-300 mm", "±0.01 mm"),
    ("2D measuring system", "Optical measurement system", "0-200 mm", "±0.005 mm"),
    ("Torque tester", "Digital torque tester", "0-20 N·m", "±0.5%"),
    ("Pin gauge set", "Go/No-Go gauge", "0.5-5.0 mm", "±0.002 mm"),
    ("Surface roughness tester", "Portable roughness tester", "Ra 0.01-40 µm", "±5%"),
    ("Resistance meter", "Four-wire resistance meter", "0.1 µΩ-20 kΩ", "±0.1%"),
    ("Thickness gauge", "Coating thickness gauge", "0-1500 µm", "±1 µm"),
    ("Digital scale", "Precision balance", "0-600 g", "0.001 g"),
)
MAKERS = (
    "Mock Instruments A",
    "Mock Instruments B",
    "Mock Instruments C",
    "Mock Instruments D",
)
CALIBRATORS = (
    "Internal QA Lab (mock)",
    "Mock Calibration Lab 01",
    "Mock Calibration Lab 02",
)
CUSTOMER_NAMES = tuple(f"Mock Customer {index:03d}" for index in range(1, 31))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--departments", type=int, default=30)
    parser.add_argument("--machines", type=int, default=5000)
    parser.add_argument("--calibrations-per-machine", type=int, default=6)
    parser.add_argument("--msa-ratio", type=float, default=0.42)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--as-of", type=date.fromisoformat, default=DEFAULT_AS_OF)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace the target database if it already exists",
    )
    args = parser.parse_args()
    if args.departments < 1:
        parser.error("--departments phải lớn hơn 0")
    if args.machines < 1:
        parser.error("--machines phải lớn hơn 0")
    if args.calibrations_per_machine < 0:
        parser.error("--calibrations-per-machine không được âm")
    if not 0 <= args.msa_ratio <= 1:
        parser.error("--msa-ratio phải nằm trong khoảng 0..1")
    if not args.schema.is_file():
        parser.error(f"Không tìm thấy schema: {args.schema}")
    return args


def add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    next_month = date(year + (month == 12), month % 12 + 1, 1)
    last_day = (next_month - timedelta(days=1)).day
    return date(year, month, min(value.day, last_day))


def epoch_seconds(value: date) -> int:
    instant = datetime.combine(value, time(12), tzinfo=timezone.utc)
    return int(instant.timestamp())


def timestamp_text(value: date, rng: random.Random) -> str:
    hour = rng.randrange(24)
    minute = rng.randrange(60)
    second = rng.randrange(60)
    return f"{value.isoformat()} {hour:02d}:{minute:02d}:{second:02d}"


def attachment(kind: str, key: str) -> str:
    return json.dumps([f"/mock/msm/{kind}/{key}.pdf"], ensure_ascii=False)


def insert_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    rows: list[tuple[object, ...]],
) -> None:
    if not rows:
        return
    names = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(
        f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})', rows
    )


def _department_rows(count: int, as_of: date, rng: random.Random):
    names = []
    rows = []
    for index in range(1, count + 1):
        code, name = DEPARTMENTS[(index - 1) % len(DEPARTMENTS)]
        department_id = f"MOCK-{code}-{index:03d}"
        names.append((department_id, name))
        created = as_of - timedelta(days=rng.randint(30, 1500))
        rows.append(
            (
                index,
                department_id,
                f"{name} {index:02d}",
                timestamp_text(created, rng),
                timestamp_text(as_of, rng),
                epoch_seconds(created),
            )
        )
    return names, rows


def _user_rows(as_of: date, rng: random.Random):
    permissions = []
    emails = []
    for index in range(1, 121):
        user_name = f"MOCK-QA-{index:04d}"
        created = as_of - timedelta(days=rng.randint(30, 1200))
        permissions.append(
            (
                index,
                user_name,
                "Y" if index % 12 == 0 else "N",
                timestamp_text(created, rng),
                timestamp_text(as_of, rng),
            )
        )
        if index % 8 != 0:
            emails.append(
                (
                    len(emails) + 1,
                    user_name,
                    f"{user_name.lower()}@example.invalid",
                    timestamp_text(created, rng),
                    timestamp_text(as_of, rng),
                    epoch_seconds(created),
                    "N" if index % 17 == 0 else "Y",
                )
            )
    return permissions, emails


def _plan_rows(as_of: date, rng: random.Random):
    rows = []
    row_id = 1
    for year in range(as_of.year - 2, as_of.year + 1):
        for month in range(1, 13):
            created = min(date(year, month, 1), as_of)
            rows.append(
                (
                    row_id,
                    month,
                    year,
                    attachment("plans", f"{year}-{month:02d}"),
                    timestamp_text(created, rng),
                    timestamp_text(min(as_of, created + timedelta(days=10)), rng),
                    epoch_seconds(created),
                )
            )
            row_id += 1
    return rows


def _trace_rows(as_of: date, rng: random.Random):
    rows = []
    for index, customer in enumerate(CUSTOMER_NAMES, start=1):
        created = as_of - timedelta(days=rng.randint(30, 1600))
        rows.append(
            (
                index,
                f"MOCK-CUSTOMER-{index:04d}",
                customer,
                attachment("traceability", f"customer-{index:04d}"),
                epoch_seconds(created),
                epoch_seconds(as_of),
                timestamp_text(created, rng),
                timestamp_text(as_of, rng),
            )
        )
    return rows


def _machine_due_offset(index: int, rng: random.Random) -> tuple[str, int]:
    slot = index % 100
    if slot < 13:
        return "Overdue", -rng.randint(1, 240)
    if slot < 23:
        return "In progress", rng.randint(-5, 20)
    if slot < 37:
        return "Start calibration", rng.randint(1, 45)
    if slot < 93:
        return "Not due yet", rng.randint(46, 540)
    if slot < 97:
        return "Active", rng.randint(541, 900)
    return "Unused", rng.randint(30, 500)


def _machine_rows(
    count: int,
    department_names: list[tuple[str, str]],
    user_count: int,
    as_of: date,
    msa_ratio: float,
    rng: random.Random,
):
    machines = []
    msa_machines = []
    msa_configs = []
    machine_offsets = {}
    for index in range(1, count + 1):
        department_id, _ = department_names[(index - 1) % len(department_names)]
        machine_type, model, spec, precision = MACHINE_TYPES[
            (index - 1) % len(MACHINE_TYPES)
        ]
        calibration_status, due_offset = _machine_due_offset(index, rng)
        requested_due_date = as_of + timedelta(days=due_offset)
        cycles = (3, 6, 12, 12, 12, 18, 24, 30)
        if calibration_status == "Unused":
            cycle = rng.choice(cycles)
            due_date = add_months(as_of, cycle)
        else:
            valid_cycles = tuple(
                value
                for value in cycles
                if add_months(as_of, value) >= requested_due_date
            )
            if valid_cycles:
                cycle = rng.choice(valid_cycles)
                due_date = requested_due_date
            else:
                cycle = max(cycles)
                due_date = add_months(as_of, cycle)
        last_calibration = add_months(due_date, -cycle)
        if calibration_status == "Overdue" and due_date > as_of:
            last_calibration = as_of - timedelta(days=cycle + 1)
            due_date = add_months(last_calibration, cycle)
        machine_id = f"MOCK-HA-{index:07d}"
        machine_name = f"{machine_type} {index:05d}"
        is_msa = rng.random() < msa_ratio
        is_archived = index % 97 == 0
        is_unused = calibration_status == "Unused"
        status = "Unused" if is_unused else ("Broken" if index % 113 == 0 else "Active")
        next_due_date = 0
        if calibration_status != "Unused" and due_date > as_of and index % 19 == 0:
            next_due = due_date + timedelta(days=rng.randint(7, 60))
            next_due_date = epoch_seconds(next_due)
        machine_offsets[machine_id] = (
            due_date,
            last_calibration,
            cycle,
            next_due_date,
        )
        created = as_of - timedelta(days=rng.randint(60, 1700))
        pic_number = (index - 1) % user_count + 1
        machines.append(
            (
                machine_name,
                model,
                f"SYN-{index:08d}",
                f"MOCK-QA-{pic_number:04d}",
                department_id,
                rng.choice(CALIBRATORS),
                epoch_seconds(last_calibration),
                epoch_seconds(due_date),
                calibration_status,
                cycle,
                spec,
                precision,
                rng.choice(MAKERS),
                "Y" if index % 4 != 0 else "N",
                "Y" if index % 53 == 0 else "N",
                rng.choice(
                    (
                        "Mock Standard MSA-01",
                        "Mock Standard ISO-01",
                        "Mock Standard JIS-01",
                    )
                ),
                f"[MOCK] Synthetic record {index:07d}",
                status,
                timestamp_text(created, rng),
                timestamp_text(as_of, rng),
                index,
                epoch_seconds(created),
                machine_id,
                f"[MOCK] Measurement use case {index:07d}",
                "Y" if is_msa else "N",
                f"/mock/msm/images/{machine_id}.png",
                "Y" if is_archived else "N",
                next_due_date,
                "MOCK-ALL"
                if index % 5 == 0
                else f"MOCK-PRODUCT-{(index % 40) + 1:03d}",
                attachment("machine", machine_id),
            )
        )
        if is_msa:
            data_type = "Attribute" if index % 5 == 0 else "Variable"
            objects = (
                ["Kappa", "Gage R&R"] if data_type == "Attribute" else ["Gage R&R"]
            )
            if data_type == "Variable":
                if index % 2 == 0:
                    objects.append("Bias")
                if index % 3 == 0:
                    objects.append("Linearity")
                if index % 4 == 0:
                    objects.append("Stability")
            config_created = as_of - timedelta(days=rng.randint(10, 1000))
            msa_machines.append(
                (
                    len(msa_machines) + 1,
                    machine_id,
                    machine_name,
                    department_id,
                    f"[MOCK] Characteristic {index:05d}",
                    f"MOCK-QA-{pic_number:04d}",
                    "[MOCK] Reference sample",
                    f"MOCK-PRD-{index % 250:04d}",
                    "[MOCK] Stratified sample method; 10 pieces",
                    data_type,
                    "Y" if "Bias" in objects else "N",
                    "Y" if "Linearity" in objects else "N",
                    "Y" if "Stability" in objects else "N",
                    "Y" if "Gage R&R" in objects else "N",
                    "Y" if data_type == "Attribute" else "N",
                    "[MOCK] Additional study notes",
                    "Y" if index % 11 == 0 else "N",
                    rng.choice((6, 12, 12, 12, 24)),
                    "[MOCK] Synthetic MSA configuration",
                    timestamp_text(config_created, rng),
                    timestamp_text(as_of, rng),
                    epoch_seconds(config_created),
                    "Y",
                    "Y" if index % 89 == 0 else "N",
                    as_of.year
                    if index % 11 == 0
                    else as_of.year - rng.choice((0, 1, 2)),
                    "Y" if index % 11 != 0 else "N",
                )
            )
            msa_configs.append(
                (machine_id, machine_name, department_id, data_type, objects, index)
            )
    return machines, msa_machines, msa_configs, machine_offsets


def _msa_judge_rows(msa_configs, as_of: date, rng: random.Random):
    rows = []
    row_id = 1
    for config in msa_configs:
        machine_id, machine_name, department_id, data_type, objects, machine_index = (
            config
        )
        product_sample = f"MOCK-PRD-{machine_index % 250:04d}"
        years = range(as_of.year - 2, as_of.year + 1)
        if machine_index % 11 == 0:
            years = (as_of.year,)
        for year in years:
            for objective in objects:
                plan_day = date(year, rng.randint(1, 12), rng.randint(1, 27))
                result_date = None
                conclusion = None
                judge_status = None
                if plan_day <= as_of and (year < as_of.year or rng.random() < 0.72):
                    result_date = min(
                        as_of, plan_day + timedelta(days=rng.randint(0, 60))
                    )
                    if objective == "Gage R&R":
                        grr = round(rng.uniform(3, 38), 1)
                        ndc = rng.randint(2, 18)
                        judge_status = "OK" if grr <= 30 and ndc >= 5 else "NG"
                        conclusion = (
                            f"[MOCK] Gage R&R simulation: %GRR={grr:.1f}; ndc={ndc}; "
                            "not an actual measurement result."
                        )
                    elif objective == "Kappa":
                        kappa = round(rng.uniform(0.45, 0.98), 2)
                        judge_status = "OK" if kappa >= 0.75 else "NG"
                        conclusion = (
                            f"[MOCK] Kappa simulation: {kappa:.2f}; "
                            "not an actual measurement result."
                        )
                    else:
                        metric = round(rng.uniform(0.1, 5.0), 2)
                        judge_status = (
                            "NG" if metric > 4.2 and rng.random() < 0.35 else "OK"
                        )
                        conclusion = (
                            f"[MOCK] {objective} simulation: index={metric:.2f}; "
                            "not an actual measurement result."
                        )
                rows.append(
                    (
                        machine_id,
                        machine_name,
                        product_sample,
                        f"[MOCK] Characteristic {machine_index:05d}",
                        data_type,
                        department_id,
                        objective,
                        str(epoch_seconds(plan_day)),
                        str(epoch_seconds(result_date)) if result_date else None,
                        conclusion,
                        judge_status,
                        timestamp_text(plan_day, rng),
                        timestamp_text(result_date or plan_day, rng),
                        epoch_seconds(plan_day),
                        attachment(
                            "msa", f"{machine_id}-{year}-{objective.replace(' ', '-')}"
                        )
                        if result_date and objective == "Gage R&R"
                        else None,
                        year,
                        row_id,
                    )
                )
                row_id += 1
    return rows


def _calibration_rows(
    machines, machine_offsets, count: int, as_of: date, rng: random.Random
):
    rows = []
    row_id = 1
    for machine in machines:
        machine_id = machine[22]
        machine_index = machine[20]
        machine_status = machine[17]
        due_date, last_calibration, cycle, next_due_date = machine_offsets[machine_id]
        calibration_due_date = due_date
        if next_due_date:
            calibration_due_date = datetime.fromtimestamp(
                next_due_date, tz=timezone.utc
            ).date()
        for offset in range(count):
            calibration_date = add_months(last_calibration, -cycle * offset)
            next_calibration_history = (
                due_date
                if offset == 0
                else add_months(last_calibration, -cycle * (offset - 1))
            )
            is_draft = calibration_date > as_of
            effective = (
                None
                if is_draft
                else "N"
                if (row_id % 13 == 0 or row_id % 29 == 0)
                else "Y"
            )
            reschedule = 0
            if effective == "N":
                if rng.random() < 0.45:
                    reschedule = epoch_seconds(
                        calibration_date + timedelta(days=rng.randint(14, 60))
                    )
            rows.append(
                (
                    row_id,
                    rng.choice(CALIBRATORS),
                    epoch_seconds(calibration_date),
                    None if is_draft else f"MOCK-QA-{(row_id - 1) % 120 + 1:04d}",
                    None
                    if is_draft
                    else attachment("calibration-certificates", f"cal-{row_id:08d}"),
                    None
                    if is_draft
                    else attachment("record-sheets", f"record-{row_id:08d}"),
                    f"[MOCK] Scheduled calibration {row_id:08d}"
                    if is_draft
                    else f"[MOCK] Synthetic calibration observation {row_id:08d}",
                    effective,
                    reschedule,
                    None if is_draft else f"MOCK-QA-{(row_id + 16) % 120 + 1:04d}",
                    None if is_draft else f"[MOCK] Synthetic judge note {row_id:08d}",
                    None if is_draft else attachment("judge", f"judge-{row_id:08d}"),
                    epoch_seconds(as_of),
                    timestamp_text(calibration_date, rng),
                    timestamp_text(as_of, rng),
                    "Y" if is_draft else "N",
                    epoch_seconds(next_calibration_history),
                    machine_id,
                )
            )
            row_id += 1

        if (
            count
            and machine_status != "Unused"
            and calibration_due_date > as_of
            and machine_index % 11 == 0
        ):
            draft_id = row_id
            rows.append(
                (
                    row_id,
                    rng.choice(CALIBRATORS),
                    epoch_seconds(calibration_due_date),
                    None,
                    None,
                    None,
                    f"[MOCK] Draft scheduled calibration {draft_id:08d}",
                    None,
                    0,
                    None,
                    None,
                    None,
                    epoch_seconds(as_of),
                    timestamp_text(as_of, rng),
                    timestamp_text(as_of, rng),
                    "Y",
                    epoch_seconds(add_months(calibration_due_date, cycle)),
                    machine_id,
                )
            )
            row_id += 1
    return rows


def build_database(
    schema_path: Path,
    db_path: Path,
    *,
    department_count: int,
    machine_count: int,
    calibrations_per_machine: int,
    msa_ratio: float,
    seed: int,
    as_of: date,
    replace: bool = False,
) -> dict[str, int]:
    if db_path.exists() and not replace:
        raise FileExistsError(
            f"Database đã tồn tại: {db_path}; dùng --replace nếu muốn tạo lại."
        )
    if department_count < 1:
        raise ValueError("department_count phải lớn hơn 0")
    if machine_count < 1:
        raise ValueError("machine_count phải lớn hơn 0")
    if calibrations_per_machine < 0:
        raise ValueError("calibrations_per_machine không được âm")
    if not 0 <= msa_ratio <= 1:
        raise ValueError("msa_ratio phải nằm trong khoảng 0..1")
    schema_sql = schema_path.read_text(encoding="utf-8")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{db_path.name}.", suffix=".tmp", dir=db_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    rng = random.Random(seed)
    try:
        with sqlite3.connect(temporary_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(schema_sql)
            connection.execute("PRAGMA user_version = 1")
            connection.executemany(
                "INSERT INTO schema_metadata (key, value) VALUES (?, ?)",
                [
                    ("schema_version", "1"),
                    ("dataset_type", "synthetic_mock"),
                    ("seed", str(seed)),
                    ("as_of", as_of.isoformat()),
                    ("department_count", str(department_count)),
                    ("machine_count", str(machine_count)),
                    ("calibrations_per_machine", str(calibrations_per_machine)),
                    ("msa_ratio", str(msa_ratio)),
                ],
            )
            department_names, departments = _department_rows(
                department_count, as_of, rng
            )
            users, emails = _user_rows(as_of, rng)
            plans = _plan_rows(as_of, rng)
            traces = _trace_rows(as_of, rng)
            machines, msa_machines, msa_configs, machine_offsets = _machine_rows(
                machine_count, department_names, len(users), as_of, msa_ratio, rng
            )
            msa_judges = _msa_judge_rows(msa_configs, as_of, rng)

            insert_rows(
                connection,
                "MSM_MACHINE_DEPARTMENT",
                (
                    "ID",
                    "DEPARTMENT_ID",
                    "DEPARTMENT_NAME",
                    "CREATE_DATE",
                    "EDIT_DATE",
                    "CREATE_DATE_UNIX",
                ),
                departments,
            )
            insert_rows(
                connection,
                "MSM_USER_PERMISSION",
                ("ID", "USER_NAME", "HAS_PERMISSION", "CREATE_DATE", "EDIT_DATE"),
                users,
            )
            insert_rows(
                connection,
                "MSM_USER_EMAIL",
                (
                    "ID",
                    "USER_NAME",
                    "EMAIL",
                    "CREATE_DATE",
                    "EDIT_DATE",
                    "CREATE_DATE_UNIX",
                    "SEND_EMAIL",
                ),
                emails,
            )
            insert_rows(
                connection,
                "MSM_PLAN_FILE",
                (
                    "ID",
                    "MONTH",
                    "YEAR",
                    "MONTH_ATTACHMENT",
                    "CREATE_DATE",
                    "EDIT_DATE",
                    "CREATE_DATE_UNIX",
                ),
                plans,
            )
            insert_rows(
                connection,
                "MSM_TRACE_DIAGRAM",
                (
                    "ID",
                    "CUSTOMER_ID",
                    "CUSTOMER_NAME",
                    "TRACE_DIAGRAM_ATTACHMENT",
                    "CREATE_DATE_UNIX",
                    "EDIT_DATE_UNIX",
                    "CREATE_DATE",
                    "EDIT_DATE",
                ),
                traces,
            )
            insert_rows(
                connection,
                "MSM_MACHINE",
                (
                    "MACHINE_NAME",
                    "MODEL",
                    "SERIAL_NUMBER",
                    "PIC",
                    "DEPARTMENT_ID",
                    "CALIBRATOR",
                    "LAST_CALIBRATION",
                    "NEXT_CALIBRATION",
                    "CALIBRATION_STATUS",
                    "INSPECT_CYCLE",
                    "SPEC",
                    "PRECISION",
                    "MAKER",
                    "CALIBRATE_SHEET",
                    "ABNORMAL_SHEET",
                    "APPLIED_STANDARD",
                    "NOTE",
                    "STATUS",
                    "CREATE_DATE",
                    "EDIT_DATE",
                    "ID",
                    "REGISTERED_DATE",
                    "MACHINE_ID",
                    "PURPOSE",
                    "IS_MSA",
                    "IMG_URL",
                    "IS_ARCHIVE",
                    "NEXT_DUE_DATE",
                    "PROJECT_NAME",
                    "MACHINE_ATTACHMENT",
                ),
                machines,
            )
            insert_rows(
                connection,
                "MSM_CALIB_JUDGE",
                (
                    "ID",
                    "CALIBRATION_UNIT",
                    "CALIBRATION_DATE",
                    "MANAGER_ID",
                    "CALIB_ATTACHMENT",
                    "CALIB_RECORD_SHEET",
                    "NOTE_CALIB",
                    "EFFECTIVE",
                    "RESCHEDULE_DATE",
                    "APPROVER_ID",
                    "NOTE_JUDGE",
                    "JUDGE_ATTACHMENT",
                    "EDIT_DATE_UNIX",
                    "CREATE_DATE",
                    "EDIT_DATE",
                    "IS_DRAFT",
                    "NEXT_CALIBRATION_HISTORY",
                    "MACHINE_ID",
                ),
                _calibration_rows(
                    machines, machine_offsets, calibrations_per_machine, as_of, rng
                ),
            )
            insert_rows(
                connection,
                "MSA_MACHINE",
                (
                    "ID",
                    "MACHINE_ID",
                    "MACHINE_NAME",
                    "DEPARTMENT_ID",
                    "CHARACTERISTICS",
                    "MANAGER_ID",
                    "MEASUREMENT_SAMPLE",
                    "PRD_SAMPLE",
                    "PRD_SAMPLE_METHOD",
                    "DATA_TYPE",
                    "BIAS",
                    "LINEARITY",
                    "STABILITY",
                    "GAGE_RR",
                    "IS_GO",
                    "OTHER",
                    "NEW_PRD",
                    "PERIODIC",
                    "NOTE",
                    "CREATE_DATE",
                    "EDIT_DATE",
                    "CREATE_DATE_UNIX",
                    "LINK_MSM",
                    "IS_ARCHIVE",
                    "YEAR",
                    "IS_PERIODIC",
                ),
                msa_machines,
            )
            insert_rows(
                connection,
                "MSA_JUDGE",
                (
                    "MACHINE_ID",
                    "MACHINE_NAME",
                    "PRD_SAMPLE",
                    "CHARACTERISTICS",
                    "DATA_TYPE",
                    "DEPARTMENT_ID",
                    "OBJECT",
                    "PLAN_DATE_UNIX",
                    "RESULT_DATE_UNIX",
                    "CONCLUSION",
                    "IS_JUDGE_OK",
                    "CREATE_DATE",
                    "EDIT_DATE",
                    "CREATE_DATE_UNIX",
                    "GAGE_RR_ATTACHMENT",
                    "YEAR",
                    "ID",
                ),
                msa_judges,
            )
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            foreign_key_errors = connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
            if integrity != "ok":
                raise sqlite3.IntegrityError(f"integrity_check thất bại: {integrity}")
            if foreign_key_errors:
                raise sqlite3.IntegrityError(
                    f"foreign_key_check thất bại: {len(foreign_key_errors)} lỗi"
                )
            counts = {
                table: int(
                    connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
                for table in (
                    "MSM_MACHINE_DEPARTMENT",
                    "MSM_USER_PERMISSION",
                    "MSM_PLAN_FILE",
                    "MSM_TRACE_DIAGRAM",
                    "MSM_MACHINE",
                    "MSM_USER_EMAIL",
                    "MSM_CALIB_JUDGE",
                    "MSA_MACHINE",
                    "MSA_JUDGE",
                )
            }
        os.replace(temporary_path, db_path)
        return counts
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    counts = build_database(
        args.schema,
        args.db,
        department_count=args.departments,
        machine_count=args.machines,
        calibrations_per_machine=args.calibrations_per_machine,
        msa_ratio=args.msa_ratio,
        seed=args.seed,
        as_of=args.as_of,
        replace=args.replace,
    )
    print(
        f"Đã tạo SQLite mock (dữ liệu tổng hợp, seed={args.seed}, as-of={args.as_of}):"
    )
    print(f"  {args.db}")
    for table, count in counts.items():
        print(f"  {table}: {count:,}")


if __name__ == "__main__":
    main()
