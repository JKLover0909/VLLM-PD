#!/usr/bin/env python3
"""Build a safe SQLite WMS snapshot from an Oracle SQL text export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.integrations.mes_wms_contract import (  # noqa: E402
    AUDIT_SEMANTIC_EPOCH,
    CURRENT_SEMANTIC_EPOCH,
    DATASET_AUDIT,
    DATASET_CURRENT,
    DATASET_LEGACY,
    DATA_CONTRACT_VERSION,
    EXPECTED_AUDIT_BASIS,
    EXPECTED_CURRENT_BASIS,
    EXPECTED_LEGACY_BASIS,
    LEGACY_SEMANTIC_EPOCH,
    PLANT_ID,
    REASON_CURRENT_GRAIN_DUPLICATE,
    REASON_DATASET_NOT_OBSERVED,
    REASON_QUANTITY_EVIDENCE_INCOMPLETE,
    REASON_TRANSACTION_HEADER_NOT_OBSERVED,
    REASON_TRANSACTION_SEMANTICS_UNAVAILABLE,
    REASON_UOM_UNAVAILABLE,
    SCHEMA_VERSION,
    SEMANTIC_CONTRACT_VERSION,
    SOURCE_SCHEMA,
    capability_statuses_for_datasets,
    validate_snapshot_connection,
)


ORACLE_TIMESTAMP_FORMATS = (
    "%d-%b-%y %I.%M.%S.%f %p",
    "%d-%b-%y %I.%M.%S %p",
)


def _normalize_oracle_timestamp(value: str) -> str:
    candidate = value.strip()
    # Oracle TIMESTAMP(9) may contain nanoseconds; Python datetime keeps six
    # fractional digits, so truncate only the excess precision.
    candidate = re.sub(
        r"(\.\d{6})\d+(\s+[AP]M)$",
        r"\1\2",
        candidate,
        flags=re.IGNORECASE,
    )
    for fmt in ORACLE_TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(candidate, fmt)
        except ValueError:
            continue
        return parsed.isoformat(sep=" ")
    return candidate


DEFAULT_SOURCE = (
    REPO_ROOT / "database" / "raw_mkac" / "mes_wms_20260728.sql"
)
DEFAULT_SCHEMA_PATH = REPO_ROOT / "database" / "schema" / "mes_wms.sql"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "mes_wms.sqlite"

RAW_TABLE_COLUMNS = {
    "PW_CURRENT_ITEM": (
        "ID",
        "CREATE_DATE",
        "EDIT_DATE",
        "ITEM_CODE",
        "ITEM_LOT_ID",
        "QTY",
        "TIME_UPDATE",
        "TIME_UPDATE_UNIX",
        "TRANS_ID",
        "PROCESS_ID",
    ),
    "PW_PROCESS": (
        "ID",
        "CREATE_DATE",
        "EDIT_DATE",
        "PROCESS_ID",
        "PROCESS_NAME",
        "PROCESS_PHYSICAL_ID",
        "STATUS",
        "NOTE",
        "IS_CHECK_MATERIAL",
    ),
    "PW_SNAPSHORT": (
        "ID",
        "CREATE_DATE",
        "EDIT_DATE",
        "SNAPSHORT_ID",
        "SNAPSHORT_DATE",
        "ITEM_CODE",
        "QTY",
        "PROCESS_ID",
        "ITEM_LOT_ID",
    ),
    "PW_TRANSACTION": (
        "ID",
        "CREATE_DATE",
        "EDIT_DATE",
        "TRANS_ID",
        "TRANS_CODE",
        "TRANS_DATE",
        "TRANS_DATE_UNIX",
        "PROCESS_ID",
        "ITEM_CODE",
        "QTY",
        "TRANS_STATUS",
        "RELATE_ID",
        "USER_ID",
        "NOTE",
        "VALUE_1",
        "VALUE_2",
        "VALUE_3",
        "CREADIT_CODE",
        "DEBIT_CODE",
        "DELETED",
    ),
    "PW_TRANSACTION_DEFINE": (
        "ID",
        "CREATE_DATE",
        "EDIT_DATE",
        "TRANS_CODE",
        "TRANS_NAME",
        "NOTE",
    ),
    "PW_TRANS_DETAIL": (
        "ID",
        "CREATE_DATE",
        "EDIT_DATE",
        "TRANS_ID",
        "ITEM_LOT_ID",
        "QTY",
        "PRODUCT_ID",
        "LOT_ID",
        "NOTE",
        "ATTACH_PATH",
    ),
}

INSERT_START = re.compile(
    r"^\s*Insert\s+into\s+"
    + re.escape(SOURCE_SCHEMA)
    + r"\.([A-Z0-9_]+)\s*\(([^)]*)\)\s*Values\s*\(",
    re.IGNORECASE,
)
EXPORT_DATE = re.compile(
    r"File\s+created\s+-\s+([A-Za-z]+)-([A-Za-z]+)-(\d{1,2})-(\d{4})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedRow:
    table: str
    values: dict[str, Any]


@dataclass
class ImportStats:
    candidate_rows: dict[str, int]
    inserted_rows: dict[str, int]
    invalid_quantity_rows: dict[str, int]


@dataclass
class ImportPlan:
    staged: dict[str, list[dict[str, Any]]]
    stats: ImportStats


@dataclass(frozen=True)
class DatasetEvidence:
    dataset: str
    status: str
    reason_code: str
    source_tables: str
    source_state: str
    candidate_row_count: int
    inserted_row_count: int
    invalid_quantity_row_count: int
    source_as_of: str
    source_as_of_state: str
    source_as_of_basis: str
    source_timezone: str
    semantic_epoch: str
    evidence_basis: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the allowlisted source without creating a SQLite file.",
    )
    mode.add_argument(
        "--validate-snapshot",
        type=Path,
        help="Validate an existing SQLite snapshot in read-only mode.",
    )
    parser.add_argument(
        "--report-json",
        metavar="PATH_OR_DASH",
        help="Write a versioned aggregate JSON report to a path or '-' for stdout.",
    )
    args = parser.parse_args()
    if args.report_json != "-":
        report_path = Path(args.report_json).resolve() if args.report_json else None
        protected = {args.schema.resolve()}
        if args.validate_snapshot is not None:
            protected.add(args.validate_snapshot.resolve())
        else:
            protected.update({args.source.resolve(), args.db.resolve()})
        if report_path in protected:
            parser.error("--report-json must not overwrite source, schema, or snapshot")
    return args


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return unicodedata.normalize("NFC", text)


def _required_text(value: Any, field: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"Trường bắt buộc {field} đang rỗng")
    return text


def _optional_int(value: Any) -> int | None:
    text = _optional_text(value)
    if text is None:
        return None
    return int(text)


def _quantity(value: Any) -> tuple[str | None, int, str | None]:
    text = _optional_text(value)
    if text is None:
        return None, 0, "empty"
    try:
        number = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None, 0, "not_numeric"
    if not number.is_finite():
        return None, 0, "not_finite"
    if number < 0:
        return None, 0, "negative"
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0", 1, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_export_date(source_path: Path) -> str:
    with source_path.open("r", encoding="latin-1", errors="replace") as source:
        header = "".join(next(source, "") for _ in range(8))
    match = EXPORT_DATE.search(header)
    if not match:
        return ""
    _weekday, month_name, day, year = match.groups()
    try:
        return datetime.strptime(
            f"{year}-{month_name}-{day}", "%Y-%B-%d"
        ).date().isoformat()
    except ValueError:
        return ""


def _parse_columns(raw_columns: str) -> tuple[str, ...]:
    return tuple(part.strip().strip('"').upper() for part in raw_columns.split(","))


def _split_oracle_values(value_text: str) -> list[Any]:
    values: list[Any] = []
    token: list[str] = []
    in_string = False
    depth = 0
    index = 0
    while index < len(value_text):
        char = value_text[index]
        if in_string:
            if char == "'" and index + 1 < len(value_text) and value_text[index + 1] == "'":
                token.extend(("'", "'"))
                index += 2
                continue
            token.append(char)
            if char == "'":
                in_string = False
            index += 1
            continue
        if char == "'":
            in_string = True
            token.append(char)
        elif char == "(":
            depth += 1
            token.append(char)
        elif char == ")":
            depth -= 1
            token.append(char)
        elif char == "," and depth == 0:
            values.append(_parse_oracle_token("".join(token)))
            token = []
        else:
            token.append(char)
        index += 1
    if in_string or depth != 0:
        raise ValueError("Oracle INSERT có literal chưa đóng")
    values.append(_parse_oracle_token("".join(token)))
    return values


def _unquote_oracle_string(token: str) -> str:
    return token[1:-1].replace("''", "'")


def _parse_oracle_token(raw: str) -> Any:
    token = raw.strip()
    if not token or token.upper() == "NULL":
        return None
    if token.startswith("'") and token.endswith("'"):
        return _unquote_oracle_string(token)
    timestamp_match = re.fullmatch(
        r"TO_TIMESTAMP\s*\(\s*('(?:''|[^'])*')\s*,.*\)",
        token,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if timestamp_match:
        return _normalize_oracle_timestamp(
            _unquote_oracle_string(timestamp_match.group(1))
        )
    return token


def _statement_values(statement: str, match: re.Match[str]) -> str:
    start = match.end()
    index = start
    in_string = False
    depth = 1
    while index < len(statement):
        char = statement[index]
        if in_string:
            if char == "'" and index + 1 < len(statement) and statement[index + 1] == "'":
                index += 2
                continue
            if char == "'":
                in_string = False
        else:
            if char == "'":
                in_string = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return statement[start:index]
        index += 1
    raise ValueError("Oracle INSERT thiếu dấu đóng Values(...)")


def iter_allowlisted_rows(source_path: Path) -> Iterator[ParsedRow]:
    statement = ""
    collecting = False
    with source_path.open("r", encoding="latin-1", errors="replace", newline="") as source:
        for line in source:
            if not collecting:
                match = INSERT_START.match(line)
                if not match:
                    continue
                table = match.group(1).upper()
                if table not in RAW_TABLE_COLUMNS:
                    continue
                statement = line
                collecting = True
            else:
                statement += line

            if collecting and _statement_complete(statement):
                match = INSERT_START.match(statement)
                if match is None:
                    raise ValueError("Không thể parse Oracle INSERT allowlist")
                table = match.group(1).upper()
                columns = _parse_columns(match.group(2))
                expected_columns = RAW_TABLE_COLUMNS[table]
                if columns != expected_columns:
                    raise ValueError(
                        f"Cột {table} không đúng contract: {columns!r}"
                    )
                values = _split_oracle_values(_statement_values(statement, match))
                if len(values) != len(columns):
                    raise ValueError(
                        f"{table} có {len(values)} giá trị, cần {len(columns)}"
                    )
                yield ParsedRow(table, dict(zip(columns, values)))
                statement = ""
                collecting = False
    if collecting:
        raise ValueError("Oracle INSERT allowlist cuối file chưa hoàn chỉnh")


def _statement_complete(statement: str) -> bool:
    in_string = False
    index = 0
    while index < len(statement):
        char = statement[index]
        if in_string:
            if char == "'" and index + 1 < len(statement) and statement[index + 1] == "'":
                index += 2
                continue
            if char == "'":
                in_string = False
        elif char == "'":
            in_string = True
        index += 1
    return not in_string and bool(re.search(r"\)\s*;\s*$", statement))


def _insert_process(target: sqlite3.Connection, row: dict[str, Any]) -> None:
    target.execute(
        """
        INSERT INTO wms_processes (
            source_id, create_date, edit_date, process_id, process_name,
            process_physical_id, status, is_check_material
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _optional_int(row["ID"]),
            _optional_text(row["CREATE_DATE"]),
            _optional_text(row["EDIT_DATE"]),
            _required_text(row["PROCESS_ID"], "PW_PROCESS.PROCESS_ID"),
            _optional_text(row["PROCESS_NAME"]),
            _optional_text(row["PROCESS_PHYSICAL_ID"]),
            _optional_text(row["STATUS"]),
            _optional_text(row["IS_CHECK_MATERIAL"]),
        ),
    )


def _insert_current_balance(target: sqlite3.Connection, row: dict[str, Any]) -> bool:
    quantity, valid, error = _quantity(row["QTY"])
    process_id = _required_text(row["PROCESS_ID"], "PW_CURRENT_ITEM.PROCESS_ID")
    process = target.execute(
        "SELECT process_pk FROM wms_processes WHERE process_id = ?", (process_id,)
    ).fetchone()
    target.execute(
        """
        INSERT INTO wms_current_balances (
            source_id, create_date, edit_date, item_code,
            quantity_decimal, quantity_valid, quantity_error, time_update,
            time_update_unix, trans_id, process_id, process_pk
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _optional_int(row["ID"]),
            _optional_text(row["CREATE_DATE"]),
            _optional_text(row["EDIT_DATE"]),
            _required_text(row["ITEM_CODE"], "PW_CURRENT_ITEM.ITEM_CODE"),
            quantity,
            valid,
            error,
            _optional_text(row["TIME_UPDATE"]),
            _optional_int(row["TIME_UPDATE_UNIX"]),
            _optional_text(row["TRANS_ID"]),
            process_id,
            int(process[0]) if process else None,
        ),
    )
    return bool(valid)


def _insert_snapshot_record(
    target: sqlite3.Connection, row: dict[str, Any]
) -> bool:
    quantity, valid, error = _quantity(row["QTY"])
    process_id = _required_text(
        row["PROCESS_ID"], "PW_SNAPSHORT.PROCESS_ID"
    )
    process = target.execute(
        "SELECT process_pk FROM wms_processes WHERE process_id = ?",
        (process_id,),
    ).fetchone()
    target.execute(
        """
        INSERT INTO wms_legacy_archive_records (
            source_id, create_date, edit_date, archive_id, archive_date,
            item_code, item_lot_id, process_id, process_pk,
            quantity_decimal, quantity_valid, quantity_error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _optional_int(row["ID"]),
            _optional_text(row["CREATE_DATE"]),
            _optional_text(row["EDIT_DATE"]),
            _required_text(row["SNAPSHORT_ID"], "PW_SNAPSHORT.SNAPSHORT_ID"),
            _required_text(
                row["SNAPSHORT_DATE"], "PW_SNAPSHORT.SNAPSHORT_DATE"
            ),
            _required_text(row["ITEM_CODE"], "PW_SNAPSHORT.ITEM_CODE"),
            _required_text(row["ITEM_LOT_ID"], "PW_SNAPSHORT.ITEM_LOT_ID"),
            process_id,
            int(process[0]) if process else None,
            quantity,
            valid,
            error,
        ),
    )
    return bool(valid)


def _insert_transaction(target: sqlite3.Connection, row: dict[str, Any]) -> bool:
    quantity, valid, error = _quantity(row["QTY"])
    target.execute(
        """
        INSERT INTO wms_raw_transaction_headers (
            source_id, create_date, edit_date, trans_id, trans_code, trans_date,
            trans_date_unix, process_id, item_code, quantity_decimal,
            quantity_valid, quantity_error, trans_status, deleted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _optional_int(row["ID"]),
            _optional_text(row["CREATE_DATE"]),
            _optional_text(row["EDIT_DATE"]),
            _required_text(row["TRANS_ID"], "PW_TRANSACTION.TRANS_ID"),
            _required_text(row["TRANS_CODE"], "PW_TRANSACTION.TRANS_CODE"),
            _optional_text(row["TRANS_DATE"]),
            _optional_int(row["TRANS_DATE_UNIX"]),
            _optional_text(row["PROCESS_ID"]),
            _optional_text(row["ITEM_CODE"]),
            quantity,
            valid,
            error,
            _optional_text(row["TRANS_STATUS"]),
            _optional_text(row["DELETED"]),
        ),
    )
    return bool(valid)


def _insert_transaction_definition(
    target: sqlite3.Connection, row: dict[str, Any]
) -> None:
    target.execute(
        "INSERT INTO wms_raw_transaction_definitions (trans_code, trans_name) VALUES (?, ?)",
        (
            _required_text(row["TRANS_CODE"], "PW_TRANSACTION_DEFINE.TRANS_CODE"),
            _optional_text(row["TRANS_NAME"]),
        ),
    )


def _insert_transaction_detail(target: sqlite3.Connection, row: dict[str, Any]) -> bool:
    quantity, valid, error = _quantity(row["QTY"])
    target.execute(
        """
        INSERT INTO wms_raw_transaction_details (
            source_id, create_date, edit_date, trans_id, item_lot_id,
            quantity_decimal, quantity_valid, quantity_error, product_id,
            production_lot_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _optional_int(row["ID"]),
            _optional_text(row["CREATE_DATE"]),
            _optional_text(row["EDIT_DATE"]),
            _required_text(row["TRANS_ID"], "PW_TRANS_DETAIL.TRANS_ID"),
            _optional_text(row["ITEM_LOT_ID"]),
            quantity,
            valid,
            error,
            _optional_text(row["PRODUCT_ID"]),
            _optional_text(row["LOT_ID"]),
        ),
    )
    return bool(valid)


def _latest_text(values: list[dict[str, Any]], column: str) -> str:
    normalized = [
        text
        for row in values
        if (text := _optional_text(row.get(column))) is not None
    ]
    return max(normalized) if normalized else ""


def _dataset_evidence(stats: ImportStats, staged: dict[str, list[dict[str, Any]]]) -> list[DatasetEvidence]:
    current_count = stats.candidate_rows["PW_CURRENT_ITEM"]
    legacy_count = stats.candidate_rows["PW_SNAPSHORT"]
    header_count = stats.candidate_rows["PW_TRANSACTION"]
    definition_count = stats.candidate_rows["PW_TRANSACTION_DEFINE"]
    detail_count = stats.candidate_rows["PW_TRANS_DETAIL"]

    current = DatasetEvidence(
        dataset=DATASET_CURRENT,
        status="PARTIAL",
        reason_code=REASON_UOM_UNAVAILABLE,
        source_tables="PW_CURRENT_ITEM",
        source_state="PRESENT_NONEMPTY",
        candidate_row_count=current_count,
        inserted_row_count=stats.inserted_rows["PW_CURRENT_ITEM"],
        invalid_quantity_row_count=stats.invalid_quantity_rows["PW_CURRENT_ITEM"],
        source_as_of=_latest_text(staged["PW_CURRENT_ITEM"], "TIME_UPDATE"),
        source_as_of_state="DERIVED_UNVERIFIED",
        source_as_of_basis=EXPECTED_CURRENT_BASIS,
        source_timezone="unverified",
        semantic_epoch=CURRENT_SEMANTIC_EPOCH,
        evidence_basis="allowlisted PW_CURRENT_ITEM rows with unique process/item grain",
    )
    legacy_invalid_count = stats.invalid_quantity_rows["PW_SNAPSHORT"]
    legacy = DatasetEvidence(
        dataset=DATASET_LEGACY,
        status=(
            "PARTIAL"
            if legacy_count and legacy_invalid_count
            else "AVAILABLE"
            if legacy_count
            else "SUPPRESSED"
        ),
        reason_code=(
            REASON_QUANTITY_EVIDENCE_INCOMPLETE
            if legacy_count and legacy_invalid_count
            else ""
            if legacy_count
            else REASON_DATASET_NOT_OBSERVED
        ),
        source_tables="PW_SNAPSHORT",
        source_state="PRESENT_NONEMPTY" if legacy_count else "NOT_OBSERVED_IN_EXPORT",
        candidate_row_count=legacy_count,
        inserted_row_count=stats.inserted_rows["PW_SNAPSHORT"],
        invalid_quantity_row_count=stats.invalid_quantity_rows["PW_SNAPSHORT"],
        source_as_of=_latest_text(staged["PW_SNAPSHORT"], "SNAPSHORT_DATE"),
        source_as_of_state="DERIVED_UNVERIFIED" if legacy_count else "UNAVAILABLE",
        source_as_of_basis=EXPECTED_LEGACY_BASIS,
        source_timezone="unverified",
        semantic_epoch=LEGACY_SEMANTIC_EPOCH,
        evidence_basis="allowlisted PW_SNAPSHORT legacy archive rows",
    )
    audit_count = header_count + definition_count + detail_count
    audit_invalid_count = (
        stats.invalid_quantity_rows["PW_TRANSACTION"]
        + stats.invalid_quantity_rows["PW_TRANS_DETAIL"]
    )
    observed_audit_tables = ",".join(
        table
        for table, count in (
            ("PW_TRANSACTION", header_count),
            ("PW_TRANSACTION_DEFINE", definition_count),
            ("PW_TRANS_DETAIL", detail_count),
        )
        if count
    )
    audit_source_as_of = _latest_text(staged["PW_TRANSACTION"], "TRANS_DATE")
    if header_count:
        audit_status = (
            "AVAILABLE"
            if definition_count
            and detail_count
            and not audit_invalid_count
            and audit_source_as_of
            else "PARTIAL"
        )
        audit_reason = (
            ""
            if audit_status == "AVAILABLE"
            else REASON_QUANTITY_EVIDENCE_INCOMPLETE
            if audit_invalid_count or not audit_source_as_of
            else REASON_TRANSACTION_SEMANTICS_UNAVAILABLE
        )
        audit_source_state = "PRESENT_NONEMPTY"
    elif audit_count:
        audit_status = "SUPPRESSED"
        audit_reason = REASON_TRANSACTION_HEADER_NOT_OBSERVED
        audit_source_state = "PARTIAL_SOURCE_OBSERVED"
    else:
        audit_status = "SUPPRESSED"
        audit_reason = REASON_DATASET_NOT_OBSERVED
        audit_source_state = "NOT_OBSERVED_IN_EXPORT"
    audit = DatasetEvidence(
        dataset=DATASET_AUDIT,
        status=audit_status,
        reason_code=audit_reason,
        source_tables=observed_audit_tables or "PW_TRANSACTION,PW_TRANSACTION_DEFINE,PW_TRANS_DETAIL",
        source_state=audit_source_state,
        candidate_row_count=audit_count,
        inserted_row_count=(
            stats.inserted_rows["PW_TRANSACTION"]
            + stats.inserted_rows["PW_TRANSACTION_DEFINE"]
            + stats.inserted_rows["PW_TRANS_DETAIL"]
        ),
        invalid_quantity_row_count=audit_invalid_count,
        source_as_of=audit_source_as_of,
        source_as_of_state=(
            "DERIVED_UNVERIFIED" if audit_source_as_of else "UNAVAILABLE"
        ),
        source_as_of_basis=EXPECTED_AUDIT_BASIS,
        source_timezone="unverified",
        semantic_epoch=AUDIT_SEMANTIC_EPOCH,
        evidence_basis="allowlisted raw transaction rows; no movement semantics inferred",
    )
    return [current, legacy, audit]


def _write_metadata(
    target: sqlite3.Connection,
    source_path: Path,
    plan: ImportPlan,
    imported_at: str,
) -> None:
    stats = plan.stats
    target.execute(
        """
        INSERT INTO import_batches (
            source_name, source_path, source_sha256, source_size_bytes,
            row_count, imported_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "MES_WMS_MKHC_PW",
            str(source_path),
            _sha256(source_path),
            source_path.stat().st_size,
            sum(stats.candidate_rows.values()),
            imported_at,
        ),
    )
    evidences = _dataset_evidence(stats, plan.staged)
    target.executemany(
        """
        INSERT INTO wms_dataset_evidence (
            dataset, status, reason_code, source_tables, source_state,
            candidate_row_count, inserted_row_count, invalid_quantity_row_count,
            source_as_of, source_as_of_state, source_as_of_basis,
            source_timezone, semantic_epoch, evidence_basis
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                e.dataset, e.status, e.reason_code, e.source_tables,
                e.source_state, e.candidate_row_count, e.inserted_row_count,
                e.invalid_quantity_row_count, e.source_as_of,
                e.source_as_of_state, e.source_as_of_basis,
                e.source_timezone, e.semantic_epoch, e.evidence_basis,
            )
            for e in evidences
        ],
    )
    quality = target.execute("SELECT * FROM v_wms_current_quality").fetchone()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "semantic_contract_version": SEMANTIC_CONTRACT_VERSION,
        "semantic_epoch": CURRENT_SEMANTIC_EPOCH,
        "source_schema": SOURCE_SCHEMA,
        "plant_id": PLANT_ID,
        "source_export_date": _source_export_date(source_path),
        "imported_at": imported_at,
        "current_row_count": quality[0],
        "valid_quantity_row_count": quality[1],
        "invalid_quantity_row_count": quality[2],
        "mapped_process_row_count": quality[3],
        "unmapped_process_row_count": quality[4],
        "distinct_item_count": quality[5],
        "distinct_process_code_count": quality[6],
    }
    for table in RAW_TABLE_COLUMNS:
        metadata[f"source_{table.lower()}_row_count"] = stats.candidate_rows[table]
        metadata[f"inserted_{table.lower()}_row_count"] = stats.inserted_rows[table]
        metadata[f"invalid_{table.lower()}_quantity_count"] = (
            stats.invalid_quantity_rows[table]
        )
    target.executemany(
        "INSERT INTO schema_metadata (key, value) VALUES (?, ?)",
        [(key, str(value)) for key, value in metadata.items()],
    )
    capability_statuses = capability_statuses_for_datasets(
        {e.dataset: (e.status, e.reason_code) for e in evidences}
    )
    target.executemany(
        """
        INSERT INTO wms_capability_status (
            capability, status, reason_code, evidence_basis, contract_version
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                capability,
                status,
                reason_code,
                "Phase 2C dataset evidence and semantic gate",
                SCHEMA_VERSION,
            )
            for capability, (status, reason_code) in capability_statuses.items()
        ],
    )


def _validate_parsed_row(parsed: ParsedRow) -> bool | None:
    row = parsed.values
    if parsed.table == "PW_PROCESS":
        _optional_int(row["ID"])
        _required_text(row["PROCESS_ID"], "PW_PROCESS.PROCESS_ID")
        return None
    if parsed.table == "PW_CURRENT_ITEM":
        _optional_int(row["ID"])
        _required_text(row["ITEM_CODE"], "PW_CURRENT_ITEM.ITEM_CODE")
        _required_text(row["PROCESS_ID"], "PW_CURRENT_ITEM.PROCESS_ID")
        _optional_int(row["TIME_UPDATE_UNIX"])
        return bool(_quantity(row["QTY"])[1])
    if parsed.table == "PW_SNAPSHORT":
        _optional_int(row["ID"])
        _required_text(row["SNAPSHORT_ID"], "PW_SNAPSHORT.SNAPSHORT_ID")
        _required_text(row["SNAPSHORT_DATE"], "PW_SNAPSHORT.SNAPSHORT_DATE")
        _required_text(row["ITEM_CODE"], "PW_SNAPSHORT.ITEM_CODE")
        _required_text(row["ITEM_LOT_ID"], "PW_SNAPSHORT.ITEM_LOT_ID")
        _required_text(row["PROCESS_ID"], "PW_SNAPSHORT.PROCESS_ID")
        return bool(_quantity(row["QTY"])[1])
    if parsed.table == "PW_TRANSACTION_DEFINE":
        _optional_int(row["ID"])
        _required_text(
            row["TRANS_CODE"], "PW_TRANSACTION_DEFINE.TRANS_CODE"
        )
        return None
    if parsed.table == "PW_TRANSACTION":
        _optional_int(row["ID"])
        _required_text(row["TRANS_ID"], "PW_TRANSACTION.TRANS_ID")
        _required_text(row["TRANS_CODE"], "PW_TRANSACTION.TRANS_CODE")
        _optional_int(row["TRANS_DATE_UNIX"])
        return bool(_quantity(row["QTY"])[1])
    if parsed.table == "PW_TRANS_DETAIL":
        _optional_int(row["ID"])
        _required_text(row["TRANS_ID"], "PW_TRANS_DETAIL.TRANS_ID")
        return bool(_quantity(row["QTY"])[1])
    raise ValueError(f"Bảng WMS ngoài allowlist: {parsed.table}")


def _build_import_plan(
    source_path: Path, schema_path: Path = DEFAULT_SCHEMA_PATH
) -> ImportPlan:
    """Parse and profile one source for both dry-run and atomic build."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy WMS export: {source_path}")
    schema_path.read_text(encoding="utf-8")

    staged: dict[str, list[dict[str, Any]]] = {
        table: [] for table in RAW_TABLE_COLUMNS
    }
    candidate_rows = {table: 0 for table in RAW_TABLE_COLUMNS}
    invalid_quantity_rows = {table: 0 for table in RAW_TABLE_COLUMNS}
    for parsed in iter_allowlisted_rows(source_path):
        candidate_rows[parsed.table] += 1
        staged[parsed.table].append(parsed.values)
        valid_quantity = _validate_parsed_row(parsed)
        if valid_quantity is False:
            invalid_quantity_rows[parsed.table] += 1

    if not candidate_rows["PW_CURRENT_ITEM"]:
        raise ValueError(
            "WMS export thiếu dataset bắt buộc có dữ liệu: PW_CURRENT_ITEM"
        )

    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for row in staged["PW_CURRENT_ITEM"]:
        key = (
            _required_text(row["PROCESS_ID"], "PW_CURRENT_ITEM.PROCESS_ID"),
            _required_text(row["ITEM_CODE"], "PW_CURRENT_ITEM.ITEM_CODE"),
        )
        if key in seen:
            duplicates += 1
        seen.add(key)
    if duplicates:
        raise ValueError(
            f"{REASON_CURRENT_GRAIN_DUPLICATE}: duplicate_count={duplicates}"
        )

    stats = ImportStats(
        candidate_rows=candidate_rows,
        inserted_rows={table: 0 for table in RAW_TABLE_COLUMNS},
        invalid_quantity_rows=invalid_quantity_rows,
    )
    return ImportPlan(staged=staged, stats=stats)


def _populate_database(
    target: sqlite3.Connection,
    source_path: Path,
    schema_path: Path,
    plan: ImportPlan,
    imported_at: str,
) -> None:
    """Populate and validate one staged plan in the supplied SQLite connection."""
    stats = plan.stats
    target.execute("PRAGMA foreign_keys = ON")
    target.executescript(schema_path.read_text(encoding="utf-8"))
    with target:
        for row in plan.staged["PW_PROCESS"]:
            _insert_process(target, row)
            stats.inserted_rows["PW_PROCESS"] += 1
        for row in plan.staged["PW_CURRENT_ITEM"]:
            valid = _insert_current_balance(target, row)
            stats.inserted_rows["PW_CURRENT_ITEM"] += 1
            assert bool(valid) == bool(_quantity(row["QTY"])[1])
        for row in plan.staged["PW_SNAPSHORT"]:
            valid = _insert_snapshot_record(target, row)
            stats.inserted_rows["PW_SNAPSHORT"] += 1
            assert bool(valid) == bool(_quantity(row["QTY"])[1])
        for row in plan.staged["PW_TRANSACTION_DEFINE"]:
            _insert_transaction_definition(target, row)
            stats.inserted_rows["PW_TRANSACTION_DEFINE"] += 1
        for row in plan.staged["PW_TRANSACTION"]:
            valid = _insert_transaction(target, row)
            stats.inserted_rows["PW_TRANSACTION"] += 1
            assert bool(valid) == bool(_quantity(row["QTY"])[1])
        for row in plan.staged["PW_TRANS_DETAIL"]:
            valid = _insert_transaction_detail(target, row)
            stats.inserted_rows["PW_TRANS_DETAIL"] += 1
            assert bool(valid) == bool(_quantity(row["QTY"])[1])
        _write_metadata(target, source_path, plan, imported_at)

    foreign_key_errors = target.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_errors:
        raise RuntimeError(f"Lỗi khóa ngoại WMS: {foreign_key_errors[:5]}")
    integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity_check WMS thất bại: {integrity}")
    compatibility = validate_snapshot_connection(target)
    if not compatibility["compatible"]:
        raise RuntimeError(
            "WMS snapshot vừa build không tương thích: "
            + json.dumps(compatibility, ensure_ascii=False, sort_keys=True)
        )


def validate_source(
    source_path: Path, schema_path: Path = DEFAULT_SCHEMA_PATH
) -> ImportStats:
    """Run the full v4 import contract without creating or replacing a DB file."""
    plan = _build_import_plan(source_path, schema_path)
    with sqlite3.connect(":memory:") as target:
        _populate_database(
            target,
            source_path,
            schema_path,
            plan,
            datetime.now(timezone.utc).isoformat(),
        )
    return plan.stats


def validate_snapshot(db_path: Path) -> dict[str, Any]:
    """Validate an existing SQLite snapshot without mutating it."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy WMS snapshot: {db_path}")
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        return validate_snapshot_connection(connection)


def _prepare_database_plan(
    source_path: Path, schema_path: Path, db_path: Path
) -> tuple[ImportPlan, Path]:
    plan = _build_import_plan(source_path, schema_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{db_path.name}.", suffix=".tmp", dir=db_path.parent, delete=False
    )
    temp_path = Path(handle.name)
    handle.close()
    target = sqlite3.connect(temp_path)
    try:
        _populate_database(
            target,
            source_path,
            schema_path,
            plan,
            datetime.now(timezone.utc).isoformat(),
        )
        target.execute("ANALYZE")
        target.execute("PRAGMA optimize")
        target.commit()
    except Exception:
        target.close()
        temp_path.unlink(missing_ok=True)
        raise
    else:
        target.close()
    return plan, temp_path


def _promote_database(temp_path: Path, db_path: Path) -> None:
    os.replace(temp_path, db_path)
    os.chmod(db_path, 0o644)


def build_database(
    source_path: Path, schema_path: Path, db_path: Path
) -> dict[str, int]:
    plan, temp_path = _prepare_database_plan(source_path, schema_path, db_path)
    try:
        _promote_database(temp_path, db_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return plan.stats.inserted_rows


def _stats_report(plan: ImportPlan, *, ok: bool = True) -> dict[str, Any]:
    evidence = _dataset_evidence(plan.stats, plan.staged)
    return {
        "report_format_version": "1",
        "ok": ok,
        "schema_version": SCHEMA_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "semantic_contract_version": SEMANTIC_CONTRACT_VERSION,
        "datasets": [
            {
                "dataset": item.dataset,
                "status": item.status,
                "reason_code": item.reason_code,
                "source_state": item.source_state,
                "candidate_row_count": item.candidate_row_count,
                "inserted_row_count": item.inserted_row_count,
                "invalid_quantity_row_count": item.invalid_quantity_row_count,
                "source_as_of": item.source_as_of,
                "source_as_of_state": item.source_as_of_state,
                "source_as_of_basis": item.source_as_of_basis,
                "source_timezone": item.source_timezone,
                "semantic_epoch": item.semantic_epoch,
            }
            for item in evidence
        ],
        "capabilities": [
            {
                "capability": capability,
                "status": status,
                "reason_code": reason_code,
            }
            for capability, (status, reason_code) in sorted(
                capability_statuses_for_datasets(
                    {
                        item.dataset: (item.status, item.reason_code)
                        for item in evidence
                    }
                ).items()
            )
        ],
        "tables": {
            table: {
                "candidate_rows": plan.stats.candidate_rows[table],
                "invalid_quantity_rows": plan.stats.invalid_quantity_rows[table],
            }
            for table in RAW_TABLE_COLUMNS
        },
        "gates": {
            "required_current_dataset": True,
            "current_balance_duplicate_count": 0,
        },
    }


def _error_report(exc: Exception) -> dict[str, Any]:
    message = str(exc)
    reason = (
        REASON_CURRENT_GRAIN_DUPLICATE
        if REASON_CURRENT_GRAIN_DUPLICATE in message
        else "IMPORT_VALIDATION_ERROR"
    )
    report = {
        "report_format_version": "1",
        "ok": False,
        "schema_version": SCHEMA_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "semantic_contract_version": SEMANTIC_CONTRACT_VERSION,
        "error": {"reason_code": reason},
    }
    if reason == REASON_CURRENT_GRAIN_DUPLICATE:
        match = re.search(r"duplicate_count=(\d+)", message)
        report["gates"] = {
            "current_balance_duplicate_count": (
                int(match.group(1)) if match else 1
            )
        }
    return report


def _write_json_report(destination: str, report: dict[str, Any]) -> None:
    content = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if destination == "-":
        sys.stdout.write(content)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False,
        mode="w", encoding="utf-8",
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(content)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    try:
        if args.validate_snapshot is not None:
            report = validate_snapshot(args.validate_snapshot)
            if args.report_json:
                _write_json_report(
                    args.report_json,
                    {
                        "report_format_version": "1",
                        "ok": bool(report["compatible"]),
                        **report,
                    },
                )
            else:
                print(f"Snapshot: {args.validate_snapshot}")
                print(f"Compatible: {str(report['compatible']).lower()}")
                print(f"Schema version: {report['schema_version'] or 'missing'}")
                if report["reason_codes"]:
                    print("Reason codes: " + ", ".join(report["reason_codes"]))
            return 0 if report["compatible"] else 1

        if args.dry_run:
            plan = _build_import_plan(args.source, args.schema)
            with sqlite3.connect(":memory:") as target:
                _populate_database(
                    target,
                    args.source,
                    args.schema,
                    plan,
                    datetime.now(timezone.utc).isoformat(),
                )
            if args.report_json:
                _write_json_report(args.report_json, _stats_report(plan))
            else:
                print(f"Source: {args.source}")
                print("Dry run: no SQLite file was created or modified")
                for table, count in plan.stats.candidate_rows.items():
                    invalid = plan.stats.invalid_quantity_rows[table]
                    print(f"{table}: candidates={count} invalid_quantity={invalid}")
            return 0

        plan, temp_path = _prepare_database_plan(args.source, args.schema, args.db)
        try:
            if args.report_json:
                _write_json_report(args.report_json, _stats_report(plan))
            _promote_database(temp_path, args.db)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        if not args.report_json:
            print(f"Database: {args.db}")
            for table, count in plan.stats.inserted_rows.items():
                print(f"{table}: {count}")
        return 0
    except Exception as exc:
        if args.report_json:
            try:
                _write_json_report(args.report_json, _error_report(exc))
            except Exception:
                pass
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
