#!/usr/bin/env python3
"""Tạo SQLite MES tối ưu truy vấn từ ba bản dump SQL thô."""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "database" / "raw"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "database" / "schema" / "mes.sql"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "mes.sqlite"
SCHEMA_VERSION = "1"

RAW_TABLE_COLUMNS = {
    "M_LOT": (
        "ID", "CREATE_DATE", "EDIT_DATE", "PRODUCT_ID", "LOT_ID", "PT_ID",
        "PT_VERSION_ID", "ROUTE_ID", "LOT_TYPE", "STATUS", "IS_RELEASE",
        "SALE_ORDER_ID", "BOARD_LOT", "SHEET_LOT", "PREV_STATUS", "DATE_CODE",
        "PRODUCE_DATE", "PRODUCE_DATE_PROCESS_ID", "PRODUCE_DATE_PROCESS_ORDER",
        "IS_RELEASE_SPLIT_LOT", "PCS_LOT", "CREATE_TIME_UNIX",
        "RELEASE_DATE_UNIX", "RELEASE_DATE", "PRODUCTION_TYPE", "USER_ID",
        "PREV_RELEASE", "PRODUCTION_PERIOD_TYPE", "USER_ID_UPDATE",
        "TIME_UPDATE_UNIX", "TIME_UPDATE",
    ),
    "D_ERROR": (
        "ID", "EDIT_DATE", "CREATE_DATE", "LOT_ID", "ROUTE_ID", "PROCESS_ID",
        "PROCESS_ORDER", "ERROR_TYPE", "ERROR_ID", "QTY", "USER_ID", "NOTE",
        "ERROR_PROCESS_TYPE", "LOT_ID_SPLIT", "PROCESS_ID_CREATE",
        "PROCESS_ORDER_CREATE", "ERROR_TIME_UNIX", "ERROR_TIME",
        "ERROR_JUDGEMENT",
    ),
    "P_ERROR": (
        "ID", "CREATE_DATE", "EDIT_DATE", "ERROR_ID", "ERROR_NAME", "ERROR_TYPE",
        "STATUS", "NOTE", "DELETED", "PROCESS_ID", "ERROR_NAME_VI",
        "ERROR_NAME_JA", "ERROR_NAME_EN", "ERROR_NAME_CH", "PRIORITY_ERROR",
        "USER_ID",
    ),
}

FILE_PATTERNS = {
    "M_LOT": "M_LOT_*.sql",
    "D_ERROR": "D_ERROR_*.sql",
    "P_ERROR": "P_ERROR_*.sql",
}


@dataclass(frozen=True)
class CatalogIndexes:
    exact: dict[tuple[str, str, str], int]
    by_error_and_type: dict[tuple[str, str], int]
    by_error: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


def _latest_source(source_dir: Path, pattern: str) -> Path:
    matches = sorted(source_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Không tìm thấy file {pattern} trong {source_dir}")
    return matches[-1]


def discover_sources(source_dir: Path) -> dict[str, Path]:
    return {
        table: _latest_source(source_dir, pattern)
        for table, pattern in FILE_PATTERNS.items()
    }


def _create_raw_table(connection: sqlite3.Connection, table: str) -> None:
    columns = RAW_TABLE_COLUMNS[table]
    definition = ", ".join(f'"{column}"' for column in columns)
    connection.execute(f'CREATE TABLE "{table}" ({definition})')


def _load_raw_dump(
    connection: sqlite3.Connection,
    table: str,
    source_path: Path,
) -> None:
    sql = source_path.read_text(encoding="utf-8-sig")
    qualified_name = f"MES_DATA_MKHC.{table}"
    if qualified_name not in sql:
        raise ValueError(f"{source_path.name} không chứa dữ liệu cho {qualified_name}")
    sql = sql.replace("MES_DATA_MKHC.", "").replace("TIMESTAMP'", "'")
    connection.executescript(sql)


def load_raw_sources(sources: dict[str, Path]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    for table in RAW_TABLE_COLUMNS:
        _create_raw_table(connection, table)
        _load_raw_dump(connection, table, sources[table])
    return connection


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return unicodedata.normalize("NFC", text) if text else None


def _required_text(value: Any, field: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"Trường bắt buộc {field} đang rỗng")
    return text


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(str(value).strip())


def _required_non_negative_int(value: Any, field: str) -> int:
    number = _optional_int(value)
    if number is None or number < 0:
        raise ValueError(f"{field} phải là số nguyên không âm, nhận được {value!r}")
    return number


def _rows(connection: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    connection.row_factory = sqlite3.Row
    return connection.execute(f'SELECT * FROM "{table}"').fetchall()


def _catalog_rank(row: sqlite3.Row, position: int) -> tuple[Any, ...]:
    return (
        bool(_optional_text(row["ERROR_NAME_VI"])),
        bool(_optional_text(row["ERROR_NAME"])),
        _optional_text(row["EDIT_DATE"]) or "",
        _optional_text(row["CREATE_DATE"]) or "",
        _optional_int(row["ID"]) or -1,
        position,
    )


def _canonical_catalog_positions(rows: list[sqlite3.Row]) -> set[int]:
    grouped: dict[tuple[str, str, str], list[tuple[int, sqlite3.Row]]] = defaultdict(list)
    for position, row in enumerate(rows):
        key = (
            _required_text(row["ERROR_ID"], "P_ERROR.ERROR_ID"),
            _required_text(row["PROCESS_ID"], "P_ERROR.PROCESS_ID"),
            _required_text(row["ERROR_TYPE"], "P_ERROR.ERROR_TYPE"),
        )
        grouped[key].append((position, row))
    return {
        max(candidates, key=lambda item: _catalog_rank(item[1], item[0]))[0]
        for candidates in grouped.values()
    }


def _insert_lots(
    target: sqlite3.Connection,
    rows: Iterable[sqlite3.Row],
) -> dict[str, int]:
    lot_pk_by_id: dict[str, int] = {}
    sql = """
        INSERT INTO lots (
            source_id, create_date, edit_date, product_id, lot_id, pt_id,
            pt_version_id, route_id, lot_type, status, is_release, sale_order_id,
            board_lot, sheet_lot, prev_status, date_code, produce_date,
            produce_date_process_id, produce_date_process_order,
            is_release_split_lot, pcs_lot, create_time_unix, release_date_unix,
            release_date, production_type, user_id, prev_release,
            production_period_type, user_id_update, time_update_unix, time_update
        ) VALUES ({})
    """.format(",".join("?" for _ in range(31)))
    for row in rows:
        lot_id = _required_text(row["LOT_ID"], "M_LOT.LOT_ID")
        values = (
            _required_non_negative_int(row["ID"], "M_LOT.ID"),
            _optional_text(row["CREATE_DATE"]), _optional_text(row["EDIT_DATE"]),
            _required_text(row["PRODUCT_ID"], "M_LOT.PRODUCT_ID"), lot_id,
            _optional_text(row["PT_ID"]), _optional_text(row["PT_VERSION_ID"]),
            _optional_text(row["ROUTE_ID"]), _optional_text(row["LOT_TYPE"]),
            _optional_text(row["STATUS"]), _optional_text(row["IS_RELEASE"]),
            _optional_text(row["SALE_ORDER_ID"]), _optional_int(row["BOARD_LOT"]),
            _optional_int(row["SHEET_LOT"]), _optional_text(row["PREV_STATUS"]),
            _optional_text(row["DATE_CODE"]), _optional_text(row["PRODUCE_DATE"]),
            _optional_text(row["PRODUCE_DATE_PROCESS_ID"]),
            _optional_int(row["PRODUCE_DATE_PROCESS_ORDER"]),
            _optional_text(row["IS_RELEASE_SPLIT_LOT"]), _optional_int(row["PCS_LOT"]),
            _optional_int(row["CREATE_TIME_UNIX"]), _optional_int(row["RELEASE_DATE_UNIX"]),
            _optional_text(row["RELEASE_DATE"]), _optional_text(row["PRODUCTION_TYPE"]),
            _optional_text(row["USER_ID"]), _optional_text(row["PREV_RELEASE"]),
            _optional_text(row["PRODUCTION_PERIOD_TYPE"]),
            _optional_text(row["USER_ID_UPDATE"]), _optional_int(row["TIME_UPDATE_UNIX"]),
            _optional_text(row["TIME_UPDATE"]),
        )
        cursor = target.execute(sql, values)
        lot_pk_by_id[lot_id] = int(cursor.lastrowid)
    return lot_pk_by_id


def _insert_catalog(
    target: sqlite3.Connection,
    rows: list[sqlite3.Row],
) -> CatalogIndexes:
    canonical_positions = _canonical_catalog_positions(rows)
    catalog_pk_by_key: dict[tuple[str, str, str], int] = {}
    fallback_candidates_by_error_and_type: dict[
        tuple[str, str], list[tuple[int, sqlite3.Row, int]]
    ] = defaultdict(list)
    fallback_candidates_by_error: dict[str, list[tuple[int, sqlite3.Row, int]]] = defaultdict(list)
    sql = """
        INSERT INTO error_catalog (
            source_id, create_date, edit_date, error_id, error_name, error_type,
            status, note, deleted, process_id, error_name_vi, error_name_ja,
            error_name_en, error_name_ch, priority_error, user_id, is_canonical
        ) VALUES ({})
    """.format(",".join("?" for _ in range(17)))
    for position, row in enumerate(rows):
        key = (
            _required_text(row["ERROR_ID"], "P_ERROR.ERROR_ID"),
            _required_text(row["PROCESS_ID"], "P_ERROR.PROCESS_ID"),
            _required_text(row["ERROR_TYPE"], "P_ERROR.ERROR_TYPE"),
        )
        is_canonical = int(position in canonical_positions)
        cursor = target.execute(
            sql,
            (
                _optional_int(row["ID"]), _optional_text(row["CREATE_DATE"]),
                _optional_text(row["EDIT_DATE"]), key[0],
                _optional_text(row["ERROR_NAME"]), key[2],
                _optional_text(row["STATUS"]), _optional_text(row["NOTE"]),
                _optional_text(row["DELETED"]), key[1],
                _optional_text(row["ERROR_NAME_VI"]),
                _optional_text(row["ERROR_NAME_JA"]),
                _optional_text(row["ERROR_NAME_EN"]),
                _optional_text(row["ERROR_NAME_CH"]),
                _optional_text(row["PRIORITY_ERROR"]),
                _optional_text(row["USER_ID"]), is_canonical,
            ),
        )
        if is_canonical:
            catalog_pk = int(cursor.lastrowid)
            catalog_pk_by_key[key] = catalog_pk
            fallback_candidates_by_error_and_type[(key[0], key[2])].append(
                (position, row, catalog_pk)
            )
            fallback_candidates_by_error[key[0]].append((position, row, catalog_pk))
    return CatalogIndexes(
        exact=catalog_pk_by_key,
        by_error_and_type={
            key: max(
                candidates,
                key=lambda item: _catalog_rank(item[1], item[0]),
            )[2]
            for key, candidates in fallback_candidates_by_error_and_type.items()
        },
        by_error={
            key: max(
                candidates,
                key=lambda item: _catalog_rank(item[1], item[0]),
            )[2]
            for key, candidates in fallback_candidates_by_error.items()
        },
    )


def _catalog_pk_for_event(
    key: tuple[str, str, str],
    catalog_indexes: CatalogIndexes,
) -> int | None:
    error_id, _process_id, error_type = key
    return (
        catalog_indexes.exact.get(key)
        or catalog_indexes.by_error_and_type.get((error_id, error_type))
        or catalog_indexes.by_error.get(error_id)
    )


def _insert_error_events(
    target: sqlite3.Connection,
    rows: Iterable[sqlite3.Row],
    lot_pk_by_id: dict[str, int],
    catalog_indexes: CatalogIndexes,
) -> None:
    sql = """
        INSERT INTO error_events (
            source_id, lot_pk, error_catalog_pk, edit_date, create_date, lot_id,
            route_id, process_id, process_order, error_type, error_id, quantity,
            user_id, note, error_process_type, lot_id_split, process_id_create,
            process_order_create, error_time_unix, error_time, error_judgement
        ) VALUES ({})
    """.format(",".join("?" for _ in range(21)))
    for row in rows:
        lot_id = _required_text(row["LOT_ID"], "D_ERROR.LOT_ID")
        key = (
            _required_text(row["ERROR_ID"], "D_ERROR.ERROR_ID"),
            _required_text(row["PROCESS_ID"], "D_ERROR.PROCESS_ID"),
            _required_text(row["ERROR_TYPE"], "D_ERROR.ERROR_TYPE"),
        )
        target.execute(
            sql,
            (
                _optional_int(row["ID"]), lot_pk_by_id.get(lot_id),
                _catalog_pk_for_event(key, catalog_indexes), _optional_text(row["EDIT_DATE"]),
                _optional_text(row["CREATE_DATE"]), lot_id,
                _optional_text(row["ROUTE_ID"]), key[1],
                _optional_int(row["PROCESS_ORDER"]), key[2], key[0],
                _required_non_negative_int(row["QTY"], "D_ERROR.QTY"),
                _optional_text(row["USER_ID"]), _optional_text(row["NOTE"]),
                _optional_text(row["ERROR_PROCESS_TYPE"]),
                _optional_text(row["LOT_ID_SPLIT"]),
                _optional_text(row["PROCESS_ID_CREATE"]),
                _optional_int(row["PROCESS_ORDER_CREATE"]),
                _optional_int(row["ERROR_TIME_UNIX"]),
                _optional_text(row["ERROR_TIME"]),
                _optional_text(row["ERROR_JUDGEMENT"]),
            ),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_metadata(
    target: sqlite3.Connection,
    sources: dict[str, Path],
    raw_counts: dict[str, int],
    imported_at: str,
) -> None:
    for table, path in sources.items():
        target.execute(
            """
            INSERT INTO import_batches (
                source_name, source_path, source_sha256, source_size_bytes,
                row_count, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (table, str(path), _sha256(path), path.stat().st_size, raw_counts[table], imported_at),
        )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "imported_at": imported_at,
        "lot_count": target.execute("SELECT COUNT(*) FROM lots").fetchone()[0],
        "error_event_count": target.execute("SELECT COUNT(*) FROM error_events").fetchone()[0],
        "error_catalog_count": target.execute("SELECT COUNT(*) FROM error_catalog").fetchone()[0],
        "orphan_error_event_count": target.execute(
            "SELECT COUNT(*) FROM error_events WHERE lot_pk IS NULL"
        ).fetchone()[0],
        "unmapped_error_name_count": target.execute(
            "SELECT COUNT(*) FROM error_events WHERE error_catalog_pk IS NULL"
        ).fetchone()[0],
    }
    target.executemany(
        "INSERT INTO schema_metadata (key, value) VALUES (?, ?)",
        [(key, str(value)) for key, value in metrics.items()],
    )


def build_database(
    source_dir: Path,
    schema_path: Path,
    db_path: Path,
) -> dict[str, int]:
    sources = discover_sources(source_dir)
    raw = load_raw_sources(sources)
    raw_rows = {table: _rows(raw, table) for table in RAW_TABLE_COLUMNS}
    raw_counts = {table: len(rows) for table, rows in raw_rows.items()}

    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{db_path.name}.", suffix=".tmp", dir=db_path.parent, delete=False
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()
    try:
        target = sqlite3.connect(temp_path)
        try:
            target.execute("PRAGMA foreign_keys = ON")
            target.executescript(schema_path.read_text(encoding="utf-8"))
            with target:
                lot_pk_by_id = _insert_lots(target, raw_rows["M_LOT"])
                catalog_pk_by_key = _insert_catalog(target, raw_rows["P_ERROR"])
                _insert_error_events(
                    target,
                    raw_rows["D_ERROR"],
                    lot_pk_by_id,
                    catalog_pk_by_key,
                )
                _write_metadata(
                    target,
                    sources,
                    raw_counts,
                    datetime.now(timezone.utc).isoformat(),
                )
            foreign_key_errors = target.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise RuntimeError(f"Lỗi khóa ngoại: {foreign_key_errors[:5]}")
            integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"SQLite integrity_check thất bại: {integrity}")
            target.execute("ANALYZE")
            target.execute("PRAGMA optimize")
            target.commit()
        finally:
            target.close()
            raw.close()
        os.replace(temp_path, db_path)
        os.chmod(db_path, 0o644)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return {
        "lots": raw_counts["M_LOT"],
        "error_events": raw_counts["D_ERROR"],
        "error_catalog": raw_counts["P_ERROR"],
    }


def main() -> int:
    args = parse_args()
    counts = build_database(args.source_dir, args.schema, args.db)
    print(f"Database:      {args.db}")
    print(f"Lots:          {counts['lots']}")
    print(f"Error events:  {counts['error_events']}")
    print(f"Error catalog: {counts['error_catalog']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
