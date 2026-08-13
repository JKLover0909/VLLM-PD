import json
import sqlite3
from pathlib import Path

import scripts.import_mes_wms as importer

import pytest

from scripts.import_mes_wms import (
    build_database,
    iter_allowlisted_rows,
    validate_snapshot,
    validate_source,
)


def _timestamp(value: str) -> str:
    return f"to_timestamp('{value}','DD-MON-RR HH.MI.SSXFF AM')"


def _insert(table: str, columns: str, values: str) -> str:
    return f"Insert into MES_WMS_MKHC.{table} ({columns}) values ({values});"


def _write_dump(path: Path, *, invalid_quantity: bool = False) -> None:
    current_qty = "'bad-value'" if invalid_quantity else "'12.50'"
    lines = [
        "-- File created - Tuesday-July-28-2026",
        "CREATE DATABASE LINK DBLINK_TEST CONNECT TO USER IDENTIFIED BY VALUES 'secret';",
        "CREATE OR REPLACE PROCEDURE BAD AS BEGIN DELETE FROM PW_CURRENT_ITEM; END;",
        _insert(
            "PW_PROCESS",
            "ID,CREATE_DATE,EDIT_DATE,PROCESS_ID,PROCESS_NAME,PROCESS_PHYSICAL_ID,STATUS,NOTE,IS_CHECK_MATERIAL",
            f"1,{_timestamp('01-JUL-26 08.00.00.000000000 AM')},null,'PROC-A','Process A','PHYS-A','1','private','Y'",
        ),
        _insert(
            "PW_CURRENT_ITEM",
            "ID,CREATE_DATE,EDIT_DATE,ITEM_CODE,ITEM_LOT_ID,QTY,TIME_UPDATE,TIME_UPDATE_UNIX,TRANS_ID,PROCESS_ID",
            f"2,{_timestamp('01-JUL-26 08.00.00.000000000 AM')},null,'ITEM-A','-',{current_qty},{_timestamp('27-JUL-26 11.00.00.000000000 PM')},'1785193200','T-1','PROC-A'",
        ),
        _insert(
            "PW_CURRENT_ITEM",
            "ID,CREATE_DATE,EDIT_DATE,ITEM_CODE,ITEM_LOT_ID,QTY,TIME_UPDATE,TIME_UPDATE_UNIX,TRANS_ID,PROCESS_ID",
            f"3,{_timestamp('01-JUL-26 08.00.00.000000000 AM')},null,'ITEM-A','-','2.25',{_timestamp('27-JUL-26 11.30.00.000000000 PM')},'1785195000','T-2','PROC-X'",
        ),
        _insert(
            "PW_CURRENT_ITEM_TEST",
            "ID,CREATE_DATE,EDIT_DATE,ITEM_CODE,ITEM_LOT_ID,QTY,TIME_UPDATE,TIME_UPDATE_UNIX,TRANS_ID,PROCESS_ID",
            "99,null,null,'ITEM-TEST','-','999',null,null,null,'PROC-A'",
        ),
        _insert(
            "PW_SNAPSHORT",
            "ID,CREATE_DATE,EDIT_DATE,SNAPSHORT_ID,SNAPSHORT_DATE,ITEM_CODE,QTY,PROCESS_ID,ITEM_LOT_ID",
            f"6,null,null,'SNAP-1',{_timestamp('27-JUL-26 10.00.00.000000000 PM')},'ITEM-A','9.5','PROC-A','LOT-MATERIAL'",
        ),
        _insert(
            "PW_TRANSACTION_DEFINE",
            "ID,CREATE_DATE,EDIT_DATE,TRANS_CODE,TRANS_NAME,NOTE",
            "1,null,null,'001','Import',null",
        ),
        _insert(
            "PW_TRANSACTION",
            "ID,CREATE_DATE,EDIT_DATE,TRANS_ID,TRANS_CODE,TRANS_DATE,TRANS_DATE_UNIX,PROCESS_ID,ITEM_CODE,QTY,TRANS_STATUS,RELATE_ID,USER_ID,NOTE,VALUE_1,VALUE_2,VALUE_3,CREADIT_CODE,DEBIT_CODE,DELETED",
            f"4,null,null,'T-1','001',{_timestamp('27-JUL-26 11.00.00.000000000 PM')},'1785193200','PROC-A','ITEM-A','12.50','1',null,'private-user','private-note',null,null,null,null,null,'N'",
        ),
        _insert(
            "PW_TRANS_DETAIL",
            "ID,CREATE_DATE,EDIT_DATE,TRANS_ID,ITEM_LOT_ID,QTY,PRODUCT_ID,LOT_ID,NOTE,ATTACH_PATH",
            "5,null,null,'T-1','LOT-MATERIAL','12.50','PRODUCT-X','LOT-X','private-note','/private/path'",
        ),
    ]
    path.write_text("\n".join(lines), encoding="latin-1")


def test_importer_builds_separate_snapshot_and_ignores_unsafe_sql(tmp_path):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    counts = build_database(source, schema, db_path)

    assert counts == {
        "PW_CURRENT_ITEM": 2,
        "PW_PROCESS": 1,
        "PW_SNAPSHORT": 1,
        "PW_TRANSACTION": 1,
        "PW_TRANSACTION_DEFINE": 1,
        "PW_TRANS_DETAIL": 1,
    }
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT process_id, process_name, item_code, quantity_decimal,
                   process_mapped, latest_update
            FROM v_wms_current_balance_by_process_item
            WHERE process_id = 'PROC-A'
            """
        ).fetchone()
        unmapped = connection.execute(
            "SELECT COUNT(*) FROM wms_current_balances WHERE process_pk IS NULL"
        ).fetchone()[0]
        metadata = dict(connection.execute("SELECT key, value FROM schema_metadata"))
        columns = {
            item[1]
            for item in connection.execute(
                "PRAGMA table_info(wms_raw_transaction_headers)"
            )
        }
        object_names = {
            item[0]
            for item in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }

    assert row == (
        "PROC-A",
        "Process A",
        "ITEM-A",
        "12.5",
        1,
        "2026-07-27 23:00:00",
    )
    assert unmapped == 1
    assert metadata["plant_id"] == "MKHC"
    assert metadata["source_schema"] == "MES_WMS_MKHC"
    assert metadata["source_pw_current_item_row_count"] == "2"
    assert "PW_CURRENT_ITEM_TEST" not in object_names
    assert "user_id" not in columns
    assert "note" not in columns


def test_importer_quarantines_invalid_quantity_instead_of_zero(tmp_path):
    source = tmp_path / "wms.sql"
    _write_dump(source, invalid_quantity=True)
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    build_database(source, schema, db_path)

    with sqlite3.connect(db_path) as connection:
        invalid = connection.execute(
            """
            SELECT quantity_decimal, quantity_valid, quantity_error
            FROM wms_current_balances
            WHERE process_id = 'PROC-A'
            """
        ).fetchone()
        metadata = dict(connection.execute("SELECT key, value FROM schema_metadata"))

    assert invalid == (None, 0, "not_numeric")
    assert metadata["invalid_quantity_row_count"] == "1"
    assert metadata["valid_quantity_row_count"] == "1"


def test_importer_rejects_duplicate_current_grain_without_replacing_target(tmp_path):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    db_path = tmp_path / "mes_wms.sqlite"
    db_path.write_bytes(b"previous-snapshot")
    content = source.read_text(encoding="latin-1")
    duplicate = _insert(
        "PW_CURRENT_ITEM",
        "ID,CREATE_DATE,EDIT_DATE,ITEM_CODE,ITEM_LOT_ID,QTY,TIME_UPDATE,TIME_UPDATE_UNIX,TRANS_ID,PROCESS_ID",
        f"6,null,null,'ITEM-A','-','0.1',{_timestamp('27-JUL-26 11.45.00.000000000 PM')},'1785195900','T-3','PROC-A'",
    )
    source.write_text(content + "\n" + duplicate, encoding="latin-1")
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    with pytest.raises(ValueError, match="CURRENT_BALANCE_GRAIN_DUPLICATE"):
        validate_source(source, schema)
    with pytest.raises(ValueError, match="CURRENT_BALANCE_GRAIN_DUPLICATE"):
        build_database(source, schema, db_path)

    assert db_path.read_bytes() == b"previous-snapshot"


def test_phase2c_snapshot_is_compatible_and_has_capability_contract(tmp_path):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    build_database(source, schema, db_path)
    report = validate_snapshot(db_path)

    assert report["compatible"] is True
    assert report["schema_version"] == "4"
    with sqlite3.connect(db_path) as connection:
        capabilities = dict(
            connection.execute(
                "SELECT capability, reason_code FROM wms_capability_status"
            )
        )
        metadata = dict(
            connection.execute("SELECT key, value FROM schema_metadata")
        )
        evidence = {
            row[0]: row[1:]
            for row in connection.execute(
                """
                SELECT dataset, status, source_state, source_as_of,
                       source_as_of_basis, semantic_epoch
                FROM wms_dataset_evidence
                """
            )
        }
        archive = connection.execute(
            """
            SELECT archive_id, archive_date, item_code, item_lot_id,
                   process_id, quantity_decimal
            FROM wms_legacy_archive_records
            """
        ).fetchone()
        object_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }

    assert capabilities["MIN_STOCK"] == "MIN_STOCK_CONTRACT_UNVERIFIED"
    assert capabilities["TREND"] == "SNAPSHOT_HISTORY_NOT_COMPARABLE"
    assert capabilities["CURRENT_LOT_LOOKUP"] == "CURRENT_GRAIN_HAS_NO_MEANINGFUL_LOT"
    assert capabilities["LEGACY_ARCHIVE_EXACT_KEY_QUERY"] == ""
    assert capabilities["RAW_TRANSACTION_AUDIT_QUERY"] == ""
    assert metadata["data_contract_version"] == "wms-current-balance-v1"
    assert metadata["semantic_contract_version"] == "wms-phase2c-v1"
    assert evidence["CURRENT_BALANCE"] == (
        "PARTIAL",
        "PRESENT_NONEMPTY",
        "2026-07-27 23:30:00",
        "MAX(PW_CURRENT_ITEM.TIME_UPDATE)",
        "CURRENT_POST_2026_01_15",
    )
    assert evidence["LEGACY_ARCHIVE"] == (
        "AVAILABLE",
        "PRESENT_NONEMPTY",
        "2026-07-27 22:00:00",
        "MAX(PW_SNAPSHORT.SNAPSHORT_DATE)",
        "LEGACY_PRE_2026_01_15",
    )
    assert archive == (
        "SNAP-1",
        "2026-07-27 22:00:00",
        "ITEM-A",
        "LOT-MATERIAL",
        "PROC-A",
        "9.5",
    )
    assert "v_wms_current_exact_key" not in object_names
    assert "v_wms_completed_movements" not in object_names


def test_dry_run_validates_source_without_creating_database(tmp_path):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    db_path = tmp_path / "must-not-exist.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    stats = validate_source(source, schema)

    assert stats.candidate_rows["PW_CURRENT_ITEM"] == 2
    assert stats.candidate_rows["PW_SNAPSHORT"] == 1
    assert stats.invalid_quantity_rows["PW_CURRENT_ITEM"] == 0
    assert not db_path.exists()


def test_dry_run_rejects_same_required_field_error_as_build(tmp_path):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    content = source.read_text(encoding="latin-1")
    content = content.replace("'T-1','PROC-A'", "'T-1',null", 1)
    source.write_text(content, encoding="latin-1")
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    with pytest.raises(ValueError, match="PW_CURRENT_ITEM.PROCESS_ID"):
        validate_source(source, schema)


def test_failed_reimport_preserves_previous_snapshot(tmp_path):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"
    build_database(source, schema, db_path)
    original_hash = db_path.read_bytes()
    source.write_text(
        _insert("PW_PROCESS", "ID,PROCESS_ID", "1,'PROC-A'"),
        encoding="latin-1",
    )

    with pytest.raises(ValueError):
        build_database(source, schema, db_path)

    assert db_path.read_bytes() == original_hash
    assert validate_snapshot(db_path)["compatible"] is True


def test_reimport_atomically_replaces_rows(tmp_path):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    build_database(source, schema, db_path)
    build_database(source, schema, db_path)

    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM wms_current_balances"
        ).fetchone()[0] == 2


def test_build_rejects_missing_required_tables_without_replacing_db(tmp_path):
    source = tmp_path / "missing-current.sql"
    source.write_text(
        _insert(
            "PW_PROCESS",
            (
                "ID,CREATE_DATE,EDIT_DATE,PROCESS_ID,PROCESS_NAME,"
                "PROCESS_PHYSICAL_ID,STATUS,NOTE,IS_CHECK_MATERIAL"
            ),
            "1,null,null,'PROC-A','Process A','PHYS-A','1',null,'Y'",
        ),
        encoding="latin-1",
    )
    db_path = tmp_path / "mes_wms.sqlite"
    db_path.write_bytes(b"previous-snapshot")
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    with pytest.raises(ValueError, match="PW_CURRENT_ITEM"):
        build_database(source, schema, db_path)

    assert db_path.read_bytes() == b"previous-snapshot"


def test_optional_sources_record_not_observed_evidence(tmp_path):
    source = tmp_path / "current-only.sql"
    source.write_text(
        "\n".join(
            [
                "-- File created - Tuesday-July-28-2026",
                _insert(
                    "PW_CURRENT_ITEM",
                    "ID,CREATE_DATE,EDIT_DATE,ITEM_CODE,ITEM_LOT_ID,QTY,TIME_UPDATE,TIME_UPDATE_UNIX,TRANS_ID,PROCESS_ID",
                    f"1,null,null,'ITEM-A','-','12.5',{_timestamp('27-JUL-26 11.00.00.000000000 PM')},'1785193200','T-1','PROC-A'",
                ),
            ]
        ),
        encoding="latin-1",
    )
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    build_database(source, schema, db_path)

    with sqlite3.connect(db_path) as connection:
        evidence = {
            row[0]: row[1:]
            for row in connection.execute(
                """
                SELECT dataset, status, reason_code, source_state,
                       source_as_of_state
                FROM wms_dataset_evidence
                ORDER BY dataset
                """
            )
        }
        capabilities = dict(
            connection.execute(
                "SELECT capability, status FROM wms_capability_status"
            )
        )

    assert evidence["LEGACY_ARCHIVE"] == (
        "SUPPRESSED",
        "DATASET_NOT_OBSERVED_IN_EXPORT",
        "NOT_OBSERVED_IN_EXPORT",
        "UNAVAILABLE",
    )
    assert evidence["RAW_TRANSACTION_AUDIT"] == (
        "SUPPRESSED",
        "DATASET_NOT_OBSERVED_IN_EXPORT",
        "NOT_OBSERVED_IN_EXPORT",
        "UNAVAILABLE",
    )
    assert capabilities["LEGACY_ARCHIVE_EXACT_KEY_QUERY"] == "SUPPRESSED"
    assert capabilities["RAW_TRANSACTION_AUDIT_QUERY"] == "SUPPRESSED"
    assert validate_snapshot(db_path)["compatible"] is True


def test_json_report_is_aggregate_and_failure_is_machine_readable(tmp_path, monkeypatch):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"
    success_report = tmp_path / "success.json"
    dry_run_db = tmp_path / "must-not-exist.sqlite"
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_mes_wms.py",
            "--source",
            str(source),
            "--schema",
            str(schema),
            "--db",
            str(dry_run_db),
            "--dry-run",
            "--report-json",
            str(success_report),
        ],
    )

    assert importer.main() == 0
    success = json.loads(success_report.read_text(encoding="utf-8"))
    serialized = success_report.read_text(encoding="utf-8")
    assert success["ok"] is True
    assert success["schema_version"] == "4"
    assert success["gates"]["current_balance_duplicate_count"] == 0
    assert not dry_run_db.exists()
    assert str(source) not in serialized
    assert "ITEM-A" not in serialized
    assert "private-user" not in serialized
    assert "private-note" not in serialized

    duplicate = _insert(
        "PW_CURRENT_ITEM",
        "ID,CREATE_DATE,EDIT_DATE,ITEM_CODE,ITEM_LOT_ID,QTY,TIME_UPDATE,TIME_UPDATE_UNIX,TRANS_ID,PROCESS_ID",
        f"99,null,null,'ITEM-A','-','1',{_timestamp('27-JUL-26 11.45.00.000000000 PM')},'1785195900','T-99','PROC-A'",
    )
    source.write_text(
        source.read_text(encoding="latin-1") + "\n" + duplicate,
        encoding="latin-1",
    )
    failure_report = tmp_path / "failure.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_mes_wms.py",
            "--source",
            str(source),
            "--schema",
            str(schema),
            "--dry-run",
            "--report-json",
            str(failure_report),
        ],
    )

    assert importer.main() == 1
    failure = json.loads(failure_report.read_text(encoding="utf-8"))
    assert failure["ok"] is False
    assert failure["error"]["reason_code"] == "CURRENT_BALANCE_GRAIN_DUPLICATE"
    assert failure["gates"]["current_balance_duplicate_count"] == 1
    assert "ITEM-A" not in failure_report.read_text(encoding="utf-8")


def test_report_json_must_not_overwrite_snapshot_or_source(tmp_path, monkeypatch):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    monkeypatch.setattr(
        "sys.argv",
        [
            "import_mes_wms.py",
            "--source",
            str(source),
            "--schema",
            str(schema),
            "--db",
            str(db_path),
            "--report-json",
            str(db_path),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        importer.parse_args()
    assert exc_info.value.code == 2

    monkeypatch.setattr(
        "sys.argv",
        [
            "import_mes_wms.py",
            "--source",
            str(source),
            "--schema",
            str(schema),
            "--dry-run",
            "--report-json",
            str(source),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        importer.parse_args()
    assert exc_info.value.code == 2


def test_build_mode_report_json_failure_preserves_previous_snapshot(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    db_path = tmp_path / "mes_wms.sqlite"
    db_path.write_bytes(b"previous-snapshot")
    report_path = tmp_path / "report.json"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_mes_wms.py",
            "--source",
            str(source),
            "--schema",
            str(schema),
            "--db",
            str(db_path),
            "--report-json",
            str(report_path),
        ],
    )

    def fail_report(*args, **kwargs):
        raise OSError("report write failed")

    monkeypatch.setattr(importer, "_write_json_report", fail_report)

    assert importer.main() == 1
    assert db_path.read_bytes() == b"previous-snapshot"


def test_build_mode_report_json_uses_completed_plan_without_reparse(tmp_path, monkeypatch):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    db_path = tmp_path / "mes_wms.sqlite"
    report_path = tmp_path / "build-report.json"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"
    original_build_plan = importer._build_import_plan
    calls = []

    def counted_build_plan(*args, **kwargs):
        calls.append(args[0])
        return original_build_plan(*args, **kwargs)

    monkeypatch.setattr(importer, "_build_import_plan", counted_build_plan)
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_mes_wms.py",
            "--source",
            str(source),
            "--schema",
            str(schema),
            "--db",
            str(db_path),
            "--report-json",
            str(report_path),
        ],
    )

    assert importer.main() == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert calls == [source]
    assert report["ok"] is True
    assert report["datasets"][0]["inserted_row_count"] > 0
    assert any(
        item["capability"] == "CURRENT_LOT_LOOKUP"
        and item["status"] == "SUPPRESSED"
        for item in report["capabilities"]
    )
    assert validate_snapshot(db_path)["compatible"] is True


def test_strict_freshness_mismatch_fails_closed(tmp_path):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"
    build_database(source, schema, db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE wms_dataset_evidence
            SET source_as_of = '2026-01-01 00:00:00'
            WHERE dataset = 'CURRENT_BALANCE'
            """
        )

    report = validate_snapshot(db_path)
    assert report["compatible"] is False
    assert "CURRENT_BALANCE:as_of_mismatch" in report["semantic_errors"]


def test_parser_rejects_changed_allowlisted_column_contract(tmp_path):
    source = tmp_path / "bad.sql"
    source.write_text(
        _insert(
            "PW_PROCESS",
            "ID,PROCESS_ID",
            "1,'PROC-A'",
        ),
        encoding="latin-1",
    )

    with pytest.raises(ValueError, match="không đúng contract"):
        list(iter_allowlisted_rows(source))


def _write_current_with_optional_audit(
    path: Path,
    *,
    header: bool = False,
    definition: bool = False,
    detail: bool = False,
) -> None:
    lines = [
        "-- File created - Tuesday-July-28-2026",
        _insert(
            "PW_CURRENT_ITEM",
            "ID,CREATE_DATE,EDIT_DATE,ITEM_CODE,ITEM_LOT_ID,QTY,TIME_UPDATE,TIME_UPDATE_UNIX,TRANS_ID,PROCESS_ID",
            f"1,null,null,'ITEM-A','-','12.5',{_timestamp('27-JUL-26 11.00.00.000000000 PM')},'1785193200','T-1','PROC-A'",
        ),
    ]
    if header:
        lines.append(
            _insert(
                "PW_TRANSACTION",
                "ID,CREATE_DATE,EDIT_DATE,TRANS_ID,TRANS_CODE,TRANS_DATE,TRANS_DATE_UNIX,PROCESS_ID,ITEM_CODE,QTY,TRANS_STATUS,RELATE_ID,USER_ID,NOTE,VALUE_1,VALUE_2,VALUE_3,CREADIT_CODE,DEBIT_CODE,DELETED",
                f"2,null,null,'T-1','001',{_timestamp('27-JUL-26 11.00.00.000000000 PM')},'1785193200','PROC-A','ITEM-A','12.5','1',null,null,null,null,null,null,null,null,'N'",
            )
        )
    if definition:
        lines.append(
            _insert(
                "PW_TRANSACTION_DEFINE",
                "ID,CREATE_DATE,EDIT_DATE,TRANS_CODE,TRANS_NAME,NOTE",
                "3,null,null,'001','Import',null",
            )
        )
    if detail:
        lines.append(
            _insert(
                "PW_TRANS_DETAIL",
                "ID,CREATE_DATE,EDIT_DATE,TRANS_ID,ITEM_LOT_ID,QTY,PRODUCT_ID,LOT_ID,NOTE,ATTACH_PATH",
                "4,null,null,'T-1','LOT-MATERIAL','12.5','PRODUCT-X','LOT-X',null,null",
            )
        )
    path.write_text("\n".join(lines), encoding="latin-1")


@pytest.mark.parametrize(
    ("definition", "detail", "expected_tables"),
    [
        (True, False, "PW_TRANSACTION_DEFINE"),
        (False, True, "PW_TRANS_DETAIL"),
        (True, True, "PW_TRANSACTION_DEFINE,PW_TRANS_DETAIL"),
    ],
)
def test_partial_audit_sources_without_header_are_valid_but_suppressed(
    tmp_path,
    definition,
    detail,
    expected_tables,
):
    source = tmp_path / "partial-audit.sql"
    _write_current_with_optional_audit(
        source,
        definition=definition,
        detail=detail,
    )
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    validate_source(source, schema)
    build_database(source, schema, db_path)

    assert validate_snapshot(db_path)["compatible"] is True
    with sqlite3.connect(db_path) as connection:
        evidence = connection.execute(
            """
            SELECT status, reason_code, source_tables, source_state,
                   candidate_row_count, inserted_row_count,
                   source_as_of_state
            FROM wms_dataset_evidence
            WHERE dataset = 'RAW_TRANSACTION_AUDIT'
            """
        ).fetchone()
        capability = connection.execute(
            """
            SELECT status, reason_code
            FROM wms_capability_status
            WHERE capability = 'RAW_TRANSACTION_AUDIT_QUERY'
            """
        ).fetchone()

    expected_count = int(definition) + int(detail)
    assert evidence == (
        "SUPPRESSED",
        "RAW_TRANSACTION_HEADER_NOT_OBSERVED",
        expected_tables,
        "PARTIAL_SOURCE_OBSERVED",
        expected_count,
        expected_count,
        "UNAVAILABLE",
    )
    assert capability == ("SUPPRESSED", "RAW_TRANSACTION_HEADER_NOT_OBSERVED")


def test_audit_header_without_other_sources_is_partial_and_compatible(tmp_path):
    source = tmp_path / "header-only.sql"
    _write_current_with_optional_audit(source, header=True)
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    build_database(source, schema, db_path)

    assert validate_snapshot(db_path)["compatible"] is True
    with sqlite3.connect(db_path) as connection:
        capability = connection.execute(
            """
            SELECT status, reason_code
            FROM wms_capability_status
            WHERE capability = 'RAW_TRANSACTION_AUDIT_QUERY'
            """
        ).fetchone()
    assert capability == ("PARTIAL", "TRANSACTION_SEMANTICS_NOT_IN_SCOPE")


def test_legacy_invalid_quantity_marks_dataset_partial(tmp_path):
    source = tmp_path / "legacy-invalid.sql"
    _write_dump(source)
    source.write_text(
        source.read_text(encoding="latin-1").replace(
            "'ITEM-A','9.5','PROC-A'",
            "'ITEM-A','bad','PROC-A'",
        ),
        encoding="latin-1",
    )
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    build_database(source, schema, db_path)

    assert validate_snapshot(db_path)["compatible"] is True
    with sqlite3.connect(db_path) as connection:
        evidence = connection.execute(
            """
            SELECT status, reason_code, invalid_quantity_row_count
            FROM wms_dataset_evidence
            WHERE dataset = 'LEGACY_ARCHIVE'
            """
        ).fetchone()
        capability = connection.execute(
            """
            SELECT status, reason_code
            FROM wms_capability_status
            WHERE capability = 'LEGACY_ARCHIVE_EXACT_KEY_QUERY'
            """
        ).fetchone()
    assert evidence == ("PARTIAL", "QUANTITY_EVIDENCE_INCOMPLETE", 1)
    assert capability == ("PARTIAL", "QUANTITY_EVIDENCE_INCOMPLETE")


@pytest.mark.parametrize(
    ("sql", "expected_error"),
    [
        (
            "UPDATE wms_dataset_evidence SET inserted_row_count = 999 WHERE dataset = 'CURRENT_BALANCE'",
            "CURRENT_BALANCE:inserted_count_mismatch",
        ),
        (
            "UPDATE wms_dataset_evidence SET invalid_quantity_row_count = 999 WHERE dataset = 'CURRENT_BALANCE'",
            "CURRENT_BALANCE:invalid_quantity_count_mismatch",
        ),
        (
            "UPDATE wms_dataset_evidence SET source_tables = '' WHERE dataset = 'CURRENT_BALANCE'",
            "CURRENT_BALANCE:missing_source_tables",
        ),
        (
            "UPDATE wms_dataset_evidence SET evidence_basis = '' WHERE dataset = 'CURRENT_BALANCE'",
            "CURRENT_BALANCE:missing_evidence_basis",
        ),
        (
            "UPDATE wms_dataset_evidence SET status = 'AVAILABLE', reason_code = '' WHERE dataset = 'CURRENT_BALANCE'",
            "CURRENT_BALANCE:current_capability_not_partial",
        ),
    ],
)
def test_validator_rejects_tampered_evidence(tmp_path, sql, expected_error):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"
    build_database(source, schema, db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(sql)

    report = validate_snapshot(db_path)
    assert report["compatible"] is False
    assert expected_error in report["semantic_errors"]


def test_audit_header_without_date_is_partial_not_snapshot_fatal(tmp_path):
    source = tmp_path / "header-no-date.sql"
    _write_current_with_optional_audit(source, header=True)
    content = source.read_text(encoding="latin-1")
    header = next(
        line for line in content.splitlines() if "PW_TRANSACTION (" in line
    )
    source.write_text(
        content.replace(
            header,
            header.replace(
                _timestamp("27-JUL-26 11.00.00.000000000 PM"),
                "null",
            ),
        ),
        encoding="latin-1",
    )
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"

    build_database(source, schema, db_path)

    report = validate_snapshot(db_path)
    assert report["compatible"] is True
    with sqlite3.connect(db_path) as connection:
        evidence = connection.execute(
            """
            SELECT status, reason_code, source_as_of, source_as_of_state
            FROM wms_dataset_evidence
            WHERE dataset = 'RAW_TRANSACTION_AUDIT'
            """
        ).fetchone()
    assert evidence == (
        "PARTIAL",
        "QUANTITY_EVIDENCE_INCOMPLETE",
        "",
        "UNAVAILABLE",
    )


def test_validator_requires_current_grain_unique_constraint(tmp_path):
    source = tmp_path / "wms.sql"
    _write_dump(source)
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"
    build_database(source, schema, db_path)

    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys = OFF;
            DROP VIEW v_wms_current_balance_by_process_item;
            DROP VIEW v_wms_current_quality;
            ALTER TABLE wms_current_balances RENAME TO old_wms_current_balances;
            CREATE TABLE wms_current_balances AS
                SELECT * FROM old_wms_current_balances;
            DROP TABLE old_wms_current_balances;
            """
        )
        schema_text = schema.read_text(encoding="utf-8")
        start = schema_text.index("CREATE VIEW v_wms_current_balance_by_process_item")
        end = schema_text.index("CREATE VIEW v_wms_legacy_archive_exact_key")
        quality = schema_text[schema_text.index("CREATE VIEW v_wms_current_quality") :]
        connection.executescript(schema_text[start:end] + quality)

    report = validate_snapshot(db_path)
    assert report["compatible"] is False
    assert "CURRENT_BALANCE:missing_grain_unique_constraint" in report["semantic_errors"]
