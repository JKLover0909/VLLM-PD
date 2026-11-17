import sqlite3
from datetime import date, datetime, time, timezone
from pathlib import Path

import pytest

from scripts.create_msm_mock_database import build_database


ROOT = Path(__file__).parents[1]
SCHEMA = ROOT / "database" / "schema" / "msm_msa.sql"


def _epoch(value: date) -> int:
    at_noon = datetime.combine(value, time(12), tzinfo=timezone.utc)
    return int(at_noon.timestamp())


TABLES = (
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


def _build(path: Path, *, seed: int = 17, machines: int = 240):
    return build_database(
        SCHEMA,
        path,
        department_count=6,
        machine_count=machines,
        calibrations_per_machine=3,
        msa_ratio=0.5,
        seed=seed,
        as_of=date(2026, 10, 17),
    )


def _rows(path: Path, table: str):
    with sqlite3.connect(path) as connection:
        query = f'SELECT * FROM "{table}" ORDER BY "ID"'
        return connection.execute(query).fetchall()


def test_builds_all_nine_tables_with_consistent_synthetic_relations(tmp_path):
    db_path = tmp_path / "msm_mock.sqlite"
    counts = _build(db_path)

    assert set(counts) == set(TABLES)
    assert counts["MSM_MACHINE_DEPARTMENT"] == 6
    assert counts["MSM_MACHINE"] == 240
    assert 80 <= counts["MSA_MACHINE"] <= 160
    assert counts["MSA_JUDGE"] > counts["MSA_MACHINE"]
    assert counts["MSM_PLAN_FILE"] == 36
    assert counts["MSM_TRACE_DIAGRAM"] == 30

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        assert integrity == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        calibration_count = connection.execute(
            "SELECT COUNT(*) FROM MSM_CALIB_JUDGE"
        ).fetchone()[0]
        machine_count = connection.execute(
            "SELECT COUNT(*) FROM MSM_MACHINE"
        ).fetchone()[0]
        expected_history = machine_count * 3
        machine_date_mismatches = connection.execute(
            "SELECT COUNT(*) FROM MSM_MACHINE AS machine "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM MSM_CALIB_JUDGE AS history "
            "WHERE history.MACHINE_ID = machine.MACHINE_ID "
            "AND history.IS_DRAFT = 'N' "
            "AND history.CALIBRATION_DATE = machine.LAST_CALIBRATION)"
        ).fetchone()[0]
        assert machine_date_mismatches == 0
        next_date_mismatches = connection.execute(
            "SELECT COUNT(*) FROM MSM_MACHINE AS machine "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM MSM_CALIB_JUDGE AS history "
            "WHERE history.MACHINE_ID = machine.MACHINE_ID "
            "AND history.IS_DRAFT = 'N' "
            "AND history.CALIBRATION_DATE = machine.LAST_CALIBRATION "
            "AND history.NEXT_CALIBRATION_HISTORY = machine.NEXT_CALIBRATION)"
        ).fetchone()[0]
        assert next_date_mismatches == 0
        expected_drafts = connection.execute(
            "SELECT COUNT(*) FROM MSM_MACHINE WHERE ID % 11 = 0 "
            "AND STATUS != 'Unused' "
            "AND (CASE WHEN NEXT_DUE_DATE > 0 THEN NEXT_DUE_DATE "
            "ELSE NEXT_CALIBRATION END) > ?",
            (_epoch(date(2026, 10, 17)),),
        ).fetchone()[0]
        assert counts["MSM_CALIB_JUDGE"] == calibration_count
        assert calibration_count == expected_history + expected_drafts
        assert calibration_count >= expected_history
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        } == {*TABLES, "schema_metadata"}
        metadata = dict(connection.execute("SELECT key, value FROM schema_metadata"))
        assert metadata["dataset_type"] == "synthetic_mock"
        assert metadata["seed"] == "17"
        assert metadata["as_of"] == "2026-10-17"
        overdue_count = connection.execute(
            "SELECT COUNT(*) FROM MSM_MACHINE WHERE CALIBRATION_STATUS='Overdue'"
        ).fetchone()[0]
        future_calibrations = connection.execute(
            "SELECT COUNT(*) FROM MSM_CALIB_JUDGE "
            "WHERE CALIBRATION_DATE > ? AND IS_DRAFT='N'",
            (_epoch(date(2026, 10, 17)),),
        ).fetchone()[0]
        assert future_calibrations == 0
        assert overdue_count > 0
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM MSM_CALIB_JUDGE WHERE EFFECTIVE='N'"
            ).fetchone()[0]
            > 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM MSM_CALIB_JUDGE WHERE IS_DRAFT='Y'"
            ).fetchone()[0]
            > 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM MSA_JUDGE WHERE IS_JUDGE_OK='NG'"
            ).fetchone()[0]
            > 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM MSA_JUDGE WHERE RESULT_DATE_UNIX IS NULL"
            ).fetchone()[0]
            > 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM MSM_MACHINE WHERE NOTE NOT LIKE '[MOCK]%'"
            ).fetchone()[0]
            == 0
        )


def test_same_seed_and_options_create_same_table_rows(tmp_path):
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    first_counts = _build(first, seed=1234, machines=40)
    second_counts = _build(second, seed=1234, machines=40)

    assert first_counts == second_counts
    for table in TABLES:
        assert _rows(first, table) == _rows(second, table)


def test_existing_database_is_not_replaced_without_explicit_opt_in(tmp_path):
    db_path = tmp_path / "existing.sqlite"
    _build(db_path, machines=10)
    before = db_path.read_bytes()

    with pytest.raises(FileExistsError, match="--replace"):
        _build(db_path, machines=20)

    assert db_path.read_bytes() == before


def test_failed_rebuild_preserves_existing_database(tmp_path):
    db_path = tmp_path / "existing.sqlite"
    _build(db_path, machines=10)
    before = db_path.read_bytes()

    with pytest.raises(ValueError):
        build_database(
            SCHEMA,
            db_path,
            department_count=0,
            machine_count=20,
            calibrations_per_machine=2,
            msa_ratio=0.5,
            seed=99,
            as_of=date(2026, 10, 17),
            replace=True,
        )

    assert db_path.read_bytes() == before


def test_generator_supports_zero_calibration_history(tmp_path):
    db_path = tmp_path / "without_history.sqlite"
    counts = build_database(
        SCHEMA,
        db_path,
        department_count=2,
        machine_count=12,
        calibrations_per_machine=0,
        msa_ratio=0,
        seed=9,
        as_of=date(2026, 10, 17),
    )

    assert counts["MSM_CALIB_JUDGE"] == 0
    assert counts["MSA_MACHINE"] == 0
    assert counts["MSA_JUDGE"] == 0
