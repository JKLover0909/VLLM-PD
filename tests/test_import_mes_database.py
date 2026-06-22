import sqlite3
from pathlib import Path

from scripts.import_mes_database import build_database


LOT_COLUMNS = (
    "ID,CREATE_DATE,EDIT_DATE,PRODUCT_ID,LOT_ID,PT_ID,PT_VERSION_ID,ROUTE_ID,"
    "LOT_TYPE,STATUS,IS_RELEASE,SALE_ORDER_ID,BOARD_LOT,SHEET_LOT,PREV_STATUS,"
    "DATE_CODE,PRODUCE_DATE,PRODUCE_DATE_PROCESS_ID,PRODUCE_DATE_PROCESS_ORDER,"
    "IS_RELEASE_SPLIT_LOT,PCS_LOT,CREATE_TIME_UNIX,RELEASE_DATE_UNIX,RELEASE_DATE,"
    "PRODUCTION_TYPE,USER_ID,PREV_RELEASE,PRODUCTION_PERIOD_TYPE,USER_ID_UPDATE,"
    "TIME_UPDATE_UNIX,TIME_UPDATE"
)
ERROR_COLUMNS = (
    "ID,EDIT_DATE,CREATE_DATE,LOT_ID,ROUTE_ID,PROCESS_ID,PROCESS_ORDER,ERROR_TYPE,"
    "ERROR_ID,QTY,USER_ID,NOTE,ERROR_PROCESS_TYPE,LOT_ID_SPLIT,PROCESS_ID_CREATE,"
    "PROCESS_ORDER_CREATE,ERROR_TIME_UNIX,ERROR_TIME,ERROR_JUDGEMENT"
)
CATALOG_COLUMNS = (
    "ID,CREATE_DATE,EDIT_DATE,ERROR_ID,ERROR_NAME,ERROR_TYPE,STATUS,NOTE,DELETED,"
    "PROCESS_ID,ERROR_NAME_VI,ERROR_NAME_JA,ERROR_NAME_EN,ERROR_NAME_CH,"
    "PRIORITY_ERROR,USER_ID"
)


def _write_raw_files(raw_dir: Path) -> None:
    raw_dir.mkdir()
    (raw_dir / "M_LOT_202601010000.sql").write_text(
        f"INSERT INTO MES_DATA_MKHC.M_LOT ({LOT_COLUMNS}) VALUES\n"
        "(1,TIMESTAMP'2026-01-01 08:00:00',NULL,'PRODUCT-1','LOT-1','PT-1','1',"
        "'ROUTE-1','0','1','Y','-',10,100,NULL,NULL,'2026-01-01','PROC-1',1,'Y',"
        "200,1,2,TIMESTAMP'2026-01-01 00:00:00','1','user',NULL,NULL,NULL,NULL,NULL);",
        encoding="utf-8",
    )
    (raw_dir / "D_ERROR_202601010000.sql").write_text(
        f"INSERT INTO MES_DATA_MKHC.D_ERROR ({ERROR_COLUMNS}) VALUES\n"
        "(10,NULL,TIMESTAMP'2026-01-01 09:00:00','LOT-1','ROUTE-1','PROC-1',1,'1',"
        "'E-1',5,'user',NULL,NULL,NULL,NULL,NULL,3,TIMESTAMP'2026-01-01 09:00:00',NULL),\n"
        "(11,NULL,TIMESTAMP'2026-01-01 09:30:00','LOT-1','ROUTE-1','PROC-2',2,'1',"
        "'0002',7,'user',NULL,NULL,NULL,NULL,NULL,3,TIMESTAMP'2026-01-01 09:30:00',NULL),\n"
        "(NULL,NULL,TIMESTAMP'2026-01-01 10:00:00','ORPHAN','ROUTE-1','PROC-X',2,'1',"
        "'E-X',2,'user',NULL,NULL,NULL,NULL,NULL,4,TIMESTAMP'2026-01-01 10:00:00',NULL);",
        encoding="utf-8",
    )
    (raw_dir / "P_ERROR_202601010000.sql").write_text(
        f"INSERT INTO MES_DATA_MKHC.P_ERROR ({CATALOG_COLUMNS}) VALUES\n"
        "(20,TIMESTAMP'2026-01-01 07:00:00',NULL,'E-1','Short','1','1',NULL,'N',"
        "'PROC-1','Ngắn mạch',NULL,'Short circuit',NULL,NULL,'user'),\n"
        "(21,TIMESTAMP'2026-01-01 07:00:00',NULL,'0002','Scratch','1','1',NULL,'N',"
        "'-','Xước',NULL,'Scratch',NULL,NULL,'user');",
        encoding="utf-8",
    )


def test_build_mes_database_preserves_orphans_and_builds_summary(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_files(raw_dir)
    db_path = tmp_path / "mes.sqlite"
    schema_path = Path(__file__).parents[1] / "database" / "schema" / "mes.sql"

    counts = build_database(raw_dir, schema_path, db_path)

    assert counts == {"lots": 1, "error_events": 3, "error_catalog": 2}
    with sqlite3.connect(db_path) as connection:
        summary = connection.execute(
            "SELECT lot_id, product_id, total_error_qty FROM v_lot_error_summary"
        ).fetchone()
        exact_detail = connection.execute(
            "SELECT error_name, error_name_mapped FROM v_error_details WHERE lot_id='LOT-1'"
            " AND error_id='E-1'"
        ).fetchone()
        fallback_detail = connection.execute(
            "SELECT error_name, error_name_mapped FROM v_error_details WHERE lot_id='LOT-1'"
            " AND error_id='0002'"
        ).fetchone()
        orphan_count = connection.execute(
            "SELECT COUNT(*) FROM error_events WHERE lot_pk IS NULL"
        ).fetchone()[0]
        metadata = dict(connection.execute("SELECT key, value FROM schema_metadata"))

    assert summary == ("LOT-1", "PRODUCT-1", 12)
    assert exact_detail == ("Ngắn mạch", 1)
    assert fallback_detail == ("Xước", 1)
    assert orphan_count == 1
    assert metadata["orphan_error_event_count"] == "1"
    assert metadata["unmapped_error_name_count"] == "1"


def test_reimport_replaces_database_instead_of_duplicating_rows(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_files(raw_dir)
    db_path = tmp_path / "mes.sqlite"
    schema_path = Path(__file__).parents[1] / "database" / "schema" / "mes.sql"

    build_database(raw_dir, schema_path, db_path)
    build_database(raw_dir, schema_path, db_path)

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM lots").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM error_events").fetchone()[0] == 3
