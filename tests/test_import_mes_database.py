import sqlite3
from pathlib import Path

from scripts.import_mes_database import build_database


LOT_COLUMNS = (
    "ID,CREATE_DATE,EDIT_DATE,PRODUCT_ID,LOT_ID,PT_ID,PT_VERSION_ID,ROUTE_ID,"
    "LOT_TYPE,STATUS,IS_RELEASE,SALE_ORDER_ID,BOARD_LOT,SHEET_LOT,PREV_STATUS,"
    "DATE_CODE,PRODUCE_DATE,PRODUCE_DATE_PROCESS_ID,PRODUCE_DATE_PROCESS_ORDER,"
    "IS_RELEASE_SPLIT_LOT,PCS_LOT,RELEASE_DATE_UNIX,RELEASE_DATE,CREATE_TIME_UNIX,"
    "PRODUCTION_TYPE,USER_ID,PREV_RELEASE,PRODUCTION_PERIOD_TYPE,USER_ID_UPDATE,"
    "TIME_UPDATE,TIME_UPDATE_UNIX"
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
MAIN_COLUMNS = (
    "ID,EDIT_DATE,CREATE_DATE,LOT_ID,ROUTE_ID,PROCESS_ID,PROCESS_ORDER,T1,T2,T3,T4,"
    "USER_ID,NOTE,STAFF_ID,STAFF_NAME,P_OK,P_NG_DEFECT,P_NG_SCRAP,S_OK,"
    "S_NG_DEFECT,S_NG_SCRAP,B_OK,B_NG_DEFECT,B_NG_SCRAP,T1_DATE,T2_DATE,T3_DATE,"
    "T4_DATE,OUTPUT_MAX_B,OUTPUT_MAX_S,OUTPUT_MAX_P,IS_MOVE_STEP,"
    "PROCESS_PHYSICAL_SUB,MOVING_STATUS"
)


def _write_raw_files(raw_dir: Path) -> None:
    raw_dir.mkdir()
    (raw_dir / "M_LOT_202601010000.sql").write_text(
        f"INSERT INTO MES_DATA.M_LOT ({LOT_COLUMNS}) VALUES\n"
        "(1,TIMESTAMP'2026-01-01 08:00:00',NULL,'PRODUCT-1','LOT-1','PT-1','1',"
        "'ROUTE-1','0','1','Y','-',10,100,NULL,NULL,'2026-01-01','PROC-1',1,'Y',"
        "200,2,TIMESTAMP'2026-01-01 00:00:00',1,'1','user',NULL,NULL,NULL,NULL,NULL);",
        encoding="utf-8",
    )
    (raw_dir / "D_ERROR_202601010000.sql").write_text(
        f"INSERT INTO MES_DATA.D_ERROR ({ERROR_COLUMNS}) VALUES\n"
        "(10,NULL,TIMESTAMP'2026-01-01 09:00:00','LOT-1','ROUTE-1','PROC-1',1,'1',"
        "'E-1',5,'user',NULL,NULL,NULL,NULL,'null',3,TIMESTAMP'2026-01-01 09:00:00',NULL),\n"
        "(11,NULL,TIMESTAMP'2026-01-01 09:30:00','LOT-1','ROUTE-1','PROC-2',2,'1',"
        "'0002',7,'user',NULL,NULL,NULL,NULL,NULL,3,TIMESTAMP'2026-01-01 09:30:00',NULL),\n"
        "(NULL,NULL,TIMESTAMP'2026-01-01 10:00:00','ORPHAN','ROUTE-1','PROC-X',2,'1',"
        "'E-X',2,'user',NULL,NULL,NULL,NULL,NULL,4,TIMESTAMP'2026-01-01 10:00:00',NULL);",
        encoding="utf-8",
    )
    (raw_dir / "P_ERROR_202601010000.sql").write_text(
        f"INSERT INTO MES_DATA.P_ERROR ({CATALOG_COLUMNS}) VALUES\n"
        "(20,TIMESTAMP'2026-01-01 07:00:00',NULL,'E-1','Short','1','1',NULL,'N',"
        "'PROC-1','Ngắn mạch',NULL,'Short circuit',NULL,NULL,'user'),\n"
        "(21,TIMESTAMP'2026-01-01 07:00:00',NULL,'0002','Scratch','1','1',NULL,'N',"
        "'-','Xước',NULL,'Scratch',NULL,NULL,'user');",
        encoding="utf-8",
    )
    (raw_dir / "D_MAIN_202601010000.sql").write_text(
        f"INSERT INTO MES_DATA.D_MAIN ({MAIN_COLUMNS}) VALUES\n"
        "(30,NULL,TIMESTAMP'2026-01-01 08:30:00','LOT-1','ROUTE-1','PROC-1',1,"
        "1,2,3,4,'private-user','private-note','EMP-1','Private Name',100,2,1,"
        "50,1,0,10,-1,0,TIMESTAMP'2026-01-01 08:30:00',"
        "TIMESTAMP'2026-01-01 08:40:00',TIMESTAMP'2026-01-01 08:50:00',"
        "TIMESTAMP'2026-01-01 09:00:00',10,50,100,'N',NULL,'0'),\n"
        "(31,NULL,TIMESTAMP'2026-01-01 09:30:00','ORPHAN','ROUTE-1','PROC-X',2,"
        "5,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,-1,NULL,NULL,NULL,NULL,NULL,NULL,NULL,"
        "TIMESTAMP'2026-01-01 09:30:00',NULL,NULL,NULL,NULL,NULL,NULL,'Y',NULL,'0');",
        encoding="utf-8",
    )


def test_build_mes_database_preserves_orphans_and_builds_summary(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_files(raw_dir)
    db_path = tmp_path / "mes.sqlite"
    schema_path = Path(__file__).parents[1] / "database" / "schema" / "mes.sql"

    counts = build_database(raw_dir, schema_path, db_path)

    assert counts == {
        "lots": 1,
        "error_events": 3,
        "error_catalog": 2,
        "process_steps": 2,
    }
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
        null_text_value = connection.execute(
            "SELECT process_order_create FROM error_events WHERE error_id='E-1'"
        ).fetchone()[0]
        metadata = dict(connection.execute("SELECT key, value FROM schema_metadata"))
        process_step = connection.execute(
            """
            SELECT lot_id, product_id, process_id, process_order,
                   b_ng_defect, lot_mapped
            FROM v_lot_process_steps
            WHERE lot_id = 'LOT-1'
            """
        ).fetchone()
        orphan_process_steps = connection.execute(
            "SELECT COUNT(*) FROM process_steps WHERE lot_pk IS NULL"
        ).fetchone()[0]
        private_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(v_lot_process_steps)"
            )
        }

    assert summary == ("LOT-1", "PRODUCT-1", 12)
    assert exact_detail == ("Ngắn mạch", 1)
    assert fallback_detail == ("Xước", 1)
    assert orphan_count == 1
    assert null_text_value is None
    assert metadata["orphan_error_event_count"] == "1"
    assert metadata["unmapped_error_name_count"] == "1"
    assert metadata["schema_version"] == "2"
    assert metadata["process_step_count"] == "2"
    assert metadata["orphan_process_step_count"] == "1"
    assert process_step == ("LOT-1", "PRODUCT-1", "PROC-1", 1, None, 1)
    assert orphan_process_steps == 1
    assert not {"user_id", "note", "staff_id", "staff_name"} & private_columns


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
        assert connection.execute("SELECT COUNT(*) FROM process_steps").fetchone()[0] == 2


def test_build_mes_database_without_optional_d_main(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_files(raw_dir)
    (raw_dir / "D_MAIN_202601010000.sql").unlink()
    db_path = tmp_path / "mes.sqlite"
    schema_path = Path(__file__).parents[1] / "database" / "schema" / "mes.sql"

    counts = build_database(raw_dir, schema_path, db_path)

    assert counts["process_steps"] == 0
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM process_steps").fetchone()[0] == 0
        metadata = dict(connection.execute("SELECT key, value FROM schema_metadata"))
        assert metadata["process_step_count"] == "0"
        assert metadata["orphan_process_step_count"] == "0"
        assert connection.execute(
            "SELECT COUNT(*) FROM import_batches WHERE source_name='D_MAIN'"
        ).fetchone()[0] == 0
