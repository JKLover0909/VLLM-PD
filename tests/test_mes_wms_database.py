import sqlite3
from pathlib import Path

import pytest

from src.integrations.mes_wms_contract import (
    SCHEMA_VERSION,
    capability_statuses_for_datasets,
)
from src.integrations.mes_wms_database import MesWmsDatabase


@pytest.fixture
def wms_database(tmp_path: Path) -> MesWmsDatabase:
    db_path = tmp_path / "mes_wms.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(schema.read_text(encoding="utf-8"))
        connection.executemany(
            """
            INSERT INTO wms_processes (
                process_pk, source_id, process_id, process_name, status
            ) VALUES (?, ?, ?, ?, '1')
            """,
            [
                (1, 1, "PROC-A", "Công đoạn A"),
                (2, 2, "PROC-B", "Công đoạn B"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO wms_current_balances (
                balance_pk, source_id, item_code,
                quantity_decimal, quantity_valid, time_update, process_id,
                process_pk
            ) VALUES (?, ?, ?, ?, 1, ?, ?, ?)
            """,
            [
                (1, 1, "ITEM-A", "12.5", "2026-07-27 10:00:00", "PROC-A", 1),
                (2, 2, "ITEM-A", "7", "2026-07-27 12:00:00", "PROC-B", 2),
                (3, 3, "ITEM-B", "100", "2026-07-27 13:00:00", "PROC-X", None),
            ],
        )
        connection.executemany(
            """
            INSERT INTO wms_legacy_archive_records (
                source_id, archive_id, archive_date, item_code,
                item_lot_id, process_id, process_pk,
                quantity_decimal, quantity_valid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            [
                (10, "SNAP-OLD", "2026-07-20 10:00:00", "ITEM-A", "LOT-MATERIAL", "PROC-A", 1, "8.5"),
                (11, "SNAP-NEW", "2026-07-21 10:00:00", "ITEM-A", "LOT-MATERIAL", "PROC-A", 1, "9.5"),
            ],
        )
        connection.execute(
            """
            INSERT INTO wms_raw_transaction_definitions (trans_code, trans_name)
            VALUES ('001', 'Source label')
            """
        )
        connection.execute(
            """
            INSERT INTO wms_raw_transaction_headers (
                source_id, trans_id, trans_code, trans_date, process_id,
                item_code, quantity_decimal, quantity_valid
            ) VALUES (20, 'TRANS-1', '001', '2026-07-22 10:00:00',
                      'PROC-A', 'ITEM-A', '10', 1)
            """
        )
        connection.execute(
            """
            INSERT INTO wms_raw_transaction_details (
                source_id, trans_id, item_lot_id, quantity_decimal,
                quantity_valid
            ) VALUES (21, 'TRANS-1', 'LOT-MATERIAL', '10', 1)
            """
        )
        connection.executemany(
            "INSERT INTO wms_dataset_evidence (dataset, status, reason_code, source_tables, source_state, candidate_row_count, inserted_row_count, invalid_quantity_row_count, source_as_of, source_as_of_state, source_as_of_basis, source_timezone, semantic_epoch, evidence_basis) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("CURRENT_BALANCE", "PARTIAL", "UOM_MASTER_UNAVAILABLE", "PW_CURRENT_ITEM", "PRESENT_NONEMPTY", 3, 3, 0, "2026-07-27 13:00:00", "DERIVED_UNVERIFIED", "MAX(PW_CURRENT_ITEM.TIME_UPDATE)", "unverified", "CURRENT_POST_2026_01_15", "synthetic fixture"),
                ("LEGACY_ARCHIVE", "AVAILABLE", "", "PW_SNAPSHORT", "PRESENT_NONEMPTY", 2, 2, 0, "2026-07-21 10:00:00", "DERIVED_UNVERIFIED", "MAX(PW_SNAPSHORT.SNAPSHORT_DATE)", "unverified", "LEGACY_PRE_2026_01_15", "synthetic fixture"),
                ("RAW_TRANSACTION_AUDIT", "AVAILABLE", "", "PW_TRANSACTION,PW_TRANSACTION_DEFINE,PW_TRANS_DETAIL", "PRESENT_NONEMPTY", 3, 3, 0, "2026-07-22 10:00:00", "DERIVED_UNVERIFIED", "MAX(PW_TRANSACTION.TRANS_DATE)", "unverified", "RAW_SOURCE_AUDIT", "synthetic fixture"),
            ],
        )
        connection.executemany(
            "INSERT INTO schema_metadata (key, value) VALUES (?, ?)",
            [
                ("schema_version", SCHEMA_VERSION),
                ("data_contract_version", "wms-current-balance-v1"),
                ("semantic_contract_version", "wms-phase2c-v1"),
                ("semantic_epoch", "CURRENT_POST_2026_01_15"),
                ("source_schema", "MES_WMS_MKHC"),
                ("plant_id", "MKHC"),
                ("source_export_date", "2026-07-28"),
                ("source_timezone", "unverified"),
                ("imported_at", "2026-07-28T12:00:00+00:00"),
                ("source_as_of", "2026-07-27 13:00:00"),
                ("source_as_of_state", "DERIVED_UNVERIFIED"),
                ("source_as_of_basis", "MAX(PW_CURRENT_ITEM.TIME_UPDATE)"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO wms_capability_status (
                capability, status, reason_code, evidence_basis,
                contract_version
            ) VALUES (?, ?, ?, 'synthetic fixture', ?)
            """,
            [
                (capability, status, reason_code, SCHEMA_VERSION)
                for capability, (status, reason_code)
                in capability_statuses_for_datasets(
                    {
                        "CURRENT_BALANCE": ("PARTIAL", "UOM_MASTER_UNAVAILABLE"),
                        "LEGACY_ARCHIVE": ("AVAILABLE", ""),
                        "RAW_TRANSACTION_AUDIT": ("AVAILABLE", ""),
                    }
                ).items()
            ],
        )
    return MesWmsDatabase(db_path)


def test_item_at_process_aggregates_same_item_only(wms_database):
    result = wms_database.query_question(
        "Mã vật tư ITEM-A tồn kho tại công đoạn PROC-A bao nhiêu?"
    )

    assert result is not None
    assert result.intent == "wms_item_at_process"
    assert result.status == "PARTIAL"
    assert result.rows[0]["quantity_decimal"] == "12.5"
    assert "12,5" in result.fallback_answer
    assert "2026-07-27 13:00:00" in result.fallback_answer
    assert "timezone chưa xác nhận" in result.fallback_answer
    assert "UOM_MASTER_UNAVAILABLE" in result.reason_codes


def test_item_breakdown_never_sums_across_processes(wms_database):
    result = wms_database.query_question("Mã vật tư ITEM-A tồn kho ở đâu?")

    assert result is not None
    assert result.intent == "wms_item_by_process"
    assert len(result.rows) == 2
    assert {row["process_id"] for row in result.rows} == {"PROC-A", "PROC-B"}
    assert "19,5" not in result.fallback_answer


def test_process_inventory_does_not_sum_different_items(wms_database):
    result = wms_database.query_question("Tồn kho công đoạn PROC-X")

    assert result is not None
    assert result.intent == "wms_process_inventory"
    assert result.rows[0]["process_mapped"] == 0
    assert "chưa ánh xạ tên" in result.fallback_answer
    assert "100" in result.fallback_answer


def test_current_listings_expose_truncation_metadata(wms_database):
    with sqlite3.connect(wms_database.db_path) as connection:
        connection.executemany(
            """
            INSERT INTO wms_current_balances (
                source_id, item_code, quantity_decimal, quantity_valid,
                time_update, process_id, process_pk
            ) VALUES (?, ?, '1', 1, '2026-07-27 13:00:00', 'PROC-A', 1)
            """,
            [
                (100 + index, f"ITEM-{index:02d}")
                for index in range(21)
            ],
        )
        connection.execute(
            """
            UPDATE wms_dataset_evidence
            SET candidate_row_count = 24,
                inserted_row_count = 24,
                source_as_of = '2026-07-27 13:00:00'
            WHERE dataset = 'CURRENT_BALANCE'
            """
        )

    result = wms_database.query_question("Tồn kho công đoạn PROC-A")

    assert result is not None
    assert len(result.rows) == 20
    assert result.pagination == {
        "page": 1,
        "page_size": 20,
        "total_count": 22,
        "has_more": True,
    }
    assert "20/22" in result.fallback_answer


@pytest.mark.parametrize(
    "question",
    [
        "Liệt kê tất cả công đoạn cần quản lý tồn kho trong WMS",
        "Liệt kê tất cả mã công đoạn WMS",
    ],
)
def test_process_catalog_matches_all_processes_in_current_balance(
    wms_database, question
):
    result = wms_database.query_question(question)

    assert result is not None
    assert result.intent == "wms_process_catalog"
    assert result.domain == "CURRENT_BALANCE"
    assert {row["process_id"] for row in result.rows} == {
        "PROC-A",
        "PROC-B",
        "PROC-X",
    }
    assert "current balance snapshot" in result.fallback_answer
    assert "IS_CHECK_MATERIAL" not in result.fallback_answer
    assert "bắt buộc kiểm tra vật tư" in result.fallback_answer


def test_process_inventory_is_not_captured_as_process_catalog(wms_database):
    result = wms_database.query_question("Tồn kho công đoạn PROC-A")

    assert result is not None
    assert result.intent == "wms_process_inventory"


def test_executive_overview_reports_quality_and_suppressed_kpis(wms_database):
    result = wms_database.query_question(
        "Tình trạng tồn kho tại các công đoạn sản xuất?"
    )

    assert result is not None
    assert result.intent == "wms_executive_overview"
    assert result.status == "PARTIAL"
    assert set(result.reason_codes) == set(MesWmsDatabase.SUPPRESSION_CODES)
    assert "2 mã vật tư" in result.fallback_answer
    assert "3 mã công đoạn" in result.fallback_answer
    assert "66,7%" in result.fallback_answer
    assert "tổng lượng tồn" not in result.fallback_answer.lower()


def test_wms_report_wording_still_routes_to_deterministic_overview(wms_database):
    result = wms_database.query_question("Lập báo cáo tổng quan tồn kho công đoạn WMS")

    assert result is not None
    assert result.intent == "wms_executive_overview"
    assert result.status == "PARTIAL"


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("Kho công đoạn WMS hiện có bao nhiêu mã vật tư?", "vi"),
        ("WMS工程倉庫には現在いくつの資材コードがありますか？", "ja"),
    ],
)
def test_snapshot_item_count_routes_to_overview(wms_database, question, language):
    result = wms_database.query_question(question, language=language)

    assert result is not None
    assert result.intent == "wms_executive_overview"
    assert result.status == "PARTIAL"
    assert "2" in result.fallback_answer


@pytest.mark.parametrize(
    "question",
    [
        "Tổng tồn kho WMS hiện tại là bao nhiêu?",
        "Công đoạn nào có tồn kho WMS nhiều nhất?",
    ],
)
def test_cross_item_total_and_ranking_are_suppressed(wms_database, question):
    result = wms_database.query_question(question)

    assert result is not None
    assert result.intent == "wms_cross_item_aggregate_suppressed"
    assert result.status == "SUPPRESSED"
    assert result.reason_codes == ("UOM_MASTER_UNAVAILABLE",)
    assert "không thể cộng hoặc xếp hạng" in result.fallback_answer
    assert "2026-07-27 13:00:00" in result.fallback_answer
    assert "timezone chưa xác nhận" in result.fallback_answer


@pytest.mark.parametrize(
    "question",
    [
        "WMSの総在庫数はいくつですか？",
        "WMS在庫が最も多い工程はどこですか？",
    ],
)
def test_japanese_cross_item_total_and_ranking_are_suppressed(
    wms_database, question
):
    result = wms_database.query_question(question, language="ja")

    assert result is not None
    assert result.intent == "wms_cross_item_aggregate_suppressed"
    assert result.status == "SUPPRESSED"
    assert result.reason_codes == ("UOM_MASTER_UNAVAILABLE",)
    assert "合算またはランキングできません" in result.fallback_answer
    assert "2026-07-27 13:00:00" in result.fallback_answer
    assert "タイムゾーン未確認" in result.fallback_answer


def test_item_scoped_ranking_wording_is_not_suppressed(wms_database):
    result = wms_database.query_question(
        "Mã vật tư ITEM-A tồn kho nhiều nhất ở công đoạn nào?"
    )

    assert result is not None
    assert result.intent == "wms_item_by_process"
    assert result.status == "PARTIAL"
    assert result.reason_codes == ("UOM_MASTER_UNAVAILABLE",)


@pytest.mark.parametrize(
    ("question", "language", "intent", "reason_code"),
    [
        (
            "Mã vật tư ITEM-A có dưới tồn tối thiểu WMS không?",
            "vi",
            "wms_min_stock_suppressed",
            "MIN_STOCK_CONTRACT_UNVERIFIED",
        ),
        (
            "Mã vật tư ITEM-A trong WMS sắp hết hạn không?",
            "vi",
            "wms_expiry_suppressed",
            "EXPIRY_SOURCE_UNAVAILABLE",
        ),
        (
            "Xu hướng tồn kho WMS của ITEM-A thế nào?",
            "vi",
            "wms_trend_suppressed",
            "SNAPSHOT_HISTORY_NOT_COMPARABLE",
        ),
        (
            "Window time lưu kho WMS của ITEM-A là bao lâu?",
            "vi",
            "wms_window_time_suppressed",
            "WINDOW_TIME_SOURCE_UNAVAILABLE",
        ),
        (
            "WIP tại kho công đoạn WMS hiện bao nhiêu?",
            "vi",
            "wms_wip_ambiguity",
            "PRODUCTION_WIP_SOURCE_UNAVAILABLE",
        ),
        (
            "Bottleneck tồn kho WMS nằm ở công đoạn nào?",
            "vi",
            "wms_bottleneck_suppressed",
            "BOTTLENECK_DEFINITION_UNAVAILABLE",
        ),
        (
            "WMS在庫の最低在庫を確認してください。",
            "ja",
            "wms_min_stock_suppressed",
            "MIN_STOCK_CONTRACT_UNVERIFIED",
        ),
        (
            "WMS在庫推移を教えてください。",
            "ja",
            "wms_trend_suppressed",
            "SNAPSHOT_HISTORY_NOT_COMPARABLE",
        ),
    ],
)
def test_unverified_phase2_kpis_are_explicitly_suppressed(
    wms_database, question, language, intent, reason_code
):
    result = wms_database.query_question(question, language=language)

    assert result is not None
    assert result.intent == intent
    assert result.status == "SUPPRESSED"
    assert result.reason_codes == (reason_code,)
    assert result.rows == []


def test_wip_and_wms_combination_fails_closed(wms_database):
    result = wms_database.query_question(
        "Tồn kho công đoạn và WIP đang sản xuất hiện thế nào?"
    )

    assert result is not None
    assert result.intent == "wms_wip_ambiguity"
    assert result.status == "SUPPRESSED"
    assert result.reason_codes == ("PRODUCTION_WIP_SOURCE_UNAVAILABLE",)


def test_japanese_inventory_query_is_deterministic(wms_database):
    result = wms_database.query_question(
        "工程 PROC-A の在庫状況を教えてください。",
        language="ja",
    )

    assert result is not None
    assert result.intent == "wms_process_inventory"
    assert "PROC-A" in result.fallback_answer
    assert "合算していません" in result.fallback_answer
    assert "2026-07-27 13:00:00" in result.fallback_answer
    assert "タイムゾーン未確認" in result.fallback_answer


def test_current_lot_lookup_is_suppressed(wms_database):
    result = wms_database.query_question(
        "WMS mã vật tư ITEM-A lot vật tư LOT-MATERIAL "
        "công đoạn PROC-A hiện có bao nhiêu?"
    )

    assert result is not None
    assert result.intent == "wms_current_lot_lookup_suppressed"
    assert result.status == "SUPPRESSED"
    assert result.rows == []
    assert result.reason_codes == ("CURRENT_GRAIN_HAS_NO_MEANINGFUL_LOT",)
    assert "không trả số lượng current theo lot" in result.fallback_answer


def test_legacy_archive_exact_key_is_not_called_trend(wms_database):
    result = wms_database.query_question(
        "WMS snapshot mã vật tư ITEM-A lot vật tư LOT-MATERIAL "
        "công đoạn PROC-A có những bản ghi nào?"
    )

    assert result is not None
    assert result.intent == "wms_legacy_archive_exact_key"
    assert [row["archive_id"] for row in result.rows] == [
        "SNAP-NEW",
        "SNAP-OLD",
    ]
    assert result.domain == "LEGACY_ARCHIVE"
    assert result.pagination == {
        "page": 1,
        "page_size": 20,
        "total_count": 2,
        "has_more": False,
    }
    assert "không kết luận xu hướng" in result.fallback_answer


def test_legacy_archive_supports_date_range_and_rejects_bad_page(wms_database):
    result = wms_database.query_question(
        "WMS snapshot mã vật tư ITEM-A lot vật tư LOT-MATERIAL công đoạn "
        "PROC-A từ 2026-07-21 đến 2026-07-21 trang 1 page size 1"
    )
    invalid = wms_database.query_question(
        "WMS snapshot mã vật tư ITEM-A lot vật tư LOT-MATERIAL công đoạn "
        "PROC-A trang 0"
    )

    assert result is not None
    assert [row["archive_id"] for row in result.rows] == ["SNAP-NEW"]
    assert result.pagination["total_count"] == 1
    assert invalid is not None
    assert invalid.intent == "wms_legacy_archive_parameters_invalid"
    assert invalid.rows == []


def test_cross_era_presence_is_suppressed_and_current_not_evaluated(wms_database):
    result = wms_database.query_question(
        "WMS snapshot mã vật tư ITEM-A lot vật tư LOT-MATERIAL công đoạn "
        "PROC-A còn trong current không?"
    )

    assert result is not None
    assert result.intent == "wms_cross_era_presence_diagnostic"
    assert result.status == "SUPPRESSED"
    assert result.reason_codes == ("CROSS_ERA_KEYS_NOT_COMPARABLE",)
    assert result.rows[0]["legacy_archive_exact_key_present"] is True
    assert result.rows[0]["current_exact_lot_presence"] == "NOT_EVALUATED"
    assert result.rows[0]["comparison_eligible"] is False
    assert {row["dataset"] for row in result.dataset_evidence} == {
        "CURRENT_BALANCE",
        "LEGACY_ARCHIVE",
    }


def test_transaction_audit_uses_source_label_without_direction(wms_database):
    result = wms_database.query_question(
        "WMS audit mã giao dịch TRANS-1"
    )

    assert result is not None
    assert result.intent == "wms_raw_transaction_audit"
    assert result.status == "PARTIAL"
    assert result.rows[0]["trans_name"] == "Source label"
    assert result.reason_codes == ("TRANSACTION_SEMANTICS_NOT_IN_SCOPE",)
    assert "không suy diễn chiều nhập/xuất" in result.fallback_answer


def test_transaction_audit_exposes_truncation_metadata(wms_database):
    with sqlite3.connect(wms_database.db_path) as connection:
        connection.executemany(
            """
            INSERT INTO wms_raw_transaction_details (
                source_id, trans_id, item_lot_id, quantity_decimal,
                quantity_valid
            ) VALUES (?, 'TRANS-1', ?, '1', 1)
            """,
            [
                (200 + index, f"LOT-{index:02d}")
                for index in range(20)
            ],
        )
        connection.execute(
            """
            UPDATE wms_dataset_evidence
            SET candidate_row_count = 23,
                inserted_row_count = 23
            WHERE dataset = 'RAW_TRANSACTION_AUDIT'
            """
        )

    result = wms_database.query_question("WMS audit mã giao dịch TRANS-1")

    assert result is not None
    assert len(result.rows) == 20
    assert result.pagination == {
        "page": 1,
        "page_size": 20,
        "total_count": 21,
        "has_more": True,
    }
    assert "20/21 bản ghi" in result.fallback_answer


def test_transaction_audit_hides_unreadable_source_label(wms_database):
    with sqlite3.connect(wms_database.db_path) as connection:
        connection.execute(
            """
            UPDATE wms_raw_transaction_definitions
            SET trans_name = 'Nh?p th?ng th??ng'
            WHERE trans_code = '001'
            """
        )
    result = wms_database.query_question("WMS audit mã giao dịch TRANS-1")

    assert result is not None
    assert "nhãn nguồn không đọc được" in result.fallback_answer
    assert "Nh?p" not in result.fallback_answer


def test_exact_key_feature_requires_all_three_identifiers(wms_database):
    result = wms_database.query_question(
        "WMS snapshot mã vật tư ITEM-A lot vật tư LOT-MATERIAL"
    )

    assert result is not None
    assert result.intent == "wms_exact_key_required"
    assert result.status == "PARTIAL"
    assert result.rows == []


def test_japanese_legacy_archive_exact_key_is_deterministic(wms_database):
    result = wms_database.query_question(
        "WMSスナップショット 資材コード ITEM-A 資材ロット "
        "LOT-MATERIAL 工程 PROC-A の記録を確認してください。",
        language="ja",
    )

    assert result is not None
    assert result.intent == "wms_legacy_archive_exact_key"
    assert "現行在庫との比較" in result.fallback_answer
    assert result.domain == "LEGACY_ARCHIVE"



def test_japanese_cross_era_presence_is_suppressed(wms_database):
    result = wms_database.query_question(
        "WMSスナップショット 資材コード ITEM-A 資材ロット LOT-MATERIAL "
        "工程 PROC-A は現行に存在しますか？",
        language="ja",
    )

    assert result is not None
    assert result.intent == "wms_cross_era_presence_diagnostic"
    assert result.status == "SUPPRESSED"
    assert result.rows[0]["current_exact_lot_presence"] == "NOT_EVALUATED"
    assert "期間・工程コード体系・粒度が異なる" in result.fallback_answer




def test_japanese_snapshot_exact_key_is_deterministic(wms_database):
    result = wms_database.query_question(
        "WMSスナップショット 資材コード ITEM-A 資材ロット "
        "LOT-MATERIAL 工程 PROC-A の記録を確認してください。",
        language="ja",
    )

    assert result is not None
    assert result.intent == "wms_legacy_archive_exact_key"
    assert "現行在庫との比較" in result.fallback_answer
    assert "推移や増減の判定ではありません" in result.fallback_answer


@pytest.mark.parametrize(
    "question",
    [
        "vật tư 10kg còn bao nhiêu ở kho công đoạn?",
        "vật tư 2026 tồn kho thế nào?",
        "vật tư 100pcs ở công đoạn nào?",
        "nguyên vật liệu 5 thùng còn không?",
        "vật tư 2,5kg tồn kho",
        "vật tư 500 tồn kho công đoạn",
        "資材10個の在庫",
    ],
)
def test_quantity_and_unit_are_not_read_as_item_code(wms_database, question):
    """Quantities, units and bare years must never be captured as an item code."""
    assert wms_database._extract_code_after(
        question, wms_database.ITEM_LABEL
    ) is None


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("mã vật tư 0202810009 tồn kho công đoạn V3500", "0202810009"),
        ("mã vật tư 04010786 ở đâu", "04010786"),
        ("mã vật tư ITEM-A tồn kho", "ITEM-A"),
        ("資材コード ITEM-A", "ITEM-A"),
        ("資材0202810009の在庫", "0202810009"),
    ],
)
def test_real_item_codes_survive_measurement_guard(
    wms_database, question, expected
):
    assert (
        wms_database._extract_code_after(question, wms_database.ITEM_LABEL)
        == expected
    )


@pytest.mark.parametrize(
    "question",
    [
        "資材コード:ITEM-A、工程PROC-A",
        "資材コード：ITEM-A、工程：PROC-A",
        "資材コード　ITEM-A、工程　PROC-A",
        "資材コードはITEM-A、工程はPROC-A",
        "資材コードがITEM-A、工程がPROC-A",
        "資材ITEM-A、工程PROC-A",
    ],
)
def test_japanese_label_code_separators_are_all_supported(
    wms_database, question
):
    """Full-width colon/space and topic particles must join label to code."""
    assert (
        wms_database._extract_code_after(question, wms_database.ITEM_LABEL)
        == "ITEM-A"
    )
    assert (
        wms_database._extract_code_after(question, wms_database.PROCESS_LABEL)
        == "PROC-A"
    )


@pytest.mark.parametrize(
    "question",
    [
        "mã vật tư ITEM-A công đoạn PROC-A",
        "mã vật tư: ITEM-A, công đoạn: PROC-A",
        "mã vật tư là ITEM-A, công đoạn là PROC-A",
    ],
)
def test_vietnamese_separators_unaffected_by_japanese_support(
    wms_database, question
):
    assert (
        wms_database._extract_code_after(question, wms_database.ITEM_LABEL)
        == "ITEM-A"
    )
    assert (
        wms_database._extract_code_after(question, wms_database.PROCESS_LABEL)
        == "PROC-A"
    )


def test_japanese_short_shizai_prefix_without_space(wms_database):
    """資材ITEM-A (no space, short prefix) must extract ITEM-A without matching 資材ロット."""
    result = wms_database.query_question(
        "資材ITEM-A、資材ロットLOT-MATERIAL、工程PROC-AのWMSスナップショット履歴",
        language="ja",
    )

    assert result is not None
    assert result.intent == "wms_legacy_archive_exact_key"
    assert len(result.rows) == 2



def test_non_wms_question_is_not_intercepted(wms_database):
    assert wms_database.query_question("Lot nào có nhiều lỗi nhất?") is None


def test_missing_snapshot_fails_closed_for_wms_question(tmp_path):
    database = MesWmsDatabase(tmp_path / "missing.sqlite")

    result = database.query_question("Tình trạng tồn kho công đoạn?")

    assert result is not None
    assert result.intent == "wms_unavailable"
    assert result.status == "SUPPRESSED"
    assert "chưa sẵn sàng" in result.fallback_answer
    assert "Mốc dữ liệu nguồn: chưa xác nhận" in result.fallback_answer


def test_v1_snapshot_is_incompatible_and_never_returns_quantities(tmp_path):
    db_path = tmp_path / "v1.sqlite"
    schema = Path(__file__).parents[1] / "database" / "schema" / "mes_wms.sql"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(schema.read_text(encoding="utf-8"))
        connection.executemany(
            "INSERT INTO schema_metadata (key, value) VALUES (?, ?)",
            [
                ("schema_version", "1"),
                ("source_schema", "MES_WMS_MKHC"),
                ("plant_id", "MKHC"),
            ],
        )
    database = MesWmsDatabase(db_path)

    result = database.query_question("Tình trạng tồn kho công đoạn WMS?")

    assert result is not None
    assert result.intent == "wms_incompatible"
    assert result.status == "SUPPRESSED"
    assert result.rows == []
    assert "WMS_SNAPSHOT_INCOMPATIBLE" in result.reason_codes
    assert database.snapshot_version() == ""
    assert database.status()["state"] == "INCOMPATIBLE"


def test_empty_source_as_of_is_shown_as_unconfirmed(wms_database):
    answer = wms_database._with_freshness(
        "Không tìm thấy dữ liệu WMS.",
        source_as_of="",
        source_timezone="unverified",
    )
    japanese = wms_database._with_freshness(
        "WMSデータが見つかりません。",
        source_as_of="",
        source_timezone="unverified",
    )

    assert "Mốc dữ liệu nguồn: chưa xác nhận" in answer
    assert "データ基準時点: 未確認" in japanese


def test_legacy_archive_uses_archive_date_as_freshness(wms_database):
    """Legacy archive freshness must not use current balance freshness."""
    result = wms_database.query_question(
        "WMS snapshot mã vật tư ITEM-A lot vật tư LOT-MATERIAL "
        "công đoạn PROC-A có những bản ghi nào?"
    )

    assert result is not None
    assert result.intent == "wms_legacy_archive_exact_key"
    assert result.domain == "LEGACY_ARCHIVE"
    assert result.source_as_of == "2026-07-21 10:00:00"
    assert "2026-07-21 10:00:00" in result.fallback_answer
    assert "2026-07-27 13:00:00" not in result.fallback_answer




def test_completed_movement_is_suppressed(wms_database):
    result = wms_database.query_question(
        "WMS completed movement của ITEM-A là gì?"
    )

    assert result is not None
    assert result.intent == "wms_completed_movements_suppressed"
    assert result.status == "SUPPRESSED"
    assert result.domain == "RAW_TRANSACTION_AUDIT"
    assert result.reason_codes == ("COMPLETED_MOVEMENTS_NOT_VERIFIED",)




def test_invalid_quantity_rows_are_not_returned_by_current_view(wms_database):
    with sqlite3.connect(wms_database.db_path) as connection:
        connection.execute(
            """
            INSERT INTO wms_current_balances (
                source_id, item_code, quantity_decimal, quantity_valid,
                quantity_error, process_id, process_pk
            ) VALUES (99, 'ITEM-INVALID', NULL, 0, 'not_numeric', 'PROC-A', 1)
            """
        )
        connection.execute(
            """
            UPDATE wms_dataset_evidence
            SET candidate_row_count = 4,
                inserted_row_count = 4,
                invalid_quantity_row_count = 1
            WHERE dataset = 'CURRENT_BALANCE'
            """
        )
    result = wms_database.query_question("Mã vật tư ITEM-INVALID tồn kho WMS")

    assert result is not None
    assert result.intent == "wms_current_quantity_invalid"
    assert result.status == "PARTIAL"
    assert result.reason_codes == ("QUANTITY_EVIDENCE_INCOMPLETE",)
    assert "không kết luận là không có tồn kho" in result.fallback_answer




def test_snapshot_presence_warns_about_namespace_mismatch(wms_database):
    """Presence diagnostic never evaluates current lot presence."""
    result = wms_database.query_question(
        "WMS snapshot mã vật tư ITEM-A lot vật tư LOT-MATERIAL công đoạn "
        "PROC-A còn trong current không?"
    )

    assert result is not None
    assert result.intent == "wms_cross_era_presence_diagnostic"
    assert result.status == "SUPPRESSED"
    assert result.reason_codes == ("CROSS_ERA_KEYS_NOT_COMPARABLE",)
    assert result.rows[0]["current_exact_lot_presence"] == "NOT_EVALUATED"
    assert result.rows[0]["comparison_eligible"] is False
    assert "không được diễn giải" in result.fallback_answer



def test_snapshot_presence_ja_warns_about_namespace_mismatch(wms_database):
    result = wms_database.query_question(
        "WMSスナップショット 資材コード ITEM-A 資材ロット LOT-MATERIAL "
        "工程 PROC-A は現行に存在しますか？",
        language="ja",
    )

    assert result is not None
    assert result.intent == "wms_cross_era_presence_diagnostic"
    assert result.status == "SUPPRESSED"
    assert result.rows[0]["current_exact_lot_presence"] == "NOT_EVALUATED"



def test_lot_id_with_spaces_is_extracted_correctly(wms_database):
    """Lot values with spaces remain accepted for legacy archive lookup."""
    result = wms_database.query_question(
        "WMS snapshot mã vật tư ITEM-A lot vật tư Lot 987654321 "
        "công đoạn PROC-A có bản ghi nào?"
    )

    assert result is not None
    assert result.intent == "wms_legacy_archive_exact_key_not_found"
    assert result.rows == []



def test_lot_id_without_spaces_is_suppressed_for_current(wms_database):
    result = wms_database.query_question(
        "WMS mã vật tư ITEM-A lot vật tư DR25K03953 "
        "công đoạn PROC-A hiện có bản ghi nào?"
    )

    assert result is not None
    assert result.intent == "wms_current_lot_lookup_suppressed"
    assert result.status == "SUPPRESSED"
    assert result.rows == []


def _set_optional_dataset_unobserved(wms_database, dataset, capability):
    with sqlite3.connect(wms_database.db_path) as connection:
        tables = {
            "LEGACY_ARCHIVE": ("wms_legacy_archive_records",),
            "RAW_TRANSACTION_AUDIT": (
                "wms_raw_transaction_headers",
                "wms_raw_transaction_definitions",
                "wms_raw_transaction_details",
            ),
        }[dataset]
        for table in tables:
            connection.execute(f"DELETE FROM {table}")
        connection.execute(
            """
            UPDATE wms_dataset_evidence
            SET status = 'SUPPRESSED',
                reason_code = 'DATASET_NOT_OBSERVED_IN_EXPORT',
                source_state = 'NOT_OBSERVED_IN_EXPORT',
                candidate_row_count = 0,
                inserted_row_count = 0,
                invalid_quantity_row_count = 0,
                source_as_of = '',
                source_as_of_state = 'UNAVAILABLE'
            WHERE dataset = ?
            """,
            (dataset,),
        )
        connection.execute(
            """
            UPDATE wms_capability_status
            SET status = 'SUPPRESSED',
                reason_code = 'DATASET_NOT_OBSERVED_IN_EXPORT'
            WHERE capability = ?
            """,
            (capability,),
        )


def test_unobserved_legacy_archive_is_suppressed_not_not_found(wms_database):
    _set_optional_dataset_unobserved(
        wms_database,
        "LEGACY_ARCHIVE",
        "LEGACY_ARCHIVE_EXACT_KEY_QUERY",
    )

    result = wms_database.query_question(
        "WMS snapshot mã vật tư ITEM-A lot vật tư LOT-MATERIAL "
        "công đoạn PROC-A có những bản ghi nào?"
    )

    assert result is not None
    assert result.intent == "wms_legacy_archive_unobserved"
    assert result.status == "SUPPRESSED"
    assert result.reason_codes == ("DATASET_NOT_OBSERVED_IN_EXPORT",)
    assert "không kết luận có hay không có bản ghi" in result.fallback_answer


def test_unobserved_raw_audit_is_suppressed_not_not_found(wms_database):
    _set_optional_dataset_unobserved(
        wms_database,
        "RAW_TRANSACTION_AUDIT",
        "RAW_TRANSACTION_AUDIT_QUERY",
    )

    result = wms_database.query_question("WMS audit mã giao dịch TRANS-404")

    assert result is not None
    assert result.intent == "wms_raw_transaction_audit_unobserved"
    assert result.status == "SUPPRESSED"
    assert result.reason_codes == ("DATASET_NOT_OBSERVED_IN_EXPORT",)
    assert "không kết luận có hay không có bản ghi" in result.fallback_answer


def test_cross_era_unobserved_archive_never_reports_absence(wms_database):
    _set_optional_dataset_unobserved(
        wms_database,
        "LEGACY_ARCHIVE",
        "LEGACY_ARCHIVE_EXACT_KEY_QUERY",
    )

    result = wms_database.query_question(
        "WMS snapshot mã vật tư ITEM-A lot vật tư LOT-MATERIAL "
        "công đoạn PROC-A còn trong current không?"
    )

    assert result is not None
    assert result.intent == "wms_cross_era_presence_unobserved"
    assert result.rows[0]["legacy_archive_exact_key_present"] == "NOT_EVALUATED"
    assert result.rows[0]["current_exact_lot_presence"] == "NOT_EVALUATED"
    assert result.status == "SUPPRESSED"
    assert "DATASET_NOT_OBSERVED_IN_EXPORT" in result.reason_codes


@pytest.mark.parametrize(
    "question",
    [
        "資材コード ITEM-A の不良数を教えてください。",
        "資材 ITEM-A の生産エラー履歴を確認してください。",
        "品番 ITEM-A の製造不良は何件ですか？",
    ],
)
def test_japanese_material_mes_questions_are_not_intercepted(wms_database, question):
    assert wms_database.query_question(question, language="ja") is None


def test_japanese_legacy_not_found_is_fully_localized(wms_database):
    result = wms_database.query_question(
        "WMSスナップショット 資材コード ITEM-Z 資材ロット LOT-Z "
        "工程 PROC-Z の記録を確認してください。",
        language="ja",
    )

    assert result is not None
    assert result.intent == "wms_legacy_archive_exact_key_not_found"
    assert "指定範囲内にありません" in result.fallback_answer
    assert "không" not in result.fallback_answer


def test_raw_audit_answer_includes_raw_quantities_without_net_semantics(wms_database):
    result = wms_database.query_question("WMS audit mã giao dịch TRANS-1")

    assert result is not None
    assert "header_qty=10" in result.fallback_answer
    assert "detail_qty=10" in result.fallback_answer
    assert "số lượng ròng" in result.fallback_answer


def test_wms_status_never_exposes_database_path(wms_database):
    status = wms_database.status()

    assert "db_path" not in status
    assert "/home/" not in str(status)
    assert "mes_wms.sqlite" not in str(status)


def test_japanese_archive_markers_route_to_legacy(wms_database):
    result = wms_database.query_question(
        "WMSアーカイブ 資材コード ITEM-A 資材ロット LOT-MATERIAL "
        "工程 PROC-A の履歴",
        language="ja",
    )

    assert result is not None
    assert result.intent == "wms_legacy_archive_exact_key"


def test_partial_legacy_capability_is_preserved_in_result(wms_database):
    with sqlite3.connect(wms_database.db_path) as connection:
        connection.execute(
            """
            UPDATE wms_legacy_archive_records
            SET quantity_decimal = NULL,
                quantity_valid = 0,
                quantity_error = 'not_numeric'
            WHERE source_id = 10
            """
        )
        connection.execute(
            """
            UPDATE wms_dataset_evidence
            SET status = 'PARTIAL',
                reason_code = 'QUANTITY_EVIDENCE_INCOMPLETE',
                invalid_quantity_row_count = 1
            WHERE dataset = 'LEGACY_ARCHIVE'
            """
        )
        connection.execute(
            """
            UPDATE wms_capability_status
            SET status = 'PARTIAL',
                reason_code = 'QUANTITY_EVIDENCE_INCOMPLETE'
            WHERE capability = 'LEGACY_ARCHIVE_EXACT_KEY_QUERY'
            """
        )

    result = wms_database.query_question(
        "WMS snapshot mã vật tư ITEM-A lot vật tư LOT-MATERIAL "
        "công đoạn PROC-A có những bản ghi nào?"
    )

    assert result is not None
    assert result.status == "PARTIAL"
    assert result.reason_codes == ("QUANTITY_EVIDENCE_INCOMPLETE",)
    assert "chưa rõ" in result.fallback_answer


def test_japanese_quantity_uses_japanese_number_separators():
    assert MesWmsDatabase._quantity("1234", "ja") == "1,234"
    assert MesWmsDatabase._quantity("1234.5", "ja") == "1,234.5"
