"""Shared Phase 2C WMS schema, capability, and compatibility contract."""

from __future__ import annotations

from datetime import datetime
import sqlite3
from typing import Any


SCHEMA_VERSION = "4"
DATA_CONTRACT_VERSION = "wms-current-balance-v1"
SEMANTIC_CONTRACT_VERSION = "wms-phase2c-v1"
CURRENT_SEMANTIC_EPOCH = "CURRENT_POST_2026_01_15"
LEGACY_SEMANTIC_EPOCH = "LEGACY_PRE_2026_01_15"
AUDIT_SEMANTIC_EPOCH = "RAW_SOURCE_AUDIT"
SOURCE_SCHEMA = "MES_WMS_MKHC"
PLANT_ID = "MKHC"

DATASET_CURRENT = "CURRENT_BALANCE"
DATASET_LEGACY = "LEGACY_ARCHIVE"
DATASET_AUDIT = "RAW_TRANSACTION_AUDIT"

REASON_WMS_DISABLED = "WMS_DISABLED"
REASON_SNAPSHOT_UNAVAILABLE = "WMS_SNAPSHOT_UNAVAILABLE"
REASON_SNAPSHOT_INCOMPATIBLE = "WMS_SNAPSHOT_INCOMPATIBLE"
REASON_SNAPSHOT_QUERY_ERROR = "WMS_SNAPSHOT_QUERY_ERROR"
REASON_SOURCE_AS_OF_UNCONFIRMED = "WMS_SOURCE_AS_OF_UNCONFIRMED"
REASON_UOM_UNAVAILABLE = "UOM_MASTER_UNAVAILABLE"
REASON_MIN_STOCK_UNVERIFIED = "MIN_STOCK_CONTRACT_UNVERIFIED"
REASON_EXPIRY_UNAVAILABLE = "EXPIRY_SOURCE_UNAVAILABLE"
REASON_WINDOW_TIME_UNAVAILABLE = "WINDOW_TIME_SOURCE_UNAVAILABLE"
REASON_TREND_UNCOMPARABLE = "SNAPSHOT_HISTORY_NOT_COMPARABLE"
REASON_WIP_UNAVAILABLE = "PRODUCTION_WIP_SOURCE_UNAVAILABLE"
REASON_BOTTLENECK_UNDEFINED = "BOTTLENECK_DEFINITION_UNAVAILABLE"
REASON_SNAPSHOT_DELTA_UNAVAILABLE = "SNAPSHOT_DELTA_NOT_IN_SCOPE"
REASON_TRANSACTION_SEMANTICS_UNAVAILABLE = "TRANSACTION_SEMANTICS_NOT_IN_SCOPE"
REASON_CURRENT_LOT_UNAVAILABLE = "CURRENT_GRAIN_HAS_NO_MEANINGFUL_LOT"
REASON_CROSS_ERA_UNCOMPARABLE = "CROSS_ERA_KEYS_NOT_COMPARABLE"
REASON_COMPLETED_MOVEMENTS_UNAVAILABLE = "COMPLETED_MOVEMENTS_NOT_VERIFIED"
REASON_DATASET_NOT_OBSERVED = "DATASET_NOT_OBSERVED_IN_EXPORT"
REASON_CURRENT_GRAIN_DUPLICATE = "CURRENT_BALANCE_GRAIN_DUPLICATE"
REASON_QUANTITY_EVIDENCE_INCOMPLETE = "QUANTITY_EVIDENCE_INCOMPLETE"
REASON_TRANSACTION_HEADER_NOT_OBSERVED = "RAW_TRANSACTION_HEADER_NOT_OBSERVED"
REASON_SQL_AGENT_UNVERIFIED = "SQL_AGENT_ANSWER_UNVERIFIED"

KPI_SUPPRESSION_REASONS = {
    "CROSS_ITEM_AGGREGATE": REASON_UOM_UNAVAILABLE,
    "MIN_STOCK": REASON_MIN_STOCK_UNVERIFIED,
    "EXPIRY": REASON_EXPIRY_UNAVAILABLE,
    "WINDOW_TIME": REASON_WINDOW_TIME_UNAVAILABLE,
    "TREND": REASON_TREND_UNCOMPARABLE,
    "PRODUCTION_WIP": REASON_WIP_UNAVAILABLE,
    "BOTTLENECK": REASON_BOTTLENECK_UNDEFINED,
}

BASE_CAPABILITY_STATUSES = {
    "CURRENT_BALANCE_ITEM_PROCESS_QUERY": ("PARTIAL", REASON_UOM_UNAVAILABLE),
    "CURRENT_BALANCE_PROCESS_ITEM_LISTING": ("PARTIAL", REASON_UOM_UNAVAILABLE),
    "CURRENT_LOT_LOOKUP": ("SUPPRESSED", REASON_CURRENT_LOT_UNAVAILABLE),
    "CROSS_ERA_PRESENCE_COMPARISON": (
        "SUPPRESSED",
        REASON_CROSS_ERA_UNCOMPARABLE,
    ),
    "COMPLETED_MOVEMENTS": (
        "SUPPRESSED",
        REASON_COMPLETED_MOVEMENTS_UNAVAILABLE,
    ),
    "SNAPSHOT_QUALITY": ("AVAILABLE", ""),
    **{
        capability: ("SUPPRESSED", reason)
        for capability, reason in KPI_SUPPRESSION_REASONS.items()
    },
}
# Compatibility alias retained for callers/tests that import the former name.
CAPABILITY_STATUSES = BASE_CAPABILITY_STATUSES

REQUIRED_OBJECT_TYPES = {
    "schema_metadata": "table",
    "import_batches": "table",
    "wms_dataset_evidence": "table",
    "wms_capability_status": "table",
    "wms_processes": "table",
    "wms_current_balances": "table",
    "wms_legacy_archive_records": "table",
    "wms_raw_transaction_definitions": "table",
    "wms_raw_transaction_headers": "table",
    "wms_raw_transaction_details": "table",
    "v_wms_current_balance_by_process_item": "view",
    "v_wms_legacy_archive_exact_key": "view",
    "v_wms_raw_transaction_audit": "view",
    "v_wms_current_quality": "view",
}
FORBIDDEN_V4_OBJECTS = {
    "wms_current_items",
    "wms_snapshot_records",
    "wms_process_item_balances",
    "v_wms_current_exact_key",
    "v_wms_snapshot_exact_key",
    "v_wms_completed_movements",
}
REQUIRED_COLUMNS = {
    "schema_metadata": {"key", "value"},
    "wms_dataset_evidence": {
        "dataset",
        "status",
        "reason_code",
        "source_tables",
        "source_state",
        "candidate_row_count",
        "inserted_row_count",
        "invalid_quantity_row_count",
        "source_as_of",
        "source_as_of_state",
        "source_as_of_basis",
        "source_timezone",
        "semantic_epoch",
        "evidence_basis",
    },
    "wms_capability_status": {
        "capability",
        "status",
        "reason_code",
        "evidence_basis",
        "contract_version",
    },
    "wms_current_balances": {
        "source_id",
        "item_code",
        "process_id",
        "quantity_decimal",
        "quantity_valid",
        "quantity_error",
        "time_update",
        "trans_id",
    },
    "wms_legacy_archive_records": {
        "source_id",
        "archive_id",
        "archive_date",
        "item_code",
        "item_lot_id",
        "process_id",
        "quantity_decimal",
        "quantity_valid",
        "quantity_error",
    },
    "v_wms_current_balance_by_process_item": {
        "source_id",
        "process_id",
        "process_name",
        "process_mapped",
        "item_code",
        "quantity_decimal",
        "quantity_valid",
        "latest_update",
        "trans_id",
    },
    "v_wms_legacy_archive_exact_key": {
        "source_id",
        "archive_id",
        "archive_date",
        "item_code",
        "item_lot_id",
        "process_id",
        "quantity_decimal",
        "quantity_valid",
    },
    "v_wms_raw_transaction_audit": {
        "trans_id",
        "trans_code",
        "trans_name",
        "trans_date",
        "process_id",
        "item_code",
        "item_lot_id",
        "header_quantity_decimal",
        "detail_quantity_decimal",
        "raw_trans_status",
        "raw_deleted",
    },
    "v_wms_current_quality": {
        "current_row_count",
        "valid_quantity_row_count",
        "invalid_quantity_row_count",
        "mapped_process_row_count",
        "unmapped_process_row_count",
        "distinct_item_count",
        "distinct_process_code_count",
        "source_as_of",
    },
}
REQUIRED_METADATA_KEYS = {
    "schema_version",
    "data_contract_version",
    "semantic_contract_version",
    "semantic_epoch",
    "source_schema",
    "plant_id",
    "source_export_date",
    "imported_at",
}
VALID_SOURCE_AS_OF_STATES = {"DERIVED_UNVERIFIED", "UNAVAILABLE"}
VALID_SOURCE_STATES = {
    "PRESENT_NONEMPTY",
    "PARTIAL_SOURCE_OBSERVED",
    "NOT_OBSERVED_IN_EXPORT",
}
VALID_DATASET_STATUSES = {"AVAILABLE", "PARTIAL", "SUPPRESSED"}
EXPECTED_DATASETS = {DATASET_CURRENT, DATASET_LEGACY, DATASET_AUDIT}
EXPECTED_CURRENT_BASIS = "MAX(PW_CURRENT_ITEM.TIME_UPDATE)"
EXPECTED_LEGACY_BASIS = "MAX(PW_SNAPSHORT.SNAPSHORT_DATE)"
EXPECTED_AUDIT_BASIS = "MAX(PW_TRANSACTION.TRANS_DATE)"
DATASET_COUNT_QUERIES = {
    DATASET_CURRENT: (
        "SELECT COUNT(*) FROM wms_current_balances",
        "SELECT COUNT(*) FROM wms_current_balances WHERE quantity_valid = 0",
    ),
    DATASET_LEGACY: (
        "SELECT COUNT(*) FROM wms_legacy_archive_records",
        "SELECT COUNT(*) FROM wms_legacy_archive_records WHERE quantity_valid = 0",
    ),
    DATASET_AUDIT: (
        """
        SELECT
            (SELECT COUNT(*) FROM wms_raw_transaction_headers) +
            (SELECT COUNT(*) FROM wms_raw_transaction_definitions) +
            (SELECT COUNT(*) FROM wms_raw_transaction_details)
        """,
        """
        SELECT
            (SELECT COUNT(*) FROM wms_raw_transaction_headers WHERE quantity_valid = 0) +
            (SELECT COUNT(*) FROM wms_raw_transaction_details WHERE quantity_valid = 0)
        """,
    ),
}


def capability_statuses_for_datasets(
    dataset_statuses: dict[str, tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    """Build snapshot capability status from observed dataset evidence."""
    statuses = dict(BASE_CAPABILITY_STATUSES)
    current_status, current_reason = dataset_statuses.get(
        DATASET_CURRENT,
        ("SUPPRESSED", REASON_DATASET_NOT_OBSERVED),
    )
    for capability in (
        "CURRENT_BALANCE_ITEM_PROCESS_QUERY",
        "CURRENT_BALANCE_PROCESS_ITEM_LISTING",
        "SNAPSHOT_QUALITY",
    ):
        statuses[capability] = (
            (current_status, current_reason)
            if current_status in {"AVAILABLE", "PARTIAL"}
            else ("SUPPRESSED", current_reason or REASON_DATASET_NOT_OBSERVED)
        )
    legacy_status, legacy_reason = dataset_statuses.get(
        DATASET_LEGACY,
        ("SUPPRESSED", REASON_DATASET_NOT_OBSERVED),
    )
    audit_status, audit_reason = dataset_statuses.get(
        DATASET_AUDIT,
        ("SUPPRESSED", REASON_DATASET_NOT_OBSERVED),
    )
    statuses["LEGACY_ARCHIVE_EXACT_KEY_QUERY"] = (
        (legacy_status, legacy_reason)
        if legacy_status in {"AVAILABLE", "PARTIAL"}
        else ("SUPPRESSED", legacy_reason or REASON_DATASET_NOT_OBSERVED)
    )
    statuses["RAW_TRANSACTION_AUDIT_QUERY"] = (
        (audit_status, audit_reason)
        if audit_status in {"AVAILABLE", "PARTIAL"}
        else ("SUPPRESSED", audit_reason or REASON_DATASET_NOT_OBSERVED)
    )
    return statuses


def _table_columns(connection: sqlite3.Connection, object_name: str) -> set[str]:
    escaped = object_name.replace('"', '""')
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
    }


def _valid_iso_timestamp(value: str) -> bool:
    if not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _dataset_evidence(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, str]]:
    rows = connection.execute(
        """
        SELECT dataset, status, reason_code, source_tables, source_state,
               candidate_row_count, inserted_row_count,
               invalid_quantity_row_count, source_as_of,
               source_as_of_state, source_as_of_basis, source_timezone,
               semantic_epoch, evidence_basis
        FROM wms_dataset_evidence
        """
    ).fetchall()
    columns = [
        "dataset",
        "status",
        "reason_code",
        "source_tables",
        "source_state",
        "candidate_row_count",
        "inserted_row_count",
        "invalid_quantity_row_count",
        "source_as_of",
        "source_as_of_state",
        "source_as_of_basis",
        "source_timezone",
        "semantic_epoch",
        "evidence_basis",
    ]
    return {
        _as_text(row[0]): {
            key: _as_text(value) for key, value in zip(columns, row)
        }
        for row in rows
    }


def _expected_capabilities_from_evidence(
    evidence: dict[str, dict[str, str]],
) -> dict[str, tuple[str, str]]:
    return capability_statuses_for_datasets(
        {
            dataset: (row.get("status", ""), row.get("reason_code", ""))
            for dataset, row in evidence.items()
        }
    )


def validate_snapshot_connection(connection: sqlite3.Connection) -> dict[str, Any]:
    """Return a non-throwing structural and semantic compatibility report."""
    try:
        object_rows = connection.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        objects = {str(row[0]): str(row[1]) for row in object_rows}
        invalid_objects = [
            name
            for name, expected_type in REQUIRED_OBJECT_TYPES.items()
            if objects.get(name) != expected_type
        ]
        forbidden_objects = sorted(FORBIDDEN_V4_OBJECTS.intersection(objects))
        invalid_columns = {
            name: sorted(required - _table_columns(connection, name))
            for name, required in REQUIRED_COLUMNS.items()
            if name in objects and required - _table_columns(connection, name)
        }

        metadata: dict[str, str] = {}
        if objects.get("schema_metadata") == "table":
            metadata = {
                str(row[0]): str(row[1])
                for row in connection.execute(
                    "SELECT key, value FROM schema_metadata"
                ).fetchall()
            }
        missing_metadata = sorted(
            key for key in REQUIRED_METADATA_KEYS if not metadata.get(key)
        )

        evidence: dict[str, dict[str, str]] = {}
        if (
            objects.get("wms_dataset_evidence") == "table"
            and not invalid_columns.get("wms_dataset_evidence")
        ):
            evidence = _dataset_evidence(connection)
        invalid_datasets = sorted(EXPECTED_DATASETS - set(evidence))

        semantic_errors: list[str] = []
        current = evidence.get(DATASET_CURRENT, {})
        legacy = evidence.get(DATASET_LEGACY, {})
        audit = evidence.get(DATASET_AUDIT, {})
        expected_specs = (
            (
                DATASET_CURRENT,
                current,
                EXPECTED_CURRENT_BASIS,
                CURRENT_SEMANTIC_EPOCH,
                "SELECT MAX(time_update) FROM wms_current_balances",
                True,
            ),
            (
                DATASET_LEGACY,
                legacy,
                EXPECTED_LEGACY_BASIS,
                LEGACY_SEMANTIC_EPOCH,
                "SELECT MAX(archive_date) FROM wms_legacy_archive_records",
                False,
            ),
            (
                DATASET_AUDIT,
                audit,
                EXPECTED_AUDIT_BASIS,
                AUDIT_SEMANTIC_EPOCH,
                "SELECT MAX(trans_date) FROM wms_raw_transaction_headers",
                False,
            ),
        )
        for dataset, row, basis, epoch, query, required in expected_specs:
            if not row:
                continue
            source_state = row.get("source_state", "")
            source_as_of = row.get("source_as_of", "")
            state = row.get("source_as_of_state", "")
            status = row.get("status", "")
            if source_state not in VALID_SOURCE_STATES:
                semantic_errors.append(f"{dataset}:invalid_source_state")
            if status not in VALID_DATASET_STATUSES:
                semantic_errors.append(f"{dataset}:invalid_status")
            observed = source_state in {"PRESENT_NONEMPTY", "PARTIAL_SOURCE_OBSERVED"}
            partial_source = source_state == "PARTIAL_SOURCE_OBSERVED"
            if required and source_state != "PRESENT_NONEMPTY":
                semantic_errors.append(f"{dataset}:required_not_observed")
            if observed and int(row.get("inserted_row_count") or 0) < 1:
                semantic_errors.append(f"{dataset}:observed_without_inserted_rows")
            if not observed and int(row.get("inserted_row_count") or 0) != 0:
                semantic_errors.append(f"{dataset}:unobserved_with_inserted_rows")
            if partial_source and status not in {"PARTIAL", "SUPPRESSED"}:
                semantic_errors.append(f"{dataset}:partial_source_invalid_status")
            if dataset == DATASET_CURRENT and status not in {"AVAILABLE", "PARTIAL"}:
                semantic_errors.append(f"{dataset}:invalid_current_status")
            if not observed and status != "SUPPRESSED":
                semantic_errors.append(f"{dataset}:invalid_unobserved_status")
            if row.get("source_as_of_basis") != basis:
                semantic_errors.append(f"{dataset}:invalid_as_of_basis")
            if row.get("semantic_epoch") != epoch:
                semantic_errors.append(f"{dataset}:invalid_semantic_epoch")
            if row.get("source_timezone") != "unverified":
                semantic_errors.append(f"{dataset}:timezone_not_unverified")
            if state not in VALID_SOURCE_AS_OF_STATES:
                semantic_errors.append(f"{dataset}:invalid_as_of_state")
            if not row.get("source_tables"):
                semantic_errors.append(f"{dataset}:missing_source_tables")
            if not row.get("evidence_basis"):
                semantic_errors.append(f"{dataset}:missing_evidence_basis")
            actual_count_sql, invalid_count_sql = DATASET_COUNT_QUERIES[dataset]
            actual_count = int(connection.execute(actual_count_sql).fetchone()[0] or 0)
            actual_invalid_count = int(
                connection.execute(invalid_count_sql).fetchone()[0] or 0
            )
            if int(row.get("candidate_row_count") or 0) != actual_count:
                semantic_errors.append(f"{dataset}:candidate_count_mismatch")
            if int(row.get("inserted_row_count") or 0) != actual_count:
                semantic_errors.append(f"{dataset}:inserted_count_mismatch")
            if int(row.get("invalid_quantity_row_count") or 0) != actual_invalid_count:
                semantic_errors.append(f"{dataset}:invalid_quantity_count_mismatch")
            actual = _as_text(connection.execute(query).fetchone()[0])
            if source_state == "PRESENT_NONEMPTY":
                if source_as_of:
                    if not _valid_iso_timestamp(source_as_of):
                        semantic_errors.append(f"{dataset}:invalid_as_of_timestamp")
                    if state != "DERIVED_UNVERIFIED":
                        semantic_errors.append(f"{dataset}:invalid_observed_as_of_state")
                    if source_as_of != actual:
                        semantic_errors.append(f"{dataset}:as_of_mismatch")
                elif (
                    dataset != DATASET_AUDIT
                    or state != "UNAVAILABLE"
                    or actual
                    or status != "PARTIAL"
                ):
                    semantic_errors.append(f"{dataset}:missing_observed_as_of")
            elif partial_source:
                if source_as_of or state != "UNAVAILABLE" or actual:
                    semantic_errors.append(f"{dataset}:partial_source_as_of_mismatch")
            elif source_as_of or state != "UNAVAILABLE" or actual:
                semantic_errors.append(f"{dataset}:unavailable_evidence_mismatch")
            elif actual_count != 0:
                semantic_errors.append(f"{dataset}:unobserved_data_present")
            if dataset == DATASET_CURRENT and (
                status != "PARTIAL" or row.get("reason_code") != REASON_UOM_UNAVAILABLE
            ):
                semantic_errors.append(f"{dataset}:current_capability_not_partial")
            if dataset == DATASET_LEGACY and actual_invalid_count and status == "AVAILABLE":
                semantic_errors.append(f"{dataset}:invalid_quantity_status")
            if dataset == DATASET_AUDIT and actual_invalid_count and status == "AVAILABLE":
                semantic_errors.append(f"{dataset}:invalid_quantity_status")
            if dataset == DATASET_AUDIT and partial_source and status == "AVAILABLE":
                semantic_errors.append(f"{dataset}:partial_source_status")

        index_rows = connection.execute(
            "PRAGMA index_list('wms_current_balances')"
        ).fetchall()
        has_grain_unique = any(
            int(row[2] or 0)
            and not int(row[4] or 0)
            and set(
                str(column[2])
                for column in connection.execute(
                    f"PRAGMA index_info('{str(row[1]).replace(chr(39), chr(39) * 2)}')"
                ).fetchall()
            ) == {"process_id", "item_code"}
            for row in index_rows
        )
        duplicate_grains = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT process_id, item_code
                FROM wms_current_balances
                GROUP BY process_id, item_code
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        if not has_grain_unique:
            semantic_errors.append("CURRENT_BALANCE:missing_grain_unique_constraint")
        if duplicate_grains:
            semantic_errors.append("CURRENT_BALANCE:duplicate_grain_rows")

        expected_capabilities = _expected_capabilities_from_evidence(evidence)
        capability_rows: dict[str, tuple[str, str, str, str]] = {}
        if (
            objects.get("wms_capability_status") == "table"
            and not invalid_columns.get("wms_capability_status")
        ):
            capability_rows = {
                str(row[0]): (
                    str(row[1]),
                    str(row[2]),
                    str(row[3]),
                    str(row[4]),
                )
                for row in connection.execute(
                    """
                    SELECT capability, status, reason_code,
                           evidence_basis, contract_version
                    FROM wms_capability_status
                    """
                ).fetchall()
            }
        invalid_capabilities: list[str] = []
        for capability, expected in expected_capabilities.items():
            row = capability_rows.get(capability)
            if row is None:
                invalid_capabilities.append(capability)
                continue
            status, reason_code, evidence_basis, contract_version = row
            if (
                (status, reason_code) != expected
                or not evidence_basis
                or contract_version != SCHEMA_VERSION
            ):
                invalid_capabilities.append(capability)

        metadata_invalid = (
            metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("data_contract_version") != DATA_CONTRACT_VERSION
            or metadata.get("semantic_contract_version")
            != SEMANTIC_CONTRACT_VERSION
            or metadata.get("semantic_epoch") != CURRENT_SEMANTIC_EPOCH
            or metadata.get("source_schema") != SOURCE_SCHEMA
            or metadata.get("plant_id") != PLANT_ID
        )
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        integrity_ok = bool(integrity and integrity[0] == "ok")
        compatible = not (
            invalid_objects
            or forbidden_objects
            or invalid_columns
            or missing_metadata
            or invalid_datasets
            or semantic_errors
            or invalid_capabilities
            or metadata_invalid
            or foreign_key_errors
            or not integrity_ok
        )
        return {
            "compatible": compatible,
            "schema_version": metadata.get("schema_version", ""),
            "data_contract_version": metadata.get("data_contract_version", ""),
            "semantic_contract_version": metadata.get(
                "semantic_contract_version", ""
            ),
            "source_schema": metadata.get("source_schema", ""),
            "plant_id": metadata.get("plant_id", ""),
            "invalid_objects": invalid_objects,
            "forbidden_objects": forbidden_objects,
            "invalid_columns": invalid_columns,
            "missing_metadata": missing_metadata,
            "invalid_datasets": invalid_datasets,
            "semantic_errors": semantic_errors,
            "invalid_capabilities": invalid_capabilities,
            "foreign_key_errors": len(foreign_key_errors),
            "integrity_ok": integrity_ok,
            "reason_codes": (
                () if compatible else (REASON_SNAPSHOT_INCOMPATIBLE,)
            ),
        }
    except (sqlite3.Error, IndexError, TypeError, ValueError):
        return {
            "compatible": False,
            "schema_version": "",
            "data_contract_version": "",
            "semantic_contract_version": "",
            "source_schema": "",
            "plant_id": "",
            "invalid_objects": list(REQUIRED_OBJECT_TYPES),
            "forbidden_objects": [],
            "invalid_columns": {},
            "missing_metadata": sorted(REQUIRED_METADATA_KEYS),
            "invalid_datasets": sorted(EXPECTED_DATASETS),
            "semantic_errors": ["validation_error"],
            "invalid_capabilities": list(BASE_CAPABILITY_STATUSES),
            "foreign_key_errors": 0,
            "integrity_ok": False,
            "reason_codes": (REASON_SNAPSHOT_INCOMPATIBLE,),
        }
