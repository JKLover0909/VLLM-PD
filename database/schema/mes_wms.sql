PRAGMA foreign_keys = ON;

CREATE TABLE schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE import_batches (
    source_name TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    source_size_bytes INTEGER NOT NULL,
    row_count INTEGER NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE wms_dataset_evidence (
    dataset TEXT PRIMARY KEY CHECK (
        dataset IN ('CURRENT_BALANCE', 'LEGACY_ARCHIVE', 'RAW_TRANSACTION_AUDIT')
    ),
    status TEXT NOT NULL CHECK (status IN ('AVAILABLE', 'PARTIAL', 'SUPPRESSED')),
    reason_code TEXT NOT NULL DEFAULT '',
    source_tables TEXT NOT NULL,
    source_state TEXT NOT NULL CHECK (
        source_state IN (
            'PRESENT_NONEMPTY',
            'PARTIAL_SOURCE_OBSERVED',
            'NOT_OBSERVED_IN_EXPORT'
        )
    ),
    candidate_row_count INTEGER NOT NULL CHECK (candidate_row_count >= 0),
    inserted_row_count INTEGER NOT NULL CHECK (inserted_row_count >= 0),
    invalid_quantity_row_count INTEGER NOT NULL CHECK (
        invalid_quantity_row_count >= 0
    ),
    source_as_of TEXT NOT NULL DEFAULT '',
    source_as_of_state TEXT NOT NULL CHECK (
        source_as_of_state IN ('DERIVED_UNVERIFIED', 'UNAVAILABLE')
    ),
    source_as_of_basis TEXT NOT NULL,
    source_timezone TEXT NOT NULL,
    semantic_epoch TEXT NOT NULL,
    evidence_basis TEXT NOT NULL
);

CREATE TABLE wms_capability_status (
    capability TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('AVAILABLE', 'PARTIAL', 'SUPPRESSED')),
    reason_code TEXT NOT NULL DEFAULT '',
    evidence_basis TEXT NOT NULL,
    contract_version TEXT NOT NULL
);

CREATE TABLE wms_processes (
    process_pk INTEGER PRIMARY KEY,
    source_id INTEGER,
    create_date TEXT,
    edit_date TEXT,
    process_id TEXT NOT NULL UNIQUE,
    process_name TEXT,
    process_physical_id TEXT,
    status TEXT,
    is_check_material TEXT
);

CREATE TABLE wms_current_balances (
    balance_pk INTEGER PRIMARY KEY,
    source_id INTEGER,
    create_date TEXT,
    edit_date TEXT,
    item_code TEXT NOT NULL,
    quantity_decimal TEXT,
    quantity_valid INTEGER NOT NULL CHECK (quantity_valid IN (0, 1)),
    quantity_error TEXT,
    time_update TEXT,
    time_update_unix INTEGER,
    trans_id TEXT,
    process_id TEXT NOT NULL,
    process_pk INTEGER REFERENCES wms_processes(process_pk),
    UNIQUE (process_id, item_code)
);

CREATE TABLE wms_legacy_archive_records (
    archive_record_pk INTEGER PRIMARY KEY,
    source_id INTEGER,
    create_date TEXT,
    edit_date TEXT,
    archive_id TEXT NOT NULL,
    archive_date TEXT NOT NULL,
    item_code TEXT NOT NULL,
    item_lot_id TEXT NOT NULL,
    process_id TEXT NOT NULL,
    process_pk INTEGER REFERENCES wms_processes(process_pk),
    quantity_decimal TEXT,
    quantity_valid INTEGER NOT NULL CHECK (quantity_valid IN (0, 1)),
    quantity_error TEXT
);

CREATE TABLE wms_raw_transaction_definitions (
    trans_code TEXT PRIMARY KEY,
    trans_name TEXT
);

CREATE TABLE wms_raw_transaction_headers (
    transaction_pk INTEGER PRIMARY KEY,
    source_id INTEGER,
    create_date TEXT,
    edit_date TEXT,
    trans_id TEXT NOT NULL UNIQUE,
    trans_code TEXT NOT NULL,
    trans_date TEXT,
    trans_date_unix INTEGER,
    process_id TEXT,
    item_code TEXT,
    quantity_decimal TEXT,
    quantity_valid INTEGER NOT NULL CHECK (quantity_valid IN (0, 1)),
    quantity_error TEXT,
    trans_status TEXT,
    deleted TEXT
);

CREATE TABLE wms_raw_transaction_details (
    transaction_detail_pk INTEGER PRIMARY KEY,
    source_id INTEGER,
    create_date TEXT,
    edit_date TEXT,
    trans_id TEXT NOT NULL,
    item_lot_id TEXT,
    quantity_decimal TEXT,
    quantity_valid INTEGER NOT NULL CHECK (quantity_valid IN (0, 1)),
    quantity_error TEXT,
    product_id TEXT,
    production_lot_id TEXT
);

CREATE INDEX idx_wms_current_process_item
    ON wms_current_balances(process_id, item_code);
CREATE INDEX idx_wms_current_item_process
    ON wms_current_balances(item_code, process_id);
CREATE INDEX idx_wms_current_update
    ON wms_current_balances(time_update);
CREATE INDEX idx_wms_current_unmapped
    ON wms_current_balances(process_id)
    WHERE process_pk IS NULL;
CREATE INDEX idx_wms_legacy_exact_key_date
    ON wms_legacy_archive_records(
        item_code, item_lot_id, process_id, archive_date, source_id
    );
CREATE INDEX idx_wms_legacy_archive_id
    ON wms_legacy_archive_records(archive_id, archive_date);
CREATE INDEX idx_wms_raw_transactions_process_item_date
    ON wms_raw_transaction_headers(process_id, item_code, trans_date);
CREATE INDEX idx_wms_raw_transaction_details_trans
    ON wms_raw_transaction_details(trans_id);

CREATE VIEW v_wms_current_balance_by_process_item AS
SELECT
    c.source_id,
    c.process_id,
    p.process_name,
    p.process_physical_id,
    p.status AS process_status,
    CASE WHEN c.process_pk IS NULL THEN 0 ELSE 1 END AS process_mapped,
    c.item_code,
    c.quantity_decimal,
    c.quantity_valid,
    c.quantity_error,
    c.time_update AS latest_update,
    c.time_update_unix AS latest_update_unix,
    c.trans_id
FROM wms_current_balances AS c
LEFT JOIN wms_processes AS p ON p.process_pk = c.process_pk
WHERE c.quantity_valid = 1;

CREATE VIEW v_wms_legacy_archive_exact_key AS
SELECT
    a.source_id,
    a.archive_id,
    a.archive_date,
    a.item_code,
    a.item_lot_id,
    a.process_id,
    p.process_name,
    a.quantity_decimal,
    a.quantity_valid,
    a.quantity_error
FROM wms_legacy_archive_records AS a
LEFT JOIN wms_processes AS p ON p.process_pk = a.process_pk;

CREATE VIEW v_wms_raw_transaction_audit AS
SELECT
    h.trans_id,
    h.trans_code,
    d.trans_name,
    h.trans_date,
    h.trans_date_unix,
    h.process_id,
    p.process_name,
    h.item_code,
    td.item_lot_id,
    h.quantity_decimal AS header_quantity_decimal,
    h.quantity_valid AS header_quantity_valid,
    h.trans_status AS raw_trans_status,
    h.deleted AS raw_deleted,
    td.quantity_decimal AS detail_quantity_decimal,
    td.quantity_valid AS detail_quantity_valid
FROM wms_raw_transaction_headers AS h
LEFT JOIN wms_raw_transaction_definitions AS d
    ON d.trans_code = h.trans_code
LEFT JOIN wms_raw_transaction_details AS td
    ON td.trans_id = h.trans_id
LEFT JOIN wms_processes AS p
    ON p.process_id = h.process_id;

CREATE VIEW v_wms_current_quality AS
SELECT
    COUNT(*) AS current_row_count,
    COALESCE(SUM(CASE WHEN quantity_valid = 1 THEN 1 ELSE 0 END), 0)
        AS valid_quantity_row_count,
    COALESCE(SUM(CASE WHEN quantity_valid = 0 THEN 1 ELSE 0 END), 0)
        AS invalid_quantity_row_count,
    COALESCE(SUM(CASE WHEN process_pk IS NOT NULL THEN 1 ELSE 0 END), 0)
        AS mapped_process_row_count,
    COALESCE(SUM(CASE WHEN process_pk IS NULL THEN 1 ELSE 0 END), 0)
        AS unmapped_process_row_count,
    COUNT(DISTINCT item_code) AS distinct_item_count,
    COUNT(DISTINCT process_id) AS distinct_process_code_count,
    MAX(time_update) AS source_as_of
FROM wms_current_balances;
