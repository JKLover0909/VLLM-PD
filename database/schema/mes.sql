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

CREATE TABLE lots (
    lot_pk INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL UNIQUE,
    create_date TEXT,
    edit_date TEXT,
    product_id TEXT NOT NULL,
    lot_id TEXT NOT NULL UNIQUE,
    pt_id TEXT,
    pt_version_id TEXT,
    route_id TEXT,
    lot_type TEXT,
    status TEXT,
    is_release TEXT,
    sale_order_id TEXT,
    board_lot INTEGER,
    sheet_lot INTEGER,
    prev_status TEXT,
    date_code TEXT,
    produce_date TEXT,
    produce_date_process_id TEXT,
    produce_date_process_order INTEGER,
    is_release_split_lot TEXT,
    pcs_lot INTEGER,
    create_time_unix INTEGER,
    release_date_unix INTEGER,
    release_date TEXT,
    production_type TEXT,
    user_id TEXT,
    prev_release TEXT,
    production_period_type TEXT,
    user_id_update TEXT,
    time_update_unix INTEGER,
    time_update TEXT
);

CREATE TABLE error_catalog (
    error_catalog_pk INTEGER PRIMARY KEY,
    source_id INTEGER,
    create_date TEXT,
    edit_date TEXT,
    error_id TEXT NOT NULL,
    error_name TEXT,
    error_type TEXT NOT NULL,
    status TEXT,
    note TEXT,
    deleted TEXT,
    process_id TEXT NOT NULL,
    error_name_vi TEXT,
    error_name_ja TEXT,
    error_name_en TEXT,
    error_name_ch TEXT,
    priority_error TEXT,
    user_id TEXT,
    is_canonical INTEGER NOT NULL DEFAULT 0 CHECK (is_canonical IN (0, 1))
);

CREATE TABLE error_events (
    error_pk INTEGER PRIMARY KEY,
    source_id INTEGER,
    lot_pk INTEGER REFERENCES lots(lot_pk),
    error_catalog_pk INTEGER REFERENCES error_catalog(error_catalog_pk),
    edit_date TEXT,
    create_date TEXT,
    lot_id TEXT NOT NULL,
    route_id TEXT,
    process_id TEXT NOT NULL,
    process_order INTEGER,
    error_type TEXT NOT NULL,
    error_id TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity >= 0),
    user_id TEXT,
    note TEXT,
    error_process_type TEXT,
    lot_id_split TEXT,
    process_id_create TEXT,
    process_order_create INTEGER,
    error_time_unix INTEGER,
    error_time TEXT,
    error_judgement TEXT
);

CREATE TABLE process_steps (
    process_step_pk INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL UNIQUE,
    lot_pk INTEGER REFERENCES lots(lot_pk),
    edit_date TEXT,
    create_date TEXT,
    lot_id TEXT NOT NULL,
    route_id TEXT NOT NULL,
    process_id TEXT NOT NULL,
    process_order INTEGER NOT NULL,
    t1_unix INTEGER,
    t2_unix INTEGER,
    t3_unix INTEGER,
    t4_unix INTEGER,
    p_ok INTEGER,
    p_ng_defect INTEGER,
    p_ng_scrap INTEGER,
    s_ok INTEGER,
    s_ng_defect INTEGER,
    s_ng_scrap INTEGER,
    b_ok INTEGER,
    b_ng_defect INTEGER,
    b_ng_scrap INTEGER,
    t1_date TEXT,
    t2_date TEXT,
    t3_date TEXT,
    t4_date TEXT,
    output_max_b INTEGER,
    output_max_s INTEGER,
    output_max_p INTEGER,
    is_move_step TEXT,
    process_physical_sub TEXT,
    moving_status TEXT
);

CREATE UNIQUE INDEX uq_error_catalog_canonical
    ON error_catalog(error_id, process_id, error_type)
    WHERE is_canonical = 1;
CREATE INDEX idx_lots_product ON lots(product_id);
CREATE INDEX idx_lots_status ON lots(status);
CREATE INDEX idx_lots_produce_date ON lots(produce_date);
CREATE INDEX idx_lots_release_date ON lots(release_date);
CREATE INDEX idx_error_catalog_key
    ON error_catalog(error_id, process_id, error_type);
CREATE INDEX idx_error_events_lot_quantity
    ON error_events(lot_pk, quantity);
CREATE INDEX idx_error_events_raw_lot ON error_events(lot_id);
CREATE INDEX idx_error_events_catalog ON error_events(error_catalog_pk);
CREATE INDEX idx_error_events_code
    ON error_events(error_id, process_id, error_type);
CREATE INDEX idx_error_events_process ON error_events(process_id);
CREATE INDEX idx_error_events_time ON error_events(error_time);
CREATE INDEX idx_error_events_unmapped_lot
    ON error_events(lot_id)
    WHERE lot_pk IS NULL;
CREATE INDEX idx_error_events_unmapped_catalog
    ON error_events(error_id, process_id, error_type)
    WHERE error_catalog_pk IS NULL;
CREATE INDEX idx_process_steps_lot ON process_steps(lot_pk, process_order);
CREATE INDEX idx_process_steps_raw_lot
    ON process_steps(lot_id, process_order, process_step_pk);
CREATE INDEX idx_process_steps_process ON process_steps(process_id);
CREATE INDEX idx_process_steps_unmapped_lot
    ON process_steps(lot_id)
    WHERE lot_pk IS NULL;

CREATE VIEW v_error_details AS
SELECT
    e.error_pk,
    e.source_id AS error_source_id,
    e.lot_id,
    l.product_id,
    l.status AS lot_status,
    l.pcs_lot,
    e.route_id,
    e.process_id,
    e.process_order,
    e.error_type,
    e.error_id,
    COALESCE(c.error_name_vi, c.error_name, c.error_name_en) AS error_name,
    c.error_name_vi,
    c.error_name_en,
    e.quantity,
    e.error_time,
    e.error_judgement,
    CASE WHEN e.lot_pk IS NULL THEN 0 ELSE 1 END AS lot_mapped,
    CASE WHEN e.error_catalog_pk IS NULL THEN 0 ELSE 1 END AS error_name_mapped
FROM error_events AS e
LEFT JOIN lots AS l ON l.lot_pk = e.lot_pk
LEFT JOIN error_catalog AS c ON c.error_catalog_pk = e.error_catalog_pk
WHERE LOWER(COALESCE(e.lot_id, '')) NOT LIKE '%test%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '%test%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '%dieuphoi%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE 'windowtime%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '9999-%'
  AND LOWER(COALESCE(l.product_id, '')) NOT IN ('666888', '666999', '223344', '212121');

CREATE VIEW v_lot_error_summary AS
SELECT
    l.lot_pk,
    l.lot_id,
    l.product_id,
    l.status,
    l.pcs_lot,
    l.produce_date,
    COUNT(e.error_pk) AS error_record_count,
    COUNT(DISTINCT e.error_id) AS distinct_error_count,
    COALESCE(SUM(e.quantity), 0) AS total_error_qty,
    COALESCE(SUM(CASE WHEN e.error_catalog_pk IS NULL THEN 1 ELSE 0 END), 0)
        AS unmapped_error_record_count
FROM lots AS l
LEFT JOIN error_events AS e ON e.lot_pk = l.lot_pk
WHERE LOWER(COALESCE(l.lot_id, '')) NOT LIKE '%test%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '%test%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '%dieuphoi%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE 'windowtime%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '9999-%'
  AND LOWER(COALESCE(l.product_id, '')) NOT IN ('666888', '666999', '223344', '212121')
GROUP BY l.lot_pk;

CREATE VIEW v_lot_error_breakdown AS
SELECT
    e.lot_id,
    l.product_id,
    e.process_id,
    e.error_type,
    e.error_id,
    COALESCE(c.error_name_vi, c.error_name, c.error_name_en) AS error_name,
    COUNT(e.error_pk) AS error_record_count,
    SUM(e.quantity) AS total_error_qty
FROM error_events AS e
LEFT JOIN lots AS l ON l.lot_pk = e.lot_pk
LEFT JOIN error_catalog AS c ON c.error_catalog_pk = e.error_catalog_pk
WHERE LOWER(COALESCE(e.lot_id, '')) NOT LIKE '%test%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '%test%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '%dieuphoi%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE 'windowtime%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '9999-%'
  AND LOWER(COALESCE(l.product_id, '')) NOT IN ('666888', '666999', '223344', '212121')
GROUP BY
    e.lot_id,
    l.product_id,
    e.process_id,
    e.error_type,
    e.error_id,
    e.error_catalog_pk;

CREATE VIEW v_product_error_summary AS
SELECT
    l.product_id,
    COUNT(DISTINCT l.lot_pk) AS lot_count,
    COUNT(e.error_pk) AS error_record_count,
    COALESCE(SUM(e.quantity), 0) AS total_error_qty
FROM lots AS l
LEFT JOIN error_events AS e ON e.lot_pk = l.lot_pk
WHERE LOWER(COALESCE(l.lot_id, '')) NOT LIKE '%test%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '%test%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '%dieuphoi%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE 'windowtime%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '9999-%'
  AND LOWER(COALESCE(l.product_id, '')) NOT IN ('666888', '666999', '223344', '212121')
GROUP BY l.product_id;

CREATE VIEW v_lot_process_steps AS
SELECT
    p.process_step_pk,
    p.lot_id,
    l.product_id,
    p.route_id,
    p.process_id,
    p.process_order,
    p.t1_unix,
    p.t2_unix,
    p.t3_unix,
    p.t4_unix,
    p.t1_date,
    p.t2_date,
    p.t3_date,
    p.t4_date,
    CASE WHEN p.p_ok >= 0 THEN p.p_ok END AS p_ok,
    CASE WHEN p.p_ng_defect >= 0 THEN p.p_ng_defect END AS p_ng_defect,
    CASE WHEN p.p_ng_scrap >= 0 THEN p.p_ng_scrap END AS p_ng_scrap,
    CASE WHEN p.s_ok >= 0 THEN p.s_ok END AS s_ok,
    CASE WHEN p.s_ng_defect >= 0 THEN p.s_ng_defect END AS s_ng_defect,
    CASE WHEN p.s_ng_scrap >= 0 THEN p.s_ng_scrap END AS s_ng_scrap,
    CASE WHEN p.b_ok >= 0 THEN p.b_ok END AS b_ok,
    CASE WHEN p.b_ng_defect >= 0 THEN p.b_ng_defect END AS b_ng_defect,
    CASE WHEN p.b_ng_scrap >= 0 THEN p.b_ng_scrap END AS b_ng_scrap,
    CASE WHEN p.output_max_b >= 0 THEN p.output_max_b END AS output_max_b,
    CASE WHEN p.output_max_s >= 0 THEN p.output_max_s END AS output_max_s,
    CASE WHEN p.output_max_p >= 0 THEN p.output_max_p END AS output_max_p,
    p.is_move_step,
    p.moving_status,
    CASE WHEN p.lot_pk IS NULL THEN 0 ELSE 1 END AS lot_mapped
FROM process_steps AS p
LEFT JOIN lots AS l ON l.lot_pk = p.lot_pk
WHERE LOWER(COALESCE(p.lot_id, '')) NOT LIKE '%test%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '%test%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '%dieuphoi%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE 'windowtime%'
  AND LOWER(COALESCE(l.product_id, '')) NOT LIKE '9999-%'
  AND LOWER(COALESCE(l.product_id, '')) NOT IN ('666888', '666999', '223344', '212121');

CREATE VIEW v_lot_process_progress AS
SELECT
    p.lot_id,
    p.product_id,
    p.route_id,
    aggregate.step_count,
    p.process_id AS latest_process_id,
    p.process_order AS latest_process_order,
    COALESCE(p.t4_date, p.t3_date, p.t2_date, p.t1_date) AS latest_recorded_at,
    p.is_move_step,
    p.moving_status,
    p.lot_mapped
FROM v_lot_process_steps AS p
JOIN (
    SELECT
        counted.lot_id,
        COUNT(*) AS step_count
    FROM v_lot_process_steps AS counted
    GROUP BY counted.lot_id
) AS aggregate
  ON aggregate.lot_id = p.lot_id
WHERE p.process_step_pk = (
    SELECT candidate.process_step_pk
    FROM v_lot_process_steps AS candidate
    WHERE candidate.lot_id = p.lot_id
    ORDER BY candidate.process_order DESC, candidate.process_step_pk DESC
    LIMIT 1
);
