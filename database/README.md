# MES snapshot database

Database SQLite `data/mes.sqlite` được tạo từ các bản dump MES MKAC trong
`database/raw_mkac/`:

- `M_LOT_*.sql`: thông tin Lot và sản phẩm (bắt buộc).
- `D_ERROR_*.sql`: các lần ghi nhận lỗi (bắt buộc).
- `P_ERROR_*.sql`: danh mục mã và tên lỗi đa ngôn ngữ (bắt buộc).
- `D_MAIN_*.sql`: lịch sử công đoạn theo Lot (tùy chọn để tương thích các bộ
  dump cũ chỉ có ba file).

Các file raw MKAC và database sinh ra đều bị Git bỏ qua vì chứa dữ liệu vận hành.
Schema và importer được commit để có thể tái tạo database:

```bash
scripts/meibook-python scripts/import_mes_database.py
```

Importer chọn file mới nhất của từng loại theo tên, tạo database tạm, kiểm tra
khóa ngoại và tính toàn vẹn rồi mới thay thế `data/mes.sqlite`. Vì vậy chạy lại
lệnh không cộng dồn dữ liệu cũ.

Importer chỉ nhận dump có schema prefix `MES_DATA.*`; không dùng lại bộ dump cũ.

## Thành phần chính

- `lots`: một bản ghi cho mỗi `lot_id`.
- `error_events`: dữ liệu lỗi chi tiết; giữ bản ghi không mapping được Lot.
- `error_catalog`: danh mục tên lỗi; một bản ghi canonical cho mỗi bộ
  `(error_id, process_id, error_type)`.
- `process_steps`: các công đoạn D_MAIN đã ghi nhận theo Lot; dùng soft FK tới
  `lots`. Bảng chuẩn hóa không lưu `USER_ID`, `STAFF_ID`, `STAFF_NAME`, `NOTE`.
- `v_error_details`: dữ liệu lỗi đã nối Lot và tên lỗi.
- `v_lot_error_summary`: tổng lỗi theo Lot.
- `v_lot_error_breakdown`: chi tiết lỗi theo Lot, công đoạn và mã lỗi.
- `v_product_error_summary`: tổng lỗi theo mã sản phẩm.
- `v_lot_process_steps`: lịch sử bước theo Lot đã loại PII; các bộ đếm âm từ
  nguồn được hiển thị là `NULL` vì là giá trị chưa rõ/sentinel.
- `v_lot_process_progress`: bước có `process_order` lớn nhất đã ghi nhận và tổng
  số bước của mỗi Lot.

Các view công đoạn chỉ phản ánh dữ liệu đã ghi nhận, không phải kế hoạch hoặc
trạng thái thời gian thực. Không gộp các bộ đếm P/S/B với `total_error_qty`,
không tự giải nghĩa mã process/status, và không tự tính yield rate khi chưa có
quy tắc nghiệp vụ.

Ví dụ truy vấn Lot có tổng lỗi cao nhất:

```sql
SELECT lot_id, product_id, total_error_qty
FROM v_lot_error_summary
ORDER BY total_error_qty DESC
LIMIT 1;
```

## MKHC process-warehouse snapshot — contract v4 / Phase 2C

Dữ liệu WMS nằm riêng trong `data/mes_wms.sqlite`; không nhập vào `mes.sqlite` và
không copy runtime data giữa Dev/Production. Schema v4 tách ba miền có grain và
freshness độc lập:

- `CURRENT_BALANCE`: nguồn bắt buộc `PW_CURRENT_ITEM`, authoritative theo
  `(process_id, item_code)`. Lot current không còn ý nghĩa nghiệp vụ.
- `LEGACY_ARCHIVE`: nguồn optional `PW_SNAPSHORT`, exact-key
  `(item_code, item_lot_id, process_id)` thuộc semantic epoch cũ.
- `RAW_TRANSACTION_AUDIT`: các nguồn optional `PW_TRANSACTION`,
  `PW_TRANSACTION_DEFINE`, `PW_TRANS_DETAIL`; chỉ dùng raw audit, không gọi là
  completed movement và không suy diễn direction/net/delta.

`PW_PROCESS` cũng optional và chỉ enrich theo exact `process_id`. Export không có
INSERT cho nguồn optional được ghi là `NOT_OBSERVED_IN_EXPORT`; điều này không
khẳng định bảng Oracle không tồn tại. Riêng raw audit, nếu definition/detail được
quan sát nhưng header `PW_TRANSACTION` không có trong export, evidence dùng
`PARTIAL_SOURCE_OBSERVED` và query bị `SUPPRESSED` với
`RAW_TRANSACTION_HEADER_NOT_OBSERVED` thay vì kết luận audit không có bản ghi.

Schema được commit tại `database/schema/mes_wms.sql`; raw export và SQLite sinh
ra không được commit. Importer chỉ parse INSERT của allowlist, loại các field
user/note/attachment/account, chạy FK/integrity/contract validation rồi atomic
replace. Duplicate current grain gây lỗi `CURRENT_BALANCE_GRAIN_DUPLICATE` trước
khi target bị thay đổi; importer không cộng hoặc chọn một row để che duplicate.

Chạy dry-run trên nguồn đã stage trong checkout Dev, không tham chiếu checkout
Production:

```bash
scripts/meibook-python scripts/import_mes_wms.py \
  --source /home/jkl/Code/VLLM-PD-dev/data/staging/mes_wms_export.sql \
  --schema database/schema/mes_wms.sql \
  --dry-run \
  --report-json -
```

Sau khi report pass, import/validate Dev snapshot:

```bash
scripts/meibook-python scripts/import_mes_wms.py \
  --source /home/jkl/Code/VLLM-PD-dev/data/staging/mes_wms_export.sql \
  --schema database/schema/mes_wms.sql \
  --db /home/jkl/Code/VLLM-PD-dev/data/mes_wms.sqlite
scripts/meibook-python scripts/import_mes_wms.py \
  --validate-snapshot /home/jkl/Code/VLLM-PD-dev/data/mes_wms.sqlite \
  --report-json -
```

### Contract và capability v4

- Snapshot phải có `schema_version=4`, data/semantic contract version, semantic
  epoch, evidence cho đủ ba miền và capability rows phù hợp evidence.
- Current query chỉ đọc quantity hợp lệ từ
  `v_wms_current_balance_by_process_item`; current lot lookup luôn
  `SUPPRESSED` với `CURRENT_GRAIN_HAS_NO_MEANINGFUL_LOT`.
- Legacy archive hỗ trợ exact-key, khoảng ngày ISO `YYYY-MM-DD` và pagination
  (`trang/page`, `page size`, tối đa 50). Freshness lấy từ
  `MAX(PW_SNAPSHORT.SNAPSHORT_DATE)`.
- Cross-era presence chỉ trả diagnostic: archive present/not-present, current là
  `NOT_EVALUATED`, `comparison_eligible=false`, reason
  `CROSS_ERA_KEYS_NOT_COMPARABLE` và freshness của cả hai miền.
- Raw transaction audit hiển thị mã/trạng thái/date/quantity thô theo allowlist;
  không filter completed và không kết luận nhập/xuất, completed, net hoặc delta.
- Current freshness lấy từ `MAX(PW_CURRENT_ITEM.TIME_UPDATE)`, audit freshness từ
  `MAX(PW_TRANSACTION.TRANS_DATE)`; timezone vẫn `unverified`, không gọi realtime.
- UOM/cross-item totals, min-stock, expiry/window-time, trend/delta, completed
  movement, WIP và bottleneck tiếp tục `SUPPRESSED` đến khi có source/business
  contract được xác nhận.
- `QTY` rỗng, phi số hoặc âm bị quarantine, không đổi thành `0`. Process mapping
  chỉ dùng exact `process_id`, không fuzzy-map và không join `FE_MATERIAL_LIST`.
- `trans_name` mojibake được ẩn bằng nhãn an toàn; không đưa raw label hỏng vào
  câu trả lời.

JSON report chỉ chứa version/status/reason/count/as-of aggregate, không chứa raw
row, sample identifier, PII, secret hoặc path nguồn nội bộ.

Runtime Dev bật bằng:

```dotenv
MES_WMS_DATABASE_ENABLED=true
MES_WMS_DATABASE_PATH=/app/data/mes_wms.sqlite
```

Production phải giữ disabled cho đến khi code/config/schema được review riêng và
Production có snapshot được import từ nguồn Production theo lifecycle dữ liệu.
