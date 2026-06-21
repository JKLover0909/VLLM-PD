# Thiết kế cơ sở dữ liệu MES

Tài liệu này mô tả database MES cục bộ được tạo từ các bản dump ngày
**20/06/2026**. Đây là snapshot phục vụ demo và phân tích, không thay thế dữ
liệu MES thời gian thực.

## 1. Mục tiêu

Database hợp nhất ba nhóm dữ liệu:

- Thông tin Lot và mã sản phẩm.
- Các lần ghi nhận lỗi của từng Lot.
- Danh mục mã lỗi và tên lỗi đa ngôn ngữ.

Các nhóm câu hỏi mục tiêu gồm: Lot có nhiều lỗi nhất, thông tin một Lot, danh
sách lỗi của Lot, tên của mã lỗi, và thống kê lỗi theo sản phẩm.

## 2. Vị trí

```text
database/
├── raw/                         # Dump gốc, không commit Git
│   ├── M_LOT_202606200852.sql
│   ├── D_ERROR_202606200951.sql
│   └── P_ERROR_202606201019.sql
├── schema/mes.sql               # Schema, index và view
└── README.md

scripts/import_mes_database.py   # Importer
data/mes.sqlite                  # Database sinh ra, không commit Git
```

`database/raw/` và `data/` đã được `.gitignore` loại khỏi Git.

## 3. Dữ liệu nguồn

### `M_LOT`

Chứa 566 Lot. `LOT_ID` là duy nhất trong snapshot hiện tại. Các trường chính:

| Trường | Ý nghĩa |
|---|---|
| `ID` | ID nguồn |
| `LOT_ID` | Mã Lot |
| `PRODUCT_ID` | Mã sản phẩm |
| `ROUTE_ID` | Tuyến sản xuất |
| `STATUS` | Trạng thái Lot |
| `PCS_LOT` | Tổng số PCS |
| `PRODUCE_DATE` | Ngày sản xuất |
| `RELEASE_DATE` | Ngày release |

### `D_ERROR`

Chứa 8.822 lần ghi nhận lỗi. `ID` nguồn có thể rỗng hoặc trùng nên không được
dùng làm khóa chính nội bộ.

| Trường | Ý nghĩa |
|---|---|
| `LOT_ID` | Lot phát sinh lỗi |
| `PROCESS_ID` | Công đoạn phát hiện lỗi |
| `ERROR_TYPE` | Loại lỗi |
| `ERROR_ID` | Mã lỗi |
| `QTY` | Số lượng lỗi |
| `ERROR_TIME` | Thời gian ghi nhận |
| `ERROR_JUDGEMENT` | Kết quả đánh giá |

### `P_ERROR`

Chứa 3.365 bản ghi danh mục lỗi. Các tên lỗi được lưu ở `ERROR_NAME`,
`ERROR_NAME_VI`, `ERROR_NAME_JA`, `ERROR_NAME_EN` và `ERROR_NAME_CH`.

Khóa mapping đầy đủ là:

```text
ERROR_ID + PROCESS_ID + ERROR_TYPE
```

Không nên nối chỉ bằng `ERROR_ID`, vì cùng mã có thể xuất hiện ở nhiều công
đoạn hoặc loại lỗi.

## 4. Schema chuẩn hóa

### `lots`

- `lot_pk`: khóa chính nội bộ.
- `source_id`: ID từ MES.
- `lot_id`: khóa nghiệp vụ duy nhất.
- Giữ thông tin sản phẩm, trạng thái, tuyến, số lượng và thời gian.

### `error_catalog`

- `error_catalog_pk`: khóa chính nội bộ.
- Giữ toàn bộ danh mục từ nguồn.
- `is_canonical=1` đánh dấu bản ghi dùng để mapping.
- Chỉ có một bản ghi canonical cho mỗi bộ
  `(error_id, process_id, error_type)`.

### `error_events`

- `error_pk`: khóa chính nội bộ.
- `source_id`: ID nguồn, không bắt buộc duy nhất.
- `lot_pk`: khóa ngoại đến `lots`, có thể rỗng nếu Lot không có trong snapshot.
- `error_catalog_pk`: khóa ngoại đến danh mục canonical, có thể rỗng nếu chưa
  mapping chính xác.
- Các mã nguồn vẫn được giữ để truy vết khi khóa ngoại bị rỗng.

### Metadata

- `import_batches`: file nguồn, SHA-256, kích thước, số dòng và thời điểm import.
- `schema_metadata`: phiên bản schema và chỉ số chất lượng dữ liệu.

## 5. Quan hệ

```text
lots (1) ---------------------- (N) error_events
error_catalog (1) ------------- (N) error_events
```

Quan hệ từ dữ liệu nguồn:

```text
M_LOT.LOT_ID = D_ERROR.LOT_ID

D_ERROR.ERROR_ID    = P_ERROR.ERROR_ID
D_ERROR.PROCESS_ID  = P_ERROR.PROCESS_ID
D_ERROR.ERROR_TYPE  = P_ERROR.ERROR_TYPE
```

## 6. View truy vấn

| View | Mục đích |
|---|---|
| `v_error_details` | Lỗi chi tiết đã nối Lot, sản phẩm và tên lỗi |
| `v_lot_error_summary` | Tổng số lượng lỗi theo Lot |
| `v_lot_error_breakdown` | Phân rã lỗi theo Lot, công đoạn và mã lỗi |
| `v_product_error_summary` | Tổng hợp lỗi theo mã sản phẩm |

`v_error_details` có cờ `lot_mapped` và `error_name_mapped` để phân biệt dữ
liệu đã mapping và chưa mapping.

## 7. Index

Các index chính bao phủ:

- Sản phẩm, trạng thái và ngày của Lot.
- `lot_pk`, `lot_id` và `error_catalog_pk` của lỗi.
- Bộ `(error_id, process_id, error_type)`.
- Công đoạn, thời gian lỗi và các bản ghi chưa mapping.

Truy vấn tổng lỗi theo Lot sử dụng covering index
`idx_error_events_lot_quantity`.

## 8. Chất lượng dữ liệu

| Chỉ số | Giá trị |
|---|---:|
| Lot | 566 |
| Bản ghi lỗi | 8.822 |
| Bản ghi danh mục lỗi | 3.365 |
| Lỗi không nối được Lot | 2 |
| Lỗi chưa mapping chính xác tên | 936 |

Giới hạn hiện tại:

1. Database là snapshot, không tự cập nhật theo MES nguồn.
2. Chưa có bảng giải nghĩa đầy đủ `STATUS`, `LOT_TYPE` và `ERROR_TYPE`.
3. Model không được tự đặt tên cho lỗi chưa mapping.
4. Dữ liệu có Lot và sản phẩm test nhưng chưa có quy tắc nghiệp vụ để loại.
5. Lot test `000316-02-000`, sản phẩm `Test_1710`, có tổng 52.300 lỗi và đứng
   đầu toàn snapshot. Lot `000432-01-000`, sản phẩm `3736-0008`, có 15.920 lỗi.
6. Kết quả snapshot có thể khác API MES thời gian thực; câu trả lời phải nói rõ
   nguồn và thời điểm dữ liệu.

## 9. Đưa dữ liệu vào hệ thống hỏi đáp

Trạng thái hiện tại: phương án dưới đây đã được triển khai trong
`src/integrations/mes_database.py` và tích hợp vào `RAGPipeline`. Frontend nhận
`answer_scope=mes_database` và hiển thị nhãn `MES snapshot`. Router MES chỉ hoạt
động trong mode `mes`; mode `mkac` được dành riêng cho hành chính, nhân sự và
tài liệu nội bộ MKAC.

LLM không tự nhìn thấy SQLite. Backend phải truy vấn database rồi chỉ đưa kết
quả có cấu trúc vào prompt:

```text
Câu hỏi
  -> router nhận diện intent MES
  -> truy vấn chỉ đọc đã kiểm soát
  -> JSON kết quả và metadata snapshot
  -> LLM diễn đạt tự nhiên
```

### Data dictionary cho model

Model cần được cung cấp ngắn gọn các định nghĩa:

- `lot_id`: mã Lot.
- `product_id`: mã sản phẩm.
- `quantity`: số lượng của một bản ghi lỗi.
- `total_error_qty`: tổng `quantity`, không phải số loại lỗi.
- `error_record_count`: số lần ghi nhận, không phải tổng số lượng lỗi.
- `error_id`: mã lỗi.
- `error_name`: tên lỗi, ưu tiên tiếng Việt.
- `process_id`: công đoạn phát hiện lỗi.
- Tên lỗi rỗng nghĩa là chưa mapping, không được suy đoán.
- Snapshot có thể chứa dữ liệu test và có thể khác API thời gian thực.

Kết quả gửi cho model nên có metadata dạng:

```json
{
  "source": "mes_snapshot",
  "snapshot_imported_at": "2026-06-20T03:52:08.714894+00:00",
  "filters": {"exclude_test_data": false},
  "rows": []
}
```

### Intent đang hỗ trợ

| Intent | Nguồn truy vấn |
|---|---|
| Lot có tổng lỗi cao nhất | `v_lot_error_summary` |
| Thông tin một Lot | `lots` và `v_lot_error_summary` |
| Danh sách lỗi của Lot | `v_lot_error_breakdown` |
| Tên của mã lỗi | `error_catalog` canonical |
| Tổng lỗi theo sản phẩm | `v_product_error_summary` |
| Lỗi phổ biến của sản phẩm | `v_error_details` |
| Các Lot có một mã lỗi | `error_events` + `lots` |

Mỗi intent nên ánh xạ đến SQL tham số hóa cố định. Không nối nội dung câu hỏi
trực tiếp vào SQL.

### Router khuyến nghị

Giai đoạn demo đang dùng phương án kết hợp:

1. Quy tắc chắc chắn cho câu phổ biến như “Lot nào lỗi nhiều nhất?”.
2. Query service read-only chọn truy vấn tham số hóa trong allowlist.
3. SQL Agent schema-aware xử lý câu phức hợp chưa có intent cố định.
4. LLM cuối chỉ diễn đạt dữ liệu đã được backend truy vấn và kiểm chứng.

Response nên dùng `answer_scope=mes_database` để phân biệt với `mes` của API
thời gian thực, `mkac` của Qdrant và `web`.

## 10. Có cần tạo skill không?

**Chưa cần skill trong giai đoạn hiện tại.** Skill không tự cấp quyền đọc
SQLite. Thành phần bắt buộc vẫn là backend query database an toàn.

Phương án phù hợp hiện tại:

```text
router MES + query service read-only + SQL Agent có validate + LLM diễn đạt
```

Skill chỉ đáng tạo khi có nhiều bảng, nhiều nhóm truy vấn và cần đóng gói lâu
dài data dictionary, quy tắc chọn API hay snapshot, quy tắc loại dữ liệu test,
giải nghĩa mã nghiệp vụ và ví dụ câu hỏi. Ngay cả khi có skill, việc đọc dữ liệu
vẫn phải đi qua tool hoặc API backend.

## 11. SQL Agent hiện tại

SQL Agent đã được thêm để xử lý các câu hỏi phức hợp mà allowlist cố định chưa
bao phủ, ví dụ:

```text
Trong Lot có số lượng lỗi nhiều nhất thì 3 loại lỗi gây lỗi nhiều nhất là gì?
```

Thiết kế hiện tại:

- Semantic model nằm ở `config/mes_semantic_model.json`.
- Model chỉ nhìn thấy các view công khai:
  - `v_lot_error_summary`
  - `v_lot_error_breakdown`
  - `v_product_error_summary`
  - `v_error_details`
- Model phải trả kế hoạch JSON có một câu `SELECT` hoặc `WITH ... SELECT`.
- Backend validate bằng SQLGlot trước khi chạy.
- SQLite được mở read-only bằng `mode=ro`, `PRAGMA query_only=ON` và authorizer.
- SQL bị chặn nếu đụng bảng raw, ghi dữ liệu, DDL, `ATTACH`, `PRAGMA` hoặc nhiều
  statement.
- Backend ép `LIMIT`, timeout và giới hạn số dòng trả về.
- Kết quả SQL được đưa lại cho LLM để diễn đạt tự nhiên.
- Nếu LLM bỏ sót trường bắt buộc hoặc trả JSON/SQL thô, backend dùng fallback
  deterministic từ kết quả đã query.

Vì vậy, LLM có thể suy luận query từ cấu trúc database đã nạp sẵn, nhưng vẫn
không được chạy SQL tùy ý. Mọi truy vấn đều đi qua lớp validate và chỉ đọc các
view công khai.
