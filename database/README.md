# MES snapshot database

Database SQLite `data/mes.sqlite` được tạo từ ba bản dump trong
`database/raw/`:

- `M_LOT_*.sql`: thông tin Lot và sản phẩm.
- `D_ERROR_*.sql`: các lần ghi nhận lỗi.
- `P_ERROR_*.sql`: danh mục mã và tên lỗi đa ngôn ngữ.

Các file raw và database sinh ra đều bị Git bỏ qua vì chứa dữ liệu vận hành.
Schema và importer được commit để có thể tái tạo database:

```bash
python scripts/import_mes_database.py
```

Importer chọn file mới nhất của từng loại theo tên, tạo database tạm, kiểm tra
khóa ngoại và tính toàn vẹn rồi mới thay thế `data/mes.sqlite`. Vì vậy chạy lại
lệnh không cộng dồn dữ liệu cũ.

## Thành phần chính

- `lots`: một bản ghi cho mỗi `lot_id`.
- `error_events`: dữ liệu lỗi chi tiết; giữ bản ghi không mapping được Lot.
- `error_catalog`: danh mục tên lỗi; một bản ghi canonical cho mỗi bộ
  `(error_id, process_id, error_type)`.
- `v_error_details`: dữ liệu lỗi đã nối Lot và tên lỗi.
- `v_lot_error_summary`: tổng lỗi theo Lot.
- `v_lot_error_breakdown`: chi tiết lỗi theo Lot, công đoạn và mã lỗi.
- `v_product_error_summary`: tổng lỗi theo mã sản phẩm.

Ví dụ truy vấn Lot có tổng lỗi cao nhất:

```sql
SELECT lot_id, product_id, total_error_qty
FROM v_lot_error_summary
ORDER BY total_error_qty DESC
LIMIT 1;
```
