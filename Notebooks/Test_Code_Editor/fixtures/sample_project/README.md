# Bộ mẫu máy tính

Dự án sandbox này được dùng để kiểm thử Coding Agent. Không dùng mã nguồn chính
của repo cho các bài test ghi file đầu tiên.

## Trạng thái ban đầu

Hàm `divide()` chưa kiểm tra mẫu số bằng `0`. Test
`test_divide_rejects_zero_denominator` yêu cầu hàm chủ động phát sinh:

```python
ValueError("b must not be zero")
```

Do đó bộ test phải có đúng một test thất bại trước khi agent sửa code.

## Chạy test

Từ thư mục này:

```bash
python -m pytest -q
```

Kết quả ban đầu mong đợi:

```text
2 passed, 1 failed
```

## Phạm vi

- Agent chỉ được thay đổi file trong dự án sandbox này khi test chỉnh sửa.
- Mốc 1 chỉ kiểm tra tool-calling giả, chưa đọc hoặc ghi các file này.
- Giữ lỗi chủ đích cho đến giai đoạn sửa code có phê duyệt.
