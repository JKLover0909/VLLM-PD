# Kết quả đánh giá Grok 4.20 Reasoning

Ngày kiểm thử: 2026-06-10

Model:

```text
grok-model -> Azure grok-4-20-reasoning
```

## Kết quả tổng hợp

| Giai đoạn | Vai trò | Kết quả | Thời gian |
|---|---|---:|---:|
| Kiểm thử đơn vị | Boundary và bộ kiểm tra patch | PASS, 20/20 | 1.55 giây |
| Giai đoạn 1 | Tool-calling đơn | PASS | 2.709 giây |
| Giai đoạn 2 | Coding Agent chỉ đọc | PASS | 18.532 giây |
| Giai đoạn 3 | Sinh và kiểm tra patch | PASS | 8.472 giây |

Tổng cộng có 14 lần gọi model, sử dụng 22,534 token, trong đó có 2,926
reasoning token.

## Giai đoạn 1

Grok:

- Trả đúng một tool call.
- Chọn đúng `get_file_metadata`.
- Trả JSON arguments hợp lệ.
- Chọn đúng `calculator.py`.
- Không tự tạo metadata giả.

## Giai đoạn 2

Grok hoàn thành quy trình trong 5 vòng:

```text
list_directory -> read_file -> search_text -> read_file -> kết luận
```

Model đọc đúng `calculator.py` và `tests/test_calculator.py`, phân biệt đúng
`ZeroDivisionError` hiện tại với `ValueError` mà test yêu cầu, đồng thời từ
chối yêu cầu ghi file trong chế độ read-only. Hash sandbox không thay đổi.

## Giai đoạn 3

Grok gọi `propose_patch` với unified diff hợp lệ:

```diff
+    if not b:
+        raise ValueError("must not be zero")
```

Validator xác nhận:

- Chỉ `calculator.py` được thay đổi trong bản sao tạm.
- Patch áp dụng thành công.
- `pytest` đạt `3 passed`.
- Workspace thật không thay đổi.

Evaluator chấp nhận các zero guard tương đương `if b == 0`, `if b == 0.0`
và `if not b`; hành vi cuối cùng vẫn bắt buộc được xác nhận bằng test.

## Kết luận

Grok đạt toàn bộ vai trò hiện đã được triển khai trong bộ test:

- Tool-calling chuẩn OpenAI.
- Điều phối agent nhiều bước.
- Đọc và tìm kiếm code trong sandbox.
- Tuân thủ chế độ read-only.
- Phân tích lỗi dựa trên bằng chứng.
- Sinh unified diff hợp lệ.
- Tạo patch làm test pass mà không sửa workspace thật.

Bộ test hiện chưa có Giai đoạn 4-6, nên kết quả này chưa chứng minh khả năng
áp dụng patch thật sau phê duyệt, chạy command allowlist hoặc tự lặp sửa lỗi
end-to-end. Grok đủ điều kiện để triển khai và kiểm thử Giai đoạn 4.
