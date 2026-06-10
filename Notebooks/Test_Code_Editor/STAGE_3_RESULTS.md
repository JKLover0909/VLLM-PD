# Giai đoạn 3: Kết quả sinh và validate patch

Ngày kiểm thử: 2026-06-06

## Phần đã triển khai

- `patch_validator.py`: kiểm tra unified diff mà không ghi workspace.
- `tests/test_patch_validator.py`: kiểm thử patch hợp lệ và patch nguy hiểm.
- `03_test_patch_generation.ipynb`: model đọc code/test rồi gọi
  `propose_patch`.

Validator thực hiện:

1. Kiểm tra path tương đối và nằm trong workspace.
2. Chặn `.env`, path traversal, binary và rename.
3. Chỉ cho phép một file có sẵn ở Giai đoạn 3.
4. Yêu cầu path khai báo trùng chính xác với path trong diff.
5. Chạy `git apply --check` trong bản sao tạm.
6. Áp dụng patch trong bản sao tạm.
7. Kiểm tra chỉ đúng file dự kiến bị thay đổi.
8. Chạy `pytest` trong bản sao tạm.
9. So hash workspace thật trước và sau.

Kiểm tra zero guard chấp nhận các biểu thức tương đương `if b == 0`,
`if b == 0.0` và `if not b`; tính đúng đắn cuối cùng vẫn được xác nhận bằng
toàn bộ test trong bản sao tạm.

`git apply --recount` được sử dụng để sửa lỗi đếm số dòng hunk của LLM. Context
code vẫn phải khớp chính xác; validator không tự sửa nội dung patch.

## Unit test

Kết quả chung cho read-only tools và patch validator:

```text
20 passed
```

Các trường hợp patch đã kiểm tra:

- Patch hợp lệ làm test pass trong bản sao tạm.
- Patch malformed.
- Sai format `apply_patch`.
- Path ngoài workspace.
- File nhạy cảm.
- Path khai báo không khớp diff.
- Patch nhiều file.
- Context không áp dụng được.
- Patch hợp lệ về cú pháp nhưng làm test thất bại.

## Kết quả model

| Model | Trạng thái | Kết quả | Thời gian |
|---|---|---:|---:|
| Gemma4 Local | SKIPPED | Không qua gate Giai đoạn 2 | - |
| Xiaomi MiMo | Tested | PASS | 17.333 giây |
| OpenAI alias trước migration | Tested | FAIL | 19.400 giây |

### Xiaomi MiMo

MiMo:

- Đọc đúng `calculator.py`.
- Đọc đúng `tests/test_calculator.py`.
- Gọi `propose_patch`.
- Chỉ sửa `calculator.py`.
- Thêm `if b == 0`.
- Phát sinh `ValueError` với message chứa `must not be zero`.
- Patch áp dụng được trong bản sao tạm.
- Toàn bộ pytest pass trong bản sao tạm.
- Workspace thật không thay đổi.

Patch MiMo:

```diff
diff --git a/calculator.py b/calculator.py
--- a/calculator.py
+++ b/calculator.py
@@ -7,4 +7,6 @@ def add(a: float, b: float) -> float:

 def divide(a: float, b: float) -> float:
     # Intentional Milestone 1 bug: no explicit zero-division validation.
+    if b == 0:
+        raise ValueError("Denominator must not be zero")
     return a / b
```

### OpenAI alias trước migration

OpenAI:

- Đọc đúng cả code và test.
- Gọi đúng `propose_patch`.
- Ý định sửa và code mới là hợp lý.
- Không sinh được unified diff có context áp dụng được.
- Lặp patch không hợp lệ cho đến giới hạn 8 vòng.
- Validator từ chối toàn bộ proposal.
- Workspace thật không thay đổi.

Patch cuối có context thiếu dòng trống so với file thật, nên Git báo:

```text
patch does not apply
```

Đây là kết quả FAIL đúng của model, không phải lỗi validator. Không nên tự sửa
context cho model vì sẽ che giấu chất lượng patch và có thể áp dụng sai vị trí.

Kết quả này được tạo trước khi `openai-model` chuyển sang GPT-5.4 mini. Cần chạy
lại Giai đoạn 1-3 để đánh giá model mới; không được coi FAIL cũ là kết quả của
GPT-5.4 mini.

### Gemma4 Local

Không chạy Giai đoạn 3 tự động vì chưa hoàn thành agent read-only nhiều bước ở
Giai đoạn 2. Quy tắc gate được giữ nguyên.

## Quyết định

- MiMo đủ điều kiện thử Giai đoạn 4: approve/reject và áp dụng patch trong
  sandbox.
- OpenAI chưa đủ điều kiện áp dụng patch tự động. Có thể cải thiện bằng:
  - model coding mạnh hơn;
  - tool chỉnh sửa có cấu trúc `replace_range`;
  - hoặc yêu cầu diff ít context hơn nhưng vẫn được validator kiểm tra.
- Local tiếp tục ở Giai đoạn 2.

Không file nào trong fixture thật đã được sửa.
