# Milestone 1 Results: Tool Calling Compatibility

Ngày kiểm thử: 2026-06-06

> Kết quả OpenAI trong báo cáo này được tạo trước khi alias `openai-model`
> chuyển sang GPT-5.4 mini. Cần chạy lại notebook để có benchmark của model mới.

## Kết quả fixture

Lệnh:

```bash
cd Notebooks/Test_Code_Editor/fixtures/sample_project
python -m pytest -q
```

Kết quả ban đầu:

```text
1 failed, 2 passed
```

Test thất bại đúng chủ đích: `divide(10, 0)` phát sinh
`ZeroDivisionError`, trong khi yêu cầu là `ValueError("b must not be zero")`.

## Kết quả tool-calling

Cả ba model nhận cùng một tool schema `get_file_metadata` và cùng prompt.

| Provider | Model alias | Kết quả | Latency | Tool | Arguments |
|---|---|---:|---:|---|---|
| Local | `local-gemma` | PASS | 6.292 giây | `get_file_metadata` | `{"file_path":"calculator.py"}` |
| MiMo | `mimo-pro` | PASS | 3.465 giây | `get_file_metadata` | `{"file_path":"calculator.py"}` |
| OpenAI | `openai-model` | PASS | 1.891 giây | `get_file_metadata` | `{"file_path":"calculator.py"}` |

Mỗi model đều:

- Trả đúng một `tool_call`.
- Chọn đúng tên tool.
- Trả arguments là JSON hợp lệ.
- Chọn đúng file `calculator.py`.
- Không tự bịa kết quả trước khi tool thực thi.

## Kết luận

Ba model đều đạt kiểm tra tương thích tool-calling ban đầu và có thể chuyển sang
Giai đoạn 2: agent `read_only`.

Kết quả hiện mới là `1/1` cho mỗi model. Chưa cấp quyền ghi file hoặc thực thi
lệnh. Trước khi bật `workspace_write`, cần chạy `RUNS_PER_MODEL = 10` và yêu cầu
ít nhất `9/10` lần đạt, đồng thời hoàn thành các kiểm tra giới hạn workspace.

Kết quả máy đọc được nằm trong:

```text
results/local.json
results/mimo.json
results/openai.json
results/summary.json
```
