# Lộ trình xây dựng AI Coding Agent với Local và Cloud APIs

## 1. Mục tiêu

Biến ba backend mô hình hiện có thành Coding Agent có thể:

1. Đọc và tìm kiếm mã nguồn trong workspace.
2. Giải thích code và lập kế hoạch thay đổi.
3. Tạo file hoặc chỉnh sửa file có kiểm soát.
4. Hiển thị diff trước khi áp dụng.
5. Chạy formatter, linter và test bằng các lệnh được cho phép.
6. Kiểm tra kết quả rồi tiếp tục sửa khi cần.
7. Hoạt động từ Web API và sau đó có thể tích hợp vào VS Code.

Ba backend cần kiểm thử:

| Lựa chọn | LiteLLM model | Backend |
|---|---|---|
| `local` | `local-gemma` | Gemma4 trên Máy 1 |
| `mimo` | `mimo-pro` | Xiaomi MiMo 2.5 Pro |
| `openai` | `openai-model` | OpenAI GPT-4o mini |
| `auto` | `coding-model` | Gemma4, fallback OpenAI |

> Lưu ý: khả năng chat tốt không đảm bảo model gọi tool đúng. Mỗi model phải vượt qua bài test tool-calling trước khi được cấp quyền sửa code.

---

## 2. Trạng thái repo hiện tại

Repo đã có:

- `POST /agent`.
- LangGraph thực hiện vòng lặp `model -> tool -> model`.
- LiteLLM cung cấp model logic `coding-model`.
- MCP Filesystem giới hạn trong `WORKSPACE_DIR`.
- MCP Git giới hạn trong `AGENT_REPOSITORY_DIR`.
- Công cụ fallback: `read_file`, `write_file`, `list_dir`.
- `AGENT_API_KEY` bảo vệ endpoint.

Repo chưa có:

- Chọn model trong request `/agent`.
- Tool tìm kiếm nội dung code.
- Tool sửa theo patch/diff.
- Terminal tool thực sự, dù system prompt đang nhắc đến terminal.
- Phê duyệt trước khi ghi file hoặc chạy lệnh.
- Giới hạn số vòng lặp và số tool call.
- Checkpoint, rollback và log tác vụ bền vững.
- Giao diện Coding Agent trên web hoặc VS Code.

---

## 3. Nguyên tắc an toàn

Không cho agent public quyền shell và quyền ghi toàn bộ máy.

Các mức quyền:

| Chế độ | Quyền |
|---|---|
| `read_only` | Liệt kê, đọc, tìm kiếm, xem Git |
| `propose` | Đọc và tạo diff, chưa ghi file |
| `workspace_write` | Áp dụng patch trong workspace sau khi duyệt |
| `execute` | Chạy lệnh nằm trong allowlist sau khi duyệt |

Quy tắc bắt buộc:

- Mọi path phải nằm trong `WORKSPACE_DIR`.
- Không cho phép path traversal hoặc symlink thoát workspace.
- Không cho phép `rm`, `sudo`, shell pipe tùy ý hoặc tải và thực thi mã từ mạng.
- Không gửi `.env`, API key, credential hoặc file bí mật lên Cloud API.
- File nhạy cảm phải bị chặn: `.env`, SSH key, token, credential, database dump.
- Mọi thay đổi phải có diff và log.
- Giới hạn kích thước file, số file và tổng số byte mỗi tác vụ.
- Có timeout, giới hạn vòng lặp và giới hạn số tool call.

---

## 4. Cấu trúc test đề xuất

Tạo dần các file sau trong thư mục này:

```text
Notebooks/Test_Code_Editor/
|-- ROADMAP.md
|-- fixtures/
|   `-- sample_project/
|       |-- calculator.py
|       |-- README.md
|       `-- tests/test_calculator.py
|-- 01_test_tool_calling.ipynb
|-- 02_test_read_only_agent.ipynb
|-- 03_test_patch_generation.ipynb
|-- 04_test_file_editing.ipynb
|-- 05_test_command_runner.ipynb
|-- 06_test_agent_loop.ipynb
`-- results/
    |-- local.json
    |-- mimo.json
    `-- openai.json
```

Chỉ dùng `fixtures/sample_project` trong các bài test ghi, xóa hoặc chạy lệnh. Không thử lần đầu trên source thật của repo.

---

## 5. Giai đoạn 0: Tạo dự án sandbox

### Việc cần lập trình

Tạo một project Python nhỏ:

```text
fixtures/sample_project/
|-- calculator.py
|-- README.md
`-- tests/test_calculator.py
```

`calculator.py` ban đầu có một lỗi có chủ đích, ví dụ phép chia chưa kiểm tra chia cho 0.

### Test

Kiểm tra thủ công:

```bash
cd Notebooks/Test_Code_Editor/fixtures/sample_project
pytest -q
```

### Tiêu chí đạt

- Test ban đầu thất bại đúng như dự kiến.
- Không có file nào ngoài sandbox bị thay đổi.
- Có thể khôi phục sandbox bằng Git hoặc script reset fixture.

---

## 6. Giai đoạn 1: Kiểm tra tool-calling của từng model

### Mục tiêu

Xác định Local, MiMo và OpenAI có trả `tool_calls` đúng chuẩn OpenAI hay không.

### Việc cần lập trình

Tạo `01_test_tool_calling.ipynb`.

Khai báo một tool giả, chưa đụng filesystem:

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_file_metadata",
            "description": "Get metadata for a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        }
    }
]
```

Gửi cùng một yêu cầu tới ba model:

```text
Hãy dùng tool get_file_metadata để kiểm tra file calculator.py.
Không tự đoán kết quả.
```

### Dữ liệu cần ghi lại

- HTTP status.
- Model thực tế.
- Có `tool_calls` hay không.
- Tên tool.
- JSON arguments có parse được không.
- Có gọi đúng một tool không.
- Thời gian phản hồi.
- Token usage.

### Tiêu chí đạt

Model đạt khi:

- Trả đúng `get_file_metadata`.
- Arguments là JSON hợp lệ.
- Có `file_path`.
- Không tự bịa kết quả tool.

Nếu model không đạt:

- Không cấp tool ghi hoặc terminal.
- Có thể chỉ dùng model đó cho chat/generate code dạng text.

---

## 7. Giai đoạn 2: Agent chỉ đọc

### Mục tiêu

Cho model đọc code nhưng chưa được phép sửa.

### Tool cần có

```text
list_directory
read_file
search_text
get_file_metadata
git_status
git_diff
git_log
```

### Việc cần lập trình

Mở rộng `src/agent/mcp_client.py` hoặc tạo module tool riêng:

```text
src/agent/tools/
|-- filesystem.py
|-- search.py
`-- git.py
```

`search_text` nên dùng `rg`, giới hạn:

- Chỉ trong workspace.
- Số kết quả tối đa.
- Kích thước output tối đa.
- Bỏ qua `.git`, `node_modules`, model weights và binary.

### Test trong `02_test_read_only_agent.ipynb`

Chạy cho từng model:

1. Liệt kê project sandbox.
2. Đọc `calculator.py`.
3. Tìm tất cả nơi gọi hàm `divide`.
4. Giải thích lỗi.
5. Yêu cầu sửa file và xác nhận agent phải từ chối vì đang `read_only`.

### Tiêu chí đạt

- Đọc đúng file.
- Không đọc được path ngoài workspace.
- Không đọc được `.env`.
- Không tạo hoặc sửa file.
- Câu trả lời dựa trên nội dung tool trả về.
- Không lặp tool vô hạn.

---

## 8. Giai đoạn 3: Sinh patch nhưng chưa ghi file

### Mục tiêu

Model đề xuất thay đổi ở dạng diff có thể kiểm tra được.

### Tool cần thêm

```text
propose_patch
validate_patch
```

Không nên để model ghi lại toàn bộ file ngay từ đầu. Dùng unified diff:

```diff
*** Begin Patch
*** Update File: calculator.py
@@
-def divide(a, b):
-    return a / b
+def divide(a, b):
+    if b == 0:
+        raise ValueError("b must not be zero")
+    return a / b
*** End Patch
```

### Việc cần lập trình

Tạo schema:

```python
class ProposedChange(BaseModel):
    path: str
    patch: str
    explanation: str
```

`validate_patch` phải:

- Kiểm tra path trong workspace.
- Kiểm tra file tồn tại hoặc được phép tạo mới.
- Dry-run patch.
- Từ chối binary.
- Từ chối file bí mật.
- Trả về diff chuẩn hóa.

### Test trong `03_test_patch_generation.ipynb`

Yêu cầu:

```text
Đọc calculator.py và test hiện tại. Đề xuất patch sửa lỗi chia cho 0.
Không áp dụng patch.
```

### Tiêu chí đạt

- Patch áp dụng dry-run thành công.
- Chỉ sửa file cần thiết.
- Không ghi file.
- Patch giải quyết đúng lỗi.
- Không thay đổi định dạng hoặc code không liên quan.

---

## 9. Giai đoạn 4: Áp dụng thay đổi có phê duyệt

### Mục tiêu

Agent chỉ được sửa file sau khi người dùng duyệt patch.

### API đề xuất

Tách quy trình thành hai bước:

```text
POST /agent/tasks
POST /agent/tasks/{task_id}/approve
POST /agent/tasks/{task_id}/reject
```

Response bước đề xuất:

```json
{
  "task_id": "uuid",
  "status": "awaiting_approval",
  "model": "openai-model",
  "changes": [
    {
      "path": "calculator.py",
      "diff": "..."
    }
  ]
}
```

### Tool cần thêm

```text
apply_patch
create_file
```

Chưa thêm `delete_file` ở giai đoạn này.

### Test trong `04_test_file_editing.ipynb`

1. Yêu cầu model đề xuất patch.
2. Kiểm tra file chưa đổi trước khi approve.
3. Reject và kiểm tra file vẫn chưa đổi.
4. Tạo lại task, approve.
5. Kiểm tra chỉ file dự kiến bị đổi.
6. Xem `git diff`.

### Tiêu chí đạt

- Không approve thì không ghi.
- Reject không để lại file tạm.
- Approve áp dụng đúng diff.
- Không thể sửa ngoài sandbox.
- Có log model, tool, path, diff, thời gian và người duyệt.

---

## 10. Giai đoạn 5: Command runner theo allowlist

### Mục tiêu

Cho agent xác minh code mà không cấp shell tùy ý.

### Không triển khai kiểu này

```python
subprocess.run(command, shell=True)
```

### Thiết kế đề xuất

Tool nhận command có cấu trúc:

```json
{
  "program": "pytest",
  "args": ["-q"],
  "cwd": "Notebooks/Test_Code_Editor/fixtures/sample_project"
}
```

Allowlist ban đầu:

```text
pytest
python -m pytest
ruff check
ruff format --check
npm test
npm run build
```

Giới hạn:

- `shell=False`.
- `cwd` phải trong workspace.
- Timeout 60 giây.
- Giới hạn output.
- Không truyền biến môi trường bí mật.
- Không cho phép toán tử shell như `|`, `>`, `&&`, `$()`.

### Test trong `05_test_command_runner.ipynb`

- Chạy `pytest -q` trong sandbox.
- Chạy formatter check.
- Thử `rm -rf`, phải bị từ chối.
- Thử cwd ngoài workspace, phải bị từ chối.
- Thử command timeout, process phải bị dừng.

### Tiêu chí đạt

- Lệnh hợp lệ chạy được.
- Lệnh ngoài allowlist bị chặn.
- Output và exit code được trả về model.
- Không process nào còn chạy sau timeout.

---

## 11. Giai đoạn 6: Vòng lặp sửa và kiểm thử

### Mục tiêu

Agent thực hiện:

```text
đọc -> đề xuất -> duyệt -> sửa -> test -> phân tích lỗi -> sửa tiếp
```

### State LangGraph cần mở rộng

```python
class AgentState(TypedDict):
    messages: ...
    model: str
    permission_mode: str
    task_id: str
    iteration: int
    max_iterations: int
    pending_changes: list
    approved: bool
    test_results: list
```

### Giới hạn đề xuất

- Tối đa 8 vòng model.
- Tối đa 20 tool call.
- Tối đa 5 file thay đổi.
- Tối đa 200 KB thay đổi.
- Tối đa 2 lần chạy test thất bại trước khi dừng xin ý kiến.

### Test trong `06_test_agent_loop.ipynb`

Yêu cầu:

```text
Sửa hàm divide để xử lý chia cho 0, bổ sung test và chạy pytest.
```

### Tiêu chí đạt

- Agent đọc code và test trước khi sửa.
- Diff được duyệt.
- Test chuyển từ fail sang pass.
- Agent trả danh sách file thay đổi.
- Agent trả command và exit code.
- Không sửa file ngoài phạm vi.

---

## 12. Giai đoạn 7: Cho phép chọn model

### Thay đổi API

Mở rộng request:

```python
class AgentRequest(BaseModel):
    session_id: str
    task: str
    model: Literal["auto", "local", "mimo", "openai"] = "auto"
    permission_mode: Literal[
        "read_only",
        "propose",
        "workspace_write",
        "execute",
    ] = "read_only"
```

Mapping:

```python
AGENT_MODEL_ROUTES = {
    "auto": "coding-model",
    "local": "local-gemma",
    "mimo": "mimo-pro",
    "openai": "openai-model",
}
```

Sửa `get_llm()` để nhận model từ state thay vì cố định:

```python
def get_llm(model_name: str):
    return ChatOpenAI(
        model=model_name,
        openai_api_key=LITELLM_MASTER_KEY,
        openai_api_base=LITELLM_URL,
        temperature=0.1,
    )
```

### Lưu ý fallback

- `auto`: Gemma4 -> OpenAI theo cấu hình hiện tại.
- `local`: không fallback, giúp test đúng khả năng Gemma4.
- `mimo`: không fallback.
- `openai`: không fallback.

Không dùng fallback khi benchmark vì sẽ không biết model nào thật sự thực hiện tool call.

### Tiêu chí đạt

- Response ghi lại model được yêu cầu.
- Response ghi lại model/backend thực tế nếu LiteLLM cung cấp metadata.
- Ba model chạy cùng một bộ test.
- Model không hỗ trợ tool-calling bị giới hạn ở chế độ generate/propose.

---

## 13. Giai đoạn 8: Bộ test đánh giá ba model

Chấm từng model theo ma trận:

| Bài test | Local | MiMo | OpenAI |
|---|---:|---:|---:|
| Gọi đúng tool | | | |
| JSON arguments hợp lệ | | | |
| Đọc đúng file | | | |
| Tìm đúng symbol | | | |
| Không bịa nội dung file | | | |
| Sinh patch hợp lệ | | | |
| Chỉ sửa phạm vi yêu cầu | | | |
| Bổ sung test đúng | | | |
| Phân tích test failure | | | |
| Hoàn thành khi test pass | | | |
| Không vượt workspace | | | |
| Thời gian | | | |
| Tổng token/chi phí | | | |

Thang điểm đề xuất:

- Tool selection: 20%.
- Độ chính xác thay đổi: 30%.
- Test và tự sửa lỗi: 20%.
- An toàn/phạm vi: 20%.
- Tốc độ và chi phí: 10%.

Model chỉ được bật `workspace_write` khi:

- Không có vi phạm workspace.
- Tool-call pass ít nhất 9/10 lần.
- Patch hợp lệ ít nhất 9/10 lần.
- Không làm hỏng test có sẵn.

---

## 14. Giai đoạn 9: Tích hợp Web

Thêm tab Coding Agent vào React:

- Chọn model.
- Chọn permission mode.
- Nhập task.
- Hiển thị kế hoạch.
- Hiển thị tool đang gọi.
- Hiển thị diff từng file.
- Nút Approve/Reject.
- Hiển thị test output.
- Hiển thị trạng thái hoàn thành/thất bại.

Không render chain-of-thought nội bộ. Chỉ hiển thị:

- Kế hoạch ngắn.
- Tool call.
- Tool result đã lọc.
- Diff.
- Test result.
- Tổng kết.

---

## 15. Giai đoạn 10: Tích hợp VS Code

Hai lựa chọn:

### Cách 1: VS Code Task hoặc script

Tạo script gọi API:

```bash
python scripts/agent_cli.py \
  --model openai \
  --mode propose \
  --task "Đọc file đang mở và đề xuất sửa lỗi"
```

Đây là cách nên làm trước vì đơn giản.

### Cách 2: VS Code Extension

Extension có sidebar:

- Lấy workspace hiện tại.
- Gửi task tới Máy 2.
- Hiển thị diff bằng VS Code Diff Editor.
- Approve/Reject.
- Refresh file sau khi agent áp dụng patch.

Không gửi toàn bộ workspace qua API. Agent trên Máy 2 phải truy cập workspace được mount hoặc đồng bộ sẵn.

---

## 16. Thứ tự triển khai thực tế

Không làm tất cả cùng lúc. Thứ tự khuyến nghị:

1. Tạo fixture sandbox.
2. Tạo notebook test tool-calling cho ba model.
3. Hoàn thiện tool chỉ đọc.
4. Test agent `read_only`.
5. Sinh và validate patch nhưng chưa ghi.
6. Thêm approve/reject.
7. Áp dụng patch trong sandbox.
8. Thêm command runner allowlist.
9. Xây vòng lặp sửa-test.
10. Thêm lựa chọn model vào API.
11. Chạy bộ benchmark chung.
12. Tích hợp web.
13. Tích hợp VS Code.

---

## 17. Mốc triển khai đầu tiên

Mốc tiếp theo nên làm ngay:

```text
Milestone 1: Tool Calling Compatibility
```

Các file cần tạo:

```text
Notebooks/Test_Code_Editor/
|-- fixtures/sample_project/calculator.py
|-- fixtures/sample_project/tests/test_calculator.py
`-- 01_test_tool_calling.ipynb
```

Kết quả Milestone 1 phải trả lời được:

1. Gemma4 có gọi tool theo schema đúng không?
2. MiMo có gọi tool theo schema đúng không?
3. OpenAI có gọi tool theo schema đúng không?
4. Model nào đủ điều kiện đi tiếp tới agent chỉ đọc?
5. Model nào chỉ nên dùng để sinh code dạng text?

Chưa cấp quyền ghi file hoặc chạy terminal trước khi hoàn thành mốc này.

