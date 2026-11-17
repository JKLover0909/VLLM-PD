# Giải thích repository Meibook theo trạng thái hiện tại

Tài liệu này là bản đọc hiểu nhanh toàn bộ repository `VLLM-PD` ở thời điểm hiện
tại. Mục tiêu là giúp nắm được hệ thống đang chạy ra sao, các module nào chịu
trách nhiệm gì, và luồng dữ liệu đi qua đâu.

## 1. Meibook đang làm gì?

Meibook là ứng dụng hỏi đáp nội bộ cho MKAC, gồm:

1. `Hỏi đáp hành chính nhân sự MKAC`
   - hỏi quy định, nội quy, quy trình, thông tin nhân sự;
   - dùng SQLite nhân sự cho câu có cấu trúc;
   - dùng RAG Qdrant cho tài liệu nội bộ.
2. `Quản lý MES`
   - hỏi dữ liệu Lot, mã hàng, lỗi sản xuất;
   - ưu tiên query deterministic vào SQLite snapshot;
   - dùng SQL Agent an toàn cho câu phức tạp.
3. `Gmail send action`
   - gửi email khi người dùng ra lệnh rõ ràng.
4. `Nghiên cứu tài liệu`
   - đã bật lại trên UI;
   - ưu tiên bộ tài liệu Nhật `DocJP` trong `docjp_knowledge`;
   - người dùng chọn topic trước khi hỏi.
5. `Coding Agent`
   - vẫn còn code `/agent`, nhưng Docker web đang tắt bằng `ENABLE_AGENT=false`.

## 2. Các service runtime

```text
Browser
  -> FastAPI :8002 Dev / :8001 Production
       -> React SPA
       -> Qdrant :6334 Dev / :6333 Production
       -> SQLite data/*.sqlite
       -> LiteLLM :4001 Dev / :4002 Production (host; container :4000)
            -> Remote Qwen3.5-9B Q4_K_M primary (HTTPS/VPN only after approval)
            -> Azure gpt-5.4-mini sequential fallback on primary failure
       -> Hardware CCTVAI summary deterministic in Meibook; no model egress
```

Trong Docker web:

- `app`: FastAPI + React build + RAG/MES/Auth/Gmail.
- `qdrant`: vector database.
- `litellm`: model proxy.
- Remote Qwen endpoint: shared single-model primary; configure only via verified TLS/VPN route.

## 3. Những file quan trọng

| File | Vai trò |
|---|---|
| `src/api/main.py` | FastAPI gateway, endpoint, auth, cache, i18n, Gmail |
| `src/api/config.py` | Cấu hình env/rate limit/cache/upload |
| `src/api/schemas.py` | Request/response models |
| `src/rag/rag_pipeline.py` | RAG MKAC/research, DocJP topic retrieval và legacy upload path |
| `src/rag/parser.py` | Parse tài liệu, hỗ trợ text source `MKAC-md` |
| `src/rag/embedder.py` | BGE-M3 embedding |
| `src/rag/vector_store.py` | Qdrant wrapper |
| `src/auth/employee_directory.py` | SQLite nhân sự và intent HR |
| `src/auth/employee_intent.py` | Nhận diện câu hỏi nhân sự |
| `src/auth/employee_answers.py` | Format câu trả lời HR VI/JA |
| `src/integrations/mes_query_service.py` | Router MES chính |
| `src/integrations/mes_database.py` | Deterministic MES SQLite queries |
| `src/integrations/mes_sql_agent.py` | SQL Agent an toàn |
| `src/integrations/mes_intent.py` | Intent/time SQL helper |
| `src/integrations/mes_answer_format.py` | Format/fallback MES |
| `src/integrations/gmail_sender.py` | Gmail send-only integration |
| `src/i18n/translation.py` | Dịch VI/JP qua LiteLLM alias (Dev hiện resolve sang Ollama) |
| `frontend/src/main.jsx` | React UI |
| `frontend/src/styles.css` | CSS desktop/mobile |
| `litellm_config.yaml` | Model routing/fallback |
| `docker-compose.web.yml` | Runtime Docker chính |

## 4. Luồng vào hệ thống

Người dùng gửi câu hỏi qua `/query` hoặc `/query/stream`.

Payload quan trọng:

```json
{
  "session_id": "...",
  "question": "...",
  "model": "auto",
  "mode": "mkac",
  "ui_language": "vi",
  "employee_id": "000001",
  "conversation_context": []
}
```

`conversation_context` chứa tối đa 16 tin nhắn gần nhất để xử lý câu nối tiếp.

## 5. Luồng xử lý chung trong API

```text
Request
  -> rate limit
  -> validate session/model/mode
  -> check employee_id nếu mode=mkac/mes/wms
  -> localize query nếu cần
  -> cache lookup nếu hợp lệ
  -> route theo mode
  -> translate answer nếu UI tiếng Nhật
  -> cache store nếu hợp lệ
  -> trả JSON hoặc SSE
```

Các câu phụ thuộc context như “anh này”, “lot đó”, “gửi thông tin này” không
được cache theo câu chữ đơn thuần.

## 6. Mode MKAC

### 6.1. Thứ tự xử lý

```text
Query MKAC
  -> quick_answers
  -> Gmail intent nếu có
  -> EmployeeDirectory nếu là câu nhân sự có cấu trúc
  -> RAG Qdrant mkac_knowledge
  -> ddgs web fallback nếu không có nguồn nội bộ
  -> LiteLLM (Dev runtime currently Ollama; remote target pending TLS/VPN)
```

### 6.2. EmployeeDirectory

SQLite nhân sự nằm ở:

```text
data/employee_directory.sqlite
```

Hiện có `154` nhân viên. Guest `000000` là hồ sơ synthetic trong code.

Các câu được trả lời trực tiếp từ SQLite:

- công ty có bao nhiêu thành viên;
- có bao nhiêu phòng ban;
- phòng nào đông nhất;
- một người cụ thể là ai;
- người vừa nhắc ở lượt trước làm vai trò gì;
- phòng ban vừa nhắc có bao nhiêu người;
- so sánh hai phòng ban.

### 6.3. RAG tài liệu

Qdrant collection:

```text
mkac_knowledge
```

Text để index ưu tiên:

```text
documents/MKAC-md
```

File gốc để preview:

```text
documents/MKAC
```

Preview trang nguồn:

```text
mkac_processed/pages
```

Trạng thái gần nhất:

- `18` tài liệu;
- `192` chunk;
- khoảng `108` ảnh trang preview.

## 7. Mode MES

MES hiện đã tách khỏi RAG tài liệu. `mode=mes` đi qua:

```text
src/integrations/mes_query_service.py
```

### 7.1. Thứ tự xử lý

```text
Query MES
  -> giữ nguyên câu Nhật nếu ui_language=ja
  -> MesDatabase deterministic query
  -> deterministic time SQL
  -> compound highest Lot query
  -> SQL Agent nếu câu phức tạp
  -> Live MES API fallback nếu cần
  -> template answer nếu đang dùng local model
```

### 7.2. Vì sao không để LLM tự sinh SQL trực tiếp?

Vì rủi ro cao:

- model có thể chọn sai bảng/cột;
- model có thể không loại lot test;
- model có thể nhầm `total_error_qty` với `error_record_count`;
- model có thể bịa dữ liệu không có trong DB;
- câu Nhật/Việt/Anh có thể làm lệch mã Lot hoặc mã hàng.

Vì vậy hệ thống ưu tiên deterministic query trước. SQL Agent chỉ là lớp sau cho
câu phức hợp và vẫn bị validate rất chặt.

### 7.3. Các intent deterministic nổi bật

- Lot có tổng lỗi cao nhất/thấp nhất/đứng thứ N.
- Mã hàng có tổng lỗi cao nhất/đứng thứ N.
- Tổng lỗi, số lot, số bản ghi lỗi theo mã hàng.
- Chi tiết lỗi theo Lot.
- Top lỗi trong Lot.
- Mapping mã lỗi hoặc tên lỗi sang process/tên lỗi.
- So sánh hai mã hàng.
- Câu theo thời gian: ngày/tháng nào nhiều lỗi, top Lot trong tháng.
- Câu mơ hồ “Có bao nhiêu Lot?” sẽ yêu cầu nói rõ phạm vi.

### 7.4. SQL Agent

SQL Agent dùng:

```text
src/integrations/mes_sql_agent.py
config/mes_semantic_model.json
```

Chỉ cho truy cập:

- `v_error_details`
- `v_lot_error_summary`
- `v_lot_error_breakdown`
- `v_product_error_summary`

SQL Agent bị chặn:

- `INSERT`, `UPDATE`, `DELETE`;
- `DROP`, `ALTER`, `CREATE`;
- `ATTACH`, `PRAGMA`;
- multi-statement;
- bảng ngoài allowlist.

## 8. Dữ liệu MES mới

Bộ raw hiện dùng:

```text
database/raw_mkac/
├── M_LOT_202606251410.sql
├── D_ERROR_202606251410.sql
└── P_ERROR_202606251411.sql
```

Database sinh ra:

```text
data/mes.sqlite
```

Health check gần nhất:

| Chỉ số | Giá trị |
|---|---:|
| Raw lots | `2592` |
| Lot hiển thị | `1325` |
| Excluded test lots | `1267` |
| Raw error events | `654` |
| Error events hiển thị | `281` |
| Excluded test error events | `373` |
| Error catalog | `969` |
| Unmapped error names | `2` |

Top Lot hiện tại sau khi loại test:

1. `000866-05-000`, mã hàng `KHTH_05`, `12870` lỗi.
2. `000866-01-000`, mã hàng `KHTH_05`, `11856` lỗi.
3. `000943-03-000`, mã hàng `0303-0303`, `10920` lỗi.

## 9. Dịch VI/JP

Module:

```text
src/i18n/translation.py
```

Model mặc định:

```text
local-qwen-chat
```

Nguyên tắc:

- `mkac`: dịch câu Nhật sang Việt trước khi xử lý.
- `mes`: không dịch câu Nhật trước khi route SQL, để bảo toàn mã kỹ thuật.
- output/error/source preview được dịch về Nhật khi UI ở JP.
- một số câu phổ biến dùng static translation để nhanh và ổn định hơn.

## 10. Model routing

Trong `litellm_config.yaml` có bốn logical aliases cho một remote primary:

| Alias | Tác vụ |
|---|---|
| `auto-model` / `local-qwen-chat` | Chat, RAG, helper và dịch |
| `local-qwen-coder` | SQL agents MES/WMS/CCTVAI |
| `coding-model` | Coding Agent (web deployment vẫn tắt theo `ENABLE_AGENT=false`) |
| `azure-chat-fallback` / `azure-coder-fallback` | Azure `gpt-5.4-mini`, chỉ sau lỗi primary tương ứng |

Target model là một Qwen3.5-9B Q4_K_M qua LiteLLM OpenAI-compatible provider.
Mọi alias chạy trên một model Ollama host (`qwen3.5:9b`, `OLLAMA_API_BASE`); Azure chỉ là fallback.

LiteLLM đặt `chat_template_kwargs.enable_thinking=false` cho Qwen primary. Không
đưa extension này hoặc Ollama options sang Azure. Giữ `max_parallel_requests: 1`,
context metadata và token budgets bảo thủ. Alias client `openai` legacy chuẩn hóa
thành `auto-model`; không có route chọn cloud trực tiếp, OpenAI aliases thứ hai,
hedging hoặc request chạy song song.

Fallback tuần tự duy nhất:

```yaml
auto-model:
  - azure-chat-fallback
local-qwen-chat:
  - azure-chat-fallback
local-qwen-coder:
  - azure-coder-fallback
coding-model:
  - azure-coder-fallback
```

Ảnh thử primary trước; chỉ lỗi/capability unsupported mới sang Azure. CCTVAI
hardware summary dùng renderer deterministic tại API, không có model alias và không
gửi telemetry/history tới bất kỳ LLM nào. BGE-M3 và OCR/Docling giữ local.

Runtime chưa chuyển cho tới khi có TLS/VPN được xác minh. Dev cần smoke dữ liệu tổng
hợp qua LiteLLM với fallback tắt; Production và dừng Ollama cần phê duyệt riêng.

## 11. Gmail

Module:

```text
src/integrations/gmail_sender.py
```

Gmail OAuth files:

```text
data/gmail_credentials.json
data/gmail_token.json
```

Scope:

```text
https://www.googleapis.com/auth/gmail.send
```

Parser chỉ xử lý intent gửi email rõ ràng. Nếu token hết hạn/revoked, chạy lại:

```bash
python scripts/init_gmail_oauth.py
```

## 12. Frontend

Frontend chủ yếu nằm ở:

```text
frontend/src/main.jsx
frontend/src/styles.css
```

Điểm hiện tại:

- UI hiện `HCNS`, `MES` và `Research`.
- Research hiển thị topic selector, ưu tiên DocJP; demo/upload cũ vẫn còn.
- UI có toggle `VN / JP`.
- Lịch sử tách theo mode/ngôn ngữ để tránh đổi ngôn ngữ làm lẫn context.
- Model label trên UI thể hiện Qwen primary; cloud fallback không phải lựa chọn người dùng.
- Source preview mở ảnh trang từ backend.

## 13. Test quan trọng

Các test nên chạy khi sửa logic:

```bash
pytest tests/test_mes_database.py
pytest tests/test_mes_sql_agent.py
pytest tests/test_mes_time_sql_routing.py
pytest tests/test_query_routing.py
pytest tests/test_employee_directory.py
pytest tests/test_gmail_sender.py
pytest tests/test_token_budgets.py
```

Regression prompt:

```text
Markdowns/TestPrompt.md
```

File `TestPrompt.md` không được cập nhật trong lần chỉnh tài liệu này theo yêu
cầu.

## 14. Những điểm cần nhớ khi phát triển tiếp

1. Không cho MES đi lẫn RAG tài liệu.
2. Không cho LLM tự query SQL tự do.
3. Không dùng `database/raw` cũ.
4. Không commit `data/`, Gmail token, SQLite, raw dump.
5. Khi sửa frontend cần build lại `frontend/dist`.
6. Khi sửa `litellm_config.yaml` cần restart `litellm`.
7. Khi đổi tài liệu MKAC cần re-index Qdrant.
8. Khi đổi raw MES cần import lại `data/mes.sqlite`.
9. Khi đổi danh sách nhân sự cần import lại `employee_directory.sqlite`.
10. Với tiếng Nhật MES, ưu tiên rule/deterministic hơn dịch tự do.
