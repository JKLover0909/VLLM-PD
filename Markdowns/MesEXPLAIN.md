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
  -> FastAPI :8001
       -> React SPA
       -> Qdrant :6333
       -> SQLite data/*.sqlite
       -> LiteLLM :4000
            -> Qwen3 chat static IP
            -> Qwen3 chat ngrok fallback
            -> Qwen2.5 3B local helper
            -> Qwen2.5 Coder
            -> OpenAI fallback
```

Trong Docker web:

- `app`: FastAPI + React build + RAG/MES/Auth/Gmail.
- `qdrant`: vector database.
- `litellm`: model proxy.
- `ollama-proxy`: bridge để container gọi Ollama host tại `11435`.

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
| `src/i18n/translation.py` | Dịch VI/JP bằng local model |
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
  -> check employee_id nếu mode=mkac/mes
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
  -> LiteLLM local chat model
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
local-qwen-small
```

Nguyên tắc:

- `mkac`: dịch câu Nhật sang Việt trước khi xử lý.
- `mes`: không dịch câu Nhật trước khi route SQL, để bảo toàn mã kỹ thuật.
- output/error/source preview được dịch về Nhật khi UI ở JP.
- một số câu phổ biến dùng static translation để nhanh và ổn định hơn.

## 10. Model routing

Trong `litellm_config.yaml`:

| Model | Tác vụ |
|---|---|
| `auto-model` | route mặc định UI |
| `local-qwen-chat` | Qwen3 14B qua IP tĩnh |
| `local-qwen-chat-ngrok` | cùng Qwen3 qua ngrok fallback |
| `local-qwen-small` | dịch/intent/rewrite/format ngắn |
| `local-qwen-coder` | SQL Agent/Coding |
| `coding-model` | tên logic ổn định cho Coding Agent |
| `openai-model` | cloud fallback |
| `grok-model` | route cũ vision/dự phòng |

Fallback quan trọng:

```yaml
auto-model:
  - local-qwen-chat-ngrok
  - openai-model
local-qwen-chat:
  - local-qwen-chat-ngrok
  - openai-model
local-qwen-small:
  - local-qwen-chat
  - openai-model
coding-model:
  - local-qwen-coder
  - local-qwen-chat
  - openai-model
```

Lưu ý: `auto-model` và `local-qwen-chat` bản chất là cùng Qwen3 14B, khác nhau ở
tên logic và fallback.

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
- Model label trên UI là `Local Model`, không lộ tên Qwen/OpenAI/Grok.
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
