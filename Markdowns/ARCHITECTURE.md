# Kiến trúc hệ thống Meibook

Tài liệu này mô tả kiến trúc hiện tại của repository `VLLM-PD` tại
`/home/jkl/Code/VLLM-PD`. Nội dung được đối chiếu với mã nguồn, Docker compose,
LiteLLM config và trạng thái API đang chạy.

> Trạng thái vận hành như số chunk, URL ngrok hoặc model endpoint có thể thay
> đổi theo thời điểm. Khi cần xác nhận lại, dùng các lệnh ở phần vận hành.

## 1. Trạng thái hiện tại

Meibook hiện là hệ thống hỏi đáp nội bộ cho MKAC với ba chế độ chính trên UI:

- `Hỏi đáp hành chính nhân sự MKAC` (`mode=mkac`).
- `Quản lý MES` (`mode=mes`).
- `Nghiên cứu tài liệu` (`mode=research`).

Chế độ `Nghiên cứu tài liệu` đã được bật lại. Luồng chính hiện dùng bộ tài liệu
Nhật `DocJP` đã index trong Qdrant collection `docjp_knowledge`, chia theo topic
cấu hình trong `config/research_topics.json`. Luồng demo/upload cũ vẫn tồn tại
qua collection `docmind_documents`.

Các điểm chính:

- Frontend React/Vite được build vào `frontend/dist` và được FastAPI phục vụ tại
  `/`.
- Backend FastAPI chạy cổng `8001`.
- LiteLLM chạy cổng nội bộ `4000`, chỉ backend gọi.
- Qdrant chạy cổng nội bộ `6333`, lưu vector tài liệu.
- SQLite dùng cho danh bạ nhân sự và MES snapshot.
- Gmail API dùng scope send-only để gửi email theo lệnh rõ ràng.
- UI có hai ngôn ngữ `VI / JP`.
- Frontend gửi `conversation_context` tối đa 16 tin nhắn gần nhất cho backend.
- ID khách demo là `000000`; đăng nhập bằng ID này tự chuyển UI sang tiếng Nhật.

## 2. Sơ đồ tổng quan

```text
Người dùng / Browser
        |
        | HTTP/HTTPS
        v
FastAPI :8001 + React SPA
        |
        |-- Auth employee_id
        |-- i18n VI/JP
        |-- query cache + minimum delay
        |
        |-- mode=mkac
        |     |-- quick answers
        |     |-- employee_directory.sqlite
        |     |-- RAG Qdrant mkac_knowledge
        |     |-- ddgs web fallback khi không có nguồn nội bộ phù hợp
        |     `-- Gmail send action
        |
        |-- mode=mes
        |     |-- deterministic/template router
        |     |-- data/mes.sqlite
        |     |-- SQL Agent an toàn cho câu phức tạp
        |     `-- Live MES API fallback cho một số intent
        |
        `-- mode=research
              |-- topic selector từ /research/topics
              |-- RAG DocJP: Qdrant docjp_knowledge
              `-- legacy upload/demo: Qdrant docmind_documents

FastAPI
        |
        v
LiteLLM :4000
        |
        |-- auto-model
        |     |-- Qwen3 14B qua IP tĩnh
        |     |-- fallback Qwen3 14B qua ngrok
        |     `-- fallback OpenAI
        |-- local-qwen-small
        |     |-- Qwen2.5 3B Instruct trên Ollama local Máy 2
        |     `-- fallback Qwen3/OpenAI
        |-- local-qwen-coder / coding-model
        |     |-- Qwen2.5 Coder 14B qua OpenAI-compatible endpoint
        |     `-- fallback Qwen3/OpenAI
        |-- openai-model
        `-- grok-model
```

## 3. Bố cục repository chính

```text
VLLM-PD/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI gateway
│   │   ├── config.py            # Cấu hình env/rate limit/cache/upload
│   │   ├── schemas.py           # Pydantic request/response
│   │   └── sse.py               # SSE event/status helpers
│   ├── rag/
│   │   ├── parser.py            # Docling/PyMuPDF, hỗ trợ text source curated
│   │   ├── embedder.py          # BAAI/bge-m3
│   │   ├── vector_store.py      # Qdrant wrapper
│   │   ├── rag_pipeline.py      # MKAC/research RAG, legacy routing
│   │   └── web_search.py        # ddgs fallback
│   ├── auth/
│   │   ├── employee_directory.py
│   │   ├── employee_intent.py
│   │   └── employee_answers.py
│   ├── integrations/
│   │   ├── mes_database.py      # MES SQLite deterministic queries
│   │   ├── mes_query_service.py # MES router/service tách khỏi RAG tài liệu
│   │   ├── mes_sql_agent.py     # Text-to-SQL có validate
│   │   ├── mes_client.py        # Live MES API client
│   │   └── gmail_sender.py      # Gmail send-only action
│   ├── i18n/
│   │   └── translation.py       # Dịch VI/JP bằng local-qwen-small
│   └── agent/                   # LangGraph Coding Agent, Docker web đang tắt
├── frontend/
│   ├── src/main.jsx
│   └── src/styles.css
├── config/
│   ├── quick_answers.json
│   ├── mkac_manifest.json
│   └── mes_semantic_model.json
├── database/
│   ├── raw_mkac/                # 3 file SQL nguồn MES mới
│   └── schema/mes.sql
├── documents/
│   ├── MKAC/                    # Tài liệu gốc để preview/trích dẫn
│   ├── MKAC-md/                 # Text curated để index MKAC
│   └── Research/                # Demo research và bộ DocJP cho Research
├── data/                        # SQLite, Gmail token/credentials, không commit
├── mkac_processed/pages/        # Ảnh trang phục vụ preview nguồn
├── docker-compose.web.yml
├── docker-compose.yml
├── litellm_config.yaml
└── Markdowns/
```

## 4. Runtime và cổng

| Thành phần | Cổng | Cách chạy | Vai trò |
|---|---:|---|---|
| FastAPI + React | `8001` | Docker service `app` hoặc uvicorn | API gateway, SSE, SPA |
| LiteLLM | `4000` | Docker service `litellm` | Model router |
| Qdrant | `6333` | Docker service `qdrant` | Vector database |
| Ollama local Máy 2 | `11434` | systemd `ollama.service` | Qwen2.5 3B phụ |
| Ollama proxy | `11435` | Docker service `ollama-proxy` | Bridge container -> host Ollama |
| Qwen3 chat | external | Ollama/ngrok | Chat model chính |
| Qwen coder | external `/v1` | llama.cpp/ngrok | SQL Agent/Coding |

`docker-compose.web.yml` bind LiteLLM và Qdrant vào `127.0.0.1`; chỉ cổng
`8001` được publish ra LAN.

## 5. Vòng đời khởi động

FastAPI dùng `lifespan` để tạo singleton:

1. `VectorStore` cho research legacy collection `docmind_documents`.
2. `VectorStore` cho MKAC collection `mkac_knowledge`.
3. `VectorStore` cho DocJP collection `docjp_knowledge`.
4. `Embedder` dùng `BAAI/bge-m3`.
5. `DocumentParser`.
6. `WebSearcher`.
7. `RAGPipeline`.
8. `MesQueryService`.

Một số singleton khác được tạo khi import module:

- `EmployeeDirectory(data/employee_directory.sqlite)`.
- `MesDatabase.from_env()`.
- `GmailSender.from_env()`.
- `TranslationService.from_env()`.

## 6. API chính

| Method | Endpoint | Chức năng |
|---|---|---|
| `GET` | `/health` | Health/config runtime, MES DB, employee DB |
| `GET` | `/models` | Model hiển thị cho UI, hiện chỉ `Local Model` |
| `GET` | `/knowledge/mkac/status` | Trạng thái Qdrant MKAC |
| `GET` | `/quick-answers` | Câu hỏi gợi ý theo mode/ngôn ngữ |
| `POST` | `/auth/employee` | Xác thực mã nhân viên hoặc guest `000000` |
| `POST` | `/sessions` | Tạo UUID session |
| `GET` | `/sessions/{session_id}` | Thông tin session tài liệu |
| `DELETE` | `/sessions/{session_id}` | Xóa session tài liệu |
| `POST` | `/sessions/{session_id}/upload` | Upload/index tài liệu research |
| `DELETE` | `/sessions/{session_id}/files/{filename}` | Xóa file khỏi session |
| `GET` | `/research/demo` | Metadata session demo research |
| `GET` | `/research/topics` | Topic Research DocJP, trạng thái file/chunk |
| `GET` | `/sources/preview` | Preview ảnh/trang nguồn trích dẫn |
| `POST` | `/query` | Query đồng bộ |
| `POST` | `/query/stream` | Query SSE streaming |
| `POST` | `/agent` | Coding Agent, chỉ bật nếu `ENABLE_AGENT=true` |

## 7. Auth nhân sự

Các mode `mkac` và `mes` yêu cầu `employee_id`.

- ID nhân viên thật được tra trong `data/employee_directory.sqlite`.
- ID guest `000000` là hồ sơ tổng hợp tạo trong code, không cần nằm trong DB.
- Guest tự chuyển UI sang tiếng Nhật.
- Greeting guest chỉ là lời chào chung, không gọi tên cá nhân.

Danh bạ hiện tại theo health check: `154` nhân viên.

## 8. i18n Việt/Nhật

Lõi backend hiểu tốt nhất tiếng Việt.

Với `mode=mkac`:

```text
Câu Nhật -> dịch sang Việt -> xử lý HR/RAG -> dịch câu trả lời sang Nhật
```

Với `mode=mes`:

```text
Câu Nhật -> giữ nguyên để rule MES nhận diện Lot/product/error -> dịch output
```

Lý do: nếu dịch trước câu MES, mã Lot, mã hàng, mã lỗi và tên lỗi dễ bị làm méo.
Bộ rule MES đã nhận diện một số marker Nhật như `ロット`, `品番`, `製品`,
`総エラー`, `2番目`, `比較`, `何ロット`.

Model dịch mặc định là `local-qwen-small`.

## 9. Context hội thoại

Frontend gửi `conversation_context` tối đa 16 tin nhắn user/assistant gần nhất.
Backend không coi đây là bộ nhớ bền vững, mà chỉ dùng cho resolver có cấu trúc:

- HR: “anh này”, “người này”, phòng ban vừa nhắc, so sánh phòng ban.
- MES: “lot đó”, “đứng thứ hai”, “so với mã hàng vừa hỏi”.
- Gmail: “gửi thông tin này...” lấy nội dung từ câu trả lời gần nhất.

Cache query tự bỏ qua các câu phụ thuộc context để tránh trả nhầm ngữ cảnh.

## 10. Dữ liệu MKAC

MKAC hiện dùng hai lớp tài liệu:

- `documents/MKAC/`: file gốc PDF/DOCX/HTML để preview/trích dẫn.
- `documents/MKAC-md/`: text/Markdown curated để index vector rõ hơn.

Trạng thái API gần nhất:

- collection: `mkac_knowledge`;
- số tài liệu: `18`;
- số chunk: `192`;
- số file Markdown curated trong `documents/MKAC-md`: `19`;
- số ảnh preview trong `mkac_processed/pages`: khoảng `108`.

Khi re-index MKAC, parser ưu tiên text source trong `documents/MKAC-md` nếu có,
nhưng vẫn giữ tên file gốc và ảnh trang để người dùng mở nguồn tham chiếu.

## 11. Pipeline `mode=mkac`

```text
Query
  -> auth employee_id
  -> quick answer/cache nếu có
  -> employee SQLite nếu là câu nhân sự có cấu trúc
  -> embed BGE-M3
  -> Qdrant mkac_knowledge
  -> prompt với sources + conversation_context
  -> LiteLLM local model
  -> dịch JP nếu cần
  -> SSE token/done
```

Nếu không tìm thấy chunk nội bộ đủ tốt, pipeline có thể dùng `ddgs` web fallback
và đánh dấu `answer_scope=web`; câu trả lời web không được coi là chính sách nội
bộ MKAC.

Token budget MKAC đã giảm để tránh local model lan man:

- general/no source: `256`;
- câu đơn giản: `512`;
- câu quy trình/chính sách/danh sách/chi tiết: `768`.

## 12. Pipeline `mode=mes`

MES không đi qua RAG tài liệu. Pipeline được tách sang `MesQueryService`.

```text
Query
  -> auth employee_id
  -> giữ nguyên câu Nhật nếu có
  -> deterministic/template router
  -> data/mes.sqlite read-only
  -> deterministic time SQL nếu là câu theo ngày/tháng
  -> SQL Agent nếu câu phức tạp và hợp lệ
  -> Live MES API fallback nếu cần
  -> template answer/local model
  -> dịch JP nếu cần
```

Các intent deterministic chính:

- tổng lỗi/số lot/số bản ghi theo mã hàng;
- thông tin một Lot;
- chi tiết lỗi theo Lot;
- Lot/mã hàng nhiều lỗi nhất, ít lỗi nhất, đứng thứ N;
- so sánh lỗi giữa hai mã hàng;
- mapping mã lỗi/tên lỗi sang process/tên lỗi;
- tổng hợp theo thời gian;
- câu mơ hồ như “Có bao nhiêu lot?” sẽ hỏi lại phạm vi.

SQL Agent chỉ nhìn `config/mes_semantic_model.json` và chỉ được query các view
allowlist. SQL được validate bằng `sqlglot`, mở SQLite read-only, ép `LIMIT`,
timeout ngắn và dùng fallback deterministic nếu model bỏ sót số liệu.

## 13. MES snapshot

MES snapshot hiện được import từ `database/raw_mkac`, không dùng bộ
`database/raw` cũ vì yêu cầu bảo mật.

Health check hiện tại:

| Chỉ số | Giá trị |
|---|---:|
| Raw lots | `2592` |
| Lot hiển thị sau khi loại test | `1325` |
| Lot test bị loại | `1267` |
| Raw error events | `654` |
| Error events hiển thị | `281` |
| Error events test bị loại | `373` |
| Error catalog | `969` |
| Tên lỗi chưa mapping | `2` |
| Imported at | `2026-06-25T09:25:15.464307+00:00` |

Top Lot hiển thị gần nhất:

| Lot | Mã hàng | Tổng lỗi |
|---|---|---:|
| `000866-05-000` | `KHTH_05` | `12870` |
| `000866-01-000` | `KHTH_05` | `11856` |
| `000943-03-000` | `0303-0303` | `10920` |
| `000866-02-000` | `KHTH_05` | `4680` |
| `000863-01-000` | `KHTH_06` | `3510` |

## 14. LiteLLM model routing

Ứng dụng gọi LiteLLM qua:

```text
http://localhost:4000/v1
```

Các model logic:

| Model | Backend | Vai trò |
|---|---|---|
| `auto-model` | `ollama_chat/qwen3:14b` | Route mặc định text |
| `local-qwen-chat` | `ollama_chat/qwen3:14b` | Chat chính local |
| `local-qwen-chat-ngrok` | `ollama_chat/qwen3:14b` | Fallback cùng model qua ngrok |
| `local-qwen-small` | `ollama_chat/qwen2.5:3b-instruct` | Dịch ngắn, intent, rewrite, format |
| `local-qwen-coder` | Ollama Qwen2.5 Coder 14B Q4 trên LAN | SQL Agent/Coding chính |
| `local-qwen-coder-ngrok` | llama.cpp Qwen2.5 Coder 14B Q5 qua ngrok | Coder fallback |
| `coding-model` | Qwen2.5 Coder 14B Q4 trên LAN | Tên ổn định cho Coding Agent |
| `openai-model` | `openai/gpt-5.4-mini` | Cloud fallback kỹ thuật |
| `grok-model` | Azure/OpenAI-compatible Grok | Route cũ cho vision/dự phòng |

Fallback hiện tại:

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
local-qwen-coder:
  - local-qwen-coder-ngrok
  - local-qwen-chat
  - openai-model
coding-model:
  - local-qwen-coder-ngrok
  - local-qwen-chat
  - openai-model
```

Lưu ý Qwen3 phải gọi qua LiteLLM provider `ollama_chat` tới root Ollama URL,
không gọi trực tiếp Ollama `/v1`, vì `/v1` có thể trả reasoning nhưng rỗng
`message.content`.

## 15. SSE

`POST /query/stream` trả `text/event-stream`.

Luồng bình thường:

```text
sources -> meta -> status... -> token... -> done
```

Các status dùng để UX biết backend đang xử lý tới đâu, ví dụ:

- đã hiểu yêu cầu;
- đang tra cứu dữ liệu;
- đang tổng hợp câu trả lời;
- đang chuyển đổi ngôn ngữ.

## 16. Frontend

Frontend vẫn là React SPA chủ yếu nằm trong `frontend/src/main.jsx` và
`frontend/src/styles.css`.

Các điểm đã cập nhật:

- UI chỉ hiện hai mode `HCNS` và `MES`.
- Model label hiển thị trung tính `Local Model`, không lộ Qwen/OpenAI/Grok trên UI.
- Phiên hỏi đáp tách theo mode và ngôn ngữ để tránh chuyển VI/JP làm lẫn lịch sử.
- Mobile có layout riêng cho top bar, input bar và nguồn tham chiếu.
- Nguồn tham chiếu có preview ảnh trang từ `mkac_processed/pages`.

## 17. Gmail send action

Gmail chỉ dùng scope:

```text
https://www.googleapis.com/auth/gmail.send
```

Credentials/token đặt trong `data/`, không commit.

Luồng:

```text
Câu hỏi có intent gửi mail rõ ràng
  -> parse địa chỉ email
  -> lấy nội dung từ câu hiện tại hoặc câu trả lời trước
  -> Gmail API gửi mail
  -> trả message id/thread id
```

Parser đã được chỉnh để không bắt nhầm các câu không phải gửi mail, ví dụ câu
“trả lời bằng tiếng Anh”.

## 18. Bảo mật và giới hạn

Các biện pháp hiện có:

- File upload có allowlist extension, giới hạn dung lượng, giới hạn số trang PDF.
- Upload/index nặng có semaphore và queue limit.
- MES SQLite mở read-only, query deterministic/parameterized.
- SQL Agent có semantic model, SQLGlot validate, authorizer read-only, timeout.
- LiteLLM/Qdrant trong Docker web chỉ bind `127.0.0.1`.
- Gmail token/credentials đặt trong `data/` và bị ignore.
- Agent tắt trong Docker web bằng `ENABLE_AGENT=false`.

Các giới hạn còn lại:

- Endpoint RAG chưa có auth user/tenant đầy đủ; employee_id là gate đơn giản.
- Rate limit/cache vẫn in-memory.
- CORS vẫn rộng trong code.
- Research mode đã bật lại, nhưng giới hạn `top_k` và `max_tokens` thấp hơn cấu
  hình cloud cũ để phù hợp local model.
- HR/RAG tiếng Nhật vẫn có thể gặp lỗi dịch lệch hoặc retrieval lệch ở câu khó.
- OpenAI/Grok vẫn tồn tại trong config làm fallback kỹ thuật.

## 19. Vận hành nhanh

Kiểm tra hệ thống:

```bash
curl -fsS http://localhost:8001/health | jq .
curl -fsS http://localhost:8001/models | jq .
curl -fsS http://localhost:8001/knowledge/mkac/status | jq .
```

Kiểm tra Docker:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml ps
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml logs -f app
```

Build frontend:

```bash
cd frontend
npm install
npm run build
```

Restart app:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml restart app
```

Test model qua LiteLLM:

```bash
KEY=$(sed -n 's/^LITELLM_MASTER_KEY=//p' .env.docker)
curl -fsS http://localhost:4000/v1/models \
  -H "Authorization: Bearer $KEY" | jq .
```

## 20. Test

Các test chính nằm trong `tests/`.

Nhóm đáng chú ý:

- `tests/test_mes_database.py`: deterministic MES intents.
- `tests/test_mes_sql_agent.py`: validate và chạy SQL Agent.
- `tests/test_mes_time_sql_routing.py`: câu MES theo ngày/tháng.
- `tests/test_query_routing.py`: tách HR/RAG/MES.
- `tests/test_employee_directory.py`: guest và match tên nhân viên.
- `tests/test_gmail_sender.py`: parser/send action.
- `tests/test_token_budgets.py`: token budget local model.

Bộ prompt regression nằm ở `Markdowns/TestPrompt.md`. File này không được cập
nhật trong lần chỉnh tài liệu này theo yêu cầu.
