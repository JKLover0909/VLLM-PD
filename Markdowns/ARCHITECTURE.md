# Kiến trúc hệ thống Meibook

Tài liệu này mô tả kiến trúc dùng chung của repository `VLLM-PD`. Runtime phải
được xác định theo checkout thực tế: Production dùng `/home/jkl/Code/VLLM-PD`,
còn Dev dùng `/home/jkl/Code/VLLM-PD-dev` trên branch `dev` với
`docker-compose.dev.yml`. Nội dung được đối chiếu với mã nguồn và Compose config;
không dùng tài liệu này để suy ra môi trường đích khi vận hành.

> Trạng thái vận hành như số chunk, URL ngrok hoặc model endpoint có thể thay
> đổi theo thời điểm. Khi cần xác nhận lại, dùng các lệnh ở phần vận hành.

## 1. Trạng thái hiện tại

Meibook hiện là hệ thống hỏi đáp nội bộ cho MKAC với bốn chế độ chính trên UI:

- `Hỏi đáp hành chính nhân sự MKAC` (`mode=mkac`).
- `Quản lý MES` (`mode=mes`).
- `Quản lý Kho WMS` (`mode=wms`).
- `Nghiên cứu tài liệu` (`mode=research`).

WMS là mode độc lập, dùng employee gate chung với MKAC/MES nhưng chỉ truy vấn
snapshot `data/mes_wms.sqlite`. Câu hỏi WMS không được fallback sang MES database,
MES SQL Agent hoặc MES API. Câu hỏi tiếng Nhật được giữ nguyên để bảo toàn mã vật
tư và mã công đoạn.

Chế độ `Nghiên cứu tài liệu` đã được bật lại. Luồng chính hiện dùng bộ tài liệu
Nhật `DocJP` đã index trong Qdrant collection `docjp_knowledge`, chia theo topic
cấu hình trong `config/research_topics.json`. Luồng demo/upload cũ vẫn tồn tại
qua collection `docmind_documents`.

Các điểm chính:

- Frontend React/Vite được build vào `frontend/dist` và được FastAPI phục vụ tại
  `/`.
- Production mặc định dùng FastAPI/LiteLLM/Qdrant tại `8001/4000/6333`.
- Dev tách riêng tại `8002/4001/6334`, container hậu tố `-dev` và bind dữ liệu từ
  `/home/jkl/Code/VLLM-PD-dev/data`.
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
FastAPI + React SPA (`8001` Production / `8002` Dev)
        |
        |-- Auth employee_id
        |-- i18n VI/JP
        |-- query cache + minimum delay
        |
        |-- mode=mkac
        |     |-- HR Executive Report aggregate-only
        |     |-- quick answers / employee_directory.sqlite
        |     |-- RAG Qdrant mkac_knowledge
        |     |-- ddgs web fallback khi không có nguồn nội bộ phù hợp
        |     `-- Gmail draft/confirm action
        |
        |-- mode=mes
        |     |-- MES quality Executive Report từ data/mes.sqlite
        |     |-- deterministic/template router + SQL Agent read-only
        |     `-- Live MES API fallback cho một số intent
        |
        |-- mode=wms
        |     |-- deterministic current-balance từ data/mes_wms.sqlite
        |     |-- WMS Executive Report contract v4
        |     |-- metadata allowlist: freshness/dataset evidence/suppression
        |     `-- fail-closed, không truy vấn hoặc suy đoán từ MES
        |
        `-- mode=research
              |-- topic selector từ /research/topics
              |-- RAG DocJP: Qdrant docjp_knowledge
              `-- legacy upload/demo: Qdrant docmind_documents

FastAPI
        |
        v
LiteLLM (`4000` Production / `4001` Dev)
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
        |-- azure-*-fallback / openai-*-fallback
        `-- grok-model
```

### 2.1 Hợp đồng API WMS

`QueryRequest.mode` chấp nhận `wms`. Cả REST và SSE đều giữ nguyên `mode="wms"`
trong response metadata. Câu trả lời current balance có
`answer_scope="wms_database"`; báo cáo tổng quan có
`answer_scope="wms_executive_report"`.

`wms_metadata` là hợp đồng additive và được allowlist bằng Pydantic trước khi trả
ra client. Các nhóm chính gồm version contract, intent/domain/status,
`reason_codes`, freshness (`source_as_of`, basis, timezone, semantic epoch),
`dataset_evidence`, grain và pagination. Path SQLite, source file và identifier
nội bộ ngoài schema không được serialize.

WMS quick answer có thể là `query` (đọc snapshot deterministic) hoặc
`server_prepared`. Entry server-prepared không gửi answer/provenance thô tới
browser; client chỉ nhận ID/câu hỏi/execution và phải gọi `/query/stream`.
Backend xác thực employee trước khi resolve, đối chiếu ID + ngôn ngữ + canonical
question + revision/provenance, sau đó vẫn kiểm tra snapshot contract/freshness.
Request `/query` không được dùng để trả prepared WMS.

SSE WMS dùng workflow `wms_verification`: các milestone allowlist gồm kiểm tra
snapshot/data contract và kiểm tra phạm vi/căn cứ trả lời. `agent_plan`,
`tool_start`, `tool_result` chỉ mô tả thao tác deterministic đã hoàn tất; không
phát SQL, raw row, prompt, danh tính hoặc chain-of-thought. Token chỉ phát sau
validation; metadata có `workflow`, `source_kind`, `cache=false`. Snapshot,
metadata hoặc prepared entry lỗi đều fail-closed với `SUPPRESSED`, không fallback
sang MES/RAG/LLM.

Khi người dùng Stop, browser abort request và đánh dấu các bước WMS đang chạy/chờ
là `cancelled`; backend không phát answer cuối hoặc `done` giả sau cancellation.
Observability WMS chỉ lưu số đếm aggregate theo `source_kind`/outcome và latency
validation; không ghi câu hỏi, câu trả lời, SQL, session hay employee ID.

Luồng WMS Q&A:

```text
status(received/routing/wms)
  -> agent_plan
  -> tool_start/tool_result(validate snapshot)
  -> tool_start/tool_result(validate answer scope)
  -> status(finalizing) -> sources -> meta -> token -> agent_done -> done
```

`WMS_VERIFICATION_STEP_PACING_SECONDS=0.55` mặc định giãn nhịp trình bày các mốc
WMS đã hoàn tất. Đặt `0` để tắt; pacing chỉ thay đổi thời điểm phát SSE, không
thay đổi verification, dữ liệu hay kết quả. WMS không dùng query response cache.

Routing WMS gọi nhánh chuyên biệt của `MesQueryService`; nhánh này bắt buộc query
WMS snapshot kể cả khi câu hỏi không lặp lại chữ "WMS". Vì vậy câu hỏi gõ nhầm
về Lot/lỗi MES trong tab WMS chỉ nhận hướng dẫn phạm vi WMS, không chạm MES
snapshot, MES SQL Agent hoặc live MES API.

Các suite chính:

- `tests/test_wms_api.py`: REST/SSE parity, auth, i18n, cache và metadata safety.
- `tests/test_mes_wms_database.py`: contract v4, intent và suppression.
- `tests/test_import_mes_wms.py`: importer và dataset evidence.
- `tests/test_report_api.py` / `tests/test_report_agent.py`: WMS Executive Report.
- `tests/test_mes_integration.py`: cách ly WMS với các route MES.

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
│   ├── actions/
│   │   ├── report_intent.py     # Executive intent + capability fail-closed
│   │   ├── report_agent.py      # Report deterministic HR/MES/WMS
│   │   └── artifact_store.py    # HTML artifact in-memory TTL/LRU
│   ├── integrations/
│   │   ├── mes_database.py      # MES SQLite deterministic queries
│   │   ├── mes_wms_database.py  # WMS contract/current balance read-only
│   │   ├── mes_wms_contract.py  # WMS semantic/data contract
│   │   ├── mes_query_service.py # MES/WMS router tách khỏi RAG tài liệu
│   │   ├── mes_sql_agent.py     # Text-to-SQL có validate
│   │   ├── mes_client.py        # Live MES API client
│   │   └── gmail_sender.py      # Gmail draft-confirm, send-only scope
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
│   └── schema/
│       ├── mes.sql
│       └── mes_wms.sql
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

| Thành phần | Production | Dev | Vai trò |
| --- | ---: | ---: | --- |
| FastAPI + React | `8001` | `8002` | API gateway, SSE, SPA |
| LiteLLM | `4000` | `4001` | Model router |
| Qdrant | `6333` | `6334` | Vector database |
| Ollama local Máy 2 | `11434` | dùng chung endpoint host theo config | Model local phụ |
| Qwen chat/coder | external | external | Chat, dịch, SQL Agent |

Production dùng `docker-compose.web.yml`; Dev chỉ dùng `docker-compose.dev.yml`.
Trước mọi lệnh Docker phải kiểm tra branch, compose project, container, port và
bind-mount. Không fallback từ Dev sang Production nếu preflight lệch.

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
| `GET` | `/reports/{report_id}` | Tải HTML report artifact ngắn hạn |
| `POST` | `/query` | Query đồng bộ, có safe artifact khi là report |
| `POST` | `/query/stream` | Query SSE streaming |
| `POST` | `/agent` | Coding Agent, chỉ bật nếu `ENABLE_AGENT=true` |

## 7. Auth nhân sự

Các mode `mkac`, `mes` và `wms` yêu cầu `employee_id`.

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

Với `mode=mes` hoặc `mode=wms`:

```text
Câu Nhật -> giữ nguyên để rule MES/WMS nhận diện mã kỹ thuật -> dịch output
```

WMS giữ nguyên câu hỏi tiếng Nhật để bảo toàn mã vật tư, mã công đoạn và mã kệ kho;
chỉ câu trả lời cuối cùng mới được dịch theo ngôn ngữ giao diện.

Lý do: nếu dịch trước câu MES, mã Lot, mã hàng, mã lỗi và tên lỗi dễ bị làm méo.
Bộ rule MES đã nhận diện một số marker Nhật như `ロット`, `品番`, `製品`,
`総エラー`, `2番目`, `比較`, `何ロット`.

Model dịch mặc định là `local-qwen-small`.

## 9. Context hội thoại

Frontend gửi `conversation_context` tối đa 16 tin nhắn user/assistant gần nhất.
Backend không coi đây là bộ nhớ bền vững, mà chỉ dùng cho resolver có cấu trúc:

- HR: “anh này”, “người này”, phòng ban vừa nhắc, so sánh phòng ban.
- MES: “lot đó”, “đứng thứ hai”, “so với mã hàng vừa hỏi”.
- Gmail: “gửi thông tin này...” hoặc “gửi báo cáo này...” lấy nội dung text hoặc đính kèm artifact HTML tương ứng từ phiên làm việc.

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

## 12. Pipeline `mode=mes` và `mode=wms`

MES và WMS không đi qua RAG tài liệu. Cả hai dùng `MesQueryService`, nhưng mỗi mode
đi vào nhánh riêng. Report capability chạy trước Q&A thông thường và chỉ kích hoạt
khi có yêu cầu tạo report rõ ràng, hoặc đủ ba nhóm marker: đối tượng điều hành +
nhu cầu tổng quan + domain HR/MES/WMS rõ.

```text
Query
  -> auth employee_id + safety guard
  -> Gmail confirm/draft nếu là email action
  -> capability gate Executive Report
       |-- HR aggregate-only: chỉ mode=mkac
       |-- MES quality/error snapshot: chỉ mode=mes
       `-- WMS current balance contract v4: chỉ mode=wms
  -> nếu mode=mes:
       data/mes.sqlite read-only
       -> deterministic SQL / guarded SQL Agent / Live MES API fallback
  -> nếu mode=wms:
       data/mes_wms.sqlite read-only
       -> deterministic current-balance contract v4
       -> fail-closed, không fallback sang route MES chung
  -> template answer/local model
  -> dịch JP nếu cần
```

MES report chỉ chứng minh chất lượng/lỗi từ snapshot; không tuyên bố sản lượng,
OEE, chi phí hay ca sản xuất. WMS report không cộng quantity xuyên mã vật tư khi
chưa có UOM master, không suy diễn trend, delta, min-stock, expiry, WIP hoặc
bottleneck. HR report chỉ xuất aggregate theo phòng ban, không có roster/profile.

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
| `azure-*-fallback` | Azure/OpenAI-compatible Grok | Fallback Azure theo role Chat / Small / Coder |
| `openai-*-fallback` | `openai/gpt-5.4-mini` | Fallback OpenAI theo role Chat / Small / Coder |
| `grok-model` | Azure/OpenAI-compatible Grok | Route cũ cho vision/dự phòng |

Fallback hiện tại: local là primary; ngrok cùng role, Azure, rồi OpenAI.

```yaml
auto-model:
  - local-qwen-chat-ngrok
  - azure-chat-fallback
  - openai-chat-fallback
local-qwen-chat:
  - local-qwen-chat-ngrok
  - azure-chat-fallback
  - openai-chat-fallback
local-qwen-small:
  - local-qwen-chat
  - azure-small-fallback
  - openai-small-fallback
local-qwen-coder:
  - local-qwen-coder-ngrok
  - azure-coder-fallback
  - openai-coder-fallback
coding-model:
  - local-qwen-coder-ngrok
  - azure-coder-fallback
  - openai-coder-fallback
```

Lưu ý Qwen3 phải gọi qua LiteLLM provider `ollama_chat` tới root Ollama URL,
không gọi trực tiếp Ollama `/v1`, vì `/v1` có thể trả reasoning nhưng rỗng
`message.content`.

## 15. SSE

`POST /query/stream` trả `text/event-stream`.

Luồng Q&A bình thường:

```text
status... -> sources -> meta -> token/replace... -> done
```

Luồng Executive Report:

```text
status -> agent_plan -> tool_start/tool_result... -> artifact -> sources/meta
       -> token -> agent_done -> done
```

`artifact` chỉ chứa allowlist JSON cho React; không có raw HTML, SVG hoặc
Markdown nội bộ. HTML đầy đủ nằm trong `ArtifactStore` và được tải qua
`GET /reports/{id}`.

Các status dùng để UX biết backend đang xử lý tới đâu, ví dụ:

- đã hiểu yêu cầu;
- đang tra cứu dữ liệu;
- đang tổng hợp câu trả lời;
- đang chuyển đổi ngôn ngữ.

## 16. Frontend

Frontend vẫn là React SPA chủ yếu nằm trong `frontend/src/main.jsx` và
`frontend/src/styles.css`.

Các điểm đã cập nhật:

- UI hiện bốn mode độc lập: `HCNS`, `MES`, `WMS` và `Nghiên cứu tài liệu`.
- Model label hiển thị trung tính `Local Model`, không lộ Qwen/OpenAI/Grok trên UI.
- Phiên hỏi đáp tách theo mode và ngôn ngữ để tránh chuyển VI/JP làm lẫn lịch sử.
- Executive Report render bằng React thuần: KPI, chart, matrix, governance,
  limitations và actions; không chèn raw HTML vào DOM.
- Nút gửi email chỉ điền composer. Backend tạo draft, yêu cầu xác nhận, rồi mới
  gọi Gmail và lấy HTML attachment từ `ArtifactStore` theo session/employee.
- Mobile có layout riêng cho top bar, input bar, report card và nguồn tham chiếu.
- Nguồn tham chiếu có preview ảnh trang từ `mkac_processed/pages`.

## 17. Gmail send action

Gmail chỉ dùng scope:

```text
https://www.googleapis.com/auth/gmail.send
```

Credentials/token đặt trong `data/`, không commit.

Luồng:

```text
Câu hỏi gửi mail
  -> parse địa chỉ email
  -> tạo bản nháp (draft) lưu người nhận/nội dung/attachment
  -> yêu cầu người dùng nhập "Xác nhận gửi email" (hoặc Hủy)
  -> khi xác nhận, Gmail API gửi mail (đính kèm HTML artifact nếu là báo cáo)
  -> trả thông báo trạng thái gửi thành công
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

Phải preflight theo môi trường trước khi chạy Docker. Với checkout Dev:

```bash
cd /home/jkl/Code/VLLM-PD-dev
test "$(git branch --show-current)" = dev
docker compose -f docker-compose.dev.yml config -q
docker compose -f docker-compose.dev.yml ps
```

Config resolve phải cho thấy app/LiteLLM/Qdrant ở `8002/4001/6334`, container hậu
tố `-dev` và bind-mount từ checkout Dev. Nếu lệch thì dừng; không dùng
`docker-compose.web.yml` thay thế.

```bash
curl -fsS http://localhost:8002/health | jq .
curl -fsS http://localhost:8002/models | jq .
curl -fsS http://localhost:8002/knowledge/mkac/status | jq .
npm --prefix frontend run build
```

Production dùng checkout `/home/jkl/Code/VLLM-PD` và
`docker-compose.web.yml`; chỉ vận hành khi có yêu cầu Production rõ ràng.

## 20. Test

Các test chính nằm trong `tests/`.

Nhóm đáng chú ý:

- `tests/test_mes_database.py`: deterministic MES intents.
- `tests/test_mes_sql_agent.py`: validate và chạy SQL Agent.
- `tests/test_mes_time_sql_routing.py`: câu MES theo ngày/tháng.
- `tests/test_query_routing.py`: tách HR/RAG/MES.
- `tests/test_employee_directory.py`: guest và match tên nhân viên.
- `tests/test_gmail_sender.py`: parser, MIME attachment, draft/confirm/cancel.
- `tests/test_report_agent.py`: capability và report deterministic HR/MES/WMS.
- `tests/test_report_api.py`: REST/SSE safe artifact và email ownership.
- `tests/test_report_artifact_store.py`: artifact TTL/LRU.
- `tests/test_mes_wms_database.py`, `tests/test_wms_api.py`: WMS contract/API.
- `tests/test_token_budgets.py`: token budget local model.

Bộ prompt regression nằm ở `Markdowns/TestPrompt.md`. File này không được cập
nhật trong lần chỉnh tài liệu này theo yêu cầu.
