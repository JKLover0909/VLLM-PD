# Meibook - Hệ thống hỏi đáp nội bộ MKAC và MES

Meibook là ứng dụng hỏi đáp nội bộ chạy trên repository `VLLM-PD`. Hệ thống phục
vụ giao diện React, API FastAPI, Qdrant, LiteLLM, dữ liệu nhân sự MKAC, dữ liệu
MES/WMS snapshot, Gmail send action và lớp dịch giao diện Việt/Nhật.

Runtime Dev của thay đổi WMS nằm tại `/home/jkl/Code/VLLM-PD-dev`, branch `dev`,
với `docker-compose.dev.yml` và cổng `8002/4001/6334`. Không dùng hướng dẫn này
để thao tác Production nếu chưa preflight checkout và Compose thực tế.

**Không commit/push hoặc chạy import/reindex trong khi chỉ kiểm tra code.**

Trạng thái tài liệu này được cập nhật theo kiến trúc đang chạy tại
`/home/jkl/Code/VLLM-PD`.

## Trạng thái hiện tại

- Frontend: React + Vite, build vào `frontend/dist` và được FastAPI phục vụ tại `/`.
- Backend: FastAPI chạy cổng `8001`.
- LiteLLM: chạy nội bộ cổng `4000`, chỉ backend gọi.
- Qdrant: chạy nội bộ cổng `6333`, lưu vector tài liệu.
- Ollama local trên Máy 2: chạy bằng `systemd`, dùng cho model phụ
  `qwen2.5:3b-instruct`.
- Giao diện đang hiển thị 4 chế độ:
  - `Hỏi đáp hành chính nhân sự MKAC` (`mode=mkac`).
  - `Quản lý MES` (`mode=mes`).
  - `Quản lý Kho WMS` (`mode=wms`), truy vấn riêng snapshot current balance.
  - `Nghiên cứu tài liệu` (`mode=research`).
- WMS dùng xác thực `employee_id` giống MKAC/MES, giữ nguyên câu hỏi tiếng Nhật,
  và không fallback sang MES snapshot, MES SQL Agent hoặc MES API.
- WMS contract v4 chỉ trả lời các intent current balance đã được kiểm chứng;
  KPI, xu hướng, cộng gộp khác đơn vị và truy vấn current lot bị suppression.

### Thay đổi gần đây: tách WMS thành mode thứ 4

Client gửi `mode="wms"` tới `/query` hoặc `/query/stream`. Backend route trực tiếp
qua `MesQueryService.query_wms_outcome`, chỉ đọc `data/mes_wms.sqlite` ở chế độ
read-only và trả `answer_scope="wms_database"` cùng metadata contract allowlist.
Báo cáo tổng quan WMS dùng `answer_scope="wms_executive_report"` và artifact HTML
private theo session/employee. Khi snapshot không khả dụng, hệ thống trả trạng thái
suppressed/unavailable thay vì suy đoán từ dữ liệu MES.

Quick answers WMS được nạp từ `config/quick_answers.json` bằng endpoint
`/quick-answers?mode=wms&language=vi|ja`. Entry `server_prepared` chỉ trả ID,
câu hỏi và execution cho browser; nội dung chuẩn bị chỉ được backend resolve sau
employee gate, đối chiếu canonical question và kiểm tra revision/provenance.
Mọi WMS quick answer đi qua `/query/stream`, không cache và không bypass snapshot
contract/freshness validation.

SSE WMS biểu đạt các mốc kiểm chứng deterministic đã thực thi (snapshot/data
contract, rồi phạm vi/căn cứ trả lời) trước token cuối. `WMS_VERIFICATION_STEP_PACING_SECONDS`
mặc định là `0.55` giây để giãn nhịp trình bày các mốc đã hoàn tất; đặt `0` để tắt.
Nó chỉ thay đổi thời điểm phát SSE, không đổi verification, dữ liệu hay kết quả.
Đây không phải chain of thought; nếu contract, metadata hoặc prepared entry không
hợp lệ, backend trả `SUPPRESSED` và không fallback sang MES/SQL Agent/RAG/LLM.
Metrics WMS chỉ có số đếm aggregate theo nguồn/outcome, validation latency và
presentation pacing; không chứa câu hỏi, câu trả lời, SQL, session hoặc employee ID.

Kiểm tra phần thay đổi bằng:

```bash
cd /home/jkl/Code/VLLM-PD-dev
scripts/meibook-python -m pytest tests/test_wms_api.py tests/test_mes_integration.py -q
npm --prefix frontend run build
```

Không chạy lệnh import WMS hoặc `--reindex` nếu chưa xác nhận rõ mục tiêu dữ liệu.

- Chế độ Research đã được bật lại trên UI. Luồng mới ưu tiên bộ tài liệu nội
  bộ Nhật `DocJP` trong collection `docjp_knowledge`, chia theo các topic cố
  định. Luồng demo/upload cũ vẫn còn qua `docmind_documents`.
- UI có chuyển ngôn ngữ `VI / JP`. Lõi xử lý chính vẫn dùng tiếng Việt. Với
  `mode=mkac`, backend dịch câu hỏi tiếng Nhật sang tiếng Việt rồi dịch câu trả
  lời trở lại tiếng Nhật. Với `mode=mes`, câu hỏi tiếng Nhật được đưa thẳng vào
  bộ rule deterministic/SQL để tránh làm méo mã Lot, mã hàng và tên lỗi; chỉ câu
  trả lời cuối cùng mới được dịch sang tiếng Nhật.
- Người dùng MKAC/MES phải có `employee_id`. ID khách demo là `000000` và dùng
  giao diện tiếng Nhật.
- UI chỉ hiển thị lựa chọn model dạng `Local Model`. Các route cloud còn tồn tại
  trong LiteLLM như fallback kỹ thuật, không còn là lựa chọn chính trên giao diện.
- Frontend gửi `conversation_context` 16 lượt gần nhất để hỗ trợ các câu nối
  tiếp như "anh này", "lot đó", "đứng thứ hai" hoặc "so với sản phẩm/phòng ban
  vừa hỏi".

## Kiến trúc nhanh

```text
Người dùng / Client
        |
        | HTTP/HTTPS tới cổng 8001
        v
FastAPI + React SPA
        |
        |-- Auth nhân sự: employee_id, có guest 000000
        |-- UI i18n: VI/JP, dịch qua local-qwen-small khi cần
        |-- Cache câu hỏi phổ biến + delay tối thiểu để UX tự nhiên
        |
        |-- mode=mkac
        |     |-- prepared answers từ config/quick_answers.json
        |     |-- SQLite employee_directory.sqlite cho câu hỏi nhân sự có cấu trúc
        |     |-- context resolver cho người/phòng ban được nhắc ở lượt trước
        |     |-- RAG MKAC: Qdrant collection mkac_knowledge
        |     |-- Web fallback ddgs nếu không có chunk nội bộ phù hợp
        |     |-- Gmail send action nếu câu hỏi là lệnh gửi email
        |
        |-- mode=mes
        |     |-- deterministic/template query trước
        |     |-- context resolver cho Lot/mã hàng/ranking/so sánh nối tiếp
        |     |-- SQLite data/mes.sqlite, loại bỏ lot test
        |     |-- SQL Agent an toàn qua semantic model nếu câu hỏi phức tạp
        |     |-- Live MES API fallback cho case cần API trực tiếp
        |
        |-- mode=wms
        |     |-- deterministic current balance từ data/mes_wms.sqlite
        |     |-- WMS Executive Report contract v4
        |     |-- metadata freshness/evidence/suppression allowlist
        |     |-- fail-closed, không fallback sang MES/SQL Agent/live API
        |
        |-- mode=research
              |-- chọn topic tài liệu DocJP từ /research/topics
              |-- RAG DocJP: Qdrant collection docjp_knowledge
              |-- fallback/legacy demo session trong docmind_documents
              |-- upload/index tài liệu riêng vẫn còn cho giai đoạn sau

FastAPI
        |
        v
LiteLLM :4000
        |
        |-- auto-model        -> Qwen3 14B qua IP tĩnh, fallback ngrok/Azure/OpenAI
        |-- local-qwen-chat   -> Qwen3 14B qua LiteLLM ollama_chat
        |-- local-qwen-small  -> Qwen2.5 3B Instruct trên Ollama system service
        |-- local-qwen-coder  -> Qwen2.5 Coder 14B qua OpenAI-compatible LAN API
        |-- coding-model      -> Qwen2.5 Coder, fallback cùng role
        |-- azure-*-fallback  -> cloud Azure theo role Chat / Small / Coder
        |-- openai-*-fallback -> cloud OpenAI theo role Chat / Small / Coder
        |-- grok-model        -> route vision/dự phòng cũ
```

Tài liệu kiến trúc chi tiết cũ hơn nằm trong [Markdowns/ARCHITECTURE.md](Markdowns/ARCHITECTURE.md).
Thiết kế database MES nằm trong [Markdowns/DATABASE.md](Markdowns/DATABASE.md).
Hướng dẫn deploy cũ nằm trong [Markdowns/DEPLOY.md](Markdowns/DEPLOY.md).

## Thành phần runtime

| Thành phần | Cổng | Chạy bằng | Vai trò |
|---|---:|---|---|
| FastAPI + React | `8001` | Docker service `app` | API gateway, SSE, SPA |
| LiteLLM | `4000` | Docker service `litellm` | Router model OpenAI-compatible |
| Qdrant | `6333` | Docker service `qdrant` | Vector database |
| Ollama local | `11434` | `ollama.service` systemd | Host model phụ Qwen2.5 3B |
| Ollama proxy | `11435` | Docker service `ollama-proxy` | Bridge Docker -> host Ollama |
| Qwen3 chat | external URL | Ollama/ngrok | Model chat chính |
| Qwen Coder | external `/v1` | llama.cpp/ngrok | SQL Agent/Coding model |

`ollama-proxy` dùng `alpine/socat`, bind vào Docker gateway và chuyển tiếp tới
`127.0.0.1:11434` của host. Vì vậy LiteLLM trong container gọi model phụ qua:

```text
http://host.docker.internal:11435
```

Ollama, Docker và các container đều chạy nền, nên model phụ vẫn hoạt động sau
khi đóng SSH hoặc VSCode Remote.

## Thư mục chính

```text
.
|-- src/
|   |-- api/main.py                  # FastAPI, REST/SSE, auth, i18n, Gmail, static React
|   |-- rag/
|   |   |-- parser.py                # Docling parser
|   |   |-- embedder.py              # BGE-M3 embedding
|   |   |-- vector_store.py          # Qdrant wrapper
|   |   `-- rag_pipeline.py          # MKAC/research RAG + legacy MES path
|   |-- integrations/
|   |   |-- mes_database.py          # Allowlisted MES snapshot queries
|   |   |-- mes_query_service.py     # MES router, deterministic query, SQL agent
|   |   |-- mes_sql_agent.py         # Text-to-SQL harness an toàn
|   |   |-- mes_client.py            # Live MES API client
|   |   `-- gmail_sender.py          # Gmail send-only integration
|   |-- i18n/translation.py          # Lớp dịch VI/JP bằng LLM local
|   |-- auth/employee_directory.py   # SQLite nhân sự, employee gate
|   `-- agent/                       # LangGraph Coding Agent, hiện tắt trong Docker web
|-- frontend/                        # React + Vite UI
|-- config/
|   |-- quick_answers.json           # Câu hỏi/câu trả lời chuẩn bị sẵn
|   |-- mes_semantic_model.json      # Semantic model cho SQL Agent
|   `-- mkac_manifest.json           # Manifest tài liệu MKAC
|-- database/schema/                 # Schema/import logic MES đã chuẩn hóa
|-- documents/MKAC                   # Tài liệu hành chính nhân sự nội bộ
|-- documents/Research               # Tài liệu research demo và DocJP
|-- data/                            # SQLite, Gmail token/credentials, không commit
|-- mkac_processed/                  # Ảnh trang tài liệu để preview nguồn
|-- docker-compose.web.yml           # Runtime Docker chính hiện tại
|-- litellm_config.yaml              # Model aliases và fallback
|-- .env.example                     # Mẫu cấu hình
`-- docmind/                         # Demo cũ, không thuộc runtime chính
```

## Dữ liệu và database

### Qdrant

Qdrant lưu vector tài liệu:

- `mkac_knowledge`: kho tài liệu nội bộ MKAC, dùng cho `mode=mkac`.
- `docmind_documents`: tài liệu upload/research theo session và demo legacy.
- `docjp_knowledge`: kho tài liệu nội bộ Nhật `DocJP`, dùng cho Research theo topic.

Vector embedding dùng `BAAI/bge-m3`, kích thước vector `1024`, distance cosine.

### SQLite nhân sự MKAC

File mặc định:

```text
data/employee_directory.sqlite
```

Nguồn trích xuất mặc định:

```text
documents/MKAC/3. DANH SÁCH KHÁM SỨC KHOẺ 2026.pdf
documents/MKAC/0. Thong tin nhan su va lanh dao MKAC.html
```

Vai trò:

- Kiểm tra `employee_id` khi vào `mkac`, `mes` hoặc `wms`.
- Trả lời trực tiếp các câu hỏi có cấu trúc về nhân sự/phòng ban.
- Cung cấp context người dùng hiện tại cho RAG MKAC.
- ID khách demo `000000` được dùng cho khách không có mã nhân viên.

### SQLite MES snapshot

File mặc định:

```text
data/mes.sqlite
```

Nguồn raw mới nằm dưới `database/raw_mkac` khi import. Bộ `database/raw` cũ
không còn được dùng vì yêu cầu bảo mật dữ liệu.

MES snapshot có các bảng chuẩn hóa và view phục vụ truy vấn:

- `lots`
- `error_events`
- `error_catalog`
- `v_lot_error_summary`
- `v_lot_error_breakdown`
- `v_product_error_summary`
- `v_error_details`

Các lot/test data bị loại khỏi kết quả trả lời bằng điều kiện loại token
`test` trong mã hàng và mã Lot.

## Các mode xử lý

### 1. Hỏi đáp hành chính nhân sự MKAC (`mode=mkac`)

Pipeline:

```text
Query -> auth employee_id -> cache/prepared answer
      -> employee SQLite nếu là câu hỏi nhân sự có cấu trúc
      -> RAG Qdrant mkac_knowledge
      -> web fallback nếu không có chunk nội bộ phù hợp
      -> LiteLLM local model
      -> dịch JP nếu UI đang ở tiếng Nhật
      -> SSE trả sources/meta/token/done
```

Điểm đáng chú ý:

- Câu hỏi chuẩn bị sẵn được đọc từ `config/quick_answers.json`.
- Cache câu hỏi phổ biến dùng `QUERY_RESPONSE_CACHE_TTL_SECONDS` và
  `QUERY_RESPONSE_CACHE_SIZE`.
- Có delay tối thiểu `MIN_QUERY_RESPONSE_SECONDS` để câu trả lời quá nhanh không
  tạo cảm giác giả.
- Câu trả lời từ RAG được giới hạn token động:
  - general/no source: `MKAC_GENERAL_MAX_TOKENS=256`
  - câu đơn giản: `MKAC_SIMPLE_MAX_TOKENS=512`
  - câu cần quy trình/chính sách/danh sách/chi tiết: `MKAC_EXTENDED_MAX_TOKENS=768`

### 2. Quản lý MES (`mode=mes`)

Pipeline:

```text
Query -> auth employee_id -> cache
      -> giữ nguyên câu Nhật nếu mode=mes để bảo toàn mã/tên
      -> deterministic/template router
      -> MES SQLite snapshot
      -> deterministic SQL cho câu thời gian/top lot quan trọng
      -> SQL Agent nếu câu hỏi phức tạp và được allowlist
      -> Live MES API fallback cho một số intent
      -> template answer nếu dùng local model
      -> dịch JP nếu UI đang ở tiếng Nhật
```

MES ưu tiên deterministic/template để giảm latency và giảm rủi ro model tự sinh
SQL sai. SQL Agent chỉ dùng cho câu hỏi phức tạp thật sự, dựa trên
`config/mes_semantic_model.json`, validate SQL AST bằng `sqlglot`, chỉ cho phép
SELECT trên view an toàn và chạy SQLite read-only.

Các intent deterministic đáng chú ý:

- tổng lỗi/số lot/số bản ghi của mã hàng;
- chi tiết lỗi theo Lot;
- Lot hoặc mã hàng có tổng lỗi cao nhất/thấp nhất/đứng thứ N;
- so sánh tổng lỗi giữa hai mã hàng;
- mapping mã lỗi hoặc tên lỗi sang tên/process;
- câu mơ hồ như "Có bao nhiêu lot?" sẽ hỏi lại phạm vi thay vì tự suy diễn;
- các câu nối tiếp dùng `conversation_context`, ví dụ:
  - "Còn đứng thứ hai là gì?";
  - "So với 0303-0303 thì sao?";
  - "Loại lỗi nào phổ biến nhất trong lot đó?".

Token budget MES:

| Nhánh | Mặc định |
|---|---:|
| MES general explanation | `MES_GENERAL_MAX_TOKENS=256` |
| Live API answer format | `MES_LIVE_API_MAX_TOKENS=192` |
| MES database answer | `MES_DATABASE_MAX_TOKENS=384` |
| SQL planner | `MES_SQL_PLANNER_MAX_TOKENS=1200` |
| SQL answer formatting | `MES_SQL_ANSWER_MAX_TOKENS=384` |

SQL planner giữ token rộng hơn vì cần sinh JSON/SQL hợp lệ. Các nhánh format
câu trả lời được giới hạn ngắn để tránh Qwen local lan man.

### 3. Nghiên cứu tài liệu (`mode=research`)

Research đã được bật lại trên UI nhưng được giới hạn theo hướng demo/local:

- Luồng chính hiện tại là chọn nhóm tài liệu nội bộ Nhật `DocJP`.
- Backend đọc topic từ `config/research_topics.json`.
- Endpoint `/research/topics` trả danh sách topic, số file/chunk và câu hỏi gợi
  ý theo `VI/JP`.
- Retrieval topic dùng collection Qdrant `docjp_knowledge`, session logic
  `docjp`, filter theo `metadata.category`.
- Topic `all` cho phép tìm trên toàn bộ `DocJP` nếu `allow_all=true`.
- Text-only Research dùng `auto-model`/local stack, không còn ép sang Grok.

Các topic DocJP hiện có:

| Topic | Category | Vai trò |
|---|---|---|
| Công nghệ thông tin & Bảo mật | `information_systems` | IT, mạng, bảo mật, phần mềm, email, họp trực tuyến |
| Pháp chế & Quản lý rủi ro | `legal_compliance` | 3rdWATCH, an toàn, khủng hoảng, hợp đồng, con dấu |
| Kế toán | `accounting` | Rakuraku Seisan, thanh toán chi phí, Q&A |
| Hành chính tổng hợp | `general_affairs` | Tai nạn lao động, biểu mẫu, vật tư, đồng phục, cơ sở vật chất |

Luồng demo/upload cũ vẫn còn để tương thích:

- Upload tài liệu.
- Parse/OCR bằng Docling.
- Index vào `docmind_documents`.
- Endpoint `/research/demo`.
- Session demo cố định:

```text
00000000-0000-4000-8000-000000000001
```

Giới hạn local model hiện tại:

| Biến | Giá trị khuyến nghị |
|---|---:|
| `RESEARCH_TOP_K` | `6` |
| `RESEARCH_MAX_TOKENS` | `768` |
| `RESEARCH_SCORE_THRESHOLD` | `0.35` |

Mục tiêu là tránh local model phải tổng hợp quá nhiều context dài như cấu hình
cloud cũ, giảm nguy cơ timeout khi demo.

Kiểm tra gần nhất:

- `/research/topics` trả `ready=true` với collection `docjp_knowledge`.
- `/research/demo` vẫn `ready=true` với 2 file demo và 39 chunk.
- Query Research theo topic `information_systems` truy xuất đúng nguồn DocJP,
  nhưng câu trả lời tiếng Nhật vẫn có thể rơi vào fallback ổn định của local
  model. Đây là điểm cần tối ưu tiếp nếu dùng Research để demo tiếng Nhật dài.

## Dịch giao diện Việt/Nhật

Lõi backend hiểu tốt nhất tiếng Việt. Khi `ui_language="ja"`:

1. Với `mode=mkac`, câu hỏi tiếng Nhật được dịch sang tiếng Việt để RAG/SQLite
   nhân sự xử lý.
2. Với `mode=mes`, câu hỏi tiếng Nhật không dịch trước khi route SQL. Bộ rule
   MES tự nhận diện các marker Nhật như `ロット`, `品番`, `製品`, `総エラー`,
   `2番目`, `比較`, `何ロット` để bảo toàn mã Lot/mã hàng/tên lỗi.
3. Câu trả lời, error message và source preview được dịch về tiếng Nhật khi cần.

Model mặc định cho lớp dịch:

```text
TRANSLATION_MODEL=local-qwen-small
```

`local-qwen-small` là `qwen2.5:3b-instruct` Q4_K_M chạy trên Ollama local của
Máy 2. Model này nhẹ hơn Qwen3 14B và phù hợp cho:

- dịch ngắn
- phân loại intent
- rewrite câu hỏi
- format câu trả lời ngắn

Một số câu cố định/error phổ biến dùng static translation, không gọi model.

Lưu ý hiện tại: phần MES tiếng Nhật đã được đẩy mạnh sang deterministic routing.
Phần HR/RAG tiếng Nhật vẫn phụ thuộc nhiều hơn vào dịch và retrieval nên vẫn có
thể gặp lỗi chất lượng như dịch lệch thuật ngữ hoặc prompt leak ở một số câu khó.

## Gmail send action

Meibook có tích hợp gửi email qua Gmail API send-only scope:

```text
https://www.googleapis.com/auth/gmail.send
```

Biến cấu hình:

```env
GMAIL_SEND_ENABLED=true
GMAIL_CREDENTIALS_PATH=data/gmail_credentials.json
GMAIL_TOKEN_PATH=data/gmail_token.json
GMAIL_ALLOW_INTERACTIVE_AUTH=false
GMAIL_SENDER_EMAIL=
```

Cách hoạt động:

- Nếu câu hỏi là lệnh gửi email có địa chỉ người nhận, backend parse intent.
- Nếu người dùng chỉ nói "gửi thông tin này...", backend dùng
  `conversation_context` từ các tin nhắn trước để tạo nội dung.
- Parser Gmail chỉ xử lý intent gửi mail rõ ràng; các câu kiểu "trả lời bằng
  tiếng Anh" hoặc câu kỹ thuật có chữ gần giống email không được biến thành lỗi
  Gmail.
- Gmail token và credentials nằm trong `data/`, không commit.
- Nếu token hết hạn hoặc bị revoke, cần authenticate lại Gmail OAuth.

## Source preview

Nguồn tham chiếu trả về trong RAG có thể có ảnh preview trang. Frontend mở modal
preview qua endpoint:

```text
GET /sources/preview?session_id=...&mode=...&file=...&page=...
```

Với MKAC, ảnh trang nằm trong:

```text
mkac_processed/pages
```

Preview chỉ trả file nằm trong thư mục cho phép, tránh path traversal.

## Model routing

LiteLLM aliases trong `litellm_config.yaml`:

| Alias | Backend | Vai trò |
|---|---|---|
| `auto-model` | `ollama_chat/qwen3:14b` | Route mặc định cho hỏi đáp text |
| `local-qwen-chat` | `ollama_chat/qwen3:14b` | Chat model local chính |
| `local-qwen-small` | `ollama_chat/qwen2.5:3b-instruct` | Model phụ cho dịch/intent/rewrite/format |
| `local-qwen-coder` | Ollama OpenAI-compatible Qwen2.5 Coder 14B Q4 trên LAN | SQL Agent/Coding chính |
| `local-qwen-coder-ngrok` | llama.cpp Qwen2.5 Coder 14B Q5 qua ngrok | Coder fallback |
| `coding-model` | Qwen2.5 Coder 14B Q4 trên LAN | LangGraph Coding Agent |
| `azure-chat-fallback` / `openai-chat-fallback` | Cloud theo vai trò Chat | Dự phòng HR/MES/Research text |
| `azure-small-fallback` / `openai-small-fallback` | Cloud theo vai trò Small | Dự phòng dịch/intent/rewrite |
| `azure-coder-fallback` / `openai-coder-fallback` | Cloud theo vai trò Coder | Dự phòng MES SQL Agent/Coding |
| `grok-model` | Azure/OpenAI-compatible Grok route | Vision/dự phòng cũ |

Fallback hiện tại (giữ model local là primary; Azure đứng trước OpenAI):

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

Các fallback cloud tự động có thể nhận prompt cùng context HR, MES và Research
khi toàn bộ route local trước đó không dùng được. Chỉ bật khi chính sách dữ liệu
của tổ chức cho phép gửi ngữ cảnh đó tới cả Azure và OpenAI.

Lưu ý với Qwen3: phải gọi qua LiteLLM provider `ollama_chat` tới root Ollama
URL, không dùng trực tiếp endpoint `/v1`, vì một số Qwen3/Ollama OpenAI-compatible
responses có thể trả reasoning nhưng rỗng `message.content`.

Kiểm tra route phải thực hiện trên đúng môi trường mục tiêu. Với Dev hiện tại,
LiteLLM chỉ xác nhận alias bằng `/v1/models`; chỉ chạy một smoke request tuần tự
mỗi route sau khi xác minh upstream tương ứng đang truy cập được.

| Route | Mục đích kiểm tra |
|---|---|
| `auto-model`, `local-qwen-chat`, `local-qwen-chat-ngrok` | Chat/RAG local và fallback tunnel |
| `local-qwen-small` | Dịch/intent/rewrite local |
| `local-qwen-coder`, `local-qwen-coder-ngrok`, `coding-model` | MES SQL/Coding local |
| `azure-*-fallback`, `openai-*-fallback` | Cloud fallback đúng role |
| `grok-model` | Vision/dự phòng cũ |

Không suy luận rằng alias đã phục vụ request chỉ vì proxy healthy: cần kiểm tra
lần lượt endpoint upstream, rồi gọi một request ngắn có xác thực qua LiteLLM.

Nghĩa là app vẫn gọi model được, nhưng Qwen chat IP tĩnh hiện đang chết; route
chat chính đang sống nhờ fallback qua ngrok, sau đó mới tới OpenAI fallback.

## API chính

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Trạng thái app, Qdrant, MES DB, Gmail, translation |
| `GET` | `/models` | Danh sách model frontend; hiện chỉ trả Local Model |
| `POST` | `/auth/employee` | Kiểm tra mã nhân viên/guest |
| `GET` | `/knowledge/mkac/status` | Trạng thái kho MKAC |
| `GET` | `/research/demo` | Trạng thái research demo session |
| `GET` | `/research/topics` | Danh sách nhóm tài liệu Research DocJP |
| `GET` | `/sources/preview` | Ảnh preview trang nguồn |
| `GET` | `/quick-answers` | Câu hỏi gợi ý theo mode/ngôn ngữ |
| `POST` | `/sessions` | Tạo session UUID |
| `GET` | `/sessions/{session_id}` | Thông tin session |
| `DELETE` | `/sessions/{session_id}` | Xóa session và file upload |
| `POST` | `/sessions/{session_id}/upload` | Upload/index tài liệu research |
| `DELETE` | `/sessions/{session_id}/files/{filename}` | Xóa file khỏi session |
| `POST` | `/query` | Hỏi đáp non-streaming |
| `POST` | `/query/stream` | Hỏi đáp SSE streaming |
| `POST` | `/agent` | Coding Agent, cần key nếu bật |

Request `query/stream` mẫu:

```bash
SESSION_ID=$(curl -fsS -X POST http://localhost:8001/sessions | jq -r .session_id)

curl -N -fsS http://localhost:8001/query/stream \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"question\":\"Liệt kê 5 Lot có số lượng lỗi cao nhất\",
    \"model\":\"auto\",
    \"mode\":\"mes\",
    \"ui_language\":\"vi\",
    \"employee_id\":\"000000\"
  }"
```

SSE event thường gặp:

```text
status -> sources -> meta -> token -> done
```

Các status được dùng để người dùng thấy pipeline đang tới bước nào, ví dụ:

- đã hiểu câu hỏi
- đang xác định loại câu hỏi
- đang truy vấn MES
- đang chuyển đổi ngôn ngữ
- đang tổng hợp câu trả lời

Frontend tự gửi `conversation_context` tối đa 16 tin nhắn user/assistant gần
nhất trong payload `/query` và `/query/stream`. Backend chỉ dùng context này cho
các resolver có cấu trúc, không coi đây là bộ nhớ bền vững:

- HR: người vừa nhắc, phòng ban vừa nhắc, so sánh phòng ban;
- MES: Lot vừa nhắc, mã hàng vừa nhắc, xếp hạng và so sánh nối tiếp;
- Gmail: lệnh "gửi thông tin này..." lấy nội dung từ câu trả lời gần nhất.

## Cấu hình quan trọng

Các biến chính nằm trong `.env.example` hoặc `.env.docker`:

```env
LITELLM_URL=http://localhost:4000/v1
LITELLM_MASTER_KEY=sk-local

QWEN_CHAT_API_BASE=http://192.168.10.124:11434
QWEN_CHAT_NGROK_API_BASE=https://carless-overarch-establish.ngrok-free.dev
QWEN_CHAT_MODEL=qwen3:14b
QWEN_SMALL_API_BASE=http://host.docker.internal:11435
QWEN_SMALL_MODEL=qwen2.5:3b-instruct
QWEN_CODER_LAN_API_BASE=http://192.168.10.14:11434/v1
QWEN_CODER_LAN_API_KEY=sk-local
QWEN_CODER_NGROK_API_BASE=https://.../v1
QWEN_CODER_NGROK_API_KEY=sk-local

TRANSLATION_ENABLED=true
TRANSLATION_MODEL=local-qwen-small
TRANSLATION_TEMPERATURE=0.1
LOCAL_CHAT_NUM_CTX=16384
LOCAL_AUX_NUM_CTX=4096

EMPLOYEE_DIRECTORY_DB_PATH=data/employee_directory.sqlite
MES_DATABASE_ENABLED=true
MES_DATABASE_PATH=data/mes.sqlite
MES_SQL_AGENT_ENABLED=true
MES_SQL_AGENT_MODEL=local-qwen-coder
MES_SEMANTIC_MODEL_PATH=config/mes_semantic_model.json

QUERY_RESPONSE_CACHE_TTL_SECONDS=600
QUERY_RESPONSE_CACHE_SIZE=256
MIN_QUERY_RESPONSE_SECONDS=2.0

MKAC_GENERAL_MAX_TOKENS=256
MKAC_SIMPLE_MAX_TOKENS=512
MKAC_EXTENDED_MAX_TOKENS=768
MES_GENERAL_MAX_TOKENS=256
MES_LIVE_API_MAX_TOKENS=192
MES_DATABASE_MAX_TOKENS=384
MES_SQL_PLANNER_MAX_TOKENS=1200
MES_SQL_ANSWER_MAX_TOKENS=384

GMAIL_SEND_ENABLED=true
GMAIL_CREDENTIALS_PATH=data/gmail_credentials.json
GMAIL_TOKEN_PATH=data/gmail_token.json
```

Không commit `.env`, `.env.docker`, Gmail token, SQLite data hoặc SQL raw có dữ
liệu thật.

## Chạy bằng Docker web

Từ root repository:

```bash
cd /home/jkl/Code/VLLM-PD
docker compose -f docker-compose.web.yml up -d --build
```

Kiểm tra:

```bash
docker compose -f docker-compose.web.yml ps
curl -fsS http://localhost:8001/health | jq .
curl -fsS http://localhost:4000/health/liveliness
```

Kiểm tra model phụ trên host:

```bash
systemctl status ollama --no-pager
ollama list
curl -fsS http://localhost:11434/api/tags | jq .
```

Kiểm tra LiteLLM gọi được `local-qwen-small`:

```bash
curl -fsS http://localhost:4000/v1/chat/completions \
  -H 'Authorization: Bearer sk-local' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"local-qwen-small",
    "messages":[
      {"role":"system","content":"Chỉ trả đúng một nhãn: MES_DATA, HR_DATA, HR_RAG, EMAIL, OTHER."},
      {"role":"user","content":"Liệt kê 5 Lot có số lượng lỗi cao nhất"}
    ],
    "temperature":0,
    "max_tokens":32,
    "extra_body":{"think":false,"num_ctx":4096}
  }' | jq .
```

## Index dữ liệu

### Index tài liệu MKAC

Host Python phải dùng Conda environment `meibook-dev` qua wrapper
`scripts/meibook-python`; không chạy bare `python` từ Conda `base`.

```bash
scripts/meibook-python scripts/index_mkac_documents.py --dry-run
scripts/meibook-python scripts/index_mkac_documents.py
```

Index lại một file:

```bash
scripts/meibook-python scripts/index_mkac_documents.py \
  --file "ten-file.pdf" \
  --reindex
```

### Import MES snapshot

MES snapshot được tạo từ các file SQL raw đã chuẩn hóa. Cấu trúc schema/import
nằm trong `database/schema` và tài liệu thiết kế nằm trong
[Markdowns/DATABASE.md](Markdowns/DATABASE.md).

Sau khi import, kiểm tra `/health` để xem:

- `lots`
- `raw_lots`
- `excluded_test_lots`
- `error_events`
- `error_catalog`
- `unmapped_error_names`
- `sql_agent_available`

## Kiểm thử

Một số test nhẹ không cần load BGE-M3:

```bash
scripts/meibook-python -m py_compile src/rag/rag_pipeline.py src/integrations/mes_query_service.py

scripts/meibook-python -m pytest \
  tests/test_query_routing.py \
  tests/test_mes_time_sql_routing.py \
  tests/test_mes_sql_agent.py \
  tests/test_translation_service.py \
  tests/test_gmail_sender.py \
  tests/test_token_budgets.py
```

Lưu ý: một số test import `RAGPipeline` đầy đủ cần các dependency nặng như
`numpy`, `torch`, `sentence-transformers`. Nếu chạy ngoài Docker/env đầy đủ mà
thiếu package, lỗi import không đồng nghĩa app Docker đang lỗi.

### Regression prompt

Bộ prompt regression nằm ở:

```text
Markdowns/TestPrompt.md
```

Kết quả chạy gần nhất:

- tổng `180` dòng test;
- chạy `178` dòng;
- bỏ qua `2` dòng gửi email thật (`80`, `JA-080`);
- không crash;
- các lỗi deterministic chính đã sửa và xác nhận lại qua API thật:
  - Lot nhiều lỗi thứ 2;
  - Lot ít lỗi nhất và các Lot đồng hạng;
  - câu "Có bao nhiêu lot?" mơ hồ;
  - MES tiếng Nhật cho `製品0303-0303`, so sánh `KHTH_05と0303-0303`,
    `2番目にエラーが多いロット`;
  - câu tiếng Nhật "trả lời bằng tiếng Anh" không còn bị parser Gmail bắt nhầm;
  - guardrail tiếng Nhật cho yêu cầu sửa/cập nhật thông tin.

Log chi tiết lần chạy gần nhất được lưu tạm tại:

```text
/tmp/meibook_testprompt_results.json
```

Các vấn đề còn lại đáng chú ý nằm chủ yếu ở HR/RAG tiếng Nhật: dịch lệch thuật
ngữ, retrieval lệch điều khoản, hoặc prompt leak ở một số câu khó. Đây là nhóm
cần xử lý riêng, không phải lỗi deterministic MES.

Smoke test MES:

```bash
SESSION_ID=$(curl -fsS -X POST http://localhost:8001/sessions | jq -r .session_id)

curl -N -fsS http://localhost:8001/query/stream \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"question\":\"Liệt kê 5 Lot có số lượng lỗi cao nhất\",
    \"model\":\"auto\",
    \"mode\":\"mes\",
    \"ui_language\":\"vi\",
    \"employee_id\":\"000000\"
  }"
```

Smoke test MKAC:

```bash
curl -N -fsS http://localhost:8001/query/stream \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"question\":\"Quy định làm thêm giờ ở MKAC như thế nào?\",
    \"model\":\"auto\",
    \"mode\":\"mkac\",
    \"ui_language\":\"vi\",
    \"employee_id\":\"000000\"
  }"
```

Smoke test tiếng Nhật:

```bash
curl -N -fsS http://localhost:8001/query/stream \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"question\":\"一番エラーが多いLotはどれですか？\",
    \"model\":\"auto\",
    \"mode\":\"mes\",
    \"ui_language\":\"ja\",
    \"employee_id\":\"000000\"
  }"
```

## Bảo mật và giới hạn

- `.env`, `.env.docker`, `data/`, SQL raw thật, token OAuth và SQLite runtime
  không được commit.
- API MKAC/MES dùng `employee_id` gate, nhưng đây chưa phải hệ thống đăng nhập
  doanh nghiệp đầy đủ.
- UUID session là định danh kỹ thuật, không phải secret bảo mật.
- LiteLLM và Qdrant chỉ bind loopback trong `docker-compose.web.yml`.
- Gmail integration chỉ có scope gửi email, không đọc mailbox.
- SQL Agent chỉ chạy read-only SQLite và validate query trước khi execute.
- MES snapshot loại test lot khỏi mọi câu trả lời thông qua query/view/filter.
- Coding Agent trong Docker web đang tắt bằng `ENABLE_AGENT=false`.

## Xử lý lỗi thường gặp

### App không lên

```bash
docker compose -f docker-compose.web.yml ps
docker compose -f docker-compose.web.yml logs --tail=160 app
```

### LiteLLM không gọi được model phụ

```bash
docker compose -f docker-compose.web.yml ps ollama-proxy litellm
docker compose -f docker-compose.web.yml logs --tail=120 ollama-proxy
systemctl status ollama --no-pager
curl -fsS http://localhost:11434/api/tags | jq .
```

Nếu container LiteLLM không thấy `host.docker.internal:11435`, kiểm tra service
`meibook-ollama-proxy`.

### Gmail báo token hết hạn

Tạo lại OAuth token Gmail với credentials hiện tại, sau đó kiểm tra:

```bash
curl -fsS http://localhost:8001/health | jq .gmail_send
```

### Câu trả lời local model bị lan man

Hệ thống đã có:

- `think:false` cho local chat model.
- dynamic `max_tokens` theo mode.
- template answer cho MES local.
- kiểm tra lặp nội dung trong `RAGPipeline._clean_model_answer`.

Nếu vẫn gặp, ưu tiên:

1. thêm deterministic/template cho intent đó;
2. giảm token budget tương ứng trong env;
3. giảm context/chunk đưa vào prompt;
4. chỉ sau đó mới đổi model.

## Ghi chú về `docmind/`

`docmind/` là demo/nhánh thử nghiệm cũ, không thuộc đường chạy chính. Runtime
hiện tại dùng `src/`, `frontend/`, `docker-compose.web.yml`,
`litellm_config.yaml`, `config/`, `documents/`, `data/` và các script ở root.
