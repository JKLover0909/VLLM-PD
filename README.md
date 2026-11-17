# Meibook - Hệ thống hỏi đáp nội bộ MKAC và MES

Meibook là ứng dụng hỏi đáp nội bộ chạy trên repository `VLLM-PD`. Hệ thống phục
vụ giao diện React, API FastAPI, Qdrant, LiteLLM, dữ liệu nhân sự MKAC, dữ liệu
MES/WMS snapshot, Agent QMS 5M1E (QA), Gmail send action và lớp dịch giao diện Việt/Nhật.

Runtime Dev của thay đổi WMS nằm tại `/home/jkl/Code/VLLM-PD-dev`, branch `dev`,
với `docker-compose.dev.yml` và cổng `8002/4001/6334`. Không dùng hướng dẫn này
để thao tác Production nếu chưa preflight checkout và Compose thực tế.

**Không commit/push hoặc chạy import/reindex trong khi chỉ kiểm tra code.**

**Trạng thái migration model (2026-11-17):** code Dev đang được chuẩn bị để mọi vai
trò LLM thông thường dùng một Qwen3.5-9B Q4_K_M từ xa làm primary, Azure chỉ fallback
khi lỗi. Runtime hiện tại chưa chuyển: Ollama trên host vẫn đang phục vụ model 9B
local. Endpoint được thử nghiệm hiện chỉ có HTTP; không cấu hình/activate để gửi dữ
liệu thật cho đến khi người dùng xác nhận TLS/VPN. Hardware telemetry được tóm tắt
tất định trong Meibook và không gửi tới model từ xa hoặc cloud. Production chưa rollout.

## Trạng thái hiện tại

- Frontend: React + Vite, build vào `frontend/dist` và được FastAPI phục vụ tại `/`.
- Backend: FastAPI trên host Production `8001` và Dev `8002`.
- LiteLLM: host Production hiện `4002`, Dev `4001`; cả hai app gọi cổng nội bộ Docker `4000`.
- Qdrant: host Production `6333`, Dev `6334`; mỗi stack giữ vector data riêng.
- Mục tiêu routing sau rollout: `auto-model`, `local-qwen-chat`, `local-qwen-coder`
  và `coding-model` sẽ dùng chung một Qwen3.5-9B Q4_K_M từ xa qua LiteLLM. Alias
  nghiệp vụ được giữ để tương thích. Runtime hiện chưa chuyển: Dev vẫn dùng Ollama;
  chưa có endpoint HTTPS/VPN được xác nhận/cấu hình, Production chưa rollout. Không
  đặt API key thật trong file tracked.
- Tóm tắt phần cứng CCTVAI được render tất định trên Meibook; telemetry không được
  gửi tới model. Chỉ lượt hardware/answer đã được đánh dấu mới lọc khỏi conversation
  context server; free text hoặc tool histories chưa được tổng quát hóa.
- BGE-M3 và Docling/OCR vẫn chạy cục bộ; chỉ LLM host Ollama dự kiến dừng sau khi
  cả hai stack đã rollout. Không gỡ Ollama/model trước khi xác minh không còn consumer.
- Azure `gpt-5.4-mini` là fallback tuần tự duy nhất cho chat và coder khi Qwen chính
  lỗi. Ảnh trước tiên thử primary; nếu server không hỗ trợ, lỗi có thể fallback
  Azure tuần tự. Cloud fallback có thể nhận prompt/ảnh theo chính sách hiện hành.
- Giao diện có các chế độ MKAC (`mode=mkac`), MES (`mode=mes`), WMS (`mode=wms`), QA (`mode=qa`), CCTVAI (`mode=cctvai`, khi khả dụng) và Nghiên cứu (`mode=research`). QA chuyển câu hỏi đến Agent QMS 5M1E và hỗ trợ hỏi nối tiếp (state agent giữ phía server theo session); hướng dẫn vận hành tại [Markdowns/QA_AGENT.md](Markdowns/QA_AGENT.md).
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
- UI chỉ hiển thị lựa chọn Qwen primary. Azure chỉ xuất hiện trong đường fallback
  sau lỗi kỹ thuật; model legacy `openai` từ client cũ được chuẩn hóa về primary.
- Frontend gửi `conversation_context` 16 lượt gần nhất để hỗ trợ các câu nối
  tiếp như "anh này", "lot đó", "đứng thứ hai" hoặc "so với sản phẩm/phòng ban
  vừa hỏi".

## Kiến trúc nhanh

```text
Người dùng / Client
        |
        | HTTP/HTTPS tới host Production :8001 / Dev :8002
        v
FastAPI + React SPA
        |
        |-- Auth nhân sự: employee_id, có guest 000000
        |-- UI i18n: VI/JP, dịch qua LiteLLM primary Qwen khi cần
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
        |-- mode=qa
        |     |-- employee gate; từ chối guest
        |     |-- Agent QMS 5M1E qua POST job + bounded polling
        |     |-- tiếng Việt; context/history của agent giữ phía server theo session
        |     |-- không dùng browser history, translation hay fallback
        |     |-- citation riêng trong qa_metadata, không dùng RAG source preview
        |
        |-- mode=research
              |-- chọn topic tài liệu DocJP từ /research/topics
              |-- RAG DocJP: Qdrant collection docjp_knowledge
              |-- fallback/legacy demo session trong docmind_documents
              |-- upload/index tài liệu riêng vẫn còn cho giai đoạn sau

FastAPI Dev :8002 / Production :8001
        |
        v
LiteLLM Dev :4001 / Production :4002 (host; container nội bộ :4000)
        |
        |-- auto-model / local-qwen-chat -> one remote Qwen primary, Azure fallback
        |-- local-qwen-coder / coding-model -> local Ollama qwen3.5:9b (Dev), Azure coder fallback
        |-- image requests -> Qwen primary; Azure only if unsupported/failure
        |-- CCTVAI hardware -> deterministic renderer; no model call
        |-- no selectable/manual cloud route; Azure is failure fallback only
```

Tài liệu kiến trúc chi tiết cũ hơn nằm trong [Markdowns/ARCHITECTURE.md](Markdowns/ARCHITECTURE.md).
Thiết kế database MES nằm trong [Markdowns/DATABASE.md](Markdowns/DATABASE.md).
Hướng dẫn deploy cũ nằm trong [Markdowns/DEPLOY.md](Markdowns/DEPLOY.md).
Hướng dẫn tích hợp và vận hành QA nằm trong [Markdowns/QA_AGENT.md](Markdowns/QA_AGENT.md).

## Thành phần runtime

| Thành phần | Cổng | Chạy bằng | Vai trò |
|---|---:|---|---|
| FastAPI + React | Dev `8002`, Production `8001` | Docker service `app` | API gateway, SSE, SPA |
| LiteLLM | Dev `4001`, Production `4002` (host) | Docker service `litellm` | Router tới Qwen primary và Azure fallback |
| Qdrant | Dev `6334`, Production `6333` | Docker service `qdrant` | Vector database |
| BGE-M3 / Docling | GPU host nếu bật cấu hình | App containers | Embedding/OCR cục bộ; không dùng Ollama |

Mọi alias dùng một model Ollama host `qwen3.5:9b` Q4_K_M (`OLLAMA_API_BASE`, bridge `meibook-ollama-proxy`); Azure chỉ là fallback khi lỗi.

`LOCAL_CHAT_NUM_CTX` and `LOCAL_AUX_NUM_CTX` are unused and should not be set as
current configuration.

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
|   |-- i18n/translation.py          # Lớp dịch VI/JP qua LiteLLM; target remote, runtime hiện Ollama
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
      -> LiteLLM (Dev runtime currently Ollama; remote Qwen target pending TLS/VPN)
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

Các giới hạn này giữ context vừa phải và giảm nguy cơ timeout khi demo. Runtime Dev hiện vẫn dùng Ollama; remote Qwen là target chờ TLS/VPN.

Kiểm tra gần nhất:

- `/research/topics` trả `ready=true` với collection `docjp_knowledge`.
- `/research/demo` vẫn `ready=true` với 2 file demo và 39 chunk.
- Query Research theo topic `information_systems` truy xuất đúng nguồn DocJP.
  Chất lượng câu trả lời tiếng Nhật cần tiếp tục kiểm thử; kết quả này không xác nhận
  remote Qwen target vì runtime Dev hiện vẫn dùng Ollama.

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
TRANSLATION_MODEL=local-qwen-chat
```

`local-qwen-chat` is a logical alias; Dev runtime currently resolves to Ollama until
the approved TLS/VPN endpoint is configured.

Chat và dịch dùng alias `local-qwen-chat`. Target sau rollout là cùng remote Qwen
primary, với Azure fallback tuần tự khi primary lỗi; hiện Dev runtime vẫn dùng Ollama.
Chỉ cấu hình endpoint thật sau khi tuyến TLS/VPN được xác minh.

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

Các alias nghiệp vụ được giữ ổn định. **Target sau rollout** là một Qwen3.5-9B
Q4_K_M remote primary trên endpoint HTTPS/VPN được xác minh; runtime Dev hiện vẫn
Ollama, endpoint chưa cấu hình:

| Alias | Vai trò |
|---|---|
| `auto-model` / `local-qwen-chat` | Chat, RAG, helper và dịch |
| `local-qwen-coder` | SQL agent MES/WMS/CCTVAI |
| `coding-model` | Coding Agent (vẫn tắt theo cấu hình web) |
| `azure-chat-fallback` | Azure `gpt-5.4-mini`, chỉ sau lỗi primary chat |
| `azure-coder-fallback` | Azure `gpt-5.4-mini`, chỉ sau lỗi primary coder |

Không có alias chọn cloud trực tiếp hoặc OpenAI provider thứ hai. Client cũ gửi
`model="openai"` được chuẩn hóa về primary. Azure fallback là tuần tự, không chạy
song song/hedge với Qwen. Model ảnh thử primary trước; nếu deployment không hỗ trợ
ảnh, lỗi capability có thể đi theo fallback Azure tuần tự.

LiteLLM gọi một model Ollama host (`ollama_chat/qwen3.5:9b`, `think=false`) cho mọi alias qua `OLLAMA_API_BASE`.

Fallback Azure có thể nhận prompt/context HR, MES, Research hoặc ảnh sau lỗi primary;
chỉ rollout khi chính sách dữ liệu cho phép gửi các nội dung đó tới Azure.

Telemetry phần cứng CCTVAI được dựng bằng `render_deterministic_answer` trên host
Meibook, không gọi alias LLM, không dịch bằng LLM, và cả lượt hỏi lẫn trả lời hardware
bị loại khỏi lịch sử gửi tới model sau đó. QA agent ngoài hệ thống vẫn là service
độc lập, không thuộc routing Qwen của Meibook.

Trước rollout thật, kiểm tra body passthrough bằng LiteLLM 1.83 với upstream giả:
Qwen phải nhận `chat_template_kwargs.enable_thinking=false`, Azure không nhận field
này; test tool/JSON/stream và lỗi/empty-response sequential fallback. Chỉ smoke dữ
liệu tổng hợp sau khi TLS/VPN cấu hình và được xác nhận. `/health` không chứng minh
kết nối upstream thành công.

## API chính

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Trạng thái app, Qdrant, MES DB, Gmail, translation |
| `GET` | `/models` | Danh sách model frontend; chỉ trả Qwen primary |
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
# Host Production: 4002; host Dev: 4001; trong Docker: http://litellm:4000/v1
LITELLM_URL=http://localhost:4002/v1
LITELLM_MASTER_KEY=sk-local

OLLAMA_API_BASE=http://host.docker.internal:11435
TRANSLATION_ENABLED=true
TRANSLATION_MODEL=local-qwen-chat
TRANSLATION_TEMPERATURE=0.1

# Qwen primary dùng thinking=false qua chat_template_kwargs.
# Không cấu hình Ollama API cho LiteLLM sau khi rollout xong.

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
curl -fsS http://localhost:4002/health/liveliness
```

Chưa có lệnh smoke test remote mặc định: phải đợi TLS/VPN endpoint được xác minh,
secret vận hành được nạp an toàn và người dùng cho phép rollout. Khi được phép, chạy
một câu tổng hợp qua LiteLLM Dev trước, đặt `fallbacks: []` để xác minh primary và
không dùng dữ liệu thật. Không truyền key trong command line, không bật `set -x`;
không chạy curl tới endpoint HTTP cũ.
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

### LiteLLM không gọi được model local

Kiểm tra `ollama ps`, service `ollama` và bridge `meibook-ollama-proxy` (`:11435`). Khi model
local lỗi, LiteLLM chỉ fallback tuần tự sang Azure `gpt-5.4-mini` cho chat/coder.

### Gmail báo token hết hạn

Tạo lại OAuth token Gmail với credentials hiện tại, sau đó kiểm tra:

```bash
curl -fsS http://localhost:8001/health | jq .gmail_send
```

### Qwen primary bị lỗi hoặc suy luận quá dài

Cấu hình primary chỉ được kích hoạt với HTTPS/VPN đã xác minh. Hệ thống gửi
`chat_template_kwargs.enable_thinking=false` tới Qwen; Azure chỉ được gọi sau lỗi
primary theo fallback tuần tự. `max_tokens` động theo mode; nếu lỗi tiếp diễn, kiểm tra
URL/model ID và trạng thái upstream sau khi TLS/VPN được xác nhận.
- template answer cho MES;
- kiểm tra lặp nội dung trong `RAGPipeline._clean_model_answer`.

Nếu vẫn gặp, ưu tiên:

1. thêm deterministic/template cho intent đó;
2. kiểm tra upstream và TLS/VPN;
3. giảm token budget tương ứng trong env;
4. giảm context/chunk đưa vào prompt.

## Ghi chú về `docmind/`

`docmind/` là demo/nhánh thử nghiệm cũ, không thuộc đường chạy chính. Runtime
hiện tại dùng `src/`, `frontend/`, `docker-compose.web.yml`,
`litellm_config.yaml`, `config/`, `documents/`, `data/` và các script ở root.
