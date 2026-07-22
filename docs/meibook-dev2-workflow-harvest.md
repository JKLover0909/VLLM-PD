# Meibook-dev2 harvest — understand-meibook-repository

- runId: `wf_3ddcee26-827`
- taskId: `wwuihyguq`
- workflow: `understand-meibook-repository`
- status: **killed** (aborted/killed — no final completion record)
- error: `Error: Workflow aborted
    at b (/$bunfs/root/src/entrypoints/cli.js:5844:7328)
    at abort (unknown)
    at zKr (/$bu`
- agents done: **23** / progress stuck: **5** / other: **0**
- totalTokens (recorded): 3177283
- totalToolCalls: 2070

## Ý nghĩa

Workflow nền chạy tối qua **bị kill** khi process Claude thoát (rời công ty / SSH đóng).
Nhiều agent map/verify **đã xong** và có `resultPreview`; phase tổng hợp cuối **không** có completion record.
Vì vậy `claude --resume` báo: *No completion record was found for background workflow...*

## Câu trả lời session (end_turn ~21:14) — nếu có

# Đã đọc hiểu codebase Meibook

Tôi đã khảo sát toàn bộ **bề mặt source code, cấu hình, frontend, backend, scripts và test** của repository; các subsystem cũng được đối chiếu chéo giữa implementation, API contract và test.

Tôi chủ động không đọc `.env`, OAuth token, SQLite/Qdrant payload, log hay tài liệu nghiệp vụ thật. Không chạy Docker, import/index, gửi Gmail hoặc tạo Calendar event.

## 1. Mô hình tổng thể

Meibook là một **modular monolith** gồm:

```text
React/Vite SPA
      │ REST + SSE
      ▼
FastAPI application
      ├── Employee access gate + HR structured lookup
      ├── MKAC document RAG
      ├── MES deterministic query + SQL-agent fallback
      ├── Research curated topics
      ├── Research session uploads
      ├── Report Agent
      ├── Calendar/Gmail actions
      └── Optional Coding Agent
             │
             ├── SQLite: Employee + MES snapshot
             ├── Qdrant: 3 logical collections
             ├── LiteLLM: local/cloud model routing
             ├── MES live API
             ├── Gmail
             └── Google Calendar MCP
```

ASGI entry point thật là `src.api.main:app` được Uvicorn khởi chạy từ [Dockerfile:48-53](Dockerfile#L48-L53). File [src/main.py](src/main.py) không phải entry point runtime hợp lệ.

---

## 2. Bản đồ module

| Lớp | Thành phần | Trách nhiệm |
|---|---|---|
| Frontend | [frontend/src/main.jsx:1031-1186](frontend/src/main.jsx#L1031-L1186) | Composition root React; quản lý workspace, session, login, messages, Research, SSE và source preview |
| Components | [frontend/src/components/](frontend/src/components/) | Login, input, message list, Research sidebar và preview dialog |
| API schemas | [src/api/schemas.py:17-121](src/api/schemas.py#L17-L121) | Contract Pydantic cho query, employee, Research và Agent |
| API composition | [src/api/main.py:130-350](src/api/main.py#L130-L350) | FastAPI app, lifespan, singleton services, middleware và route assembly |
| SSE | [src/api/sse.py:12-69](src/api/sse.py#L12-L69) | Chuẩn hóa SSE event |
| API helpers | [src/api/helpers.py:24-227](src/api/helpers.py#L24-L227) | UUID, filename, client IP, cache key và follow-up detection |
| RAG orchestration | [src/rag/rag_pipeline.py:76-536](src/rag/rag_pipeline.py#L76-L536) | Retrieval, prompt, model routing, streaming, fallback và Calendar read |
| Vector store | [src/rag/vector_store.py:28-406](src/rag/vector_store.py#L28-L406) | Qdrant collections, filtering, search và session/file lifecycle |
| Parsing | [src/rag/parser.py:51-120](src/rag/parser.py#L51-L120) | Docling/PyMuPDF, OCR, chunking và page preview |
| Embedding | [src/rag/embedder.py:19](src/rag/embedder.py#L19) | BGE-M3 embedding |
| HR | [src/auth/employee_directory.py:21-220](src/auth/employee_directory.py#L21-L220) | Employee access gate và structured HR lookup |
| MES | [src/integrations/mes_query_service.py:40-168](src/integrations/mes_query_service.py#L40-L168) | Deterministic routing, SQLite/live API và Text-to-SQL fallback |
| Report | [src/actions/report_agent.py:448-545](src/actions/report_agent.py#L448-L545) | Lập plan và tạo báo cáo MES deterministic |
| Calendar | [src/actions/calendar_action.py:243-467](src/actions/calendar_action.py#L243-L467) | Draft, confirm, room validation và create event |
| Coding Agent | [src/agent/graph.py:45-118](src/agent/graph.py#L45-L118) | LangGraph tool loop |
| MCP | [src/agent/mcp_client.py:31-166](src/agent/mcp_client.py#L31-L166) | Tool discovery cho filesystem, Git và Calendar |
| Ingestion | [scripts/](scripts/) | OCR, import SQLite, index Qdrant và Research evaluation |
| Deployment | [Dockerfile](Dockerfile), [docker-compose.dev.yml](docker-compose.dev.yml), [docker-compose.web.yml](docker-compose.web.yml) | Build và topology Dev/Production |

---

## 3. Startup và runtime

Trong lifespan, backend khởi tạo tuần tự:

1. Ba `VectorStore`;
2. BGE-M3 embedder;
3. document parser;
4. web search;
5. `RAGPipeline`;
6. MES service;
7. Report Agent;
8. Calendar Action Service.

Composition nằm tại [src/api/main.py:278-331](src/api/main.py#L278-L331).

Một số side effect xảy ra ngay khi import module, trước lifespan, ví dụ tạo upload directory tại [src/api/main.py:130-139](src/api/main.py#L130-L139).

Nếu `frontend/dist` tồn tại khi backend được import, FastAPI mount SPA tại `/` ở [src/api/main.py:2542-2549](src/api/main.py#L2542-L2549). Build frontend sau khi process đã chạy không tự thêm mount; cần restart app.

---

## 4. Các luồng nghiệp vụ chính

### HR/MKAC

Employee ID hiện là **access gate**, không phải authentication mạnh:

- nhận ID sáu chữ số;
- hỗ trợ guest `000000`;
- hoặc kiểm tra record tồn tại trong SQLite;
- không có password, token hoặc server-side authenticated session.

Implementation chính: [src/auth/employee_directory.py:21-77](src/auth/employee_directory.py#L21-L77).

Thứ tự xử lý `/query` tổng quát:

1. rate limit và validation;
2. Research static/runtime cache;
3. safety guard;
4. structured HR lookup;
5. prepared MKAC answer;
6. dịch JA→VI và structured lookup lần hai;
7. Calendar;
8. Report Agent;
9. Gmail;
10. cuối cùng mới đến MKAC RAG hoặc MES.

Luồng nằm tại [src/api/main.py:1750-1948](src/api/main.py#L1750-L1948).

Có một khác biệt đáng chú ý: `/query/stream` đặt prepared answer trước safety/HR lookup, nên REST và SSE có thể chọn đường xử lý khác nhau cho cùng input tại [src/api/main.py:1951-2170](src/api/main.py#L1951-L2170).

Nếu không khớp structured answer, MKAC tìm trong collection `mkac_knowledge`, session `mkac`. Một số intent nhân sự/company profile được hậu lọc category; nếu lọc không còn kết quả thì pipeline dùng lại tập kết quả ban đầu tại [src/rag/rag_pipeline.py:1285-1318](src/rag/rag_pipeline.py#L1285-L1318).

### MES

MES ưu tiên deterministic logic thay vì để LLM tự dựng số liệu:

1. live max-Lot hoặc SQLite snapshot;
2. deterministic time SQL;
3. deterministic compound/highest-Lot SQL;
4. LLM Text-to-SQL có validation;
5. general MES explanation;
6. fail-closed nếu không hỗ trợ.

Cascade nằm tại [src/integrations/mes_query_service.py:63-156](src/integrations/mes_query_service.py#L63-L156).

MES không dùng Qdrant document RAG cho Q&A dữ liệu thông thường. `query_stream()` hiện tính toàn bộ answer trước rồi phát thành một token, chưa phải token streaming thực.

### Research curated topics

Research topic dùng:

- registry [config/research_topics.json](config/research_topics.json);
- collection DocJP;
- fixed DocJP session;
- filter `metadata.category`.

Retrieval được thực hiện tại [src/rag/rag_pipeline.py:1332-1380](src/rag/rag_pipeline.py#L1332-L1380).

Hiện có nguy cơ drift giữa ba nguồn:

1. `collection`/`session_id` trong topic registry;
2. env/runtime config;
3. indexer hard-code session `docjp`.

Các field collection/session trong registry được trả ra API nhưng không phải nguồn cấu hình authoritative cho retrieval.

### Research upload theo session

Luồng upload:

1. frontend tạo UUID session;
2. backend kiểm UUID, basename, traversal, extension và kích thước;
3. lưu file;
4. parse và embed;
5. xóa vectors cũ của cùng filename;
6. upsert chunks mới;
7. query lọc collection `docmind_documents` theo UUID.

API lifecycle nằm tại [src/api/main.py:1638-1747](src/api/main.py#L1638-L1747).

Session UUID chỉ là namespace, không phải ownership credential. Ai biết UUID và truy cập được API có thể thao tác session nếu không có lớp ACL/reverse proxy bên ngoài.

Replace upload cũng chưa transactional: ghi file, xóa vectors và thêm vectors là các bước riêng. Các phương thức delete/remove của Qdrant còn có xu hướng chuyển lỗi thành `0` hoặc kết quả rỗng tại [src/rag/vector_store.py:99-204](src/rag/vector_store.py#L99-L204).

### Source preview

Endpoint chọn store theo mode/scope, lấy `metadata.image_path`, re-root đường dẫn và chỉ trả ảnh trong ba root cho phép tại [src/api/main.py:1388-1445](src/api/main.py#L1388-L1445).

Topic preview chưa enforce category/topic tương ứng; category hiện là retrieval scope, không phải authorization boundary.

### Report Agent

Report Agent:

1. phân loại capability;
2. từ chối report ngoài allowlist;
3. dựng deterministic SQL plan;
4. chạy từng SQL step;
5. tạo KPI/sections/limitations;
6. lưu HTML artifact;
7. phát timeline qua SSE.

Implementation: [src/actions/report_agent.py:448-545](src/actions/report_agent.py#L448-L545).

Artifact được lưu in-memory, TTL khoảng sáu giờ, tối đa 200 item tại [src/actions/artifact_store.py:27-53](src/actions/artifact_store.py#L27-L53). Restart hoặc nhiều worker có thể khiến artifact biến mất/không đồng nhất.

### Calendar và Gmail

Calendar có hai luồng:

- read-only Calendar tools trong RAG pipeline;
- create flow qua draft → confirm → `create-event`.

Draft/confirm nằm tại [src/actions/calendar_action.py:340-467](src/actions/calendar_action.py#L340-L467).

Gmail có thể gửi ngay khi command hợp lệ, không có bước confirmation riêng, recipient-domain allowlist hoặc role gate rõ ràng trong application layer; orchestration nằm tại [src/api/main.py:1036-1181](src/api/main.py#L1036-L1181).

### Optional Coding Agent

`POST /agent` bị vô hiệu nếu `ENABLE_AGENT=false`. Nếu bật, endpoint gọi LangGraph với MCP tools tại [src/api/main.py:2477-2530](src/api/main.py#L2477-L2530).

Điểm cần lưu ý:

- source config mặc định bật Agent;
- API key mặc định có thể rỗng;
- fallback MCP có cả `write_file`;
- Compose Dev/Production hiện chủ đích đặt `ENABLE_AGENT=false`.

Cấu hình liên quan nằm tại [src/api/config.py:30-33](src/api/config.py#L30-L33) và [src/api/config.py:78-84](src/api/config.py#L78-L84).

---

## 5. Frontend và API contract

`QueryRequest` được định nghĩa tại [src/api/schemas.py:17-44](src/api/schemas.py#L17-L44), gồm:

- `session_id`;
- `question`;
- `model`;
- `mode`;
- `ui_language`;
- `employee_id`;
- `conversation_context`;
- `research_scope`;
- `research_topic`.

Field `stream` chỉ mang tính compatibility; endpoint quyết định transport:

- `/query`: JSON;
- `/query/stream`: SSE.

SSE backend phát frame dạng `data: <JSON>\n\n` tại [src/api/sse.py:56-57](src/api/sse.py#L56-L57). Frontend tự parse tại [frontend/src/main.jsx:822-853](frontend/src/main.jsx#L822-L853).

Frontend xử lý status, agent plan/tool events, artifact, sources, metadata, token, replace và error. Nó chưa có handler riêng cho `citations` và `done`; completion chủ yếu dựa trên EOF/finally.

State được chia thành sáu workspace:

- `mkac:vi`, `mkac:ja`;
- `mes:vi`, `mes:ja`;
- `research:vi`, `research:ja`.

Employee profile, theme, language, session IDs và titles được lưu localStorage. Messages, sources và artifact cards chỉ ở React memory nên mất khi reload.

---

## 6. Dữ liệu và external boundaries

| Boundary | Nội dung |
|---|---|
| Browser | localStorage và React state |
| FastAPI process | query cache, rate limiter, metrics, report artifacts, Calendar drafts |
| SQLite | employee directory và MES snapshot |
| Qdrant | `mkac_knowledge`, `docjp_knowledge`, `docmind_documents` |
| Filesystem | uploads, previews, processed documents, JSON configuration |
| LiteLLM | model routing local/cloud |
| MES API | dữ liệu live cho một số route |
| Gmail | side effect gửi email |
| Google Calendar | read/create event |
| Coding MCP | filesystem, Git và Calendar tools |

Query cache, artifacts và Calendar drafts đều là **per-process memory**. Khi chạy nhiều worker, request sau có thể sang worker khác và không thấy state vừa tạo.

---

## 7. Dev và Production

### Production

[Docker Compose Production](docker-compose.web.yml) dùng:

- web mặc định `8001`;
- LiteLLM `4000`;
- Qdrant `6333`;
- container không có hậu tố `-dev`.

### Dev local hiện tại

[Docker Compose Dev](docker-compose.dev.yml) dùng:

- web `8002`;
- LiteLLM `4001`;
- Qdrant `6334`;
- container có hậu tố `-dev`.

Dev vẫn chia sẻ Docker daemon, GPU/cache và một số model upstream với Production. Hai stack còn dùng chung image tags `:latest`, nên không hoàn toàn isolated ở tầng image/model.

Cả hai Compose bind-mount `./src:/app/src:ro`, có thể che `src/agent/node_modules` đã cài trong image. Calendar runtime vì vậy phụ thuộc bản dependency local ignored trong checkout.

`docker-compose.dev.yml`, host Python wrapper và host dependency overlay hiện là local/untracked, nên clone sạch chưa tái tạo đầy đủ topology Dev chỉ từ Git.

Ngoài ra, Vite dev proxy hiện trỏ mặc định sang port `8001` và thiếu một số API prefix tại [frontend/vite.config.js:10-19](frontend/vite.config.js#L10-L19). Chạy `npm run dev` không điều chỉnh config có nguy cơ gọi nhầm Production; frontend được FastAPI phục vụ trên port Dev `8002` không có vấn đề này.

---

## 8. Test coverage

Test suite bao phủ khá tốt:

- Employee/HR routing;
- MES deterministic SQL và SQL Agent;
- parser/table chunking;
- MKAC retrieval;
- Research topics/cache/routing/upload preview;
- Report Agent/API/artifact store;
- Calendar/Gmail;
- translation;
- token budgets.

Các cụm có thể bắt đầu từ [tests/](tests/).

Khoảng trống chính:

- không có frontend test/lint script; [frontend/package.json:6-20](frontend/package.json#L6-L20) chỉ có `dev`, `build`, `preview`;
- ít route-level ASGI/lifespan test;
- normal SSE, Calendar và Gmail chưa có coverage end-to-end đầy đủ;
- thiếu test transaction/rollback cho Research upload;
- thiếu Qdrant outage semantics;
- thiếu auth/action perimeter;
- thiếu startup test cho MCP/Calendar;
- Coding Agent API chưa được bảo vệ bằng test tương xứng capability;
- một số Research evaluation scripts là report generator, không phải pass/fail gate đáng tin.

---

## 9. Các điểm nóng ưu tiên

1. **Coding Agent có default source không an toàn** nếu khởi chạy ngoài Compose hiện tại.
2. **Research upload/Qdrant mutation không transactional**.
3. **Employee ID là access gate yếu** nhưng có thể mở Gmail/Calendar side effects.
4. **Nhiều endpoint session/preview/report không có application-level authorization**.
5. **Qdrant error thường bị biến thành zero hit**, dễ nhầm outage với “không có dữ liệu”.
6. **`/health` chủ yếu là process liveness**, không chứng minh model, retrieval hay external integration hoạt động.
7. **REST và SSE có routing order khác nhau**.
8. **Cache key/freshness chưa phản ánh đầy đủ conversation và corpus version**.
9. **Research registry/env/indexer có thể drift**.
10. **Vite proxy có nguy cơ trỏ Dev frontend sang Production API**.
11. **State per-process không phù hợp nếu tăng nhiều Uvicorn worker**.
12. Hai “god file” lớn nhất vẫn là [src/api/main.py](src/api/main.py) và [frontend/src/main.jsx](frontend/src/main.jsx).
13. `RAGPipeline` còn giữ MES compatibility branches trong khi API đã có `MesQueryService` riêng, tạo duplication và một số đường khó tiếp cận.

---

## 10. Thứ tự đọc đề xuất

Nếu cần onboard nhanh, nên đọc theo thứ tự:

1. [AGENTS.md](AGENTS.md) và [CLAUDE.md](CLAUDE.md).
2. [Dockerfile](Dockerfile).
3. [docker-compose.dev.yml](docker-compose.dev.yml) và [docker-compose.web.yml](docker-compose.web.yml).
4. [src/api/config.py](src/api/config.py).
5. [src/api/schemas.py](src/api/schemas.py).
6. [src/api/main.py:130-350](src/api/main.py#L130-L350).
7. [src/api/main.py:1750-2449](src/api/main.py#L1750-L2449).
8. [src/rag/rag_pipeline.py](src/rag/rag_pipeline.py).
9. [src/rag/vector_store.py](src/rag/vector_store.py).
10. [src/integrations/mes_query_service.py](src/integrations/mes_query_service.py).
11. [src/auth/employee_directory.py](src/auth/employee_directory.py).
12. [src/actions/](src/actions/).
13. [frontend/src/main.jsx](frontend/src/main.jsx).
14. [tests/](tests/).

## Phạm vi xác minh

- Không có file nào được tôi chỉnh sửa.
- Không chạy test/build vì yêu cầu là đọc hiểu, không thay đổi sản phẩm.
- Không chạy Docker hoặc truy cập runtime Dev/Production.
- Các kết luận trên là **static analysis đã đối chiếu chéo source, config và tests**.
- Trạng thái container, resolved mounts, Qdrant schema thực, model availability và hành vi end-to-end chưa được xác minh runtime.

## Agent đã DONE (kết quả rút gọn)

### map:api-runtime (`a2f0c3659f4ef40a4`)
- model: tier1 · tools: 51 · tokens: 138096
- preview: {"subsystem":"FastAPI application assembly và runtime request handling của Meibook API Gateway","summary":"SỰ THẬT ĐÃ XÁC MINH: Runtime chính là một FastAPI monolith trong `src/api/main.py`: Docker khởi chạy `uvicorn src.api.main:app` (`Dockerfile:53`), còn khối chạy trực tiếp dùng cùng import string (`src/api/main.py:2546`). `src/main.py` chỉ chứa token `main` ở dòng 1 và không lắp ráp hay export…

### map:rag-core (`ab4e195d4c2292227`)
- model: tier1 · tools: 48 · tokens: 103225
- preview: {"subsystem":"RAG core (`src/rag`) — nghiên cứu kiến trúc chỉ đọc trên checkout `/home/jkl/Code/VLLM-PD-dev`, nhánh `dev`","summary":"SỰ KIỆN ĐÃ XÁC MINH: RAG core gồm bảy module: embedding BGE-M3, parser/chunker page-aware, Qdrant vector store, prompt builder đa phương thức, pipeline điều phối query/generation, bộ phân giải đường dẫn preview và DDGS web fallback (`src/rag/embedder.py:19`, `src/ra…

### map:research (`a4527ba52093f6b6c`)
- model: tier1 · tools: 61 · tokens: 158126
- preview: {"subsystem":"Research / NotebookLM-like document QA","summary":"Đã khảo sát tĩnh, chỉ đọc trên nhánh `dev`; không sửa file, không gọi Docker/Qdrant/API runtime và không đọc dữ liệu nghiệp vụ. Subsystem Research có hai corpus tách biệt: (1) curated-topic dùng collection Qdrant `docjp_knowledge`, session cố định `docjp`, lọc `metadata.category` theo registry bốn chủ đề; (2) upload dùng collection `…

### map:hr-auth (retry 1) (`a8dc865e30dcf7d9b`)
- model: tier1 · tools: 120 · tokens: 157238
- preview: {"subsystem":"Cổng truy cập bằng mã nhân viên và luồng hỏi đáp HR/MKAC có cấu trúc","summary":"Kết luận đã xác minh bằng đọc tĩnh: đây là một access gate dựa trên việc mã nhân viên tồn tại, không phải cơ chế authentication theo nghĩa chứng minh danh tính. Runtime chỉ kiểm tra chuỗi 6 chữ số trong SQLite, hoặc chấp nhận guest cố định `000000`; không có mật khẩu, phiên đăng nhập phía server, cookie,…

### map:mes (`a66e8b2a43f36a7aa`)
- model: tier1 · tools: 42 · tokens: 157358
- preview: {"subsystem":"MES query stack: API mode routing, deterministic SQLite intents, time SQL, live MES max-Lot API, validated LLM Text-to-SQL fallback, and answer verification","summary":"Verified from source only; no files were edited, no services/data were accessed, and no tests were executed. The active API path is `/query` or `/query/stream` with `mode=\"mes\"`, which authenticates an employee, res…

### map:actions (retry 2) (`a1ae9734df566763a`)
- model: tier1 · tools: 155 · tokens: 141782
- preview: {"subsystem":"External actions and generated artifacts (read-only architecture map)","summary":"Verified facts: The main action surface is multiplexed through POST /query and POST /query/stream rather than separate Gmail/Calendar/report mutation routes (/home/jkl/Code/VLLM-PD-dev/src/api/main.py:1750,1951). Routing order after authorization/localization is Calendar action, MES report, then Gmail s…

### map:agent-mcp (retry 1) (`a6adf85c385e9e8f7`)
- model: tier1 · tools: 79 · tokens: 61085
- preview: {"subsystem":"Coding Agent LangGraph + shared MCP/Google Calendar integration","summary":"Verified read-only study on `/home/jkl/Code/VLLM-PD-dev` branch `dev`; no files were edited and no runtime, Docker, external Calendar, or sensitive-data command was executed. The subsystem has two distinct consumers of the same cached MCP discovery layer: (1) the optional Coding Agent exposed at `POST /agent`…

### map:frontend (`a43ef9042fa8290c3`)
- model: tier1 · tools: 50 · tokens: 131235
- preview: {"subsystem":"Frontend React/Vite của Meibook — nghiên cứu kiến trúc chỉ đọc trên nhánh dev; không chỉnh sửa file, không chạy ứng dụng, không gọi API và không đọc dữ liệu nghiệp vụ nhạy cảm.","summary":"ĐÃ XÁC MINH: Đây là SPA React 18 dùng Vite 6, không có router hay thư viện quản lý state toàn cục. `createRoot(...).render(<StrictMode><App /></StrictMode>)` khởi động toàn bộ ứng dụng từ `frontend…

### map:deployment (`a40aabe958f12a1b6`)
- model: tier1 · tools: 45 · tokens: 76794
- preview: {"subsystem":"Meibook build, deployment, and runtime topology","summary":"Nghiên cứu chỉ-đọc tại `/home/jkl/Code/VLLM-PD-dev`, nhánh `dev`; không sửa file, không chạy Docker/Compose, không đọc `.env`/`.env.docker`, token, log hay dữ liệu nghiệp vụ. Topology chính là React/Vite được build thành static assets, FastAPI/Uvicorn vừa phục vụ API/SSE vừa mount SPA, Qdrant lưu vector, và LiteLLM làm gatew…

### map:ingestion-ops (`abe46e5030a8c388f`)
- model: tier1 · tools: 61 · tokens: 145482
- preview: {"subsystem":"Bản đồ các script vận hành Meibook: OCR/tách văn bản, index Qdrant, import SQLite nhân sự/MES, đánh giá Research, deploy/start/ngrok và guardrail an toàn","summary":"Đã khảo sát chỉ đọc checkout `/home/jkl/Code/VLLM-PD-dev` trên nhánh `dev`; không sửa file, không chạy Docker/import/index/evaluation, không đọc `.env`, token, SQLite, Qdrant payload, raw dump, log hay nội dung tài liệu …

### map:tests (`ab813cfefa29ff44b`)
- model: tier1 · tools: 58 · tokens: 99835
- preview: {"subsystem":"Bản đồ kiến trúc test suite của Meibook (nghiên cứu chỉ đọc)","summary":"Đã đọc cấu hình pytest và toàn bộ 27 tệp có tên test được tìm thấy, không sửa file, không chạy test, không truy cập secret hay dữ liệu nghiệp vụ. Suite pytest chính được cấu hình chỉ thu thập từ hai root: `/home/jkl/Code/VLLM-PD-dev/tests` và `/home/jkl/Code/VLLM-PD-dev/Notebooks/Test_Code_Editor/tests`; project…

### map:config-data (retry 1) (`ac193a333923ca9f9`)
- model: tier1 · tools: 81 · tokens: 136702
- preview: {"subsystem":"Bản đồ cấu hình và artifact dữ liệu của Meibook (khảo sát chỉ đọc trên /home/jkl/Code/VLLM-PD-dev, nhánh dev)","summary":"Đã xác minh cấu trúc bằng mã nguồn và danh sách Git, không đọc SQLite/Qdrant/log/runtime payload, không đọc secret, không chạy Docker và không sửa file. Hệ thống có bốn lớp artifact chính: (1) JSON cấu hình được Git track trong /home/jkl/Code/VLLM-PD-dev/config; (…

### map:dependencies (`aa8e9a8c21d67aa16`)
- model: tier1 · tools: 94 · tokens: 174548
- preview: {"subsystem":"Meibook (VLLM-PD) dependency and subsystem map — Python + Node","summary":"Meibook is a Docker-Compose app: a FastAPI backend (src/) serving RAG (HR/MKAC + Research/DocJP), a deterministic+LLM MES query subsystem, external actions (Gmail send, Google Calendar via a Node MCP server, MES report agent), and an optional LangGraph coding agent. A Vite/React SPA (frontend/) is prebuilt to …

### map:docs-history (`a000c482831a1d80b`)
- model: tier1 · tools: 82 · tokens: 189529
- preview: {"subsystem":"Meibook (VLLM-PD) — bản đồ kiến trúc read-only tại checkout /home/jkl/Code/VLLM-PD-dev, nhánh dev","summary":"Đã đối chiếu tài liệu, 20 commit gần nhất, cây mã nguồn và các file topology local mà không chạy Docker, service, test, truy vấn DB/Qdrant hay đọc secret/dữ liệu nghiệp vụ. Thứ tự thẩm quyền trong phiên này là: chỉ thị người dùng; /home/jkl/Code/VLLM-PD-dev/CLAUDE.md:3 (inclu…

### verify:mes (`ae26dad89bc05865f`)
- model: tier1 · tools: 62 · tokens: 144427
- preview: {"verdict":"needs-corrections","confirmedEvidence":["Luồng Q&A MES thông thường đúng là tách khỏi document RAG: `route_query` và `route_query_stream` chuyển `mode=\"mes\"` sang `MesQueryService`, còn `mkac|research` mới gọi `RAGPipeline` (/home/jkl/Code/VLLM-PD-dev/src/api/main.py:857-914).","Cascade chính của `MesQueryService.query` được mô tả đúng: route snapshot/live trước, fixed time SQL, fixe…

### verify:rag-core (`ab2b9b509b00959d9`)
- model: tier1 · tools: 60 · tokens: 147456
- preview: {"verdict":"needs-corrections","confirmedEvidence":["Phần lõi đang được triển khai đúng là bảy module dưới `/home/jkl/Code/VLLM-PD-dev/src/rag/`: embedder, parser, vector store, prompts, pipeline, media resolver và web search. Composition root của gateway dựng ba `VectorStore`, một `Embedder`, một `DocumentParser`, một `WebSearcher` và một `RAGPipeline` tại `/home/jkl/Code/VLLM-PD-dev/src/api/main…

### verify:frontend (`a98b6db218568bc51`)
- model: tier1 · tools: 81 · tokens: 135838
- preview: {"verdict":"needs-corrections","confirmedEvidence":["Đã xác minh cấu trúc cốt lõi: React 18/Vite 6 SPA, mount bằng `createRoot(...).render(<StrictMode><App /></StrictMode>)` tại `frontend/src/main.jsx:2960-2964`; `App` bắt đầu tại `frontend/src/main.jsx:1031` và giữ phần lớn state/effect/orchestration. Không thấy router, Context hay thư viện state toàn cục trong `frontend/src`; dependencies trực t…

### verify:tests (`a7ef27ebf44b98f57`)
- model: tier1 · tools: 81 · tokens: 157558
- preview: {"verdict":"needs-corrections","confirmedEvidence":["Cấu hình discovery được mô tả đúng: khi gọi pytest không kèm path, `testpaths` chỉ gồm `tests` và `Notebooks/Test_Code_Editor/tests`; project root và Code Editor root được thêm vào `pythonpath` tại `/home/jkl/Code/VLLM-PD-dev/pytest.ini:1-7`. Có 22 tệp `test_*.py` trong root thứ nhất và 2 tệp trong root thứ hai.","Nhận định về nested fixture đún…

### verify:deployment (retry 1) (`a06e1112d872cb045`)
- model: tier1 · tools: 114 · tokens: 146822
- preview: {"verdict":"needs-corrections","confirmedEvidence":["Phần lõi của bản đồ là đúng: image app build Vite bằng Node 20 rồi chạy FastAPI/Uvicorn trên Python 3.10; artifact được copy vào `/app/frontend/dist`, healthcheck gọi cổng nội bộ 8001 và CMD chạy `src.api.main:app` (`/home/jkl/Code/VLLM-PD-dev/Dockerfile:1-9`, `/home/jkl/Code/VLLM-PD-dev/Dockerfile:44-53`).","Topology Compose chính được mô tả đú…

### verify:api-runtime (`a4e7d114d109bae3f`)
- model: tier1 · tools: 59 · tokens: 145962
- preview: {"verdict":"needs-corrections","confirmedEvidence":["Đã xác minh entry point nguồn chính: `/home/jkl/Code/VLLM-PD-dev/Dockerfile:48-53` expose/healthcheck cổng 8001 và khai báo `python -m uvicorn src.api.main:app`; `/home/jkl/Code/VLLM-PD-dev/src/api/main.py:2546-2549` dùng cùng import string khi chạy trực tiếp. Cả Compose Production và Dev không override command của service app (`/home/jkl/Code/V…

### verify:ingestion-ops (retry 1) (`a36379439f27df313`)
- model: tier1 · tools: 116 · tokens: 216833
- preview: {"verdict":"needs-corrections","confirmedEvidence":["Phân loại side effect tổng thể là đúng: OCR ghi Markdown (`scripts/ocr_docjp.py:135-173`, `scripts/ocr_vision_api.py:87-122`); ba indexer chạy thật tạo/đọc collection rồi prune/remove/upsert Qdrant và ghi report (`scripts/index_docjp_documents.py:107-228`, `scripts/index_mkac_documents.py:109-236`, `scripts/index_research_demo_documents.py:104-2…

### verify:agent-mcp (`aaa28bd3ae828f685`)
- model: tier1 · tools: 67 · tokens: 96615
- preview: {"verdict":"needs-corrections","confirmedEvidence":["Phần cốt lõi của map là đúng: `ENABLE_AGENT` chỉ điều khiển import và endpoint Coding Agent tại `/home/jkl/Code/VLLM-PD-dev/src/api/main.py:130-133,2477-2495`, trong khi lifespan luôn dựng `RAGPipeline` và `CalendarActionService` tại `/home/jkl/Code/VLLM-PD-dev/src/api/main.py:279-327`. Dev Compose đặt `ENABLE_AGENT: \"false\"` tại `/home/jkl/Co…

### verify:config-data (`a76b46e23a1523c12`)
- model: tier1 · tools: 60 · tokens: 114737
- preview: {"verdict":"needs-corrections","confirmedEvidence":["Đã xác nhận checkout khảo sát là `/home/jkl/Code/VLLM-PD-dev` trên nhánh `dev`; khảo sát không kết nối SQLite/Qdrant, không đọc payload runtime, không chạy Docker và không sửa repository.","Các số lượng cấu trúc chính là đúng: 4 Research topic, 12 cached answer, 18 MKAC quick answer, 7 MES quick answer, 2 Calendar room, 4 MES view, 3 relationshi…

## Agent còn progress khi bị kill

- **verify:research (retry 1)** state=progress lastTool=Bash summary=rg --files /home/jkl/Code/VLLM-PD-dev/src /home/jkl/Code/VL…
- **verify:dependencies (retry 2)** state=progress lastTool=Bash summary=git -C /home/jkl/Code/VLLM-PD-dev check-ignore -v docmind/b…
- **verify:docs-history (retry 1)** state=progress lastTool=Bash summary=git -C /home/jkl/Code/VLLM-PD-dev branch --show-current && …
- **verify:hr-auth (retry 1)** state=progress lastTool=Bash summary=git -C /home/jkl/Code/VLLM-PD-dev diff -- frontend/src/main…
- **verify:actions** state=progress lastTool=Bash summary=git -C /home/jkl/Code/VLLM-PD-dev ls-files src/agent/node_m…

## Cách tiếp tục khuyến nghị

1. **Không** cố resume task `wwuihyguq` / workflow cũ trong panel IDE.
2. Dùng file harvest này làm context.
3. Session mới hoặc resume rồi **gõ prompt mới** (bỏ qua recovery workflow), ví dụ:
   > Bỏ qua background workflow understand-meibook-repository đã bị kill.
   > Dựa trên survey/map agent đã done (file harvest), tiếp tục: ...
4. Chỉ relaunch Workflow với resumeFromRunId nếu thật sự muốn chạy nốt verify agents còn dở (tốn token, có thể stall lại).