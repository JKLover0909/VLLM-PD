# Kiến trúc hệ thống Meibook

> Trạng thái target ngày 2026-11-17: mọi alias LLM thông thường dùng chung remote
> Qwen primary, Azure chỉ fallback khi lỗi; hardware telemetry deterministic. Runtime
> chưa chuyển cho đến khi có HTTPS/VPN xác nhận. BGE/OCR vẫn local. Trạng thái runtime
> phải được xác minh theo checkout, Compose và container, không suy từ target này.

---

## 1. Năm chế độ hỏi đáp

| Mode | `mode=` | Nguồn dữ liệu | Ghi chú |
|---|---|---|---|
| Hỏi đáp hành chính nhân sự | `mkac` | SQLite nhân sự + Qdrant `mkac_knowledge` | Hỗ trợ VI/JP, ddgs web fallback |
| Quản lý sản xuất MES | `mes` | SQLite `data/mes.sqlite` (read-only) | Deterministic trước, SQL Agent fallback |
| Quản lý kho WMS | `wms` | SQLite `data/mes_wms.sqlite` (read-only) | Fail-closed, không fallback sang MES |
| Camera AI CCTVAI | `cctvai` | PostgreSQL replica `cctvai` schema (read-only) | Replica không realtime; Dev bật mặc định |
| Nghiên cứu tài liệu | `research` | Qdrant `docjp_knowledge` (theo topic) | Upload session dùng `docmind_documents` |

---

## 2. Kiến trúc tổng quan

```mermaid
graph TB
    Browser["🌐 Browser / React SPA"]
    GW["FastAPI Gateway\n8001 Production · 8002 Dev"]
    LLM["LiteLLM Proxy\n4002 Production · 4001 Dev (host)"]
    Ollama["Current Dev upstream: host Ollama\nRemote target not configured"]
    Qdrant["Qdrant\n6333 Production · 6334 Dev"]
    SQLite["SQLite\ndata/*.sqlite"]
    PG["PostgreSQL Replica\ncctvai schema · read-only"]
    Qwen["Target: remote Qwen3.5-9B Q4_K_M\nTLS/VPN pending; Dev runtime still Ollama"]
    Cloud["Azure gpt-5.4-mini\nsequential failure fallback"]
    Hardware["Deterministic hardware renderer"]

    Browser <-->|"HTTP/SSE"| GW
    GW --> LLM
    GW --> Qdrant
    GW --> SQLite
    GW --> PG
    LLM --> Ollama
    LLM -. future TLS/VPN rollout .-> Qwen
    Qwen -->|failure only| Cloud
    PG --> Hardware
    Hardware --> GW
```

---

## 3. Luồng xử lý theo mode

### 3.1 MKAC — Hỏi đáp nhân sự

```mermaid
flowchart LR
    Q["Query"] --> Auth["Auth\nemployee_id"]
    Auth --> Cache{"Cache\nhit?"}
    Cache -->|Có| R["Response"]
    Cache -->|Không| EI{"Employee\nintent?"}
    EI -->|Cấu trúc| DB["employee_directory\n.sqlite"]
    EI -->|RAG| EMB["BGE-M3\nEmbedder"]
    EMB --> QD["Qdrant\nmkac_knowledge"]
    DB --> LLM["LiteLLM\nlocal-qwen-chat"]
    QD --> LLM
    LLM --> I18N{"UI = JP?"}
    I18N -->|Có| TR["Dịch\nlocal-qwen-chat (Qwen3.5 9B)"]
    I18N -->|Không| R
    TR --> R
```

### 3.2 MES — Quản lý sản xuất

```mermaid
flowchart LR
    Q["Query"] --> Auth["Auth + safety"]
    Auth --> ER{"Executive\nReport?"}
    ER -->|Có| RPT["Report Agent\ndeterministic"]
    ER -->|Không| DET{"Deterministic\nintent?"}
    DET -->|Khớp| SQL["Parameterized\nSQL read-only"]
    DET -->|Không khớp| AGT["SQL Agent\nsqlglot + allowlist"]
    SQL --> LLM["LiteLLM"]
    AGT --> LLM
    RPT --> LLM
    LLM --> R["SSE / Response"]
```

### 3.3 WMS — Quản lý kho

```mermaid
flowchart LR
    Q["Query"] --> Auth["Auth"]
    Auth --> DET{"Deterministic\ncurrent-balance?"}
    DET -->|Khớp| SNAP["mes_wms.sqlite\ncontract v4"]
    DET -->|Không khớp| FB["Shared remote Qwen SQL fallback\n(low confidence)"]
    SNAP --> VAL["Validate\nsnapshot + scope"]
    VAL -->|Lỗi| SUP["SUPPRESSED\nfail-closed"]
    VAL -->|OK| SSE["SSE\nwms_verification"]
    FB --> SSE
```

> WMS không truy vấn MES snapshot, MES SQL Agent hoặc MES API trong bất kỳ
> trường hợp nào. Câu hỏi tiếng Nhật giữ nguyên để bảo toàn mã vật tư/công đoạn.

### 3.4 CCTVAI — Camera giám sát AI

```mermaid
flowchart LR
    Q["Query"] --> DET{"7 intent\ntất định?"}
    DET -->|Có| PSQL["Parameterized SQL\ndel_flag trong ON"]
    DET -->|Không| AGT{"SQL Agent\nbật?"}
    AGT -->|Không| REJ["Từ chối\nnêu phạm vi"]
    AGT -->|Có| GEN["LLM sinh SQL\nlocal-qwen-coder (Qwen3.5 9B)"]
    GEN --> VAL["sqlglot validate\n+ read-only + timeout"]
    VAL -->|Từ chối| RT["Thử lại\nrồi từ chối"]
    PSQL --> ANS["Trả lời"]
    VAL -->|OK| UNV["Trả lời\n+ nhãn CHƯA KIỂM CHỨNG"]
```

**Ba ràng buộc bắt buộc của CCTVAI:**

| Vấn đề | Nguyên nhân | Hậu quả nếu sai |
|---|---|---|
| `del_flag` trong `ON` của JOIN | `camera_id`/`violation_code` không unique trên toàn bảng | Tăng 43% số dòng — số liệu sai hoàn toàn |
| `detected_time` là epoch **ms** | Không phải giây | Lệch hàng nghìn năm |
| Liệt kê cột tường minh | Quyền cấp theo cột, `SELECT *` bị từ chối | Lỗi permission |

### 3.5 Research — Nghiên cứu tài liệu

```mermaid
flowchart LR
    Q["Query"] --> TS["Topic selector\n/research/topics"]
    TS -->|topic scope| DJ["Qdrant\ndocjp_knowledge\n+ filter category"]
    TS -->|upload scope| DM["Qdrant\ndocmind_documents\n+ filter session_id"]
    DJ --> LLM["LiteLLM"]
    DM --> LLM
    LLM --> R["Response + sources"]
```

---

## 4. Model routing

```mermaid
graph LR
    APP["App"] -->|"auto-model / local-qwen-chat"| QWEN["Target: remote Qwen3.5-9B Q4_K_M\nTLS/VPN pending; Dev runtime still Ollama"]
    QWEN -->|"primary failure only"| AZ["azure-chat-fallback\ngpt-5.4-mini"]

    APP -->|"translation default: local-qwen-chat"| QWEN
    APP -->|"image request"| QWEN

    APP -->|"local-qwen-coder / coding-model"| QWEN
    QWEN -->|"primary failure only"| AZC["azure-coder-fallback\ngpt-5.4-mini"]

    APP -->|"CCTVAI hardware"| HW["Deterministic renderer\nno LLM / no export"]
    LOCAL["BGE-M3 + Docling remain local"] --- APP
```

> **Dev (2026-11-17):** every LLM alias runs on one host Ollama model, `qwen3.5:9b`
> Q4_K_M (`think=false`, context 8192, one request at a time) via the Ollama bridge.
> Azure `gpt-5.4-mini` is only a sequential failure fallback. Production is not changed.
>
> Hardware metrics use the deterministic Meibook renderer and never enter an LLM
> request, translation prompt or later conversation history. The Qwen endpoint is not
> used for this data. BGE-M3 and Docling/OCR remain local. Ollama remains installed
> and active until both stacks are migrated, consumers audited and a separate stop
> approval granted; stopping it is not part of Dev preparation.

---

## 5. Cổng và môi trường

| Thành phần | Production | Dev |
|---|:---:|:---:|
| FastAPI + React SPA | `8001` | `8002` |
| LiteLLM Proxy (host) | `4002` (`LITELLM_PROXY_PORT`; container `4000`) | `4001` (container `4000`) |
| Qdrant | `6333` | `6334` |
| Compose file | `docker-compose.web.yml` | `docker-compose.dev.yml` |
| Checkout | `/home/jkl/Code/VLLM-PD` · `main` | `/home/jkl/Code/VLLM-PD-dev` · `dev` |
| WMS bật | ❌ | ✅ |
| CCTVAI bật | main (schema đã merge) | ✅ |

**Ranh giới dữ liệu:** SQLite, Qdrant storage, uploads, logs, credentials Dev và
Production là tách biệt hoàn toàn. Không copy/merge dữ liệu runtime giữa hai môi
trường.

---

## 6. Khởi động — thứ tự singleton

```mermaid
sequenceDiagram
    participant L as lifespan()
    participant V as VectorStore ×3
    participant E as Embedder BGE-M3
    participant P as DocumentParser
    participant W as WebSearcher
    participant R as RAGPipeline
    participant M as MesQueryService
    participant C as CctvaiQueryService
    L->>V: docmind · mkac_knowledge · docjp_knowledge
    L->>E: BAAI/bge-m3
    L->>P: Docling/PyMuPDF
    L->>W: ddgs
    L->>R: RAGPipeline
    L->>M: MesQueryService
    L->>C: CctvaiQueryService (nếu cấu hình)
```

Các singleton import-time: `EmployeeDirectory`, `MesDatabase`, `GmailSender`,
`TranslationService`, `CctvaiHardwareService`.

---

## 7. SSE event sequence

```mermaid
sequenceDiagram
    participant B as Browser
    participant G as Gateway
    B->>G: POST /query/stream
    G-->>B: status (received / routing / ...)
    G-->>B: sources
    G-->>B: meta
    G-->>B: token (×N)
    G-->>B: done
    Note over G,B: Executive Report chèn thêm<br/>agent_plan · tool_start/result · artifact
```

---

## 8. Bảo mật — tóm tắt

| Lớp | Biện pháp |
|---|---|
| Upload | Allowlist extension · giới hạn dung lượng/trang PDF · semaphore |
| MES/WMS SQLite | Read-only · parameterized query · SQL Agent qua sqlglot + authorizer |
| CCTVAI PostgreSQL | Role `cctvai_llm_ro` · quyền cấp theo cột · statement timeout |
| LiteLLM / Qdrant | Bind `127.0.0.1` trong Docker |
| Gmail | Scope send-only · token/credentials trong `data/` không commit |
| Coding Agent | Tắt mặc định `ENABLE_AGENT=false` |

**Giới hạn còn lại:** `employee_id` là access gate đơn giản (chưa có auth
user/tenant đầy đủ); rate limit/cache in-memory; CORS còn rộng.

---

## 9. Bố cục mã nguồn chính

```
src/
├── api/          main.py · schemas.py · config.py · sse.py
├── rag/          rag_pipeline.py · embedder.py · vector_store.py · parser.py
├── auth/         employee_directory.py · employee_intent.py
├── actions/      report_intent.py · report_agent.py · artifact_store.py
├── integrations/ mes_*.py · cctvai_*.py · gmail_sender.py
├── i18n/         translation.py
└── agent/        LangGraph Coding Agent (tắt trong Docker web)
tools/
└── host_metrics/ schemas.py  ← dùng chung với CCTVAI hardware collector
config/           quick_answers.json · mkac_manifest.json · *_semantic_model.json
database/         raw_mkac/ · schema/mes.sql · schema/mes_wms.sql
documents/        MKAC/ · MKAC-md/ · Research/DocJP/
Markdowns/        tài liệu dự án (xem README.md để tra cứu)
```

---

## 10. Vận hành nhanh

```bash
# Preflight Dev
cd /home/jkl/Code/VLLM-PD-dev && git branch --show-current   # dev
docker compose -f docker-compose.dev.yml config -q
docker compose -f docker-compose.dev.yml ps

# Liveness check Dev
curl -fsS http://localhost:8002/health | jq .
curl -fsS http://localhost:4001/health/liveliness
curl -fsS http://localhost:6334/healthz

# Liveness check Production
curl -fsS http://localhost:8001/health | jq .
```

> **Test:** `scripts/meibook-python -m pytest tests/ -q`
> — không dùng bare `python`/`pytest`; không chạy `docker-compose.web.yml` khi
> đang trong checkout Dev.

---

*Tài liệu liên quan: [`DATABASE.md`](DATABASE.md) · [`DEPLOY.md`](DEPLOY.md) ·
[`TestPrompt.md`](TestPrompt.md) · [`CCTVAI_HARDWARE.md`](CCTVAI_HARDWARE.md)*
