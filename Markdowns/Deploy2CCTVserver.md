# Triển khai Meibook CCTVAI lên Server CCTV AI

> **Phiên bản tài liệu:** 2026-09-29
> **Mục đích:** Hướng dẫn triển khai hệ thống hỏi đáp Chatbot CCTVAI (module của Meibook) lên máy chủ CCTV AI gốc tại `192.86.201.99`.
> **Đối tượng:** Team vận hành server CCTV AI.
> **Trạng thái:** Bản nháp — chưa triển khai.

---

## 1. Tổng quan

Hệ thống hiện tại đang chạy trên máy chủ Meibook (Machine 2 — `192.84.106.87`), kết nối tới PostgreSQL read replica trên server CCTV AI (`192.86.201.99:55434`). Mục tiêu là **triển khai bản sao Meibook trực tiếp lên server CCTV AI** để:

- Giảm độ trễ mạng — database PostgreSQL nằm ngay trên localhost.
- Tự chủ hạ tầng — server CCTV AI hoạt động độc lập, không phụ thuộc máy Meibook.
- Phục vụ chatbot CCTVAI chuyên biệt cho team CCTV.

### 1.1 Ràng buộc phần cứng

| Thông số | Giá trị |
|---|---|
| GPU | **1× NVIDIA RTX PRO 5000** (Blackwell, 48 GB hoặc 72 GB GDDR7 — xác minh bằng `nvidia-smi`) |
| Mục tiêu | Host **đúng 1 model LLM** trên GPU duy nhất |
| Database | PostgreSQL CCTVAI **đã có sẵn** trên localhost (`127.0.0.1:55434`) |
| Host Metrics | Service `cctvai-host-metrics` đã/sẽ cài trên server này (xem `Markdowns/CCTVAI_HARDWARE.md`) |

> ⚠️ **Bắt buộc đo VRAM còn trống** khi các container CCTV AI đang chạy (`nvidia-smi`). Không giả định toàn bộ 48/72 GB đều khả dụng cho LLM.

---

## 2. Lựa chọn model LLM

### 2.1 Hệ thống hiện tại (Machine 2 — nhiều GPU)

| Model | Kích thước | Vai trò | GPU riêng |
|---|---|---|---|
| `qwen3:14b` | ~10-12 GB VRAM | Chat chung, diễn giải phần cứng (`local-hardware-summary`) | GPU 1 |
| `qwen2.5-coder:14b` | ~9-10 GB VRAM | SQL Agent cho CCTVAI/MES/WMS | GPU 2 (LAN) |
| `qwen2.5:3b-instruct` | ~2-3 GB VRAM | Dịch JA↔VI, phân loại intent | GPU 1 (cùng host) |

### 2.2 Khuyến nghị cho CCTV server (1 GPU)

**Chọn: `qwen2.5-coder:14b`** — model duy nhất trên Ollama.

| Lý do | Giải thích |
|---|---|
| **Bảo toàn chất lượng SQL** | Đây chính xác là model đã kiểm chứng cho nhánh SQL Agent — nơi sai lệch dễ dẫn tới trả lời sai số liệu. Thay model SQL là rủi ro lớn nhất. |
| **VRAM vừa đủ** | ~9-10 GB cho Q4 quantization, dư rộng trên RTX PRO 5000 (48-72 GB). |
| **Đa năng** | Qwen2.5-Coder 14B xử lý tốt cả text generation thông thường (diễn giải phần cứng, dịch ngắn), không chỉ viết code. |
| **Context window** | 32K token native — đủ cho mọi tác vụ CCTVAI. |

**Đánh đổi cần biết:**

| Tác vụ | Trên Machine 2 | Trên CCTV server | Ảnh hưởng |
|---|---|---|---|
| SQL Agent (CCTVAI DB) | `qwen2.5-coder:14b` ✅ | `qwen2.5-coder:14b` ✅ | **Không đổi** — giữ nguyên model |
| Diễn giải phần cứng | `qwen3:14b` | `qwen2.5-coder:14b` | Cần kiểm thử — chất lượng diễn giải có thể khác |
| Dịch JA↔VI | `qwen2.5:3b-instruct` (nhanh) | `qwen2.5-coder:14b` (chậm hơn) | Chậm hơn ~3-4× cho tác vụ dịch; hoặc tắt dịch nếu không cần JA |
| Chat chung | `qwen3:14b` | `qwen2.5-coder:14b` | Chấp nhận được cho câu hỏi CCTVAI |

---

## 3. Kiến trúc triển khai

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Server CCTV AI (192.86.201.99)                │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │ Ollama        │     │ LiteLLM      │     │ Qdrant       │    │
│  │ qwen2.5-coder │◀────│ Proxy        │     │ Vector DB    │    │
│  │ :14b          │     │ (port 4000)  │     │ (port 6333)  │    │
│  │ (port 11434)  │     └──────┬───────┘     └──────┬───────┘    │
│  │ GPU: RTX PRO  │            │                     │            │
│  │ 5000          │            │                     │            │
│  └──────────────┘     ┌──────┴─────────────────────┴───────┐    │
│                       │ Meibook App (FastAPI)                │    │
│                       │ Port: 8001                           │    │
│                       │ mode=cctvai                          │    │
│                       │ Embedder: BGE-M3 (CPU)               │    │
│                       └──────────┬───────────────────────────┘    │
│                                  │                                │
│  ┌──────────────┐     ┌─────────┴────────┐     ┌──────────────┐ │
│  │ PostgreSQL    │◀────│ psycopg (read-   │     │ host_metrics │ │
│  │ CCTVAI DB     │     │ only, localhost)  │     │ service      │ │
│  │ (port 55434)  │     └──────────────────┘     │ (port 8099)  │ │
│  └──────────────┘                                └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Thành phần:**

| Service | Vai trò | Chạy trên |
|---|---|---|
| **Ollama** | Serve model `qwen2.5-coder:14b` trên GPU | Host (systemd) |
| **Meibook App** | FastAPI backend + React frontend | Docker container |
| **LiteLLM Proxy** | Route model alias → Ollama local | Docker container |
| **Qdrant** | Vector DB cho embedding (RAG, Research) | Docker container |
| **PostgreSQL** | CCTVAI database (đã có sẵn) | Host (đã chạy) |
| **Host Metrics** | Đọc số liệu phần cứng server | Host (systemd, xem CCTVAI_HARDWARE.md) |

> **Không cần `ollama-proxy` (alpine/socat)** — Ollama chạy trực tiếp trên host, container gọi qua `host.docker.internal`.

---

## 4. Chuẩn bị trước khi triển khai

### 4.1 Kiểm tra phần cứng và phần mềm

```bash
# GPU — xác minh VRAM tổng và VRAM còn trống khi CCTV đang chạy
nvidia-smi

# Docker
docker --version
docker compose version

# Python (cho host_metrics service, không cho Meibook app)
python3 --version

# Cổng đã dùng — xác minh 8001, 4000, 6333, 11434 đều trống
ss -tlnp | grep -E '8001|4000|6333|11434'

# Disk — cần ~25 GB cho Docker images + model + Qdrant data
df -h /

# PostgreSQL CCTVAI đang chạy
psql -h 127.0.0.1 -p 55434 -U cctvai_llm_ro -d cctvai -c "SELECT count(*) FROM cctvai.event_snapshots;"
```

### 4.2 Cài đặt Ollama

```bash
# Cài Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model — ~9 GB download
ollama pull qwen2.5-coder:14b

# Xác minh model chạy được trên GPU
ollama run qwen2.5-coder:14b "SELECT 1;" --verbose
# Kiểm tra output: `eval duration` cho thấy GPU inference

# Cấu hình Ollama lắng nghe trên 0.0.0.0 (để Docker container gọi được)
sudo systemctl edit ollama.service
```

Thêm vào override:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_CONTEXT_LENGTH=16384"
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama

# Xác minh Ollama lắng nghe
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

---

## 5. Triển khai Docker Compose

### 5.1 Cấu trúc thư mục

```bash
# Tạo thư mục triển khai
sudo mkdir -p /opt/meibook-cctvai
cd /opt/meibook-cctvai

# Copy từ repo (hoặc clone)
# Cần: Dockerfile, Dockerfile.litellm, frontend/, src/, config/, scripts/,
#       database/schema/, tools/, requirements.txt, litellm_config.yaml
```

### 5.2 File `docker-compose.cctvai.yml`

Tạo file Compose riêng cho server CCTV AI — **không dùng `docker-compose.web.yml` của Machine 2**:

```yaml
# docker-compose.cctvai.yml
# Meibook CCTVAI — triển khai trên server CCTV AI (192.86.201.99)
# 1 GPU (RTX PRO 5000), 1 model LLM (qwen2.5-coder:14b via Ollama host)

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    image: meibook-cctvai:latest
    container_name: meibook-cctvai
    # KHÔNG có gpus: all — embedding chạy CPU, LLM chạy qua Ollama host
    env_file:
      - .env.cctvai
    environment:
      # --- Qdrant (container nội bộ) ---
      QDRANT_HOST: qdrant
      QDRANT_PORT: "6333"

      # --- LiteLLM (container nội bộ) ---
      LITELLM_URL: http://litellm:4000/v1

      # --- CCTVAI Database (PostgreSQL localhost qua host network) ---
      CCTVAI_DATABASE_ENABLED: "true"
      CCTVAI_DB_HOST: host.docker.internal
      CCTVAI_DB_PORT: "55434"
      CCTVAI_DB_NAME: cctvai
      CCTVAI_DB_USER: cctvai_llm_ro
      # CCTVAI_DB_PASSWORD đặt trong .env.cctvai (không commit)
      CCTVAI_DB_CONNECT_TIMEOUT: "3"
      CCTVAI_DB_STATEMENT_TIMEOUT_MS: "5000"
      CCTVAI_HEALTH_TTL_SECONDS: "30"
      CCTVAI_HEALTH_BREAKER_THRESHOLD: "3"
      CCTVAI_HEALTH_BREAKER_COOLDOWN_SECONDS: "60"
      CCTVAI_MAX_ROWS: "50"

      # --- CCTVAI SQL Agent ---
      CCTVAI_SQL_AGENT_ENABLED: "true"
      CCTVAI_SEMANTIC_MODEL_PATH: /app/config/cctvai_semantic_model.json
      CCTVAI_SQL_AGENT_MAX_ROWS: "50"
      CCTVAI_SQL_AGENT_MAX_ATTEMPTS: "2"
      CCTVAI_SQL_AGENT_MODEL: local-qwen-coder
      CCTVAI_SQL_PLANNER_MAX_TOKENS: "1200"
      CCTVAI_SQL_ANSWER_MAX_TOKENS: "384"

      # --- CCTVAI Hardware Metrics ---
      CCTVAI_HARDWARE_ENABLED: "true"
      CCTVAI_HARDWARE_URL: https://localhost:8099/metrics
      CCTVAI_HARDWARE_TOKEN_PATH: /app/data/cctvai_hardware_token
      CCTVAI_HARDWARE_CA_PATH: /app/data/cctvai_hardware_ca.pem
      CCTVAI_HARDWARE_TIMEOUT_SECONDS: "5"
      CCTVAI_HARDWARE_CACHE_SECONDS: "5"
      CCTVAI_HARDWARE_MAX_AGE_SECONDS: "30"
      CCTVAI_HARDWARE_MODEL: local-hardware-summary
      CCTVAI_HARDWARE_ANSWER_MAX_TOKENS: "384"

      # --- Embedding (CPU — GPU dành cho Ollama) ---
      EMBEDDING_DEVICE: cpu
      EMBEDDING_DTYPE: float32
      EMBEDDING_BATCH_SIZE: "4"
      DOCLING_DEVICE: cpu

      # --- LLM context ---
      LOCAL_CHAT_NUM_CTX: "16384"
      LOCAL_AUX_NUM_CTX: "4096"

      # --- Translation ---
      TRANSLATION_MODEL: local-qwen-small
      TRANSLATION_TEMPERATURE: "0.1"
      # Đặt TRANSLATION_ENABLED=false trong .env.cctvai nếu không cần JA

      # --- Module tắt (không cần trên server CCTV) ---
      ENABLE_AGENT: "false"
      GMAIL_SEND_ENABLED: "false"
      MES_DATABASE_ENABLED: "false"
      MES_SQL_AGENT_ENABLED: "false"
      MES_WMS_DATABASE_ENABLED: "false"
      MKAC_WEB_SEARCH_ENABLED: "false"

      # --- Token budget giữ nguyên ---
      MKAC_GENERAL_MAX_TOKENS: "256"
      MKAC_SIMPLE_MAX_TOKENS: "512"
      MKAC_EXTENDED_MAX_TOKENS: "768"
      RESEARCH_TOP_K: "6"
      RESEARCH_MAX_TOKENS: "768"
      RESEARCH_SCORE_THRESHOLD: "0.35"

      # --- Qdrant collections ---
      DOCJP_COLLECTION_NAME: docjp_knowledge
      DOCJP_SESSION_ID: docjp

      # --- Paths ---
      UPLOAD_DIR: /app/uploads
      RESEARCH_TOPICS_PATH: /app/config/research_topics.json
    ports:
      - "0.0.0.0:8001:8001"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - easyocr_cache:/root/.EasyOCR
      - hf_cache:/root/.cache/huggingface
      - torch_cache:/root/.cache/torch
      - ./config:/app/config
      - ./uploads:/app/uploads
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - qdrant
      - litellm
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: meibook-cctvai-qdrant
    ports:
      - "127.0.0.1:6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    restart: unless-stopped

  litellm:
    build:
      context: .
      dockerfile: Dockerfile.litellm
    image: meibook-cctvai-litellm:latest
    container_name: meibook-cctvai-litellm
    env_file:
      - .env.cctvai
    environment:
      LITELLM_LOCAL_MODEL_COST_MAP: "True"
      # Tất cả trỏ về Ollama trên host
      QWEN_CHAT_API_BASE: http://host.docker.internal:11434
      QWEN_CHAT_NGROK_API_BASE: ""
      QWEN_SMALL_API_BASE: http://host.docker.internal:11434
      QWEN_CODER_LAN_API_BASE: http://host.docker.internal:11434/v1
      QWEN_CODER_LAN_API_KEY: sk-local
      QWEN_CODER_NGROK_API_BASE: ""
      QWEN_CODER_NGROK_API_KEY: ""
    extra_hosts:
      - "host.docker.internal:host-gateway"
    command: ["--local", "--config", "/app/config.yaml", "--port", "4000", "--host", "0.0.0.0"]
    ports:
      - "127.0.0.1:4000:4000"
    volumes:
      - ./litellm_config_cctvai.yaml:/app/config.yaml:ro
    depends_on:
      - qdrant
    restart: unless-stopped

volumes:
  hf_cache:
  torch_cache:
  easyocr_cache:
```

### 5.3 File `litellm_config_cctvai.yaml`

Tạo cấu hình LiteLLM riêng — **tất cả alias trỏ về cùng 1 model `qwen2.5-coder:14b` trên Ollama host**:

```yaml
# litellm_config_cctvai.yaml
# Server CCTV AI — 1 Ollama, 1 model: qwen2.5-coder:14b
#
# QUAN TRỌNG: Provider phải đúng cho từng loại route:
#   - ollama_chat/ : dùng Ollama native API, hỗ trợ num_ctx, think
#   - openai/      : dùng Ollama OpenAI-compat /v1, KHÔNG hỗ trợ num_ctx
#
# QWEN_CHAT_API_BASE KHÔNG được kết thúc bằng /v1
# QWEN_CODER_LAN_API_BASE PHẢI kết thúc bằng /v1

model_list:
  # Chat routes — dùng ollama_chat provider (hỗ trợ num_ctx)
  - model_name: auto-model
    litellm_params:
      model: ollama_chat/qwen2.5-coder:14b
      api_base: os.environ/QWEN_CHAT_API_BASE
      timeout: 120
      think: false
      stream: true
      max_parallel_requests: 1
    model_info:
      max_input_tokens: 16384
      max_output_tokens: 4096
      input_cost_per_token: 0.0
      output_cost_per_token: 0.0

  - model_name: local-qwen-chat
    litellm_params:
      model: ollama_chat/qwen2.5-coder:14b
      api_base: os.environ/QWEN_CHAT_API_BASE
      timeout: 120
      think: false
      stream: true
      max_parallel_requests: 1
    model_info:
      max_input_tokens: 16384
      max_output_tokens: 4096
      input_cost_per_token: 0.0
      output_cost_per_token: 0.0

  # Small helper — cùng model, alias riêng cho translation/intent
  - model_name: local-qwen-small
    litellm_params:
      model: ollama_chat/qwen2.5-coder:14b
      api_base: os.environ/QWEN_SMALL_API_BASE
      timeout: 30
      stream: true
      max_parallel_requests: 2
    model_info:
      max_input_tokens: 32768
      max_output_tokens: 1024
      input_cost_per_token: 0.0
      output_cost_per_token: 0.0

  # Coder route — dùng openai/ provider (Ollama /v1 endpoint)
  - model_name: local-qwen-coder
    litellm_params:
      model: openai/qwen2.5-coder:14b
      api_base: os.environ/QWEN_CODER_LAN_API_BASE
      api_key: os.environ/QWEN_CODER_LAN_API_KEY
      timeout: 120
      stream: true
      max_parallel_requests: 1
    model_info:
      max_input_tokens: 16384
      max_output_tokens: 4096
      input_cost_per_token: 0.0
      output_cost_per_token: 0.0

  # Hardware summariser — local-only, KHÔNG có cloud fallback
  - model_name: local-hardware-summary
    litellm_params:
      model: ollama_chat/qwen2.5-coder:14b
      api_base: os.environ/QWEN_CHAT_API_BASE
      timeout: 60
      think: false
      stream: true
      max_parallel_requests: 1
    model_info:
      max_input_tokens: 16384
      max_output_tokens: 1024
      input_cost_per_token: 0.0
      output_cost_per_token: 0.0

router_settings:
  # Không có cloud fallback — server CCTV chạy hoàn toàn local
  # Nếu muốn thêm Azure fallback, thêm alias và key vào .env.cctvai
  fallbacks:
    - auto-model: ["local-qwen-chat"]
    - local-qwen-chat: []
    - local-qwen-small: ["local-qwen-chat"]
    - local-qwen-coder: []
  num_retries: 1
  timeout: 120
  routing_strategy: simple-shuffle

litellm_settings:
  global_max_parallel_requests: 4
```

> ⚠️ **`local-hardware-summary` cố ý KHÔNG có fallback cloud** — số liệu phần cứng server không được gửi qua Azure/OpenAI.

### 5.4 File `.env.cctvai`

```bash
# .env.cctvai — KHÔNG COMMIT FILE NÀY VÀO GIT
# Chỉ chứa secret và override cho server CCTV AI

# PostgreSQL CCTVAI
CCTVAI_DB_PASSWORD=<mật_khẩu_cctvai_llm_ro>

# LiteLLM master key (dùng nội bộ giữa app ↔ litellm container)
LITELLM_MASTER_KEY=sk-local

# Tắt dịch JA nếu không cần (tiết kiệm tài nguyên LLM)
TRANSLATION_ENABLED=false
# Nếu cần JA, đổi thành: TRANSLATION_ENABLED=true

# Cloud fallback (để trống = tắt)
OPENAI_API_KEY=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
```

### 5.5 File `.dockerignore` bổ sung

Thêm vào `.dockerignore` **trước khi build** trên server CCTV:

```
data/
documents/
GmailBot/
Markdowns/
docjp_processed/
mkac_processed/
*.patch
```

Lý do: các thư mục này không được `COPY` trong Dockerfile nhưng vẫn bị gửi làm build context cho Docker daemon (chậm + có thể chứa credential).

---

## 6. Xác thực (Auth)

### 6.1 Employee ID

API `/query` và `/query/stream` với `mode=cctvai` yêu cầu `employee_id` trong request body. Hệ thống chấp nhận:

- **Mã nhân viên thật** từ database nhân sự (`employee_directory.sqlite`).
- **Mã khách `000000`** — guest access, không cần danh sách nhân sự.

Đối với server CCTV AI, khuyến nghị **dùng mã khách `000000`** vì không cần import danh sách nhân sự MKAC. Frontend gửi:

```json
{
  "employee_id": "000000",
  "mode": "cctvai",
  ...
}
```

### 6.2 Không có API Key

Các endpoint `/query`, `/query/stream`, `/health`, `/sessions`, `/quick-answers` **không yêu cầu API key hay Bearer token**. Bảo vệ bằng:

- Rate limiting: 15 request/phút/IP (mặc định).
- Firewall: giới hạn IP truy cập port 8001.

---

## 7. Quy trình triển khai từng bước

### Bước 1: Chuẩn bị server

```bash
# Kiểm tra GPU, VRAM, Docker, cổng (mục 4.1)
nvidia-smi
docker compose version
ss -tlnp | grep -E '8001|4000|6333|11434'
df -h /
```

### Bước 2: Cài Ollama và pull model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:14b

# Cấu hình lắng nghe 0.0.0.0
sudo systemctl edit ollama.service
# Thêm: Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl daemon-reload
sudo systemctl restart ollama

# Xác minh
curl -s http://localhost:11434/api/tags
ollama ps  # kiểm tra model đã load vào GPU
```

### Bước 3: Chuẩn bị thư mục triển khai

```bash
sudo mkdir -p /opt/meibook-cctvai
cd /opt/meibook-cctvai

# Copy source code (KHÔNG copy data/, documents/, .env, qdrant_storage/)
# Cần: Dockerfile, Dockerfile.litellm, frontend/, src/, config/,
#       scripts/, database/schema/, tools/, requirements.txt
```

### Bước 4: Tạo file cấu hình

```bash
# Tạo docker-compose.cctvai.yml (nội dung mục 5.2)
# Tạo litellm_config_cctvai.yaml (nội dung mục 5.3)
# Tạo .env.cctvai (nội dung mục 5.4 — điền mật khẩu DB thật)
# Bổ sung .dockerignore (mục 5.5)
```

### Bước 5: Chuẩn bị dữ liệu runtime

```bash
mkdir -p data logs uploads config qdrant_storage

# Copy file cấu hình cần thiết
# config/cctvai_semantic_model.json — bắt buộc cho SQL Agent
# config/quick_answers.json — gợi ý câu hỏi
# config/research_topics.json — nếu dùng Research mode

# Hardware metrics token + CA cert (nếu host_metrics đã cài)
# cp /etc/cctvai-host-metrics/token data/cctvai_hardware_token
# cp /path/to/ca.pem data/cctvai_hardware_ca.pem
```

### Bước 6: Build và khởi động

```bash
cd /opt/meibook-cctvai

# Build images
docker compose -f docker-compose.cctvai.yml build

# Khởi động
docker compose -f docker-compose.cctvai.yml up -d

# Theo dõi log khởi động (90 giây đầu là load embedding model BGE-M3)
docker logs -f meibook-cctvai
```

### Bước 7: Xác minh

```bash
# Health check — chờ ~90 giây cho embedding model load xong
curl -s http://localhost:8001/health | python3 -m json.tool

# Kiểm tra các trường quan trọng:
# - status: "healthy"
# - cctvai_database.available: true
# - cctvai_database.enabled: true
# - cctvai_hardware.available: true (nếu host_metrics đã cài)

# Test LiteLLM
curl -s http://localhost:4000/health/liveliness

# Test Qdrant
curl -s http://localhost:6333/healthz

# Test hỏi đáp CCTVAI
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-001",
    "question": "Hôm nay có bao nhiêu sự kiện vi phạm?",
    "mode": "cctvai",
    "employee_id": "000000",
    "ui_language": "vi",
    "model": "auto",
    "conversation_context": []
  }'

# Test phần cứng (nếu host_metrics đã cài)
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-001",
    "question": "Kiểm tra tình trạng CPU và RAM server",
    "mode": "cctvai",
    "employee_id": "000000",
    "ui_language": "vi",
    "model": "auto",
    "conversation_context": []
  }'
```

---

## 8. Qdrant — Dữ liệu vector

### 8.1 CCTVAI có cần Qdrant không?

| Tính năng | Cần Qdrant? | Giải thích |
|---|---|---|
| CCTVAI Database Q&A | **Không** | Dùng deterministic SQL + SQL Agent, không dùng RAG |
| CCTVAI Hardware Q&A | **Không** | Dùng API metrics + LLM diễn giải, không dùng RAG |
| MKAC HR Q&A | **Có** | RAG trên collection `mkac_knowledge` |
| Research Document Q&A | **Có** | RAG trên collection `docjp_knowledge` |

Nếu server CCTV **chỉ chạy mode `cctvai`**, Qdrant vẫn cần khởi động (vì Meibook app khởi tạo VectorStore trong lifespan), nhưng **không cần import dữ liệu** — các collection sẽ trống và mode `cctvai` không truy vấn chúng.

Nếu muốn dùng thêm MKAC/Research trên server CCTV, cần chạy script import riêng (ngoài phạm vi tài liệu này).

---

## 9. Lưu ý quan trọng

### 9.1 Provider mismatch — `ollama_chat/` vs `openai/`

Đây là cạm bẫy lớn nhất khi cấu hình LiteLLM:

| Provider | API endpoint | `num_ctx` | `think` | Dùng cho |
|---|---|---|---|---|
| `ollama_chat/` | `http://host:11434` (KHÔNG có `/v1`) | ✅ Hỗ trợ | ✅ Hỗ trợ | Chat, translation, hardware |
| `openai/` | `http://host:11434/v1` (PHẢI có `/v1`) | ❌ Bị bỏ qua | ❌ Bị bỏ qua | SQL Agent (coder) |

**Nếu trỏ `QWEN_CHAT_API_BASE` tới `/v1` → `num_ctx` sẽ bị bỏ qua, model dùng context mặc định.**

### 9.2 Concurrent requests trên 1 GPU

Ollama chỉ có 1 model trên 1 GPU. Khi nhiều request đồng thời (ví dụ 1 câu SQL + 1 câu hardware), chúng sẽ xếp hàng tuần tự. `litellm_settings.global_max_parallel_requests: 4` giới hạn tổng request đang xử lý.

### 9.3 Embedding model load time

`BAAI/bge-m3` (~1.5 GB) load lần đầu mất 60-90 giây trên CPU. Dockerfile HEALTHCHECK có `start-period: 90s` để chờ. Lần chạy tiếp theo nhanh hơn nhờ Docker volume cache (`hf_cache`, `torch_cache`).

### 9.4 Database localhost vs Docker

PostgreSQL chạy trên host (`127.0.0.1:55434`). Từ trong container Docker, truy cập qua `host.docker.internal`. Cần:

- `extra_hosts: ["host.docker.internal:host-gateway"]` trong Compose.
- `CCTVAI_DB_HOST: host.docker.internal` (không phải `localhost` hay `127.0.0.1`).

### 9.5 Firewall

Chỉ mở port `8001` cho mạng nội bộ (frontend chatbot gọi tới). Các port khác (`4000`, `6333`, `11434`) chỉ lắng nghe `127.0.0.1` hoặc Docker internal network.

```bash
# Ví dụ: chỉ cho phép subnet nội bộ truy cập 8001
sudo ufw allow from 192.84.0.0/16 to any port 8001
sudo ufw allow from 192.168.0.0/16 to any port 8001
```

---

## 10. Vận hành

### 10.1 Lệnh thường dùng

```bash
cd /opt/meibook-cctvai

# Xem trạng thái
docker compose -f docker-compose.cctvai.yml ps

# Xem log
docker logs meibook-cctvai --tail 50
docker logs meibook-cctvai-litellm --tail 20

# Restart app (sau khi sửa config)
docker compose -f docker-compose.cctvai.yml restart app

# Recreate (sau khi đổi env/port/mount)
docker compose -f docker-compose.cctvai.yml up -d app

# Rebuild image (sau khi đổi source code/Dockerfile)
docker compose -f docker-compose.cctvai.yml build app
docker compose -f docker-compose.cctvai.yml up -d app

# Dừng toàn bộ
docker compose -f docker-compose.cctvai.yml down
```

### 10.2 Kiểm tra Ollama

```bash
# Model đã load chưa
ollama ps

# VRAM đang dùng
nvidia-smi

# Restart Ollama
sudo systemctl restart ollama
```

### 10.3 Rollback

Dừng Meibook không ảnh hưởng các container CCTV AI đang chạy:

```bash
docker compose -f docker-compose.cctvai.yml down
# Chỉ dừng Meibook stack — PostgreSQL, container CCTV AI, host_metrics vẫn chạy bình thường
```

---

## 11. Sự cố thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Container `meibook-cctvai` không start | `gpus: all` trong Compose nhưng không có NVIDIA runtime | Xoá `gpus: all` (Compose mẫu đã bỏ) |
| `/health` trả `cctvai_database.available: false` | Sai mật khẩu DB hoặc `host.docker.internal` không resolve | Kiểm tra `.env.cctvai`, thử `docker exec meibook-cctvai ping host.docker.internal` |
| LLM trả lời rất chậm hoặc timeout | Ollama chưa chạy hoặc model chưa load vào GPU | `ollama ps`, `nvidia-smi`, `systemctl status ollama` |
| Embedding model load 5+ phút | CPU yếu hoặc lần đầu download model | Chờ; kiểm tra `docker logs meibook-cctvai` tìm dòng "Loading BAAI/bge-m3" |
| SQL Agent trả "cannot answer" cho mọi câu | `litellm_config_cctvai.yaml` thiếu alias `local-qwen-coder` | Kiểm tra file config, `docker logs meibook-cctvai-litellm` |
| `num_ctx` không hoạt động | `api_base` kết thúc bằng `/v1` cho route `ollama_chat/` | Xoá `/v1` khỏi `QWEN_CHAT_API_BASE` |
| Port 8001 đã bị chiếm | Service khác đang dùng | `ss -tlnp \| grep 8001`, đổi port trong Compose |
| `TRANSLATION_ENABLED` mặc định `true` gây lỗi | App cố gọi model dịch nhưng alias/model không sẵn sàng | Đặt `TRANSLATION_ENABLED=false` trong `.env.cctvai` |

---

## 12. Checklist triển khai

- [ ] Kiểm tra GPU: `nvidia-smi` — xác minh tên GPU và VRAM còn trống
- [ ] Cài Ollama, pull `qwen2.5-coder:14b`, cấu hình `OLLAMA_HOST=0.0.0.0:11434`
- [ ] Kiểm tra PostgreSQL: `psql -h 127.0.0.1 -p 55434 -U cctvai_llm_ro -d cctvai`
- [ ] Tạo thư mục `/opt/meibook-cctvai`, copy source code
- [ ] Tạo `docker-compose.cctvai.yml`, `litellm_config_cctvai.yaml`, `.env.cctvai`
- [ ] Bổ sung `.dockerignore` trước build
- [ ] Tạo `data/`, `logs/`, `uploads/`, `config/`, `qdrant_storage/`
- [ ] Copy `config/cctvai_semantic_model.json`, `config/quick_answers.json`
- [ ] Nếu dùng host_metrics: copy token + CA cert vào `data/`
- [ ] `docker compose -f docker-compose.cctvai.yml build`
- [ ] `docker compose -f docker-compose.cctvai.yml up -d`
- [ ] Chờ ~90 giây, kiểm tra `curl http://localhost:8001/health`
- [ ] Test query CCTVAI database (mục 7, Bước 7)
- [ ] Test query hardware (nếu host_metrics đã cài)
- [ ] Cấu hình firewall — chỉ mở port 8001 cho mạng nội bộ
- [ ] Gửi Base URL (`http://192.86.201.99:8001`) cho team frontend

---

## 13. Tham khảo

| Tài liệu | Đường dẫn |
|---|---|
| API Frontend CCTVAI | `Markdowns/Handle_CCTVChatbot_API.md` |
| Host Metrics Runbook | `Markdowns/CCTVAI_HARDWARE.md` |
| Kiến trúc Meibook | `Markdowns/ARCHITECTURE.md` |
| LiteLLM Config gốc | `litellm_config.yaml` |
| Biến môi trường mẫu | `.env.docker.example` |
| CCTVAI Semantic Model | `config/cctvai_semantic_model.json` |
