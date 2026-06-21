# Meibook - Commit đầu tiên: Hệ thống hỏi đáp tài liệu và Coding Agent

Meibook là hệ thống hỏi đáp tài liệu, nghiên cứu tài liệu và thử nghiệm Coding
Agent chạy trên Máy 2. Hệ thống chính phục vụ web React, API FastAPI, Qdrant
vector database và LiteLLM router trong cùng một repository.

Thư mục `docmind/` là phần demo tách riêng từ các thử nghiệm trước, không phải
đường chạy chính của repository hiện tại.

## Trạng thái hiện tại

- Web người dùng: React SPA, được FastAPI phục vụ tại `/`.
- API Gateway: FastAPI chạy ở cổng `8001`.
- Public URL cho máy hiện tại: ngrok expose cổng `8001`; script khởi động tự lấy
  URL mới và in ra terminal.
- LiteLLM: chạy nội bộ ở cổng `4000`, không cần public cho người dùng.
- Qdrant: chạy nội bộ ở cổng `6333`.
- Embedding: `BAAI/bge-m3` chạy trên Máy 2.
- Parser tài liệu: Docling.
- LLM upstream:
  - Gemma4 local trên Máy 1 qua Ollama/ngrok.
  - OpenAI GPT-5.4 mini.
  - Grok 4.20 Reasoning qua Azure.
- Chế độ `Nghiên cứu` luôn dùng Grok để xử lý tài liệu và ảnh.
- Coding Agent: LangGraph + MCP tools, endpoint `/agent` được bảo vệ bằng
  `AGENT_API_KEY`.

## Kiến trúc nhanh

```text
Người dùng / Máy 3
        |
        | HTTPS ngrok hoặc IP nội bộ, cổng 8001
        v
Máy 2: FastAPI + React
        |
        |-- Hỏi đáp MKAC -> mkac_knowledge -> LiteLLM
        |
        |-- Nghiên cứu -> Upload -> Docling/OCR -> docmind_documents
        |
        |-- /agent -> LangGraph -> MCP filesystem/git -> LiteLLM
        |
        v
Máy 2: LiteLLM, cổng 4000 nội bộ
        |
        |-- auto-model -> OpenAI -> fallback Grok -> fallback Gemma4 local
        |-- local-gemma -> Gemma4 local trên Máy 1
        |-- openai-model -> GPT-5.4 mini
        |-- grok-model -> Grok 4.20 Reasoning qua Azure
        |-- coding-model -> Gemma4 local -> fallback OpenAI
```

Chi tiết hơn xem [ARCHITECTURE.md](ARCHITECTURE.md).

## Thư mục chính

```text
.
|-- src/
|   |-- api/main.py          # FastAPI, REST/SSE, static React mount
|   |-- rag/
|   |   |-- parser.py        # Docling parser
|   |   |-- embedder.py      # BGE-M3 embedding
|   |   |-- vector_store.py  # Qdrant session/file/chunk storage
|   |   `-- rag_pipeline.py  # Retrieval + prompt + LiteLLM call
|   `-- agent/
|       |-- graph.py         # LangGraph Coding Agent
|       `-- mcp_client.py    # MCP filesystem/git tools
|-- frontend/                # React + Vite web app
|-- config/mkac_manifest.json
|-- scripts/index_mkac_documents.py
|-- docker-compose.yml       # Qdrant + LiteLLM cho cách chạy local/systemd
|-- docker-compose.web.yml   # Bản Docker web nội bộ
|-- litellm_config.yaml      # Model aliases và fallback routing
|-- requirements.txt         # Python dependencies cho hệ thống chính
|-- .env.example             # Mẫu cấu hình local/systemd
|-- .env.docker.example      # Mẫu cấu hình Docker web nội bộ
`-- docmind/                 # Demo/runtime tách riêng, không phải luồng chính
```

## Yêu cầu

- Linux trên Máy 2 hoặc máy chủ nội bộ.
- Docker và Docker Compose.
- Conda hoặc Python 3.10 environment nếu chạy không dùng Docker app.
- NVIDIA GPU nếu muốn chạy BGE-M3 và OCR nhanh trên CUDA.
- Node.js 20+ để build frontend khi chạy local/systemd.
- Máy 1 đang expose Ollama/Gemma4 qua `OLLAMA_API_BASE` nếu muốn dùng model
  local.

Repository hiện đang dùng conda env tên `docmind`, nhưng đây chỉ là tên môi
trường. Thư mục `docmind/` vẫn là demo riêng.

## Cài đặt môi trường local/systemd

Từ root repository:

```bash
cd /home/jkl0909/Code/llm/Meibook

conda create -n docmind python=3.10 -y
conda activate docmind
pip install --upgrade pip
pip install -r requirements.txt

# Cài Node.js nếu env chưa có npm/node.
conda install -n docmind -c conda-forge nodejs=20 -y
```

Build frontend:

```bash
cd /home/jkl0909/Code/llm/Meibook/frontend
npm install
npm run build
```

Giao diện web có ba chế độ màu: `Sáng`, `Tối` và `Theo hệ thống`. Nút theme
trên thanh công cụ dùng để chuyển chế độ; lựa chọn được lưu trong
`localStorage` của trình duyệt.

## Cấu hình `.env`

Tạo file `.env`:

```bash
cd /home/jkl0909/Code/llm/Meibook
cp .env.example .env
```

Các biến quan trọng:

```env
OLLAMA_API_BASE=https://your-machine-1-ollama-ngrok-url
OLLAMA_MODEL=gemma4:latest

OPENAI_API_KEY=...

AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/openai/v1
AZURE_OPENAI_DEPLOYMENT=grok-4-20-reasoning

LITELLM_URL=http://localhost:4000/v1
LITELLM_MASTER_KEY=sk-local

MACHINE2_API_HOST=0.0.0.0
MACHINE2_API_PORT=8001
MACHINE2_API_LOCAL_URL=http://localhost:8001
NGROK_RESERVED_DOMAIN=

QDRANT_HOST=localhost
QDRANT_PORT=6333
MKAC_COLLECTION_NAME=mkac_knowledge
MKAC_SOURCE_DIR=documents/MKAC
MKAC_MANIFEST_PATH=config/mkac_manifest.json
MKAC_SCORE_THRESHOLD=0.48

UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE_MB=25
MAX_DOCUMENT_PAGES=100
DOCUMENT_PROCESSING_TIMEOUT_SECONDS=300
UPLOAD_PROCESSING_CONCURRENCY=1
UPLOAD_QUEUE_SIZE=4
QUERY_RATE_LIMIT_PER_MINUTE=15
UPLOAD_RATE_LIMIT_PER_HOUR=10

EMBEDDING_DEVICE=cuda
EMBEDDING_DTYPE=float16
EMBEDDING_BATCH_SIZE=8
DOCLING_DEVICE=cuda
DOCLING_NUM_THREADS=4
DOCLING_OCR_LANGUAGES=vi,en
MKAC_INDEX_EMBEDDING_DEVICE=cpu

AGENT_API_KEY=replace_with_a_long_random_secret
WORKSPACE_DIR=/home/jkl0909/Code/llm
AGENT_REPOSITORY_DIR=/home/jkl0909/Code/llm/Meibook
```

Không commit `.env`. File này có thể chứa API key thật.

## Chạy hệ thống local/systemd

### 1. Khởi động Qdrant và LiteLLM

```bash
cd /home/jkl0909/Code/llm/Meibook
docker compose up -d
docker compose ps
```

Kiểm tra LiteLLM:

```bash
curl http://localhost:4000/health/liveliness

KEY=$(sed -n 's/^LITELLM_MASTER_KEY=//p' .env)
curl -H "Authorization: Bearer $KEY" http://localhost:4000/v1/models
```

### 2. Build React

```bash
cd /home/jkl0909/Code/llm/Meibook/frontend
npm install
npm run build
```

FastAPI chỉ mount web khi `frontend/dist` tồn tại.

### 3. Chạy FastAPI cổng 8001

Chạy foreground để debug:

```bash
cd /home/jkl0909/Code/llm/Meibook
conda activate docmind
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

Hoặc dùng service đã cấu hình trên Máy 2:

```bash
systemctl --user status meibook-api
systemctl --user restart meibook-api
journalctl --user -u meibook-api -n 120 --no-pager
```

Máy hiện tại đã bật `loginctl enable-linger`, nên user service có thể chạy sau
khi logout.

### 4. Public bằng ngrok

Nếu dùng ngrok Free, để ngrok tự sinh URL random cho cổng `8001`:

```bash
ngrok http 8001
```

Nếu có reserved/static domain của ngrok trả phí, đặt `NGROK_RESERVED_DOMAIN` để
script đưa domain này vào tham số `ngrok --url`.

Chỉ cần public cổng `8001`. Không public LiteLLM cổng `4000` nếu không có lý do
riêng.

## Triển khai Docker web nội bộ

Hướng dẫn triển khai Docker web-only cho máy chủ trong mạng công ty nằm ở
[DEPLOY.md](DEPLOY.md). Chế độ này không chạy ngrok và không bật Coding Agent.

## Sử dụng web

Mở:

```text
http://localhost:8001
```

Hoặc public URL được `./scripts/meibook.sh start` in ra terminal.

Người dùng có thể:

- Tạo phiên mới.
- Chọn `Hỏi đáp MKAC` để tra cứu kho tài liệu nội bộ dùng chung.
- Chọn `Nghiên cứu` để upload PDF, DOCX, XLSX, PPTX, HTML, PNG, JPG, JPEG.
- Hai chế độ dùng session và lịch sử hỏi đáp riêng. Khi chuyển tab, frontend
  chuyển sang UUID của chế độ đó; tài liệu upload chỉ gắn với session
  `Nghiên cứu`.
- Chế độ `Nghiên cứu` yêu cầu có tài liệu đã index trước khi hỏi; có thể thêm
  tài liệu từ nút chính giữa màn hình, thanh bên hoặc nút đính kèm cạnh ô nhập.
- Panel nguồn mặc định thu gọn; nút nguồn hiển thị số trích dẫn của câu trả lời
  mới nhất và chỉ mở khi người dùng cần đối chiếu.
- Chế độ `Hỏi đáp MKAC` chỉ hiển thị hai lựa chọn `Cloud Model` và `Local Model`; mặc định là `Cloud Model`.
- Chế độ `Nghiên cứu` luôn dùng model nghiên cứu Grok ở backend.
- Đặt câu hỏi và xem sources từ tài liệu.

### Index kho tài liệu MKAC

Tài liệu được quản lý bởi `config/mkac_manifest.json` và index theo từng trang:

```bash
python scripts/index_mkac_documents.py --dry-run
python scripts/index_mkac_documents.py
```

Chạy lại một file:

```bash
python scripts/index_mkac_documents.py \
  --file "Quy định giờ làm thêm.pdf" \
  --reindex
```

PDF scan được OCR theo trang, ảnh trang được lưu trong `mkac_processed/` và
không commit Git.

## Các API endpoint

| Phương thức | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Kiểm tra API và cấu hình Qdrant |
| `GET` | `/models` | Danh sách model cho frontend |
| `GET` | `/knowledge/mkac/status` | Trạng thái kho tri thức MKAC |
| `POST` | `/sessions` | Tạo session RAG |
| `GET` | `/sessions/{session_id}` | Lấy thông tin session |
| `DELETE` | `/sessions/{session_id}` | Xóa session và file upload |
| `POST` | `/sessions/{session_id}/upload` | Upload và index tài liệu |
| `DELETE` | `/sessions/{session_id}/files/{filename}` | Xóa một file trong session |
| `POST` | `/query` | Hỏi đáp non-streaming |
| `POST` | `/query/stream` | Hỏi đáp SSE streaming |
| `POST` | `/agent` | Coding Agent, cần `X-Agent-API-Key` |

Request query mẫu:

```bash
SESSION_ID=$(curl -fsS -X POST http://localhost:8001/sessions | jq -r .session_id)

curl -fsS http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"question\":\"Quy định làm thêm giờ tại MKAC như thế nào?\",
    \"model\":\"openai\",
    \"mode\":\"mkac\",
    \"stream\":false
  }" | jq .
```

Upload mẫu:

```bash
curl -fsS -X POST \
  "http://localhost:8001/sessions/$SESSION_ID/upload" \
  -F "file=@documents/test1.pdf" | jq .
```

Agent mẫu:

```bash
AGENT_KEY=$(sed -n 's/^AGENT_API_KEY=//p' .env)

curl -fsS http://localhost:8001/agent \
  -H "X-Agent-API-Key: $AGENT_KEY" \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"task\":\"Không gọi công cụ. Chỉ trả lời đúng một từ: OK\"
  }" | jq .
```

## Định tuyến model

LiteLLM aliases trong `litellm_config.yaml`:

| Lựa chọn UI/API | Model group LiteLLM | Backend |
|---|---|---|
| `Cloud Model` / `openai` | `openai-model` | OpenAI GPT-5.4 mini |
| `Local Model` / `local` | `local-gemma` | Ollama/Gemma4 trên Máy 1 |
| Research Model / `grok` | `grok-model` | Grok 4.20 Reasoning qua Azure, dùng riêng cho `Nghiên cứu` |
| `auto` | `auto-model` | Route kỹ thuật dự phòng: OpenAI, fallback Grok, fallback Gemma4 local |
| Agent | `coding-model` | Gemma4 local, fallback OpenAI |

Trong mode `mkac`, retrieval dùng collection `mkac_knowledge`. Nếu không có
chunk đạt ngưỡng, backend tìm trên web với câu hỏi gắn thêm ngữ cảnh MKAC. Kết
quả có `answer_scope="web"` và kèm URL để kiểm chứng; thông tin này không được
coi là quy định nội bộ MKAC.

Tìm web dùng `ddgs` và không cần API key. Có thể tắt hoặc điều chỉnh trong
`.env` bằng `MKAC_WEB_SEARCH_ENABLED`, `MKAC_WEB_SEARCH_CONTEXT`,
`MKAC_WEB_SEARCH_REGION`, `MKAC_WEB_SEARCH_MAX_RESULTS` và
`MKAC_WEB_SEARCH_TIMEOUT`.

Nếu câu hỏi nhắc đến bảng, hình, sơ đồ hoặc biểu đồ và chunk có ảnh trang,
backend route sang `openai-model` để dùng Vision. Mode `research` vẫn tự động
dùng Vision cho file ảnh upload.

## Bảo mật và giới hạn

- `.env` bị ignore bởi Git.
- `/agent` yêu cầu header `X-Agent-API-Key` nếu `AGENT_API_KEY` được set.
- Upload chỉ nhận extension cho phép.
- Tên file và session ID được validate để tránh path traversal.
- Upload giới hạn mặc định 25 MB/file.
- Query giới hạn mặc định 15 request/IP/phút.
- Upload giới hạn mặc định 10 request/IP/giờ.
- MCP filesystem tool chỉ được phép truy cập `WORKSPACE_DIR`.
- Git MCP tool khóa vào `AGENT_REPOSITORY_DIR`.

## Kiểm thử nhanh

```bash
cd /home/jkl0909/Code/llm/Meibook

# Kiểm tra cú pháp Python.
conda run -n docmind python -m py_compile \
  src/api/main.py \
  src/rag/rag_pipeline.py \
  src/agent/graph.py \
  src/agent/mcp_client.py

# Build frontend.
cd frontend
conda run -n docmind npm run build

# Health local.
curl -fsS http://localhost:8001/health | jq .

# Health public qua ngrok.
PUBLIC_URL=$(curl -fsS http://localhost:4040/api/tunnels \
  | jq -r '.tunnels[] | select(.proto == "https") | .public_url')
curl -fsS -H 'ngrok-skip-browser-warning: true' "$PUBLIC_URL/health" | jq .
```

Đã kiểm thử trên Máy 2:

- React build production thành công.
- Desktop/mobile UI đã kiểm tra bằng screenshot trình duyệt.
- Qdrant và LiteLLM đang chạy Docker.
- Local Gemma4, OpenAI và Grok trả `200`.
- Upload `documents/test1.pdf`, index 3 chunks, query SSE trả sources và token.
- `/agent` có key trả `200`, không có key trả `401`.

## Xử lý lỗi thường gặp

### LiteLLM không lên

```bash
docker compose logs --tail=120 litellm
docker compose up -d --force-recreate litellm
```

Nếu thấy lỗi config router, kiểm tra `litellm_config.yaml`.

### Web không hiện React

Build lại frontend và restart FastAPI:

```bash
cd frontend
npm run build
systemctl --user restart meibook-api
```

### FastAPI khởi động chậm

Lần đầu có thể chậm vì BGE-M3 và Docling/OCR model được nạp. Xem log:

```bash
journalctl --user -u meibook-api -n 160 --no-pager
```

### Ngrok URL đổi

Ngrok Free có thể cấp URL mới sau mỗi lần restart. Chạy
`./scripts/meibook.sh restart`; script sẽ tự lấy URL hiện tại từ ngrok API và
in dòng `Public web: ...` ra terminal. Không cần lưu URL tạm thời trong `.env`.

## Ghi chú về `docmind/`

`docmind/` có requirements và backend riêng từ các thử nghiệm trước. Trong kiến
trúc hiện tại, không dùng `docmind/` làm hệ thống chính. Dùng `src/`,
`frontend/`, `docker-compose.yml`, `litellm_config.yaml` và `.env` ở root
repository.
