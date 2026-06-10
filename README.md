# VLLM-PD

VLLM-PD la he thong RAG va Coding Agent chay tren May 2. He thong hien tai phuc vu web React, API FastAPI, Qdrant vector database va LiteLLM router trong cung mot repo.

`docmind/` la phan demo/tach rieng, khong phai duong chay chinh cua repo hien tai.

## Trang thai hien tai

- Web cho nguoi dung: React SPA, duoc FastAPI phuc vu tai `/`.
- API Gateway: FastAPI chay o port `8001`.
- Public URL: ngrok expose port `8001`; script khoi dong tu lay va in URL moi ra terminal.
- LiteLLM: chay noi bo o port `4000`, khong can public cho nguoi dung.
- Qdrant: chay noi bo o port `6333`.
- Embedding: `BAAI/bge-m3` chay tren May 2.
- Parser tai lieu: Docling.
- LLM upstream:
  - Gemma4 local tren May 1 qua Ollama/ngrok.
  - MiMo 2.5 Pro qua Xiaomi MiMo API.
  - OpenAI GPT-5.4 mini.
- Session co file anh `.png/.jpg/.jpeg` tu dong route sang OpenAI Vision de doc noi dung anh.
- Coding Agent: LangGraph + MCP tools, endpoint `/agent` duoc bao ve bang `AGENT_API_KEY`.

## Kien truc nhanh

```text
Nguoi dung / May 3
        |
        | HTTPS ngrok, port 8001
        v
May 2: FastAPI + React
        |
        |-- Upload tai lieu -> Docling -> BGE-M3 -> Qdrant
        |
        |-- Query RAG -> Qdrant -> LiteLLM
        |
        |-- /agent -> LangGraph -> MCP filesystem/git -> LiteLLM
        |
        v
May 2: LiteLLM, port 4000 noi bo
        |
        |-- auto-model -> MiMo 2.5 Pro -> fallback OpenAI -> fallback local Gemma4
        |-- local-gemma -> Gemma4 local tren May 1
        |-- mimo-pro -> MiMo 2.5 Pro
        |-- openai-model -> GPT-5.4 mini
        |-- grok-model -> Grok 4.20 Reasoning qua Azure
        |-- coding-model -> Gemma4 local -> fallback OpenAI
```

Chi tiet hon xem [ARCHITECTURE.md](ARCHITECTURE.md).

## Thu muc chinh

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
|       |-- graph.py         # LangGraph coding agent
|       `-- mcp_client.py    # MCP filesystem/git tools
|-- frontend/                # React + Vite web app
|-- docker-compose.yml       # Qdrant + LiteLLM
|-- litellm_config.yaml      # Model aliases and fallback routing
|-- requirements.txt         # Python dependencies for main system
|-- .env.example             # Config template
`-- docmind/                 # Separate demo/runtime area, not main flow
```

## Yeu cau

- Linux tren May 2.
- Docker va Docker Compose.
- Conda hoac Python 3.10 environment.
- NVIDIA GPU neu muon chay BGE-M3 nhanh tren CUDA.
- Node.js 20+ de build frontend.
- May 1 dang expose Ollama/Gemma4 qua `OLLAMA_API_BASE`.

Repo hien dang dung conda env ten `docmind`, nhung day chi la ten moi truong. Thu muc `docmind/` van la demo rieng.

## Cai dat moi truong

Tu root repo:

```bash
cd /home/jkl0909/Code/llm/VLLM-PD

conda create -n docmind python=3.10 -y
conda activate docmind
pip install --upgrade pip
pip install -r requirements.txt

# Node.js neu env chua co npm/node
conda install -n docmind -c conda-forge nodejs=20 -y
```

Frontend:

```bash
cd /home/jkl0909/Code/llm/VLLM-PD/frontend
npm install
npm run build
```

## Cau hinh `.env`

Tao file `.env`:

```bash
cd /home/jkl0909/Code/llm/VLLM-PD
cp .env.example .env
```

Nhung bien quan trong:

```env
OLLAMA_API_BASE=https://your-machine-1-ollama-ngrok-url
OLLAMA_MODEL=gemma4:latest

OPENAI_API_KEY=...
MIMO_API_KEY=...
MIMO_API_BASE=https://token-plan-sgp.xiaomimimo.com/v1

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

UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE_MB=25
QUERY_RATE_LIMIT_PER_MINUTE=15
UPLOAD_RATE_LIMIT_PER_HOUR=10

AGENT_API_KEY=replace_with_a_long_random_secret
WORKSPACE_DIR=/home/jkl0909/Code/llm
AGENT_REPOSITORY_DIR=/home/jkl0909/Code/llm/VLLM-PD
```

Khong commit `.env`. File nay co the chua API key that.

## Chay he thong

### 1. Khoi dong Qdrant va LiteLLM

```bash
cd /home/jkl0909/Code/llm/VLLM-PD
docker compose up -d
docker compose ps
```

Kiem tra LiteLLM:

```bash
curl http://localhost:4000/health/liveliness

KEY=$(sed -n 's/^LITELLM_MASTER_KEY=//p' .env)
curl -H "Authorization: Bearer $KEY" http://localhost:4000/v1/models
```

### 2. Build React

```bash
cd /home/jkl0909/Code/llm/VLLM-PD/frontend
npm install
npm run build
```

FastAPI chi mount web khi `frontend/dist` ton tai.

### 3. Chay FastAPI port 8001

Chay foreground de debug:

```bash
cd /home/jkl0909/Code/llm/VLLM-PD
conda activate docmind
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

Hoac dung service da cau hinh tren May 2:

```bash
systemctl --user status vllm-pd-api
systemctl --user restart vllm-pd-api
journalctl --user -u vllm-pd-api -n 120 --no-pager
```

May hien tai da bat `loginctl enable-linger`, nen user service co the chay sau khi logout.

### 4. Public bang ngrok

Neu dung ngrok Free, de ngrok tu sinh URL random cho port `8001`:

```bash
ngrok http 8001
```

Neu co reserved/static domain cua ngrok tra phi, dat `NGROK_RESERVED_DOMAIN`
de script dua domain nay vao tham so `ngrok --url`.

Chi can public port `8001`. Khong public LiteLLM port `4000` neu khong co ly do rieng.

## Su dung web

Mo:

```text
http://localhost:8001
```

Hoac public URL duoc `./scripts/vllm-pd.sh start` in ra terminal.

Nguoi dung co the:

- Tao phien moi.
- Upload PDF, DOCX, XLSX, PPTX, HTML, PNG, JPG, JPEG.
- Chon mode `Hoi dap` hoac `Nghien cuu`.
- Chon model `Tu dong`, `Gemma4 Local`, `MiMo 2.5 Pro`, `OpenAI`.
- Dat cau hoi va xem sources tu tai lieu.

## API endpoints

| Method | Endpoint | Mo ta |
|---|---|---|
| `GET` | `/health` | Kiem tra API va Qdrant config |
| `GET` | `/models` | Danh sach model cho frontend |
| `POST` | `/sessions` | Tao session RAG |
| `GET` | `/sessions/{session_id}` | Lay thong tin session |
| `DELETE` | `/sessions/{session_id}` | Xoa session va file upload |
| `POST` | `/sessions/{session_id}/upload` | Upload va index tai lieu |
| `DELETE` | `/sessions/{session_id}/files/{filename}` | Xoa mot file trong session |
| `POST` | `/query` | Hoi dap non-streaming |
| `POST` | `/query/stream` | Hoi dap SSE streaming |
| `POST` | `/agent` | Coding Agent, can `X-Agent-API-Key` |

Request query mau:

```bash
SESSION_ID=$(curl -fsS -X POST http://localhost:8001/sessions | jq -r .session_id)

curl -fsS http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"question\":\"Tai lieu nay noi ve gi?\",
    \"model\":\"local\",
    \"mode\":\"chat\",
    \"stream\":false
  }" | jq .
```

Upload mau:

```bash
curl -fsS -X POST \
  "http://localhost:8001/sessions/$SESSION_ID/upload" \
  -F "file=@documents/test1.pdf" | jq .
```

Agent mau:

```bash
AGENT_KEY=$(sed -n 's/^AGENT_API_KEY=//p' .env)

curl -fsS http://localhost:8001/agent \
  -H "X-Agent-API-Key: $AGENT_KEY" \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"task\":\"Khong goi cong cu. Chi tra loi dung mot tu: OK\"
  }" | jq .
```

## Model routing

LiteLLM aliases trong `litellm_config.yaml`:

| UI/API option | LiteLLM model group | Backend |
|---|---|---|
| `auto` | `auto-model` | MiMo 2.5 Pro, fallback OpenAI, fallback Gemma4 local |
| `local` | `local-gemma` | Ollama/Gemma4 tren May 1 |
| `mimo` | `mimo-pro` | MiMo 2.5 Pro |
| `openai` | `openai-model` | OpenAI GPT-5.4 mini |
| `grok` | `grok-model` | Grok 4.20 Reasoning qua Azure |
| Agent | `coding-model` | Gemma4 local, fallback OpenAI |

Neu session co file anh `.png`, `.jpg` hoac `.jpeg`, backend bo qua lua chon
model cua nguoi dung, bao gom Grok, va route truy van sang `openai-model` de su dung Vision.

## Bao mat va gioi han

- `.env` bi ignore boi Git.
- `/agent` yeu cau header `X-Agent-API-Key` neu `AGENT_API_KEY` duoc set.
- Upload chi nhan extension cho phep.
- Ten file va session ID duoc validate de tranh path traversal.
- Upload gioi han mac dinh 25 MB/file.
- Query gioi han mac dinh 15 request/IP/phut.
- Upload gioi han mac dinh 10 request/IP/gio.
- MCP filesystem tool chi duoc phep truy cap `WORKSPACE_DIR`.
- Git MCP tool khoa vao `AGENT_REPOSITORY_DIR`.

## Kiem thu nhanh

```bash
cd /home/jkl0909/Code/llm/VLLM-PD

# Python syntax
conda run -n docmind python -m py_compile \
  src/api/main.py \
  src/rag/rag_pipeline.py \
  src/agent/graph.py \
  src/agent/mcp_client.py

# Frontend build
cd frontend
conda run -n docmind npm run build

# Health
curl -fsS http://localhost:8001/health | jq .

# Public health
PUBLIC_URL=$(curl -fsS http://localhost:4040/api/tunnels \
  | jq -r '.tunnels[] | select(.proto == "https") | .public_url')
curl -fsS -H 'ngrok-skip-browser-warning: true' "$PUBLIC_URL/health" | jq .
```

Da kiem thu tren May 2:

- React build production thanh cong.
- Desktop/mobile UI bang Playwright screenshot.
- Qdrant va LiteLLM dang chay Docker.
- Local Gemma4 va OpenAI tra `200`.
- MiMo Token Plan SGP tra `200`.
- Upload `documents/test1.pdf`, index 3 chunks, query SSE tra sources va token.
- `/agent` co key tra `200`, khong co key tra `401`.

## Troubleshooting

### LiteLLM khong len

```bash
docker compose logs --tail=120 litellm
docker compose up -d --force-recreate litellm
```

Neu thay loi config router, kiem tra `litellm_config.yaml`.

### MiMo tra 401

Kiem tra `MIMO_API_BASE`. Token Plan key `tp-*` thuong can endpoint Token Plan, vi du:

```env
MIMO_API_BASE=https://token-plan-sgp.xiaomimimo.com/v1
```

Pay-as-you-go key co the dung:

```env
MIMO_API_BASE=https://api.xiaomimimo.com/v1
```

### Web khong hien React

Build lai frontend va restart FastAPI:

```bash
cd frontend
npm run build
systemctl --user restart vllm-pd-api
```

### FastAPI khoi dong cham

Lan dau co the cham vi BGE-M3 va Docling/OCR model duoc nap. Xem log:

```bash
journalctl --user -u vllm-pd-api -n 160 --no-pager
```

### Ngrok URL doi

Ngrok Free co the cap URL moi sau moi lan restart. Chay
`./scripts/vllm-pd.sh restart`; script se tu lay URL hien tai tu ngrok API va
in dong `Public web: ...` ra terminal. Khong can luu URL tam thoi trong `.env`.

## Ghi chu ve `docmind/`

`docmind/` co requirements va backend rieng tu cac thu nghiem truoc. Trong kien truc hien tai, khong dung `docmind/` lam he thong chinh. Dung `src/`, `frontend/`, `docker-compose.yml`, `litellm_config.yaml` va `.env` o root repo.
