# ARCHITECTURE

Tai lieu nay mo ta kien truc hien tai cua repo VLLM-PD tai May 2. Cac mo ta cu ve Streamlit, FAISS, FastAPI port 8000, MCP Hub rieng va vLLM host truc tiep khong con la duong chay chinh.

`docmind/` duoc xem la demo/tach rieng va khong nam trong kien truc van hanh chinh.

## 1. Muc tieu he thong

VLLM-PD gom hai nhom chuc nang:

1. Web RAG cho nguoi dung:
   - Upload tai lieu.
   - Parse bang Docling.
   - Embed bang BGE-M3.
   - Luu va search Qdrant theo session.
   - Goi LLM qua LiteLLM.
   - Cho phep nguoi dung chon model.

2. Coding Agent:
   - Endpoint `/agent`.
   - LangGraph dieu phoi agent.
   - LLM qua LiteLLM `coding-model`.
   - MCP filesystem/git tools.
   - Bao ve bang `AGENT_API_KEY`.

## 2. Vai tro cac may

| May | Vai tro hien tai | Dich vu lien quan |
|---|---|---|
| May 1 | Host LLM local | Ollama/Gemma4, expose qua ngrok vao `OLLAMA_API_BASE` |
| May 2 | Server chinh | FastAPI, React static, RAG pipeline, BGE-M3, Docling, Qdrant, LiteLLM, LangGraph Agent |
| May 3 | Client | Browser truy cap web/API public cua May 2, hoac coding client goi `/agent` |

He thong hien public mot URL chinh qua port `8001` cua May 2. LiteLLM port `4000` la noi bo.

## 3. So do tong the

```text
                           Public HTTPS ngrok
Nguoi dung / May 3  --------------------------------+
                                                     |
                                                     v
                                       +---------------------------+
                                       | May 2: FastAPI :8001     |
                                       | - React SPA at /         |
                                       | - REST/SSE API           |
                                       | - upload/session/query   |
                                       | - protected /agent       |
                                       +------------+--------------+
                                                    |
                +-----------------------------------+-----------------------------------+
                |                                   |                                   |
                v                                   v                                   v
      +-------------------+               +-------------------+              +-------------------+
      | RAG pipeline      |               | LangGraph Agent   |              | Static frontend   |
      | Docling           |               | MCP fs/git tools  |              | frontend/dist     |
      | BGE-M3            |               | coding-model      |              +-------------------+
      | Qdrant search     |               +---------+---------+
      +---------+---------+                         |
                |                                   |
                +-------------------+---------------+
                                    |
                                    v
                         +----------------------+
                         | LiteLLM proxy :4000  |
                         | internal only        |
                         +------+----+-----+----+
                                |    |     |
              +-----------------+    |     +----------------+
              |                      |                      |
              v                      v                      v
    +-------------------+   +-------------------+   +-------------------+
    | May 1 Ollama      |   | Xiaomi MiMo API   |   | OpenAI API        |
    | Gemma4 local      |   | MiMo 2.5 Pro      |   | GPT-4o mini       |
    +-------------------+   +-------------------+   +-------------------+
```

## 4. Runtime services

| Service | Port | Public | Owner | Mo ta |
|---|---:|---|---|---|
| FastAPI + React | 8001 | Co, qua ngrok | systemd user service | Web, REST API, SSE, Agent endpoint |
| LiteLLM | 4000 | Khong | Docker Compose | Router OpenAI-compatible |
| Qdrant | 6333/6334 | Khong | Docker Compose | Vector database |
| Ollama May 1 | 11434 local May 1 | Co, qua ngrok May 1 | May 1 | Gemma4 upstream |

FastAPI service:

```text
/home/jkl0909/.config/systemd/user/vllm-pd-api.service
```

Quan ly:

```bash
systemctl --user status vllm-pd-api
systemctl --user restart vllm-pd-api
journalctl --user -u vllm-pd-api -n 120 --no-pager
```

Docker services:

```bash
docker compose ps
docker compose logs --tail=120 litellm
docker compose logs --tail=120 qdrant
```

## 5. Backend API layer

File chinh: `src/api/main.py`.

Chuc nang:

- Load `.env` bang `python-dotenv`.
- Khoi tao singleton trong FastAPI lifespan:
  - `VectorStore`
  - `Embedder`
  - `DocumentParser`
  - `RAGPipeline`
- Mount React build tai `/` neu `frontend/dist` ton tai.
- CORS mo cho public web/API.
- Validate UUID session ID.
- Validate filename va file extension.
- Gioi han upload/query theo IP bang in-memory rate limiter.
- Bao ve `/agent` bang constant-time compare voi `AGENT_API_KEY`.

Endpoints:

| Endpoint | Vai tro |
|---|---|
| `GET /health` | Health check |
| `GET /models` | Model list cho UI |
| `POST /sessions` | Tao session |
| `GET /sessions/{session_id}` | Doc thong tin session |
| `DELETE /sessions/{session_id}` | Xoa session va file |
| `POST /sessions/{session_id}/upload` | Upload, parse, embed, index |
| `DELETE /sessions/{session_id}/files/{filename}` | Xoa file khoi Qdrant/session |
| `POST /query` | RAG non-streaming |
| `POST /query/stream` | RAG SSE streaming |
| `POST /agent` | Coding Agent |

## 6. Frontend layer

Thu muc: `frontend/`.

Stack:

- React 18.
- Vite.
- `lucide-react`.
- `react-markdown`.

Man hinh chinh:

- Sidebar tai lieu.
- Nut tao phien moi.
- Upload nhieu file.
- Tabs `Hoi dap` va `Nghien cuu`.
- Model selector.
- Chat composer.
- Panel sources tren desktop.
- Sources inline tren mobile.

Frontend dung relative API path, vi vay cung mot public URL port `8001` phuc vu ca web va API:

```text
/              -> React SPA
/health        -> FastAPI
/models        -> FastAPI
/sessions      -> FastAPI
/query/stream  -> FastAPI SSE
```

Build:

```bash
cd frontend
npm install
npm run build
```

`frontend/dist` la build output va khong can commit.

## 7. RAG pipeline

Thanh phan:

| File | Vai tro |
|---|---|
| `src/rag/parser.py` | Dung Docling de convert tai lieu sang markdown, sau do chunk |
| `src/rag/embedder.py` | Load `BAAI/bge-m3`, embed query/documents |
| `src/rag/vector_store.py` | Qdrant collection, session filtering, add/search/delete chunks |
| `src/rag/rag_pipeline.py` | Retrieval, prompt, model routing, LiteLLM call |

Luon upload:

```text
User upload
  -> FastAPI validate session/filename/extension/size
  -> Save file vao UPLOAD_DIR/session_id
  -> Docling process_file
  -> Split markdown thanh TextChunk
  -> BGE-M3 embed_documents
  -> Qdrant upsert payload co session_id va source_file
```

Luon query:

```text
Question
  -> BGE-M3 embed_query
  -> Qdrant search theo session_id
  -> build_rag_prompt
  -> LiteLLM chat.completions
  -> answer + formatted sources
```

Mode `chat`:

- Retrieval top_k mac dinh 5.
- Prompt tra loi ngan gon, co citation.

Mode `research`:

- Retrieval top_k 10.
- Prompt yeu cau bao cao nghien cuu:
  - Tom tat dieu hanh.
  - Phat hien chinh.
  - Bang chung.
  - Diem chua ro.
  - Cau hoi nghien cuu tiep theo.
- Max tokens cao hon mode chat.

## 8. LiteLLM routing

File: `litellm_config.yaml`.

Model groups:

| Group | Backend | Dung cho |
|---|---|---|
| `auto-model` | Ollama/Gemma4 | Lua chon tu dong trong UI |
| `local-gemma` | Ollama/Gemma4 | Khi user chon Gemma4 Local |
| `mimo-pro` | OpenAI-compatible MiMo API | Khi user chon MiMo 2.5 Pro |
| `openai-model` | OpenAI GPT-4o mini | Khi user chon OpenAI |
| `coding-model` | Ollama/Gemma4 | LangGraph Coding Agent |

Fallback:

```yaml
router_settings:
  fallbacks:
    - auto-model: ["mimo-pro", "openai-model"]
    - coding-model: ["openai-model"]
```

Y nghia:

- Web chon `auto`: uu tien local Gemma4, neu fail thi MiMo, neu fail tiep thi OpenAI.
- Agent: uu tien local Gemma4, fallback OpenAI.
- Neu user chon truc tiep `mimo` hoac `openai`, LiteLLM goi dung group do, khong fallback sang group khac.

MiMo Token Plan:

```env
MIMO_API_BASE=https://token-plan-sgp.xiaomimimo.com/v1
```

Pay-as-you-go MiMo co the dung endpoint khac:

```env
MIMO_API_BASE=https://api.xiaomimimo.com/v1
```

## 9. Coding Agent architecture

File:

- `src/agent/graph.py`
- `src/agent/mcp_client.py`

LangGraph:

```text
AgentState(messages)
      |
      v
call_model
      |
      |-- neu co tool_calls -> ToolNode -> call_model
      |
      `-- neu khong -> END
```

LLM:

- `ChatOpenAI`
- `model="coding-model"`
- `openai_api_base=LITELLM_URL`
- `openai_api_key=LITELLM_MASTER_KEY`

MCP:

- Dung `langchain-mcp-adapters==0.2.2`.
- Dung `MultiServerMCPClient`.
- Filesystem MCP:
  - `@modelcontextprotocol/server-filesystem`
  - allowed directory: `WORKSPACE_DIR`
- Git MCP:
  - `mcp-server-git`
  - repository: `AGENT_REPOSITORY_DIR`
- Tools duoc cache bang `lru_cache`.
- Neu MCP fail, fallback local tools chi doc/ghi/list ben trong `WORKSPACE_DIR`.

Security:

- Public `/agent` can header:

```text
X-Agent-API-Key: <AGENT_API_KEY>
```

Khong nen de `AGENT_API_KEY` trong `.env` rong khi public API.

## 10. Data model

Qdrant collection:

```text
docmind_documents
```

Vector dimension:

```text
1024
```

Payload moi point:

```json
{
  "session_id": "...",
  "text": "...",
  "source_file": "test1.pdf",
  "page_number": 1,
  "chunk_index": 0,
  "content_type": "text",
  "metadata": {
    "source": "docling"
  }
}
```

Session isolation dua vao payload filter `session_id`.

## 11. Public networking

Hien tai chi can mot public URL cho nguoi dung va May 3:

```text
MACHINE2_API_PUBLIC_URL -> ngrok -> http://localhost:8001
```

Khong can public LiteLLM:

```text
MACHINE2_LITELLM_LOCAL_URL=http://localhost:4000
```

Ly do:

- Web va API da cung domain/port 8001.
- Frontend dung relative paths.
- Coding client co the goi `/agent` qua cung public URL.
- LiteLLM co master key va provider credentials, nen nen o noi bo.

## 12. Security boundaries

| Boundary | Co che |
|---|---|
| Secret/API key | `.env`, `.gitignore` |
| Agent endpoint | `AGENT_API_KEY` va `X-Agent-API-Key` |
| Session ID | UUID validation |
| File path | `Path(filename).name`, extension allowlist |
| Upload size | `MAX_UPLOAD_SIZE_MB` |
| Abuse control | In-memory per-IP rate limit |
| MCP filesystem | `WORKSPACE_DIR` |
| MCP git | `AGENT_REPOSITORY_DIR` |
| LiteLLM | Noi bo port 4000 |

## 13. Current operational state on May 2

Da thiet lap:

- `docker compose` chay Qdrant va LiteLLM.
- `systemd --user` chay FastAPI/React:

```text
vllm-pd-api.service
```

- `loginctl enable-linger` da bat, nen service co the chay sau khi logout.
- React da build production.
- LiteLLM da nhan 5 model groups.
- MCP filesystem/git da load du 26 tools.

Da test:

- `GET /health` local va public ngrok.
- `GET /models`.
- React desktop/mobile bang Playwright screenshot.
- Upload `documents/test1.pdf`.
- Qdrant index 3 chunks.
- `POST /query/stream` tra sources, meta, token va done.
- Local Gemma4 tra `200`.
- MiMo Token Plan SGP tra `200`.
- OpenAI tra `200`.
- `/agent` co key tra `200`, khong key tra `401`.

## 14. Failure modes

### Machine 1 Ollama/ngrok fail

Anh huong:

- `local-gemma` fail.
- `auto-model` co the fallback sang MiMo/OpenAI.
- `coding-model` co the fallback sang OpenAI.

Can lam:

```bash
curl "$OLLAMA_API_BASE/api/tags"
docker compose restart litellm
```

### LiteLLM fail config

Anh huong:

- Query RAG va Agent khong goi duoc model.

Can lam:

```bash
docker compose logs --tail=120 litellm
docker compose up -d --force-recreate litellm
```

### FastAPI fail startup

Thuong gap khi:

- Qdrant chua len.
- Python env thieu package.
- GPU/model embedding load loi.

Can lam:

```bash
journalctl --user -u vllm-pd-api -n 160 --no-pager
systemctl --user restart vllm-pd-api
```

### Web trang trang hoac 404 asset

Can build lai frontend:

```bash
cd frontend
npm run build
systemctl --user restart vllm-pd-api
```

## 15. Deployment checklist

```bash
cd /home/jkl0909/Code/llm/VLLM-PD

# Python syntax
conda run -n docmind python -m py_compile \
  src/api/main.py \
  src/rag/rag_pipeline.py \
  src/agent/graph.py \
  src/agent/mcp_client.py

# Frontend
cd frontend
npm install
npm run build

# Infra
cd ..
docker compose up -d
docker compose ps

# API service
systemctl --user restart vllm-pd-api
curl -fsS http://localhost:8001/health | jq .

# Public URL
PUBLIC_URL=$(sed -n 's/^MACHINE2_API_PUBLIC_URL=//p' .env)
curl -fsS -H 'ngrok-skip-browser-warning: true' "$PUBLIC_URL/health" | jq .
```

## 16. Nhung gi khong con la duong chinh

- `docmind/` khong phai runtime chinh.
- Streamlit khong phai frontend chinh.
- FAISS khong phai vector store chinh.
- FastAPI port 8000 khong phai port chinh.
- LiteLLM public URL khong can thiet cho May 3.
- vLLM local server khong phai backend inference dang dung trong repo hien tai.
