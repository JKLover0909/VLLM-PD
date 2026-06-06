# VLLM-PD Presentation Outline

## Slide 1: Bai toan va boi canh

- Nguoi dung can hoi dap va nghien cuu tai lieu qua web.
- He thong can ho tro model local va cloud.
- Coding Agent can API rieng de goi tac vu lap trinh.
- May 1, May 2, May 3 can phoi hop qua public URL don gian.

Speaker notes:
Mo dau bang nhu cau thuc te: tai lieu nhieu dinh dang, can tom tat/co nguon, va can co mot endpoint rieng cho agent lap trinh. Diem quan trong la he thong khong chi la chatbot, ma la nen tang RAG + model router + coding agent.

## Slide 2: Muc tieu he thong

- Mot public URL cho web va API.
- Upload, index va hoi dap tai lieu.
- Cho phep user chon model.
- Uu tien local, fallback cloud khi can.
- Tach web nguoi dung va Coding Agent co khoa bao ve.

Speaker notes:
Nhan manh mot quyet dinh thiet ke: chi public FastAPI port 8001. LiteLLM va Qdrant o noi bo de giam be mat tan cong.

## Slide 3: Cong nghe su dung

| Tang | Cong nghe |
|---|---|
| Frontend | React, Vite, lucide-react, react-markdown |
| API | FastAPI, SSE streaming, Pydantic |
| RAG | Docling, BGE-M3, Qdrant |
| Router | LiteLLM |
| Agent | LangGraph, MCP filesystem/git |
| Infra | Docker Compose, systemd user service, ngrok |

Speaker notes:
Trinh bay theo tang de nguoi nghe khong bi ngop vi danh sach cong nghe. Moi cong nghe co vai tro ro rang trong pipeline.

## Slide 4: Model va API model routing

| Lua chon | Model group | Backend |
|---|---|---|
| Tu dong | auto-model | Gemma4 -> MiMo -> OpenAI |
| Gemma4 Local | local-gemma | Ollama/Gemma4 tren May 1 |
| MiMo 2.5 Pro | mimo-pro | Xiaomi MiMo API |
| OpenAI | openai-model | GPT-5.4 mini |
| Agent | coding-model | Gemma4 -> OpenAI |

Speaker notes:
Giai thich model group la ten logic de frontend/backend khong phu thuoc truc tiep vao provider. LiteLLM giu vai tro router va fallback.

## Slide 5: Kien truc tong the

```text
May 3 / User
    |
    | ngrok HTTPS, port 8001
    v
May 2: FastAPI + React
    |-- RAG: Docling -> BGE-M3 -> Qdrant
    |-- Agent: LangGraph -> MCP tools
    v
May 2: LiteLLM internal :4000
    |-- May 1 Ollama/Gemma4
    |-- MiMo API
    `-- OpenAI API
```

Speaker notes:
Day la slide trung tam. Chi ra service nao public, service nao noi bo. Public duy nhat la FastAPI port 8001.

## Slide 6: Luong RAG tai lieu

1. User upload tai lieu.
2. FastAPI validate file, session va size.
3. Docling parse sang markdown.
4. BGE-M3 embed chunks.
5. Qdrant luu vector theo session.
6. Query -> retrieve -> prompt -> LiteLLM -> answer + sources.

Speaker notes:
Nhan manh sources/citation: cau tra loi khong chi sinh van ban, ma kem bang chung tu chunk tai lieu.

## Slide 7: Luong chon model va fallback

- `local`: chi goi Gemma4 local.
- `mimo`: chi goi MiMo.
- `openai`: chi goi OpenAI.
- `auto`: Gemma4 local fail thi fallback MiMo, roi OpenAI.
- Agent dung `coding-model`: Gemma4 local, fallback OpenAI.

Speaker notes:
Day la phan giai thich tai sao can LiteLLM. Nguoi dung chon model o UI, backend anh xa sang model group, LiteLLM xu ly provider/fallback.

## Slide 8: Web public va API tren May 2

- React SPA duoc FastAPI serve tai `/`.
- API cung domain: `/health`, `/models`, `/sessions`, `/query/stream`.
- SSE streaming de hien cau tra loi theo token.
- Ngrok expose `http://localhost:8001`.
- LiteLLM khong public.

Speaker notes:
Day la diem thiet ke giup May 3 dung don gian: mot URL duy nhat vua la web vua la API. Khong can public them LiteLLM.

## Slide 9: Coding Agent va MCP

- Endpoint: `POST /agent`.
- Bao ve bang `X-Agent-API-Key`.
- LangGraph dieu phoi vong lap agent/tool.
- MCP filesystem gioi han trong `WORKSPACE_DIR`.
- MCP git khoa vao `AGENT_REPOSITORY_DIR`.
- LLM agent goi `coding-model` qua LiteLLM.

Speaker notes:
Phan nay tach khoi RAG. Agent co quyen dung tool doc/ghi file/git nen can security boundary rieng.

## Slide 10: Bao mat va van hanh

- `.env` bi ignore, khong commit API key.
- Validate UUID session ID.
- Chan path traversal qua filename validation.
- Allowlist extension upload.
- Rate limit query/upload theo IP.
- systemd user service tu restart.
- Docker Compose quan ly Qdrant va LiteLLM.

Speaker notes:
Neu trinh bay truoc hoi dong ky thuat, slide nay rat quan trong. No cho thay he thong khong chi chay duoc demo, ma co suy nghi ve van hanh.

## Slide 11: Ket qua kiem thu / Demo

- React build production thanh cong.
- UI desktop/mobile da kiem tra bang Playwright screenshot.
- LiteLLM nhan 5 model groups.
- Gemma4, MiMo Token Plan SGP va OpenAI goi duoc.
- Upload `documents/test1.pdf`, index 3 chunks.
- `/query/stream` tra sources, meta, token, done.
- `/agent` co key tra 200, khong key tra 401.

Speaker notes:
Day la slide de chot rang cac thanh phan khong chi nam tren so do, ma da duoc test end-to-end.

## Slide 12: Trade-off va huong phat trien

Trade-off hien tai:
- Ngrok URL co the thay doi.
- Rate limit dang in-memory.
- Chua co user login/history rieng.
- Upload lon nen dua vao job queue.

Huong phat trien:
- Domain co dinh.
- Auth/user management.
- Redis rate limit.
- Luu lich su hoi thoai.
- Metrics/log dashboard.
- Queue cho parse/index tai lieu lon.

Speaker notes:
Ket thuc bang nhung gi da lam va nhung gi can nang cap. Cach nay giup bai trinh bay thuc te, khong tao cam giac he thong da hoan hao tuyet doi.
