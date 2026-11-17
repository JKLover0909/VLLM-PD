# progress.md — Nhật ký bàn giao giữa các phiên

> Quy tắc: entry mới nhất ở trên cùng; giữ khoảng 10 entry gần nhất (entry cũ hơn
> có thể xóa vì đã nằm trong Git history). Chỉ ghi kết quả của lệnh đã thật sự chạy.
> **Không** ghi secret, token, mã nhân viên, tên người hay số liệu HR/MES thật.
> Trạng thái tính năng nằm ở `feature_list.json`; quy trình phiên nằm ở `CLAUDE.md`
> → "Session Harness".

## Current state

_Cập nhật: 2026-11-17 (tất cả alias dùng Ollama local `qwen3.5:9b`; Production chưa đổi)_

- **Checkout:** `/home/jkl/Code/VLLM-PD-dev`, nhánh `dev` @ `810f720`. `main` đi trước
  `dev` 1 commit (`7fec558`); `dev` không có commit nào chưa vào `main`.
- **Working tree:** nhiều thay đổi chưa commit từ các phiên trước (trước phiên
  2026-10-31: 40 file sửa, 27 file untracked) cộng file harness của phiên này. Chưa
  review nội dung; không revert/format/stage file mình không sửa.
- **Baseline kiểm thử (2026-10-31, Dev, trước khi thêm harness):**
  pytest suite mặc định 909 passed; `npm --prefix frontend test` 44/44 pass;
  pyflakes trên `src/` sạch; `git diff --check` sạch.
- **Đang làm dở:** `guard.cross_turn` (test pass, chờ review), `mode.qa` (chưa live
  test với QMS :8600), `llm.routing` (code Dev chuẩn bị remote Qwen target; chưa có
  tất cả alias dùng Ollama local `qwen3.5:9b`; Production chưa đổi).
- **Cần người dùng quyết định:**
  1. Review + commit nhóm thay đổi chưa commit (nên tách commit theo nhóm bên dưới).
  2. Có merge `7fec558` từ `main` về `dev` không.
  3. Ưu tiên sửa `hygiene.test_fixture_pii` (fixture test dùng tên thật).

### Nhóm thay đổi chưa commit (theo tên file, chưa review nội dung)

| Nhóm | File chính |
| --- | --- |
| QA mode | `src/integrations/qa_agent_service.py`, `src/api/qa_handlers.py`, `src/api/schemas.py`, `frontend/src/qa-*.js`, `Markdowns/QA_AGENT.md`, `tests/test_qa_*.py`, `frontend/test/qa-*.test.js` |
| Guards | `src/api/prompt_guard.py`, `query_guard.py`, `scope_guard.py`, `src/api/main.py`, `tests/test_prompt_*guard.py`, `tests/test_query_*guard*.py` |
| Frontend modules | `frontend/src/markdown-tables.js`, `query-stream.js`, `mode-tabs.js`, `main.jsx`, `components/*.jsx`, `styles.css`, `package*.json` |
| MSM mock DB | `scripts/create_msm_mock_database.py`, `database/schema/msm_msa.sql`, `database/MSM_MSA_SQL_CONTEXT.md`, `tests/test_create_msm_mock_database.py` |
| Model routing | `litellm_config.yaml`, `tests/test_litellm_routing_config.py`, `tests/test_model_routes.py` |
| Backend khác | `src/actions/*.py`, `src/i18n/translation.py`, `src/integrations/mes_*.py`, `cctvai_hardware_service.py`, `src/rag/*.py`, `src/api/helpers.py`, `sse.py` |
| Infra / docs | `docker-compose*.yml`, `.env*.example`, `README.md`, `AGENTS.md`, `Markdowns/*.md`, `database/README.md` |
| Harness (2026-10-31) | `CLAUDE.md`, `AGENTS.md` (2 chỗ), `feature_list.json`, `progress.md`, `scripts/harness-check.sh`, `.claude/settings.json`, `.claude/hooks/*.sh` |

## Session log

### 2026-11-17 — Chuẩn bị remote Qwen routing, TLS/VPN gated

- **Yêu cầu:** mọi tác vụ LLM thường dùng một Qwen3.5-9B Q4_K_M remote; Azure
  `gpt-5.4-mini` chỉ fallback tuần tự khi primary lỗi. Người dùng chốt telemetry
  hardware không được gửi tới remote/cloud và yêu cầu TLS/VPN trước dữ liệu thật.
- **Thay đổi trong Dev:** LiteLLM aliases 4 role đều trỏ đến một provider/env
  primary; bỏ alias cloud/OpenAI thứ hai và alias hardware. Compose bỏ Ollama env,
  host-gateway/proxy dependency/service khỏi topology web; ví dụ env để endpoint/key
  trống. `openai` từ client legacy map về primary; bỏ provider options per caller;
  hardware renderer deterministic và history hardware bị lọc ở API/frontend. Cập nhật
  README, AGENTS, Architecture/Deploy/MesEXPLAIN và model labels. Production không sửa.
- **Xác minh:** 196 tests pass qua `scripts/meibook-python`; Dev Compose config -q
  và git diff --check sạch tại lần chạy. Trước đó YAML syntax check tool Ruby/Node
  thiếu dependency; parse độc lập chưa làm được. Không gọi endpoint remote, không đọc
  `.env.docker`, không restart/rebuild container.
- **Blocker:** chưa có TLS/VPN endpoint được cung cấp/xác minh và key vận hành chưa
  cấu hình. Vì vậy không thể smoke-test wire thật hoặc kết luận model primary hoạt động.
  Ollama service/model trên host vẫn đang active; Dev runtime cũng chưa chuyển.
  Production chưa thay code/config hoặc restart. Không dừng Ollama, không xóa model.
- **Còn phải làm:** kiểm thử LiteLLM 1.83 với upstream giả để chứng minh
  `chat_template_kwargs.enable_thinking` passthrough và strip khỏi Azure, test ảnh,
  JSON/tools/stream/fallback; sửa tài liệu con còn stale và review `git diff` vì file
  Dev có thay đổi chưa commit từ trước. Sau TLS/VPN: user-approved Dev smoke → xin duyệt
  riêng Production rollout và Ollama stop. Không commit.

### 2026-11-17 — Robot hiện diện + sân khấu thích ứng (frontend, Dev)

- **Yêu cầu:** giao diện "sống" theo phương án C (robot dẫn dắt, lựa chọn là chính), chưa có giọng nói.
- **Thêm:** `frontend/src/agent-presence.js` (suy ra trạng thái từ busy/pending/status/timeline/stopped/error/QA clarification,
  không bịa tiến độ), `components/AgentRobot.jsx` (robot SVG 2.5D, mắt/đầu theo con trỏ, chạm và focus; rAF, dừng khi tab ẩn
  hoặc reduced-motion), `components/AgentPresence.jsx` (sân khấu lớn khi chưa có hội thoại, thanh gọn khi đã có; nút tắt chuyển động;
  nhãn trạng thái `role="status"`), `frontend/test/agent-presence.test.js` (6 test, fail trước khi có module).
- **Sửa:** `main.jsx` (thay icon empty state bằng robot, tối đa 4 gợi ý + "Xem thêm", vẫn dùng `sendMessage`/`handleQuickAnswerClick`);
  `styles.css` (khối "Agent presence", sáng/tối, phone, reduced-motion). Không đổi backend/API/SSE.
- **Xác minh:** `npm --prefix frontend test` 50/50 (baseline 44/44). Playwright trên build tạm + server tĩnh không proxy, mọi API mock:
  43/43 (sáng/tối, desktop/phone, ánh mắt theo con trỏ, working→complete, không tràn ngang, không lỗi console, reduced-motion dừng robot).
  Build vào `frontend/dist` Dev; asset 8002 khớp bản build; `/health` 200. Không restart Docker.
- **Sự cố cần biết:** một lần thăm dò Vite dev server không mock đã đi qua proxy mặc định `vite.config.js` tới cổng **8001 (Production)**,
  gọi các GET khởi động và `POST /sessions` (chỉ ghi log, không ghi Qdrant/DB). `vite.config.js` proxy về 8001 là rủi ro cho kiểm thử Dev.
- **Còn lại:** robot chưa có thao tác Có/Không mới (không có contract backend); chưa kiểm thử với model thật trên 8002.

### 2026-11-16 — Giữ bản 9B Q4_K_M để dành VRAM cho BGE/OCR

- **Quyết định mới của người dùng:** mọi vai trò local dùng `qwen3.5:9b`
  (Q4_K_M, file ~6.6 GB), không chọn bản Q8_0 vừa tải. Không xóa model nào.
- **Đã sửa trên Dev:** `litellm_config.yaml` (5 alias cùng tag Q4_K_M),
  `tests/test_litellm_routing_config.py`, `feature_list.json`, `progress.md`.
- **Xác minh:** 25 targeted tests passed; preflight xác nhận project `meibook-dev`,
  container `meibook-litellm-dev`, port `127.0.0.1:4001`, mount config từ Dev.
  Restart chỉ LiteLLM Dev; cả 5 alias smoke pass với fallback tắt, không có reasoning,
  header upstream `:11435`, attempted fallbacks 0. Web 8002 healthy.
  `ollama ps`: chỉ Q4_K_M đang loaded, 100% GPU, ~5.7 GB, context 8192.
  Tổng GPU 9808/16303 MiB (còn ~6.3 GiB trống tại thời điểm kiểm tra).
- **Còn lại:** docs/env examples chưa cập nhật (agent isolation guard chặn, không bypass).
  Production chưa rollout: file config vẫn Q8_0 từ bước trước, container chưa restart.
  Không chạy Production Compose hoặc xóa model. KHÔNG xóa `qwen3.5:9b` như kế hoạch cũ.

### 2026-11-16 — Hợp nhất mọi alias local về một model Ollama (trước quyết định Q4)

- **Mục tiêu:** bỏ Swift-27B ở máy LAN (`192.168.10.188:8998`, kiểm tra ngày
  2026-11-16 thấy connection refused), chạy mọi vai trò local trên một model host.
  `qwen3.5:14b` không tồn tại trên Ollama registry nên dùng `qwen3.5:9b-q8_0`.
- **Thay đổi (cả Dev và Production checkout, chưa commit):** `litellm_config.yaml`
  (5 alias local → `ollama_chat/qwen3.5:9b-q8_0`, `think: false`, bỏ `CHAT_LAN_*`);
  `tests/test_litellm_routing_config.py`, `tests/test_translation_service.py`;
  bỏ `CHAT_LAN_API_BASE` khỏi `docker-compose*.yml`; `.env*.example`; tài liệu
  (`AGENTS.md`, `README.md`, `Markdowns/{ARCHITECTURE,DEPLOY,MesEXPLAIN}.md`).
- **Đã kiểm tra:** pytest routing/model routes/translation 25 passed ở cả 2 checkout;
  `docker compose -f docker-compose.dev.yml config -q` OK; `git diff --check` sạch.
  Model chạy 100% GPU, khoảng 9 GB, context 8192; GPU tổng ~12.9/16.3 GB.
  Recreate `meibook-litellm-dev`: 5 alias trả lời với `fallbacks: []`, header
  `x-litellm-model-api-base` = `:11435`, 0 fallback. Smoke `/query/stream` trên 8002
  (guest, UI tiếng Nhật, mkac) trả lời đầy đủ trong khoảng 23 s, 14/14 request LiteLLM 200.
- **Chưa làm (cần xác nhận):** recreate `meibook-litellm` Production; sau đó mới
  `ollama rm qwen3.5:9b`. Trước khi Production chuyển, Ollama (`MAX_LOADED_MODELS=1`)
  sẽ phải đổi qua lại giữa 2 model khi có request coder của Production.
- **Rủi ro:** Dev và Production dùng chung một slot Ollama và GPU, nên request phải
  xếp hàng. Nếu Ollama tắt, chat/coder fallback sang Azure; hardware summary
  fail-closed. `.env.docker` thật vẫn còn key `CHAT_LAN_*` (không còn được dùng).

### 2026-10-31 — Thiết lập session harness

- **Mục tiêu:** harness cho các phiên agent; không đổi business logic.
- **Thay đổi:**
  - `CLAUDE.md`: thêm mục "Session Harness" (quy trình bắt đầu/kết thúc phiên,
    validation matrix, mô tả hook).
  - `AGENTS.md`: sửa 2 thông tin lỗi thời (`docker-compose.dev.yml` đã được track;
    frontend đã có `npm test`, vẫn chưa có lint).
  - Mới: `feature_list.json`, `progress.md`, `scripts/harness-check.sh`,
    `.claude/hooks/{session-context,guard-environment,guard-files,post-edit-check}.sh`
    và thư viện dùng chung `.claude/hooks/lib-paths.sh`.
  - `.claude/settings.json`: đăng ký thêm SessionStart, PreToolUse, PostToolUse hook.
- **Đã kiểm tra:** baseline ở "Current state"; kết quả kiểm thử hook và
  `scripts/harness-check.sh full` ghi ở dòng dưới.
- **Kết quả xác minh harness (2026-10-31):**
  - `scripts/harness-check.sh full`: 6/6 PASS (whitespace, pyflakes `src/`, frontend
    `node --test`, JSON, pytest 909 passed, `docker compose -f docker-compose.dev.yml
    config -q`). Thêm file có trailing whitespace thì `quick` trả exit 1; mode sai trả exit 2.
  - `guard-environment.sh`: 100/100 case payload khớp (34 allow, 27 deny, 39 ask);
    bộ chạy test được kiểm chéo bằng case cố tình sai. Hook chạy khoảng 10 ms.
  - `guard-files.sh`: 35/35 case khớp (14 allow, 11 deny, 10 ask).
  - `post-edit-check.sh`: báo đúng JSON lỗi, import thừa, trailing whitespace; im lặng
    với file sạch, file ngoài repo, file secret và dữ liệu runtime.
  - `session-context.sh`: 21 dòng trong Dev; không in gì ngoài 2 checkout Meibook.
  - Hook chỉ có hiệu lực từ phiên Claude Code mới (hoặc sau khi reload settings);
    phiên thiết lập không được hook mới bảo vệ.
  - Bộ case test hook nằm ngoài repo (`/tmp/meibook-harness-test/`), chưa thành
    test chính thức.
- **Phát hiện HR (không sửa trong phiên này):** các tên placeholder người dùng
  nghi ngờ không tồn tại trong DB nhân sự; ngược lại phần lớn tên trong fixture
  `tests/test_query_routing.py` trùng nhân viên thật → `hygiene.test_fixture_pii`.
- **Rủi ro:** hook dựa trên regex, chỉ là lưới an toàn; `.claude/` được track nên
  hook sẽ chạy cả ở checkout Production sau khi merge; Codex/agent khác không chạy
  Claude Code hook.
- **Bước tiếp theo:** review và commit theo nhóm; sửa fixture PII; smoke test
  `guard.cross_turn` trên cổng Dev 8002 khi được phép.
