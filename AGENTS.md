# AGENTS.md

Tài liệu này là hướng dẫn chính thức (Onboarding Guide) dành cho bất kỳ AI Agent nào tiếp nhận, bảo trì hoặc phát triển thêm tính năng cho dự án **Meibook (VLLM-PD)**. Bắt buộc phải đọc kỹ trước khi thực hiện thay đổi codebase.

---

## 0. Bối cảnh Worktree Dev hiện tại

**Local topology ngày 2026-07-14:** checkout `/home/jkl/Code/VLLM-PD-dev` trên nhánh `dev` là môi trường phát triển độc lập của Meibook, không phải checkout đang phục vụ Production. Chỉ áp dụng mặc định Dev bên dưới khi `pwd`, branch và Compose config khớp topology này; nếu tài liệu được merge sang `main`, clone ở đường dẫn khác hoặc thiếu `docker-compose.dev.yml`, phải xác định lại môi trường thay vì áp dụng máy móc.

| Môi trường | Đường dẫn / nhánh | Compose | Web | LiteLLM | Qdrant | Container chính |
| --- | --- | --- | --- | --- | --- | --- |
| Production local | `/home/jkl/Code/VLLM-PD` / `main` | `docker-compose.web.yml` | mặc định `8001`, có thể override bằng `MACHINE2_API_PORT` | mặc định `127.0.0.1:4000`, có thể override bằng `LITELLM_PROXY_PORT` | mặc định `127.0.0.1:6333`, có thể override bằng `QDRANT_PORT` | `meibook-web`, `meibook-litellm`, `meibook-qdrant` |
| Dev local | `/home/jkl/Code/VLLM-PD-dev` / `dev` | `docker-compose.dev.yml` (file local chưa được Git track tại thời điểm 2026-07-14) | `0.0.0.0:8002` | `127.0.0.1:4001` | `127.0.0.1:6334` | `meibook-web-dev`, `meibook-litellm-dev`, `meibook-qdrant-dev` |

### 0.1 Mục đích

* Mọi tính năng và sửa lỗi phải được phát triển và kiểm thử trong worktree/stack Dev trước khi review và merge sang `main`; chạy smoke test Dev khi thay đổi có bề mặt runtime cần xác minh end-to-end.
* Stack Dev có bản sao riêng của SQLite, Qdrant storage, documents, preview images, uploads và logs để thử nghiệm mà không sửa dữ liệu Production. Compose Dev và `.env.docker` hiện là local state, không tự xuất hiện trong clone/worktree mới.
* Git chỉ merge **source code, schema, config và tài liệu đã được review**. Không bao giờ merge/copy ngược SQLite, Qdrant storage, uploads, logs hoặc dữ liệu thử nghiệm từ Dev sang Production. `docjp_processed/` hiện có preview legacy đang được Git track; không stage output reindex Dev cùng code. Mọi thay đổi preview tracked phải được review riêng và xác nhận là artifact phát hành có chủ đích.
* Khi thay đổi schema dữ liệu, phải viết/chạy migration hoặc import có kiểm soát riêng cho từng môi trường; không thay thế dữ liệu production bằng bản Dev.

### 0.2 Mức cô lập và tài nguyên dùng chung

* Tên container, Docker network, host port và các bind-mount dữ liệu được tách nhờ hai checkout vật lý khác nhau.
* Dev vẫn dùng chung Docker daemon, GPU/VRAM, image tag và các Docker volume cache HuggingFace/Torch/EasyOCR với Production.
* Cấu hình Dev hiện tại **không chạy `meibook-ollama-proxy-dev` hay port `11436`**; LiteLLM Dev dùng chung proxy/model upstream qua `host.docker.internal:11435`. Không được giả định hai stack độc lập hoàn toàn ở tầng model.
* Các biến `QWEN_*` khai báo trực tiếp trong service `litellm.environment` hiện thắng giá trị cùng tên trong `.env.docker`; sửa env file không đổi các endpoint này nếu Compose chưa được cập nhật.
* Web Dev được chủ đích publish `0.0.0.0:8002` để dùng trên LAN. Employee ID/guest hiện chỉ là access gate, không phải xác thực mạnh; đây là quyết định vận hành cho môi trường demo hiện tại, không tự đổi sang loopback nếu người dùng chưa yêu cầu.
* `ENABLE_AGENT=false` chỉ tắt endpoint Coding Agent `/agent`; không tắt MES SQL Agent, Report Agent, Gmail hoặc Calendar. Dev chủ đích dùng cùng tài khoản demo Gmail/Calendar đang hoạt động trên Production qua các bản sao credential/token trong `data/`; không gửi email hoặc tạo Calendar event trong smoke test nếu người dùng chưa yêu cầu rõ.
* Do `./src:/app/src:ro` che dependency trong image, worktree Dev hiện cần bản sao local ignored `src/agent/node_modules` được khởi tạo từ dependency tree Production/lockfile trước khi app chạy Calendar. Compose mount credential/token Dev vào package này; không merge `node_modules` hay OAuth state vào Git.

### 0.3 Guardrail bắt buộc cho Agent

Trước mọi lệnh Docker, import/index, migration hoặc thao tác dữ liệu có thay đổi trạng thái, phải kiểm tra `pwd`, branch hiện tại, Compose file, container đích, host port và resolved bind-mount. Mặc định chỉ thao tác Dev:

```bash
pwd
git branch --show-current
git status --short
test -f docker-compose.dev.yml
docker compose -f docker-compose.dev.yml config -q
docker compose -f docker-compose.dev.yml config --services
docker compose -f docker-compose.dev.yml ps
```

`config -q` chỉ kiểm tra cú pháp và `config --services` chỉ xác nhận service names. Trước thao tác stateful, phải dùng output chọn lọc từ resolved config/`docker inspect` để xác minh Compose project, `container_name`, published port và bind source. Không in toàn bộ `docker compose config` vào chat/log vì output có thể chứa biến môi trường hoặc secret đã resolve.

* Không chạy `docker-compose.web.yml`, `scripts/docker-deploy.sh`, restart/stop/down container không có hậu tố `-dev`, hoặc truy cập các cổng Production đã resolve (mặc định `8001/4000/6333`) nếu người dùng chưa yêu cầu rõ thao tác Production.
* Không dùng lệnh Docker mơ hồ thiếu `-f docker-compose.dev.yml` khi preflight xác nhận đang ở worktree Dev local. Nếu file này không tồn tại hoặc không hợp lệ, dừng lại và báo người dùng; không tự chuyển sang Compose Production.
* Không chạy `down -v`, drop/clear collection, xóa storage, reindex/import toàn bộ, gửi Gmail, tạo Calendar event, commit, merge, push hoặc deploy nếu chưa được người dùng yêu cầu/xác nhận rõ.
* Không copy nóng `qdrant_storage`; nếu cần làm mới Dev từ Production, dùng Qdrant snapshot/restore nhất quán theo runbook được duyệt.
* Không đọc hoặc hiển thị `.env`, OAuth token, dữ liệu HR/MES, Qdrant payload, uploads hay logs nhạy cảm nếu không thật sự cần thiết.

### 0.4 Snapshot khởi tạo Dev ngày 2026-07-14

Người vận hành đã báo cáo stack Dev được khởi tạo từ snapshot nhất quán và smoke test thành công: `docjp_knowledge` 938 points, `docmind_documents` 104 points, `mkac_knowledge` 192 points; Employee SQLite có 154 nhân sự; MES SQLite có 1.325 lot; `/health`, LiteLLM liveliness và một lời gọi thật qua `local-qwen-small` hoạt động. Đây chỉ là **snapshot theo thời điểm**, không phải invariant; agent phải đo lại trước khi dùng các con số này để kết luận.

## 1. Project Overview

**Meibook (VLLM-PD)** là một hệ thống Chatbot AI nội bộ đa năng được phát triển cho doanh nghiệp (MKAC). Hệ thống hỗ trợ song ngữ (Tiếng Việt và Tiếng Nhật) với 3 luồng nghiệp vụ chính:

1. **Hỏi đáp Hành chính nhân sự (HR/MKAC):** Tra cứu quy định, nội quy, thông tin nhân sự. Kết hợp tìm kiếm cấu trúc qua SQLite và RAG trên Qdrant.
2. **Quản lý Sản xuất (MES):** Tra cứu thông tin mã Lot, mã hàng, thống kê lỗi từ hệ thống MES.
3. **Nghiên cứu tài liệu (Research):** Cho phép người dùng chọn các nhóm tài liệu có sẵn (NotebookLM-like) để đặt câu hỏi chuyên sâu, trích dẫn nguồn cụ thể.

## 2. System Architecture

Kiến trúc Client-Server, triển khai bằng Docker Compose: `docker-compose.web.yml` cho Production và file local `docker-compose.dev.yml` cho topology Dev hiện tại.

* **Frontend:** ReactJS + Vite + CSS thuần. SPA render tin nhắn qua Server-Sent Events (SSE).
* **Backend:** FastAPI (Python). Xử lý định tuyến, auth, session, rate limit, gọi module RAG/MES.
* **Vector Database:** Qdrant. Lưu trữ embedding (`mkac_knowledge`, `docjp_knowledge`).
* **LLM Gateway:** LiteLLM Proxy. Điều phối các truy vấn tới mô hình Local (Qwen) qua Ollama hoặc Cloud (OpenAI, Azure) dựa trên fallback chain.
* **Database (Structured):** SQLite. Dùng để lưu trữ dữ liệu nhân sự và snapshot dữ liệu sản xuất.
* **Ingestion Pipeline:** Sử dụng `Docling` và `PyMuPDF` để OCR/parse, cắt chunk có overlap, sinh ảnh preview, dùng `BAAI/bge-m3` để tạo embedding (1024-dim).

## 3. Important Directory Structure

* `frontend/src/`: Giao diện (React). `main.jsx` chứa logic chính, `styles.css` chứa CSS.
* `src/api/`: Khởi tạo FastAPI (`main.py`), định nghĩa API schemas (`schemas.py`), config (`config.py`), topics (`research_topics.py`).
* `src/rag/`: Chứa core logic RAG (`rag_pipeline.py`, `vector_store.py`, `parser.py`, `prompts.py`, `media_paths.py`).
* `src/integrations/`: Xử lý nghiệp vụ MES (`mes_query_service.py`, `mes_sql_agent.py`) và tích hợp ngoài (Gmail).
* `src/auth/`: Quản lý hỏi đáp cấu trúc cho HR.
* `src/i18n/`: Dịch thuật tự động qua `translation.py`.
* `scripts/`: Các file thực thi (import DB, index Qdrant, evaluation).
* `tests/`: Thư mục Pytest.
* `config/`: Chứa file manifest (`docjp_manifest.json`, `research_topics.json`).

## 4. Main Workflows

### 4.1 HR / MKAC Q&A

Nhận diện intent xem có phải câu hỏi cấu trúc không (lấy từ SQLite). Nếu không, tạo embedding bằng BGE-M3 → tìm kiếm dense cosine trong collection `mkac_knowledge` → dựng prompt có nguồn/context → gọi LLM. Hỗ trợ dịch câu hỏi tiếng Nhật sang tiếng Việt trước retrieval khi nhánh structured HR chưa khớp; runtime hiện chưa có CrossEncoder reranker.

### 4.2 MES Q&A

Parse thông số (mã Lot, Mã hàng, Thời gian). Ưu tiên deterministic/rule-based SQL cho MES vì dễ kiểm chứng, ổn định và ít hallucination hơn LLM-generated SQL. Tuy nhiên agent vẫn phải kiểm tra schema, dữ liệu đầu vào, điều kiện lọc và kết quả trả về trước khi kết luận. Nếu phức tạp, gọi `mes_sql_agent` (LLM-based) làm fallback.

### 4.2b WMS Q&A (Dev extension) — chính sách LLM-generated SQL

WMS (`mode=wms`) đọc riêng `data/mes_wms.sqlite`. Luồng mặc định vẫn là deterministic intent + SQL viết tay trong `src/integrations/mes_wms_database.py`, vì kết quả kiểm chứng được và ổn định.

**Quyết định vận hành hiện tại:** khi câu hỏi không khớp intent deterministic hoặc bị deterministic guardrail từ chối vì semantics chưa xác minh, WMS **được phép** fallback sang LLM-generated SQL (`wms_sql_agent`) thay vì dừng ở clarification/suppression. Người dùng đã chấp nhận đánh đổi này để tăng độ phủ câu hỏi, sau khi được cảnh báo rõ các rủi ro bên dưới.

**Rủi ro đã được chấp nhận** — agent phải hiểu là chúng có thật, không phải chỉ là cảnh báo hình thức:

* **Cộng số lượng khác UOM.** Schema `database/schema/mes_wms.sql` không có cột đơn vị tính và không có master quy đổi. Một câu `SUM(quantity_decimal)` chạy qua nhiều `item_code` sẽ ra con số vô nghĩa nhưng trông hợp lệ.
* **So sánh cross-era.** Dữ liệu current balance và legacy archive khác grain, khác key, khác freshness. LLM rất dễ `JOIN` hoặc `UNION` hai era như thể chúng tương thích.
* **Suy diễn WIP / bottleneck / trend / min-stock / expiry.** Snapshot không chứa các khái niệm này; LLM có thể tự dựng chúng từ các cột sẵn có và trình bày như sự thật nghiệp vụ.

**Ràng buộc bắt buộc khi implement hoặc sửa nhánh fallback này:**

* Chỉ read-only, chỉ chạy qua allowlisted view, có row limit và timeout như MES SQL Agent hiện tại.
* Các suppression ngữ nghĩa sẵn có (`_cross_item_aggregate_suppression`, `_completed_movements_suppressed`, WIP ambiguity, unsupported KPI) chỉ là kết quả deterministic; WMS được phép chuyển tiếp chúng sang `wms_sql_agent`. Chỉ các lỗi hạ tầng/contract khiến snapshot không thể truy vấn an toàn (disabled, unavailable, incompatible, query error) mới phải fail-closed trước LLM.
* Kết quả từ LLM-generated SQL phải được đánh dấu là suy luận độ tin cậy thấp trên cả metadata lẫn UI, phân biệt rõ với kết quả deterministic.
* Không trình bày số liệu từ nhánh này như dữ liệu đã kiểm chứng hợp đồng.

Khi một dạng câu hỏi trở nên phổ biến, ưu tiên "nâng cấp" nó thành intent deterministic + SQL viết tay + test, thay vì để nó sống mãi ở nhánh fallback.

### 4.3 Research / NotebookLM-like Document QA

Research có hai scope: topic dùng collection `docjp_knowledge` với filter `metadata.category`, còn tài liệu người dùng upload được cô lập theo session trong `docmind_documents`. Không dùng double-translation đối với tài liệu tiếng Nhật để tránh nhiễu RAG.

### 4.4 Source Preview / Citation Flow

Nhấn vào nguồn trên UI, API `/sources/preview` đọc file ảnh PNG (được generate lúc index) từ ổ cứng (`docjp_processed` hoặc `mkac_processed`). Hàm `resolve_processed_image_path` trong `src/rag/media_paths.py` xử lý mapping đường dẫn từ Host (`/home/...`) vào Docker (`/app/...`).

## 5. How to Run the Project

Khi preflight xác nhận đúng worktree Dev local và `docker-compose.dev.yml` hợp lệ, mặc định dùng Compose Dev:

* **Restart app Dev sau khi sửa Python hoặc config app bị cache:** `docker compose -f docker-compose.dev.yml restart app`.
* **Restart LiteLLM sau khi sửa `litellm_config.yaml`:** `docker compose -f docker-compose.dev.yml restart litellm`.
* **Recreate sau khi đổi biến môi trường, port, mount hoặc network:** `docker compose -f docker-compose.dev.yml up -d <service-bị-ảnh-hưởng>`. Nếu thay `MEIBOOK_ENV_FILE` hoặc `.env.docker`, phải xác minh lại resolved target mà không in secret.
* **Build Frontend/source dependency:** chạy `cd frontend && npm ci && npm run build` trong worktree mới (hoặc chỉ `npm run build` khi dependency đúng lockfile đã có). Phải build trước lần `docker compose up` đầu tiên và bảo đảm `frontend/dist` ghi được bởi user hiện tại; nếu Docker đã tạo thư mục rỗng/root-owned, dừng và xử lý ownership có chủ đích. `frontend/dist` được bind-mount read-only nên không cần rebuild app image chỉ vì đổi React/CSS hoặc `frontend/package.json`.
* **Rebuild image:** cần khi đổi Python/system dependency, `src/agent/package.json` hoặc Dockerfile. Tuy nhiên mount `./src:/app/src:ro` che `src/agent/node_modules` đã cài trong image; topology local hiện dùng bản sao ignored tại `src/agent/node_modules` trong từng checkout. Khi dependency Agent thay đổi, cập nhật bản sao Dev theo `src/agent/package-lock.json` hoặc bản Production đã kiểm chứng trước khi recreate app. Không merge `node_modules`, không tự build tag `latest` dùng chung với Production; thiết kế bền vững hơn là image/tag Dev và volume dependency riêng.
* **Index/import dữ liệu:** mọi lệnh `--reindex`, import hoặc prune là thao tác dữ liệu, không phải bước chạy thông thường. Nếu script hỗ trợ `--dry-run`, phải dùng trước; nếu không (ví dụ MES importer), hãy kiểm tra input/schema, backup DB đích và dùng đường dẫn staging trước khi thay thế có chủ đích. Luôn giới hạn phạm vi và chỉ chạy thật khi người dùng yêu cầu rõ. Với tài liệu Research được duyệt dài đến 200 trang, giữ giới hạn tường minh như `MAX_DOCUMENT_PAGES=200`; lưu ý dry-run index hiện không khởi tạo parser nên không phát hiện lỗi giới hạn trang.

Sau khi code Dev được kiểm thử, quy trình chuẩn là review diff → commit/merge code từ `dev` vào `main` khi được yêu cầu → áp dụng đúng bước Production theo loại thay đổi (build frontend trong checkout Production vì `frontend/dist` bị ignore/bind-mount, restart service, recreate hoặc rebuild image). Dữ liệu runtime Dev không tham gia merge.

## 6. How to Test, Lint, Build and Verify

* **Host Python bắt buộc:** dùng `scripts/meibook-python` để chạy Python trong Conda `meibook-dev` (Python 3.10); không dùng bare `python`, `python3`, `pip`, `pytest`, `pyflakes` hoặc Conda `base`. Wrapper mặc định ép embedding/OCR sang CPU để không cạnh tranh GPU Production; chỉ đặt `MEIBOOK_ALLOW_GPU=1` cho tác vụ GPU đã được phê duyệt.
* **Backend targeted test (mặc định trong vòng lặp phát triển):** chạy file/case liên quan bằng `scripts/meibook-python -m pytest <test-path> -q`.
* **Backend suite trước review/merge hoặc thay đổi rộng:** `scripts/meibook-python -m pytest tests/ -q --ignore=tests/test_mes_integration.py --ignore=tests/test_mkac_pipeline.py`.
* **Backend static check:** `scripts/meibook-python -m pyflakes src/api/*.py src/rag/*.py src/integrations/*.py src/auth/*.py src/actions/*.py src/i18n/*.py src/agent/*.py` (nếu môi trường đã cài `pyflakes`).
* **Frontend Build:** `cd frontend && npm run build`. Một lần build vừa tạo artifact vừa xác minh bundling; hiện `frontend/package.json` chưa có script `lint` hoặc `test`, vì vậy không báo đã lint nếu chỉ chạy build.
* **Compose validation:** `docker compose -f docker-compose.dev.yml config -q`; cần chạy khi Compose/env/target thay đổi hoặc trong preflight cho lệnh stateful, không cần lặp lại vô ích. Dùng `config --services`, `ps` và inspect/query chọn lọc để kiểm tra target; không xuất toàn bộ resolved config có thể chứa secret.
* **Dev liveness:** `curl -fsS http://localhost:8002/health`, `curl -fsS http://localhost:4001/health/liveliness`, `curl -fsS http://localhost:6334/healthz`.
* **Xem log API/Preview Dev:** `docker logs meibook-web-dev --tail 50` (tránh log dữ liệu nhạy cảm trong báo cáo).

`/health` chỉ là liveness/startup-level signal, không chứng minh Qdrant retrieval, LiteLLM model upstream, MES, Gmail hay Calendar end-to-end. Chạy smoke test không nhạy cảm trên cổng Dev `8002` khi thay đổi ảnh hưởng runtime/API/SSE/RAG/MES/UI interaction; thay đổi docs, CSS thuần hoặc hàm thuần có thể dùng phép kiểm tra targeted phù hợp hơn để tránh gọi shared GPU/model upstream không cần thiết.

## 7. Development Conventions

* **Coding Style:** Tên biến, tên hàm, code comment và commit message phải tuân theo convention hiện có của repo (ưu tiên tiếng Anh kỹ thuật ngắn gọn).
* **Schemas:** Cẩn trọng khi đổi `schemas.py` vì có thể làm frontend React hỏng payload.
* **UI/UX:** Cần giữ dark mode, responsive, dùng CSS thuần trong `styles.css`. Giao diện phải chuyên nghiệp, rõ luồng người dùng.

## 8. Agent Working Rules

* **Ngôn ngữ giao tiếp:** Luôn trao đổi, giải thích và báo cáo kết quả với người dùng bằng tiếng Việt, trừ khi người dùng yêu cầu tiếng Anh.
* **Làm thật, không đoán mò:** Luôn đọc log, phân tích nguyên nhân gốc trước khi sửa code.
* **Test-driven:** Sửa code xong phải chạy test. Nếu test fail, trước tiên phải xác định nguyên nhân (code sai, test lỗi thời, thiếu dependency). Chỉ cập nhật test khi logic/spec mới đã được xác nhận hợp lệ; nếu không, phải sửa code.
* **Tác động dây chuyền:** Khi sửa API, frontend, hoặc DB, phải kiểm tra các module bị ảnh hưởng.
* **Tối ưu RAG:** Ưu tiên chất lượng câu trả lời, bám sát nguồn tài liệu, trích dẫn rõ ràng, giảm hallucination.
* **Chủ động rà soát:** Đề xuất cải tiến khi thấy code chưa tối ưu, nhưng vẫn phải ưu tiên hoàn thành task chính trước.

## 9. Data, Security and Privacy Rules

* **Tuyệt đối không:** `git push`, `git reset --hard`, xóa file lớn, drop database, clear Qdrant collection, hoặc reindex toàn bộ nếu người dùng chưa yêu cầu rõ.
* **Không tự ý commit** nếu người dùng chưa yêu cầu.
* **Thao tác nguy hiểm:** Phải giải thích trước và chờ người dùng xác nhận.
* **Bảo mật:** Không hard-code API key, token, password. Không log dữ liệu nhạy cảm (mã nhân viên, dữ liệu HR/MES thật). Không đưa dữ liệu thật vào test/prompt nếu chưa ẩn danh.
* **Anti-hallucination:** Với câu hỏi không có trong nguồn, chatbot phải nói rõ "không tìm thấy thông tin", cấm tự suy diễn quy định công ty. Không để LLM tự bịa số liệu MES.

## 10. Debugging Guide

* **Preview image not found:** Kiểm tra log `docker logs meibook-web-dev --tail 50`. Lỗi thường do đường dẫn mount Host/Docker lệch, hoặc PDF chưa được sinh ảnh lúc index.
* **Giao diện web không cập nhật:** Quên chạy `npm run build` hoặc chưa Hard Reload (`Ctrl + Shift + R`) trình duyệt.
* **ERR_NGROK_334:** URL ngrok bị chiếm bởi máy khác. Với Dev, tạo tunnel mới tới cổng `8002` bằng config rỗng (ví dụ `ngrok http 8002 --config=none`) hoặc dùng cloudflared; không thay tunnel Production nếu chưa được yêu cầu.
* **Test Token Budget Fail:** Thường do sửa cấu hình `MAX_TOKENS` trong `.env` / `rag_pipeline.py` nhưng quên cập nhật số expected trong `test_token_budgets.py`.
* **Nhầm môi trường:** Nếu kết quả khác dự kiến, kiểm tra Docker Compose project, suffix `-dev`, host port và bind source trước khi sửa dữ liệu. Không suy luận môi trường chỉ từ tên service `app`.

## 11. Definition of Done

A task is only considered done when:

1. The requested change is implemented within scope.
2. Related frontend/backend/API/database impacts are checked.
3. Relevant test, lint, build, or smoke check has been run, or the reason for not running is clearly stated.
4. Changed files are summarized.
5. Verification result and remaining risks are reported to the user.

## 12. Known Weak Points / Future Improvements

* `main.jsx` đang rất lớn (>3000 dòng), cần chia nhỏ thành các React Components.
* Luồng upload Research theo session đã hoạt động nhưng cần tiếp tục kiểm thử isolation, preview và cleanup trên collection `docmind_documents`.
* Truy vấn tiếng Việt trên kho tài liệu tiếng Nhật (Research) có thể được tối ưu thêm bằng HyDE (Hypothetical Document Embeddings).
* Chưa có cảnh báo MES tự động khi tỷ lệ lỗi vượt ngưỡng.
