# Triển khai Docker web nội bộ Meibook

> **Ranh giới môi trường:** Phần 8 là runbook Dev cho
> `/home/jkl/Code/VLLM-PD-dev` và `docker-compose.dev.yml`. Các phần dùng
> `/home/jkl/Code/VLLM-PD`, port `8001` hoặc `docker-compose.web.yml` là
> Production. Từ 2026-11-17 Dev dùng một model Ollama host (`qwen3.5:9b`) cho mọi alias, Azure chỉ
> fallback; hardware telemetry deterministic. Production chưa đổi; tài liệu này không cấp quyền
> chạy lệnh Production.

Tài liệu này mô tả cách triển khai Meibook theo cấu hình Docker web hiện tại
trong repository `VLLM-PD`. Chế độ này phục vụ web hỏi đáp MKAC/MES, Qdrant,
LiteLLM, SQLite nhân sự, MES snapshot và Gmail send action. Coding Agent tắt
mặc định.

## 1. Mô hình triển khai

Người dùng truy cập:

```text
http://<IP_MAY_CHU>:8001
```

Các service trong `docker-compose.web.yml`:

| Service | Vai trò |
|---|---|
| `app` | FastAPI + React build + RAG/MES/Auth/Gmail |
| `qdrant` | Vector database |
| `litellm` | Model router |
| `ollama-proxy` | Bridge container -> host Ollama (`:11435`), cần cho mọi alias |

Các cổng:

| Service | Cổng host | Ghi chú |
|---|---:|---|
| `app` | `8001` | Publish ra LAN |
| `qdrant` | `127.0.0.1:6333` | Chỉ nội bộ host |
| `litellm` | `127.0.0.1:4002` | Production uses `LITELLM_PROXY_PORT`; container listens on `4000` |
| Ollama host | `172.17.0.1:11435` qua `ollama-proxy` | Model `qwen3.5:9b` dùng chung cho mọi alias |

## 2. Thành phần cần chuẩn bị

### 2.1. Mã nguồn

```bash
cd /home/jkl/Code/VLLM-PD
```

### 2.2. Tài liệu MKAC

Cần có:

```text
documents/MKAC/
documents/MKAC-md/
config/mkac_manifest.json
```

Ý nghĩa:

- `documents/MKAC/`: tài liệu gốc để preview/trích dẫn.
- `documents/MKAC-md/`: text curated để index nhanh và sạch hơn.

### 2.3. MES raw mới

Chỉ dùng bộ raw mới:

```text
database/raw_mkac/
├── M_LOT_202606251410.sql
├── D_ERROR_202606251410.sql
└── P_ERROR_202606251411.sql
```

Không dùng lại `database/raw/` cũ.

### 2.4. Gmail OAuth nếu dùng gửi mail

Các file runtime:

```text
data/gmail_credentials.json
data/gmail_token.json
```

Không commit các file này.

## 3. Cấu hình môi trường

Nếu chưa có `.env.docker`, tạo từ mẫu và điền credentials phù hợp với checkout.
Không chạy lệnh `cp` để ghi đè file env hiện có:

```bash
test -e .env.docker || cp .env.docker.example .env.docker
nano .env.docker
```

Dev dùng một model Ollama host cho mọi alias; Azure chỉ fallback tuần tự. Credentials Azure đặt
trong `.env.docker` ignored, không đưa secret vào Markdown hoặc commit. Chỉ tạo env từ ví dụ khi
file thật chưa tồn tại.

```env
OLLAMA_API_BASE=http://host.docker.internal:11435
TRANSLATION_MODEL=local-qwen-chat
MES_SQL_AGENT_MODEL=local-qwen-coder

# Azure gpt-5.4-mini is used only after primary chat/coder failure.
# Hardware telemetry is summarized deterministically; no LLM alias is configured.

EMPLOYEE_DIRECTORY_DB_PATH=data/employee_directory.sqlite
MES_DATABASE_PATH=data/mes.sqlite
GMAIL_SEND_ENABLED=true
GMAIL_CREDENTIALS_PATH=data/gmail_credentials.json
GMAIL_TOKEN_PATH=data/gmail_token.json
```

Production local hiện dùng host port `4002` qua `LITELLM_PROXY_PORT`; Dev dùng
host port `4001`. Nếu không đặt biến, Compose mapping fallback là host `4000` —
đây không phải host port hiện tại của hai stack. LiteLLM trong container luôn nghe
cổng `4000`. `env_file:` của service không tự cấp biến cho Compose interpolation,
nên các lệnh Production phải truyền cả `--env-file`:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose --env-file .env.docker \
  -f docker-compose.web.yml config -q
```

Trước thay đổi trạng thái, kiểm tra thêm project/container, port và bind source
đã resolve; không in toàn bộ config vì có thể lộ secret. Không cấu hình
`LOCAL_CHAT_NUM_CTX`, `LOCAL_AUX_NUM_CTX`, `CHAT_LAN_API_BASE` hoặc
`CHAT_LAN_API_KEY`; các biến này không còn được sử dụng.

## 4. Model endpoints

Target routing giữ bốn alias nghiệp vụ: `auto-model`, `local-qwen-chat` (chat và
helper/dịch), `local-qwen-coder` (MES/WMS/CCTVAI SQL, Dev: Ollama local `qwen3.5:9b`) và `coding-model` (Coding
Agent). Cả bốn gọi cùng một model `qwen3.5:9b` qua `ollama_chat`; Azure `gpt-5.4-mini` chỉ fallback
tuần tự khi model local lỗi. Production chưa đổi. API không cung cấp lựa chọn cloud trực tiếp;
`model="openai"` từ client cũ chuẩn hóa về model local.

Mọi alias dùng một model Ollama host (`qwen3.5:9b`) qua `OLLAMA_API_BASE`. Endpoint remote `192.84.106.81` đã bị bỏ.

Thinking tắt bằng `think: false`. `max_parallel_requests: 1`, context 8192 và `max_tokens` giữ
bảo thủ để vừa VRAM dùng chung với BGE-M3/Docling.

Hardware metrics **không có model alias**: `CctvaiHardwareService` render số liệu
trực tiếp trên Meibook. Không gửi snapshot, câu hỏi hardware, câu trả lời hay lượt
history hardware tới Qwen/Azure/translation. BGE-M3 và Docling/OCR vẫn chạy local.

Ollama host và proxy `:11435` là phụ thuộc bắt buộc của mọi alias; không dừng hay xóa model
nếu chưa được phê duyệt. Dev và Production dùng chung GPU, nên tránh chạy OCR/index nặng cùng lúc.
QA Agent/QMS là microservice độc lập ngoài routing này.

## 5. Build và chạy

Runbook này không dùng script deploy chung vì script đó build/up toàn bộ services.
Sau preflight và approval riêng cho Production, dùng Compose với env file rõ ràng;
chỉ build/recreate đúng service bị ảnh hưởng:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose --env-file .env.docker \
  -f docker-compose.web.yml config -q
```

Chỉ chạy `up -d <service>` hoặc `build <service>` sau khi xác định rõ checkout,
branch, Compose project, host port và bind mounts. Dev dùng riêng
`docker-compose.dev.yml`; không trộn hai stack.

Kiểm tra:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml ps
curl -fsS http://localhost:8001/health | jq .
```

## 6. Import danh bạ nhân sự

Danh bạ nhân sự dùng cho:

- đăng nhập bằng mã nhân viên;
- hỏi người/phòng ban;
- context người dùng hiện tại;
- guest demo `000000`.

Chạy import:

```bash
set -a
source .env.docker
set +a

MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml run --rm \
  -e ENABLE_AGENT=false \
  app python scripts/import_employee_directory.py
```

Kết quả:

```text
data/employee_directory.sqlite
documents/MKAC/0. Thong tin nhan su va lanh dao MKAC.html
```

Restart app:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml restart app
```

Kiểm tra:

```bash
curl -fsS http://localhost:8001/health | jq '.employee_directory'
```

Kỳ vọng hiện tại:

```json
{
  "db_path": "/app/data/employee_directory.sqlite",
  "employees": 154
}
```

## 7. Import MES snapshot

Chạy importer để tạo `data/mes.sqlite` từ `database/raw_mkac`:

```bash
set -a
source .env.docker
set +a

MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml run --rm \
  -e ENABLE_AGENT=false \
  app python scripts/import_mes_database.py \
    --source-dir /app/database/raw_mkac \
    --db /app/data/mes.sqlite
```

Nếu script hiện tại dùng tham số mặc định, có thể chạy ngắn hơn:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml run --rm \
  -e ENABLE_AGENT=false \
  app python scripts/import_mes_database.py
```

Lưu ý: `database/raw_mkac` không được mount riêng trong
`docker-compose.web.yml`; nó nằm trong image khi build. Nếu thay bộ dump SQL
mới trên host, hãy rebuild image hoặc chạy importer trên host để đảm bảo
container nhìn thấy file mới.

Restart app sau khi import:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml restart app
```

Kiểm tra:

```bash
curl -fsS http://localhost:8001/health | jq '.mes_database'
```

Kỳ vọng hiện tại:

```json
{
  "available": true,
  "lots": 1325,
  "raw_lots": 2592,
  "excluded_test_lots": 1267,
  "error_events": 281,
  "raw_error_events": 654,
  "excluded_test_error_events": 373,
  "error_catalog": 969,
  "unmapped_error_names": 2,
  "sql_agent_available": true
}
```

## 8. Import WMS kho công đoạn MKHC — contract v4 / Phase 2C Dev

> **Chỉ chạy trong Dev:** mọi lệnh trong phần này phải dùng checkout
> `/home/jkl/Code/VLLM-PD-dev`, nhánh `dev`, `docker-compose.dev.yml`, app port
> `8002` và bind mount Dev. Không thay bằng Compose hoặc port Production.

WMS dùng snapshot riêng `data/mes_wms.sqlite`; không nhập vào `mes.sqlite` và
không chạy nguyên Oracle export. Importer chỉ parse INSERT thuộc allowlist, bỏ
qua database link/DDL/procedure/grant/bảng `_TEST` và field nhạy cảm. Contract v4
tách `CURRENT_BALANCE`, `LEGACY_ARCHIVE` và `RAW_TRANSACTION_AUDIT` với evidence,
capability và freshness riêng.

Trước mọi dry-run/import/restart phải xác minh đúng Dev; không in toàn Compose
config vì có thể làm lộ secret đã resolve:

```bash
cd /home/jkl/Code/VLLM-PD-dev
pwd                         # /home/jkl/Code/VLLM-PD-dev
git branch --show-current   # dev
test -f docker-compose.dev.yml
docker compose -f docker-compose.dev.yml config -q
docker compose -f docker-compose.dev.yml config --services
docker compose -f docker-compose.dev.yml ps
```

Kiểm tra chọn lọc resolved Compose và `docker inspect`: project `meibook-dev`,
container `meibook-web-dev`, ports `8002/4001/6334`, bind source
`/home/jkl/Code/VLLM-PD-dev/data` vào `/app/data`. Nếu lệch, dừng thay vì fallback
sang Production.

Stage export trong vùng Dev-approved, ví dụ
`/home/jkl/Code/VLLM-PD-dev/data/staging/mes_wms_export.sql`. Không dùng raw path
từ checkout Production. Dry-run machine-readable phải pass và duplicate count
phải bằng 0:

```bash
scripts/meibook-python scripts/import_mes_wms.py \
  --source /home/jkl/Code/VLLM-PD-dev/data/staging/mes_wms_export.sql \
  --schema database/schema/mes_wms.sql \
  --dry-run \
  --report-json -
```

`PW_CURRENT_ITEM` bắt buộc có dữ liệu và unique theo `(process_id,item_code)`.
`PW_PROCESS`, `PW_SNAPSHORT`, `PW_TRANSACTION`, `PW_TRANSACTION_DEFINE`,
`PW_TRANS_DETAIL` optional; không thấy INSERT được báo `NOT_OBSERVED_IN_EXPORT`.
Chỉ khi report pass mới import atomically vào Dev:

```bash
scripts/meibook-python scripts/import_mes_wms.py \
  --source /home/jkl/Code/VLLM-PD-dev/data/staging/mes_wms_export.sql \
  --schema database/schema/mes_wms.sql \
  --db /home/jkl/Code/VLLM-PD-dev/data/mes_wms.sqlite
scripts/meibook-python scripts/import_mes_wms.py \
  --validate-snapshot /home/jkl/Code/VLLM-PD-dev/data/mes_wms.sqlite \
  --report-json -
```

Cấu hình Dev container:

```env
MES_WMS_DATABASE_ENABLED=true
MES_WMS_DATABASE_PATH=/app/data/mes_wms.sqlite
```

Smoke sau khi nạp code/runtime Dev:

```bash
curl -fsS http://127.0.0.1:8002/health | jq '.mes_wms_database'
```

Syntax legacy archive: nêu đủ mã vật tư + lot vật tư + công đoạn, tùy chọn khoảng
ngày ISO và phân trang, ví dụ: `WMS snapshot mã vật tư ITEM-A lot vật tư LOT-A
công đoạn PROC-A từ 2026-01-01 đến 2026-01-31 trang 1 page size 20`.

Current authoritative theo `(process_id,item_code)` và không trả current theo lot.
Cross-era presence luôn là diagnostic `SUPPRESSED`: archive có/không có exact-key,
current `NOT_EVALUATED`, không kết luận hết tồn/nhập-xuất/delta/trend. Raw audit chỉ
hiển thị code/status/date/quantity thô; không gọi completed movement, không suy
diễn direction/net. Khi chỉ thấy definition/detail nhưng không thấy header,
evidence là `PARTIAL_SOURCE_OBSERVED` và capability audit bị `SUPPRESSED` với
`RAW_TRANSACTION_HEADER_NOT_OBSERVED`; không diễn giải thành dataset vắng hoàn
toàn. UOM/cross-item totals, min-stock, HSD/window-time, trend, completed
movement, WIP, bottleneck và valuation tiếp tục bị khóa.

Production giữ `MES_WMS_DATABASE_ENABLED=false` đến khi release/data lifecycle
riêng được phê duyệt; tuyệt đối không copy SQLite Dev sang Production.

## 9. Index MKAC từ Markdown curated

Index tài liệu MKAC:

```bash
./scripts/docker-index-mkac.sh
```

Script dùng:

```text
MKAC_SOURCE_DIR=/app/documents/MKAC
MKAC_TEXT_SOURCE_DIR=/app/documents/MKAC-md
MKAC_PAGE_IMAGE_DIR=/app/mkac_processed/pages
```

Nghĩa là:

- text lấy từ `documents/MKAC-md` nếu có;
- file gốc và ảnh trang vẫn lấy từ `documents/MKAC`.

Kiểm tra:

```bash
curl -fsS http://localhost:8001/knowledge/mkac/status | jq .
```

Kỳ vọng gần nhất:

```json
{
  "ready": true,
  "collection": "mkac_knowledge",
  "num_documents": 18,
  "num_chunks": 192
}
```

## 9. Build frontend khi sửa UI

Nếu sửa `frontend/src/main.jsx` hoặc `frontend/src/styles.css`, cần build lại:

```bash
cd frontend
npm install
npm run build
cd ..
```

Sau đó restart app nếu FastAPI đang phục vụ `frontend/dist` từ container/image
cũ. Với Docker image build sẵn, cần rebuild image:

```bash
set -a
source .env.docker
set +a

MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml build app
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml up -d app
```

## 10. Gmail OAuth

Nếu đổi email gửi đi hoặc token hết hạn, thay credentials và tạo token mới:

```bash
cp -p data/client_secret_*.json data/gmail_credentials.json
chmod 600 data/gmail_credentials.json
rm -f data/gmail_token.json
python scripts/init_gmail_oauth.py
```

Nếu port OAuth mặc định `8080` bị chiếm:

```bash
python scripts/init_gmail_oauth.py --port 8081
```

Sau khi auth xong, restart app:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml restart app
```

Test gửi mail nên dùng địa chỉ nội bộ hoặc địa chỉ được phép. Tránh chạy test
gửi mail thật hàng loạt.

## 11. Kiểm tra model

Gọi một câu tổng hợp qua LiteLLM Dev (`:4001`) với `fallbacks: []` lần lượt cho từng alias để
không bị Azure che lỗi. Không ghi API key trong shell history, process listing hoặc log.

LiteLLM host ports là Dev `4001` và Production `4002`
(container `4000`). `/v1/models` chỉ xác nhận alias đã nạp, không chứng minh upstream
request hoạt động. Cần xác minh response header `x-litellm-model-api-base` trỏ
`host.docker.internal:11435` và `x-litellm-attempted-fallbacks` bằng 0.

Azure `gpt-5.4-mini` chỉ là fallback chat/coder. Không có alias hardware LLM hoặc
provider cloud trực tiếp. Hardware summary deterministic trên Meibook.

## 12. Kiểm tra API chính

Health:

```bash
curl -fsS http://localhost:8001/health | jq .
```

Models hiển thị UI:

```bash
curl -fsS http://localhost:8001/models | jq .
```

Hiện chỉ nên thấy:

```json
{
  "id": "auto",
  "name": "Local Model"
}
```

Auth guest:

```bash
curl -fsS -X POST http://localhost:8001/auth/employee \
  -H 'Content-Type: application/json' \
  -d '{"employee_id":"000000"}' | jq .
```

Auth nhân viên:

```bash
curl -fsS -X POST http://localhost:8001/auth/employee \
  -H 'Content-Type: application/json' \
  -d '{"employee_id":"000001"}' | jq .
```

Query MKAC:

```bash
SESSION_ID=$(curl -fsS -X POST http://localhost:8001/sessions | jq -r .session_id)

curl -fsS -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"question\":\"Meiko Automation có bao nhiêu phòng ban?\",
    \"model\":\"auto\",
    \"mode\":\"mkac\",
    \"ui_language\":\"vi\",
    \"employee_id\":\"000001\"
  }" | jq .
```

Query MES:

```bash
SESSION_ID=$(curl -fsS -X POST http://localhost:8001/sessions | jq -r .session_id)

curl -fsS -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"question\":\"Mã Lot nào có số lượng lỗi nhiều nhất?\",
    \"model\":\"auto\",
    \"mode\":\"mes\",
    \"ui_language\":\"vi\",
    \"employee_id\":\"000001\"
  }" | jq .
```

Query MES tiếng Nhật:

```bash
SESSION_ID=$(curl -fsS -X POST http://localhost:8001/sessions | jq -r .session_id)

curl -fsS -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"question\":\"2番目にエラーが多いロットはどれですか？\",
    \"model\":\"auto\",
    \"mode\":\"mes\",
    \"ui_language\":\"ja\",
    \"employee_id\":\"000000\"
  }" | jq .
```

## 13. Lệnh quản trị

Xem trạng thái:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml ps
```

Log app:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml logs -f app
```

Log LiteLLM:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml logs -f litellm
```

Restart app:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml restart app
```

Restart LiteLLM:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml restart litellm
```

Dừng toàn bộ:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml down
```

## 14. Backup

Nên backup:

```text
documents/MKAC/
documents/MKAC-md/
config/
data/mes.sqlite
data/employee_directory.sqlite
data/gmail_credentials.json
data/gmail_token.json
qdrant_storage/
mkac_processed/
uploads/
logs/
```

Không commit:

```text
.env
.env.docker
data/
database/raw/
database/raw_mkac/
*.sqlite
client_secret_*.json
gmail_token.json
gmail_credentials.json
```

## 15. Lỗi thường gặp

### App khởi động chậm

Lần đầu có thể tải model embedding/OCR. Xem log:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml logs -f app
```

### Không truy cập được từ máy khác

Kiểm tra IP:

```bash
hostname -I
```

Kiểm tra port:

```bash
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml ps
```

Mở firewall cho cổng `8001` nếu cần.

### LiteLLM không gọi được model local

Kiểm tra `ollama ps` và bridge `meibook-ollama-proxy` (`:11435`). Khi model local lỗi, chat/coder fallback tuần tự sang Azure `gpt-5.4-mini`. Hardware telemetry không đi qua model.

### Đăng nhập nhân viên lỗi

Kiểm tra:

```bash
curl -fsS http://localhost:8001/health | jq '.employee_directory'
```

Nếu `employees` bằng `0`, chạy lại import danh bạ.

### MES trả dữ liệu cũ/lệch

Kiểm tra DB:

```bash
curl -fsS http://localhost:8001/health | jq '.mes_database'
```

Nếu `imported_at` hoặc số dòng không đúng, import lại từ `database/raw_mkac`.

### Câu hỏi MKAC không tìm đúng nguồn

Chạy lại index:

```bash
./scripts/docker-index-mkac.sh
```

Đảm bảo file `.md` trong `documents/MKAC-md` khớp với tài liệu gốc trong
`documents/MKAC`.

### Gmail token hết hạn

Chạy lại:

```bash
rm -f data/gmail_token.json
python scripts/init_gmail_oauth.py
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml restart app
```

## 16. Test sau deploy

Test nhanh trên host qua môi trường Python được quản lý:

```bash
scripts/meibook-python -m pytest tests/test_mes_database.py
scripts/meibook-python -m pytest tests/test_mes_sql_agent.py
scripts/meibook-python -m pytest tests/test_query_routing.py
scripts/meibook-python -m pytest tests/test_gmail_sender.py
```

Test prompt đầy đủ:

```text
Markdowns/TestPrompt.md
```

Không chạy các case gửi email thật nếu chưa kiểm soát người nhận.
