# Triển khai Docker web nội bộ Meibook

> **Ranh giới môi trường:** Phần 8 là runbook Dev cho
> `/home/jkl/Code/VLLM-PD-dev` và `docker-compose.dev.yml`. Các phần dùng
> `/home/jkl/Code/VLLM-PD`, port `8001` hoặc `docker-compose.web.yml` là
> Production và không được chạy khi đang hoàn thiện WMS Phase 2C Dev.

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
| `ollama-proxy` | Bridge container -> Ollama host local |

Các cổng:

| Service | Cổng host | Ghi chú |
|---|---:|---|
| `app` | `8001` | Publish ra LAN |
| `qdrant` | `127.0.0.1:6333` | Chỉ nội bộ host |
| `litellm` | `127.0.0.1:4000` | Chỉ nội bộ host |
| `ollama-proxy` | `172.17.0.1:11435` | Docker gateway -> host Ollama |

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

Tạo file `.env.docker`:

```bash
cp .env.docker.example .env.docker
nano .env.docker
```

Các nhóm biến quan trọng:

```env
MACHINE2_API_PORT=8001

LITELLM_MASTER_KEY=sk-local
OPENAI_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...

QWEN_CHAT_API_BASE=http://192.168.10.124:11434
QWEN_CHAT_NGROK_API_BASE=https://carless-overarch-establish.ngrok-free.dev
QWEN_SMALL_API_BASE=http://host.docker.internal:11435
QWEN_CODER_LAN_API_BASE=http://192.168.10.14:11434/v1
QWEN_CODER_LAN_API_KEY=sk-local
QWEN_CODER_NGROK_API_BASE=https://your-qwen-coder-tunnel.example/v1
QWEN_CODER_NGROK_API_KEY=sk-local

# Role-specific cloud fallback credentials (Azure is attempted before OpenAI).
OPENAI_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/openai/v1
CHAT_FALLBACK_MODEL=azure-chat-fallback

TRANSLATION_MODEL=local-qwen-small
MES_SQL_AGENT_MODEL=local-qwen-coder

EMPLOYEE_DIRECTORY_DB_PATH=data/employee_directory.sqlite
MES_DATABASE_PATH=data/mes.sqlite

GMAIL_SEND_ENABLED=true
GMAIL_CREDENTIALS_PATH=data/gmail_credentials.json
GMAIL_TOKEN_PATH=data/gmail_token.json
```

Trong `docker-compose.web.yml`, các biến model cũng được truyền trực tiếp vào
service `litellm` để đảm bảo container thấy đúng endpoint.

## 4. Ollama local model phụ trên máy host

Model phụ `qwen2.5:3b-instruct` chạy trên host qua systemd Ollama. Container
LiteLLM gọi qua `ollama-proxy`:

```text
http://host.docker.internal:11435
```

Kiểm tra host Ollama:

```bash
curl -fsS http://localhost:11434/api/tags | jq .
```

Kiểm tra qua proxy sau khi Docker chạy:

```bash
curl -fsS http://172.17.0.1:11435/api/tags | jq .
```

## 5. Build và chạy

Chạy script deploy:

```bash
./scripts/docker-deploy.sh
```

Hoặc chạy thủ công:

```bash
set -a
source .env.docker
set +a

MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml build
MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml up -d
```

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

Kiểm tra LiteLLM models:

```bash
KEY=$(sed -n 's/^LITELLM_MASTER_KEY=//p' .env.docker)

curl -fsS http://localhost:4000/v1/models \
  -H "Authorization: Bearer $KEY" | jq .
```

Test chat qua LiteLLM:

```bash
curl -fsS http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto-model",
    "messages": [{"role": "user", "content": "Trả lời ngắn gọn: OK"}],
    "max_tokens": 16
  }' | jq .
```

Endpoint gốc cần biết:

| Endpoint | Vai trò |
|---|---|
| `http://192.168.10.124:11434` | Qwen3 IP tĩnh, route chính nếu sống |
| `https://carless-overarch-establish.ngrok-free.dev` | Qwen3 ngrok fallback |
| `http://host.docker.internal:11435` | Qwen small qua host Ollama proxy |
| `https://d3d2-...ngrok-free.app/v1` | Qwen Coder OpenAI-compatible |

Nếu IP tĩnh Qwen3 `connection refused`, LiteLLM sẽ fallback sang ngrok rồi
OpenAI theo `litellm_config.yaml`.

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

### LiteLLM không gọi được Qwen3 IP tĩnh

Nếu endpoint `192.168.10.124:11434` lỗi, kiểm tra trực tiếp:

```bash
curl -fsS http://192.168.10.124:11434/api/tags | jq .
```

Nếu lỗi `connection refused`, hệ thống vẫn có thể chạy qua ngrok fallback. Tuy
nhiên nên khôi phục server Qwen3 để giảm phụ thuộc ngrok/cloud fallback.

### Qwen3 trả reasoning nhưng rỗng content

Không gọi Qwen3 qua Ollama `/v1`. Trong LiteLLM phải dùng:

```yaml
model: ollama_chat/qwen3:14b
api_base: http://...:11434
think: false
```

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
