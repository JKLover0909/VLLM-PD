# Triển khai Docker web nội bộ

Tài liệu này mô tả cách triển khai VLLM-PD lên một máy chủ khác trong mạng nội
bộ công ty. Chế độ này chỉ chạy web hỏi đáp, RAG, Qdrant và LiteLLM. Không chạy
ngrok và không bật Coding Agent.

Người dùng trong công ty sẽ truy cập bằng IP nội bộ của máy chủ:

```text
http://<IP_NỘI_BỘ_MÁY_CHỦ>:8001
```

## Phạm vi triển khai

Các thành phần được chạy:

- `app`: FastAPI + React build + RAG pipeline.
- `qdrant`: vector database.
- `litellm`: model router nội bộ.

Các thành phần không chạy trong chế độ này:

- Ngrok.
- Coding Agent.
- MCP filesystem/git tools.

Trong `docker-compose.web.yml`, chỉ web/API được expose ra LAN qua cổng `8001`.
Qdrant `6333` và LiteLLM `4000` chỉ bind trên `127.0.0.1` của máy chủ.

## File liên quan

```text
Dockerfile
docker-compose.web.yml
.env.docker.example
scripts/docker-deploy.sh
scripts/import_employee_directory.py
scripts/docker-index-mkac.sh
```

## Yêu cầu máy chủ

- Linux.
- Docker và Docker Compose.
- Dung lượng đủ cho image Python/PyTorch/Docling và cache model.
- Nếu chạy GPU: NVIDIA driver và `nvidia-container-toolkit`.
- Nếu không chạy GPU: dùng cấu hình CPU mặc định trong `.env.docker.example`.

## 1. Chuẩn bị mã nguồn

```bash
cd /home/<user>/Code/llm
git clone <repo-url> VLLM-PD
cd VLLM-PD
```

Nếu không dùng Git remote, copy repository sang máy mới và bảo đảm có các thư
mục dữ liệu cần thiết:

```text
documents/MKAC/
config/mkac_manifest.json
```

Cần có file danh sách nhân sự trong thư mục MKAC để tạo SQLite đăng nhập và tra
cứu nhân viên:

```text
documents/MKAC/3. DANH SÁCH KHÁM SỨC KHOẺ 2026.pdf
```

File summary nhân sự có thể copy từ máy cũ hoặc tạo lại ở bước import SQLite:

```text
documents/MKAC/0. Thong tin nhan su va lanh dao MKAC.html
```

## 2. Tạo cấu hình Docker

```bash
cp .env.docker.example .env.docker
nano .env.docker
```

Sửa các API key và URL model:

```env
OLLAMA_API_BASE=...
OPENAI_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
MACHINE2_API_PORT=8001
ENABLE_AGENT=false
```

Kiểm tra thêm các biến SQLite danh bạ nhân viên trong `.env.docker`:

```env
EMPLOYEE_DIRECTORY_SOURCE_DIR=/app/documents/MKAC
EMPLOYEE_DIRECTORY_SOURCE_GLOB="3. DANH SÁCH KHÁM SỨC KHOẺ 2026.pdf"
EMPLOYEE_DIRECTORY_DB_PATH=/app/data/employee_directory.sqlite
EMPLOYEE_DIRECTORY_SUMMARY_PATH="/app/documents/MKAC/0. Thong tin nhan su va lanh dao MKAC.html"
```

Mặc định `.env.docker.example` dùng CPU để dễ chuyển máy:

```env
EMBEDDING_DEVICE=cpu
DOCLING_DEVICE=cpu
EMBEDDING_DTYPE=float32
```

Nếu máy deploy có NVIDIA GPU và đã cài `nvidia-container-toolkit`, có thể đổi:

```env
EMBEDDING_DEVICE=cuda
DOCLING_DEVICE=cuda
EMBEDDING_DTYPE=float16
```

## 3. Build và chạy

```bash
./scripts/docker-deploy.sh
```

Script sẽ:

1. Build image `vllm-pd-web`.
2. Start `app`, `qdrant` và `litellm`.
3. Kiểm tra `/health`.
4. In URL local và URL LAN.

Ví dụ:

```text
VLLM-PD Docker web deployment is running.
Local URL: http://localhost:8001
LAN URL:   http://192.168.1.20:8001
```

## 4. Tạo hoặc copy SQLite danh bạ nhân viên

Các chức năng sau phụ thuộc vào SQLite danh bạ nhân viên:

- Đăng nhập bằng mã nhân viên.
- Lời chào theo tên, chức danh và phòng ban.
- Hỏi theo tên người, ví dụ `Nguyễn Đình Sơn là ai?`.
- Hỏi theo phòng ban, ví dụ `bộ phận của tôi gồm những ai?`.
- Hỏi thông tin cá nhân, ví dụ `tôi tên gì`, `tôi làm bộ phận nào`.

SQLite được lưu tại:

```text
data/employee_directory.sqlite
```

Có hai cách triển khai.

### Cách A: copy SQLite từ máy cũ

Dùng khi muốn triển khai nhanh và dữ liệu nhân sự chưa đổi:

```bash
mkdir -p data
cp /path/to/old/VLLM-PD/data/employee_directory.sqlite data/
```

Nên copy kèm summary nếu có:

```bash
cp "/path/to/old/VLLM-PD/documents/MKAC/0. Thong tin nhan su va lanh dao MKAC.html" \
  "documents/MKAC/"
```

### Cách B: tạo lại SQLite trên máy mới

Dùng khi deploy sạch hoặc vừa cập nhật danh sách nhân sự. Cần chạy sau khi
`./scripts/docker-deploy.sh` đã start container:

```bash
set -a
source .env.docker
set +a
VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml run --rm \
  -e ENABLE_AGENT=false \
  app python scripts/import_employee_directory.py
```

Lệnh này đọc:

```text
documents/MKAC/3. DANH SÁCH KHÁM SỨC KHOẺ 2026.pdf
```

và tạo/cập nhật:

```text
data/employee_directory.sqlite
documents/MKAC/0. Thong tin nhan su va lanh dao MKAC.html
```

Script cũng tự thêm mã cố định:

```text
000001 - Nguyễn Văn Thuận - Giám đốc Meiko Automation
```

Sau khi import, restart app để chắc chắn API đọc DB mới:

```bash
set -a
source .env.docker
set +a
VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml restart app
```

Kiểm tra nhanh:

```bash
curl -fsS http://localhost:8001/health | jq '.employee_directory'
```

Kết quả mong đợi hiện tại:

```json
{
  "db_path": "/app/data/employee_directory.sqlite",
  "employees": 154
}
```

Nếu thiếu SQLite hoặc import lỗi, đăng nhập MKAC sẽ bị từ chối và các câu hỏi
theo nhân viên/phòng ban sẽ không trả lời đúng.

## 5. Index kho MKAC

Sau lần deploy đầu tiên, index tài liệu nội bộ:

```bash
./scripts/docker-index-mkac.sh
```

Script này chạy `scripts/index_mkac_documents.py` bên trong container `app` và
ghi vector vào Qdrant Docker.

Nên chạy import SQLite trước khi index, vì script import tạo/cập nhật file:

```text
documents/MKAC/0. Thong tin nhan su va lanh dao MKAC.html
```

File này cũng là một tài liệu trong kho MKAC, dùng để trả lời các câu hỏi về số
phòng ban, số nhân sự, lãnh đạo và thống kê theo phòng ban.

Nếu thêm hoặc sửa tài liệu MKAC, chạy lại lệnh này. Với một file cụ thể, có thể
vào container hoặc chạy script index thủ công với tham số `--file` nếu cần.

Nếu chỉ cập nhật file danh sách nhân sự, chạy theo thứ tự:

```bash
set -a
source .env.docker
set +a
VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml run --rm \
  -e ENABLE_AGENT=false \
  app python scripts/import_employee_directory.py

./scripts/docker-index-mkac.sh

VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml restart app
```

## 6. Lệnh quản trị

Xem trạng thái:

```bash
set -a
source .env.docker
set +a
VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml ps
```

Xem log app:

```bash
set -a
source .env.docker
set +a
VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml logs -f app
```

Restart app:

```bash
set -a
source .env.docker
set +a
VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml restart app
```

Dừng toàn bộ:

```bash
set -a
source .env.docker
set +a
VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml down
```

## 7. Kiểm tra sau triển khai

Kiểm tra health:

```bash
curl -fsS http://localhost:8001/health | jq .
```

Kiểm tra danh sách model:

```bash
curl -fsS http://localhost:8001/models | jq .
```

Kết quả model ở web MKAC chỉ nên hiển thị `Cloud Model` và `Local Model`.
`Research Model` dùng riêng cho chế độ nghiên cứu và bị ẩn khỏi dropdown MKAC.

Kiểm tra SQLite danh bạ:

```bash
curl -fsS http://localhost:8001/health | jq '.employee_directory'
```

Kiểm tra auth nhân viên:

```bash
curl -fsS -X POST http://localhost:8001/auth/employee \
  -H 'Content-Type: application/json' \
  -d '{"employee_id":"000001"}' | jq .
```

Kiểm tra hỏi theo tên nhân viên:

```bash
SESSION_ID=$(curl -fsS -X POST http://localhost:8001/sessions | jq -r .session_id)

curl -fsS -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"question\":\"Nguyễn Đình Sơn là ai?\",
    \"model\":\"openai\",
    \"mode\":\"mkac\",
    \"employee_id\":\"000001\"
  }" | jq .
```

Kiểm tra LiteLLM nội bộ:

```bash
KEY=$(sed -n 's/^LITELLM_MASTER_KEY=//p' .env.docker)
curl -fsS http://localhost:4000/v1/models \
  -H "Authorization: Bearer $KEY" | jq .
```

## 8. Dữ liệu cần backup

Khi chuyển máy hoặc backup hệ thống, lưu các đường dẫn sau:

```text
documents/MKAC/
config/mkac_manifest.json
data/employee_directory.sqlite
qdrant_storage/
uploads/
logs/
mkac_processed/
```

Nếu muốn deploy sạch, có thể không copy `qdrant_storage/` và chạy lại:

```bash
set -a
source .env.docker
set +a
VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml run --rm \
  -e ENABLE_AGENT=false \
  app python scripts/import_employee_directory.py

./scripts/docker-index-mkac.sh
```

Không nên commit `data/employee_directory.sqlite` nếu danh sách nhân sự là dữ
liệu nội bộ nhạy cảm. Khi cần chuyển máy, copy qua kênh nội bộ an toàn.

## 9. Lưu ý bảo mật

- Không commit `.env.docker`.
- Không commit SQLite danh bạ nhân viên nếu dữ liệu nhân sự là thông tin nội bộ.
- Không public cổng `4000` của LiteLLM ra LAN nếu không cần.
- Không public cổng `6333` của Qdrant ra LAN nếu không cần.
- Chế độ Docker web nội bộ đã đặt `ENABLE_AGENT=false`, nên endpoint `/agent`
  không dùng được.
- Nếu máy chủ có firewall, chỉ cần mở cổng `8001` cho người dùng nội bộ.

## 10. Xử lý lỗi thường gặp

### App khởi động chậm

Lần đầu chạy có thể chậm vì container tải model embedding/OCR. Xem log:

```bash
set -a
source .env.docker
set +a
VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml logs -f app
```

### Không truy cập được từ máy khác

Kiểm tra IP nội bộ:

```bash
hostname -I
```

Kiểm tra container đã publish cổng chưa:

```bash
set -a
source .env.docker
set +a
VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml ps
```

Kiểm tra firewall của máy chủ và bảo đảm cổng `8001` được phép truy cập từ mạng
công ty.

### LiteLLM không gọi được provider

Kiểm tra `.env.docker`:

```env
OPENAI_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
OLLAMA_API_BASE=...
```

Sau khi sửa, restart LiteLLM:

```bash
set -a
source .env.docker
set +a
VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml restart litellm
```

### Đăng nhập mã nhân viên bị lỗi

Kiểm tra file SQLite có tồn tại không:

```bash
ls -lh data/employee_directory.sqlite
```

Kiểm tra API đọc được bao nhiêu nhân viên:

```bash
curl -fsS http://localhost:8001/health | jq '.employee_directory'
```

Nếu `employees` bằng `0` hoặc file không tồn tại, chạy lại import:

```bash
set -a
source .env.docker
set +a
VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml run --rm \
  -e ENABLE_AGENT=false \
  app python scripts/import_employee_directory.py

VLLM_PD_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml restart app
```

### Hỏi tên nhân viên hoặc phòng ban không đúng

Các câu như `Nguyễn Đình Sơn là ai?`, `bộ phận của tôi gồm những ai?` lấy dữ
liệu từ SQLite, không lấy từ Qdrant. Nếu sai dữ liệu:

1. Kiểm tra PDF danh sách nhân sự trong `documents/MKAC/`.
2. Chạy lại import SQLite.
3. Restart `app`.
4. Nếu câu hỏi thống kê tổng hợp vẫn sai, chạy lại `./scripts/docker-index-mkac.sh`
   để cập nhật file summary vào Qdrant.

### Import SQLite không ghi được file summary

Nếu gặp lỗi quyền ghi khi import, kiểm tra mount trong `docker-compose.web.yml`.
Thư mục `documents/MKAC` cần được mount read-write để script tạo/cập nhật:

```text
documents/MKAC/0. Thong tin nhan su va lanh dao MKAC.html
```
