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

## 2. Tạo cấu hình Docker

```bash
cp .env.docker.example .env.docker
nano .env.docker
```

Sửa các API key và URL model:

```env
OLLAMA_API_BASE=...
OPENAI_API_KEY=...
MIMO_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
MACHINE2_API_PORT=8001
ENABLE_AGENT=false
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

## 4. Index kho MKAC

Sau lần deploy đầu tiên, index tài liệu nội bộ:

```bash
./scripts/docker-index-mkac.sh
```

Script này chạy `scripts/index_mkac_documents.py` bên trong container `app` và
ghi vector vào Qdrant Docker.

Nếu thêm hoặc sửa tài liệu MKAC, chạy lại lệnh này. Với một file cụ thể, có thể
vào container hoặc chạy script index thủ công với tham số `--file` nếu cần.

## 5. Lệnh quản trị

Xem trạng thái:

```bash
VLLM_PD_ENV_FILE=.env.docker docker compose \
  --env-file .env.docker \
  -f docker-compose.web.yml ps
```

Xem log app:

```bash
VLLM_PD_ENV_FILE=.env.docker docker compose \
  --env-file .env.docker \
  -f docker-compose.web.yml logs -f app
```

Restart app:

```bash
VLLM_PD_ENV_FILE=.env.docker docker compose \
  --env-file .env.docker \
  -f docker-compose.web.yml restart app
```

Dừng toàn bộ:

```bash
VLLM_PD_ENV_FILE=.env.docker docker compose \
  --env-file .env.docker \
  -f docker-compose.web.yml down
```

## 6. Kiểm tra sau triển khai

Kiểm tra health:

```bash
curl -fsS http://localhost:8001/health | jq .
```

Kiểm tra danh sách model:

```bash
curl -fsS http://localhost:8001/models | jq .
```

Kiểm tra LiteLLM nội bộ:

```bash
KEY=$(sed -n 's/^LITELLM_MASTER_KEY=//p' .env.docker)
curl -fsS http://localhost:4000/v1/models \
  -H "Authorization: Bearer $KEY" | jq .
```

## 7. Dữ liệu cần backup

Khi chuyển máy hoặc backup hệ thống, lưu các đường dẫn sau:

```text
documents/MKAC/
config/mkac_manifest.json
qdrant_storage/
uploads/
logs/
mkac_processed/
```

Nếu muốn deploy sạch, có thể không copy `qdrant_storage/` và chạy lại:

```bash
./scripts/docker-index-mkac.sh
```

## 8. Lưu ý bảo mật

- Không commit `.env.docker`.
- Không public cổng `4000` của LiteLLM ra LAN nếu không cần.
- Không public cổng `6333` của Qdrant ra LAN nếu không cần.
- Chế độ Docker web nội bộ đã đặt `ENABLE_AGENT=false`, nên endpoint `/agent`
  không dùng được.
- Nếu máy chủ có firewall, chỉ cần mở cổng `8001` cho người dùng nội bộ.

## 9. Xử lý lỗi thường gặp

### App khởi động chậm

Lần đầu chạy có thể chậm vì container tải model embedding/OCR. Xem log:

```bash
VLLM_PD_ENV_FILE=.env.docker docker compose \
  --env-file .env.docker \
  -f docker-compose.web.yml logs -f app
```

### Không truy cập được từ máy khác

Kiểm tra IP nội bộ:

```bash
hostname -I
```

Kiểm tra container đã publish cổng chưa:

```bash
VLLM_PD_ENV_FILE=.env.docker docker compose \
  --env-file .env.docker \
  -f docker-compose.web.yml ps
```

Kiểm tra firewall của máy chủ và bảo đảm cổng `8001` được phép truy cập từ mạng
công ty.

### LiteLLM không gọi được provider

Kiểm tra `.env.docker`:

```env
OPENAI_API_KEY=...
MIMO_API_KEY=...
MIMO_API_BASE=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
OLLAMA_API_BASE=...
```

Sau khi sửa, restart LiteLLM:

```bash
VLLM_PD_ENV_FILE=.env.docker docker compose \
  --env-file .env.docker \
  -f docker-compose.web.yml restart litellm
```
