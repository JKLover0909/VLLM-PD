# CCTVAI Host Metrics — Runbook

> **Phạm vi tài liệu này:** package độc lập `tools/host_metrics/` (collector +
> API đọc chỉ số phần cứng server CCTVAI) và cách triển khai nó trên máy chủ
> Ubuntu CCTVAI. Đây **không phải** service của Meibook: nó không dùng
> Python/Conda host của Meibook, không nằm trong Docker image Meibook, và
> không có quyền ghi/điều khiển gì lên hệ thống — chỉ đọc số liệu.
>
> **Không thuộc phạm vi tài liệu này:** client/service tích hợp phía Meibook
> (`src/integrations/cctvai_hardware_client.py`,
> `src/integrations/cctvai_hardware_service.py`, routing `/query`,
> `/query/stream`, UI). Các phần đó có tài liệu/agent riêng theo plan
> `snoopy-percolating-puffin.md`.

## 1. Mục đích

Người dùng muốn Meibook trả lời câu hỏi tự nhiên về tình trạng phần cứng
server CCTVAI (CPU, RAM, GPU, disk, network, uptime, tiến trình, trạng thái
Docker) mà không cấp cho chatbot bất kỳ khả năng điều khiển/thay đổi hệ
thống nào. Giải pháp: một service **chỉ đọc** (read-only), độc lập hoàn
toàn với Meibook, chạy trên chính server CCTVAI, expose một API HTTPS duy
nhất (`GET /metrics`) mà Meibook Dev gọi trực tiếp.

Không có: remote shell, restart container, xóa/chạy lệnh tùy ý, điều khiển
phần cứng qua chatbot. Service này không biết gì về Meibook, LLM, hay
nghiệp vụ CCTV — nó chỉ đo và trả số liệu hệ thống.

## 2. Kiến trúc

```text
┌─────────────────────────────┐        ┌──────────────────────────┐
│ Ubuntu CCTVAI server         │        │ Meibook Dev (app container)│
│                              │        │                            │
│ ┌────────────┐  snapshot.json│        │  cctvai_hardware_client.py │
│ │ collector   │─────────────▶│        │  (không thuộc package này) │
│ │ (psutil +   │  atomic write│        │                            │
│ │  nvidia-smi │  /run/       │  HTTPS │                            │
│ │  + docker)  │  cctvai-     │  GET   │                            │
│ └────────────┘  metrics/     │ /metrics◀───────────────────────────┤
│        ▲                     │        │  Bearer token              │
│        │ read-only, no shell │        │                            │
│ ┌────────────┐               │        │                            │
│ │  server.py  │──────────────┼────────┘                            │
│ │ (FastAPI +  │  reads snapshot.json (read-only)                   │
│ │  TLS)       │  NOT in docker group, no docker socket              │
│ └────────────┘               │                                     │
└─────────────────────────────┘        └──────────────────────────┘
```

Hai tiến trình (`collector.py` và `server.py`) chạy dưới **hai user
systemd riêng**, không chia sẻ in-memory state — chỉ chia sẻ một file JSON
(`snapshot.json`) trên đĩa, ghi atomic (`os.replace` từ file tạm cùng thư
mục). Nếu API bị chiếm quyền (RCE) nó vẫn không có quyền Docker/GPU vì
không nằm trong nhóm `docker` và không mount socket Docker.

## 3. Cấu trúc package

```text
tools/host_metrics/
├── __init__.py          # rỗng/docstring — KHÔNG import collector/server
├── schemas.py           # contract Pydantic v2 dùng chung (client Meibook cũng import module này)
├── collector.py         # sampler psutil + subprocess nvidia-smi/docker, ghi snapshot atomic
├── server.py            # FastAPI + TLS, đọc snapshot, bearer auth, freshness check
├── requirements.txt     # dependency cô lập, venv riêng — KHÔNG chung với Meibook
└── deploy/
    ├── cctvai-host-metrics-collector.service
    ├── cctvai-host-metrics-api.service
    ├── collector.env.example
    └── api.env.example
```

`schemas.py` **không** import psutil/fastapi/uvicorn và không có I/O lúc
import, để phía Meibook app container có thể `import tools.host_metrics.
schemas` an toàn mà không kéo theo dependency hệ thống. `collector.py` và
`server.py` được phép import psutil/fastapi tự do — hai file này **không**
chạy trong container Meibook, chỉ chạy trong venv riêng trên server
CCTVAI.

## 4. Data contract (v1)

Định nghĩa đầy đủ nằm trong `tools/host_metrics/schemas.py`
(`HostMetricsSnapshot`, `SCHEMA_VERSION = 1`). Tóm tắt:

| Trường | Kiểu | Ghi chú |
|---|---|---|
| `schema_version` | `1` | literal, dùng để phát hiện breaking change |
| `collected_at` | datetime UTC | validator bắt buộc timezone-aware |
| `sample_interval_seconds` | float\|null | null ở sample đầu tiên |
| `cpu` | object\|null | utilization%, load 1/5/15m, core count |
| `memory` | object\|null | total/used/available bytes, percent |
| `gpus[]` | tối đa 8 | index, name, utilization%, VRAM used/total bytes, nhiệt độ °C, công suất W |
| `disks[]` | tối đa 32 | chỉ filesystem thật (đã lọc tmpfs/overlay/proc/...) |
| `network[]` | tối đa 32 | rx/tx bytes/sec tính từ chênh lệch counter |
| `uptime_seconds` | float\|null | |
| `processes[]` | tối đa 10 | **chỉ** tên + CPU%/RAM — không cmdline, không env |
| `containers[]` | tối đa 100 | tên, state, health, restart_count — không env/log/mount |
| `components` | object cố định field | trạng thái từng phần: `ok\|warming_up\|degraded\|unavailable\|error` + `reason` ngắn đã lọc |
| `truncated` | object | cờ `true` cho từng mảng nếu bị cắt bớt — không bao giờ khai danh sách bị cắt là đầy đủ |

Nguyên tắc bắt buộc:

- **Không có giá trị thiếu = 0.** Một chỉ số không đo được là `null`.
- **Sample đầu tiên = "warming_up", không phải 0 giả.** CPU%, network
  rate và per-process CPU% cần 2 lần lấy mẫu để tính delta; lần đầu tiên
  sau khi collector khởi động (hoặc sau khi counter bị reset — ví dụ
  interface mạng restart) component tương ứng báo `warming_up`/`degraded`
  với field số để `null`.
- **Mọi mảng đều có giới hạn cứng** (`max_length` ở Pydantic) và có cờ
  `truncated.*` nếu bị cắt — client không được suy diễn danh sách là đầy
  đủ.
- **Không suy luận:** container `running` không có nghĩa camera stream
  đang hoạt động; snapshot hiện tại không có nghĩa gì về xu hướng lịch sử.
  Đây là việc của tầng diễn giải LLM phía Meibook, không phải của service
  này — service chỉ trả số liệu tất định.

## 5. Bảo mật / cô lập quyền

- **Hai user systemd riêng:**
  - `cctvai-metrics-collector`: là thành viên nhóm `docker` (để đọc `docker
    ps`/`docker inspect`), có quyền chạy `nvidia-smi`. Ghi snapshot vào
    `/run/cctvai-metrics/snapshot.json` (RuntimeDirectory, mode `0750`,
    file mode `0640`).
  - `cctvai-metrics-api`: **không** ở nhóm `docker`, không có Docker
    socket, chỉ đọc snapshot qua nhóm phụ trợ chia sẻ với
    `cctvai-metrics-collector` (đọc, không ghi).
- **Subprocess:** mọi lệnh gọi `nvidia-smi`/`docker` dùng argv cố định,
  `shell=False`, timeout, output bị cắt ở một ngưỡng cứng trước khi parse.
  Không có input người dùng nào chạm tới argv.
- **Auth API:** Bearer token đọc từ file (`HOST_METRICS_TOKEN_FILE`), so
  sánh bằng `secrets.compare_digest` (constant-time, cùng nguyên tắc với
  `/agent` của Meibook nhưng **không dùng chung token**). Thiếu cấu hình
  token → **fail startup**, không có chế độ "auth tắt".
- **TLS:** server chỉ chạy HTTPS (`ssl_certfile`/`ssl_keyfile`), khởi động
  thất bại nếu thiếu cert/key. Chứng thư nên do một CA riêng ký (không
  dùng CA public), SAN đúng hostname/IP mà Meibook Dev sẽ pin. Không bao
  giờ verify=False phía client (nằm ngoài phạm vi package này, nhưng ghi
  chú cho người triển khai client).
- **systemd hardening:** `NoNewPrivileges`, `ProtectSystem=strict`,
  `ProtectHome`, `PrivateTmp`, `RestrictSUIDSGID`,
  `MemoryDenyWriteExecute`, `ReadOnlyPaths`/`ReadWritePaths` giới hạn tối
  đa theo từng service — xem chi tiết trong hai file `.service` ở
  `deploy/`.
- **Ưu tiên CPU/IO thấp cho collector:** `Nice=19`, `CPUWeight=10`,
  `IOSchedulingClass=idle` — collector không được cạnh tranh tài nguyên
  với các container CCTV đang chạy.
- **Không log dữ liệu nhạy cảm.** `reason` trong `components.*` là chuỗi
  ngắn đã lọc (bị cắt về `MAX_REASON_LENGTH`), không bao giờ chứa
  traceback thô hay đường dẫn hệ thống chi tiết.

## 6. Cài đặt trên server Ubuntu CCTVAI

> Các bước dưới đây là hướng dẫn triển khai — **chưa được thực hiện** cho
> tới khi người dùng duyệt và một agent có quyền SSH/triển khai thực hiện
> theo checklist preflight ở phần 1 của plan `snoopy-percolating-puffin.md`
> (xác minh SSH read-only, host key pin, OS/Python/systemd/GPU CLI/quyền
> Docker trước khi làm bất cứ gì).

### 6.1. Tạo user và thư mục

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin cctvai-metrics-collector
sudo useradd --system --no-create-home --shell /usr/sbin/nologin cctvai-metrics-api
sudo usermod -aG docker cctvai-metrics-collector
sudo usermod -aG cctvai-metrics-collector cctvai-metrics-api

sudo mkdir -p /opt/cctvai-host-metrics/app /etc/cctvai-host-metrics/tls
sudo chown -R root:root /opt/cctvai-host-metrics /etc/cctvai-host-metrics
```

### 6.2. Triển khai mã nguồn + venv cô lập

```bash
# Copy CHỈ package độc lập, không copy toàn bộ repo Meibook.
sudo cp -r tools/host_metrics /opt/cctvai-host-metrics/app/tools_host_metrics_src
# (cấu trúc thật: /opt/cctvai-host-metrics/app/tools/host_metrics/... để import path khớp)

sudo python3 -m venv /opt/cctvai-host-metrics/venv
sudo /opt/cctvai-host-metrics/venv/bin/pip install --upgrade pip
sudo /opt/cctvai-host-metrics/venv/bin/pip install -r tools/host_metrics/requirements.txt
```

### 6.3. Sinh token và chứng thư TLS

```bash
# Bearer token — KHÔNG commit, KHÔNG log, KHÔNG đưa vào prompt LLM.
sudo sh -c 'openssl rand -hex 32 > /etc/cctvai-host-metrics/token'
sudo chown root:cctvai-metrics-api /etc/cctvai-host-metrics/token
sudo chmod 0640 /etc/cctvai-host-metrics/token

# TLS: CA riêng + cert cho hostname/IP thật của server CCTVAI, SAN đúng
# host mà Meibook Dev sẽ pin. Không dùng self-signed không SAN, không
# verify=False phía client.
```

### 6.4. Cấu hình môi trường

```bash
sudo cp tools/host_metrics/deploy/collector.env.example /etc/cctvai-host-metrics/collector.env
sudo cp tools/host_metrics/deploy/api.env.example /etc/cctvai-host-metrics/api.env
# Sửa HOST_METRICS_TLS_CERTFILE / HOST_METRICS_TLS_KEYFILE trong api.env
# theo đường dẫn cert/key thật đã tạo ở bước 6.3.
sudo chmod 0640 /etc/cctvai-host-metrics/*.env
sudo chown root:cctvai-metrics-collector /etc/cctvai-host-metrics/collector.env
sudo chown root:cctvai-metrics-api /etc/cctvai-host-metrics/api.env
```

### 6.5. Cài systemd unit

```bash
sudo cp tools/host_metrics/deploy/cctvai-host-metrics-collector.service /etc/systemd/system/
sudo cp tools/host_metrics/deploy/cctvai-host-metrics-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cctvai-host-metrics-collector.service
sudo systemctl enable --now cctvai-host-metrics-api.service
```

### 6.6. Kiểm chứng sau khi cài

```bash
sudo systemctl status cctvai-host-metrics-collector.service
sudo systemctl status cctvai-host-metrics-api.service
sudo cat /run/cctvai-metrics/snapshot.json | python3 -m json.tool | head -30

# Từ Meibook Dev (hoặc máy có quyền mạng tới cổng 8099), test bằng token thật:
curl -sS --cacert /path/to/ca.crt \
  -H "Authorization: Bearer $(sudo cat /etc/cctvai-host-metrics/token)" \
  https://<CCTVAI_HOST>:8099/metrics | python3 -m json.tool

# Test sai token phải bị từ chối:
curl -sS -o /dev/null -w "%{http_code}\n" --cacert /path/to/ca.crt \
  -H "Authorization: Bearer wrong-token" \
  https://<CCTVAI_HOST>:8099/metrics   # kỳ vọng: 401
```

Trước khi bật service thật, kiểm tra cổng `8099` (hoặc cổng khác đã chọn)
đang trống trên server, và firewall hiện có chỉ cho phép Meibook Dev truy
cập cổng này — không mở rộng/flush rule chung.

## 7. Vận hành / rollback

- **Restart collector** (không ảnh hưởng API đang chạy, sẽ tạm mất
  snapshot mới trong lúc restart):
  `sudo systemctl restart cctvai-host-metrics-collector.service`
- **Restart API:**
  `sudo systemctl restart cctvai-host-metrics-api.service`
- **Rollback:** service này độc lập hoàn toàn với các container CCTV hiện
  có — dừng/gỡ hai unit này (`systemctl disable --now ...`) không đụng tới
  bất kỳ dịch vụ CCTV nào đang chạy.
  ```bash
  sudo systemctl disable --now cctvai-host-metrics-api.service
  sudo systemctl disable --now cctvai-host-metrics-collector.service
  ```
- **Xoay vòng token:** ghi token mới vào file, sau đó
  `systemctl restart cctvai-host-metrics-api.service` (token chỉ đọc lúc
  khởi động, không tự động reload).

## 8. Test cục bộ (trước khi triển khai)

Chạy trong checkout `/home/jkl/Code/VLLM-PD-dev` — **không** dùng bare
`python`/`pytest`, luôn qua wrapper `scripts/meibook-python`:

```bash
scripts/meibook-python -m pytest tests/test_host_metrics_schemas.py \
  tests/test_host_metrics_collector.py tests/test_host_metrics_server.py -q

scripts/meibook-python -m pyflakes tools/host_metrics/*.py tests/test_host_metrics_*.py
```

Toàn bộ test dùng fixture giả (monkeypatch `psutil`, subprocess
`nvidia-smi`/`docker`, file snapshot tạm) — **không** gọi hệ thống thật,
không cần server CCTVAI để chạy được. Bao phủ: GPU đủ/thiếu, lỗi
Docker/quyền, timeout, network counter reset, disk filtering (loại
tmpfs/overlay), process top-N filtering, ghi snapshot atomic, auth
đúng/sai/thiếu, snapshot cũ/thiếu/hỏng schema/oversized.

## 9. Sự cố thường gặp

| Triệu chứng | Nguyên nhân khả dĩ | Cách kiểm tra |
|---|---|---|
| API trả `503 Snapshot not available yet` | collector chưa chạy lần nào, hoặc đường dẫn snapshot lệch giữa hai `.env` | `systemctl status cctvai-host-metrics-collector`; so `HOST_METRICS_SNAPSHOT_PATH` hai file env |
| API trả `503 Snapshot is stale` | collector bị crash/dừng, hoặc `HOST_METRICS_MAX_AGE_SECONDS` quá chặt | `journalctl -u cctvai-host-metrics-collector -n 50` |
| `components.gpu.status = "unavailable"` | Không có GPU NVIDIA, hoặc `nvidia-smi` không có trên PATH của user collector | chạy thử `sudo -u cctvai-metrics-collector nvidia-smi` |
| `components.docker.status = "unavailable"`, reason chứa "permission" | user collector chưa thực sự nằm trong nhóm `docker` (cần re-login/restart service sau khi `usermod`) | `id cctvai-metrics-collector`; restart service |
| API `401` dù token đúng | token file có khoảng trắng/newline không mong muốn, hoặc hai bên dùng file token khác nhau | so sánh `sha256sum` file token client/server |
| `cpu`/`network` mãi `warming_up` | collector bị restart liên tục (interval quá ngắn so với chu kỳ crash) | `journalctl -u cctvai-host-metrics-collector` tìm exception lặp lại |
