# Tài liệu API — Chatbot CCTV AI

> **Phiên bản tài liệu:** 2026-09-17  
> **Mục đích:** Hướng dẫn tích hợp backend cho frontend chatbot của hệ thống CCTVAI.  
> **Trạng thái:** Gửi đối tác front-end.

---

## 1. Tổng quan

Backend cung cấp API chatbot CCTVAI để trả lời các câu hỏi về:

- **Dữ liệu vi phạm camera** — thống kê sự kiện theo loại vi phạm, camera, tuyến sản xuất (từ PostgreSQL read replica).
- **Phần cứng máy chủ** — trạng thái CPU, RAM, GPU, ổ đĩa, mạng, Docker container.
- **Gửi email** — tổng hợp kết quả và gửi qua Gmail (tuỳ chọn).

Giao tiếp qua hai kiểu:
- **REST (`POST /query`)** — trả một JSON response đầy đủ sau khi xử lý xong.
- **SSE Streaming (`POST /query/stream`)** — trả kết quả từng phần theo chuẩn Server-Sent Events, phù hợp hiển thị dần dần.

> **Khuyến nghị:** Dùng `/query/stream` cho trải nghiệm chat trực quan. Dùng `/query` cho trường hợp cần toàn bộ câu trả lời trước khi render.

---

## 2. URL gốc và xác thực

```
Base URL: http://192.84.106.87:8001
```

> **Lưu ý mạng:** IP `192.84.106.87` là địa chỉ của máy chủ backend trong mạng nội bộ. Nếu frontend chạy ngoài mạng này, cần hỏi team backend về cấu hình tường lửa hoặc URL public (ngrok/cloudflared). API hiện chạy HTTP — nếu triển khai qua HTTPS reverse proxy, URL sẽ được cập nhật.
>
> Không dùng API Key hay Bearer token ở cấp request. Backend chặn bằng rate limit và session. Mọi yêu cầu về thông tin xác thực (CCTVAI DB, phần cứng) được cấu hình phía server — **frontend không cần và không được biết**.

---

## 3. Kiểm tra trạng thái server

### `GET /health`

Trả về trạng thái tất cả subsystem. Frontend nên gọi endpoint này khi khởi động để xác định tính năng nào đang hoạt động.

**Response (JSON) — các trường liên quan CCTVAI:**

> Response thực tế còn chứa nhiều field khác (`qdrant_host`, `mkac_documents`, `mes_database`, v.v.) dành cho các module Meibook khác. Frontend chatbot CCTVAI chỉ cần đọc 3 trường dưới đây.

```jsonc
{
  "status": "healthy",
  // ... nhiều field khác của Meibook ...
  "cctvai_database": {
    "available": true,        // true = truy vấn vi phạm được
    "enabled": true           // đã bật trong cấu hình server
  },
  "cctvai_hardware": {
    "available": true,        // true = truy vấn phần cứng được
    "enabled": true
  },
  "gmail_send": {
    "available": true,        // true = tính năng gửi email hoạt động
    "enabled": true
  }
}
```

**Lưu ý:**
- `cctvai_database.available = false` → câu hỏi về dữ liệu vi phạm sẽ trả lỗi 503.
- `cctvai_hardware.available = false` → câu hỏi về phần cứng sẽ trả lỗi 503.
- Hai subsystem này **độc lập**: một cái hỏng không ảnh hưởng cái kia.
- `gmail_send.available = false` → lệnh gửi email sẽ trả lỗi 503.

---

## 4. Quản lý phiên (Session)

Mỗi cuộc hội thoại cần một `session_id` duy nhất. Frontend tạo và lưu ID này.

### `POST /sessions`

Tạo phiên mới.

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Session created successfully in Qdrant context."
}
```

> **Thực tế:** `session_id` có thể là bất kỳ chuỗi UUID nào do frontend tự tạo. Gọi endpoint này nếu muốn server đảm bảo slot tồn tại, hoặc tự sinh UUID v4 phía frontend.

---

## 5. Câu hỏi gợi ý

### `GET /quick-answers?mode=cctvai&language=vi`

Trả về danh sách câu hỏi gợi ý để hiển thị dưới chatbox hoặc làm nút gợi ý.

**Tham số query:**

| Tham số    | Bắt buộc | Giá trị                 | Mặc định |
|------------|----------|-------------------------|----------|
| `mode`     | Không    | `cctvai`                | `mkac`   |
| `language` | Không    | `vi` hoặc `ja`          | `vi`     |

**Response:**

```jsonc
{
  "mode": "cctvai",
  "language": "vi",
  "short_answer_threshold": 450,
  "max_suggestions": 3,
  "suggestions": [
    {
      "question": "Kiểm tra tình trạng phần cứng server CCTVAI: CPU, RAM, GPU và ổ đĩa",
      "keywords": [],
      "answer": "",
      "live": true   // luôn true với cctvai — phải gọi /query hoặc /query/stream
    },
    {
      "question": "Tổng quan dữ liệu CCTVAI hiện có bao nhiêu sự kiện?",
      "keywords": [],
      "answer": "",
      "live": true
    },
    {
      "question": "7 ngày qua sự kiện vi phạm theo mức độ nghiêm trọng như thế nào?",
      "keywords": [],
      "answer": "",
      "live": true
    }
    // ... thêm các gợi ý khác ...
  ]
}
```

**Gợi ý có sẵn cho mode `cctvai` (tiếng Việt):**

- Kiểm tra tình trạng phần cứng server CCTVAI: CPU, RAM, GPU và ổ đĩa
- Tổng quan dữ liệu CCTVAI hiện có bao nhiêu sự kiện?
- 7 ngày qua sự kiện vi phạm theo mức độ nghiêm trọng như thế nào?
- Danh sách camera đang hoạt động trong hệ thống?
- Độ tin cậy trung bình của mô hình nhận diện là bao nhiêu?

> **Lưu ý:** Gợi ý lập báo cáo tổng quan (`cctvai-report`) **không được dùng** cho chatbot này — tính năng báo cáo đã bị loại bỏ. Frontend nên lọc hoặc bỏ qua gợi ý này nếu server trả về.

**Trường `live`:**
- `true` → câu hỏi được gửi vào pipeline thật khi người dùng bấm (phải gọi `/query` hoặc `/query/stream`).
- `false` → câu trả lời đã có sẵn trong trường `answer`, hiển thị ngay không cần gọi API. Với mode `cctvai`, tất cả gợi ý đều là `live: true`.

> **Khuyến nghị UX:** Hiển thị tối đa `max_suggestions` gợi ý xáo trộn ngẫu nhiên theo từng tin nhắn trả lời. Luôn gửi vào pipeline khi người dùng bấm — không cần đọc trường `answer`.

---

## 6. Hỏi đáp — REST (không streaming)

### `POST /query`

Gửi câu hỏi, nhận câu trả lời hoàn chỉnh.

**Headers:**
```
Content-Type: application/json
```

**Request body:**
```jsonc
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "question": "Hôm nay có bao nhiêu sự kiện vi phạm?",
  "mode": "cctvai",
  "ui_language": "vi",
  "model": "auto",
  "stream": false,
  "conversation_context": []
}
```

**Mô tả các trường:**

| Trường                | Kiểu     | Bắt buộc | Mặc định | Mô tả |
|-----------------------|----------|----------|----------|-------|
| `session_id`          | string   | Có       | —        | UUID phiên hội thoại |
| `question`            | string   | Có       | —        | Nội dung câu hỏi của người dùng |
| `mode`                | string   | Không    | `mkac`   | **Đặt `"cctvai"` cho chatbot CCTVAI** |
| `ui_language`         | string   | Không    | `vi`     | `"vi"` hoặc `"ja"` |
| `model`               | string   | Không    | `auto`   | `"auto"`, `"local"`, `"openai"`. Nên để `"auto"` |
| `stream`              | boolean  | Không    | `true`   | Không dùng với `/query` |
| `conversation_context`| array    | Không    | `[]`     | Lịch sử tin nhắn trước (xem §9) |

**Response (200 OK):**
```jsonc
{
  "answer": "Hôm nay có **47 sự kiện vi phạm**:\n\n- **Cảnh báo:** 23\n- **Thường:** 24",
  "sources": [],
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "local-qwen-small",
  "mode": "cctvai",
  "answer_scope": "cctvai_database",
  "cctvai_metadata": {
    "intent": "cctvai_events_by_severity",
    "domain": "cctvai",
    "status": "PARTIAL",
    "reason_codes": ["CCTVAI_REPLICA_LAG_UNVERIFIED"],
    "latest_event_at": "2026-09-17 10:15",
    "replica_lag_state": "REPLICA_LAG_UNVERIFIED",
    "grain": "severity",
    "schema_version": "1",
    "data_contract_version": "cctvai-replica-v1",
    "semantic_contract_version": "cctvai-report-v1",
    "source_system": "CCTVAI_REPORTING_REPLICA"
  }
}
```

**Lỗi phổ biến:**

| HTTP | Nghĩa |
|------|-------|
| `429` | Quá giới hạn rate (15 request/phút/IP mặc định) |
| `503` | Dịch vụ chưa sẵn sàng (`cctvai_database.available = false`) |

---

## 7. Hỏi đáp — SSE Streaming (khuyến nghị)

### `POST /query/stream`

Cùng request body như `/query`. Server trả kết quả theo Server-Sent Events.

**Headers:**
```
Content-Type: application/json
Accept: text/event-stream
```

**Luồng sự kiện SSE:**

Mỗi sự kiện có dạng:
```
data: {"type": "<loại>", ...}\n\n
```

### Các loại sự kiện (theo thứ tự xuất hiện)

#### `status` — Trạng thái xử lý (thông tin cho UX)
```json
{"type": "status", "message": "Đang xử lý..."}
```
Xuất hiện nhiều lần trong quá trình xử lý. Có thể hiển thị dưới dạng loading spinner hoặc text trạng thái. **Không phải câu trả lời.**

#### `sources` — Nguồn tham chiếu
```json
{"type": "sources", "sources": []}
```
Với mode `cctvai`, `sources` luôn là mảng rỗng `[]` (không có tài liệu RAG). Nhận sự kiện này trước `token` để chuẩn bị vùng hiển thị.

#### `meta` — Metadata của câu trả lời
```json
{
  "type": "meta",
  "model": "local-qwen-small",
  "mode": "cctvai",
  "answer_scope": "cctvai_hardware",
  "cctvai_metadata": {
    "intent": "cctvai_hardware",
    "domain": "cctvai",
    "status": "PARTIAL",
    "reason_codes": [],
    "replica_lag_state": ""
  }
}
```
Nhận trước `token`. Dùng `answer_scope` để biết câu trả lời thuộc loại gì (xem §8).

#### `token` — Nội dung câu trả lời (streaming)
```json
{"type": "token", "content": "Tình trạng phần cứng máy chủ CCTV AI:\n\n- **CPU:** 11% (48 lõi logic)..."}
```
> ⚠️ Với `/query/stream`, toàn bộ nội dung câu trả lời được gửi trong **một sự kiện `token` duy nhất** (không chia nhỏ từng chữ như giao thức streaming cổ điển). Frontend nhận và render trực tiếp.

#### `done` — Kết thúc stream
```json
{"type": "done"}
```
Nhận sự kiện này thì đóng kết nối SSE.

#### `error` — Lỗi trong quá trình xử lý
```json
{"type": "error", "message": "CCTVAI query service is not available."}
```

### Ví dụ xử lý SSE (JavaScript)

```javascript
const eventSource = new EventSourcePolyfill('/query/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: sessionId,
    question: userMessage,
    mode: 'cctvai',
    ui_language: 'vi',
    model: 'auto',
    conversation_context: conversationHistory
  })
});

let answer = '';

eventSource.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'status') {
    showLoadingMessage(data.message);
  } else if (data.type === 'meta') {
    // Lưu answer_scope để hiển thị badge
    currentAnswerScope = data.answer_scope;
  } else if (data.type === 'token') {
    // Nối nội dung (trong thực tế thường chỉ có 1 token lớn)
    answer += data.content;
    renderMarkdown(answer);
  } else if (data.type === 'done') {
    eventSource.close();
    hideLoadingSpinner();
  } else if (data.type === 'error') {
    showError(data.message);
    eventSource.close();
  }
});
```

> **Lưu ý CORS:** Nếu frontend chạy trên domain khác, cần cấu hình proxy hoặc CORS phía server. Hỏi team backend về domain được phép.

---

## 8. Phân loại câu trả lời theo `answer_scope`

Trường `answer_scope` trong `meta` và response JSON cho biết câu hỏi được xử lý bởi subsystem nào.

| `answer_scope`       | Nghĩa |
|----------------------|-------|
| `cctvai_database`    | Trả lời từ dữ liệu vi phạm PostgreSQL replica |
| `cctvai_hardware`    | Trả lời từ dữ liệu phần cứng máy chủ (CPU/RAM/GPU/disk/network) |
| `cctvai_database` + `reason_codes: ["CCTVAI_SQL_AGENT_ANSWER_UNVERIFIED"]` | SQL Agent LLM tạo câu trả lời — chưa kiểm chứng, độ tin cậy thấp hơn |
| `email_action`       | Hành động gửi email (tạo nháp, xác nhận, huỷ) |

**Khuyến nghị UI:** Hiển thị badge hoặc chú thích nhỏ để người dùng biết nguồn dữ liệu. Ví dụ:
- `cctvai_database` → "📊 Dữ liệu CCTVAI"
- `cctvai_hardware` → "🖥️ Phần cứng máy chủ"
- Có `CCTVAI_REPLICA_LAG_UNVERIFIED` trong `reason_codes` → "⚠️ Dữ liệu replica, độ trễ chưa xác minh"

---

## 9. Lịch sử hội thoại (`conversation_context`)

Để model hiểu câu hỏi nối tiếp (ví dụ: "hôm qua thì sao?", "còn camera B thì sao?"), frontend phải gửi kèm lịch sử tin nhắn gần nhất.

**Cấu trúc:**
```json
"conversation_context": [
  {"role": "user", "content": "Hôm nay có bao nhiêu vi phạm?"},
  {"role": "assistant", "content": "Hôm nay có 47 sự kiện vi phạm..."}
]
```

**Quy tắc:**
- Mỗi phần tử gồm `role` (`"user"` hoặc `"assistant"`) và `content` (nội dung text).
- Giữ tối đa **6 lượt** gần nhất (3 cặp user-assistant) để tránh vượt giới hạn token.
- Nếu gửi quá dài, backend tự cắt bớt (tối đa 1500 ký tự/lượt).
- Có thể gửi mảng rỗng `[]` nếu bắt đầu cuộc trò chuyện mới.

---

## 10. Phân loại câu hỏi tự động

Backend tự phân loại câu hỏi — frontend **không cần** phân loại thủ công.

### Câu hỏi về phần cứng máy chủ

Tự động nhận dạng khi câu hỏi chứa từ khoá liên quan đến CPU, RAM, GPU, ổ đĩa, mạng, Docker, nhiệt độ, thời gian hoạt động... (tiếng Việt, tiếng Anh, tiếng Nhật đều được nhận).

Ví dụ câu hỏi về phần cứng:
- "CPU đang sử dụng bao nhiêu %?"
- "RAM còn trống bao nhiêu?"
- "GPU nhiệt độ bao nhiêu độ?"
- "Bao nhiêu container Docker đang chạy?"
- "サーバーのメモリ使用率は？"

### Câu hỏi về dữ liệu vi phạm CCTVAI

Các câu hỏi không khớp phần cứng sẽ được route vào hệ thống truy vấn dữ liệu vi phạm với 7 loại intent:

| Intent | Câu hỏi ví dụ |
|--------|--------------|
| Thống kê theo mức độ (`severity`) | "Có bao nhiêu vi phạm cảnh báo hôm nay?" |
| Thống kê theo loại vi phạm | "Loại vi phạm nào xảy ra nhiều nhất?" |
| Thống kê theo camera/tuyến | "Camera nào ghi nhận nhiều vi phạm nhất?" |
| Sự kiện gần đây | "Cho tôi xem 10 vi phạm gần nhất" |
| Camera đang hoạt động | "Danh sách camera đang hoạt động" |
| Sự kiện không có camera | "Có sự kiện nào không khớp camera không?" |
| Tổng quan dữ liệu | "Tổng quan dữ liệu CCTVAI" |

Nếu câu hỏi ngoài phạm vi trên, hệ thống trả lời: *"Câu hỏi này chưa khớp với các nội dung tôi tra cứu được. Bạn muốn xem thống kê vi phạm theo loại/camera/tuyến...?"*

### Câu hỏi ngoài phạm vi (không trả lời)

Backend sẽ từ chối (trả thông báo, không thực thi) các yêu cầu:
- Thay đổi cấu hình camera, khởi động lại hệ thống, xóa dữ liệu
- Lấy thông tin credential camera (username/password/URL camera)
- Danh tính người phụ trách tuyến (chỉ có user_id nội bộ, không có tên/email)
- Tính thời gian kéo dài của sự kiện đang mở (không có trong dữ liệu replica)

---

## 11. Gửi email (tuỳ chọn)

Nếu `gmail_send.available = true` (kiểm tra từ `/health`), người dùng có thể dùng câu lệnh tự nhiên để gửi email kết quả.

> **Lưu ý:** Tính năng này chỉ gửi được nếu đã cấu hình Gmail OAuth phía server. Frontend chỉ cần gửi câu lệnh văn bản — không cần xử lý email logic.

### Luồng 3 bước

**Bước 1 — Tạo email nháp:**

Người dùng nhập:
```
gửi email cho nguyen.van.a@cty.vn báo thống kê vi phạm hôm nay
```

Hoặc dạng ngắn gọn hơn:
```
send email to nguyen.van.a@cty.vn thống kê vi phạm hôm nay
```

Server trả về (`answer_scope: "email_action"`):
```
Đã tạo bản nháp email:
- Gửi đến: nguyen.van.a@cty.vn
- Tiêu đề: Meibook - Thống kê vi phạm hôm nay

Nhập "xác nhận gửi email" để gửi, hoặc "hủy gửi email" để huỷ.
```

**Bước 2 — Xác nhận gửi:**

```
xác nhận gửi email
```

Hoặc tiếng Nhật: `メール送信を確定`

Server gửi email thật và trả xác nhận.

**Bước 3 — Huỷ (tuỳ chọn):**

```
hủy gửi email
```

Hoặc tiếng Nhật: `メール送信をキャンセル`

### Ràng buộc email
- Địa chỉ email phải hợp lệ.
- Chỉ gửi email từ `session_id` đang hoạt động — không thể gửi thay session khác.
- Nội dung email lấy từ câu trả lời gần nhất trong cùng phiên.

---

## 12. Giới hạn và lưu ý triển khai

### Rate limiting
- Mặc định: **15 request / phút / IP** cho endpoint `/query` và `/query/stream`.
- Vượt giới hạn trả `HTTP 429`.

### Dữ liệu replica không phải real-time
- Dữ liệu vi phạm từ `cctvai_database` đồng bộ bất đồng bộ — thường dưới vài giây nhưng không đảm bảo tuyệt đối.
- Câu trả lời luôn kèm cảnh báo "dữ liệu replica" trong `reason_codes: ["CCTVAI_REPLICA_LAG_UNVERIFIED"]`.
- **Không dùng để ra quyết định nghiệp vụ cần real-time tuyệt đối.**

### Dữ liệu phần cứng
- Cache 5 giây phía server — nhiều user hỏi cùng lúc không gây spike.
- Snapshot cũ hơn 30 giây bị từ chối, trả lỗi rõ ràng.

### Markdown trong câu trả lời
- Trường `answer` chứa Markdown (in đậm `**...**`, danh sách `- `, xuống dòng `\n`).
- **Frontend cần render Markdown** để hiển thị đúng định dạng (có thể dùng `marked.js`, `react-markdown`, v.v.).

### Ngôn ngữ
- Backend hỗ trợ câu hỏi tiếng Việt và tiếng Nhật ở `mode=cctvai`.
- `ui_language` điều khiển ngôn ngữ của câu trả lời — nên đặt đúng theo UI người dùng đang dùng.

---

## 13. Checklist tích hợp cho frontend

Trước khi go-live, kiểm tra các mục sau:

- [ ] Gọi `/health` khi khởi động, xử lý `available = false` cho từng subsystem.
- [ ] Tạo/lưu `session_id` cho mỗi cuộc hội thoại (không hardcode).
- [ ] Gửi đúng `mode: "cctvai"` trong mọi request.
- [ ] Render Markdown trong trường `answer`.
- [ ] Gửi `conversation_context` với tối đa 6 lượt gần nhất.
- [ ] Xử lý `HTTP 429` — hiển thị thông báo "Vui lòng thử lại sau" và không retry tự động quá nhanh.
- [ ] Xử lý `HTTP 503` — hiển thị thông báo dịch vụ tạm không khả dụng.
- [ ] Đóng kết nối SSE sau khi nhận sự kiện `done` hoặc `error`.
- [ ] Hiển thị badge phân biệt nguồn dữ liệu theo `answer_scope`.
- [ ] Với email: chỉ hiển thị tính năng khi `gmail_send.available = true`.

---

## 14. Ví dụ hoàn chỉnh (curl)

### Hỏi về phần cứng máy chủ

```bash
curl -X POST http://192.84.106.87:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "question": "RAM và CPU hiện tại như thế nào?",
    "mode": "cctvai",
    "ui_language": "vi",
    "model": "auto",
    "conversation_context": []
  }'
```

**Response:**
```json
{
  "answer": "📊 **Tình trạng phần cứng máy chủ CCTV AI:**\n\n- **CPU:** 11% (48 lõi logic)\n- **Bộ nhớ RAM:** 32% (39.8GB / 124.8GB)",
  "sources": [],
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "local-hardware-summary",
  "mode": "cctvai",
  "answer_scope": "cctvai_hardware"
}
```

### Hỏi về dữ liệu vi phạm

```bash
curl -X POST http://192.84.106.87:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "question": "Hôm nay có bao nhiêu sự kiện vi phạm theo mức độ?",
    "mode": "cctvai",
    "ui_language": "vi",
    "model": "auto",
    "conversation_context": []
  }'
```

### Hỏi streaming

```bash
curl -X POST http://192.84.106.87:8001/query/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "question": "Camera nào ghi nhận nhiều vi phạm nhất?",
    "mode": "cctvai",
    "ui_language": "vi",
    "model": "auto",
    "conversation_context": []
  }'
```
