# Dàn ý trình bày VLLM-PD

## Slide 1: Bài toán và bối cảnh

- Người dùng cần hỏi đáp và nghiên cứu tài liệu qua web.
- Tài liệu có nhiều định dạng: PDF, Office, HTML và ảnh.
- Hệ thống cần hỗ trợ cả model local và cloud API.
- Khi chạy trong công ty, người dùng truy cập qua IP nội bộ của máy chủ.

Ghi chú trình bày:
Mở đầu bằng nhu cầu thực tế: tài liệu nhiều, cần tóm tắt có nguồn, cần hỏi đáp
nhanh và cần giao diện web dễ dùng. Điểm quan trọng là hệ thống không chỉ là
chatbot, mà là nền tảng RAG có router model và khả năng mở rộng sang agent.

## Slide 2: Mục tiêu hệ thống

- Một web nội bộ cho người dùng công ty.
- Upload, index và hỏi đáp tài liệu.
- Cho phép người dùng chọn model.
- Ưu tiên MiMo, fallback sang OpenAI và Gemma4 local khi cần.
- Tách riêng chế độ `Hỏi đáp MKAC` và `Nghiên cứu`.

Ghi chú trình bày:
Nhấn mạnh quyết định thiết kế: người dùng chỉ cần vào một URL web. LiteLLM và
Qdrant là thành phần nội bộ, không cần public cho người dùng cuối.

## Slide 3: Công nghệ sử dụng

| Tầng | Công nghệ |
|---|---|
| Frontend | React, Vite, lucide-react, react-markdown |
| API | FastAPI, Pydantic, SSE streaming |
| RAG | Docling, BGE-M3, Qdrant |
| Router model | LiteLLM |
| Agent thử nghiệm | LangGraph, MCP filesystem/git |
| Hạ tầng | Docker Compose, systemd user service, ngrok cho môi trường demo |

Ghi chú trình bày:
Trình bày theo tầng để người nghe không bị ngợp vì danh sách công nghệ. Mỗi
công nghệ có vai trò rõ ràng trong pipeline.

## Slide 4: Model và API được gọi trong hệ thống

| Lựa chọn | Model group | Backend |
|---|---|---|
| Tự động | `auto-model` | MiMo 2.5 Pro, fallback OpenAI, fallback Gemma4 local |
| Gemma4 Local | `local-gemma` | Ollama/Gemma4 trên Máy 1 |
| MiMo 2.5 Pro | `mimo-pro` | Xiaomi MiMo API |
| OpenAI | `openai-model` | GPT-5.4 mini |
| Grok | `grok-model` | Grok 4.20 Reasoning qua Azure |
| Agent | `coding-model` | Gemma4 local, fallback OpenAI |

Ghi chú trình bày:
Giải thích `model group` là tên logic để frontend/backend không phụ thuộc trực
tiếp vào provider. LiteLLM giữ vai trò router và fallback.

## Slide 5: Kiến trúc tổng thể

```text
Máy người dùng trong công ty
    |
    | HTTP, IP nội bộ, cổng 8001
    v
Máy chủ ứng dụng: FastAPI + React
    |-- Hỏi đáp MKAC: Qdrant collection mkac_knowledge
    |-- Nghiên cứu: Upload -> Docling/OCR -> BGE-M3 -> Qdrant
    v
LiteLLM nội bộ :4000
    |-- Máy 1 Ollama/Gemma4
    |-- MiMo API
    |-- OpenAI API
    `-- Azure/Grok API
```

Ghi chú trình bày:
Đây là slide trung tâm. Chỉ ra service nào cho người dùng truy cập, service nào
chỉ chạy nội bộ. Khi demo ngoài mạng công ty có thể dùng ngrok, nhưng khi deploy
nội bộ thì không cần ngrok.

## Slide 6: Luồng RAG tài liệu

1. Người dùng upload tài liệu.
2. FastAPI kiểm tra file, session, extension và dung lượng.
3. Docling parse/OCR nội dung.
4. BGE-M3 sinh embedding cho từng chunk.
5. Qdrant lưu vector theo session hoặc kho MKAC.
6. Khi hỏi, hệ thống retrieve chunk liên quan.
7. Backend dựng prompt và gọi LiteLLM.
8. Frontend hiển thị câu trả lời kèm nguồn.

Ghi chú trình bày:
Nhấn mạnh nguồn trích dẫn: câu trả lời không chỉ là văn bản sinh ra, mà có bằng
chứng từ chunk tài liệu để người dùng đối chiếu.

## Slide 7: Chọn model và fallback

- `auto`: ưu tiên MiMo, fallback sang OpenAI rồi Gemma4 local.
- `local`: chỉ gọi Gemma4 local.
- `mimo`: chỉ gọi MiMo.
- `openai`: chỉ gọi OpenAI.
- `grok`: gọi Grok qua Azure.
- Session có ảnh tự route sang OpenAI Vision nếu model đã chọn không hỗ trợ ảnh.

Ghi chú trình bày:
Đây là phần giải thích tại sao cần LiteLLM. Người dùng chọn model ở UI, backend
ánh xạ sang model group, còn LiteLLM xử lý provider và fallback.

## Slide 8: Hai chế độ sử dụng chính

### Hỏi đáp MKAC

- Dùng kho tài liệu nội bộ MKAC đã index sẵn.
- Không cho upload tài liệu ở chế độ này.
- Nếu không có thông tin nội bộ, hệ thống có thể tìm web với ngữ cảnh MKAC.

### Nghiên cứu

- Người dùng tự upload tài liệu theo phiên.
- Tài liệu chỉ gắn với session nghiên cứu hiện tại.
- Phù hợp để phân tích hồ sơ, PDF, ảnh hoặc tài liệu thử nghiệm.

Ghi chú trình bày:
Slide này giúp phân biệt rõ “tri thức công ty dùng chung” và “tài liệu người
dùng upload tạm thời”.

## Slide 9: Giao diện người dùng

- Web React chạy cùng origin với API.
- Hỗ trợ dark mode, light mode và theo hệ thống.
- Có chọn model, chọn chế độ, upload tài liệu và panel nguồn.
- Trả lời streaming để người dùng thấy phản hồi dần.
- Có cơ chế dừng câu trả lời và sao chép nội dung.

Ghi chú trình bày:
Nên demo trực tiếp UI ở slide này: đổi theme, chuyển chế độ MKAC/Nghiên cứu,
chọn model và xem nguồn trích dẫn.

## Slide 10: Bảo mật và vận hành

- `.env` và `.env.docker` bị ignore, không commit API key.
- Validate UUID session ID.
- Chặn path traversal qua filename validation.
- Allowlist extension upload.
- Rate limit query/upload theo IP.
- Giới hạn queue xử lý tài liệu để tránh quá tải VRAM.
- Docker web nội bộ chỉ expose cổng `8001`.

Ghi chú trình bày:
Nếu trình bày trước hội đồng kỹ thuật, slide này rất quan trọng. Nó cho thấy hệ
thống không chỉ chạy được demo, mà đã có suy nghĩ về vận hành và rủi ro.

## Slide 11: Triển khai

Hai cách chạy chính:

- Local/systemd trên Máy 2: dùng `docker-compose.yml`, FastAPI chạy bằng
  Uvicorn hoặc user service, có thể dùng ngrok khi demo ngoài mạng.
- Docker web nội bộ: dùng `docker-compose.web.yml`, chạy `app`, `qdrant` và
  `litellm`, không chạy ngrok, không bật Coding Agent.

Ghi chú trình bày:
Nhấn mạnh Docker web nội bộ là hướng triển khai sang máy khác trong công ty.
Người dùng truy cập bằng `http://IP_NỘI_BỘ:8001`.

## Slide 12: Kết quả kiểm thử / demo

- React build production thành công.
- UI desktop/mobile đã kiểm tra bằng screenshot trình duyệt.
- Qdrant và LiteLLM chạy bằng Docker.
- Local Gemma4, MiMo Token Plan SGP, OpenAI và Grok gọi được.
- Upload PDF, index chunk và query SSE trả sources.
- Coding Agent có test riêng, nhưng không bật trong bản Docker web nội bộ.

Ghi chú trình bày:
Đây là slide để chốt rằng các thành phần không chỉ nằm trên sơ đồ mà đã được
kiểm thử end-to-end.

## Slide 13: Trade-off và hướng phát triển

Trade-off hiện tại:

- Ngrok URL có thể thay đổi trong môi trường demo.
- Rate limit đang in-memory.
- Chưa có đăng nhập người dùng và phân quyền theo phòng ban.
- Upload tài liệu lớn nên đưa vào job queue chuyên dụng.

Hướng phát triển:

- Domain nội bộ cố định.
- Auth/user management.
- Redis rate limit.
- Lưu lịch sử hội thoại theo người dùng.
- Dashboard log và metrics.
- Queue cho parse/index tài liệu lớn.

Ghi chú trình bày:
Kết thúc bằng những gì đã làm và những gì cần nâng cấp. Cách này giúp bài trình
bày thực tế, không tạo cảm giác hệ thống đã hoàn hảo tuyệt đối.
