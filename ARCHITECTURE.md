# Kiến trúc hệ thống VLLM-PD

Tài liệu này mô tả kiến trúc của luồng chạy chính trong repository `VLLM-PD`, dựa trên mã nguồn và cấu hình được đối chiếu ngày **05/06/2026**.

Các phát biểu về cấu trúc phần mềm, endpoint, model logic và luồng dữ liệu được lấy trực tiếp từ mã nguồn. Trạng thái dịch vụ đang chạy, URL ngrok, số lượng MCP tool hoặc kết quả kiểm thử trên một máy cụ thể là thông tin vận hành theo thời điểm và cần được xác minh lại bằng các lệnh ở phần [Vận hành và kiểm tra](#18-vận-hành-và-kiểm-tra).

> **Phạm vi:** `src/`, `frontend/`, `docker-compose.yml`, `litellm_config.yaml` và `.env` ở thư mục gốc tạo thành hệ thống chính. Thư mục `docmind/` là demo hoặc phiên bản thử nghiệm tách biệt, không thuộc đường chạy chính được mô tả trong tài liệu này.

## 1. Mục tiêu và phạm vi chức năng

VLLM-PD cung cấp hai nhóm chức năng độc lập nhưng dùng chung API Gateway và LiteLLM:

### 1.1. Không gian hỏi đáp và nghiên cứu tài liệu

- Tạo phiên làm việc theo UUID.
- Tải lên nhiều định dạng tài liệu.
- Trích xuất nội dung bằng Docling.
- Chia nội dung thành các đoạn nhỏ có chồng lấn.
- Sinh vector embedding bằng `BAAI/bge-m3`.
- Lưu kho MKAC dùng chung trong collection `mkac_knowledge`.
- Lưu tài liệu nghiên cứu theo phiên trong collection `docmind_documents`.
- Tạo prompt RAG từ các đoạn liên quan.
- Gọi mô hình ngôn ngữ thông qua LiteLLM.
- Trả lời đồng bộ hoặc streaming bằng Server-Sent Events (SSE).
- Tách ba chế độ `Hỏi đáp hành chính nhân sự MKAC`, `Quản lý MES` và
  `Nghiên cứu tài liệu` với session, lịch sử và nguồn dữ liệu riêng.
- Chế độ hành chính nhân sự và MES cho chọn `Cloud Model` hoặc `Local Model`;
  Grok được giữ riêng cho chế độ `Nghiên cứu tài liệu`.

### 1.2. Coding Agent

- Nhận tác vụ tại endpoint `POST /agent`.
- Dùng LangGraph để điều phối vòng lặp giữa mô hình và công cụ.
- Gọi mô hình logic `coding-model` qua LiteLLM.
- Nạp công cụ filesystem và Git qua MCP.
- Dùng bộ công cụ file cục bộ giới hạn trong workspace khi MCP không khả dụng.
- Có thể bảo vệ endpoint bằng `AGENT_API_KEY`.

## 2. Các nguyên tắc kiến trúc

Hệ thống hiện tuân theo các nguyên tắc chính sau:

1. **Một cổng vào cho người dùng:** FastAPI phục vụ cả REST/SSE API và React SPA trên cổng `8001`.
2. **Tách model vật lý khỏi ứng dụng:** mã nguồn chỉ gọi các tên model logic của LiteLLM.
3. **Tách kho dữ liệu:** MKAC dùng khóa logic `session_id=mkac`; tài liệu nghiên cứu lọc theo UUID phiên.
4. **Xử lý nặng ngoài event loop:** parse, embedding và thao tác Qdrant chính trong luồng upload/query được chuyển sang thread bằng `asyncio.to_thread`.
5. **Định tuyến mô hình có fallback:** route web mặc định `auto-model` ưu tiên OpenAI rồi mới fallback sang Grok/Gemma4, còn `coding-model` giữ đường dự phòng qua OpenAI.
6. **Giới hạn phạm vi công cụ Agent:** MCP filesystem và bộ công cụ fallback bị ràng buộc bởi `WORKSPACE_DIR`; Git MCP được gắn với `AGENT_REPOSITORY_DIR`.

## 3. Bố cục repository

```text
VLLM-PD/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI, REST, SSE, static frontend
│   ├── rag/
│   │   ├── parser.py            # Docling và chunking
│   │   ├── embedder.py          # BGE-M3
│   │   ├── vector_store.py      # Qdrant
│   │   └── rag_pipeline.py      # Retrieval, prompt, gọi LiteLLM
│   ├── integrations/
│   │   ├── mes_client.py        # Gọi MES thời gian thực
│   │   ├── mes_database.py      # Truy vấn MES snapshot read-only
│   │   └── mes_sql_agent.py     # Text-to-SQL có validate cho MES snapshot
│   └── agent/
│       ├── graph.py             # Đồ thị LangGraph
│       └── mcp_client.py        # MCP và công cụ fallback
├── frontend/
│   ├── src/main.jsx             # React SPA
│   ├── src/styles.css
│   └── vite.config.js
├── docker-compose.yml           # Qdrant và LiteLLM
├── litellm_config.yaml          # Model logic và fallback
├── .env.example                 # Mẫu cấu hình
├── requirements.txt
├── tests/
└── docmind/                     # Demo/nhánh thử nghiệm, không phải runtime chính
```

## 4. Mô hình triển khai ba máy

| Máy | Trách nhiệm | Thành phần chính |
|---|---|---|
| Máy 1 | Máy chủ suy luận local | Ollama, model `gemma4:latest`, có thể được Máy 2 truy cập qua ngrok |
| Máy 2 | Máy chủ ứng dụng | FastAPI, React build, Docling, BGE-M3, Qdrant, LiteLLM, LangGraph và MCP client |
| Máy 3 | Máy khách | Trình duyệt hoặc client gọi API của Máy 2 |

Sơ đồ tổng thể:

```text
                         HTTPS, thường qua ngrok
┌──────────────────┐    URL tạm do ngrok cấp lúc khởi động
│ Người dùng/Máy 3 │ ─────────────────────────────────────────┐
└──────────────────┘                                          │
                                                              ▼
                                                ┌──────────────────────────┐
                                                │ Máy 2: FastAPI :8001    │
                                                │                          │
                                                │ React SPA tại /          │
                                                │ REST API                 │
                                                │ SSE /query/stream        │
                                                │ Coding Agent /agent      │
                                                └────────────┬─────────────┘
                                                             │
                          ┌──────────────────────────────────┼───────────────────────┐
                          │                                  │                       │
                          ▼                                  ▼                       ▼
               ┌────────────────────┐             ┌──────────────────┐   ┌──────────────────┐
               │ RAG pipeline       │             │ LangGraph Agent  │   │ frontend/dist    │
               │ Docling            │             │ MCP filesystem   │   │ React static     │
               │ BGE-M3             │             │ MCP Git           │   └──────────────────┘
               │ Qdrant retrieval   │             └─────────┬────────┘
               └──────────┬─────────┘                       │
                          └──────────────────┬───────────────┘
                                             ▼
                                  ┌──────────────────────┐
                                  │ LiteLLM Proxy :4000 │
                                  └──────┬──────┬───────┘
                                         │      │
                         ┌───────────────┘      └────────────────┐
                         ▼                                       ▼
              ┌──────────────────────┐                ┌─────────────────────┐
              │ Máy 1: Ollama       │                │ Cloud providers     │
              │ Gemma4 local        │                │ Grok / OpenAI       │
              └──────────────────────┘                └─────────────────────┘

Máy 2 còn chạy Qdrant :6333/:6334 để lưu vector và payload tài liệu.
```

## 5. Thành phần runtime và cổng mạng

| Thành phần | Cổng mặc định | Cách chạy | Chức năng |
|---|---:|---|---|
| FastAPI + React | `8001` | Uvicorn hoặc user service | API Gateway, SSE và SPA |
| LiteLLM | `4000` | Docker Compose | Proxy OpenAI-compatible và model router |
| Qdrant REST | `6333` | Docker Compose | API vector database |
| Qdrant gRPC | `6334` | Docker Compose | Giao thức gRPC của Qdrant |
| Ollama trên Máy 1 | `11434` | Ngoài repository | Suy luận Gemma4 local |
| Vite dev server | do Vite cấp | Chỉ khi phát triển | Frontend development và proxy API |

`docker-compose.yml` đang publish `4000`, `6333` và `6334` lên tất cả interface của Docker host theo cú pháp `HOST:CONTAINER`. Vì vậy, “nội bộ” là chủ đích triển khai, **không phải bảo đảm do Compose tự tạo ra**. Máy 2 cần firewall, security group hoặc binding loopback nếu không muốn các cổng này bị truy cập từ mạng ngoài.

Ví dụ binding chỉ vào loopback:

```yaml
ports:
  - "127.0.0.1:4000:4000"
```

## 6. Vòng đời khởi động

FastAPI dùng `lifespan` để tạo bốn singleton theo thứ tự:

1. `VectorStore`
   - Kết nối Qdrant.
   - Kiểm tra collection.
   - Tạo collection và payload index nếu chưa tồn tại.
2. `Embedder`
   - Nạp `BAAI/bge-m3`.
   - Mặc định dùng CUDA FP16 và batch 8.
   - Khóa các lệnh `encode()` dùng chung model để tránh nhiều batch CUDA chồng nhau.
3. `DocumentParser`
   - Khởi tạo `DocumentConverter` của Docling.
   - Mặc định dùng CUDA để OCR tài liệu scan nhanh hơn, batch nội bộ bằng 1.
4. `RAGPipeline`
   - Giữ tham chiếu đến embedder và vector store.
   - Khởi tạo `AsyncOpenAI` trỏ đến LiteLLM.

Hệ quả vận hành:

- API không sẵn sàng nếu Qdrant không kết nối được.
- Lần khởi động đầu có thể chậm do tải BGE-M3 và model của Docling.
- Việc nạp Coding Agent xảy ra khi import `src.api.main`; quá trình này có thể thực hiện khám phá MCP trước khi ứng dụng nhận request.
- React chỉ được mount nếu `frontend/dist` đã tồn tại tại thời điểm module FastAPI được import.

## 7. Kiến trúc API Gateway

Tệp chính: `src/api/main.py`.

Trách nhiệm:

- Nạp `.env` bằng `python-dotenv`.
- Khởi tạo tài nguyên dùng chung.
- Validate UUID công khai.
- Kiểm tra tên và phần mở rộng file.
- Giới hạn dung lượng upload.
- Giới hạn số trang PDF và thời gian xử lý tài liệu.
- Admission control cho parse/index: một tác vụ chạy, tối đa bốn tác vụ chờ.
- Rate limit query/upload theo IP trong bộ nhớ.
- Điều phối upload, parse, embedding và index.
- Cung cấp query đồng bộ và streaming.
- Chuyển tác vụ đến LangGraph Agent.
- Phục vụ build React tại `/`.

### 7.1. Danh sách endpoint

| Phương thức | Endpoint | Chức năng | Xác thực |
|---|---|---|---|
| `GET` | `/health` | Trả trạng thái tiến trình và cấu hình Qdrant | Không |
| `GET` | `/models` | Danh sách model cho UI | Không |
| `GET` | `/knowledge/mkac/status` | Số tài liệu/chunk trong kho MKAC | Không |
| `POST` | `/sessions` | Sinh UUID phiên mới | Không |
| `GET` | `/sessions/{session_id}` | Tổng hợp file và chunk trong Qdrant | Không |
| `DELETE` | `/sessions/{session_id}` | Xóa vector và thư mục upload của phiên | Không |
| `POST` | `/sessions/{session_id}/upload` | Upload, parse, embed và index | Không |
| `DELETE` | `/sessions/{session_id}/files/{filename}` | Xóa một file khỏi Qdrant và ổ đĩa | Không |
| `POST` | `/query` | RAG đồng bộ | Không |
| `POST` | `/query/stream` | RAG streaming bằng SSE | Không |
| `POST` | `/agent` | Thực thi Coding Agent | Header API key nếu đã cấu hình |

### 7.2. Lưu ý về health check

`GET /health` trả trạng thái dịch vụ, kho MKAC và tải xử lý tài liệu:

```json
{
  "status": "healthy",
  "qdrant_host": "localhost",
  "qdrant_port": 6333,
  "document_processing": {
    "active": 0,
    "waiting": 0,
    "concurrency": 1,
    "queue_size": 4,
    "embedding_device": "cuda",
    "embedding_dtype": "float16",
    "ocr_device": "cuda"
  }
}
```

Endpoint này không thực hiện truy vấn sống đến Qdrant, LiteLLM, model upstream hoặc MCP. Vì các singleton được tạo khi startup, response `200` cho thấy tiến trình FastAPI đã khởi động thành công, nhưng không bảo đảm tất cả dependency vẫn khỏe tại thời điểm kiểm tra.

## 8. Vòng đời session

Session là UUID do API sinh ra và dùng làm khóa phân vùng logic.

```text
POST /sessions
  └─ sinh UUID
     └─ VectorStore.create_session() chỉ ghi log, không tạo bản ghi

Upload file đầu tiên
  └─ Qdrant bắt đầu có point mang session_id

GET /sessions/{id}
  ├─ có point      -> trả thông tin session
  └─ chưa có point -> 404 "Session not found or empty"

DELETE /sessions/{id}
  ├─ xóa point theo payload filter
  └─ xóa uploads/{session_id}
```

Frontend lưu UUID tại `localStorage` với khóa `vllm-pd-session`. Nếu session mới chưa có tài liệu, `GET /sessions/{id}` trả `404`; frontend coi đây là phiên rỗng hợp lệ và tiếp tục giữ UUID.

Không có bảng session riêng, thời hạn hết hạn hoặc tác vụ tự động dọn session. Dữ liệu chỉ bị xóa khi client gọi endpoint xóa hoặc quản trị viên dọn trực tiếp.

## 9. Luồng nạp và lập chỉ mục tài liệu

### 9.1. Định dạng được API cho phép

```text
.pdf, .docx, .xlsx, .pptx, .html, .htm, .png, .jpg, .jpeg
```

Frontend hiện không khai báo `.htm` trong thuộc tính `accept`, nhưng backend vẫn chấp nhận định dạng này.

### 9.2. Trình tự xử lý

```text
UploadFile
  │
  ├─ rate limit theo IP
  ├─ validate session UUID
  ├─ loại bỏ nguy cơ path traversal trong filename
  ├─ kiểm tra extension
  ├─ ghi file theo từng khối 1 MiB
  └─ kiểm tra MAX_UPLOAD_SIZE_MB
       │
       ▼
Docling DocumentConverter.convert()
       │
       ▼
export_to_markdown()
       │
       ▼
chia chunk khoảng 1.000 ký tự, chồng lấn khoảng 20% số dòng
       │
       ▼
BGE-M3 embed_documents(), batch mặc định 32
       │
       ├─ xóa các vector cũ của cùng filename trong session
       └─ upsert vector và payload mới vào Qdrant
```

Nếu parse, embedding hoặc index lỗi, API cố gắng xóa file vật lý vừa tải lên và trả lỗi.

### 9.3. Đặc điểm chunking hiện tại

- `CHUNK_SIZE = 1000` ký tự.
- `CHUNK_OVERLAP = 200` được khai báo nhưng chưa được dùng trực tiếp.
- Phần chồng lấn thực tế là khoảng 20% số dòng cuối của chunk trước.
- Chunk được đánh dấu `table` khi phát hiện mẫu Markdown table đơn giản.
- Metadata mặc định là `{"source": "docling"}`.

### 9.4. Giới hạn cần biết

- `page_number` hiện luôn bắt đầu và giữ ở `1`; parser chưa ánh xạ provenance trang thật từ cấu trúc Docling.
- Logic nhận diện bảng dùng heuristic chuỗi, không phải kiểu phần tử từ Docling.
- Chunking dựa trên số ký tự và dòng, chưa dựa trên token hoặc cấu trúc heading.
- Mã nguồn RAG có khả năng gửi tối đa hai ảnh từ `metadata.image_path`, nhưng parser hiện chưa tạo trường này. Vì vậy luồng vision chưa hoàn chỉnh theo đường chạy mặc định.
- Khi upload lại cùng tên file, vector cũ bị xóa trước khi vector mới được upsert; file vật lý mới đã ghi đè file cũ.

## 10. Embedding và lưu trữ vector

### 10.1. Embedding

| Thuộc tính | Giá trị |
|---|---|
| Model | `BAAI/bge-m3` |
| Kích thước vector | `1024` |
| Thiết bị | CUDA nếu có, ngược lại CPU |
| Chuẩn hóa | L2 normalization |
| Batch mặc định | `32` |
| Query prefix | Chuỗi rỗng |

Vector được chuẩn hóa L2 và Qdrant dùng cosine similarity.

### 10.2. Qdrant

| Thuộc tính | Giá trị |
|---|---|
| Collection | `docmind_documents` |
| Vector size | `1024` |
| Distance | `COSINE` |
| Payload index | `session_id`, kiểu keyword |

Payload của mỗi point:

```json
{
  "session_id": "b2fbd2b3-4c72-4ea7-8e06-57ac6b43163b",
  "text": "Nội dung chunk...",
  "source_file": "tai-lieu.pdf",
  "page_number": 1,
  "chunk_index": 0,
  "content_type": "text",
  "metadata": {
    "source": "docling"
  }
}
```

Mỗi lần index tạo UUID point mới. Cô lập dữ liệu dựa trên payload filter:

```text
session_id == UUID của request
```

Đây là cô lập logic ở tầng ứng dụng, chưa phải cơ chế multi-tenant có xác thực chủ sở hữu. Bất kỳ client nào biết UUID đều có thể gọi API đọc, query hoặc xóa session vì các endpoint RAG hiện chưa yêu cầu đăng nhập.

## 11. Luồng truy vấn RAG

```text
Câu hỏi + session_id + model + mode
  │
  ├─ rate limit theo IP
  ├─ validate UUID
  └─ BGE-M3 embed_query()
       │
       ▼
Qdrant cosine search
  ├─ filter session_id
  ├─ score_threshold = 0,25
  └─ top_k = 5 hoặc 10
       │
       ▼
build_rag_prompt()
  ├─ system prompt theo mode
  ├─ context từ các chunk
  └─ ảnh từ metadata.image_path nếu có
       │
       ▼
LiteLLM /v1/chat/completions
       │
       ├─ /query        -> JSON hoàn chỉnh
       └─ /query/stream -> SSE
```

### 11.1. Chế độ `mkac`

- `top_k = 5`.
- `max_tokens = 1024`.
- Tìm trong collection `mkac_knowledge` với `session_id=mkac`.
- Lọc phần đuôi kết quả có điểm thấp hơn nhiều so với kết quả tốt nhất.
- Trả lời dựa trên tài liệu nội bộ và trích dẫn đúng trang.
- Câu hỏi về bảng/hình/sơ đồ có thể đính kèm ảnh trang và route sang OpenAI Vision.

### 11.2. Chế độ `research`

- `top_k = 10`.
- `max_tokens = 1800`.
- Yêu cầu cấu trúc:
  - Tóm tắt điều hành.
  - Phát hiện chính.
  - Bằng chứng.
  - Điểm chưa rõ.
  - Câu hỏi nghiên cứu tiếp theo.

### 11.3. Khi không tìm thấy kết quả MKAC

Pipeline tìm kiếm web với câu hỏi được gắn thêm ngữ cảnh MKAC. Khi có kết quả,
model chỉ tổng hợp từ snippet, dẫn URL và response/meta có `answer_scope=web`.
Thông tin web luôn được phân biệt với chính sách nội bộ MKAC bằng metadata và
nguồn liên kết; câu trả lời không thêm câu mở đầu cảnh báo lặp lại.

Nếu tìm web không có kết quả hoặc gặp lỗi, pipeline mới gọi model không kèm
context và dùng `answer_scope=general`; model chỉ thông báo ngắn gọn rằng chưa
tìm thấy thông tin, không trả lời bằng kiến thức chung.

### 11.4. Truy vấn dữ liệu lỗi theo Lot từ MES

Ở chế độ `mes`, các câu hỏi có đủ ý định `Lot + lỗi/NG + nhiều/cao nhất` được
định tuyến tới MES trước bước embedding và retrieval. Backend gọi API MES bằng
Bearer token, parse ba trường `Lot_Id`, `Product_Id`, `Total_Error_Qty`, tự chọn
giá trị lớn nhất và giữ đầy đủ các Lot đồng hạng.

Kết quả đã chuẩn hóa được đưa vào model người dùng đang chọn để diễn đạt thành
một câu tiếng Việt tự nhiên. Response dùng `answer_scope=mes`, không trả nguồn
Qdrant và frontend hiển thị nhãn `Dữ liệu MES`. Nếu LLM lỗi trước khi stream bắt
đầu, backend dùng câu trả lời deterministic chứa đủ mã Lot, mã hàng và số lỗi.

Token MES chỉ tồn tại ở backend qua biến môi trường; frontend không gọi trực
tiếp API MES và không nhận token.

### 11.5. Truy vấn MES snapshot cục bộ

Database `data/mes.sqlite` hợp nhất thông tin Lot, bản ghi lỗi và danh mục tên
lỗi từ ba dump MES. `MesDatabase` mở SQLite ở chế độ read-only và chỉ thực thi
các truy vấn tham số hóa trong allowlist; câu hỏi người dùng không được chuyển
thẳng thành SQL.

Các intent hiện hỗ trợ gồm thông tin Lot, chi tiết lỗi theo Lot, tên mã lỗi,
thống kê và chi tiết lỗi theo sản phẩm, sản phẩm có tổng lỗi cao nhất, các Lot
có một mã lỗi và Lot có tổng lỗi cao nhất trong snapshot.

API MES vẫn được ưu tiên cho câu hỏi Lot lỗi nhiều nhất theo dữ liệu thời gian
thực. Snapshot được dùng khi người dùng nói rõ `snapshot/database`, khi API MES
lỗi, hoặc cho các intent chi tiết mà API hiện tại chưa cung cấp. Response dùng
`answer_scope=mes_database`; frontend hiển thị nhãn `MES snapshot`.

Snapshot không tự loại dữ liệu test và có thể khác API MES thời gian thực. Nếu
LLM không khả dụng hoặc bỏ sót trường bắt buộc, backend trả câu deterministic từ
kết quả SQL.

### 11.6. SQL Agent cho câu hỏi MES phức hợp

Với các câu hỏi MES phức hợp chưa có intent cố định, ví dụ “trong Lot có số lỗi
nhiều nhất thì 3 loại lỗi gây lỗi nhiều nhất là gì”, backend dùng
`MesSqlAgent`. Agent này không cho model truy cập bảng raw trực tiếp. Thay vào
đó, model chỉ nhận semantic model ở `config/mes_semantic_model.json` gồm các
view công khai:

- `v_lot_error_summary`
- `v_lot_error_breakdown`
- `v_product_error_summary`
- `v_error_details`

LLM chỉ sinh kế hoạch JSON chứa một câu `SELECT` hoặc `WITH ... SELECT`.
Backend validate SQL bằng SQLGlot, chỉ cho truy cập các view allowlist, chặn
DDL/DML/`ATTACH`/`PRAGMA`, ép `LIMIT`, mở SQLite `mode=ro`, bật
`PRAGMA query_only=ON`, dùng authorizer read-only và timeout ngắn. Sau khi chạy
SQL, kết quả JSON đã kiểm chứng mới được đưa lại cho LLM để diễn đạt tiếng Việt
tự nhiên. Nếu câu trả lời thiếu mã/tên lỗi/số liệu bắt buộc, backend dùng câu
fallback deterministic từ chính kết quả SQL.

## 12. Giao thức SSE

`POST /query/stream` trả `Content-Type: text/event-stream`. Mỗi event nằm trong trường `data` dưới dạng JSON.

Thứ tự bình thường:

```text
sources -> meta -> token... -> done
```

Ví dụ:

```text
data: {"type":"sources","sources":[...]}

data: {"type":"meta","model":"auto-model","mode":"mkac","answer_scope":"mkac"}

data: {"type":"token","content":"Nội"}

data: {"type":"token","content":" dung"}

data: {"type":"done"}
```

Khi lỗi xảy ra sau khi stream đã bắt đầu:

```text
data: {"type":"error","message":"..."}
```

Response đặt:

```text
Cache-Control: no-cache
X-Accel-Buffering: no
```

Frontend hiện xử lý `sources`, `token` và `error`; event `meta` và `done` chưa được dùng để cập nhật trạng thái giao diện.

## 13. LiteLLM và định tuyến model

Ứng dụng gọi LiteLLM qua URL mặc định:

```text
http://localhost:4000/v1
```

### 13.1. Ánh xạ từ UI/API

| Giá trị API | Model logic LiteLLM | Backend chính |
|---|---|---|
| `auto` | `auto-model` | `openai/gpt-5.4-mini` |
| `local` | `local-gemma` | Ollama `gemma4:latest` |
| `openai` | `openai-model` | `openai/gpt-5.4-mini` |
| `grok` | `grok-model` | Azure `grok-4-20-reasoning` |
| Coding Agent | `coding-model` | Ollama `gemma4:latest` |

### 13.2. Chuỗi fallback

```yaml
auto-model:
  - grok-model
  - local-gemma

coding-model:
  - openai-model
```

Ý nghĩa:

- `auto`: OpenAI → Grok → Gemma4 local.
- Agent: Gemma4 local → OpenAI.
- `local`, `openai` và `grok` không có fallback riêng trong cấu hình hiện tại.
- Chế độ `Nghiên cứu` luôn route sang `grok-model`; ảnh `.png`, `.jpg` hoặc `.jpeg` được đính kèm base64 vào prompt để dùng Vision.
- Router dùng `simple-shuffle`, retry một lần và timeout tổng quát 120 giây.

### 13.3. Điểm cần lưu ý

- `OLLAMA_MODEL` có trong `.env.example` nhưng `litellm_config.yaml` đang ghi trực tiếp `gemma4:latest`; đổi riêng biến này chưa làm thay đổi model.
- `LITELLM_MASTER_KEY` bảo vệ LiteLLM, nhưng giá trị mặc định `sk-local` chỉ phù hợp môi trường tin cậy.
- Provider key được truyền vào container LiteLLM qua `.env`; không được đưa các key này vào frontend.

## 14. Coding Agent và MCP

### 14.1. Đồ thị LangGraph

```text
START
  │
  ▼
agent: gọi coding-model
  │
  ├─ có tool_calls ──► tools: ToolNode
  │                         │
  │                         └────────► agent
  │
  └─ không có tool_calls ──► END
```

State hiện chỉ chứa:

```python
messages: Sequence[BaseMessage]
```

Agent chưa có plan, checkpoint, giới hạn số vòng lặp, bộ nhớ bền vững hoặc hàng đợi tác vụ riêng trong mã nguồn hiện tại.

### 14.2. Kết nối mô hình

`ChatOpenAI` được dùng như client OpenAI-compatible:

```text
model               = coding-model
openai_api_base     = LITELLM_URL
openai_api_key      = LITELLM_MASTER_KEY
temperature         = 0,2
```

### 14.3. MCP servers

| Tên | Runtime | Phạm vi |
|---|---|---|
| `filesystem` | `npx -y @modelcontextprotocol/server-filesystem` | `WORKSPACE_DIR` |
| `git` | `uvx mcp-server-git --repository ...` | `AGENT_REPOSITORY_DIR` |

Tool được khám phá qua `MultiServerMCPClient` và cache bằng `lru_cache(maxsize=1)`.

Nếu MCP adapter không được cài hoặc không server nào nạp được tool, hệ thống dùng ba tool cục bộ:

- `read_file`
- `write_file`
- `list_dir`

Các đường dẫn fallback được resolve và bắt buộc nằm trong `WORKSPACE_DIR`.

### 14.4. Hợp đồng `/agent`

Request:

```json
{
  "session_id": "chuỗi do client cung cấp",
  "task": "Mô tả tác vụ lập trình"
}
```

Response:

```json
{
  "session_id": "chuỗi từ request",
  "status": "completed",
  "output": "Kết quả cuối",
  "steps": [
    {
      "role": "ai",
      "content": "...",
      "tool_calls": []
    }
  ]
}
```

`session_id` của Agent hiện không được validate, không liên kết với session RAG và không được dùng để lưu trạng thái Agent. Nó chỉ được phản hồi lại cho client.

## 15. Frontend

Frontend là React 18 SPA build bằng Vite.

Chức năng chính:

- Kiểm tra `/health` và tải `/models` khi khởi động.
- Khôi phục UUID session riêng cho từng chế độ từ `localStorage`.
- Tạo phiên mới.
- Trong chế độ `Hỏi đáp hành chính nhân sự MKAC`, sử dụng kho tài liệu nội bộ,
  danh bạ nhân sự dùng chung và yêu cầu xác thực mã nhân viên.
- Trong chế độ `Quản lý MES`, định tuyến riêng tới API MES hoặc MES snapshot;
  không fallback sang RAG hành chính, nhân sự hoặc web.
- Trong chế độ `Nghiên cứu tài liệu`, chọn, upload và xóa nhiều file theo phiên.
- Chọn model.
- Chuyển giữa ba chế độ hành chính nhân sự, MES và nghiên cứu tài liệu.
- Mỗi chế độ giữ UUID, lịch sử hội thoại và nguồn tham chiếu riêng; session cũ
  dùng một khóa được migrate sang chế độ `Nghiên cứu tài liệu`.
- Gửi câu hỏi qua SSE.
- Render Markdown.
- Hiển thị nguồn ở panel desktop và trong từng tin nhắn.
- Cho phép dừng câu trả lời streaming, sao chép phản hồi và xem nhãn AI với
  model/phạm vi nguồn.
- Panel nguồn dùng progressive disclosure: mặc định thu gọn và hiển thị số
  nguồn của câu trả lời mới nhất trên nút mở panel.
- Hỗ trợ giao diện `Sáng`, `Tối` và `Theo hệ thống`; lựa chọn được lưu bằng
  khóa `vllm-pd-theme` trong `localStorage`. Chế độ hệ thống theo dõi
  `prefers-color-scheme` và cập nhật ngay khi thiết lập hệ điều hành thay đổi.
- Script nhỏ trong `frontend/index.html` áp dụng theme trước khi React mount để
  tránh nháy nền sáng khi mở dark mode.

Production:

```text
frontend/dist -> FastAPI StaticFiles -> /
```

Development:

```text
Vite dev server
  └─ proxy /health, /models, /sessions, /query -> http://localhost:8001
```

Frontend dùng URL tương đối nên web và API có thể chạy cùng origin. API bật CORS `*`, nhưng trong production cùng origin thì CORS rộng thường không cần thiết.

## 16. Bảo mật và ranh giới tin cậy

### 16.1. Biện pháp đã có

| Rủi ro | Biện pháp hiện tại |
|---|---|
| Path traversal qua tên file | So sánh filename với `Path(filename).name` |
| File không hỗ trợ | Allowlist phần mở rộng |
| File quá lớn | Đọc theo block và giới hạn `MAX_UPLOAD_SIZE_MB` |
| PDF quá nhiều trang | Từ chối trước OCR bằng `MAX_DOCUMENT_PAGES` |
| Upload đồng thời làm tăng VRAM/RAM | Semaphore và hàng đợi có giới hạn |
| OCR chiếm VRAM | Docling batch 1 và toàn bộ upload được giới hạn concurrency 1 |
| Session ID không hợp lệ | Parse bằng `uuid.UUID` |
| Lạm dụng query/upload | Rate limit trong bộ nhớ theo IP |
| Truy cập Agent | So sánh API key bằng `secrets.compare_digest` |
| Agent đọc/ghi ngoài workspace | Giới hạn `WORKSPACE_DIR` |
| Git tool truy cập sai repository | Cố định `AGENT_REPOSITORY_DIR` |
| Lộ provider key cho browser | Browser chỉ gọi API Gateway, không gọi LiteLLM trực tiếp |

### 16.2. Rủi ro và hạn chế hiện tại

1. **Endpoint RAG không có xác thực.** UUID là định danh, không phải secret hoặc quyền sở hữu.
2. **`AGENT_API_KEY` rỗng sẽ vô hiệu hóa xác thực Agent.** Điều kiện kiểm tra chỉ chạy khi biến này có giá trị.
3. **CORS cho phép mọi origin, method và header.**
4. **Rate limit nằm trong RAM.** Dữ liệu mất khi restart và không dùng chung giữa nhiều worker/instance.
5. **Tin tưởng `X-Forwarded-For`.** Client có thể giả header nếu API không đứng sau reverse proxy đáng tin cậy.
6. **Docker publish cổng ra host.** LiteLLM và Qdrant cần firewall hoặc bind loopback.
7. **Thông báo lỗi có thể lộ chi tiết nội bộ.** Một số exception được trả nguyên văn trong HTTP/SSE response.
8. **Upload kiểm tra extension, chưa kiểm tra MIME hoặc magic bytes.**
9. **Không có quét mã độc, quota ổ đĩa hoặc chính sách lưu giữ file.**
10. **Agent có quyền ghi file.** Đây là năng lực có rủi ro cao, cần workspace riêng, backup và nguyên tắc đặc quyền tối thiểu.

Khuyến nghị tối thiểu khi public:

- Bắt buộc `AGENT_API_KEY` mạnh.
- Thêm xác thực cho toàn bộ API và kiểm tra chủ sở hữu session.
- Bind LiteLLM/Qdrant vào `127.0.0.1` hoặc chặn bằng firewall.
- Thu hẹp CORS về domain frontend.
- Chỉ tin `X-Forwarded-For` từ reverse proxy được kiểm soát.
- Thay rate limiter in-memory bằng Redis hoặc gateway rate limit nếu chạy nhiều instance.
- Không chạy Agent trên repository hoặc tài khoản hệ thống có quyền vượt quá nhu cầu.

## 17. Cấu hình

### 17.1. Nhóm biến môi trường

| Nhóm | Biến chính |
|---|---|
| Ollama | `OLLAMA_API_BASE`, `OLLAMA_MODEL` |
| Provider cloud | `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |
| LiteLLM | `LITELLM_URL`, `LITELLM_MASTER_KEY` |
| API public | `MACHINE2_API_HOST`, `MACHINE2_API_PORT`, `NGROK_RESERVED_DOMAIN` |
| Qdrant | `QDRANT_HOST`, `QDRANT_PORT` |
| Upload | `UPLOAD_DIR`, `MAX_UPLOAD_SIZE_MB`, `MAX_DOCUMENT_PAGES`, `DOCUMENT_PROCESSING_TIMEOUT_SECONDS`, `UPLOAD_PROCESSING_CONCURRENCY`, `UPLOAD_QUEUE_SIZE` |
| Embedding | `EMBEDDING_DEVICE`, `EMBEDDING_DTYPE`, `EMBEDDING_BATCH_SIZE` |
| OCR | `DOCLING_DEVICE`, `DOCLING_NUM_THREADS`, `DOCLING_OCR_LANGUAGES` |
| Rate limit | `QUERY_RATE_LIMIT_PER_MINUTE`, `UPLOAD_RATE_LIMIT_PER_HOUR` |
| Agent | `AGENT_API_KEY`, `WORKSPACE_DIR`, `AGENT_REPOSITORY_DIR` |
| MES API | `MES_API_URL`, `MES_API_TOKEN`, `MES_API_TIMEOUT`, `MES_VERIFY_TLS`, `MES_CA_CERT` |
| MES snapshot | `MES_DATABASE_ENABLED`, `MES_DATABASE_PATH` |
| MES SQL Agent | `MES_SQL_AGENT_ENABLED`, `MES_SEMANTIC_MODEL_PATH`, `MES_SQL_AGENT_MAX_ROWS`, `MES_SQL_AGENT_TIMEOUT`, `MES_SQL_AGENT_MAX_ATTEMPTS` |
| Log | `LOG_LEVEL` |

Không commit `.env`. File này chứa provider key, LiteLLM master key và Agent API key.

### 17.2. Giá trị mặc định đáng chú ý

| Cấu hình | Mặc định trong mã |
|---|---|
| `QDRANT_HOST` | `localhost` |
| `QDRANT_PORT` | `6333` |
| `UPLOAD_DIR` | `./uploads` |
| `LITELLM_URL` | `http://localhost:4000/v1` |
| `LITELLM_MASTER_KEY` | `sk-local` |
| `MAX_UPLOAD_SIZE_MB` | `25` |
| `MAX_DOCUMENT_PAGES` | `100` |
| Upload processing concurrency | `1` |
| Upload queue size | `4` |
| Embedding | CUDA, FP16, batch `8` |
| Docling/EasyOCR | CUDA, batch 1 |
| Query rate limit | `15` request/IP/phút |
| Upload rate limit | `10` request/IP/giờ |

## 18. Vận hành và kiểm tra

### 18.1. Build frontend

```bash
cd frontend
npm install
npm run build
```

### 18.2. Khởi động hạ tầng

```bash
docker compose up -d
docker compose ps
```

### 18.3. Chạy API foreground

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

### 18.4. User service trên Máy 2

Nếu máy đã cài service `vllm-pd-api`:

```bash
systemctl --user status vllm-pd-api
systemctl --user restart vllm-pd-api
journalctl --user -u vllm-pd-api -n 160 --no-pager
```

Đường dẫn service từng được sử dụng:

```text
~/.config/systemd/user/vllm-pd-api.service
```

Repository không chứa unit file này, vì vậy cần xác minh nội dung thực tế trên Máy 2.

### 18.5. Kiểm tra dependency

```bash
# FastAPI
curl -fsS http://localhost:8001/health

# LiteLLM
curl -fsS http://localhost:4000/health/liveliness

# Qdrant
curl -fsS http://localhost:6333/healthz

# Model list qua LiteLLM
curl -fsS \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  http://localhost:4000/v1/models
```

### 18.6. Smoke test Hỏi đáp MKAC

```bash
SESSION_ID=$(
  curl -fsS -X POST http://localhost:8001/sessions |
  jq -r .session_id
)

curl -N -fsS http://localhost:8001/query/stream \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"question\":\"Quy định nghỉ phép hàng năm của MKAC như thế nào?\",
    \"model\":\"auto\",
    \"mode\":\"mkac\"
  }"
```

Để smoke test upload tài liệu ngoài, upload file vào session như trước rồi gửi
query với `"mode":"research"`.

### 18.7. Smoke test Agent

```bash
curl -fsS http://localhost:8001/agent \
  -H "X-Agent-API-Key: $AGENT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id":"smoke-test",
    "task":"Không gọi công cụ. Chỉ trả lời: OK"
  }' |
  jq .
```

## 19. Failure modes và cách chẩn đoán

### 19.1. Qdrant không sẵn sàng khi API khởi động

Biểu hiện:

- FastAPI không hoàn tất startup.
- Log lỗi kết nối hoặc tạo collection.

Kiểm tra:

```bash
docker compose ps qdrant
docker compose logs --tail=120 qdrant
curl -fsS http://localhost:6333/healthz
```

### 19.2. BGE-M3 hoặc Docling nạp lỗi

Nguyên nhân thường gặp:

- Thiếu package.
- Không tải được model lần đầu.
- CUDA/PyTorch không tương thích.
- Không đủ RAM/VRAM hoặc dung lượng đĩa.

Kiểm tra log FastAPI và thử khởi động foreground để thấy stack trace đầy đủ.

### 19.3. Ollama hoặc ngrok của Máy 1 lỗi

Ảnh hưởng:

- `local-gemma` lỗi trực tiếp.
- `auto-model` có thể chuyển sang Grok hoặc Gemma4 local.
- `coding-model` có thể chuyển sang OpenAI.

Kiểm tra:

```bash
curl "$OLLAMA_API_BASE/api/tags"
docker compose logs --tail=120 litellm
```

### 19.4. LiteLLM lỗi cấu hình hoặc provider

Ảnh hưởng:

- RAG vẫn retrieval được nhưng không sinh câu trả lời.
- Agent không gọi được model.

Kiểm tra:

```bash
docker compose logs --tail=160 litellm
docker compose up -d --force-recreate litellm
```

### 19.5. React trắng trang hoặc thiếu asset

Kiểm tra `frontend/dist`, build lại và restart API:

```bash
cd frontend
npm run build
systemctl --user restart vllm-pd-api
```

### 19.6. Session vừa tạo trả 404

Đây là hành vi hiện tại nếu session chưa có vector. Upload ít nhất một tài liệu rồi gọi lại `GET /sessions/{id}`.

### 19.7. OCR tài liệu MKAC

PDF được xử lý theo từng trang. Trang có text native dùng PyMuPDF; trang scan
được render thành PNG rồi OCR bằng Docling. `TextChunk.page_number` giữ số
trang thật và ảnh trang nằm trong `mkac_processed/pages`. OCR mặc định chạy CUDA
để cải thiện thời gian upload ở chế độ Nghiên cứu. Batch Docling và concurrency
upload đều bằng 1 để tránh nhiều tác vụ chiếm VRAM cùng lúc. Script index MKAC
vẫn mặc định embed trên CPU; có thể chuyển sang CUDA khi API đã dừng.

## 20. Khả năng mở rộng và điểm cải tiến

Các hướng ưu tiên theo tác động:

1. **Xác thực và phân quyền session**
   - Gắn session với user/tenant.
   - Kiểm tra quyền trên upload, query và delete.
2. **Tách worker xử lý tài liệu**
   - Thay hàng đợi trong RAM bằng Redis/Celery hoặc worker tương đương.
   - Dành worker embedding GPU riêng, ưu tiên query và hỗ trợ dynamic batching.
   - Dành worker OCR GPU concurrency 1 riêng và theo dõi tiến độ job.
3. **Parser có provenance chính xác**
   - Dùng cấu trúc document của Docling thay vì chỉ `export_to_markdown()`.
   - Lưu page, bounding box, heading và loại phần tử.
4. **Chunking tốt hơn**
   - Chunk theo token và cấu trúc tài liệu.
   - Giữ bảng, heading và đoạn văn theo ngữ nghĩa.
5. **Vision pipeline hoàn chỉnh**
   - Trích xuất/lưu ảnh.
   - Tạo `metadata.image_path`.
   - Chỉ route request có ảnh đến model hỗ trợ vision.
6. **Session registry**
   - Lưu session độc lập với vector.
   - Hỗ trợ phiên rỗng, TTL và cleanup job.
7. **Health/readiness chuyên sâu**
   - Ping Qdrant, LiteLLM và model upstream.
   - Tách liveness và readiness.
8. **Rate limit phân tán**
   - Redis, API gateway hoặc reverse proxy.
9. **Quan sát hệ thống**
   - Structured logging, request ID, latency từng stage, token usage và tracing.
10. **Agent bền vững và an toàn hơn**
   - Checkpoint, timeout, recursion limit, approval gate cho thao tác ghi/chạy lệnh.
11. **Triển khai nhiều instance**
    - Đưa upload sang object storage.
    - Dùng session/auth store dùng chung.
    - Loại bỏ state chỉ tồn tại trong RAM.

## 21. Những thành phần không thuộc đường chạy chính

- `docmind/` không phải backend/frontend chính.
- Streamlit không phải frontend production.
- FAISS không phải vector store của hệ thống chính.
- Port `8000` không phải port API mặc định.
- Máy 3 không cần truy cập trực tiếp LiteLLM.
- Repository hiện không chạy vLLM server làm backend suy luận chính; tên dự án `VLLM-PD` không đồng nghĩa runtime hiện tại đang dùng vLLM.

## 22. Tóm tắt

VLLM-PD là một hệ thống RAG và Coding Agent triển khai tập trung trên Máy 2:

- FastAPI là cổng vào duy nhất cho web và API.
- React cung cấp giao diện hỏi đáp/nghiên cứu.
- Docling, BGE-M3 và Qdrant tạo pipeline nạp và retrieval tài liệu.
- LiteLLM tách ứng dụng khỏi model vật lý và cung cấp fallback.
- LangGraph kết hợp LLM với MCP để tạo Coding Agent.

Kiến trúc phù hợp với triển khai một máy chủ ứng dụng và lượng người dùng nhỏ. Trước khi mở rộng hoặc public rộng rãi, các hạng mục quan trọng nhất là xác thực session, cô lập cổng hạ tầng, provenance trang chính xác, health check dependency và kiểm soát quyền của Agent.
