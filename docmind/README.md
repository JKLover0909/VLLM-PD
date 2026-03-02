# DocMind 📚

**RAG-powered Document Q&A Webapp** — Phân tích tài liệu có ảnh, hỏi đáp thông minh, hỗ trợ tiếng Việt & tiếng Anh.

## Stack

| Thành phần | Công nghệ |
|---|---|
| LLM | Llama 3.1 8B Instruct FP16 via **vLLM** |
| Embedding | **BAAI/bge-m3** (multilingual, 8192 ctx) |
| Vector DB | **FAISS** IndexFlatIP (cosine similarity) |
| OCR | **EasyOCR** (vi + en) |
| PDF Parse | **PyMuPDF** |
| Backend | **FastAPI** + async streaming |
| Frontend | **Streamlit** |

---

## Yêu cầu

- Python 3.10+
- GPU ≥ 16GB VRAM (Llama 3.1 8B FP16)
- RAM ≥ 16GB
- Storage ≥ 20GB (model weights)
- CUDA 11.8+

---

## Cài đặt

```bash
cd docmind
pip install -r requirements.txt
```

---

## Chạy

### Bước 1: Khởi động vLLM server
```bash
bash scripts/start_vllm.sh
# Chạy trên port 8000
# Có thể tùy chỉnh:
# MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct" PORT=8000 bash scripts/start_vllm.sh
```

### Bước 2: Khởi động FastAPI backend
```bash
cd backend
cp .env.example .env   # Chỉnh sửa nếu cần
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
# API docs: http://localhost:8001/docs
```

### Bước 3: Khởi động Streamlit frontend
```bash
cd frontend
streamlit run app.py --server.port 8501
# Mở: http://localhost:8501
```

---

## Sử dụng

1. Mở `http://localhost:8501` trên trình duyệt
2. Click **"Tạo Session Mới"** ở sidebar
3. Upload tài liệu (PDF, ảnh) → Click **"Index Tài liệu"**
4. Nhập câu hỏi và nhận câu trả lời kèm nguồn trích dẫn

---

## Cấu trúc Project

```
docmind/
├── backend/
│   ├── main.py                # FastAPI app
│   ├── document_processor.py  # PDF + OCR
│   ├── embedder.py            # bge-m3 wrapper
│   ├── vector_store.py        # FAISS sessions
│   ├── rag_pipeline.py        # RAG orchestration
│   ├── vllm_client.py         # vLLM API client
│   └── .env.example
├── frontend/
│   └── app.py                 # Streamlit UI
├── scripts/
│   └── start_vllm.sh          # vLLM launcher
├── uploads/                   # Session uploads (auto-created)
└── requirements.txt
```

---

## API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/health` | Kiểm tra trạng thái |
| POST | `/sessions` | Tạo session mới |
| GET | `/sessions/{id}` | Thông tin session |
| DELETE | `/sessions/{id}` | Xóa session |
| POST | `/sessions/{id}/upload` | Upload & index file |
| POST | `/query` | Hỏi đáp (non-streaming) |
| POST | `/query/stream` | Hỏi đáp (SSE streaming) |
| DELETE | `/sessions/{id}/files/{name}` | Xóa file |

---

## Lưu ý

- **OCR ảnh nặng**: EasyOCR load lần đầu ~30s. Sau đó cache.
- **bge-m3**: Load ~2GB VRAM. Nếu thiếu VRAM, dùng `device="cpu"`.
- **Session TTL**: 1 giờ không dùng → tự xóa.
- **FAISS remove_file**: Cần re-index sau khi xóa file. Xem `vector_store.py` line 118.


