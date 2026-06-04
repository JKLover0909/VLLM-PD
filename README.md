# 🤖 VLLM-PD: AI Coding Agent & RAG Document System

> **Hệ thống AI Agent tích hợp xử lý tài liệu thông minh**
> 
> Kết hợp AI Coding Agent (LangGraph + MCP) với hệ thống RAG xử lý tài liệu đa định dạng, hỗ trợ song ngữ Anh-Việt.

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Các thành phần chính](#-các-thành-phần-chính)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [API Endpoints](#-api-endpoints)
- [Troubleshooting](#-troubleshooting)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

---

## 🌟 Giới thiệu

**VLLM-PD** là hệ thống AI hybrid kết hợp hai chức năng chính:

1. **AI Coding Agent**: Hỗ trợ lập trình thông qua giao thức MCP (Model Context Protocol), tích hợp với LangGraph và VS Code
2. **DocMind RAG**: Hệ thống xử lý và hỏi đáp tài liệu thông minh, hỗ trợ PDF, ảnh, OCR đa ngôn ngữ

### ✨ Tính năng nổi bật

- 🔧 **AI Coding Agent** với LangGraph orchestration
- 📚 **RAG Pipeline** xử lý tài liệu đa định dạng (PDF, ảnh)
- 🌐 **OCR đa ngôn ngữ** (tiếng Việt + tiếng Anh)
- 🚀 **vLLM** để inference nhanh với Llama/Qwen models
- 🔍 **Vector Database** với FAISS/Qdrant
- 🎯 **Embedding đa ngôn ngữ** với BGE-M3
- 🔄 **LiteLLM Router** hỗ trợ fallback sang Cloud APIs
- 🔐 **Tailscale VPN** kết nối bảo mật giữa các máy

---

## 🏗️ Kiến trúc hệ thống

Hệ thống được thiết kế theo mô hình **microservices phân tán**, chạy trên nhiều máy kết nối qua Tailscale VPN:

```
┌─────────────────────────────────────────────────────────────┐
│              TAILSCALE MESH VPN (Private Network)           │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌─────────────┐ ┌────────────┐
│   Máy 3      │ │   Máy 2     │ │   Máy 1    │
│ Workstation  │ │ RAG Server  │ │ LLM Host   │
│              │ │  (Linux)    │ │ (Windows)  │
│  VS Code     │ │             │ │            │
│  + Cline     │◄┤ LangGraph   │ │  vLLM      │
│              │ │  Agent      │ │  Server    │
│              │ │             │ │            │
│  Browser     │ │ FastAPI     │ │  Ollama    │
│  Chat UI     │◄┤ Backend     │◄┤  (backup)  │
│              │ │             │ │            │
│              │ │ Qdrant      │ │  Qwen3     │
│              │ │ LiteLLM     │ │  Llama3.1  │
└──────────────┘ └─────────────┘ └────────────┘
```

Chi tiết đầy đủ xem tại: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 🧩 Các thành phần chính

### 1. **DocMind** - RAG Document System

Hệ thống xử lý và hỏi đáp tài liệu với các tính năng:

- ✅ Upload và parse PDF, ảnh (PNG, JPG)
- ✅ OCR với EasyOCR (vi + en)
- ✅ Embedding đa ngôn ngữ (BGE-M3)
- ✅ Vector search với FAISS
- ✅ Streaming response từ vLLM
- ✅ Session management
- ✅ Web UI với Streamlit

**Stack:**
- LLM: Llama 3.1 8B / Qwen3 14B (vLLM)
- Embedding: BAAI/bge-m3
- Vector DB: FAISS / Qdrant
- Backend: FastAPI
- Frontend: Streamlit

### 2. **Agent System** - AI Coding Assistant

Agent hỗ trợ lập trình với:

- ✅ LangGraph orchestration
- ✅ MCP (Model Context Protocol)
- ✅ Tool calling & function execution
- ✅ Tích hợp VS Code (qua Cline/Roo-Code)

**Stack:**
- Framework: LangGraph
- Protocol: MCP
- Backend: FastAPI
- Client: Python SDK

### 3. **Infrastructure Services**

- **Qdrant**: Vector database (Docker)
- **LiteLLM**: Router/Proxy với fallback sang Cloud APIs
- **vLLM**: High-performance LLM inference server

---

## 💻 Yêu cầu hệ thống

### Hardware

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|---|---|---|
| **CPU** | 8 cores | 16+ cores |
| **RAM** | 16GB | 32GB+ |
| **GPU** | NVIDIA RTX 4060 (12GB) | RTX 5070/5060 Ti (16GB) |
| **Storage** | 50GB free | 100GB+ SSD |
| **VRAM** | 12GB | 16GB+ |

### Software

- **OS**: Linux (Ubuntu 22.04+) / Windows 11 + WSL2
- **Python**: 3.10 hoặc 3.11
- **CUDA**: 11.8+ / 12.1+
- **Docker** & **Docker Compose**
- **Conda** (khuyến nghị dùng Miniconda)

---

## 🚀 Cài đặt

### Bước 1: Clone repository

```bash
git clone https://github.com/JKLover0909/VLLM-PD.git
cd VLLM-PD
```

### Bước 2: Tạo môi trường Conda

```bash
# Tạo environment từ file
cd docmind
conda env create -f environment.yml

# Hoặc tạo thủ công
conda create -n docmind python=3.11 -y
conda activate docmind
```

### Bước 3: Cài đặt dependencies

```bash
# Cài đặt PyTorch (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Cài đặt packages từ requirements
pip install -r requirements.txt

# Nếu dùng DocMind
cd docmind
pip install -r requirements.txt
```

### Bước 4: Khởi động Infrastructure Services

```bash
# Khởi động Qdrant + LiteLLM
docker-compose up -d

# Kiểm tra trạng thái
docker-compose ps
```

### Bước 5: Cấu hình môi trường

```bash
# Tạo file .env
cp .env.example .env

# Chỉnh sửa các biến môi trường
nano .env
```

**Nội dung .env mẫu:**

```env
# LLM Configuration
VLLM_API_URL=http://localhost:8000/v1
OLLAMA_API_URL=http://localhost:11434

# Vector Database
QDRANT_URL=http://localhost:6333

# LiteLLM
LITELLM_URL=http://localhost:4000

# Cloud APIs (optional, for fallback)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

---

## 🎯 Hướng dẫn sử dụng

### Phương án 1: Chạy nhanh với script tự động (Khuyến nghị)

Script `run_all.sh` sẽ tự động khởi động vLLM + Backend + Frontend:

```bash
# Kích hoạt môi trường
conda activate docmind

# Di chuyển vào thư mục scripts
cd docmind/scripts

# Khởi động toàn bộ hệ thống
./run_all.sh start

# Kiểm tra trạng thái
./run_all.sh status

# Dừng toàn bộ
./run_all.sh stop
```

**Tùy chọn khởi động:**

```bash
# Chỉ chạy vLLM + Backend (không có Streamlit frontend)
./run_all.sh start --no-frontend

# Khởi động lại một service cụ thể
./run_all.sh restart vllm    # hoặc backend, frontend
```

**Log files được lưu tại:** `docmind/logs/`

---

### Phương án 2: Chạy thủ công từng service (Debug mode)

Phù hợp khi cần debug hoặc chạy riêng từng phần.

#### 2.1. Khởi động vLLM Server (Bắt buộc)

```bash
conda activate docmind
cd docmind/scripts
bash start_vllm.sh
```

vLLM sẽ chạy tại: `http://localhost:8000`

**Kiểm tra:**
```bash
curl http://localhost:8000/v1/models
```

#### 2.2. Khởi động FastAPI Backend (Bắt buộc)

```bash
conda activate docmind
cd docmind/backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Backend API sẽ chạy tại: `http://localhost:8001`

**Xem API docs:** http://localhost:8001/docs

#### 2.3. Khởi động Streamlit Frontend (Tùy chọn)

```bash
conda activate docmind
cd docmind/frontend
streamlit run app.py --server.port 8501
```

Web UI sẽ mở tại: `http://localhost:8501`

---

### Phương án 3: Sử dụng Jupyter Notebooks

Thư mục `Notebooks/` chứa các notebook để test:

```bash
conda activate docmind
jupyter lab

# Mở các notebook:
# - Test_docmind.ipynb: Test RAG pipeline
# - Test_LocalAPI.ipynb: Test backend API
```

---

## 📁 Cấu trúc dự án

```
VLLM-PD/
├── README.md                    # File này
├── ARCHITECTURE.md              # Tài liệu kiến trúc chi tiết
├── AGENTS.md                    # Hướng dẫn cấu hình Agent
├── requirements.txt             # Dependencies chung
├── docker-compose.yml           # Qdrant + LiteLLM
├── litellm_config.yaml          # Cấu hình LiteLLM Router
│
├── docmind/                     # 📚 RAG Document System
│   ├── README.md
│   ├── environment.yml          # Conda environment
│   ├── requirements.txt
│   ├── backend/
│   │   ├── main.py              # FastAPI app
│   │   ├── document_processor.py
│   │   ├── embedder.py          # BGE-M3 wrapper
│   │   ├── vector_store.py      # FAISS session manager
│   │   ├── rag_pipeline.py      # RAG orchestration
│   │   ├── vllm_client.py       # vLLM client
│   │   └── uploads/             # Session storage
│   ├── frontend/
│   │   └── app.py               # Streamlit UI
│   ├── scripts/
│   │   ├── run_all.sh           # 🚀 One-command launcher
│   │   └── start_vllm.sh        # vLLM starter
│   └── logs/                    # Log files
│
├── src/                         # 🤖 Agent System
│   ├── main.py
│   ├── agent/
│   │   ├── graph.py             # LangGraph workflow
│   │   └── mcp_client.py        # MCP protocol client
│   ├── api/
│   │   └── main.py              # Agent API server
│   └── rag/
│       ├── embedder.py
│       ├── parser.py
│       ├── rag_pipeline.py
│       └── vector_store.py
│
├── Notebooks/                   # 📓 Jupyter notebooks
│   ├── Test_docmind.ipynb
│   └── Test_LocalAPI.ipynb
│
├── tests/                       # 🧪 Unit tests
│   └── test_imports.py
│
└── documents/                   # 📄 Sample documents
```

---

## 🔌 API Endpoints

### DocMind Backend API

**Base URL:** `http://localhost:8001`

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/health` | Health check |
| `POST` | `/sessions` | Tạo session mới |
| `GET` | `/sessions` | Liệt kê các session |
| `POST` | `/sessions/{id}/upload` | Upload tài liệu |
| `POST` | `/sessions/{id}/index` | Index tài liệu vào vector DB |
| `POST` | `/sessions/{id}/query` | Hỏi đáp RAG (streaming) |
| `DELETE` | `/sessions/{id}` | Xóa session |

**Swagger UI:** http://localhost:8001/docs

### Agent API (src/)

**Base URL:** `http://localhost:8002` (nếu đã khởi động)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/agent/execute` | Thực thi Agent task |
| `GET` | `/agent/status` | Trạng thái Agent |

---

## 🛠️ Troubleshooting

### Vấn đề 1: vLLM không khởi động được

```bash
# Kiểm tra GPU
nvidia-smi

# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Giảm model size nếu thiếu VRAM
# Sửa trong start_vllm.sh: thay Llama 8B → 3B hoặc dùng quantization
```

### Vấn đề 2: Port đã được sử dụng

```bash
# Kiểm tra port 8000, 8001, 8501
lsof -i :8000
lsof -i :8001
lsof -i :8501

# Kill process nếu cần
kill -9 <PID>

# Hoặc dùng script stop
cd docmind/scripts
./run_all.sh stop
```

### Vấn đề 3: Docker services không chạy

```bash
# Kiểm tra logs
docker-compose logs qdrant
docker-compose logs litellm

# Restart services
docker-compose restart

# Rebuild nếu cần
docker-compose up -d --build
```

### Vấn đề 4: Import errors

```bash
# Reinstall dependencies
conda activate docmind
pip install -r requirements.txt --force-reinstall

# Hoặc tạo lại environment
conda env remove -n docmind
conda env create -f docmind/environment.yml
```

### Vấn đề 5: Out of Memory (OOM)

```bash
# Giảm batch size trong vLLM
# Sửa start_vllm.sh:
--max-model-len 2048 --gpu-memory-utilization 0.8

# Hoặc dùng CPU offloading
--cpu-offload-gb 8
```

---

## 📚 Tài liệu tham khảo

### Tài liệu chính

- [ARCHITECTURE.md](ARCHITECTURE.md) - Kiến trúc chi tiết
- [AGENTS.md](AGENTS.md) - Cấu hình Agent
- [docmind/README.md](docmind/README.md) - DocMind RAG system

### Công nghệ sử dụng

- [vLLM](https://docs.vllm.ai/) - LLM Inference Engine
- [LangGraph](https://python.langchain.com/docs/langgraph) - Agent Framework
- [Qdrant](https://qdrant.tech/documentation/) - Vector Database
- [LiteLLM](https://docs.litellm.ai/) - LLM Router/Proxy
- [FastAPI](https://fastapi.tiangolo.com/) - Web Framework
- [Streamlit](https://docs.streamlit.io/) - Web UI
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Multilingual Embedding
- [Docling](https://github.com/DS4SD/docling) - Document Processing
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - OCR Engine

### Model weights

- [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [Qwen3 14B Coder](https://huggingface.co/Qwen/Qwen3-14B-Coder)
- [Qwen3 VL 7B](https://huggingface.co/Qwen/Qwen3-VL-7B)

---

## 📝 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Hãy tạo issue hoặc pull request.

---

## 📧 Liên hệ

- GitHub: [@JKLover0909](https://github.com/JKLover0909)
- Repository: [VLLM-PD](https://github.com/JKLover0909/VLLM-PD)

---

**Phát triển bởi JKLover0909** • Last updated: June 2026