# 🏗️ ARCHITECTURE: Hệ thống AI Coding Agent & RAG Server Hybrid (Local-Cloud)

> **Phiên bản:** v1.0 — Ngày tạo: 2026-06-03
>
> **Mục tiêu:** Thiết kế kiến trúc và quy trình triển khai hệ thống AI Agent hỗ trợ lập trình (qua giao thức MCP) kết hợp hệ thống RAG xử lý tài liệu đa định dạng song ngữ Anh-Việt. Ưu tiên self-hosted, bảo mật dữ liệu và có khả năng fallback linh hoạt sang Cloud API.

---

## 📋 Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Sơ đồ kiến trúc & Luồng dữ liệu](#2-sơ-đồ-kiến-trúc--luồng-dữ-liệu)
3. [Chi tiết từng máy chủ](#3-chi-tiết-từng-máy-chủ)
4. [Đề xuất mô hình AI](#4-đề-xuất-mô-hình-ai)
5. [Stack công nghệ chi tiết](#5-stack-công-nghệ-chi-tiết)
6. [Thiết kế hệ thống Agent & MCP](#6-thiết-kế-hệ-thống-agent--mcp)
7. [Cơ chế Router/Fallback thông minh](#7-cơ-chế-routerfallback-thông-minh)
8. [Hệ thống RAG chi tiết](#8-hệ-thống-rag-chi-tiết)
9. [Bảo mật & Kết nối mạng](#9-bảo-mật--kết-nối-mạng)
10. [Sử dụng Cloud Credits](#10-sử-dụng-cloud-credits)
11. [Kế hoạch triển khai theo giai đoạn](#11-kế-hoạch-triển-khai-theo-giai-đoạn)

---

## 1. Tổng quan kiến trúc

Hệ thống bao gồm **3 máy tính** phân tán trên **2 dải mạng khác nhau**, kết nối qua mạng riêng ảo (VPN), hoạt động theo mô hình **microservices** với các vai trò rõ ràng:

| Máy | Vai trò | Hệ điều hành | GPU | VRAM |
|:---|:---|:---|:---|:---|
| **Máy 1** — LLM Host | Host mô hình LLM/VLM, phục vụ inference qua API chuẩn OpenAI | Windows 11 | RTX 5070 Ti | 16GB |
| **Máy 2** — RAG + Agent Server | Xử lý tài liệu, Vector DB, điều phối Agent, Embedding, MCP Hub | Linux (Ubuntu) | RTX 5060 Ti | 16GB |
| **Máy 3** — Developer Workstation | Máy trạm lập trình, VS Code, kết nối từ xa vào hệ thống | Bất kỳ | — | — |

---

## 2. Sơ đồ kiến trúc & Luồng dữ liệu

### 2.1. Sơ đồ tổng thể

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TAILSCALE MESH VPN (Private Network)                │
│                     Tất cả 3 máy kết nối bảo mật qua đây                   │
└───────┬─────────────────────────────┬───────────────────────┬───────────────┘
        │                             │                       │
        ▼                             ▼                       ▼
┌───────────────┐           ┌─────────────────────┐   ┌──────────────────┐
│   MÁY 3       │           │      MÁY 2          │   │     MÁY 1        │
│  Developer    │           │  RAG + Agent Server  │   │   LLM Host       │
│  Workstation  │           │     (Linux)          │   │   (Windows 11)   │
│               │           │                      │   │                  │
│  ┌──────────┐ │  MCP/SSE  │  ┌────────────────┐ │   │  ┌────────────┐ │
│  │ VS Code  │◄├───────────┤► │  LangGraph     │ │   │  │  Ollama    │ │
│  │ + Cline/ │ │           │  │  Agent Engine  │ │   │  │  Server    │ │
│  │ Roo Code │ │           │  │                │ │   │  │            │ │
│  └──────────┘ │           │  │  ┌──────────┐  │ │   │  │ ┌────────┐│ │
│               │           │  │  │ MCP Hub  │  │ │   │  │ │Qwen3   ││ │
│  ┌──────────┐ │           │  │  │ Server   │  │ │   │  │ │Coder   ││ │
│  │ Browser  │ │  HTTP     │  │  └──────────┘  │ │   │  │ │14B     ││ │
│  │ Chat UI  │◄├───────────┤► │                │ │   │  │ └────────┘│ │
│  └──────────┘ │           │  └───────┬────────┘ │   │  │ ┌────────┐│ │
│               │           │          │          │   │  │ │Qwen3   ││ │
│               │           │          ▼          │   │  │ │VL-7B   ││ │
│               │           │  ┌────────────────┐ │   │  │ └────────┘│ │
│               │           │  │  LiteLLM       │ │   │  └────────────┘ │
│               │           │  │  Router/Proxy  │─┤───┤──►  API :11434  │
│               │           │  │                │ │   │                  │
│               │           │  │  ┌──────────┐  │ │   └──────────────────┘
│               │           │  │  │Fallback  │  │ │
│               │           │  │  │→ OpenAI  │  │ │          ┌──────────┐
│               │           │  │  │→ Mimo    │  │─┤─────────►│ Cloud    │
│               │           │  │  │→ Azure   │  │ │          │ APIs     │
│               │           │  │  └──────────┘  │ │          └──────────┘
│               │           │  └────────────────┘ │
│               │           │                      │
│               │           │  ┌────────────────┐ │
│               │           │  │  RAG Pipeline  │ │
│               │           │  │                │ │
│               │           │  │  ┌──────────┐  │ │
│               │           │  │  │ Docling  │  │ │
│               │           │  │  │ Parser   │  │ │
│               │           │  │  └──────────┘  │ │
│               │           │  │  ┌──────────┐  │ │
│               │           │  │  │ BGE-M3   │  │ │
│               │           │  │  │Embedding │  │ │
│               │           │  │  └──────────┘  │ │
│               │           │  │  ┌──────────┐  │ │
│               │           │  │  │ Qdrant   │  │ │
│               │           │  │  │VectorDB  │  │ │
│               │           │  │  └──────────┘  │ │
│               │           │  └────────────────┘ │
└───────────────┘           └─────────────────────┘
```

### 2.2. Luồng dữ liệu chính

```
  LUỒNG 1: Coding Agent (MCP)
  ════════════════════════════
  Máy 3 (VS Code) ──MCP/SSE──► Máy 2 (LangGraph Agent)
                                    │
                                    ├──► LiteLLM Router ──► Máy 1 (Ollama: Qwen3-Coder-14B)
                                    │                   └──► [Fallback] OpenAI / Mimo Pro
                                    │
                                    ├──► MCP Server: filesystem (đọc/ghi code)
                                    ├──► MCP Server: terminal (chạy lệnh)
                                    └──► MCP Server: web-search (tìm kiếm web)


  LUỒNG 2: RAG Hỏi đáp Tài liệu
  ════════════════════════════════
  Máy 3 (Browser/Chat UI) ──HTTP──► Máy 2 (FastAPI Gateway)
                                        │
                                        ├──► Docling (parse PDF/DOCX/XLSX/ảnh)
                                        ├──► BGE-M3 Embedding (chạy trên GPU Máy 2)
                                        ├──► Qdrant Vector DB (tìm kiếm ngữ nghĩa)
                                        │
                                        └──► LiteLLM Router ──► Máy 1 (Ollama: Qwen3-VL-7B)
                                                            └──► [Fallback] OpenAI GPT-4o


  LUỒNG 3: Xử lý Hình ảnh / OCR tiếng Việt
  ═══════════════════════════════════════════
  Tài liệu ảnh ──► Máy 2 (Docling + Vintern-1B hoặc Qwen3-VL-7B)
                        │
                        ├──► OCR local (ưu tiên)
                        └──► [Fallback nặng] Google Document AI (dùng $300 credit)
```

---

## 3. Chi tiết từng máy chủ

### 3.1. Máy 1 — LLM Host (Windows 11, RTX 5070 Ti 16GB)

**Vai trò cốt lõi:** Chuyên chạy inference cho các mô hình LLM/VLM lớn, phục vụ API chuẩn OpenAI-compatible cho toàn hệ thống.

**Phần mềm triển khai:**

| Thành phần | Công cụ | Lý do chọn |
|:---|:---|:---|
| Inference Engine | **Ollama** | Dễ cài trên Windows, tự động quản lý VRAM, hỗ trợ hot-swap model, API chuẩn OpenAI |
| Kết nối mạng | **Tailscale** (thay Ngrok) | Bảo mật hơn Ngrok, không cần cổng public, ping thấp hơn |

**Phân bổ VRAM (16GB) — Chiến lược đa model:**

Ollama hỗ trợ **tự động load/unload** model theo nhu cầu (chỉ load model đang dùng vào VRAM), do đó bạn có thể cài **nhiều model** nhưng chỉ 1 model hoạt động tại một thời điểm:

| Model | Kích thước trên VRAM | Dùng cho | Ghi chú |
|:---|:---|:---|:---|
| **Qwen3-Coder-14B** (Q4_K_M) | ~10GB | Coding Agent, lập trình, sửa lỗi | Model coding chuyên dụng, hỗ trợ function calling tốt |
| **Qwen3-VL-7B** (Q4_K_M) | ~6GB | RAG hỏi đáp tài liệu + hình ảnh | Hiểu ảnh, bảng biểu, tiếng Việt tốt |
| **Gemma 4 26B MoE** (Q4_K_M) | ~14GB | Task phức tạp cần suy luận sâu | Chỉ load khi cần, dùng Thinking Mode |

> ⚠️ **Lưu ý quan trọng:** Với Q4_K_M quantization, model 14B chiếm ~10GB VRAM, còn lại ~6GB cho KV Cache (context window). Đủ cho context ~16K-32K tokens. Nếu cần context dài hơn, hãy chuyển sang dùng model 7B (chỉ chiếm ~6GB, dư ~10GB cho context rất dài).

### 3.2. Máy 2 — RAG + Agent Server (Linux, RTX 5060 Ti 16GB)

**Vai trò cốt lõi:** Bộ não điều phối toàn bộ hệ thống — chạy Agent Engine, RAG Pipeline, Vector DB, Embedding Model và LiteLLM Router.

**Phân bổ VRAM (16GB):**

| Thành phần | VRAM sử dụng | Ghi chú |
|:---|:---|:---|
| **BGE-M3 Embedding** | ~2GB | Mô hình embedding đa ngữ, chạy thường trực |
| **Vintern-1B** (OCR tiếng Việt) | ~1.5GB | Mô hình VLM nhỏ chuyên OCR tiếng Việt, chạy thường trực |
| **Docling** (AI layout models) | ~2GB | TableFormer + Layout Analysis |
| **Dự phòng / KV Cache** | ~10.5GB | Dành cho burst processing hoặc model phụ |

**Các service chạy trên Máy 2:**

| Service | Port | Mô tả |
|:---|:---|:---|
| FastAPI Gateway | `:8000` | API Gateway chính, xử lý mọi request |
| LiteLLM Proxy | `:4000` | Router chuyển hướng request LLM |
| Qdrant | `:6333` | Vector Database |
| LangGraph Agent | Internal | Điều phối Agent logic |
| MCP Hub | `:3000` | Tập trung các MCP Server |
| Chat UI (Open WebUI) | `:8080` | Giao diện chat cho RAG hỏi đáp |

### 3.3. Máy 3 — Developer Workstation

**Vai trò:** Máy trạm lập trình, nơi bạn ngồi code hàng ngày.

**Cấu hình:**

| Thành phần | Chi tiết |
|:---|:---|
| IDE | VS Code + Extension **Cline** hoặc **Roo Code** |
| Kết nối | Tailscale VPN → SSH remote vào Máy 2 (VS Code Remote SSH) |
| MCP Client | Tích hợp sẵn trong VS Code, kết nối tới MCP Hub trên Máy 2 |

---

## 4. Đề xuất mô hình AI

### 4.1. Mô hình chạy Local (Tự host)

| Tên model | Kích thước | Quantize | Máy chạy | Tác vụ | Ưu điểm |
|:---|:---|:---|:---|:---|:---|
| **Qwen3-Coder-14B** | 14B | Q4_K_M | Máy 1 | Coding Agent chính | Chuyên coding, function calling chuẩn, hỗ trợ Thinking Mode |
| **Qwen3-VL-7B** | 7B | Q4_K_M | Máy 1 | RAG hỏi đáp + Vision | Đọc hiểu ảnh, bảng biểu, hỗ trợ tiếng Việt tốt |
| **Gemma 4 26B MoE** | 26B (4B active) | Q4_K_M | Máy 1 | Reasoning nâng cao | MoE nên nhanh, Thinking Mode mạnh |
| **BGE-M3** | ~568M | FP16 | Máy 2 | Embedding đa ngữ | Hỗ trợ dense + sparse retrieval, tiếng Việt tốt |
| **Vintern-1B** | 1B | FP16 | Máy 2 | OCR tiếng Việt chuyên sâu | Nhẹ, nhanh, chuyên biệt cho tài liệu Việt |

### 4.2. Mô hình Cloud API (Dự phòng / Fallback)

| Provider | Model | Dùng khi |
|:---|:---|:---|
| **OpenAI** | GPT-4o / GPT-4.1 | Task coding cực khó, logic phức tạp vượt khả năng local |
| **Xiaomi Mimo Pro** | Mimo Pro | Backup cho các tác vụ tổng hợp, hỏi đáp thông thường |
| **Google Cloud** | Document AI | OCR tài liệu scan phức tạp (dùng $300 credit) |
| **Azure OpenAI** | GPT-4o (Azure) | Backup enterprise-grade (dùng $100 student credit) |

---

## 5. Stack công nghệ chi tiết

### 5.1. Tổng quan Stack

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                    │
│  VS Code + Cline/Roo Code │ Open WebUI (Chat) │ Browser │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│                    GATEWAY LAYER (Máy 2)                 │
│              FastAPI + LiteLLM Proxy Router              │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│                ORCHESTRATION LAYER (Máy 2)               │
│           LangGraph (Agent Engine) + MCP Hub             │
└───────┬─────────────┬──────────────┬────────────────────┘
        │             │              │
┌───────▼──────┐ ┌────▼─────┐ ┌─────▼──────────┐
│  RAG         │ │  MCP     │ │  INFERENCE     │
│  Pipeline    │ │  Servers │ │  LAYER         │
│              │ │          │ │                │
│ • Docling    │ │ • FS     │ │ • Máy 1:      │
│ • BGE-M3    │ │ • Term   │ │   Ollama       │
│ • Qdrant    │ │ • Web    │ │ • Cloud:       │
│ • Vintern   │ │ • Git    │ │   OpenAI/Mimo  │
└──────────────┘ └──────────┘ └────────────────┘
```

### 5.2. Chi tiết từng thành phần

#### 🔧 Inference Engine (Máy 1)

| Công cụ | Phiên bản | Lý do chọn |
|:---|:---|:---|
| **Ollama** | Latest | Cài đặt đơn giản trên Windows, API chuẩn OpenAI, tự quản lý VRAM, hỗ trợ hot-swap model |

> **Tại sao Ollama mà không phải vLLM trên Windows?**
> - vLLM chủ yếu tối ưu cho Linux, cài trên Windows phức tạp.
> - Ollama có bản Windows native, cài 1 click, API tương thích OpenAI.
> - Với 1 user (bạn), throughput của Ollama là đủ. vLLM chỉ cần khi phục vụ nhiều user đồng thời.

#### 🧠 Agent Orchestration (Máy 2)

| Công cụ | Vai trò | Lý do chọn |
|:---|:---|:---|
| **LangGraph** | Điều phối Agent, quản lý state | Framework mạnh nhất cho agentic workflow phức tạp, hỗ trợ cycles, persistence, human-in-the-loop |
| **langchain-mcp-adapters** | Kết nối MCP tools → LangGraph | Chuyển đổi MCP tools thành LangChain tools chuẩn để LangGraph sử dụng |
| **FastAPI** | API Gateway | Async, hiệu năng cao, dễ tùy biến, tích hợp tốt với LangGraph |

#### 📄 Document Processing (Máy 2)

| Công cụ | Vai trò | Lý do chọn |
|:---|:---|:---|
| **Docling** (IBM) | Parse PDF, DOCX, XLSX, ảnh | Xử lý đa định dạng trong 1 pipeline, trích xuất bảng biểu bằng AI (TableFormer), export Markdown/JSON |
| **PyMuPDF** | Backup cho PDF nhanh | Tốc độ cao khi cần extract text đơn giản từ PDF |
| **openpyxl** | Xử lý Excel nâng cao | Đọc dữ liệu có cấu trúc từ Excel sheets |

#### 🔍 Vector Database & Embedding (Máy 2)

| Công cụ | Vai trò | Lý do chọn |
|:---|:---|:---|
| **Qdrant** | Vector Database | Hiệu năng cao (Rust), hỗ trợ metadata filtering mạnh, tự host dễ, phù hợp quy mô vừa |
| **BGE-M3** (BAAI) | Embedding Model | Hỗ trợ dense + sparse + multi-vector, tiếng Việt tốt, ~568M params chạy nhẹ trên GPU |

> **Tại sao Qdrant mà không phải ChromaDB?**
> - ChromaDB phù hợp prototype/dev, nhưng Qdrant mạnh hơn rất nhiều về filtering, performance và production-ready.
> - Qdrant viết bằng Rust, tốc độ query nhanh, hỗ trợ disk-based index (tiết kiệm RAM).

#### 🔀 LLM Router (Máy 2)

| Công cụ | Vai trò | Lý do chọn |
|:---|:---|:---|
| **LiteLLM Proxy** | Router/Load Balancer | Hỗ trợ 100+ LLM providers, fallback tự động, health check, API chuẩn OpenAI |

#### 📊 Observability (Máy 2)

| Công cụ | Vai trò | Lý do chọn |
|:---|:---|:---|
| **Langfuse** (self-hosted) | Tracing & Monitoring | Theo dõi chuỗi suy luận của Agent, debug lỗi, đo latency, tự host được |

---

## 6. Thiết kế hệ thống Agent & MCP

### 6.1. Kiến trúc MCP

```
┌──────────────────────────────────────────────────────────────────┐
│                       MÁY 2 (Linux Server)                       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  LangGraph Agent Engine                   │    │
│  │                                                           │    │
│  │   ┌─────────┐    ┌──────────┐    ┌──────────────────┐   │    │
│  │   │  Plan   │───►│ Execute  │───►│ Evaluate/Retry   │   │    │
│  │   │  Node   │    │  Node    │    │     Node         │   │    │
│  │   └─────────┘    └────┬─────┘    └──────────────────┘   │    │
│  │                       │                                   │    │
│  │              ┌────────▼────────┐                          │    │
│  │              │  MCP Client Hub │ (langchain-mcp-adapters) │    │
│  │              └────────┬────────┘                          │    │
│  └───────────────────────┼───────────────────────────────────┘    │
│                          │                                        │
│          ┌───────────────┼───────────────────────┐               │
│          │               │                       │               │
│    ┌─────▼─────┐   ┌────▼──────┐   ┌───────────▼──────────┐   │
│    │ MCP Server│   │MCP Server │   │  MCP Server          │   │
│    │filesystem │   │ terminal  │   │  brave-search / tavily│   │
│    │           │   │           │   │                       │   │
│    │ • read    │   │ • exec    │   │  • web_search        │   │
│    │ • write   │   │ • run_test│   │  • fetch_url         │   │
│    │ • list    │   │ • install │   │                       │   │
│    └───────────┘   └───────────┘   └───────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2. Các MCP Server cần thiết lập

| MCP Server | Transport | Chức năng | Cách cài |
|:---|:---|:---|:---|
| **@modelcontextprotocol/server-filesystem** | stdio | Đọc/ghi/sửa file code | `npx @modelcontextprotocol/server-filesystem <workspace_path>` |
| **mcp-server-terminal** | stdio | Chạy lệnh terminal, test, build | Cài qua pip hoặc npx |
| **@anthropic/mcp-server-brave-search** hoặc **tavily** | stdio | Tìm kiếm web | Cần API key của Brave Search hoặc Tavily |
| **mcp-server-git** | stdio | Thao tác Git (commit, diff, log) | `npx @modelcontextprotocol/server-git` |
| **mcp-server-qdrant** | stdio | Truy vấn Vector DB trực tiếp | Cài qua pip |

### 6.3. Cấu hình MCP trong VS Code (Máy 3)

Tạo file `.vscode/mcp.json` trong workspace:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/jkl0909/Code"
      ]
    },
    "terminal": {
      "command": "npx",
      "args": ["-y", "mcp-server-terminal"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "${env:BRAVE_API_KEY}"
      }
    },
    "git": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-git",
        "--repository", "/home/jkl0909/Code"
      ]
    }
  }
}
```

### 6.4. Luồng hoạt động của Coding Agent

```
  Người dùng (VS Code) gửi yêu cầu: "Hãy sửa lỗi import ở file utils.py"
       │
       ▼
  ┌──────────────────────────────────────┐
  │  1. PLAN NODE (LangGraph)            │
  │  LLM phân tích yêu cầu → Lên kế hoạch│
  │  "Cần đọc file utils.py trước"       │
  └────────────┬─────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────┐
  │  2. EXECUTE NODE                     │
  │  Gọi MCP Tool: filesystem.read_file  │
  │  → Nhận nội dung utils.py            │
  └────────────┬─────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────┐
  │  3. REASON NODE                      │
  │  LLM (Qwen3-Coder-14B via LiteLLM)  │
  │  Phân tích lỗi → Sinh code sửa      │
  └────────────┬─────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────┐
  │  4. EXECUTE NODE                     │
  │  Gọi MCP Tool: filesystem.write_file │
  │  → Ghi code đã sửa vào utils.py     │
  └────────────┬─────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────┐
  │  5. VERIFY NODE                      │
  │  Gọi MCP Tool: terminal.run_command  │
  │  → Chạy test: "python -m pytest"     │
  │  → Kiểm tra kết quả pass/fail       │
  └────────────┬─────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────┐
  │  6. EVALUATE NODE                    │
  │  Test pass? → Trả kết quả cho user   │
  │  Test fail? → Quay lại bước 3 (retry)│
  └──────────────────────────────────────┘
```

---

## 7. Cơ chế Router/Fallback thông minh

### 7.1. Thiết kế Router bằng LiteLLM Proxy

LiteLLM Proxy chạy trên **Máy 2** như một middleware trung tâm, mọi request LLM từ Agent hoặc RAG đều đi qua đây:

```
  Mọi request ──► LiteLLM Proxy (:4000)
                      │
                      ├── Ưu tiên 1: Local Model (Máy 1 Ollama)
                      │     ├── Healthy? → Forward request
                      │     └── Lỗi / Timeout? ──► Fallback
                      │
                      ├── Ưu tiên 2: Xiaomi Mimo Pro API
                      │     ├── Available? → Forward
                      │     └── Lỗi? ──► Fallback tiếp
                      │
                      └── Ưu tiên 3: OpenAI API
                            └── Luôn available (trả phí)
```

### 7.2. Cấu hình LiteLLM Proxy (`config.yaml`)

```yaml
# File: /home/jkl0909/Code/llm/VLLM-PD/litellm_config.yaml

model_list:
  # ============================================================
  # CODING MODEL — Dùng cho Agent lập trình
  # ============================================================
  - model_name: coding-model
    litellm_params:
      model: ollama/qwen3-coder-14b
      api_base: http://<tailscale-ip-may1>:11434  # Máy 1 qua Tailscale
      timeout: 120
      stream: true

  - model_name: coding-model-fallback
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      timeout: 60

  # ============================================================
  # VISION MODEL — Dùng cho RAG hình ảnh & tài liệu
  # ============================================================
  - model_name: vision-model
    litellm_params:
      model: ollama/qwen3-vl-7b
      api_base: http://<tailscale-ip-may1>:11434
      timeout: 90
      stream: true

  - model_name: vision-model-fallback
    litellm_params:
      model: openai/gpt-4o  # GPT-4o hỗ trợ vision
      api_key: os.environ/OPENAI_API_KEY

  # ============================================================
  # GENERAL MODEL — Hỏi đáp, tóm tắt, phân loại
  # ============================================================
  - model_name: general-model
    litellm_params:
      model: ollama/gemma4-26b-a4b
      api_base: http://<tailscale-ip-may1>:11434
      timeout: 120

  - model_name: general-model-fallback-mimo
    litellm_params:
      model: openai/mimo-pro  # Xiaomi Mimo Pro (OpenAI-compatible)
      api_key: os.environ/MIMO_API_KEY
      api_base: https://api.mimo.xiaomi.com/v1  # URL ví dụ

  - model_name: general-model-fallback-openai
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

# ============================================================
# ROUTER SETTINGS
# ============================================================
router_settings:
  # Khi model chính lỗi → tự động chuyển sang fallback
  fallbacks:
    - coding-model: ["coding-model-fallback"]
    - vision-model: ["vision-model-fallback"]
    - general-model: ["general-model-fallback-mimo", "general-model-fallback-openai"]

  # Health check mỗi 30 giây để phát hiện sớm Máy 1 offline
  enable_health_check: true
  health_check_interval: 30

  # Không retry model lỗi, chuyển fallback ngay lập tức
  num_retries: 0

  # Timeout mặc định
  timeout: 120

  # Routing strategy
  routing_strategy: "simple-shuffle"  # Hoặc "latency-based-routing"

litellm_settings:
  # Log chi tiết để debug
  set_verbose: false
  # Cache response để tiết kiệm (optional)
  cache: true
  cache_params:
    type: "local"
    ttl: 600  # 10 phút
```

### 7.3. Khởi chạy LiteLLM Proxy

```bash
# Trên Máy 2 (Linux)
litellm --config /home/jkl0909/Code/llm/VLLM-PD/litellm_config.yaml \
        --port 4000 \
        --host 0.0.0.0
```

### 7.4. Sử dụng trong code Python

```python
# Mọi service trên Máy 2 gọi LLM thông qua LiteLLM Proxy
# Cú pháp giống hệt OpenAI SDK — dễ migrate

from openai import OpenAI

client = OpenAI(
    api_key="sk-any-key",  # LiteLLM Proxy không cần key thật
    base_url="http://localhost:4000/v1"  # LiteLLM Proxy trên Máy 2
)

# Gọi Coding Model (tự động fallback nếu Máy 1 lỗi)
response = client.chat.completions.create(
    model="coding-model",
    messages=[
        {"role": "system", "content": "Bạn là trợ lý lập trình chuyên nghiệp."},
        {"role": "user", "content": "Hãy viết hàm quicksort bằng Python"}
    ],
    stream=True
)
```

---

## 8. Hệ thống RAG chi tiết

### 8.1. Pipeline xử lý tài liệu

```
  Tài liệu đầu vào
  (PDF, DOCX, XLSX, ảnh)
       │
       ▼
  ┌──────────────────────────────────────┐
  │  BƯỚC 1: DOCUMENT PARSING           │
  │                                      │
  │  Docling (IBM)                       │
  │  ├── PDF → Text + Table + Layout     │
  │  ├── DOCX → Markdown structured      │
  │  ├── XLSX → Bảng dữ liệu có headers │
  │  └── Ảnh → Vintern-1B (OCR Việt)     │
  │          hoặc Qwen3-VL-7B (Máy 1)   │
  └────────────┬─────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────┐
  │  BƯỚC 2: CHUNKING                   │
  │                                      │
  │  LangChain RecursiveCharacterSplitter│
  │  ├── chunk_size: 1000 tokens         │
  │  ├── chunk_overlap: 200 tokens       │
  │  └── Giữ nguyên cấu trúc bảng biểu  │
  └────────────┬─────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────┐
  │  BƯỚC 3: EMBEDDING                  │
  │                                      │
  │  BGE-M3 (chạy trên GPU Máy 2)       │
  │  ├── Dense vector (1024 dims)        │
  │  ├── Sparse vector (BM25-like)       │
  │  └── Multi-vector (ColBERT-like)     │
  └────────────┬─────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────┐
  │  BƯỚC 4: INDEXING                   │
  │                                      │
  │  Qdrant Vector Database              │
  │  ├── Collection: "documents_vi_en"   │
  │  ├── Payload: metadata (filename,    │
  │  │   page, language, doc_type)       │
  │  └── Hybrid search: dense + sparse   │
  └──────────────────────────────────────┘
```

### 8.2. Pipeline truy vấn RAG (Query)

```
  Câu hỏi người dùng: "Trong báo cáo Q3, doanh thu là bao nhiêu?"
       │
       ▼
  ┌──────────────────────────────────────┐
  │  1. QUERY EXPANSION                 │
  │  LLM tạo thêm các biến thể câu hỏi  │
  │  để tăng recall:                     │
  │  - "revenue Q3 report"              │
  │  - "doanh thu quý 3 báo cáo"       │
  └────────────┬─────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────┐
  │  2. HYBRID SEARCH (Qdrant)          │
  │  ├── Dense search (semantic)         │
  │  ├── Sparse search (keyword)         │
  │  └── Rerank: Cross-encoder hoặc LLM │
  │      → Top-K chunks (K=5)           │
  └────────────┬─────────────────────────┘
               │
               ▼
  ┌──────────────────────────────────────┐
  │  3. GENERATION                      │
  │  LLM (qua LiteLLM Router):          │
  │  - Context: top-K chunks             │
  │  - System prompt: "Trả lời bằng     │
  │    tiếng Việt, trích dẫn nguồn"     │
  │  → Câu trả lời + citations          │
  └──────────────────────────────────────┘
```

---

## 9. Bảo mật & Kết nối mạng

### 9.1. Thay thế Ngrok bằng Tailscale VPN

> **Khuyến nghị mạnh:** Chuyển từ Ngrok sang **Tailscale** để kết nối Máy 1 và Máy 2.

| Tiêu chí | Ngrok (Hiện tại) | Tailscale (Đề xuất) |
|:---|:---|:---|
| **Bảo mật** | ⚠️ Tạo URL public, ai cũng truy cập được | ✅ Mạng riêng tư, Zero Trust, mã hóa WireGuard |
| **Tốc độ** | ⚠️ Đi qua relay server của Ngrok | ✅ Kết nối P2P trực tiếp giữa 2 máy (nếu có thể) |
| **Ổn định** | ⚠️ URL thay đổi khi restart (bản free) | ✅ IP Tailscale cố định (100.x.x.x) |
| **Chi phí** | ⚠️ Bản free giới hạn bandwidth | ✅ Miễn phí cho cá nhân (lên đến 100 thiết bị) |
| **Setup** | Đơn giản | Đơn giản (cài client trên cả 2 máy, login cùng tài khoản) |

**Cách cài Tailscale:**

```bash
# Trên Máy 1 (Windows 11):
# Tải installer từ https://tailscale.com/download/windows
# Đăng nhập bằng Google/GitHub account

# Trên Máy 2 (Linux):
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Trên Máy 3 (Developer Workstation):
# Cài Tailscale client và đăng nhập cùng tài khoản
```

Sau khi cài, cả 3 máy sẽ có IP nội bộ dạng `100.x.x.x` và có thể giao tiếp trực tiếp như trong cùng LAN.

### 9.2. Ma trận bảo mật

| Lớp bảo mật | Giải pháp |
|:---|:---|
| **Mạng** | Tailscale VPN (WireGuard encryption) |
| **API Authentication** | API Key cho LiteLLM Proxy, Bearer token cho FastAPI |
| **Dữ liệu** | Toàn bộ dữ liệu lưu local trên Máy 2, không gửi lên cloud (trừ khi fallback) |
| **MCP** | Giới hạn filesystem access trong workspace path cụ thể |
| **Cloud Fallback** | Chỉ gửi prompt/query, KHÔNG gửi toàn bộ tài liệu gốc lên cloud |

---

## 10. Sử dụng Cloud Credits

### 10.1. Google Cloud ($300 Free Trial)

| Dịch vụ | Chi phí ước tính | Mục đích |
|:---|:---|:---|
| **Document AI (OCR)** | ~$1.5/1000 trang | Xử lý tài liệu scan phức tạp mà local OCR (Vintern-1B) chưa chính xác |
| **Cloud Storage** | ~$0.02/GB/tháng | Backup tài liệu gốc (optional) |
| **Vertex AI** | Pay-per-use | Thử nghiệm Gemma 4 trên cloud (nếu muốn so sánh với local) |

> 💡 **Chiến lược:** Dùng $300 chủ yếu cho Document AI OCR. Với giá ~$1.5/1000 trang, bạn có thể OCR ~200,000 trang tài liệu. Đủ dùng rất lâu.

### 10.2. Azure Student ($100)

| Dịch vụ | Chi phí ước tính | Mục đích |
|:---|:---|:---|
| **Azure OpenAI Service** | Pay-per-token | Backup API cho coding agent khi cần model cực mạnh (GPT-4o) |
| **Azure AI Search** | $0 (Free tier) | Thử nghiệm so sánh với Qdrant (optional) |

> 💡 **Chiến lược:** Reserve $100 Azure cho Azure OpenAI API. Dùng làm "bảo hiểm" cho những task coding cực khó mà local model + OpenAI trực tiếp đều thất bại.

---

## 11. Kế hoạch triển khai theo giai đoạn

### Phase 1: Nền tảng mạng & Inference (Tuần 1)

- [ ] Cài Tailscale trên cả 3 máy, xác nhận kết nối P2P
- [ ] Cài Ollama trên Máy 1 (Windows), tải Qwen3-Coder-14B + Qwen3-VL-7B
- [ ] Kiểm tra API Ollama hoạt động qua Tailscale IP từ Máy 2
- [ ] Gỡ bỏ Ngrok (sau khi Tailscale ổn định)

### Phase 2: LiteLLM Router & Fallback (Tuần 1-2)

- [ ] Cài LiteLLM Proxy trên Máy 2
- [ ] Cấu hình `config.yaml` với local model + OpenAI + Mimo Pro fallback
- [ ] Test fallback: tắt Ollama → xác nhận request chuyển sang OpenAI
- [ ] Cài Langfuse (self-hosted) để monitoring

### Phase 3: RAG Pipeline (Tuần 2-3)

- [ ] Cài Qdrant trên Máy 2 (Docker hoặc binary)
- [ ] Cài BGE-M3 embedding model trên Máy 2 (qua sentence-transformers hoặc TEI)
- [ ] Cài Docling, viết script ingest tài liệu (PDF/DOCX/XLSX)
- [ ] Cài Vintern-1B cho OCR tiếng Việt
- [ ] Xây dựng FastAPI endpoint cho RAG query
- [ ] Cài Open WebUI làm giao diện chat cho RAG

### Phase 4: Agent & MCP (Tuần 3-4)

- [ ] Xây dựng LangGraph Agent Engine trên Máy 2
- [ ] Cài đặt các MCP Server (filesystem, terminal, web-search, git)
- [ ] Tích hợp langchain-mcp-adapters để kết nối MCP → LangGraph
- [ ] Cài Cline/Roo Code trên VS Code (Máy 3), cấu hình kết nối
- [ ] Test end-to-end: yêu cầu sửa code → Agent đọc file → sửa → chạy test

### Phase 5: Tối ưu & Production (Tuần 4+)

- [ ] Fine-tune prompt template cho từng tác vụ (coding, RAG, OCR)
- [ ] Cấu hình Google Document AI làm backup OCR (dùng $300 credit)
- [ ] Thiết lập Azure OpenAI làm backup cho coding model
- [ ] Viết script tự động khởi động tất cả services (systemd trên Linux)
- [ ] Viết tài liệu vận hành & troubleshooting

---

## 📎 Phụ lục: Cấu trúc thư mục dự án

```
VLLM-PD/
├── ARCHITECTURE.md              # File này
├── AGENTS.md                    # Quy tắc cho AI Agent
├── docker-compose.yml           # Orchestrate services trên Máy 2
├── litellm_config.yaml          # Cấu hình LiteLLM Router
├── .vscode/
│   └── mcp.json                 # Cấu hình MCP cho VS Code
├── src/
│   ├── agent/
│   │   ├── graph.py             # LangGraph Agent definition
│   │   ├── nodes.py             # Agent nodes (plan, execute, evaluate)
│   │   ├── state.py             # Agent state schema
│   │   └── prompts.py           # System prompts cho từng tác vụ
│   ├── rag/
│   │   ├── ingest.py            # Document ingestion pipeline
│   │   ├── chunker.py           # Text chunking logic
│   │   ├── embedder.py          # BGE-M3 embedding wrapper
│   │   ├── retriever.py         # Qdrant hybrid search
│   │   └── generator.py         # RAG generation logic
│   ├── mcp/
│   │   ├── servers/             # Custom MCP server definitions
│   │   └── client.py            # MCP client integration
│   ├── router/
│   │   └── middleware.py        # Custom routing logic (nếu cần)
│   └── api/
│       ├── main.py              # FastAPI application
│       ├── routes/
│       │   ├── chat.py          # Chat/RAG endpoints
│       │   ├── agent.py         # Agent task endpoints
│       │   └── documents.py     # Document upload/manage
│       └── models/
│           └── schemas.py       # Pydantic models
├── scripts/
│   ├── setup_may1.sh            # Script cài đặt cho Máy 1
│   ├── setup_may2.sh            # Script cài đặt cho Máy 2
│   └── start_services.sh        # Khởi động toàn bộ services
├── Notebooks/
│   └── Test_LocalAPI.ipynb      # Notebook test API
└── tests/
    ├── test_router.py           # Test LiteLLM fallback
    ├── test_rag.py              # Test RAG pipeline
    └── test_agent.py            # Test Agent workflow
```

---

> **Ghi chú cuối:** Kiến trúc này được thiết kế theo nguyên tắc **modular** — mỗi thành phần có thể thay thế hoặc nâng cấp độc lập. Ví dụ: Bạn có thể đổi Ollama sang vLLM khi chuyển Máy 1 sang Linux, hoặc thay Qdrant bằng Milvus khi dữ liệu tăng lên hàng triệu documents, mà không cần thay đổi phần còn lại của hệ thống.
