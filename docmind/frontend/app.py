"""
app.py — DocMind Streamlit Frontend
Giao diện hỏi đáp tài liệu kiểu NotebookLM với hỗ trợ tiếng Việt.
"""

import json
import time
import uuid
import requests
import streamlit as st

# ─────────────────────────────────────
# Config
# ─────────────────────────────────────
BACKEND_URL = "http://localhost:8001"
ALLOWED_TYPES = ["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp"]

st.set_page_config(
    page_title="DocMind — Trợ lý Tài liệu AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────
# Custom CSS (dark premium theme)
# ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Header */
.docmind-header {
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* Chat bubble — user */
.chat-user {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0;
    margin-left: 20%;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3);
}

/* Chat bubble — assistant */
.chat-assistant {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    margin-right: 10%;
    backdrop-filter: blur(10px);
    line-height: 1.7;
}

/* Source badge */
.source-badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px 3px;
}

/* File card */
.file-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 10px 14px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.2s;
}

/* Status pill */
.status-ok {
    background: rgba(16,185,129,0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
}
.status-warn {
    background: rgba(245,158,11,0.15);
    color: #fcd34d;
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
}

/* Input styling */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important;
}

/* Scrollable chat area */
.chat-container {
    max-height: 65vh;
    overflow-y: auto;
    padding-right: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────
# Session State Init
# ─────────────────────────────────────
def init_state():
    defaults = {
        "session_id": None,
        "messages": [],       # [{"role": "user"|"assistant", "content": str, "sources": list}]
        "indexed_files": [],  # list of filenames
        "vllm_status": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ─────────────────────────────────────
# API Helpers
# ─────────────────────────────────────
def api_create_session() -> str | None:
    try:
        r = requests.post(f"{BACKEND_URL}/sessions", timeout=5)
        r.raise_for_status()
        return r.json()["session_id"]
    except Exception as e:
        st.error(f"❌ Không thể kết nối backend: {e}")
        return None


def api_health() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "error", "vllm_connected": False}


def api_upload(session_id: str, file) -> dict | None:
    try:
        r = requests.post(
            f"{BACKEND_URL}/sessions/{session_id}/upload",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        st.error(f"❌ Upload lỗi: {detail}")
        return None
    except Exception as e:
        st.error(f"❌ Upload lỗi: {e}")
        return None


def api_query(session_id: str, question: str) -> tuple[str, list]:
    """Non-streaming query, trả về (answer, sources)."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/query",
            json={"session_id": session_id, "question": question, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return data["answer"], data.get("sources", [])
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        return f"❌ Lỗi: {detail}", []
    except Exception as e:
        return f"❌ Lỗi kết nối: {e}", []


def api_query_stream(session_id: str, question: str):
    """Streaming query via SSE. Yields (type, data) tuples."""
    try:
        with requests.post(
            f"{BACKEND_URL}/query/stream",
            json={"session_id": session_id, "question": question, "stream": True},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    try:
                        data = json.loads(line[6:])
                        yield data
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield {"type": "error", "message": str(e)}


# ─────────────────────────────────────
# UI Components
# ─────────────────────────────────────
def render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    if role == "user":
        st.markdown(
            f'<div class="chat-user">👤 {content}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-assistant">🤖 {content}</div>',
            unsafe_allow_html=True,
        )
        if sources:
            badges = " ".join(
                f'<span class="source-badge">📄 {s["file"]} trang {s["page"]}</span>'
                for s in sources
            )
            st.markdown(f"<div style='margin-top:4px'>{badges}</div>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown('<p class="docmind-header">📚 DocMind</p>', unsafe_allow_html=True)
        st.caption("Trợ lý Tài liệu AI · Tiếng Việt & English")
        st.markdown("---")

        # Status
        health = api_health()
        vllm_ok = health.get("vllm_connected", False)
        if health.get("status") == "ok":
            status_html = '<span class="status-ok">● Backend Online</span>'
        else:
            status_html = '<span class="status-warn">● Backend Offline</span>'

        vllm_html = (
            '<span class="status-ok">● vLLM Connected</span>'
            if vllm_ok
            else '<span class="status-warn">● vLLM Offline</span>'
        )
        st.markdown(f"{status_html}  {vllm_html}", unsafe_allow_html=True)
        st.markdown("---")

        # Session
        if st.session_state.session_id is None:
            if st.button("🚀 Tạo Session Mới", use_container_width=True):
                sid = api_create_session()
                if sid:
                    st.session_state.session_id = sid
                    st.session_state.messages = []
                    st.session_state.indexed_files = []
                    st.rerun()
        else:
            sid_short = st.session_state.session_id[:8]
            st.markdown(f"**Session:** `{sid_short}...`")

            if st.button("🔄 Session Mới", use_container_width=True, type="secondary"):
                sid = api_create_session()
                if sid:
                    st.session_state.session_id = sid
                    st.session_state.messages = []
                    st.session_state.indexed_files = []
                    st.rerun()

        st.markdown("---")

        # Upload
        st.markdown("#### 📁 Tài liệu")
        uploaded = st.file_uploader(
            "Kéo thả hoặc chọn file",
            type=ALLOWED_TYPES,
            accept_multiple_files=True,
            help="Hỗ trợ: PDF, PNG, JPG, BMP, TIFF, WEBP",
        )

        if uploaded and st.session_state.session_id:
            if st.button("⚡ Index Tài liệu", use_container_width=True):
                new_files = [f for f in uploaded if f.name not in st.session_state.indexed_files]
                if not new_files:
                    st.info("Tất cả file đã được index.")
                else:
                    progress = st.progress(0, text="Đang xử lý...")
                    for i, f in enumerate(new_files):
                        progress.progress(
                            int((i / len(new_files)) * 100),
                            text=f"Đang index: {f.name}",
                        )
                        result = api_upload(st.session_state.session_id, f)
                        if result:
                            st.session_state.indexed_files.append(f.name)
                            st.success(
                                f"✅ {f.name} — {result['num_chunks']} chunks"
                            )
                    progress.progress(100, text="Hoàn tất!")
                    time.sleep(0.5)
                    st.rerun()
        elif uploaded and not st.session_state.session_id:
            st.warning("⚠️ Hãy tạo session trước.")

        # Indexed files list
        if st.session_state.indexed_files:
            st.markdown("**Đã index:**")
            for fname in st.session_state.indexed_files:
                ext = fname.split(".")[-1].upper()
                icon = "📄" if ext == "PDF" else "🖼️"
                st.markdown(
                    f'<div class="file-card">{icon} <span style="font-size:0.85rem">{fname}</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        # Tips
        with st.expander("💡 Gợi ý câu hỏi"):
            tips = [
                "Tóm tắt nội dung chính của tài liệu này?",
                "Các điểm quan trọng nhất trong tài liệu là gì?",
                "Giải thích khái niệm X trong tài liệu?",
                "So sánh các phương pháp được đề cập?",
                "Kết luận của tài liệu là gì?",
            ]
            for tip in tips:
                st.markdown(f"• _{tip}_")


def render_chat_area():
    st.markdown("#### 💬 Hỏi đáp Tài liệu")

    if not st.session_state.session_id:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; opacity:0.5">
            <div style="font-size:4rem">📚</div>
            <h3>Chào mừng đến với DocMind</h3>
            <p>Tạo session và upload tài liệu để bắt đầu hỏi đáp</p>
        </div>
        """, unsafe_allow_html=True)
        return

    if not st.session_state.indexed_files:
        st.info("📁 Hãy upload và index tài liệu ở thanh bên trái để bắt đầu.")
        return

    # Render messages
    for msg in st.session_state.messages:
        render_message(msg)

    # Chat input
    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input(
                "Câu hỏi",
                placeholder="Hỏi về nội dung tài liệu... (Tiếng Việt hoặc English)",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Gửi ➤", use_container_width=True)

    if submitted and question.strip():
        # Thêm user message
        st.session_state.messages.append({"role": "user", "content": question})

        # Streaming response
        answer_placeholder = st.empty()
        full_answer = ""
        sources = []

        with st.spinner("🤔 Đang suy nghĩ..."):
            for event in api_query_stream(st.session_state.session_id, question):
                if event["type"] == "sources":
                    sources = event["sources"]
                elif event["type"] == "token":
                    full_answer += event["content"]
                    answer_placeholder.markdown(
                        f'<div class="chat-assistant">🤖 {full_answer}▌</div>',
                        unsafe_allow_html=True,
                    )
                elif event["type"] == "done":
                    answer_placeholder.empty()
                elif event["type"] == "error":
                    full_answer = f"❌ Lỗi: {event['message']}"

        # Lưu assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
        })
        st.rerun()

    # Nếu có messages, hiển thị nút xóa chat
    if st.session_state.messages:
        if st.button("🗑️ Xóa lịch sử chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()


# ─────────────────────────────────────
# Main Layout
# ─────────────────────────────────────
render_sidebar()

# Main content
st.markdown('<h1 class="docmind-header">DocMind — AI Document Assistant</h1>', unsafe_allow_html=True)
st.caption("Phân tích tài liệu thông minh · Hỗ trợ PDF có ảnh, OCR tiếng Việt")
st.markdown("---")

render_chat_area()
