"""
Meiko Automation - DocMind QA Testbench (Streamlit)
Webapp kiểm thử hệ thống RAG đa phương thức (text + image).
"""

import json
import os
import time
from typing import Any

import requests
import streamlit as st

BACKEND_URL = os.getenv("DOCMIND_BACKEND_URL", "http://localhost:8001").rstrip("/")
ALLOWED_TYPES = ["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp"]

st.set_page_config(
    page_title="Meiko Automation · DocMind QA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at top right, #132f4c 0%, #0b1220 55%, #080d18 100%);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.86);
    border-right: 1px solid rgba(148, 163, 184, 0.25);
}

.brand-title {
    font-size: 1.85rem;
    font-weight: 700;
    background: linear-gradient(90deg, #22d3ee, #38bdf8, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.subtitle {
    color: #cbd5e1;
    margin-bottom: 14px;
}

.status-chip {
    display: inline-block;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.76rem;
    font-weight: 600;
    margin: 2px 4px 2px 0;
}

.ok { background: rgba(34,197,94,.16); color: #86efac; border: 1px solid rgba(34,197,94,.35); }
.warn { background: rgba(245,158,11,.16); color: #fcd34d; border: 1px solid rgba(245,158,11,.35); }

.card {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 8px;
}

.metric {
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.24);
    background: rgba(30, 41, 59, 0.58);
    padding: 12px;
    text-align: center;
}

.metric .label { color: #94a3b8; font-size: 0.82rem; }
.metric .value { color: #e2e8f0; font-size: 1.1rem; font-weight: 700; margin-top: 2px; }

.stButton > button {
    border-radius: 10px !important;
    border: 1px solid rgba(56, 189, 248, 0.36) !important;
}

.stTextInput > div > div > input {
    border-radius: 10px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "session_id": None,
        "messages": [],
        "indexed_files": [],
        "last_latency": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def api_health() -> dict[str, Any]:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=4)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {"status": "error", "vllm_connected": False, "sessions": 0}


def api_create_session() -> str | None:
    try:
        response = requests.post(f"{BACKEND_URL}/sessions", timeout=8)
        response.raise_for_status()
        return response.json().get("session_id")
    except Exception as exc:
        st.error(f"Không thể tạo session: {exc}")
        return None


def api_upload(session_id: str, file_obj) -> dict[str, Any] | None:
    try:
        response = requests.post(
            f"{BACKEND_URL}/sessions/{session_id}/upload",
            files={"file": (file_obj.name, file_obj.getvalue(), file_obj.type)},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
        st.error(f"Upload lỗi: {detail}")
        return None
    except Exception as exc:
        st.error(f"Upload lỗi: {exc}")
        return None


def api_query_stream(session_id: str, question: str):
    try:
        with requests.post(
            f"{BACKEND_URL}/query/stream",
            json={"session_id": session_id, "question": question, "stream": True},
            stream=True,
            timeout=240,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    try:
                        yield json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="brand-title">Meiko Automation</div>', unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>DocMind QA Testbench · Multimodal RAG</div>", unsafe_allow_html=True)

        health = api_health()
        backend_chip = (
            '<span class="status-chip ok">Backend Online</span>'
            if health.get("status") == "ok"
            else '<span class="status-chip warn">Backend Offline</span>'
        )
        vllm_chip = (
            '<span class="status-chip ok">vLLM Connected</span>'
            if health.get("vllm_connected")
            else '<span class="status-chip warn">vLLM Offline</span>'
        )
        st.markdown(f"{backend_chip}{vllm_chip}", unsafe_allow_html=True)

        st.markdown("---")
        if st.session_state.session_id is None:
            if st.button("Tạo session mới", use_container_width=True):
                session_id = api_create_session()
                if session_id:
                    st.session_state.session_id = session_id
                    st.session_state.messages = []
                    st.session_state.indexed_files = []
                    st.rerun()
        else:
            st.caption(f"Session: {st.session_state.session_id}")
            if st.button("Đổi session", use_container_width=True):
                session_id = api_create_session()
                if session_id:
                    st.session_state.session_id = session_id
                    st.session_state.messages = []
                    st.session_state.indexed_files = []
                    st.rerun()

        st.markdown("---")
        st.markdown("### Upload tài liệu test")
        uploads = st.file_uploader(
            "Hỗ trợ PDF + ảnh",
            type=ALLOWED_TYPES,
            accept_multiple_files=True,
            help="Dùng để test khả năng OCR + RAG của hệ thống.",
        )

        if uploads and st.session_state.session_id and st.button("Index tài liệu", use_container_width=True):
            new_uploads = [f for f in uploads if f.name not in st.session_state.indexed_files]
            if not new_uploads:
                st.info("Các file này đã được index trước đó.")
            else:
                progress = st.progress(0, text="Đang index...")
                for idx, file_obj in enumerate(new_uploads):
                    progress.progress(int((idx / len(new_uploads)) * 100), text=f"Xử lý: {file_obj.name}")
                    result = api_upload(st.session_state.session_id, file_obj)
                    if result:
                        st.session_state.indexed_files.append(file_obj.name)
                progress.progress(100, text="Hoàn tất index")
                time.sleep(0.25)
                st.rerun()
        elif uploads and not st.session_state.session_id:
            st.warning("Cần tạo session trước khi upload.")

        if uploads:
            image_uploads = [f for f in uploads if f.type and f.type.startswith("image/")][:2]
            for image_file in image_uploads:
                st.image(image_file, caption=f"Preview: {image_file.name}", use_column_width=True)

        if st.session_state.indexed_files:
            st.markdown("---")
            st.markdown("### File đã index")
            for file_name in st.session_state.indexed_files:
                icon = "🖼️" if file_name.lower().endswith(("png", "jpg", "jpeg", "bmp", "tiff", "webp")) else "📄"
                st.markdown(f"- {icon} {file_name}")


def render_metrics() -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div class='metric'><div class='label'>Session Active</div><div class='value'>{'Yes' if st.session_state.session_id else 'No'}</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='metric'><div class='label'>Indexed Files</div><div class='value'>{len(st.session_state.indexed_files)}</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        latency_text = f"{st.session_state.last_latency:.2f}s" if st.session_state.last_latency else "-"
        st.markdown(
            f"<div class='metric'><div class='label'>Last Response</div><div class='value'>{latency_text}</div></div>",
            unsafe_allow_html=True,
        )


def _format_source_item(source: Any) -> str:
    if isinstance(source, dict):
        return f"{source.get('file', '-')}:trang {source.get('page', '-')}"
    return str(source)


def render_chat() -> None:
    st.markdown("### Chat kiểm thử")
    st.markdown(
        "<div class='card'>Gợi ý test: <b>“Trong ảnh này có thông tin gì?”</b>, <b>“Tóm tắt nội dung tài liệu theo 3 ý chính”</b>, <b>“Trích dẫn nguồn liên quan tới ...”</b></div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.session_id:
        st.info("Tạo session ở sidebar để bắt đầu.")
        return
    if not st.session_state.indexed_files:
        st.info("Upload và index file để hệ thống có ngữ cảnh trả lời.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                source_text = " | ".join(
                    f"{source.get('file', '-')}:trang {source.get('page', '-')}"
                    for source in message["sources"]
                )
                st.caption(f"Nguồn: {source_text}")

    question = st.chat_input("Nhập câu hỏi để test hệ thống...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    full_answer = ""
    sources = []
    start_time = time.time()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        for event in api_query_stream(st.session_state.session_id, question):
            event_type = event.get("type")
            if event_type == "sources":
                sources = event.get("sources", [])
            elif event_type == "token":
                full_answer += event.get("content", "")
                placeholder.markdown(full_answer + "▌")
            elif event_type == "done":
                placeholder.markdown(full_answer)
            elif event_type == "error":
                full_answer = f"Lỗi truy vấn: {event.get('message', 'Unknown error')}"
                placeholder.error(full_answer)

        if sources:
            source_text = " | ".join(
                _format_source_item(source)
                for source in sources
            )
            st.caption(f"Nguồn: {source_text}")

    st.session_state.last_latency = time.time() - start_time
    st.session_state.messages.append(
        {"role": "assistant", "content": full_answer, "sources": sources}
    )
    st.rerun()


def main() -> None:
    init_state()
    render_sidebar()

    st.markdown('<div class="brand-title">DocMind Multimodal QA Demo</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Môi trường kiểm thử hệ thống hỏi đáp tài liệu cho Meiko Automation</div>",
        unsafe_allow_html=True,
    )
    render_metrics()
    st.markdown("---")
    render_chat()


if __name__ == "__main__":
    main()
