"""Hàm phụ trợ thuần túy (không giữ trạng thái) cho API Gateway.

Gồm validate/định dạng session-id, tên file upload, chuẩn hóa văn bản, khóa
cache, và dựng nội dung email. Các hàm này không tham chiếu singleton toàn cục
của ``main.py`` nên tách ra được an toàn. Giữ nguyên hành vi so với bản gốc.
"""

import json
import re
import shutil
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request

from src.actions.report_intent import is_report_request
from src.api import config
from src.api.schemas import QueryRequest
from src.integrations.gmail_sender import try_parse_email_send_command


def normalize_session_id(session_id: str) -> str:
    """Validate and normalize a public session identifier."""
    try:
        return str(uuid.UUID(session_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid session ID.") from exc


def session_upload_dir(session_id: str) -> Path:
    """Return a filesystem-safe upload directory for a UUID session."""
    return config.UPLOAD_DIR / normalize_session_id(session_id)


def safe_upload_filename(filename: Optional[str]) -> str:
    """Reject path traversal and unsupported public uploads."""
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    safe_name = Path(filename).name
    if safe_name != filename or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if Path(safe_name).suffix.lower() not in config.ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_UPLOAD_EXTENSIONS))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed extensions: {allowed}",
        )
    return safe_name


def research_demo_source_files() -> List[str]:
    """List supported documents prepared for the built-in research demo."""
    if not config.RESEARCH_DEMO_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in config.RESEARCH_DEMO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in config.ALLOWED_UPLOAD_EXTENSIONS
    )


def path_is_inside(path: Path, root: Path) -> bool:
    """Return whether path is equal to or nested inside root after resolving."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def cleanup_failed_upload(file_path: Path, page_dir: Path) -> None:
    """Remove partial upload artifacts after parsing or indexing fails."""
    if file_path.exists():
        file_path.unlink()
    if page_dir.exists():
        shutil.rmtree(page_dir)


def client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else "unknown"


def message_text(message: Any) -> str:
    """Normalize LangChain message content into client-friendly text."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
                return parsed["content"]
        except json.JSONDecodeError:
            pass
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def normalize_query_cache_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value or "").strip().lower()
    return re.sub(r"\s+", " ", normalized)


def query_cache_key(req: QueryRequest, *, snapshot_version: str = "") -> Optional[str]:
    if config.QUERY_RESPONSE_CACHE_SIZE <= 0 or req.mode not in {"mkac", "mes", "wms", "research"}:
        return None
    if try_parse_email_send_command(req.question) is not None:
        return None
    # Mọi yêu cầu tạo report đều bỏ cache: report được hỗ trợ tạo artifact mới,
    # còn report ngoài capability phải đi qua refusal mới thay vì lấy fallback cũ.
    if is_report_request(req.question):
        return None
    # Uploaded Research documents are private to a mutable session. Until the
    # cache has corpus-version invalidation, disabling it is safer than risking
    # stale results or cross-session answers/sources.
    if req.mode == "research" and req.research_scope == "upload":
        return None
    # Câu follow-up ("phòng này", "còn X thì sao"...) có đáp án phụ thuộc
    # conversation_context — cache theo câu chữ sẽ trả nhầm ngữ cảnh người khác.
    if is_followup_question(req.question):
        return None
    # TTL áp dụng theo mode: MES/WMS dùng TTL dài riêng vì snapshot tĩnh.
    # Research dùng TTL riêng vì tài liệu DocJP gần như tĩnh (chỉ đổi khi reindex).
    if req.mode in {"mes", "wms"}:
        ttl = config.MES_QUERY_CACHE_TTL_SECONDS
    elif req.mode == "research":
        ttl = config.RESEARCH_QUERY_CACHE_TTL_SECONDS
    else:
        ttl = config.QUERY_RESPONSE_CACHE_TTL_SECONDS
    if ttl <= 0:
        return None
    employee_key = req.employee_id or ""
    # research_topic phân biệt phạm vi tài liệu; câu hỏi giống nhau nhưng khác
    # topic phải cho ra hai khóa cache khác nhau vì nguồn dữ liệu khác nhau.
    research_scope_key = (req.research_scope or "") if req.mode == "research" else ""
    research_topic_key = (req.research_topic or "") if req.mode == "research" else ""
    # snapshot_version gắn vào khóa cho MES: khi re-import (imported_at đổi),
    # mọi khóa cũ không còn khớp → cache tự vô hiệu, không lo trả dữ liệu cũ.
    return "|".join(
        (
            req.mode,
            req.ui_language,
            req.model,
            employee_key,
            research_scope_key,
            research_topic_key,
            snapshot_version,
            normalize_query_cache_text(req.question),
        )
    )


def normalize_prepared_question(value: str) -> str:
    normalized = normalize_query_cache_text(value)
    normalized = re.sub(r"[?？!！。.,;:]+$", "", normalized).strip()
    return normalized


def _normalize_reference_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"\s+", " ", normalized).strip()


def is_context_reference(value: str) -> bool:
    normalized = _normalize_reference_text(value)
    reference_markers = (
        "thong tin nay",
        "noi dung nay",
        "ket qua nay",
        "cau tra loi nay",
        "bao cao nay",
        "report nay",
        "ban bao cao nay",
        "phan tren",
        "o tren",
        "vua roi",
        "ben tren",
        "this report",
        "このレポート",
        "この報告書",
        "この内容",
        "上記のレポート",
    )
    return any(marker in normalized for marker in reference_markers)


def is_followup_question(value: str) -> bool:
    """Câu hỏi tham chiếu ngữ cảnh hội thoại (deictic/elliptic).

    Trả lời của các câu này phụ thuộc ``conversation_context`` nên không được
    cache theo câu chữ — hai người hỏi cùng câu "liệt kê thành viên phòng này"
    trong hai ngữ cảnh khác nhau phải nhận hai câu trả lời khác nhau.
    """
    if is_context_reference(value):
        return True
    normalized = _normalize_reference_text(value)
    followup_markers = (
        "phong nay",
        "phong do",
        "phong ban nay",
        "phong ban do",
        "bo phan nay",
        "bo phan do",
        "thi sao",
        "cai nay",
        "cai do",
        "cai kia",
        "nguoi nay",
        "nguoi do",
        "anh ay",
        "chi ay",
        "ong ay",
        "ba ay",
        "lot nay",
        "lot do",
        "san pham nay",
        "san pham do",
        "ma hang nay",
        "ma hang do",
    )
    return any(marker in normalized for marker in followup_markers)


def latest_assistant_context(
    conversation_context: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for item in reversed(conversation_context or []):
        if item.get("role") != "assistant":
            continue
        content = str(item.get("content") or "").strip()
        artifact_id = str(item.get("artifact_id") or "").strip()
        if content or artifact_id:
            return {
                "content": content,
                "answer_scope": str(item.get("answer_scope") or "conversation_context"),
                "model": str(item.get("model") or ""),
                "artifact_id": artifact_id,
                "artifact_type": str(item.get("artifact_type") or ""),
                "artifact_title": str(item.get("artifact_title") or ""),
            }
    return None


def build_email_body(
    *,
    original_question: str,
    data_question: str,
    answer: str,
    answer_scope: str,
) -> str:
    return (
        "Xin chào,\n\n"
        "Meibook gửi bạn thông tin theo yêu cầu:\n\n"
        f"{answer.strip()}\n\n"
        "---\n"
        f"Yêu cầu gốc: {original_question.strip()}\n"
        f"Câu hỏi dữ liệu: {data_question.strip()}\n"
        f"Nguồn trả lời: {answer_scope}\n"
    )


def build_direct_email_body(
    *,
    original_question: str,
    body: str,
) -> str:
    return (
        "Xin chào,\n\n"
        f"{body.strip()}\n\n"
        "---\n"
        "Email được gửi từ Meibook theo yêu cầu trực tiếp của người dùng.\n"
        f"Yêu cầu gốc: {original_question.strip()}\n"
    )
