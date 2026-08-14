"""Tiện ích Server-Sent Events (SSE) và thông điệp trạng thái VI/JA.

Tách khỏi ``main.py`` để phần định dạng sự kiện stream và các câu trạng thái
hiển thị cho người dùng nằm gọn một chỗ. Giữ nguyên hành vi so với bản gốc.
"""

import json
from typing import Any, Dict

from src.api.schemas import QueryRequest

QUERY_STREAM_STATUS_TEXT = {
    "vi": {
        "received": "Tôi đã hiểu rồi, bạn chờ chút nhé...",
        "cache": "Đang kiểm tra kết quả đã xử lý trước đó...",
        "routing": "Đang xác định loại câu hỏi...",
        "quick_answer": "Đang kiểm tra câu trả lời đã chuẩn bị...",
        "hr_directory": "Đang kiểm tra danh bạ nhân sự...",
        "email": "Đang chuẩn bị nội dung email...",
        "calendar": "Đang kiểm tra lịch và phòng họp...",
        "mes": "Đang truy vấn dữ liệu MES...",
        "wms": "Đang chuẩn bị kiểm chứng dữ liệu WMS...",
        "report": "Đang tổng hợp báo cáo điều hành...",
        "rag": "Đang tìm nguồn tài liệu phù hợp...",
        "research_cache": "Đang phân tích tài liệu liên quan...",
        "translation": "Đang chuyển đổi ngôn ngữ...",
        "finalizing": "Đang tổng hợp câu trả lời...",
        "almost_done": "Sắp ra rồi...",
    },
    "ja": {
        "received": "承知しました。少々お待ちください...",
        "cache": "以前の処理結果を確認しています...",
        "routing": "質問の種類を判定しています...",
        "quick_answer": "準備済みの回答を確認しています...",
        "hr_directory": "人事名簿を確認しています...",
        "email": "メール内容を準備しています...",
        "calendar": "カレンダーと会議室を確認しています...",
        "mes": "MESデータを照会しています...",
        "wms": "WMSデータの検証を準備しています...",
        "report": "エグゼクティブレポートを集計しています...",
        "rag": "関連資料を検索しています...",
        "research_cache": "関連資料を分析しています...",
        "translation": "言語を変換しています...",
        "finalizing": "回答をまとめています...",
        "almost_done": "もうすぐ結果が出ます...",
    },
}


def query_status_text(req: QueryRequest, key: str) -> str:
    language_text = QUERY_STREAM_STATUS_TEXT.get(
        req.ui_language,
        QUERY_STREAM_STATUS_TEXT["vi"],
    )
    return language_text.get(key, QUERY_STREAM_STATUS_TEXT["vi"].get(key, key))


def sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_status(req: QueryRequest, key: str) -> str:
    return sse_event({"type": "status", "message": query_status_text(req, key)})


def sse_agent_plan(
    *,
    title: str,
    steps: list[dict[str, str]],
    workflow: str,
) -> str:
    """Format a safe, high-level deterministic workflow plan."""
    return sse_event(
        {
            "type": "agent_plan",
            "title": title,
            "steps": steps,
            "workflow": workflow,
        }
    )


def sse_tool_start(*, step_id: str, tool: str, title: str) -> str:
    """Format a public milestone that has just begun."""
    return sse_event(
        {
            "type": "tool_start",
            "step_id": step_id,
            "tool": tool,
            "title": title,
        }
    )


def sse_tool_result(*, step_id: str, status: str, summary: str) -> str:
    """Format the allowlisted outcome of a deterministic milestone."""
    return sse_event(
        {
            "type": "tool_result",
            "step_id": step_id,
            "status": status,
            "summary": summary,
        }
    )


def query_processing_status_key(req: QueryRequest) -> str:
    if req.mode == "mes":
        return "mes"
    if req.mode == "wms":
        return "wms"
    if req.mode == "mkac":
        return "rag"
    return "rag"
