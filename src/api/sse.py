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
        "mes": "Đang truy vấn dữ liệu MES...",
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
        "mes": "MESデータを照会しています...",
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


def query_processing_status_key(req: QueryRequest) -> str:
    if req.mode == "mes":
        return "mes"
    if req.mode == "mkac":
        return "rag"
    return "rag"
