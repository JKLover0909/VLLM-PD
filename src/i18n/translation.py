"""LLM-backed translation layer between the UI and the Vietnamese core."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from openai import AsyncOpenAI


UiLanguage = Literal["vi", "ja"]


class TranslationError(RuntimeError):
    """Raised when the UI translation layer cannot complete a request."""


@dataclass(frozen=True)
class TranslatedQuery:
    original_question: str
    backend_question: str
    ui_language: UiLanguage

    @property
    def translated(self) -> bool:
        return self.ui_language == "ja" and self.backend_question != self.original_question


class TranslationService:
    """Translate Japanese UI input/output while keeping the core Vietnamese."""

    SUPPORTED_LANGUAGES = {"vi", "ja"}

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.model = model or os.getenv("TRANSLATION_MODEL", "openai-model")
        self.client = client or AsyncOpenAI(
            api_key=os.getenv("LITELLM_MASTER_KEY", "sk-local"),
            base_url=os.getenv("LITELLM_URL", "http://localhost:4000/v1"),
        )

    @classmethod
    def from_env(cls) -> "TranslationService | None":
        enabled = os.getenv("TRANSLATION_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return None
        return cls(enabled=enabled)

    @classmethod
    def normalize_language(cls, value: str | None) -> UiLanguage:
        language = (value or "vi").strip().lower()
        return "ja" if language == "ja" else "vi"

    async def translate_query(
        self,
        question: str,
        *,
        ui_language: str | None,
        mode: str,
    ) -> TranslatedQuery:
        language = self.normalize_language(ui_language)
        clean_question = (question or "").strip()
        if not self.enabled or language == "vi" or not clean_question:
            return TranslatedQuery(clean_question, clean_question, language)

        translated = await self._complete(
            system=(
                "Bạn là lớp dịch truy vấn cho hệ thống Meibook. "
                "Lõi backend hiểu tốt nhất tiếng Việt, vì vậy hãy dịch câu hỏi "
                "từ tiếng Nhật sang tiếng Việt tự nhiên, ngắn gọn và rõ intent. "
                "Nếu đầu vào đã là tiếng Việt hoặc là mã kỹ thuật, giữ nguyên phần đó."
            ),
            user=(
                f"Chế độ hệ thống: {mode}\n"
                "Nhiệm vụ: dịch câu hỏi sau sang tiếng Việt cho backend xử lý.\n\n"
                "Quy tắc bắt buộc:\n"
                "- Giữ nguyên mã Lot, mã hàng, mã lỗi, mã nhân viên, email, URL, "
                "tên file, số liệu và ký hiệu kỹ thuật.\n"
                "- Không thêm giải thích, không markdown, chỉ trả về câu đã dịch.\n"
                "- Với MES, dịch ロット/ロット番号 thành Lot, 品番/製品コード thành mã hàng, "
                "不良/エラー thành lỗi.\n"
                "- Với yêu cầu gửi email, dịch thành câu lệnh tiếng Việt bắt đầu tự nhiên "
                "bằng 'Gửi email...' hoặc 'Gửi thông tin này...'.\n\n"
                f"Câu hỏi: {clean_question}"
            ),
            max_tokens=700,
        )
        return TranslatedQuery(clean_question, translated.strip() or clean_question, language)

    async def translate_answer(
        self,
        answer: str,
        *,
        ui_language: str | None,
        mode: str,
    ) -> str:
        language = self.normalize_language(ui_language)
        clean_answer = answer or ""
        if not self.enabled or language == "vi" or not clean_answer.strip():
            return clean_answer

        return await self._complete(
            system=(
                "Bạn là lớp dịch kết quả cho giao diện Meibook tiếng Nhật. "
                "Hãy dịch từ tiếng Việt sang tiếng Nhật tự nhiên cho người dùng doanh nghiệp. "
                "Không thêm, không bớt thông tin."
            ),
            user=(
                f"Chế độ hệ thống: {mode}\n"
                "Dịch câu trả lời sau sang tiếng Nhật.\n\n"
                "Quy tắc bắt buộc:\n"
                "- Giữ nguyên markdown, xuống dòng, danh sách bullet/numbered list.\n"
                "- Giữ nguyên mã Lot, mã hàng, mã lỗi, mã nhân viên, email, URL, "
                "tên file, số trang, số liệu và ký hiệu kỹ thuật.\n"
                "- Dịch các cụm mô tả tiếng Việt thông thường sang tiếng Nhật, "
                "bao gồm tên lỗi sau mã lỗi. Ví dụ giữ 'LA001' nhưng dịch "
                "'Trầy xước' sang tiếng Nhật.\n"
                "- Không dịch tên field kỹ thuật nếu có trong mã, nhưng tránh thêm tên field mới.\n"
                "- MES snapshot dịch là MESスナップショット.\n"
                "- Tuyệt đối không thêm câu giải thích mới về JSON, filters, "
                "chính sách hiển thị hoặc hệ thống nếu câu gốc không có.\n"
                "- Chỉ trả về bản dịch, không giải thích.\n\n"
                f"Câu trả lời tiếng Việt:\n{clean_answer}"
            ),
            max_tokens=1800 if mode == "research" else 1000,
        )

    async def _complete(self, *, system: str, user: str, max_tokens: int) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover - provider-specific failures
            raise TranslationError("Không thể dịch nội dung cho giao diện.") from exc

        content = response.choices[0].message.content or ""
        return content.strip()
