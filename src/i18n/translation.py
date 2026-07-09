"""LLM-backed translation layer between the UI and the Vietnamese core."""

from __future__ import annotations

import os
import re
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
    JAPANESE_SCRIPT_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
    LOCAL_MODEL_ALIASES = {
        "auto-model",
        "local-qwen-chat",
        "local-qwen-small",
        "local-gemma",
    }
    DEFAULT_MODEL = "local-qwen-small"
    GLOSSARY = (
        "- Meibook: Meibook\n"
        "- MKAC / Meiko Automation: Meiko Automation\n"
        "- MES: MES\n"
        "- MES snapshot: MESスナップショット\n"
        "- Lot / mã Lot: Lot\n"
        "- mã hàng / mã sản phẩm: 品番\n"
        "- mã lỗi: エラーコード\n"
        "- tên lỗi: エラー名\n"
        "- tổng số lỗi: 総エラー数\n"
        "- số bản ghi lỗi: エラー記録数\n"
        "- lỗi / NG / defect / error: エラー\n"
        "- phòng ban: 部署\n"
        "- nhân viên / thành viên: 社員\n"
        "- mã nhân viên: 社員番号\n"
        "- làm thêm giờ: 残業\n"
        "- nghỉ phép: 休暇\n"
        "- quy định / nội quy: 規定\n"
    )
    STATIC_VI_TO_JA = {
        "Không có dữ liệu.": "データがありません。",
        "Không thể dịch nội dung cho giao diện.": "画面表示用の翻訳を完了できませんでした。",
        "Chưa tạo được câu trả lời ổn định từ model local. Vui lòng thử lại với câu hỏi ngắn gọn hơn.": (
            "ローカルモデルから安定した回答を生成できませんでした。質問を短くしてもう一度お試しください。"
        ),
        "Không thể hoàn tất yêu cầu: Gmail OAuth token đã hết hạn hoặc bị Google thu hồi. Vui lòng tạo lại token Gmail OAuth.": (
            "リクエストを完了できませんでした。Gmail OAuthトークンの有効期限が切れたか、Googleにより無効化されています。Gmail OAuthトークンを再作成してください。"
        ),
    }

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.model = model or os.getenv("TRANSLATION_MODEL", self.DEFAULT_MODEL)
        self.num_ctx = int(os.getenv("LOCAL_CHAT_NUM_CTX", "16384"))
        self.aux_num_ctx = int(os.getenv("LOCAL_AUX_NUM_CTX", "4096"))
        self.temperature = float(os.getenv("TRANSLATION_TEMPERATURE", "0.1"))
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

    @classmethod
    def should_translate_query(cls, question: str, ui_language: str | None) -> bool:
        language = cls.normalize_language(ui_language)
        clean_question = (question or "").strip()
        return (
            language == "ja"
            and bool(clean_question)
            and bool(cls.JAPANESE_SCRIPT_PATTERN.search(clean_question))
        )

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
        if not self.should_translate_query(clean_question, language):
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
                "Glossary thuật ngữ bắt buộc:\n"
                f"{self.GLOSSARY}\n"
                f"Câu hỏi: {clean_question}"
            ),
            max_tokens=700,
        )
        return TranslatedQuery(
            clean_question,
            self._clean_completion(translated) or clean_question,
            language,
        )

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
        static_translation = self._translate_static_vi_to_ja(clean_answer)
        if static_translation is not None:
            return static_translation

        translated = await self._complete(
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
                "Glossary thuật ngữ bắt buộc:\n"
                f"{self.GLOSSARY}\n"
                f"Câu trả lời tiếng Việt:\n{clean_answer}"
            ),
            max_tokens=1800 if mode == "research" else 1000,
        )
        return self._clean_completion(translated) or clean_answer

    async def translate_ui_text(
        self,
        text: str,
        *,
        ui_language: str | None,
        mode: str,
        purpose: str = "ui",
    ) -> str:
        """Translate short dynamic UI fragments such as source snippets."""
        language = self.normalize_language(ui_language)
        clean_text = text or ""
        if not self.enabled or language == "vi" or not clean_text.strip():
            return clean_text
        static_translation = self._translate_static_vi_to_ja(clean_text)
        if static_translation is not None:
            return static_translation

        translated = await self._complete(
            system=(
                "Bạn là lớp dịch các đoạn text ngắn cho giao diện Meibook tiếng Nhật. "
                "Hãy dịch từ tiếng Việt sang tiếng Nhật tự nhiên, giữ đúng ý và không "
                "thêm nhận xét."
            ),
            user=(
                f"Chế độ hệ thống: {mode}\n"
                f"Loại nội dung: {purpose}\n"
                "Dịch đoạn sau sang tiếng Nhật.\n\n"
                "Quy tắc bắt buộc:\n"
                "- Giữ nguyên mã Lot, mã hàng, mã lỗi, mã nhân viên, email, URL, "
                "tên file, số trang, số liệu và ký hiệu kỹ thuật.\n"
                "- Nếu là đoạn trích tài liệu, chỉ dịch nội dung đoạn trích; không "
                "thêm tiêu đề hoặc giải thích.\n"
                "- Chỉ trả về bản dịch.\n\n"
                "Glossary thuật ngữ bắt buộc:\n"
                f"{self.GLOSSARY}\n"
                f"Đoạn text:\n{clean_text}"
            ),
            max_tokens=700,
        )
        return self._clean_completion(translated) or clean_text

    async def _complete(self, *, system: str, user: str, max_tokens: int) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        provider_options = self._provider_options()
        if provider_options:
            kwargs.update(provider_options)
        try:
            response = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # pragma: no cover - provider-specific failures
            raise TranslationError("Không thể dịch nội dung cho giao diện.") from exc

        content = response.choices[0].message.content or ""
        return self._clean_completion(content)

    def _provider_options(self) -> dict[str, Any]:
        if self.model in self.LOCAL_MODEL_ALIASES:
            num_ctx = self.aux_num_ctx if self.model == "local-qwen-small" else self.num_ctx
            return {"extra_body": {"think": False, "num_ctx": num_ctx}}
        return {}

    @classmethod
    def _translate_static_vi_to_ja(cls, text: str) -> str | None:
        normalized = (text or "").strip()
        return cls.STATIC_VI_TO_JA.get(normalized)

    @staticmethod
    def _clean_completion(text: str) -> str:
        clean_text = (text or "").strip()
        clean_text = re.sub(r"(?is)<think>.*?</think>", "", clean_text).strip()
        if "</think>" in clean_text:
            clean_text = clean_text.split("</think>", 1)[1].strip()
        clean_text = re.sub(r"(?is)^```(?:\w+)?\s*|\s*```$", "", clean_text).strip()
        clean_text = re.sub(
            r"(?is)^(?:đây là|dưới đây là|kết quả|bản dịch|câu đã dịch|translated query|translation)"
            r"[^:\n]{0,160}:\s*",
            "",
            clean_text,
        ).strip()
        quoted = re.match(r'(?is)^(?:đây là|dưới đây là)[^"\n]{0,180}"(.+)"\s*$', clean_text)
        if quoted:
            clean_text = quoted.group(1).strip()
        return clean_text.strip(' "\'')
