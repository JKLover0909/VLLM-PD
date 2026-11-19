import asyncio
from types import SimpleNamespace

from src.i18n.translation import TranslationService


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0)
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeChat:
    def __init__(self, replies):
        self.completions = FakeCompletions(replies)


class FakeClient:
    def __init__(self, replies):
        self.chat = FakeChat(replies)


def test_vietnamese_query_bypasses_translation_model():
    client = FakeClient(["unused"])
    service = TranslationService(client=client, model="translation-model")

    result = asyncio.run(
        service.translate_query(
            "Lot nào có số lượng lỗi nhiều nhất?",
            ui_language="vi",
            mode="mes",
        )
    )

    assert result.backend_question == "Lot nào có số lượng lỗi nhiều nhất?"
    assert client.chat.completions.calls == []


def test_translation_defaults_to_local_chat_alias():
    client = FakeClient(["unused"])
    service = TranslationService(client=client)

    assert service.model == "local-qwen-chat"


def test_japanese_query_is_translated_for_vietnamese_backend():
    client = FakeClient(["Lot nào có số lượng lỗi nhiều nhất?"])
    service = TranslationService(client=client, model="translation-model")

    result = asyncio.run(
        service.translate_query(
            "一番不良が多いロットは？",
            ui_language="ja",
            mode="mes",
        )
    )

    assert result.backend_question == "Lot nào có số lượng lỗi nhiều nhất?"
    call = client.chat.completions.calls[0]
    assert call["model"] == "translation-model"
    assert "mã Lot" in call["messages"][0]["content"]
    assert call["messages"][1]["content"] == "一番不良が多いロットは？"


def test_translation_client_leaves_upstream_options_to_litellm():
    client = FakeClient(["Lot nào có số lượng lỗi nhiều nhất?"])
    service = TranslationService(client=client, model="local-qwen-chat")

    result = asyncio.run(
        service.translate_query(
            "一番不良が多いロットは？",
            ui_language="ja",
            mode="mes",
        )
    )

    assert result.backend_question == "Lot nào có số lượng lỗi nhiều nhất?"
    call = client.chat.completions.calls[0]
    assert call["model"] == "local-qwen-chat"
    assert "extra_body" not in call


def test_japanese_ui_vietnamese_query_bypasses_translation_model():
    client = FakeClient(["unused"])
    service = TranslationService(client=client, model="translation-model")

    result = asyncio.run(
        service.translate_query(
            "Gửi thông tin này cho email 12wuu115@gmail.com",
            ui_language="ja",
            mode="mes",
        )
    )

    assert result.backend_question == "Gửi thông tin này cho email 12wuu115@gmail.com"
    assert client.chat.completions.calls == []


def test_japanese_answer_translation_preserves_backend_content_request():
    client = FakeClient(["MESスナップショットによると、Lot 000866-05-000です。"])
    service = TranslationService(client=client, model="translation-model")

    answer = asyncio.run(
        service.translate_answer(
            "Theo MES snapshot, Lot 000866-05-000 có 12.870 lỗi.",
            ui_language="ja",
            mode="mes",
        )
    )

    assert "Lot 000866-05-000" in answer
    call = client.chat.completions.calls[0]
    assert "Giữ nguyên markdown" in call["messages"][1]["content"]
    assert "dịch 'Trầy xước'" in call["messages"][1]["content"]
    assert "không thêm câu giải thích mới" in call["messages"][1]["content"]
    assert "12.870" in call["messages"][1]["content"]


def test_static_japanese_answer_translation_bypasses_model():
    client = FakeClient(["unused"])
    service = TranslationService(client=client, model="auto-model")

    answer = asyncio.run(
        service.translate_answer(
            "Không có dữ liệu.",
            ui_language="ja",
            mode="mes",
        )
    )

    assert answer == "データがありません。"
    assert client.chat.completions.calls == []


ECHOED_QUERY_PROMPT = (
    "Chế độ hệ thống: mkac\n"
    "Nhiệm vụ: dịch câu hỏi sau sang tiếng Việt cho backend xử lý.\n\n"
    "Quy tắc bắt buộc:\n"
    "- Với yêu cầu gửi email, dịch thành câu lệnh tiếng Việt bắt đầu tự nhiên "
    "bằng 'Gửi email...' hoặc 'Gửi thông tin này...'.\n\n"
    "Glossary thuật ngữ bắt buộc:\n"
    "- MKAC / Meiko Automation: Meiko Automation\n"
    "Câu hỏi: MKACの残業申請は誰の承認が必要ですか？"
)


def test_echoed_query_prompt_is_retried_once():
    client = FakeClient([ECHOED_QUERY_PROMPT, "Đăng ký làm thêm giờ ở MKAC cần ai phê duyệt?"])
    service = TranslationService(client=client, model="translation-model")

    result = asyncio.run(
        service.translate_query("MKACの残業申請は誰の承認が必要ですか？", ui_language="ja", mode="mkac")
    )

    assert result.backend_question == "Đăng ký làm thêm giờ ở MKAC cần ai phê duyệt?"
    assert len(client.chat.completions.calls) == 2


def test_persistently_echoed_query_prompt_falls_back_to_original_question():
    client = FakeClient([ECHOED_QUERY_PROMPT, ECHOED_QUERY_PROMPT])
    service = TranslationService(client=client, model="translation-model")

    result = asyncio.run(
        service.translate_query("MKACの残業申請は誰の承認が必要ですか？", ui_language="ja", mode="mkac")
    )

    # The prompt text (which mentions "Gửi email") must never reach the backend.
    assert result.backend_question == "MKACの残業申請は誰の承認が必要ですか？"
    assert "email" not in result.backend_question.lower()


def test_echoed_answer_prompt_keeps_the_vietnamese_answer():
    echoed = (
        "Chế độ hệ thống: mkac\nDịch câu trả lời sau sang tiếng Nhật.\n\n"
        "Quy tắc bắt buộc:\n- Chỉ trả về bản dịch, không giải thích.\n\n"
        "Glossary thuật ngữ bắt buộc:\n- MES: MES\nCâu trả lời tiếng Việt:\nCó 3 phòng ban."
    )
    client = FakeClient([echoed, echoed])
    service = TranslationService(client=client, model="translation-model")

    answer = asyncio.run(service.translate_answer("Có 3 phòng ban.", ui_language="ja", mode="mkac"))

    assert answer == "Có 3 phòng ban."


def test_translation_mentioning_one_rule_word_is_not_treated_as_echo():
    client = FakeClient(["Quy tắc bắt buộc về làm thêm giờ là gì?"])
    service = TranslationService(client=client, model="translation-model")

    result = asyncio.run(service.translate_query("残業の必須ルールは？", ui_language="ja", mode="mkac"))

    assert result.backend_question == "Quy tắc bắt buộc về làm thêm giờ là gì?"
    assert len(client.chat.completions.calls) == 1


def test_leaked_prompt_header_prefix_is_stripped():
    client = FakeClient([
        "Chế độ hệ thống: mkac. Nhiệm vụ của bạn là xác nhận ai cần phê duyệt cho đơn làm thêm giờ?"
    ])
    service = TranslationService(client=client, model="translation-model")

    result = asyncio.run(
        service.translate_query("MKACの残業申請は誰の承認が必要ですか？", ui_language="ja", mode="mkac")
    )

    assert result.backend_question == "xác nhận ai cần phê duyệt cho đơn làm thêm giờ?"


def test_translation_cannot_introduce_email_context():
    question = "MKACの残業申請は誰の承認が必要ですか？"
    service = TranslationService(client=FakeClient(["Gửi email xin phê duyệt làm thêm giờ."]))
    result = asyncio.run(service.translate_query(question, ui_language="ja", mode="mkac"))
    assert result.backend_question == question


def test_real_email_translation_is_preserved_without_executing_action():
    question = "レポートを demo@example.invalid にメールで送信してください"
    translated = "Gửi email báo cáo đến demo@example.invalid."
    service = TranslationService(client=FakeClient([translated]))
    result = asyncio.run(service.translate_query(question, ui_language="ja", mode="mkac"))
    assert result.backend_question == translated


def test_leaked_question_label_is_stripped():
    client = FakeClient(["Câu hỏi: Lot nào có số lượng lỗi nhiều nhất?"])
    service = TranslationService(client=client, model="translation-model")

    result = asyncio.run(service.translate_query("一番不良が多いロットは？", ui_language="ja", mode="mes"))

    assert result.backend_question == "Lot nào có số lượng lỗi nhiều nhất?"
