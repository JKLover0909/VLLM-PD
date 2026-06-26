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
    assert "mã Lot" in call["messages"][1]["content"]
    assert "一番不良が多いロットは？" in call["messages"][1]["content"]


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
