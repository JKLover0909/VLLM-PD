"""Routing contract tests for CCTVAI host-hardware questions.

Mirrors test_cctvai_api.py: fake objects only, no HTTP, no psutil, no LLM.

The point of these tests is the *independence* claim. Hardware metrics come
from a separate host service, so they must stay answerable when the RAG
pipeline, the MES service and the CCTVAI Postgres replica are all down — and
they must not borrow anything from those services (no format_sources call, no
replica metadata).
"""

import asyncio
import os

import pytest

os.environ["ENABLE_AGENT"] = "false"

from fastapi import HTTPException

from src.api import main
from src.api.schemas import QueryRequest
from src.integrations.mes_query_service import MesQueryOutcome, MesQueryStreamOutcome


HARDWARE_ANSWER = "CPU 12%, RAM 31/124 GB, GPU 0 ở 41°C, ổ / còn 1.2 TB."


class ExplodingRagPipeline:
    """Any use of the RAG pipeline during a hardware answer is a bug."""

    @staticmethod
    def format_sources(results, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("format_sources must not be called for hardware answers")


class FakeHardwareService:
    def __init__(self, *, available=True):
        self.calls = []
        self._available = available
        self.closed = False

    def status(self):
        return {"enabled": True, "available": self._available}

    async def refresh_health(self):
        return None

    async def close(self):
        self.closed = True

    async def query_hardware_outcome(self, *, question, model, language):
        self.calls.append(("outcome", question, model, language))
        return MesQueryOutcome(
            answer=HARDWARE_ANSWER,
            results=[],
            routed_model="local-hardware-summary",
            answer_scope="cctvai_hardware",
        )

    async def query_hardware_stream_outcome(self, *, question, model, language):
        self.calls.append(("stream", question, model, language))

        async def tokens():
            yield "token", HARDWARE_ANSWER

        return MesQueryStreamOutcome(
            token_stream=tokens(),
            results=[],
            routed_model="local-hardware-summary",
            answer_scope="cctvai_hardware",
        )


def hardware_request(**updates):
    values = {
        "session_id": "00000000-0000-4000-8000-000000000998",
        "question": "Server CCTV AI đang dùng bao nhiêu CPU và RAM?",
        "mode": "cctvai",
        "model": "local",
        "ui_language": "vi",
        "employee_id": "000000",
    }
    values.update(updates)
    return QueryRequest(**values)


def configure_hardware_only(monkeypatch, *, available=True):
    """Everything except the hardware service is unavailable."""
    service = FakeHardwareService(available=available)
    monkeypatch.setattr(main, "cctvai_hardware_service", service)
    # Deliberately None: proves hardware does not depend on any of them.
    monkeypatch.setattr(main, "rag_pipeline", None)
    monkeypatch.setattr(main, "mes_query_service", None)
    monkeypatch.setattr(main, "cctvai_query_service", None)
    return service


# ---------------------------------------------------------------------------
# classify_hardware_question — pure, bilingual, narrow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question",
    [
        "Server CCTV AI đang dùng bao nhiêu CPU và RAM?",
        "Tình trạng ổ đĩa của server CCTV AI thế nào?",
        "GPU của server còn trống bao nhiêu VRAM?",
        "CCTVAIサーバーのCPUとメモリの使用率は？",
        "What is the disk usage on the CCTVAI server?",
    ],
)
def test_hardware_questions_are_classified(question):
    assert main.classify_hardware_question(question)


@pytest.mark.parametrize(
    "question",
    [
        # Existing deterministic replica intents must keep their questions.
        "7 ngày qua sự kiện theo mức độ nghiêm trọng?",
        "Camera nào ghi nhận nhiều vi phạm nhất tháng này?",
        "Có bao nhiêu sự kiện chưa được xử lý?",
        "",
    ],
)
def test_replica_questions_are_not_claimed_by_hardware(question):
    assert not main.classify_hardware_question(question)


# ---------------------------------------------------------------------------
# Dispatch independence
# ---------------------------------------------------------------------------

def test_hardware_answers_without_rag_mes_or_replica(monkeypatch):
    service = configure_hardware_only(monkeypatch)

    outcome = asyncio.run(main.route_query_outcome(hardware_request()))

    assert outcome.answer == HARDWARE_ANSWER
    assert outcome.answer_scope == "cctvai_hardware"
    assert outcome.results == []
    assert outcome.cctvai_metadata is None
    assert outcome.wms_metadata is None
    assert service.calls[0][0] == "outcome"


def test_hardware_stream_answers_without_rag_mes_or_replica(monkeypatch):
    service = configure_hardware_only(monkeypatch)

    outcome = asyncio.run(main.route_query_stream_outcome(hardware_request()))

    assert outcome.answer_scope == "cctvai_hardware"
    assert outcome.results == []
    assert outcome.cctvai_metadata is None
    assert service.calls[0][0] == "stream"


def test_non_hardware_cctvai_question_still_hits_the_readiness_gate(monkeypatch):
    """Hardware dispatch must not become a bypass for every cctvai question."""
    configure_hardware_only(monkeypatch)
    req = hardware_request(question="7 ngày qua sự kiện theo mức độ nghiêm trọng?")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.route_query_outcome(req))
    assert exc.value.status_code == 503


def test_hardware_intent_outside_cctvai_mode_is_not_claimed(monkeypatch):
    """A CPU question asked in mode=mkac belongs to RAG, not to metrics."""
    service = configure_hardware_only(monkeypatch)
    req = hardware_request(mode="mkac")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.route_query_outcome(req))
    assert exc.value.status_code == 503
    assert service.calls == []


def test_hardware_service_absent_falls_through_to_normal_routing(monkeypatch):
    monkeypatch.setattr(main, "cctvai_hardware_service", None)
    monkeypatch.setattr(main, "rag_pipeline", None)
    monkeypatch.setattr(main, "mes_query_service", None)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.route_query_outcome(hardware_request()))
    # Falls through to the existing gate rather than silently answering.
    assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# Citations and translation
# ---------------------------------------------------------------------------

def test_hardware_scope_skips_the_rag_source_formatter(monkeypatch):
    monkeypatch.setattr(main, "rag_pipeline", ExplodingRagPipeline())

    assert main.format_sources_for_scope([], "cctvai_hardware") == []
    # A stray result payload must still not reach the RAG formatter.
    assert main.format_sources_for_scope([{"x": 1}], "cctvai_hardware") == []


def test_hardware_answers_are_not_retranslated_for_japanese_ui(monkeypatch):
    """The metrics service already renders JA; a second pass can mangle numbers."""
    class ExplodingTranslator:
        async def translate(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("hardware answers must not be re-translated")

    monkeypatch.setattr(main, "translation_service", ExplodingTranslator())
    req = hardware_request(ui_language="ja", question="CCTVAIサーバーのCPU使用率は？")

    answer = asyncio.run(
        main.translate_answer_for_ui(
            HARDWARE_ANSWER,
            req,
            answer_scope="cctvai_hardware",
        )
    )
    assert answer == HARDWARE_ANSWER


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_reports_hardware_independently_of_the_replica(monkeypatch):
    service = FakeHardwareService(available=True)
    monkeypatch.setattr(main, "cctvai_hardware_service", service)
    monkeypatch.setattr(main, "cctvai_query_service", None)

    payload = asyncio.run(main.health())

    assert payload["cctvai_hardware"]["available"] is True
    assert payload["cctvai_database"]["available"] is False


def test_health_reports_hardware_unavailable_when_unconfigured(monkeypatch):
    monkeypatch.setattr(main, "cctvai_hardware_service", None)

    payload = asyncio.run(main.health())

    assert payload["cctvai_hardware"] == {"available": False, "enabled": False}
