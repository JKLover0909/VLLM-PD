"""API contract tests for the CCTVAI mode.

Mirrors test_wms_api.py: no real Postgres, no real LLM — fake objects only.
Verified contracts:
  - authorize_query requires employee_id for mode="cctvai"
  - build_query_cache_key always returns None for cctvai (near-realtime data)
  - Japanese queries are NOT translated (cctvai_database answers natively)
  - /health exposes cctvai_database key without credentials
  - Non-streaming REST response carries cctvai_metadata, other modes do not
  - SSE meta event carries cctvai_metadata
  - Route fails closed 503 when cctvai_query_service is None
  - Non-cctvai response contract omits cctvai_metadata (null excluded)
"""

import asyncio
import json
import os

import pytest

os.environ["ENABLE_AGENT"] = "false"

from starlette.requests import Request
from starlette.responses import StreamingResponse

from src.api import main
from src.api.schemas import QueryRequest
from src.integrations.mes_query_service import MesQueryOutcome, MesQueryStreamOutcome
from src.integrations import cctvai_contract as contract


# ---------------------------------------------------------------------------
# Shared CCTVAI metadata fixture
# ---------------------------------------------------------------------------

CCTVAI_METADATA = {
    "intent": contract.INTENT_EVENTS_BY_SEVERITY,
    "domain": contract.DOMAIN,
    "status": "PARTIAL",
    "reason_codes": [contract.REASON_REPLICA_LAG_UNVERIFIED],
    "latest_event_at": "2026-09-03 06:00:00+00",
    "replica_lag_state": contract.REPLICA_LAG_STATE_UNVERIFIED,
    "grain": "severity",
    "schema_version": contract.SCHEMA_VERSION,
    "data_contract_version": contract.DATA_CONTRACT_VERSION,
    "semantic_contract_version": contract.SEMANTIC_CONTRACT_VERSION,
    "source_system": contract.SOURCE_SYSTEM,
}

_SENSITIVE = ("cctvai_llm_ro", "55434", "password", "fake-pass")


# ---------------------------------------------------------------------------
# Fake objects
# ---------------------------------------------------------------------------


class FakeRagPipeline:
    @staticmethod
    def format_sources(results, **kwargs):
        return []


class FakeCctvaiQueryService:
    def __init__(self, *, available=True):
        self.calls = []
        self._available = available

    @property
    def available(self):
        return self._available

    def status(self):
        return {
            "enabled": True,
            "available": self._available,
            "state": contract.STATE_READY if self._available else contract.STATE_UNAVAILABLE,
        }

    async def query_cctvai_outcome(self, *, question, model, language):
        self.calls.append(("query_cctvai_outcome", question, model, language))
        return MesQueryOutcome(
            answer="Có 8305 sự kiện trong 7 ngày qua.",
            results=[],
            routed_model="local",
            answer_scope="cctvai_database",
            cctvai_metadata=CCTVAI_METADATA,
        )

    async def query_cctvai_stream_outcome(self, *, question, model, language):
        self.calls.append(("query_cctvai_stream_outcome", question, model, language))

        async def tokens():
            yield "token", "Có 8305 sự kiện trong 7 ngày qua."

        return MesQueryStreamOutcome(
            token_stream=tokens(),
            results=[],
            routed_model="local",
            answer_scope="cctvai_database",
            cctvai_metadata=CCTVAI_METADATA,
        )


async def no_rate_limit(*args, **kwargs):
    return None


async def no_wait(*args, **kwargs):
    return None


def cctvai_request(**updates):
    values = {
        "session_id": "00000000-0000-4000-8000-000000000999",
        "question": "7 ngày qua sự kiện theo mức độ nghiêm trọng?",
        "mode": "cctvai",
        "model": "local",
        "ui_language": "vi",
        "employee_id": "000000",
    }
    values.update(updates)
    return QueryRequest(**values)


def parse_sse(body: str):
    return [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


class FakeMesQueryService:
    """Minimal stub so ensure_query_services_ready() passes."""
    pass


def configure_cctvai_api(monkeypatch, *, available=True):
    service = FakeCctvaiQueryService(available=available)
    monkeypatch.setattr(main, "cctvai_query_service", service)
    monkeypatch.setattr(main, "rag_pipeline", FakeRagPipeline())
    monkeypatch.setattr(main, "mes_query_service", FakeMesQueryService())
    monkeypatch.setattr(main, "authorize_query", lambda req: None)
    monkeypatch.setattr(main, "enforce_rate_limit", no_rate_limit)
    monkeypatch.setattr(main, "wait_for_min_query_latency", no_wait)
    monkeypatch.setattr(main, "translation_service", None)
    return service


def request_context(path="/query"):
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )


# ---------------------------------------------------------------------------
# authorize_query — cctvai requires employee_id
# ---------------------------------------------------------------------------


def test_cctvai_mode_requires_employee_id():
    # mode="research" → authorize_query returns None immediately (no employee check)
    research_req = cctvai_request(mode="research", employee_id="000000")
    assert main.authorize_query(research_req) is None

    # mode="cctvai" → authorize_query must call verify_mkac_employee.
    # The invariant: "cctvai" is in the authorized set, so the function proceeds to
    # verify_mkac_employee rather than short-circuiting to None.
    # An invalid format raises 403; a valid format returns None (not found) or a record.
    from fastapi import HTTPException
    cctvai_none = cctvai_request(employee_id=None)
    with pytest.raises(HTTPException) as exc:
        main.authorize_query(cctvai_none)
    # 403 = the check ran (cctvai IS in the authorized set)
    assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# build_query_cache_key — never caches cctvai
# ---------------------------------------------------------------------------


def test_cctvai_requests_are_not_cached(monkeypatch):
    class VersionedMesSnapshot:
        def snapshot_version(self):
            return "mes-v2"

    monkeypatch.setattr(main, "mes_database", VersionedMesSnapshot())
    assert main.build_query_cache_key(cctvai_request()) is None


def test_non_cctvai_requests_can_still_cache(monkeypatch):
    class VersionedMesSnapshot:
        def snapshot_version(self):
            return "mes-v2"

    monkeypatch.setattr(main, "mes_database", VersionedMesSnapshot())
    mes_req = cctvai_request(
        mode="mes",
        question="Lot 000432-01-000 có bao nhiêu lỗi?",
        employee_id="000000",
    )
    assert main.build_query_cache_key(mes_req) is not None


# ---------------------------------------------------------------------------
# Japanese queries are NOT translated for cctvai
# ---------------------------------------------------------------------------


def test_japanese_cctvai_query_skips_translation(monkeypatch):
    """localize_query_request must return the request unchanged for mode=cctvai."""
    called = []

    class FakeTranslation:
        async def translate_query(self, question, **kwargs):
            called.append(question)
            from src.i18n.translation import TranslationResult
            return TranslationResult(
                original_question=question,
                backend_question="translated",
                ui_language="ja",
            )

    monkeypatch.setattr(main, "translation_service", FakeTranslation())
    req = cctvai_request(ui_language="ja", question="カメラの違反イベントは？")
    result = asyncio.run(main.localize_query_request(req))
    assert result.question == req.question
    assert called == [], "translation_service.translate_query must NOT be called for cctvai"


# ---------------------------------------------------------------------------
# /health — cctvai_database key present, no credentials
# ---------------------------------------------------------------------------


def test_health_exposes_cctvai_database_key(monkeypatch):
    service = FakeCctvaiQueryService(available=True)
    monkeypatch.setattr(main, "cctvai_query_service", service)
    monkeypatch.setattr(main, "rag_pipeline", FakeRagPipeline())
    monkeypatch.setattr(main, "mes_database", None)
    monkeypatch.setattr(main, "mes_wms_database", None)
    monkeypatch.setattr(main, "gmail_sender", None)
    monkeypatch.setattr(main, "translation_service", None)
    monkeypatch.setattr(main, "embedder", None)
    monkeypatch.setattr(main, "doc_parser", None)

    response = asyncio.run(main.health())
    assert "cctvai_database" in response
    payload = str(response["cctvai_database"])
    for s in _SENSITIVE:
        assert s not in payload, f"Sensitive string {s!r} in /health cctvai_database"


def test_health_cctvai_disabled_when_service_none(monkeypatch):
    monkeypatch.setattr(main, "cctvai_query_service", None)
    monkeypatch.setattr(main, "rag_pipeline", FakeRagPipeline())
    monkeypatch.setattr(main, "mes_database", None)
    monkeypatch.setattr(main, "mes_wms_database", None)
    monkeypatch.setattr(main, "gmail_sender", None)
    monkeypatch.setattr(main, "translation_service", None)
    monkeypatch.setattr(main, "embedder", None)
    monkeypatch.setattr(main, "doc_parser", None)

    response = asyncio.run(main.health())
    assert response["cctvai_database"]["enabled"] is False
    assert response["cctvai_database"]["available"] is False


# ---------------------------------------------------------------------------
# Non-streaming REST — cctvai_metadata present; other modes exclude it
# ---------------------------------------------------------------------------


def test_non_streaming_cctvai_response_includes_metadata(monkeypatch):
    configure_cctvai_api(monkeypatch)

    req = cctvai_request(stream=False)
    response = asyncio.run(main.query_documents(req, request_context()))
    assert response.cctvai_metadata is not None
    assert response.cctvai_metadata.intent == contract.INTENT_EVENTS_BY_SEVERITY
    assert contract.REASON_REPLICA_LAG_UNVERIFIED in response.cctvai_metadata.reason_codes
    assert response.cctvai_metadata.source_system == contract.SOURCE_SYSTEM


def test_non_cctvai_response_omits_cctvai_metadata(monkeypatch):
    """A WMS response must not carry a cctvai_metadata field."""
    from tests.test_wms_api import configure_wms_api, wms_request

    configure_wms_api(monkeypatch)
    req = wms_request(stream=False)
    response = asyncio.run(main.query_documents(req, request_context()))
    dumped = response.model_dump(exclude_none=True)
    assert "cctvai_metadata" not in dumped


# ---------------------------------------------------------------------------
# SSE streaming — meta event carries cctvai_metadata
# ---------------------------------------------------------------------------


def test_streaming_cctvai_meta_event_has_metadata(monkeypatch):
    configure_cctvai_api(monkeypatch)

    req = cctvai_request(stream=True)
    streaming_response = asyncio.run(main.query_stream(req, request_context()))
    assert isinstance(streaming_response, StreamingResponse)

    async def collect():
        chunks = []
        async for chunk in streaming_response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    body = asyncio.run(collect())
    events = parse_sse(body)
    meta = next((e for e in events if e.get("type") == "meta"), None)
    assert meta is not None, "No meta event in SSE stream"
    assert "cctvai_metadata" in meta, f"cctvai_metadata missing from meta: {meta}"
    assert meta["cctvai_metadata"]["intent"] == contract.INTENT_EVENTS_BY_SEVERITY
    assert meta["answer_scope"] == "cctvai_database"


# ---------------------------------------------------------------------------
# 503 when service is None
# ---------------------------------------------------------------------------


def test_cctvai_mode_fails_closed_503_when_service_missing(monkeypatch):
    monkeypatch.setattr(main, "cctvai_query_service", None)
    monkeypatch.setattr(main, "rag_pipeline", FakeRagPipeline())
    monkeypatch.setattr(main, "mes_query_service", None)
    monkeypatch.setattr(main, "authorize_query", lambda req: None)
    monkeypatch.setattr(main, "enforce_rate_limit", no_rate_limit)
    monkeypatch.setattr(main, "wait_for_min_query_latency", no_wait)
    monkeypatch.setattr(main, "translation_service", None)

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            main.route_query_outcome(cctvai_request())
        )
    assert exc_info.value.status_code == 503

    with pytest.raises(HTTPException) as stream_exc:
        asyncio.run(
            main.route_query_stream_outcome(cctvai_request())
        )
    assert stream_exc.value.status_code == 503


# ---------------------------------------------------------------------------
# safe_cctvai_metadata — allowlist gate
# ---------------------------------------------------------------------------


def test_safe_cctvai_metadata_accepts_valid_payload():
    result = main.safe_cctvai_metadata(CCTVAI_METADATA)
    assert result is not None
    assert result.intent == contract.INTENT_EVENTS_BY_SEVERITY
    assert result.source_system == contract.SOURCE_SYSTEM


def test_safe_cctvai_metadata_returns_none_for_empty():
    assert main.safe_cctvai_metadata(None) is None
    assert main.safe_cctvai_metadata({}) is None


def test_safe_cctvai_metadata_drops_unknown_fields():
    payload = {**CCTVAI_METADATA, "injected_field": "evil_value"}
    result = main.safe_cctvai_metadata(payload)
    assert result is not None
    assert not hasattr(result, "injected_field")
