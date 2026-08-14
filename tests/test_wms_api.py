import asyncio
import json
import os

import pytest

os.environ["ENABLE_AGENT"] = "false"

from starlette.requests import Request
from starlette.responses import StreamingResponse

from src.actions.report_intent import report_capability_for_mode
from src.api import main
from src.api.schemas import QueryRequest
from src.integrations.mes_query_service import MesQueryOutcome, MesQueryStreamOutcome


WMS_METADATA = {
    "contract_version": "4",
    "data_contract_version": "wms-current-balance-v1",
    "semantic_contract_version": "wms-phase2c-v1",
    "intent": "wms_current_lot_lookup_suppressed",
    "domain": "CURRENT_BALANCE",
    "status": "SUPPRESSED",
    "reason_codes": ["CURRENT_GRAIN_HAS_NO_MEANINGFUL_LOT"],
    "imported_at": "2026-07-30T10:00:00+00:00",
    "source_as_of": "2026-07-27 23:49:17",
    "source_as_of_state": "DERIVED_UNVERIFIED",
    "source_as_of_basis": "MAX(PW_CURRENT_ITEM.TIME_UPDATE)",
    "source_timezone": "unverified",
    "semantic_epoch": "CURRENT_POST_2026_01_15",
    "dataset_evidence": [
        {
            "dataset": "CURRENT_BALANCE",
            "status": "PARTIAL",
            "reason_code": "UOM_MASTER_UNAVAILABLE",
            "source_state": "PRESENT_NONEMPTY",
            "candidate_row_count": 2,
            "inserted_row_count": 2,
            "invalid_quantity_row_count": 0,
            "source_as_of": "2026-07-27 23:49:17",
            "source_as_of_state": "DERIVED_UNVERIFIED",
            "source_as_of_basis": "MAX(PW_CURRENT_ITEM.TIME_UPDATE)",
            "source_timezone": "unverified",
            "semantic_epoch": "CURRENT_POST_2026_01_15",
        }
    ],
    "grain": "process_id,item_code",
    "pagination": None,
}


class FakeRagPipeline:
    @staticmethod
    def format_sources(results, **kwargs):
        return []


class FakeWmsQueryService:
    def __init__(self):
        self.calls = []

    async def query_wms_outcome(self, *, question, model, language):
        self.calls.append(("query_wms_outcome", question, model, language))
        return MesQueryOutcome(
            answer="Current WMS không hỗ trợ tra theo lot.",
            results=[],
            routed_model="local-qwen-chat",
            answer_scope="wms_database",
            wms_metadata=WMS_METADATA,
        )

    async def query_wms_stream_outcome(self, *, question, model, language):
        self.calls.append(("query_wms_stream_outcome", question, model, language))

        async def tokens():
            yield "token", "Current WMS không hỗ trợ tra theo lot."

        return MesQueryStreamOutcome(
            token_stream=tokens(),
            results=[],
            routed_model="local-qwen-chat",
            answer_scope="wms_database",
            wms_metadata=WMS_METADATA,
        )

    async def query(self, *, question, model, language):
        outcome = await self.query_outcome(
            question=question,
            model=model,
            language=language,
        )
        return outcome.as_tuple()

    async def query_stream(self, *, question, model, language):
        outcome = await self.query_stream_outcome(
            question=question,
            model=model,
            language=language,
        )
        return outcome.as_tuple()


async def no_rate_limit(*args, **kwargs):
    return None


async def no_wait(*args, **kwargs):
    return None


def wms_request(**updates):
    values = {
        "session_id": "00000000-0000-4000-8000-000000000210",
        "question": "Kho công đoạn WMS hiện có bao nhiêu mã vật tư?",
        "mode": "wms",
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


def configure_wms_api(monkeypatch):
    service = FakeWmsQueryService()
    monkeypatch.setattr(main, "mes_query_service", service)
    monkeypatch.setattr(main, "rag_pipeline", FakeRagPipeline())
    monkeypatch.setattr(main, "authorize_query", lambda req: None)
    monkeypatch.setattr(main, "enforce_rate_limit", no_rate_limit)
    monkeypatch.setattr(main, "wait_for_min_query_latency", no_wait)
    monkeypatch.setattr(main, "pace_wms_verification_step", no_wait)
    monkeypatch.setattr(main, "translation_service", None)
    return service


def request_context(path):
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )


def test_wms_requests_are_not_cached(monkeypatch):
    class VersionedMesSnapshot:
        def snapshot_version(self):
            return "mes-v2"

    monkeypatch.setattr(main, "mes_database", VersionedMesSnapshot())

    # WMS answers carry contract/availability metadata: never served from cache.
    assert main.build_query_cache_key(wms_request()) is None
    # A plain MES question in MES mode still caches with its snapshot version.
    assert main.build_query_cache_key(
        wms_request(mode="mes", question="Lot 000432-01-000 có bao nhiêu lỗi?")
    ) is not None


def test_non_streaming_wms_response_includes_safe_metadata(monkeypatch):
    service = configure_wms_api(monkeypatch)

    response = asyncio.run(
        main.query_documents(wms_request(), request_context("/query"))
    )

    assert response.answer_scope == "wms_database"
    assert response.sources == []
    assert response.mode == "wms"
    assert response.wms_metadata.model_dump() == WMS_METADATA
    serialized = response.model_dump_json()
    assert "/home/" not in serialized
    assert "mes_wms.sqlite" not in serialized
    assert "private-user" not in serialized
    assert service.calls == [
        (
            "query_wms_outcome",
            wms_request().question,
            "local",
            "vi",
        )
    ]


def test_streaming_wms_response_has_metadata_parity(monkeypatch):
    service = configure_wms_api(monkeypatch)
    response = asyncio.run(
        main.query_stream(wms_request(), request_context("/query/stream"))
    )
    assert isinstance(response, StreamingResponse)

    async def collect():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    events = parse_sse(asyncio.run(collect()))
    event_types = [event["type"] for event in events]
    meta = next(event for event in events if event["type"] == "meta")

    assert event_types == [
        "status",
        "status",
        "status",
        "agent_plan",
        "tool_start",
        "tool_result",
        "tool_start",
        "tool_result",
        "status",
        "sources",
        "meta",
        "token",
        "agent_done",
        "done",
    ]
    assert meta["type"] == "meta"
    assert meta["model"] == "local-qwen-chat"
    assert meta["mode"] == "wms"
    assert meta["answer_scope"] == "wms_database"
    assert meta["wms_metadata"] == main.safe_wms_metadata(WMS_METADATA)
    assert meta["workflow"] == "wms_verification"
    assert meta["cache"] is False
    assert event_types.index("token") > event_types.index("tool_result")
    assert not {"artifact", "error"}.intersection(event_types)
    assert service.calls == [
        (
            "query_wms_outcome",
            wms_request().question,
            "local",
            "vi",
        )
    ]


def test_non_wms_response_contract_omits_null_metadata():
    response = main.QueryResponse(
        answer="MES answer",
        sources=[],
        session_id="00000000-0000-4000-8000-000000000210",
        model="local-qwen-chat",
        mode="mes",
        answer_scope="mes_database",
    )

    assert "wms_metadata" not in response.model_dump(exclude_none=True)


def test_wms_mode_fails_closed_when_isolated_service_is_missing(monkeypatch):
    class LegacyMesQueryService:
        def __init__(self):
            self.calls = []

        async def query_outcome(self, **kwargs):  # pragma: no cover
            self.calls.append(("query_outcome", kwargs))
            return MesQueryOutcome(
                answer="MES answer",
                results=[],
                routed_model="legacy-model",
                answer_scope="mes_database",
            )

        async def query_stream_outcome(self, **kwargs):  # pragma: no cover
            self.calls.append(("query_stream_outcome", kwargs))
            raise AssertionError("WMS không được fallback sang route MES")

    service = LegacyMesQueryService()
    monkeypatch.setattr(main, "mes_query_service", service)
    monkeypatch.setattr(main, "rag_pipeline", FakeRagPipeline())

    with pytest.raises(main.HTTPException) as rest_error:
        asyncio.run(main.route_query_outcome(wms_request()))
    with pytest.raises(main.HTTPException) as stream_error:
        asyncio.run(main.route_query_stream_outcome(wms_request()))

    assert rest_error.value.status_code == 503
    assert "WMS query service" in rest_error.value.detail
    assert stream_error.value.status_code == 503
    assert "WMS streaming query service" in stream_error.value.detail
    assert service.calls == []


@pytest.mark.parametrize("language", ["vi", "ja"])
def test_wms_quick_answers_preserve_execution_boundary(monkeypatch, language):
    main._quick_answers_cache = None
    payload = asyncio.run(main.quick_answers(mode="wms", language=language))
    suggestions = payload["suggestions"]
    prepared = next(
        item for item in suggestions if item["id"] == "wms-snapshot-explainer"
    )
    query = next(item for item in suggestions if item["id"] == "wms-item-count")

    assert [item["id"] for item in suggestions] == [
        "wms-executive-overview",
        "wms-item-count",
        "wms-snapshot-explainer",
    ]
    assert all(item["question"].strip() for item in suggestions)
    assert prepared["execution"] == "server_prepared"
    assert query["execution"] == "query"
    assert "answer" not in prepared
    assert "answer_ja" not in prepared


def test_wms_query_suggestion_id_fails_closed(monkeypatch):
    service = configure_wms_api(monkeypatch)
    available_metadata = {
        **WMS_METADATA,
        "domain": "CURRENT_BALANCE",
        "status": "PARTIAL",
        "reason_codes": ["UOM_MASTER_UNAVAILABLE"],
    }

    async def available_snapshot(**kwargs):
        return MesQueryOutcome(
            answer="Snapshot count answer",
            results=[],
            routed_model="local-qwen-chat",
            answer_scope="wms_database",
            wms_metadata=available_metadata,
        )

    service.query_wms_outcome = available_snapshot
    req = wms_request(
        question="Kho công đoạn WMS hiện có bao nhiêu mã vật tư?",
        quick_answer_id="wms-item-count",
    )
    response = asyncio.run(main.query_stream(req, request_context("/query/stream")))

    async def collect():
        return "".join(
            [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]
        )

    events = parse_sse(asyncio.run(collect()))
    meta = next(event for event in events if event["type"] == "meta")
    token = next(event for event in events if event["type"] == "token")
    assert meta["wms_metadata"]["status"] == "SUPPRESSED"
    assert meta["wms_metadata"]["reason_codes"] == [
        "WMS_PREPARED_VALIDATION_FAILED"
    ]
    assert token["content"] != "Snapshot count answer"
    assert events[-1]["type"] == "done"


def test_wms_quick_answers_hide_prepared_payload(monkeypatch):
    main._quick_answers_cache = None
    payload = asyncio.run(main.quick_answers(mode="wms", language="vi"))
    prepared = next(
        item
        for item in payload["suggestions"]
        if item["id"] == "wms-snapshot-explainer"
    )
    assert prepared["execution"] == "server_prepared"
    assert "answer" not in prepared
    assert "answer_ja" not in prepared


def test_wms_prepared_resolver_requires_allowlisted_match():
    req = wms_request(
        question="Dữ liệu tồn kho WMS có phải thời gian thực không?",
        quick_answer_id="wms-snapshot-explainer",
    )
    outcome = main.resolve_wms_prepared_response(req)
    assert outcome is not None
    assert outcome.answer_scope == "wms_database"
    assert outcome.wms_metadata["data_contract_version"] == "wms-current-balance-v1"

    mismatch = main.resolve_wms_prepared_response(
        req.model_copy(update={"question": "Tồn kho WMS ITEM-A ở đâu?"})
    )
    assert mismatch is None


@pytest.mark.parametrize("quick_answer_id", [None, "wms-snapshot-explainer"])
def test_wms_prepared_synchronous_query_is_rejected(quick_answer_id):
    with pytest.raises(main.HTTPException) as exc_info:
        asyncio.run(
            main.query_documents(
                wms_request(
                    question="Dữ liệu tồn kho WMS có phải thời gian thực không?",
                    quick_answer_id=quick_answer_id,
                ),
                request_context("/query"),
            )
        )
    assert exc_info.value.status_code == 409
    assert "WMS_STREAM_REQUIRED" in exc_info.value.detail


def test_wms_metadata_schema_drops_unknown_evidence_fields():
    metadata = {
        **WMS_METADATA,
        "runtime_path": "/home/private/mes_wms.sqlite",
        "dataset_evidence": [
            {
                **WMS_METADATA["dataset_evidence"][0],
                "source_path": "/home/private/mes_wms.sql",
                "raw_identifier": "ITEM-SECRET",
            }
        ],
    }

    parsed = main.safe_wms_metadata(metadata)
    assert parsed is not None
    serialized = json.dumps(parsed)

    assert "runtime_path" not in parsed
    assert "source_path" not in parsed["dataset_evidence"][0]
    assert "raw_identifier" not in parsed["dataset_evidence"][0]
    assert "/home/private" not in serialized
    assert "ITEM-SECRET" not in serialized


def test_non_streaming_wms_invalid_metadata_fails_closed(monkeypatch):
    class InvalidMetadataWmsService:
        async def query_wms_outcome(self, **kwargs):
            return MesQueryOutcome(
                answer="Số liệu không được xác minh",
                results=[],
                routed_model="local-qwen-chat",
                answer_scope="wms_database",
                wms_metadata={"dataset_evidence": "not-a-list"},
            )

    monkeypatch.setattr(main, "mes_query_service", InvalidMetadataWmsService())
    monkeypatch.setattr(main, "rag_pipeline", FakeRagPipeline())
    monkeypatch.setattr(main, "authorize_query", lambda req: None)
    monkeypatch.setattr(main, "enforce_rate_limit", no_rate_limit)
    monkeypatch.setattr(main, "wait_for_min_query_latency", no_wait)

    response = asyncio.run(
        main.query_documents(wms_request(), request_context("/query"))
    )

    assert response.answer_scope == "wms_database"
    assert response.wms_metadata is not None
    assert response.wms_metadata.status == "SUPPRESSED"
    assert response.wms_metadata.reason_codes == ["WMS_METADATA_VALIDATION_FAILED"]
    assert "Số liệu không được xác minh" not in response.answer


def test_wms_prepared_contract_epoch_mismatch_fails_closed(monkeypatch):
    service = configure_wms_api(monkeypatch)
    mismatch_metadata = {
        **WMS_METADATA,
        "domain": "CURRENT_BALANCE",
        "status": "PARTIAL",
        "reason_codes": ["UOM_MASTER_UNAVAILABLE"],
        "semantic_epoch": "STALE_EPOCH",
    }

    async def mismatched_snapshot(**kwargs):
        service.calls.append(("mismatched_snapshot", kwargs["question"]))
        return MesQueryOutcome(
            answer="Snapshot answer",
            results=[],
            routed_model="local-qwen-chat",
            answer_scope="wms_database",
            wms_metadata=mismatch_metadata,
        )

    service.query_wms_outcome = mismatched_snapshot
    req = wms_request(
        question="Dữ liệu tồn kho WMS có phải thời gian thực không?",
        quick_answer_id="wms-snapshot-explainer",
    )
    response = asyncio.run(main.query_stream(req, request_context("/query/stream")))

    async def collect():
        return "".join(
            [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]
        )

    events = parse_sse(asyncio.run(collect()))
    token = next(event for event in events if event["type"] == "token")
    meta = next(event for event in events if event["type"] == "meta")

    assert meta["wms_metadata"]["status"] == "SUPPRESSED"
    assert meta["wms_metadata"]["reason_codes"] == [
        "WMS_PREPARED_VALIDATION_FAILED"
    ]
    assert "snapshot tĩnh cuối ngày" not in token["content"]
    assert "Không thể xác minh gợi ý WMS" in token["content"]


def test_wms_stream_invalid_metadata_suppresses_before_token(monkeypatch):
    class InvalidMetadataWmsService:
        async def query_wms_outcome(self, **kwargs):
            return MesQueryOutcome(
                answer="Số liệu không được xác minh",
                results=[],
                routed_model="local-qwen-chat",
                answer_scope="wms_database",
                wms_metadata={"dataset_evidence": "not-a-list"},
            )

    monkeypatch.setattr(main, "mes_query_service", InvalidMetadataWmsService())
    monkeypatch.setattr(main, "rag_pipeline", FakeRagPipeline())
    monkeypatch.setattr(main, "authorize_query", lambda req: None)
    monkeypatch.setattr(main, "enforce_rate_limit", no_rate_limit)

    response = asyncio.run(
        main.query_stream(wms_request(), request_context("/query/stream"))
    )

    async def collect():
        return "".join(
            [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]
        )

    events = parse_sse(asyncio.run(collect()))
    token_index = next(
        index for index, event in enumerate(events) if event["type"] == "token"
    )
    meta = next(event for event in events if event["type"] == "meta")

    assert meta["wms_metadata"]["status"] == "SUPPRESSED"
    assert meta["wms_metadata"]["reason_codes"] == [
        "WMS_METADATA_VALIDATION_FAILED"
    ]
    assert "Số liệu không được xác minh" not in events[token_index]["content"]
    assert token_index > max(
        index
        for index, event in enumerate(events)
        if event["type"] == "tool_result"
    )


def test_wms_stream_paces_completed_milestones_before_finalization(monkeypatch):
    configure_wms_api(monkeypatch)
    pacing_multipliers = []

    async def record_pacing(multiplier=1.0):
        pacing_multipliers.append(multiplier)
        return 0.0

    monkeypatch.setattr(main, "pace_wms_presentation", record_pacing)
    response = asyncio.run(
        main.query_stream(wms_request(), request_context("/query/stream"))
    )

    async def collect():
        return "".join(
            [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]
        )

    events = parse_sse(asyncio.run(collect()))
    assert pacing_multipliers == [0.45, 0.45, 0.45, 1.0, 1.0]
    assert [event["type"] for event in events][-4:] == [
        "meta",
        "token",
        "agent_done",
        "done",
    ]


def test_wms_stream_cancellation_does_not_emit_final_events_or_metrics(monkeypatch):
    configure_wms_api(monkeypatch)
    completed_metrics = []

    class CancelledWmsService:
        async def query_wms_outcome(self, **kwargs):
            raise asyncio.CancelledError()

    async def record_completion(**kwargs):
        completed_metrics.append(kwargs)

    monkeypatch.setattr(main, "mes_query_service", CancelledWmsService())
    monkeypatch.setattr(main, "record_wms_verification_metric", record_completion)
    monkeypatch.setattr(main, "record_query_metric", record_completion)
    response = asyncio.run(
        main.query_stream(wms_request(), request_context("/query/stream"))
    )

    async def collect_until_cancelled():
        events = []
        with pytest.raises(asyncio.CancelledError):
            async for chunk in response.body_iterator:
                body = chunk.decode() if isinstance(chunk, bytes) else chunk
                events.extend(parse_sse(body))
        return events

    events = asyncio.run(collect_until_cancelled())
    event_types = [event["type"] for event in events]
    assert "done" not in event_types
    assert "agent_done" not in event_types
    assert "token" not in event_types
    assert completed_metrics == []


def test_wms_verification_metrics_are_aggregate_only(monkeypatch):
    service = configure_wms_api(monkeypatch)
    previous_metrics = main.query_metrics
    main.query_metrics = {
        "total": 0,
        "cache_hits": 0,
        "errors": 0,
        "by_scope": main.defaultdict(int),
        "by_mode": main.defaultdict(int),
        "latency_ms": main.deque(maxlen=500),
        "wms_verification": {
            "by_source_kind": main.defaultdict(int),
            "by_outcome": main.defaultdict(int),
            "duration_ms": main.deque(maxlen=500),
            "snapshot_validation_ms": main.deque(maxlen=500),
            "answer_validation_ms": main.deque(maxlen=500),
            "presentation_pacing_ms": main.deque(maxlen=500),
        },
    }
    try:
        response = asyncio.run(
            main.query_stream(wms_request(), request_context("/query/stream"))
        )

        async def consume():
            async for _ in response.body_iterator:
                pass

        asyncio.run(consume())
        metrics = asyncio.run(main.metrics())
    finally:
        main.query_metrics = previous_metrics

    assert service.calls
    assert metrics["wms_verification"]["by_source_kind"] == {"snapshot": 1}
    assert metrics["wms_verification"]["by_outcome"] == {"suppressed": 1}
    assert metrics["wms_verification"]["duration_ms"]["count"] == 1
    assert metrics["wms_verification"]["snapshot_validation_ms"]["count"] == 1
    assert metrics["wms_verification"]["answer_validation_ms"]["count"] == 1
    assert metrics["wms_verification"]["presentation_pacing_ms"]["count"] == 1
    serialized = json.dumps(metrics)
    assert wms_request().question not in serialized
    assert wms_request().employee_id not in serialized
    assert wms_request().session_id not in serialized


def test_wms_mode_requires_employee_authorization(monkeypatch):
    checked = []

    def fake_verify(employee_id):
        checked.append(employee_id)
        return None

    monkeypatch.setattr(main, "verify_mkac_employee", fake_verify)

    main.authorize_query(wms_request())
    main.authorize_query(wms_request(mode="research"))

    # Chỉ mode cần danh tính nhân viên mới gọi verify; research thì không.
    assert checked == ["000000"]


def test_wms_mode_keeps_japanese_question_untranslated(monkeypatch):
    class ExplodingTranslationService:
        async def translate(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("WMS phải giữ nguyên câu hỏi tiếng Nhật")

    monkeypatch.setattr(main, "translation_service", ExplodingTranslationService())

    req = wms_request(
        ui_language="ja",
        question="WMS工程倉庫には現在いくつの資材コードがありますか？",
    )
    localized = asyncio.run(main.localize_query_request(req))

    assert localized.question == req.question


def test_wms_mode_refuses_mes_report_intent():
    # Câu báo cáo MES gõ nhầm trong tab WMS phải fail closed có hướng dẫn.
    capability = report_capability_for_mode("Lập báo cáo tổng hợp lỗi MES", "wms")

    assert capability.status == "unsupported"
    assert capability.shape == "mode_mismatch"
    assert capability.domain == "mes"


def test_wms_route_isolated_from_mes_service(monkeypatch):
    class IsolatedService:
        def __init__(self):
            self.calls = []

        async def query_wms_outcome(self, *, question, model, language):
            self.calls.append((question, model, language))
            return MesQueryOutcome(
                answer="WMS only",
                results=[],
                routed_model="local-qwen-chat",
                answer_scope="wms_database",
                wms_metadata=WMS_METADATA,
            )

        async def query_outcome(self, **kwargs):  # pragma: no cover
            raise AssertionError("WMS không được gọi route MES")

    service = IsolatedService()
    monkeypatch.setattr(main, "mes_query_service", service)
    monkeypatch.setattr(main, "rag_pipeline", FakeRagPipeline())

    outcome = asyncio.run(main.route_query_outcome(wms_request()))

    assert outcome.answer == "WMS only"
    assert service.calls == [(wms_request().question, "local", "vi")]


def test_wms_mode_refuses_mes_report_before_query_service(monkeypatch):
    class FailingService:
        async def query_wms_outcome(self, **kwargs):  # pragma: no cover
            raise AssertionError("MES report phải bị chặn trước routing")

    monkeypatch.setattr(main, "mes_query_service", FailingService())
    response = asyncio.run(
        main.handle_report_query(
            wms_request(question="Lập báo cáo tổng hợp lỗi MES")
        )
    )

    assert response is not None
    assert response.answer_scope == "mes_report_unsupported"
    assert "chế độ MES" in response.answer


def test_wms_health_uses_allowlisted_top_level_counts(monkeypatch):
    class SafeWmsDatabase:
        @staticmethod
        def status():
            return {
                "enabled": True,
                "available": True,
                "compatible": True,
                "state": "READY",
                "distinct_items": 4,
                "distinct_process_codes": 2,
            }

    monkeypatch.setattr(main, "mes_wms_database", SafeWmsDatabase())
    payload = asyncio.run(main.health())

    assert payload["mes_wms_database"]["distinct_items"] == 4
    assert payload["mes_wms_database"]["distinct_process_codes"] == 2
    assert "db_path" not in payload["mes_wms_database"]


def test_mes_mode_refuses_wms_report_intent():
    # Ngược lại: câu báo cáo WMS trong tab MES cũng phải fail closed.
    capability = report_capability_for_mode(
        "Lập báo cáo tổng quan tồn kho WMS",
        "mes",
    )

    assert capability.status == "unsupported"
    assert capability.shape == "mode_mismatch"
    assert capability.domain == "wms"


def test_wms_mode_opens_wms_executive_report():
    capability = report_capability_for_mode(
        "Lập báo cáo tổng quan tồn kho WMS",
        "wms",
    )

    assert capability.status == "supported"
    assert capability.domain == "wms"
    assert capability.shape == "wms_executive"


def test_health_omits_wms_filesystem_path(monkeypatch):
    class SafeWmsDatabase:
        @staticmethod
        def status():
            return {
                "enabled": True,
                "available": True,
                "compatible": True,
                "state": "READY",
            }

    monkeypatch.setattr(main, "mes_wms_database", SafeWmsDatabase())

    payload = asyncio.run(main.health())
    serialized = json.dumps(payload)

    assert payload["mes_wms_database"]["state"] == "READY"
    assert "db_path" not in payload["mes_wms_database"]
    assert "mes_wms.sqlite" not in serialized
