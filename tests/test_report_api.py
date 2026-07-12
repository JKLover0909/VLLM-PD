import asyncio
import json
import os

import pytest

os.environ["ENABLE_AGENT"] = "false"

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from src.actions.artifact_store import ArtifactStore, StoredArtifact
from src.api import main
from src.api.helpers import query_cache_key
from src.api.schemas import QueryRequest


REPORT_ID = "00000000-0000-4000-8000-000000000321"


def report_payload():
    return {
        "id": REPORT_ID,
        "title": "Báo cáo lỗi sản xuất MES — fixture",
        "period_label": "toàn bộ dữ liệu snapshot",
        "generated_at": "10:00 01/07/2026",
        "snapshot_imported_at": "2026-06-20",
        "kpis": [
            {"key": "total_error_qty", "label": "Tổng lỗi", "value": 35},
            {"key": "error_record_count", "label": "Số bản ghi lỗi", "value": 3},
            {"key": "lot_count", "label": "Số Lot có lỗi", "value": 2},
            {"key": "product_count", "label": "Số mã hàng có lỗi", "value": 2},
        ],
        "sections": [],
        "observations": ["LOT-B có tổng lỗi cao nhất."],
        "limitations": ["Số liệu lấy từ MES snapshot."],
        "markdown": "## Báo cáo fixture",
    }


class FakeReportAgent:
    available = True

    async def build_report(self, question):
        assert "báo cáo" in question.lower()
        return report_payload(), "Đã tạo báo cáo fixture."

    async def run(self, question):
        assert "báo cáo" in question.lower()
        yield {
            "event": "plan",
            "title": "Báo cáo fixture",
            "period_label": "toàn bộ dữ liệu snapshot",
            "steps": [{"id": "kpi", "title": "Tổng quan lỗi trong kỳ"}],
        }
        yield {"event": "step_start", "step_id": "kpi", "title": "Tổng quan lỗi trong kỳ"}
        yield {
            "event": "step_result",
            "step_id": "kpi",
            "status": "done",
            "summary": "Tổng lỗi trong kỳ: 35",
        }
        yield {
            "event": "report",
            "report": report_payload(),
            "summary_text": "Đã tạo báo cáo fixture.",
        }


class UnavailableReportAgent:
    available = False


class FailIfCalledReportAgent:
    available = True

    async def build_report(self, question):
        raise AssertionError("Unsupported report must not build an artifact")

    async def run(self, question):
        raise AssertionError("Unsupported report must not start the agent")
        yield  # pragma: no cover


async def no_rate_limit(*args, **kwargs):
    return None


def report_request(**updates):
    values = {
        "session_id": "00000000-0000-4000-8000-000000000123",
        "question": "Lập báo cáo top 3 lỗi sản xuất",
        "mode": "mes",
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


def test_report_queries_are_not_cached():
    req = report_request()
    unsupported = report_request(
        question="Tạo báo cáo so sánh lỗi tháng 5/2026 và tháng 6/2026"
    )
    normal = report_request(question="Lot nào có tổng lỗi cao nhất?")

    assert query_cache_key(req, snapshot_version="v1") is None
    assert query_cache_key(unsupported, snapshot_version="v1") is None
    assert query_cache_key(normal, snapshot_version="v1") is not None


def test_handle_report_query_routes_only_mes_reports(monkeypatch):
    store = ArtifactStore()
    monkeypatch.setattr(main, "mes_report_agent", FakeReportAgent())
    monkeypatch.setattr(main, "artifact_store", store)

    response = asyncio.run(main.handle_report_query(report_request()))
    normal = asyncio.run(
        main.handle_report_query(
            report_request(question="Lot nào có tổng lỗi cao nhất?")
        )
    )
    wrong_mode = asyncio.run(
        main.handle_report_query(
            report_request(mode="mkac", question="Lập báo cáo lỗi sản xuất")
        )
    )

    assert response is not None
    assert response.model == "report-agent"
    assert response.answer_scope == "mes_report"
    assert response.answer.startswith("Đã tạo báo cáo fixture")
    assert asyncio.run(store.get(REPORT_ID)) is not None
    assert normal is None
    assert wrong_mode is None


def test_handle_report_query_rejects_unavailable_agent(monkeypatch):
    monkeypatch.setattr(main, "mes_report_agent", UnavailableReportAgent())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.handle_report_query(report_request()))

    assert exc_info.value.status_code == 503
    assert "chưa sẵn sàng" in exc_info.value.detail


@pytest.mark.parametrize(
    "question",
    [
        "Tạo báo cáo so sánh lỗi tháng 5/2026 và tháng 6/2026",
        "Tạo báo cáo chất lượng riêng cho mã hàng PRODUCT-B",
        "Chỉ tạo báo cáo xu hướng tổng lỗi theo tháng",
        "Tạo báo cáo các Lot có trên 100 lỗi",
        "Lập báo cáo sản lượng sản xuất theo ca",
        "Lập báo cáo công tác nhân sự quý 2",
        "Lập báo cáo lỗi tháng 13/2026",
    ],
)
def test_handle_report_query_refuses_unsupported_without_running_agent(
    monkeypatch,
    question,
):
    store = ArtifactStore()
    monkeypatch.setattr(main, "mes_report_agent", FailIfCalledReportAgent())
    monkeypatch.setattr(main, "artifact_store", store)

    response = asyncio.run(
        main.handle_report_query(report_request(question=question))
    )

    assert response is not None
    assert response.answer_scope == "mes_report_unsupported"
    assert response.model == "report-agent"
    assert "không tự đổi yêu cầu sang mẫu mặc định" in response.answer
    assert asyncio.run(store.get(REPORT_ID)) is None


def test_download_report_contract(monkeypatch):
    store = ArtifactStore()
    artifact = StoredArtifact(
        id=REPORT_ID,
        kind="report_html",
        content="<!doctype html><h1>Report</h1>",
        media_type="text/html; charset=utf-8",
        filename="mes-report-fixture.html",
        meta={"title": "Report"},
    )
    asyncio.run(store.put(artifact))
    monkeypatch.setattr(main, "artifact_store", store)

    response = asyncio.run(main.download_report(REPORT_ID))

    assert isinstance(response, Response)
    assert response.status_code == 200
    assert response.media_type.startswith("text/html")
    assert response.headers["content-disposition"] == (
        'attachment; filename="mes-report-fixture.html"'
    )
    assert response.headers["cache-control"] == "private, max-age=300"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert b"<h1>Report</h1>" in response.body


@pytest.mark.parametrize(
    ("report_id", "status_code"),
    [
        ("not-a-uuid", 400),
        ("00000000-0000-4000-8000-000000000999", 404),
    ],
)
def test_download_report_rejects_invalid_or_missing_ids(
    monkeypatch, report_id, status_code
):
    monkeypatch.setattr(main, "artifact_store", ArtifactStore())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.download_report(report_id))

    assert exc_info.value.status_code == status_code


def test_streaming_unsupported_report_returns_refusal_without_agent_protocol(monkeypatch):
    minimum_latency_calls = []

    async def record_minimum_latency(started_at):
        minimum_latency_calls.append(started_at)

    monkeypatch.setattr(main, "mes_report_agent", FailIfCalledReportAgent())
    monkeypatch.setattr(main, "artifact_store", ArtifactStore())
    monkeypatch.setattr(main, "authorize_query", lambda req: None)
    monkeypatch.setattr(main, "enforce_rate_limit", no_rate_limit)
    monkeypatch.setattr(main, "wait_for_min_query_latency", record_minimum_latency)

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/query/stream",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )
    response = asyncio.run(
        main.query_stream(
            report_request(
                question="Tạo báo cáo so sánh lỗi tháng 5/2026 và tháng 6/2026"
            ),
            request,
        )
    )

    async def collect():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    events = parse_sse(asyncio.run(collect()))
    event_types = [event["type"] for event in events]

    assert event_types == ["status", "status", "sources", "meta", "token", "done"]
    assert not {
        "agent_plan",
        "tool_start",
        "tool_result",
        "artifact",
        "agent_done",
        "error",
    }.intersection(event_types)
    assert next(event for event in events if event["type"] == "meta") == {
        "type": "meta",
        "model": "report-agent",
        "mode": "mes",
        "answer_scope": "mes_report_unsupported",
    }
    assert "không tự đổi yêu cầu sang mẫu mặc định" in next(
        event["content"] for event in events if event["type"] == "token"
    )
    assert minimum_latency_calls == []



def test_streaming_report_emits_complete_agent_protocol(monkeypatch):
    store = ArtifactStore()
    minimum_latency_calls = []

    async def record_minimum_latency(started_at):
        minimum_latency_calls.append(started_at)

    monkeypatch.setattr(main, "mes_report_agent", FakeReportAgent())
    monkeypatch.setattr(main, "artifact_store", store)
    monkeypatch.setattr(main, "authorize_query", lambda req: None)
    monkeypatch.setattr(main, "enforce_rate_limit", no_rate_limit)
    monkeypatch.setattr(main, "wait_for_min_query_latency", record_minimum_latency)

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/query/stream",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )
    response = asyncio.run(main.query_stream(report_request(), request))
    assert isinstance(response, StreamingResponse)

    async def collect():
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    events = parse_sse(asyncio.run(collect()))
    event_types = [event["type"] for event in events]

    assert event_types == [
        "status",
        "status",
        "status",
        "agent_plan",
        "tool_start",
        "tool_result",
        "artifact",
        "meta",
        "token",
        "agent_done",
        "done",
    ]
    artifact_event = next(event for event in events if event["type"] == "artifact")
    assert artifact_event["artifact_type"] == "mes_report"
    assert "markdown" not in artifact_event["artifact"]
    assert artifact_event["artifact"]["download_url"] == f"/reports/{REPORT_ID}"
    assert next(event for event in events if event["type"] == "meta") == {
        "type": "meta",
        "model": "report-agent",
        "mode": "mes",
        "answer_scope": "mes_report",
    }
    stored = asyncio.run(store.get(REPORT_ID))
    assert stored is not None
    assert stored.kind == "report_html"
    assert "Báo cáo lỗi sản xuất MES" in stored.content
    assert len(minimum_latency_calls) == 1
