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
from src.integrations.gmail_sender import (
    EmailDraftStore,
    GmailSendResult,
)


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


class FakeHrReportAgent:
    available = True

    async def build_report(self, question, language="vi"):
        assert language in {"vi", "ja"}
        return (
            {
                "id": REPORT_ID,
                "report_type": "hr_executive_report",
                "language": language,
                "title": "Báo cáo Tổng quan Nhân sự Cấp Điều hành",
                "generated_at": "2026-07-30 10:00:00",
                "period_label": "Danh bạ nhân sự hiện tại",
                "html_content": (
                    '<!doctype html><html lang="vi"><body>HR fixture</body></html>'
                ),
                "kpis": [{"key": "headcount", "label": "Tổng nhân sự", "value": 10}],
                "charts": [
                    {
                        "id": "headcount-chart",
                        "title": "Headcount",
                        "rows": [{"department": "A", "size": 10}],
                        "svg": "<svg>private</svg>",
                    }
                ],
                "matrices": [],
                "sections": [],
                "observations": ["HR snapshot fixture"],
                "governance": ["Aggregate only"],
                "limitations": ["Not realtime"],
            },
            "Đã tạo Báo cáo Nhân sự Cấp Điều hành.",
        )


class FakeWmsReportAgent:
    available = True

    async def generate_report(self, question, language="vi"):
        assert "WMS" in question
        assert language in {"vi", "ja"}
        return (
            {
                "id": REPORT_ID,
                "report_type": "wms_executive_report",
                "title": "Báo cáo Tồn kho WMS Cấp Điều hành",
                "generated_at": "2026-07-30 10:00:00",
                "period_label": "Snapshot WMS 2026-07-27 23:49:17",
                "html_content": (
                    '<!doctype html><html lang="vi"><body>WMS fixture</body></html>'
                ),
                "kpis": [
                    {
                        "key": "distinct_item_count",
                        "label": "Mã vật tư",
                        "value": 3,
                    }
                ],
                "observations": ["WMS snapshot fixture"],
                "limitations": ["UOM chưa được xác minh"],
            },
            "Đã tạo Báo cáo WMS Cấp Điều hành.",
        )


class IncompatibleWmsReportAgent:
    available = False


class FakeReportAgent:
    available = True

    async def build_report(self, question, language="vi"):
        assert "báo cáo" in question.lower()
        return report_payload(), "Đã tạo báo cáo fixture."

    async def run(self, question, language="vi"):
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


class FailIfCalledTranslationService:
    async def translate_answer(self, *args, **kwargs):
        raise AssertionError("Localized report summaries must not be translated again")


class FakeGmailSender:
    available = True

    def __init__(self):
        self.calls = []

    def send_email(self, to_email, subject, body, *, attachments=None):
        self.calls.append(
            {
                "to_email": to_email,
                "subject": subject,
                "body": body,
                "attachments": attachments or [],
            }
        )
        return GmailSendResult(
            message_id="gmail-message-1",
            to_email=to_email,
            subject=subject,
        )


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
    wms_wrong_mode = asyncio.run(
        main.handle_report_query(
            report_request(
                mode="mkac",
                question="Lập báo cáo tổng quan tồn kho công đoạn WMS",
            )
        )
    )
    wms_unsupported_wrong_mode = asyncio.run(
        main.handle_report_query(
            report_request(mode="mkac", question="Lập báo cáo tổng tồn kho WMS")
        )
    )

    assert response is not None
    assert response.model == "report-agent"
    assert response.answer_scope == "mes_report"
    assert response.answer.startswith("Đã tạo báo cáo fixture")
    assert asyncio.run(store.get(REPORT_ID)) is not None
    assert normal is None
    assert wrong_mode is not None
    assert wrong_mode.answer_scope == "mes_report_unsupported"
    assert wms_wrong_mode is not None
    assert wms_wrong_mode.answer_scope == "mes_report_unsupported"
    assert wms_unsupported_wrong_mode is not None
    assert wms_unsupported_wrong_mode.answer_scope == "mes_report_unsupported"


def test_handle_hr_report_query_returns_safe_artifact(monkeypatch):
    store = ArtifactStore()
    monkeypatch.setattr(main, "hr_report_agent", FakeHrReportAgent())
    monkeypatch.setattr(main, "artifact_store", store)

    response = asyncio.run(
        main.handle_report_query(
            report_request(
                mode="mkac",
                question="Ban giám đốc cần tổng quan nhân sự",
            )
        )
    )

    assert response is not None
    assert response.answer_scope == "hr_executive_report"
    assert response.artifact["report_type"] == "hr_executive_report"
    assert response.artifact["governance"] == ["Aggregate only"]
    assert "html_content" not in response.artifact
    assert "svg" not in response.artifact["charts"][0]
    stored = asyncio.run(store.get(REPORT_ID))
    assert stored is not None
    assert stored.session_id == response.session_id
    assert stored.employee_id == "000000"
    assert stored.meta == {
        "session_id": response.session_id,
        "employee_id": "000000",
        "report_type": "hr_executive_report",
        "title": "Báo cáo Tổng quan Nhân sự Cấp Điều hành",
    }


def test_handle_wms_report_query_persists_native_html(monkeypatch):
    store = ArtifactStore()
    monkeypatch.setattr(main, "mes_wms_report_agent", FakeWmsReportAgent())
    monkeypatch.setattr(main, "artifact_store", store)

    response = asyncio.run(
        main.handle_report_query(
            report_request(
                mode="wms",
                question="Lập báo cáo tổng quan tồn kho công đoạn WMS",
            )
        )
    )

    assert response is not None
    assert response.answer_scope == "wms_executive_report"
    stored = asyncio.run(store.get(REPORT_ID))
    assert stored is not None
    assert stored.filename == "wms-report-00000000.html"
    assert "WMS fixture" in stored.content


def test_rest_japanese_wms_report_preserves_native_summary(monkeypatch):
    store = ArtifactStore()
    monkeypatch.setattr(main, "mes_wms_report_agent", FakeWmsReportAgent())
    monkeypatch.setattr(main, "artifact_store", store)
    monkeypatch.setattr(main, "translation_service", FailIfCalledTranslationService())
    monkeypatch.setattr(main, "authorize_query", lambda req: None)
    monkeypatch.setattr(main, "enforce_rate_limit", no_rate_limit)
    monkeypatch.setattr(main, "wait_for_min_query_latency", no_rate_limit)

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/query",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )
    response = asyncio.run(
        main.query_documents(
            report_request(
                mode="wms",
                question="WMS在庫レポートを作成",
                ui_language="ja",
            ),
            request,
        )
    )

    assert response.answer == "Đã tạo Báo cáo WMS Cấp Điều hành."
    assert response.answer_scope == "wms_executive_report"


def test_handle_wms_report_rejects_incompatible_snapshot(monkeypatch):
    monkeypatch.setattr(
        main,
        "mes_wms_report_agent",
        IncompatibleWmsReportAgent(),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            main.handle_report_query(
                report_request(
                    mode="wms",
                    question="Lập báo cáo tổng quan tồn kho công đoạn WMS",
                )
            )
        )

    assert exc_info.value.status_code == 503
    assert "WMS Report Agent chưa sẵn sàng" in exc_info.value.detail


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
        main.handle_report_query(
            report_request(
                question=question,
                mode="mkac" if "nhân sự" in question else "mes",
            )
        )
    )

    assert response is not None
    assert response.answer_scope == "mes_report_unsupported"
    assert response.model == "report-agent"
    assert "chưa" in response.answer or "không tự đổi" in response.answer
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



def test_streaming_wms_report_emits_safe_complete_protocol(monkeypatch):
    store = ArtifactStore()
    minimum_latency_calls = []

    async def record_minimum_latency(started_at):
        minimum_latency_calls.append(started_at)

    monkeypatch.setattr(main, "mes_wms_report_agent", FakeWmsReportAgent())
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
    response = asyncio.run(
        main.query_stream(
            report_request(
                mode="wms",
                question="Lập báo cáo tổng quan tồn kho công đoạn WMS",
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
    artifact_event = next(event for event in events if event["type"] == "artifact")

    assert event_types == [
        "status",
        "status",
        "status",
        "agent_plan",
        "tool_start",
        "tool_result",
        "tool_start",
        "tool_result",
        "artifact",
        "sources",
        "meta",
        "token",
        "agent_done",
        "done",
    ]
    assert artifact_event["artifact_type"] == "wms_executive_report"
    assert artifact_event["artifact"]["download_url"] == f"/reports/{REPORT_ID}"
    assert "html_content" not in artifact_event["artifact"]
    assert "markdown" not in artifact_event["artifact"]
    assert "<html" not in json.dumps(artifact_event)
    assert next(event for event in events if event["type"] == "meta") == {
        "type": "meta",
        "model": "report-agent",
        "mode": "wms",
        "answer_scope": "wms_executive_report",
    }
    stored = asyncio.run(store.get(REPORT_ID))
    assert stored is not None
    assert "WMS fixture" in stored.content
    assert len(minimum_latency_calls) == 1


def test_report_email_draft_confirm_and_cancel(monkeypatch):
    store = ArtifactStore()
    sender = FakeGmailSender()
    draft_store = EmailDraftStore()
    req = report_request(
        question="Gửi báo cáo này cho email test@example.com",
        conversation_context=[
            {
                "role": "assistant",
                "content": "Đã tạo báo cáo.",
                "answer_scope": "mes_report",
                "artifact_id": REPORT_ID,
                "artifact_type": "mes_report",
                "artifact_title": "MES Executive Report",
            }
        ],
    )
    asyncio.run(
        store.put(
            StoredArtifact(
                id=REPORT_ID,
                kind="report_html",
                content="<!doctype html><h1>MES report</h1>",
                media_type="text/html; charset=utf-8",
                filename="mes-report.html",
                meta={"title": "MES Executive Report"},
                session_id=req.session_id,
                employee_id=req.employee_id,
            )
        )
    )
    monkeypatch.setattr(main, "artifact_store", store)
    monkeypatch.setattr(main, "email_draft_store", draft_store)
    monkeypatch.setattr(main, "gmail_sender", sender)

    draft_response = asyncio.run(main.handle_email_send_query(req, None))
    assert draft_response.answer_scope == "email_action"
    assert "Xác nhận gửi email" in draft_response.answer
    assert sender.calls == []

    confirm_response = asyncio.run(
        main.handle_email_send_query(
            report_request(question="Xác nhận gửi email"),
            None,
        )
    )
    assert "Đã gửi email" in confirm_response.answer
    assert len(sender.calls) == 1
    assert sender.calls[0]["attachments"] == [
        {
            "filename": "mes-report.html",
            "content": "<!doctype html><h1>MES report</h1>",
            "media_type": "text/html; charset=utf-8",
        }
    ]

    repeated = asyncio.run(
        main.handle_email_send_query(
            report_request(question="Xác nhận gửi email"),
            None,
        )
    )
    assert "đã được gửi trước đó" in repeated.answer
    assert len(sender.calls) == 1

    new_draft = asyncio.run(
        main.handle_email_send_query(
            req.model_copy(update={"question": "Gửi báo cáo này cho email other@example.com"}),
            None,
        )
    )
    assert "bản nháp" in new_draft.answer.lower()
    cancelled = asyncio.run(
        main.handle_email_send_query(
            report_request(question="Hủy gửi email"),
            None,
        )
    )
    assert "Đã hủy" in cancelled.answer
    assert asyncio.run(draft_store.get(req.session_id)) is None


def test_report_email_rejects_expired_or_cross_owner_artifact(monkeypatch):
    sender = FakeGmailSender()
    monkeypatch.setattr(main, "gmail_sender", sender)
    monkeypatch.setattr(main, "email_draft_store", EmailDraftStore())
    monkeypatch.setattr(main, "artifact_store", ArtifactStore())
    req = report_request(
        question="Gửi báo cáo này cho email test@example.com",
        conversation_context=[
            {
                "role": "assistant",
                "content": "Đã tạo báo cáo.",
                "answer_scope": "mes_report",
                "artifact_id": REPORT_ID,
                "artifact_type": "mes_report",
            }
        ],
    )

    with pytest.raises(main.GmailSenderError, match="hết hạn"):
        asyncio.run(main.handle_email_send_query(req, None))

    store = ArtifactStore()
    asyncio.run(
        store.put(
            StoredArtifact(
                id=REPORT_ID,
                kind="report_html",
                content="<html></html>",
                media_type="text/html; charset=utf-8",
                filename="report.html",
                meta={"title": "Report"},
                session_id=req.session_id,
                employee_id="another-employee",
            )
        )
    )
    monkeypatch.setattr(main, "artifact_store", store)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(main.handle_email_send_query(req, None))
    assert exc_info.value.status_code == 403
    assert sender.calls == []


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
