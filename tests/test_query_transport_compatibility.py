import asyncio
from types import SimpleNamespace

from starlette.requests import Request

from src.api import main
from src.api.schemas import QueryRequest


SESSION_ID = "00000000-0000-4000-8000-000000000777"


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


async def no_rate_limit(*args, **kwargs):
    return None


async def no_wait(*args, **kwargs):
    return None


def protected_request(question):
    return QueryRequest(
        session_id=SESSION_ID,
        question=question,
        mode="mkac",
        model="local",
        ui_language="vi",
        employee_id="000000",
    )


def install_rest_test_defaults(monkeypatch):
    monkeypatch.setattr(main, "enforce_rate_limit", no_rate_limit)
    monkeypatch.setattr(main, "wait_for_min_query_latency", no_wait)
    monkeypatch.setattr(main, "authorize_query", lambda req: None)
    monkeypatch.setattr(main, "safety_guard_response", lambda req: None)
    monkeypatch.setattr(main, "research_cached_query_response", lambda req: None)
    monkeypatch.setattr(main, "get_cached_query_response", no_cached_response)
    monkeypatch.setattr(main, "query_response_cache", main.OrderedDict())


async def no_cached_response(cache_key):
    return None


def test_rest_prepared_mkac_answer_bypasses_generic_router(monkeypatch):
    install_rest_test_defaults(monkeypatch)
    prepared = main.QueryResponse(
        answer="Câu trả lời MKAC đã kiểm chứng.",
        sources=[{"file": "policy.pdf", "page": 1}],
        session_id=SESSION_ID,
        model="auto-model",
        mode="mkac",
        answer_scope="mkac",
    )
    monkeypatch.setattr(main, "prepared_query_response", lambda req: prepared)

    async def fail_route(*args, **kwargs):  # pragma: no cover
        raise AssertionError("Prepared MKAC answer must not call the generic router")

    monkeypatch.setattr(main, "route_query_outcome", fail_route)

    response = asyncio.run(
        main.query_documents(
            protected_request("Quy định làm thêm giờ ở MKAC như thế nào?"),
            request_context(),
        )
    )

    assert response.answer == prepared.answer
    assert response.sources == prepared.sources
    assert response.answer_scope == "mkac"


def test_rest_calendar_answer_bypasses_generic_router(monkeypatch):
    install_rest_test_defaults(monkeypatch)
    monkeypatch.setattr(main, "prepared_query_response", lambda req: None)
    monkeypatch.setattr(main, "employee_context_for_query", lambda *args: None)
    monkeypatch.setattr(main, "employee_directory_query_response", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "handle_email_send_query", return_none)
    monkeypatch.setattr(main, "handle_report_query", return_none)
    monkeypatch.setattr(main, "translation_service", None)
    calls = []

    async def calendar_handler(req, employee):
        calls.append((req.question, employee))
        return main.QueryResponse(
            answer="Đã chuẩn bị bản nháp lịch họp.",
            sources=[],
            session_id=req.session_id,
            model="calendar-agent",
            mode=req.mode,
            answer_scope="calendar_draft",
        )

    async def fail_route(*args, **kwargs):  # pragma: no cover
        raise AssertionError("Calendar REST action must not call the generic router")

    monkeypatch.setattr(main, "handle_calendar_action_query", calendar_handler)
    monkeypatch.setattr(main, "route_query_outcome", fail_route)

    response = asyncio.run(
        main.query_documents(
            protected_request("Đặt phòng họp 1 lúc 9 giờ sáng mai"),
            request_context(),
        )
    )

    assert response.answer_scope == "calendar_draft"
    assert response.model == "calendar-agent"
    assert calls == [("Đặt phòng họp 1 lúc 9 giờ sáng mai", None)]


async def return_none(*args, **kwargs):
    return None
