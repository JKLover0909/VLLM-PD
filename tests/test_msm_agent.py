import httpx
import pytest
from fastapi.testclient import TestClient

from src.api import msm_agent
from src.api.msm_agent import (
    MsmAgentUnavailable,
    MsmTokenCache,
    load_config,
    to_meibook_event,
)


def test_load_config_fails_closed_when_a_key_is_missing():
    with pytest.raises(MsmAgentUnavailable):
        load_config({"MSM_AGENT_BASE": "http://x", "MSM_AGENT_USER": "u"})


def test_load_config_strips_trailing_slash():
    cfg = load_config({
        "MSM_AGENT_BASE": "http://192.168.10.171:8700/",
        "MSM_AGENT_USER": "agent-integration",
        "MSM_AGENT_PASS": "secret",
    })
    assert cfg.base_url == "http://192.168.10.171:8700"


@pytest.mark.asyncio
async def test_token_is_fetched_once_then_reused():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"token": "jwt-1"})

    cfg = load_config({
        "MSM_AGENT_BASE": "http://agent", "MSM_AGENT_USER": "u", "MSM_AGENT_PASS": "p",
    })
    cache = MsmTokenCache(cfg, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert await cache.token() == "jwt-1"
    assert await cache.token() == "jwt-1"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_invalidate_forces_a_new_login():
    replies = [{"token": "jwt-1"}, {"token": "jwt-2"}]
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=replies[len(calls) - 1])

    cfg = load_config({
        "MSM_AGENT_BASE": "http://agent", "MSM_AGENT_USER": "u", "MSM_AGENT_PASS": "p",
    })
    cache = MsmTokenCache(cfg, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert await cache.token() == "jwt-1"
    cache.invalidate()
    assert await cache.token() == "jwt-2"


def test_answer_becomes_a_token_event():
    assert to_meibook_event({"type": "answer", "text": "Xin chào"}) == {
        "type": "token", "content": "Xin chào",
    }


def test_done_carries_pending_and_views_untouched():
    result = {"answer": "ok", "pending": [{"action_id": "a1"}], "views": [{"kind": "table"}]}
    assert to_meibook_event({"type": "done", "result": result})["result"] == result


def test_progress_events_become_meta():
    assert to_meibook_event({"type": "tool_start", "tool": "search_machines", "label": "Đang tra máy"}) == {
        "type": "meta", "label": "Đang tra máy",
    }


def test_unknown_events_are_dropped():
    assert to_meibook_event({"type": "llm_wait", "step": "think"}) is None


def _client(handler, monkeypatch):
    cfg = msm_agent.load_config({
        "MSM_AGENT_BASE": "http://agent", "MSM_AGENT_USER": "u", "MSM_AGENT_PASS": "p",
    })
    monkeypatch.setattr(msm_agent, "_config", cfg)
    monkeypatch.setattr(msm_agent, "_http", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    msm_agent.SESSIONS = msm_agent.MsmSessions()
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(msm_agent.router)
    return TestClient(app)


def test_unconfigured_returns_503(monkeypatch):
    monkeypatch.setattr(msm_agent, "_config", None)
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(msm_agent.router)
    r = TestClient(app).post("/msm/query/stream", json={
        "session_id": "s1", "question": "máy nào quá hạn", "employee_id": "E1",
    })
    assert r.status_code == 503


def test_first_question_creates_one_thread_and_streams_tokens(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/agent/v1/threads":
            return httpx.Response(200, json={"thread_id": "u:conv-1"})
        if request.url.path.endswith("/messages/stream"):
            return httpx.Response(200, content=(
                b'data: {"type":"answer","text":"Co 3 may"}\n\n'
                b'data: {"type":"done","result":{"answer":"Co 3 may","pending":[]}}\n\n'
            ))
        return httpx.Response(200, json={"token": "jwt-1"})

    client = _client(handler, monkeypatch)
    with client.stream("POST", "/msm/query/stream", json={
        "session_id": "s1", "question": "máy nào quá hạn", "employee_id": "E1",
    }) as r:
        body = b"".join(r.iter_bytes()).decode()

    assert '"type": "token"' in body.replace('{"type":', '{"type":')
    assert '"content": "Co 3 may"' in body
    assert seen.count("/agent/v1/threads") == 1
    assert msm_agent.SESSIONS.thread("s1") == "u:conv-1"


def test_second_question_reuses_the_thread(monkeypatch):
    created = []

    def handler(request):
        if request.url.path == "/agent/v1/threads":
            created.append(1)
            return httpx.Response(200, json={"thread_id": "u:conv-1"})
        if request.url.path.endswith("/messages/stream"):
            return httpx.Response(200, content=b'data: {"type":"done","result":{}}\n\n')
        return httpx.Response(200, json={"token": "jwt-1"})

    client = _client(handler, monkeypatch)
    for _ in range(2):
        with client.stream("POST", "/msm/query/stream", json={
            "session_id": "s1", "question": "hỏi lại", "employee_id": "E1",
        }) as r:
            b"".join(r.iter_bytes())
    assert len(created) == 1


def test_confirm_without_a_thread_returns_404(monkeypatch):
    client = _client(lambda request: httpx.Response(200, json={"token": "x"}), monkeypatch)
    r = client.post("/msm/actions/abc/confirm", json={"session_id": "nope", "employee_id": "E1"})
    assert r.status_code == 404
