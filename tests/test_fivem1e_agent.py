import httpx
import pytest
from fastapi.testclient import TestClient

from src.api import fivem1e_agent
from src.api.fivem1e_agent import (
    Fivem1eAgentUnavailable,
    Fivem1eTokenCache,
    load_config,
    to_meibook_event,
)


def test_load_config_fails_closed_when_a_key_is_missing():
    with pytest.raises(Fivem1eAgentUnavailable):
        load_config({"FIVEM1E_AGENT_BASE": "http://x", "FIVEM1E_AGENT_USER": "u"})


def test_load_config_strips_trailing_slash():
    cfg = load_config({
        "FIVEM1E_AGENT_BASE": "http://192.168.10.171:8703/",
        "FIVEM1E_AGENT_USER": "agent-integration",
        "FIVEM1E_AGENT_PASS": "secret",
    })
    assert cfg.base_url == "http://192.168.10.171:8703"


@pytest.mark.asyncio
async def test_token_is_fetched_once_then_reused():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"token": "jwt-5m1e"})

    cfg = load_config({
        "FIVEM1E_AGENT_BASE": "http://agent", "FIVEM1E_AGENT_USER": "u", "FIVEM1E_AGENT_PASS": "p",
    })
    cache = Fivem1eTokenCache(cfg, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert await cache.token() == "jwt-5m1e"
    assert await cache.token() == "jwt-5m1e"
    assert len(calls) == 1


def test_to_meibook_event():
    assert to_meibook_event({"type": "answer", "text": "hello"}) == {"type": "token", "content": "hello"}
    assert to_meibook_event({"type": "token", "content": "world"}) == {"type": "token", "content": "world"}
    assert to_meibook_event({"type": "plan"}) == {"type": "meta", "label": "Lập kế hoạch tra cứu"}
    assert to_meibook_event({"type": "tool_start", "tool": "search_reports"}) == {"type": "meta", "label": "Đang tra cứu: search_reports"}
    assert to_meibook_event({"type": "done", "result": {"k": "v"}}) == {"type": "done", "result": {"k": "v"}}
