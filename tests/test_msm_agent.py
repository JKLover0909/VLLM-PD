import httpx
import pytest

from src.api.msm_agent import MsmAgentUnavailable, MsmTokenCache, load_config


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
