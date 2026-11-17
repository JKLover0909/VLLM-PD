"""Every LLM alias routes to host Ollama, so each LiteLLM service must reach it.

Production owns the socat bridge (`ollama-proxy`); Dev reuses it. Dropping the
bridge or `host.docker.internal` silently sends every request to the Azure
fallback, which is what this guards against.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
OLLAMA_BASE = "http://host.docker.internal:11435"


def _services(name):
    with (ROOT / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)["services"]


@pytest.mark.parametrize("compose", ["docker-compose.web.yml", "docker-compose.dev.yml"])
def test_litellm_can_reach_host_ollama(compose):
    litellm = _services(compose)["litellm"]
    assert litellm["environment"]["OLLAMA_API_BASE"] == OLLAMA_BASE
    assert "host.docker.internal:host-gateway" in litellm["extra_hosts"]


def test_production_runs_the_shared_ollama_bridge():
    services = _services("docker-compose.web.yml")
    proxy = services["ollama-proxy"]
    assert proxy["network_mode"] == "host"
    assert "TCP-LISTEN:11435,bind=172.17.0.1,fork,reuseaddr" in proxy["command"]
    assert "TCP:127.0.0.1:11434" in proxy["command"]
    assert "ollama-proxy" in services["litellm"]["depends_on"]


def test_dev_does_not_start_a_second_bridge():
    # A second socat on 11435 would collide with Production's bridge.
    assert "ollama-proxy" not in _services("docker-compose.dev.yml")
