from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "litellm_config.yaml"
LAN_MODEL = "openai/qwen2.5-coder:14b"
NGROK_MODEL = (
    "openai//home/jkl0909/models/qwen2.5-coder-14b/"
    "Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf"
)
CODER_FALLBACKS = [
    "local-qwen-coder-ngrok",
    "local-qwen-chat",
    "openai-model",
]


def _load_config():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _models_by_name(config):
    return {model["model_name"]: model for model in config["model_list"]}


def _fallbacks_by_name(config):
    return {
        model_name: fallback_names
        for fallback in config["router_settings"]["fallbacks"]
        for model_name, fallback_names in fallback.items()
    }


def test_coder_aliases_use_lan_primary_and_q5_ngrok_fallback():
    models = _models_by_name(_load_config())

    for model_name in ("local-qwen-coder", "coding-model"):
        params = models[model_name]["litellm_params"]
        assert params["model"] == LAN_MODEL
        assert params["api_base"] == "os.environ/QWEN_CODER_LAN_API_BASE"
        assert params["api_key"] == "os.environ/QWEN_CODER_LAN_API_KEY"
        assert "headers" not in params

    ngrok_params = models["local-qwen-coder-ngrok"]["litellm_params"]
    assert ngrok_params["model"] == NGROK_MODEL
    assert ngrok_params["api_base"] == "os.environ/QWEN_CODER_NGROK_API_BASE"
    assert ngrok_params["api_key"] == "os.environ/QWEN_CODER_NGROK_API_KEY"
    assert ngrok_params["headers"] == {"ngrok-skip-browser-warning": "true"}


def test_coder_fallbacks_prefer_q5_ngrok_without_cycles():
    config = _load_config()
    fallbacks = _fallbacks_by_name(config)

    assert fallbacks["local-qwen-coder"] == CODER_FALLBACKS
    assert fallbacks["coding-model"] == CODER_FALLBACKS
    assert "local-qwen-coder-ngrok" not in fallbacks

    coder_nodes = {
        "local-qwen-coder",
        "coding-model",
        "local-qwen-coder-ngrok",
    }
    coder_edges = {
        source: [target for target in targets if target in coder_nodes]
        for source, targets in fallbacks.items()
        if source in coder_nodes
    }

    def has_cycle(node, visiting, visited):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for target in coder_edges.get(node, []):
            if has_cycle(target, visiting, visited):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    visited = set()
    assert not any(has_cycle(node, set(), visited) for node in coder_nodes)
