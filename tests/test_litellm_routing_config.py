from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "litellm_config.yaml"
LAN_MODEL = "openai/qwen2.5-coder:14b"
NGROK_MODEL = (
    "openai//home/jkl0909/models/qwen2.5-coder-14b/"
    "Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf"
)
AZURE_MODEL = "openai/grok-4-20-reasoning"
OPENAI_MODEL = "openai/gpt-5.4-mini"
CLOUD_MODELS = {
    "azure-chat-fallback",
    "azure-small-fallback",
    "azure-coder-fallback",
    "openai-chat-fallback",
    "openai-small-fallback",
    "openai-coder-fallback",
}
ROLE_FALLBACKS = {
    "auto-model": [
        "local-qwen-chat-ngrok",
        "azure-chat-fallback",
        "openai-chat-fallback",
    ],
    "local-qwen-chat": [
        "local-qwen-chat-ngrok",
        "azure-chat-fallback",
        "openai-chat-fallback",
    ],
    "local-qwen-small": [
        "local-qwen-chat",
        "azure-small-fallback",
        "openai-small-fallback",
    ],
    "local-qwen-coder": [
        "local-qwen-coder-ngrok",
        "azure-coder-fallback",
        "openai-coder-fallback",
    ],
    "coding-model": [
        "local-qwen-coder-ngrok",
        "azure-coder-fallback",
        "openai-coder-fallback",
    ],
}


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


def test_role_specific_cloud_aliases_do_not_reuse_openai_model_alias():
    models = _models_by_name(_load_config())

    assert CLOUD_MODELS <= models.keys()
    assert "openai-model" not in models

    for model_name in (
        "azure-chat-fallback",
        "azure-small-fallback",
        "azure-coder-fallback",
    ):
        params = models[model_name]["litellm_params"]
        assert params["model"] == AZURE_MODEL
        assert params["api_key"] == "os.environ/AZURE_OPENAI_API_KEY"
        assert params["api_base"] == "os.environ/AZURE_OPENAI_ENDPOINT"

    for model_name in (
        "openai-chat-fallback",
        "openai-small-fallback",
        "openai-coder-fallback",
    ):
        params = models[model_name]["litellm_params"]
        assert params["model"] == OPENAI_MODEL
        assert params["api_key"] == "os.environ/OPENAI_API_KEY"


def test_fallbacks_preserve_role_and_prefer_azure_before_openai():
    config = _load_config()
    models = _models_by_name(config)
    fallbacks = _fallbacks_by_name(config)

    for model_name, expected in ROLE_FALLBACKS.items():
        assert fallbacks[model_name] == expected
        assert all(target in models for target in expected)

    assert "local-qwen-chat" not in fallbacks["local-qwen-coder"]
    assert "local-qwen-chat" not in fallbacks["coding-model"]


def test_fallback_graph_has_no_cycles():
    fallbacks = _fallbacks_by_name(_load_config())

    def has_cycle(node, visiting, visited):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for target in fallbacks.get(node, []):
            if has_cycle(target, visiting, visited):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    visited = set()
    assert not any(has_cycle(node, set(), visited) for node in fallbacks)
