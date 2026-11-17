from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "litellm_config.yaml"
LOCAL_MODEL = "os.environ/REMOTE_LLM_MODEL"
LOCAL_ALIASES = (
    "auto-model",
    "local-qwen-chat",
    "local-qwen-coder",
    "coding-model",
)
AZURE_MODEL = "openai/gpt-5.4-mini"
CLOUD_MODELS = {
    "azure-chat-fallback",
    "azure-coder-fallback",
}
ROLE_FALLBACKS = {
    "auto-model": ["azure-chat-fallback"],
    "local-qwen-chat": ["azure-chat-fallback"],
    "local-qwen-coder": ["azure-coder-fallback"],
    "coding-model": ["azure-coder-fallback"],
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


def test_all_role_aliases_use_the_same_remote_qwen_primary():
    models = _models_by_name(_load_config())

    for model_name in LOCAL_ALIASES:
        params = models[model_name]["litellm_params"]
        assert params["model"] == LOCAL_MODEL
        assert params["custom_llm_provider"] == "openai"
        assert params["api_base"] == "os.environ/REMOTE_LLM_API_BASE"
        assert params["api_key"] == "os.environ/REMOTE_LLM_API_KEY"
        assert params["allowed_openai_params"] == ["extra_body"]
        assert params["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert params["max_parallel_requests"] == 1
        assert models[model_name]["model_info"]["max_input_tokens"] == 8192


def test_remote_primary_requires_a_protected_configured_endpoint():
    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    assert "192.84.106.81" not in config_text
    assert "http://" not in config_text
    assert "REMOTE_LLM_API_BASE" in config_text
    assert "REMOTE_LLM_API_KEY" in config_text
    assert "REMOTE_LLM_MODEL" in config_text


def test_config_has_no_ollama_or_legacy_cloud_primary_routes():
    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    models = _models_by_name(_load_config())

    assert "OLLAMA_API_BASE" not in config_text
    assert "ollama_chat" not in config_text
    assert "CHAT_LAN_" not in config_text
    assert "swift" not in config_text.lower()
    assert "local-hardware-summary" not in models
    assert "local-qwen-small" not in models
    assert "azure-small-fallback" not in models
    assert "grok-model" not in models
    primary_models = {
        model["litellm_params"]["model"]
        for name, model in models.items()
        if name not in CLOUD_MODELS
    }
    assert primary_models == {LOCAL_MODEL}


def test_primary_has_no_ollama_reasoning_parameters():
    models = _models_by_name(_load_config())
    for model_name in LOCAL_ALIASES:
        params = models[model_name]["litellm_params"]
        assert "think" not in params
        assert "reasoning_effort" not in params
        assert params["allowed_openai_params"] == ["extra_body"]


def test_azure_is_the_only_cloud_fallback_provider():
    models = _models_by_name(_load_config())

    assert CLOUD_MODELS <= models.keys()
    assert not any(name.startswith("openai-") for name in models)

    for model_name in CLOUD_MODELS:
        params = models[model_name]["litellm_params"]
        assert params["model"] == AZURE_MODEL
        assert params["api_key"] == "os.environ/AZURE_OPENAI_API_KEY"
        assert params["api_base"] == "os.environ/AZURE_OPENAI_ENDPOINT"


def test_fallbacks_are_sequential_role_specific_azure_only():
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
