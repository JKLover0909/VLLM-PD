"""Focused offline regression coverage for model routing contracts."""

import pytest
from pydantic import ValidationError

from src.api.schemas import QueryRequest
from src.integrations.mes_config import (
    prefer_template_answers,
    resolve_model as resolve_mes_model,
    resolve_sql_agent_model,
)
from src.rag.rag_pipeline import RAGPipeline


@pytest.fixture
def pipeline():
    # These resolver methods do not initialize retrieval or external services.
    return object.__new__(RAGPipeline)


@pytest.mark.parametrize("ui_model", ["auto", "local", "openai"])
def test_rag_image_requests_stay_on_primary_for_capability_fallback(pipeline, ui_model):
    assert pipeline._resolve_model(ui_model, has_images=True) == {
        "auto": "auto-model",
        "local": "local-qwen-chat",
        "openai": "auto-model",
    }[ui_model]


@pytest.mark.parametrize(
    ("ui_model", "expected"),
    [
        ("auto", "auto-model"),
        ("local", "local-qwen-chat"),
        ("openai", "auto-model"),
    ],
)
def test_rag_text_requests_resolve_ui_models(pipeline, ui_model, expected):
    assert pipeline._resolve_model(ui_model) == expected


def test_removed_grok_is_rejected_by_schema_and_rag_resolver(pipeline):
    with pytest.raises(ValidationError):
        QueryRequest(session_id="session", question="question", model="grok")
    with pytest.raises(ValueError, match="Unsupported model option"):
        pipeline._resolve_model("grok")


def test_mes_alias_routing_is_preserved():
    assert resolve_mes_model("auto") == "auto-model"
    assert resolve_mes_model("local") == "local-qwen-chat"
    assert resolve_mes_model("openai") == "auto-model"
    assert resolve_sql_agent_model("local") == "local-qwen-coder"


def test_mes_deterministic_template_preference_remains_local_only(monkeypatch):
    monkeypatch.setenv("MES_TEMPLATE_ANSWERS_FOR_LOCAL", "true")
    assert prefer_template_answers("auto-model") is True
    assert prefer_template_answers("local-qwen-chat") is True
    assert prefer_template_answers("auto-model") is True


def test_provider_options_do_not_override_litellm_upstream_body(pipeline):
    for model in (
        "auto-model",
        "local-qwen-chat",
        "local-qwen-coder",
        "coding-model",
        "azure-chat-fallback",
    ):
        assert pipeline._provider_options(model) == {}


def test_mes_sql_planner_uses_shared_remote_qwen_alias():
    assert resolve_sql_agent_model("openai") == "local-qwen-coder"
