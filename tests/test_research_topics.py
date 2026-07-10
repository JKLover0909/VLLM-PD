"""Tests for the Research topic registry and query schema."""

import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api import research_topics as topics_module
from src.api.research_topics import (
    ALL_TOPIC_ID,
    load_research_topic_config,
    research_topic_by_id,
    research_topic_category,
    research_topic_ids,
    validate_research_topic,
)
from src.api.schemas import QueryRequest


@pytest.fixture(autouse=True)
def fresh_registry():
    """Reload the registry from disk for every test."""
    topics_module._registry_cache = None
    yield
    topics_module._registry_cache = None


def test_registry_loads_four_topics():
    registry = load_research_topic_config()
    ids = [topic["id"] for topic in registry["topics"]]
    assert ids == [
        "information_systems",
        "legal_compliance",
        "accounting",
        "general_affairs",
    ]
    assert registry["collection"] == "docjp_knowledge"
    assert registry["session_id"] == "docjp"
    assert registry["allow_all"] is True


def test_topic_ids_include_all_when_allowed():
    assert ALL_TOPIC_ID in research_topic_ids()


def test_every_topic_has_bilingual_labels():
    registry = load_research_topic_config()
    for topic in registry["topics"]:
        assert topic.get("label_vi"), topic["id"]
        assert topic.get("label_ja"), topic["id"]


def test_validate_known_topic():
    assert validate_research_topic("accounting") == "accounting"
    assert validate_research_topic("all") == "all"


def test_validate_unknown_topic_returns_none():
    assert validate_research_topic("nonexistent") is None
    assert validate_research_topic("") is None
    assert validate_research_topic(None) is None


def test_category_mapping():
    assert research_topic_category("legal_compliance") == "legal_compliance"
    # "all" searches the whole corpus without a category filter.
    assert research_topic_category("all") is None
    assert research_topic_category(None) is None


def test_topic_by_id_all_requires_allow_all(tmp_path, monkeypatch):
    config_path = tmp_path / "research_topics.json"
    config_path.write_text(
        json.dumps(
            {
                "collection": "docjp_knowledge",
                "session_id": "docjp",
                "allow_all": False,
                "topics": [{"id": "accounting", "category": "accounting"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(topics_module.config, "RESEARCH_TOPICS_PATH", config_path)
    topics_module._registry_cache = None
    assert research_topic_by_id("all") is None
    assert validate_research_topic("all") is None


def test_missing_config_yields_empty_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(
        topics_module.config,
        "RESEARCH_TOPICS_PATH",
        tmp_path / "does_not_exist.json",
    )
    topics_module._registry_cache = None
    registry = load_research_topic_config()
    assert registry["topics"] == []
    assert validate_research_topic("accounting") is None


def test_query_request_accepts_research_topic():
    req = QueryRequest(
        session_id="s",
        question="q",
        mode="research",
        research_topic="accounting",
    )
    assert req.research_topic == "accounting"


def test_query_request_research_topic_defaults_to_none():
    req = QueryRequest(session_id="s", question="q", mode="research")
    assert req.research_topic is None
    assert req.research_scope is None


def test_query_request_accepts_explicit_upload_scope():
    req = QueryRequest(
        session_id="s",
        question="q",
        mode="research",
        research_scope="upload",
    )
    assert req.research_scope == "upload"
    assert req.research_topic is None
