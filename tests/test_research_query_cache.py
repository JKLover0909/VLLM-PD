"""Tests for Research mode being included in the runtime query response cache."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api import config
from src.api.helpers import query_cache_key
from src.api.schemas import QueryRequest


def _research_request(**overrides) -> QueryRequest:
    fields = {
        "session_id": "s1",
        "question": "3rdWATCH đăng nhập thế nào?",
        "mode": "research",
        "ui_language": "vi",
        "model": "auto",
        "research_scope": "topic",
        "research_topic": "legal_compliance",
    }
    fields.update(overrides)
    return QueryRequest(**fields)


def test_research_mode_is_cacheable(monkeypatch):
    monkeypatch.setattr(config, "QUERY_RESPONSE_CACHE_SIZE", 256)
    monkeypatch.setattr(config, "RESEARCH_QUERY_CACHE_TTL_SECONDS", 3600)
    key = query_cache_key(_research_request())
    assert key is not None


def test_research_cache_key_differs_by_topic(monkeypatch):
    monkeypatch.setattr(config, "QUERY_RESPONSE_CACHE_SIZE", 256)
    monkeypatch.setattr(config, "RESEARCH_QUERY_CACHE_TTL_SECONDS", 3600)
    key_legal = query_cache_key(_research_request(research_topic="legal_compliance"))
    key_accounting = query_cache_key(_research_request(research_topic="accounting"))
    assert key_legal != key_accounting


def test_research_cache_key_same_question_same_topic_is_stable(monkeypatch):
    monkeypatch.setattr(config, "QUERY_RESPONSE_CACHE_SIZE", 256)
    monkeypatch.setattr(config, "RESEARCH_QUERY_CACHE_TTL_SECONDS", 3600)
    key_a = query_cache_key(_research_request())
    key_b = query_cache_key(_research_request())
    assert key_a == key_b


def test_research_cache_disabled_when_ttl_zero(monkeypatch):
    monkeypatch.setattr(config, "QUERY_RESPONSE_CACHE_SIZE", 256)
    monkeypatch.setattr(config, "RESEARCH_QUERY_CACHE_TTL_SECONDS", 0)
    assert query_cache_key(_research_request()) is None


def test_research_cache_disabled_when_cache_size_zero(monkeypatch):
    monkeypatch.setattr(config, "QUERY_RESPONSE_CACHE_SIZE", 0)
    monkeypatch.setattr(config, "RESEARCH_QUERY_CACHE_TTL_SECONDS", 3600)
    assert query_cache_key(_research_request()) is None


def test_uploaded_research_session_never_uses_shared_response_cache(monkeypatch):
    monkeypatch.setattr(config, "QUERY_RESPONSE_CACHE_SIZE", 256)
    monkeypatch.setattr(config, "RESEARCH_QUERY_CACHE_TTL_SECONDS", 3600)

    assert (
        query_cache_key(
            _research_request(
                session_id="private-session",
                research_scope="upload",
                research_topic=None,
            )
        )
        is None
    )
