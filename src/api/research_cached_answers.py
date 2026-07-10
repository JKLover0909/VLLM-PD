"""Cached answers for predefined Research quick prompts.

Các câu hỏi gợi ý trong chế độ Research là dữ liệu demo ổn định. Module này
đọc manifest JSON, match theo topic + ngôn ngữ + câu hỏi đã normalize và trả
về ``QueryResponse`` sẵn có, không gọi retrieval hoặc LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from src.api import config
from src.api.helpers import normalize_prepared_question
from src.api.schemas import QueryRequest, QueryResponse
from src.api.research_topics import validate_research_topic

logger = logging.getLogger(__name__)

_cache_manifest: Optional[Dict[str, Any]] = None
_answer_index: Optional[Dict[tuple[str, str, str], Dict[str, Any]]] = None


def load_research_cached_answers(*, force_reload: bool = False) -> Dict[str, Any]:
    """Load and cache ``config/research_cached_answers.json``."""
    global _cache_manifest, _answer_index
    if _cache_manifest is not None and not force_reload:
        return _cache_manifest

    empty: Dict[str, Any] = {"version": 1, "answers": []}
    try:
        raw = json.loads(
            config.RESEARCH_CACHED_ANSWERS_PATH.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        logger.info(
            "Research cached answers config not found at %s",
            config.RESEARCH_CACHED_ANSWERS_PATH,
        )
        _cache_manifest = empty
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot load research cached answers: %s", exc)
        _cache_manifest = empty
    else:
        answers = [item for item in raw.get("answers", []) if item.get("research_topic")]
        _cache_manifest = {
            "version": raw.get("version", 1),
            "description": raw.get("description", ""),
            "answers": answers,
        }

    _answer_index = None
    return _cache_manifest


def _build_answer_index() -> Dict[tuple[str, str, str], Dict[str, Any]]:
    global _answer_index
    if _answer_index is not None:
        return _answer_index

    manifest = load_research_cached_answers()
    index: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for item in manifest.get("answers", []):
        topic = validate_research_topic(str(item.get("research_topic") or ""))
        if not topic:
            continue
        for language in ("vi", "ja"):
            question_field = f"question_{language}"
            aliases_field = f"aliases_{language}"
            answer_field = f"answer_{language}"
            if not item.get(answer_field):
                continue
            candidates = [str(item.get(question_field) or "")]
            aliases = item.get(aliases_field) or []
            if isinstance(aliases, list):
                candidates.extend(str(alias) for alias in aliases)
            for candidate in candidates:
                normalized = normalize_prepared_question(candidate)
                if normalized:
                    index[(topic, language, normalized)] = item

    _answer_index = index
    return _answer_index


def research_cached_query_response(req: QueryRequest) -> Optional[QueryResponse]:
    """Return a cached Research quick-prompt answer if available."""
    if req.mode != "research":
        return None
    if req.research_scope == "upload":
        return None

    topic = validate_research_topic(req.research_topic)
    if not topic:
        return None

    language = req.ui_language if req.ui_language in {"vi", "ja"} else "vi"
    normalized_question = normalize_prepared_question(req.question)
    item = _build_answer_index().get((topic, language, normalized_question))
    if item is None:
        return None

    answer = item.get(f"answer_{language}") or item.get("answer_vi") or ""
    if not answer:
        return None

    sources = item.get("sources") or []
    if not isinstance(sources, list):
        sources = []
    sources = [dict(source) for source in sources]
    for source in sources:
        source.setdefault("source_scope", "topic")

    return QueryResponse(
        answer=answer,
        sources=sources,
        session_id=req.session_id,
        model=item.get("model") or "auto-model",
        mode="research",
        answer_scope=item.get("answer_scope") or "research",
    )


def research_cached_answer_metadata(req: QueryRequest) -> Dict[str, Any]:
    """Optional metadata for diagnostic endpoints/tests."""
    if req.research_scope == "upload":
        return {}
    topic = validate_research_topic(req.research_topic)
    if not topic:
        return {}
    language = req.ui_language if req.ui_language in {"vi", "ja"} else "vi"
    normalized_question = normalize_prepared_question(req.question)
    item = _build_answer_index().get((topic, language, normalized_question))
    if item is None:
        return {}
    return {
        "id": item.get("id", ""),
        "citations": item.get("citations", []),
        "metadata": item.get("metadata", {}),
    }
