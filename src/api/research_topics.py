"""Registry các nhóm chủ đề tài liệu cho chế độ Research.

Đọc ``config/research_topics.json`` một lần lúc import và cache lại. Mỗi topic
map một ``category`` trong metadata của collection DocJP; retrieval sẽ filter
theo category đó. Topic id ``all`` (khi ``allow_all=true``) tìm trên toàn bộ
corpus không filter.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from src.api import config

logger = logging.getLogger(__name__)

ALL_TOPIC_ID = "all"

_registry_cache: Optional[Dict[str, Any]] = None


def load_research_topic_config(*, force_reload: bool = False) -> Dict[str, Any]:
    """Load and cache the research topic registry; empty registry on error."""
    global _registry_cache
    if _registry_cache is not None and not force_reload:
        return _registry_cache

    empty: Dict[str, Any] = {
        "knowledge_base": "",
        "collection": config.DOCJP_COLLECTION_NAME,
        "session_id": config.DOCJP_SESSION_ID,
        "default_topic": "",
        "allow_all": False,
        "topics": [],
    }
    try:
        raw = json.loads(config.RESEARCH_TOPICS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning(
            "Research topics config not found at %s", config.RESEARCH_TOPICS_PATH
        )
        _registry_cache = empty
        return _registry_cache
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load research topics config: %s", exc)
        _registry_cache = empty
        return _registry_cache

    topics = [item for item in raw.get("topics", []) if item.get("id")]
    _registry_cache = {
        "knowledge_base": raw.get("knowledge_base", ""),
        "collection": raw.get("collection", config.DOCJP_COLLECTION_NAME),
        "session_id": raw.get("session_id", config.DOCJP_SESSION_ID),
        "default_topic": raw.get("default_topic", ""),
        "allow_all": bool(raw.get("allow_all", False)),
        "topics": topics,
    }
    return _registry_cache


def research_topic_ids() -> List[str]:
    registry = load_research_topic_config()
    ids = [topic["id"] for topic in registry["topics"]]
    if registry["allow_all"]:
        ids.append(ALL_TOPIC_ID)
    return ids


def research_topic_by_id(topic_id: str) -> Optional[Dict[str, Any]]:
    registry = load_research_topic_config()
    if topic_id == ALL_TOPIC_ID and registry["allow_all"]:
        return {"id": ALL_TOPIC_ID, "category": None}
    for topic in registry["topics"]:
        if topic["id"] == topic_id:
            return topic
    return None


def validate_research_topic(topic_id: Optional[str]) -> Optional[str]:
    """Return the topic id if valid; None for missing/unknown ids.

    Unknown ids are treated as "no topic" (legacy behaviour) rather than an
    error so old clients keep working.
    """
    if not topic_id:
        return None
    if research_topic_by_id(topic_id) is not None:
        return topic_id
    logger.warning("Ignoring unknown research topic id: %r", topic_id)
    return None


def research_topic_category(topic_id: Optional[str]) -> Optional[str]:
    """Category filter value for a validated topic id (None for `all`)."""
    if not topic_id:
        return None
    topic = research_topic_by_id(topic_id)
    if topic is None:
        return None
    return topic.get("category") or None
