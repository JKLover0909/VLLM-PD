"""Small, configurable web-search adapter for MKAC fallback questions."""

import logging
import os
from typing import Any, Dict, List

from src.rag.parser import TextChunk
from src.rag.vector_store import SearchResult

logger = logging.getLogger(__name__)


class WebSearcher:
    """Search the public web without making the RAG pipeline provider-specific."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        max_results: int | None = None,
        timeout: int | None = None,
        region: str | None = None,
        context: str | None = None,
    ):
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("MKAC_WEB_SEARCH_ENABLED", "true").lower()
            in {"1", "true", "yes", "on"}
        )
        self.max_results = max_results or int(
            os.getenv("MKAC_WEB_SEARCH_MAX_RESULTS", "5")
        )
        self.timeout = timeout or int(os.getenv("MKAC_WEB_SEARCH_TIMEOUT", "10"))
        self.region = region or os.getenv("MKAC_WEB_SEARCH_REGION", "vn-vi")
        self.context = context or os.getenv("MKAC_WEB_SEARCH_CONTEXT", "MKAC công ty")

    def search(self, question: str) -> List[SearchResult]:
        if not self.enabled:
            return []

        try:
            from ddgs import DDGS
        except ImportError:
            logger.warning(
                "MKAC web search is enabled but package 'ddgs' is not installed."
            )
            return []

        query = f'"{self.context}" {question}'.strip()
        try:
            raw_results = DDGS(timeout=self.timeout).text(
                query,
                region=self.region,
                safesearch="moderate",
                max_results=self.max_results,
                backend="auto",
            )
        except Exception:
            logger.exception("MKAC web search failed for query: %s", query)
            return []

        results: List[SearchResult] = []
        seen_urls = set()
        for rank, item in enumerate(raw_results or [], 1):
            normalized = self._normalize_result(item)
            if not normalized or normalized["url"] in seen_urls:
                continue
            seen_urls.add(normalized["url"])
            chunk = TextChunk(
                text=normalized["snippet"],
                source_file=normalized["title"],
                page_number=0,
                chunk_index=rank - 1,
                content_type="web",
                metadata={
                    "title": normalized["title"],
                    "url": normalized["url"],
                    "search_query": query,
                    "source": "web-search",
                },
            )
            results.append(SearchResult(chunk=chunk, score=max(0.5, 1 - rank * 0.1)))

        logger.info(
            "Found %s web results for MKAC fallback query: '%s'",
            len(results),
            query[:100],
        )
        return results

    @staticmethod
    def _normalize_result(item: Dict[str, Any]) -> Dict[str, str] | None:
        title = str(item.get("title") or "").strip()
        url = str(item.get("href") or item.get("url") or "").strip()
        snippet = str(item.get("body") or item.get("snippet") or "").strip()
        if not title or not url or not snippet:
            return None
        if not url.startswith(("http://", "https://")):
            return None
        return {"title": title, "url": url, "snippet": snippet}
