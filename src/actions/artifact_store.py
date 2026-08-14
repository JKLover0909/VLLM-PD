"""Kho artifact in-memory cho các Action Agent (báo cáo, sau này là draft).

Per-process, LRU + TTL giống ``query_response_cache`` trong ``main.py``.
Artifact chỉ phục vụ nút "Tải HTML" trên UI nên mất khi restart là chấp nhận
được ở giai đoạn demo; khi cần bền vững sẽ chuyển sang SQLite.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class StoredArtifact:
    id: str
    kind: str  # report_html | ...
    content: str
    media_type: str
    filename: str
    meta: dict[str, Any]
    session_id: str = ""
    employee_id: str = ""


class ArtifactStore:
    def __init__(self, *, max_items: int = 200, ttl_seconds: float = 6 * 3600):
        self._items: "OrderedDict[str, tuple[float, StoredArtifact]]" = OrderedDict()
        self._lock = asyncio.Lock()
        self.max_items = max(1, max_items)
        self.ttl_seconds = ttl_seconds

    async def put(self, artifact: StoredArtifact) -> None:
        expiry_at = time.monotonic() + self.ttl_seconds
        async with self._lock:
            self._items[artifact.id] = (expiry_at, artifact)
            self._items.move_to_end(artifact.id)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    async def get(self, artifact_id: str) -> Optional[StoredArtifact]:
        now = time.monotonic()
        async with self._lock:
            cached = self._items.get(artifact_id)
            if cached is None:
                return None
            expiry_at, artifact = cached
            if now > expiry_at:
                self._items.pop(artifact_id, None)
                return None
            self._items.move_to_end(artifact_id)
            return artifact
