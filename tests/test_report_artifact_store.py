import asyncio

from src.actions.artifact_store import ArtifactStore, StoredArtifact


def artifact(artifact_id: str, *, kind: str = "report_html") -> StoredArtifact:
    return StoredArtifact(
        id=artifact_id,
        kind=kind,
        content=f"<html>{artifact_id}</html>",
        media_type="text/html; charset=utf-8",
        filename=f"{artifact_id}.html",
        meta={"title": artifact_id},
    )


def test_artifact_store_put_get_roundtrip():
    async def scenario():
        store = ArtifactStore(max_items=2, ttl_seconds=60)
        expected = artifact("report-1")
        await store.put(expected)
        return await store.get("report-1"), await store.get("missing")

    stored, missing = asyncio.run(scenario())
    assert stored == artifact("report-1")
    assert missing is None


def test_artifact_store_expires_items(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "src.actions.artifact_store.time.monotonic",
        lambda: now[0],
    )

    async def scenario():
        store = ArtifactStore(ttl_seconds=5)
        await store.put(artifact("expired"))
        now[0] = 106.0
        return await store.get("expired")

    assert asyncio.run(scenario()) is None


def test_artifact_store_evicts_least_recently_used_item():
    async def scenario():
        store = ArtifactStore(max_items=2, ttl_seconds=60)
        await store.put(artifact("first"))
        await store.put(artifact("second"))
        assert await store.get("first") is not None  # refresh first in LRU order
        await store.put(artifact("third"))
        return (
            await store.get("first"),
            await store.get("second"),
            await store.get("third"),
        )

    first, second, third = asyncio.run(scenario())
    assert first is not None
    assert second is None
    assert third is not None


def test_artifact_store_replaces_existing_artifact():
    async def scenario():
        store = ArtifactStore(max_items=1, ttl_seconds=60)
        await store.put(artifact("same"))
        updated = StoredArtifact(
            id="same",
            kind="report_html",
            content="updated",
            media_type="text/html; charset=utf-8",
            filename="updated.html",
            meta={"title": "updated"},
        )
        await store.put(updated)
        return await store.get("same")

    stored = asyncio.run(scenario())
    assert stored is not None
    assert stored.content == "updated"
    assert stored.filename == "updated.html"
