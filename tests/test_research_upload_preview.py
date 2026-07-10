"""Preview routing for private Research upload sessions."""

import asyncio
import os
from pathlib import Path

os.environ["ENABLE_AGENT"] = "false"

from fastapi.responses import FileResponse

from src.api import main


class PreviewStore:
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.calls = []

    def get_page_image_path(self, session_id, filename, page_number):
        self.calls.append((session_id, filename, page_number))
        return str(self.image_path)


def test_uploaded_research_preview_uses_session_vector_store(tmp_path, monkeypatch):
    session_id = "00000000-0000-4000-8000-000000000123"
    image_path = tmp_path / session_id / "_pages" / "manual" / "page-0001.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    upload_store = PreviewStore(image_path)
    topic_store = PreviewStore(image_path)
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(main, "vector_store", upload_store)
    monkeypatch.setattr(main, "docjp_vector_store", topic_store)

    response = asyncio.run(
        main.source_page_preview(
            session_id=session_id,
            mode="research",
            file="manual.pdf",
            page=1,
            language="vi",
            source_scope="upload",
        )
    )

    assert isinstance(response, FileResponse)
    assert upload_store.calls == [(session_id, "manual.pdf", 1)]
    assert topic_store.calls == []
