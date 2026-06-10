from pathlib import Path

from src.rag.parser import DocumentParser, TextChunk
from src.rag.rag_pipeline import RAGPipeline
from src.rag.vector_store import SearchResult


class FakeEmbedder:
    def embed_query(self, question):
        return [0.0] * 1024


class FakeStore:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


def test_split_text_preserves_page_and_metadata():
    parser = DocumentParser.__new__(DocumentParser)
    chunks = parser._split_text(
        "Quy định làm thêm giờ\nNhân viên cần đăng ký trước.",
        filename="overtime.pdf",
        page_number=7,
        start_index=3,
        metadata={"category": "working_time"},
    )

    assert len(chunks) == 1
    assert chunks[0].page_number == 7
    assert chunks[0].chunk_index == 3
    assert chunks[0].metadata["category"] == "working_time"


def test_mkac_query_uses_general_fallback_without_relevant_chunks():
    pipeline = RAGPipeline(
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
        mkac_vector_store=FakeStore(),
    )

    results, images, scope = pipeline._prepare_query_context(
        "ignored-session",
        "Thủ đô của Pháp là gì?",
        "mkac",
    )

    assert results == []
    assert images == []
    assert scope == "general"


def test_mkac_query_uses_shared_store_and_page_image(tmp_path):
    image = tmp_path / "page-0002.png"
    image.write_bytes(b"image")
    result = SearchResult(
        TextChunk(
            text="Bảng định mức công tác",
            source_file="travel.pdf",
            page_number=2,
            chunk_index=0,
            content_type="table",
            metadata={"image_path": str(image)},
        ),
        score=0.8,
    )
    mkac_store = FakeStore([result])
    pipeline = RAGPipeline(
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
        mkac_vector_store=mkac_store,
    )

    results, images, scope = pipeline._prepare_query_context(
        "ignored-session",
        "Bảng định mức công tác là gì?",
        "mkac",
    )

    assert results == [result]
    assert images == [Path(image)]
    assert scope == "mkac"
    assert mkac_store.calls[0]["session_id"] == "mkac"


def test_relative_filter_removes_weak_tail_results():
    def result(score):
        return SearchResult(
            TextChunk("text", "file.pdf", 1, 0, "text"),
            score=score,
        )

    filtered = RAGPipeline._filter_relative_results(
        [result(0.60), result(0.55), result(0.43)]
    )

    assert [item.score for item in filtered] == [0.60, 0.55]
