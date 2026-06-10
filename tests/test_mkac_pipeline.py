from pathlib import Path

from src.rag.parser import DocumentParser, TextChunk
from src.rag.rag_pipeline import RAGPipeline
from src.rag.rag_pipeline import GENERAL_SYSTEM_PROMPT, WEB_SYSTEM_PROMPT
from src.rag.vector_store import SearchResult
from scripts.index_mkac_documents import build_embedding_text


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


class FakeWebSearcher:
    def __init__(self, results=None):
        self.results = results or []
        self.questions = []

    def search(self, question):
        self.questions.append(question)
        return self.results


def test_web_prompt_answers_directly_without_repeated_fallback_banner():
    assert "Bắt đầu bằng câu" not in WEB_SYSTEM_PROMPT
    assert (
        "Không tìm thấy nội dung này trong tài liệu nội bộ MKAC; dưới đây"
        not in WEB_SYSTEM_PROMPT
    )
    assert "Chưa tìm thấy thông tin phù hợp về nội dung này." in GENERAL_SYSTEM_PROMPT
    assert "Không bổ sung kiến thức chung" in GENERAL_SYSTEM_PROMPT


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


def test_embedding_text_includes_curated_company_identity():
    chunk = TextChunk("Nội dung OCR", "registration.pdf", 1, 0, "text")
    text = build_embedding_text(
        chunk,
        {
            "knowledge_base": "MKAC",
            "title": "Giấy chứng nhận đăng ký doanh nghiệp",
            "category": "corporate_identity",
            "organization": {
                "short_name": "MKAC",
                "legal_name_vi": "Công ty Cổ phần Meiko Automation",
                "legal_name_en": "Meiko Automation Joint Stock Company",
                "enterprise_id": "0108918123",
            },
        },
    )

    assert "Công ty Cổ phần Meiko Automation" in text
    assert "0108918123" in text
    assert text.endswith("Nội dung OCR")


def test_mkac_retrieval_question_removes_company_name_from_policy_question():
    assert (
        RAGPipeline._mkac_retrieval_question(
            "Nhân viên MKAC làm thêm giờ được tính như thế nào?"
        )
        == "Nhân viên làm thêm giờ được tính như thế nào?"
    )
    identity_question = "MKAC là viết tắt của công ty nào?"
    assert RAGPipeline._mkac_retrieval_question(identity_question) == identity_question


def test_company_profile_question_uses_legal_documents_and_lower_threshold():
    legal = SearchResult(
        TextChunk(
            "Ngành nghề đăng ký",
            "investment.pdf",
            2,
            0,
            "text",
            {"category": "investment_registration"},
        ),
        score=0.47,
    )
    unrelated = SearchResult(
        TextChunk(
            "Quy định làm thêm",
            "overtime.pdf",
            1,
            0,
            "text",
            {"category": "working_time"},
        ),
        score=0.46,
    )
    pipeline = RAGPipeline(
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
        mkac_vector_store=FakeStore([legal, unrelated]),
    )
    question = "MKAC có những lĩnh vực hoạt động nào theo hồ sơ đăng ký?"

    results, _, scope = pipeline._prepare_query_context("ignored", question, "mkac")

    assert pipeline._mkac_retrieval_threshold(question) == 0.42
    assert results == [legal]
    assert scope == "mkac"


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


def test_mkac_query_uses_web_fallback_when_internal_store_has_no_match():
    web_result = SearchResult(
        TextChunk(
            text="Thông tin công khai về MKAC",
            source_file="MKAC public page",
            page_number=0,
            chunk_index=0,
            content_type="web",
            metadata={"url": "https://example.com/mkac"},
        ),
        score=0.9,
    )
    web_searcher = FakeWebSearcher([web_result])
    pipeline = RAGPipeline(
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
        mkac_vector_store=FakeStore(),
        web_searcher=web_searcher,
    )

    results, images, scope = pipeline._prepare_query_context(
        "ignored-session",
        "MKAC có những sản phẩm nào?",
        "mkac",
    )

    assert results == [web_result]
    assert images == []
    assert scope == "web"
    assert web_searcher.questions == ["MKAC có những sản phẩm nào?"]
    assert pipeline.format_sources(results)[0]["url"] == "https://example.com/mkac"


def test_mkac_query_rejects_weak_internal_match_before_web_fallback():
    weak_internal_result = SearchResult(
        TextChunk("MKAC footer", "internal.pdf", 1, 0, "text"),
        score=0.43,
    )
    web_result = SearchResult(
        TextChunk(
            "Website MKAC",
            "MKAC website",
            0,
            0,
            "web",
            {"url": "https://example.com/mkac"},
        ),
        score=0.9,
    )
    pipeline = RAGPipeline(
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
        mkac_vector_store=FakeStore([weak_internal_result]),
        web_searcher=FakeWebSearcher([web_result]),
    )
    pipeline.mkac_score_threshold = 0.48

    results, _, scope = pipeline._prepare_query_context(
        "ignored-session",
        "Website chính thức của MKAC là gì?",
        "mkac",
    )

    assert results == [web_result]
    assert scope == "web"


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
