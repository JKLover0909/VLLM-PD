import pytest

from src.integrations.mes_database import MesDatabaseResult
from src.integrations.mes_query_service import MesQueryService
from src.integrations.mes_sql_agent import MesSqlQueryResult


def test_mes_token_budgets_default_to_short_answers(monkeypatch):
    for key in (
        "MES_GENERAL_MAX_TOKENS",
        "MES_LIVE_API_MAX_TOKENS",
        "MES_DATABASE_MAX_TOKENS",
        "MES_SQL_PLANNER_MAX_TOKENS",
        "MES_SQL_ANSWER_MAX_TOKENS",
    ):
        monkeypatch.delenv(key, raising=False)

    assert MesQueryService.general_answer_max_tokens() == 256
    assert MesQueryService.live_api_answer_max_tokens() == 192
    assert MesQueryService.sql_planner_max_tokens() == 1200

    small_result = MesDatabaseResult(
        intent="lot_details",
        rows=[{"lot_id": "000001-01-000"}],
        imported_at="",
        fallback_answer="",
    )
    large_result = MesDatabaseResult(
        intent="lot_details",
        rows=[{"row": index} for index in range(4)],
        imported_at="",
        fallback_answer="",
    )
    assert MesQueryService.database_answer_max_tokens(small_result) == 256
    assert MesQueryService.database_answer_max_tokens(large_result) == 384

    sql_result = MesSqlQueryResult(
        columns=["lot_id"],
        rows=[{"lot_id": str(index)} for index in range(11)],
        imported_at="",
        truncated=False,
    )
    assert MesQueryService.sql_answer_max_tokens(sql_result) == 512


def test_mkac_token_budget_is_dynamic(monkeypatch):
    pytest.importorskip("numpy")
    from src.rag.rag_pipeline import RAGPipeline

    for key in (
        "RESEARCH_MAX_TOKENS",
        "MKAC_GENERAL_MAX_TOKENS",
        "MKAC_SIMPLE_MAX_TOKENS",
        "MKAC_EXTENDED_MAX_TOKENS",
    ):
        monkeypatch.delenv(key, raising=False)

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.max_tokens = 1024

    assert (
        pipeline._rag_answer_max_tokens(
            question="MKAC có bao nhiêu phòng ban?",
            mode="mkac",
            search_results=[],
            answer_scope="mkac",
            has_images=False,
        )
        == 512
    )
    assert (
        pipeline._rag_answer_max_tokens(
            question="Nội dung này là gì?",
            mode="mkac",
            search_results=[],
            answer_scope="general",
            has_images=False,
        )
        == 256
    )
    assert (
        pipeline._rag_answer_max_tokens(
            question="Quy trình đăng ký làm thêm giờ ở MKAC gồm những bước nào?",
            mode="mkac",
            search_results=[],
            answer_scope="mkac",
            has_images=False,
        )
        == 768
    )
    assert (
        pipeline._rag_answer_max_tokens(
            question="Tổng hợp tài liệu này",
            mode="research",
            search_results=[],
            answer_scope="research",
            has_images=False,
        )
        == 1800
    )
