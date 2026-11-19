"""Regression: the "no information" branch must answer instead of raising."""

import asyncio

import pytest

from src.rag.rag_pipeline import (
    MKAC_NO_INFORMATION_ANSWER,
    RESEARCH_NO_INFORMATION_ANSWER,
    RAGPipeline,
)


def _pipeline(answer_scope: str):
    pipeline = object.__new__(RAGPipeline)

    async def no_mes_route(question, mode):
        return None, None

    pipeline._get_mes_route = no_mes_route
    pipeline._prepare_query_context = lambda *args: ([], [], answer_scope)
    return pipeline


@pytest.mark.parametrize(
    ("mode", "scope", "expected"),
    [
        ("mkac", "general", MKAC_NO_INFORMATION_ANSWER),
        ("research", "research", RESEARCH_NO_INFORMATION_ANSWER),
    ],
)
def test_query_returns_no_information_answer(mode, scope, expected):
    answer, sources, _model, answer_scope = asyncio.run(
        _pipeline(scope).query("s", "Câu hỏi không có trong tài liệu", "auto", mode=mode)
    )

    assert answer == expected
    assert sources == []
    assert answer_scope == scope


@pytest.mark.parametrize(
    ("mode", "scope", "expected"),
    [
        ("mkac", "general", MKAC_NO_INFORMATION_ANSWER),
        ("research", "research", RESEARCH_NO_INFORMATION_ANSWER),
    ],
)
def test_query_stream_returns_no_information_answer(mode, scope, expected):
    async def collect():
        stream, sources, _model, answer_scope = await _pipeline(scope).query_stream(
            "s", "Câu hỏi không có trong tài liệu", "auto", mode=mode
        )
        tokens = [token async for _kind, token in stream]
        return "".join(tokens), sources, answer_scope

    answer, sources, answer_scope = asyncio.run(collect())

    assert answer == expected
    assert sources == []
    assert answer_scope == scope
