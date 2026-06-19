import asyncio

import httpx
import pytest

from src.integrations.mes_client import MesApiError, MesClient, MesLotError
from src.rag.rag_pipeline import RAGPipeline


def test_mes_client_sends_bearer_payload_and_returns_all_highest_lots():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer demo-token"
        assert request.url == "https://mes.example/api/dynamics"
        assert request.read().decode() == (
            '{"ServiceName":"mes_data","ActionName":"DEMO_GET_TOTAL_ERROR",'
            '"Condition":{"Schema_Data":"MES_DATA_MKHC"}}'
        )
        return httpx.Response(
            200,
            json={
                "code": 200,
                "message": "ok",
                "data": [
                    {
                        "Lot_Id": "LOT-LOW",
                        "Product_Id": "PRODUCT-A",
                        "Total_Error_Qty": "120",
                    },
                    {
                        "Lot_Id": "LOT-MAX-1",
                        "Product_Id": "PRODUCT-B",
                        "Total_Error_Qty": "15,920",
                    },
                    {
                        "Lot_Id": "LOT-MAX-2",
                        "Product_Id": "PRODUCT-C",
                        "Total_Error_Qty": "15920",
                    },
                ],
            },
        )

    client = MesClient(
        "https://mes.example/api/dynamics",
        "demo-token",
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(client.get_lots_with_highest_error())

    assert result == [
        MesLotError("LOT-MAX-1", "PRODUCT-B", 15920),
        MesLotError("LOT-MAX-2", "PRODUCT-C", 15920),
    ]


def test_mes_client_rejects_invalid_error_quantity():
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={
                "code": 200,
                "data": [
                    {
                        "Lot_Id": "LOT-1",
                        "Product_Id": "PRODUCT-1",
                        "Total_Error_Qty": "not-a-number",
                    }
                ],
            },
        )
    )
    client = MesClient(
        "https://mes.example/api/dynamics",
        "demo-token",
        transport=transport,
    )

    with pytest.raises(MesApiError, match="số lượng lỗi không hợp lệ"):
        asyncio.run(client.get_lots_with_highest_error())


@pytest.mark.parametrize(
    "question",
    [
        "Mã Lot nào có số lượng lỗi nhiều nhất?",
        "Lot nào nhiều lỗi nhất hiện nay?",
        "Lô sản xuất có tổng lỗi cao nhất là lô nào?",
        "Cho tôi Lot đứng đầu về số lỗi.",
        "Lot có NG cao nhất là mã nào?",
        "Which lot has the most defects?",
    ],
)
def test_highest_lot_error_intent_matches_natural_questions(question):
    assert RAGPipeline._is_highest_lot_error_question(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Lot 000432 đang sản xuất mã hàng nào?",
        "MKAC có bao nhiêu phòng ban?",
        "Số lượng lỗi hôm nay là bao nhiêu?",
        "Lot nào được tạo gần nhất?",
    ],
)
def test_highest_lot_error_intent_rejects_other_questions(question):
    assert RAGPipeline._is_highest_lot_error_question(question) is False


def test_mes_fallback_answer_contains_required_fields():
    answer = RAGPipeline._format_mes_fallback(
        [MesLotError("000432-01-000", "3736-0008", 15920)]
    )

    assert "000432-01-000" in answer
    assert "3736-0008" in answer
    assert "15.920" in answer


def test_mes_answer_validation_rejects_missing_product_id():
    lots = [MesLotError("000432-01-000", "3736-0008", 15920)]

    assert RAGPipeline._mes_answer_has_required_fields(
        "Lot 000432-01-000, mã hàng 3736-0008, có 15.920 lỗi.",
        lots,
    )
    assert not RAGPipeline._mes_answer_has_required_fields(
        "Lot 000432-01-000 có 15.920 lỗi.",
        lots,
    )
