import asyncio

import httpx
import pytest

from src.integrations.mes_client import MesApiError, MesClient, MesLotError
from src.integrations.mes_database import MesDatabaseResult
from src.rag.rag_pipeline import RAGPipeline


class FakeEmbedder:
    pass


class FakeStore:
    pass


class FakeMesDatabase:
    available = True

    def __init__(self):
        self.calls = []

    def is_snapshot_question(self, question):
        return "database" in question.lower() or "snapshot" in question.lower()

    def query_question(self, question, *, allow_highest_lot=False):
        self.calls.append((question, allow_highest_lot))
        if "Lot 000432-01-000" in question:
            return MesDatabaseResult(
                intent="lot_details",
                rows=[{"lot_id": "000432-01-000", "product_id": "3736-0008"}],
                imported_at="2026-06-20T03:52:08+00:00",
                fallback_answer="Lot 000432-01-000 thuộc mã hàng 3736-0008.",
                required_terms=("000432-01-000", "3736-0008"),
            )
        if allow_highest_lot:
            return MesDatabaseResult(
                intent="highest_error_lot",
                rows=[
                    {
                        "lot_id": "SNAPSHOT-LOT",
                        "product_id": "SNAPSHOT-PRODUCT",
                        "total_error_qty": 52300,
                    }
                ],
                imported_at="2026-06-20T03:52:08+00:00",
                fallback_answer="Snapshot Lot có 52.300 lỗi.",
                required_terms=("SNAPSHOT-LOT", "SNAPSHOT-PRODUCT", "52300"),
            )
        return None


class FakeEmptyMesDatabase(FakeMesDatabase):
    def query_question(self, question, *, allow_highest_lot=False):
        self.calls.append((question, allow_highest_lot))
        return None


class FakeMesClient:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    async def get_lots_with_highest_error(self):
        self.calls += 1
        if self.error:
            raise self.error
        return [MesLotError("LIVE-LOT", "LIVE-PRODUCT", 15920)]


def make_pipeline(mes_client, mes_database):
    return RAGPipeline(
        embedder=FakeEmbedder(),
        vector_store=FakeStore(),
        mes_client=mes_client,
        mes_database=mes_database,
    )


def test_mes_client_sends_bearer_payload_and_returns_all_highest_lots():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer demo-token"
        assert request.url == "https://mes.example/api/dynamics"
        assert request.read().decode() == (
            '{"ServiceName":"mes_data","ActionName":"DEMO_GET_TOTAL_ERROR",'
            '"Condition":{"Schema_Data":"MES_DATA"}}'
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


def test_lot_detail_question_routes_to_snapshot_without_calling_live_api():
    mes_client = FakeMesClient()
    mes_database = FakeMesDatabase()
    pipeline = make_pipeline(mes_client, mes_database)

    source, result = asyncio.run(
        pipeline._get_mes_route("Lot 000432-01-000 sản xuất mã hàng nào?", "mes")
    )

    assert source == "mes_database"
    assert result.intent == "lot_details"
    assert mes_client.calls == 0


def test_highest_lot_question_prefers_snapshot_database():
    mes_client = FakeMesClient()
    mes_database = FakeMesDatabase()
    pipeline = make_pipeline(mes_client, mes_database)

    source, result = asyncio.run(
        pipeline._get_mes_route("Lot nào có số lượng lỗi nhiều nhất?", "mes")
    )

    assert source == "mes_database"
    assert result.intent == "highest_error_lot"
    assert result.rows[0]["lot_id"] == "SNAPSHOT-LOT"
    assert mes_client.calls == 0
    assert mes_database.calls == [("Lot nào có số lượng lỗi nhiều nhất?", True)]


@pytest.mark.parametrize(
    "question",
    [
        "Liệt kê danh sách 5 lot nhiều lỗi nhất",
        "Top 5 lot có nhiều lỗi nhất",
    ],
)
def test_top_n_highest_lot_questions_prefer_snapshot_database(question):
    mes_client = FakeMesClient()
    mes_database = FakeMesDatabase()
    pipeline = make_pipeline(mes_client, mes_database)

    source, result = asyncio.run(pipeline._get_mes_route(question, "mes"))

    assert source == "mes_database"
    assert result.intent == "highest_error_lot"
    assert mes_client.calls == 0
    assert mes_database.calls == [(question, True)]


def test_live_api_is_used_when_snapshot_has_no_matching_data():
    mes_client = FakeMesClient()
    mes_database = FakeEmptyMesDatabase()
    pipeline = make_pipeline(mes_client, mes_database)

    source, result = asyncio.run(
        pipeline._get_mes_route("Lot nào có số lượng lỗi nhiều nhất?", "mes")
    )

    assert source == "mes"
    assert result[0].lot_id == "LIVE-LOT"
    assert mes_client.calls == 1
    assert mes_database.calls == [("Lot nào có số lượng lỗi nhiều nhất?", True)]


def test_explicit_snapshot_question_bypasses_live_api():
    mes_client = FakeMesClient()
    mes_database = FakeMesDatabase()
    pipeline = make_pipeline(mes_client, mes_database)

    source, result = asyncio.run(
        pipeline._get_mes_route(
            "Theo database, Lot nào có số lượng lỗi nhiều nhất?",
            "mes",
        )
    )

    assert source == "mes_database"
    assert result.intent == "highest_error_lot"
    assert mes_client.calls == 0


def test_snapshot_is_used_without_calling_live_api():
    mes_client = FakeMesClient(MesApiError("offline"))
    mes_database = FakeMesDatabase()
    pipeline = make_pipeline(mes_client, mes_database)

    source, result = asyncio.run(
        pipeline._get_mes_route("Lot nào có số lượng lỗi nhiều nhất?", "mes")
    )

    assert source == "mes_database"
    assert result.intent == "highest_error_lot"
    assert mes_client.calls == 0


def test_mes_database_answer_rejects_internal_field_names():
    result = MesDatabaseResult(
        intent="product_error_summary",
        rows=[{"product_id": "3736-0008", "total_error_qty": 40727}],
        imported_at="2026-06-20T03:52:08+00:00",
        fallback_answer="Mã hàng 3736-0008 có tổng 40.727 lỗi.",
        required_terms=("3736-0008", "40727"),
    )

    assert RAGPipeline._mes_database_answer_has_required_terms(
        "Mã hàng 3736-0008 có tổng 40.727 lỗi.",
        result,
    )
    assert not RAGPipeline._mes_database_answer_has_required_terms(
        "Mã hàng 3736-0008 có 40.727 lỗi (total_error_qty).",
        result,
    )


def test_mkac_mode_does_not_route_mes_questions():
    mes_client = FakeMesClient()
    mes_database = FakeMesDatabase()
    pipeline = make_pipeline(mes_client, mes_database)

    source, result = asyncio.run(
        pipeline._get_mes_route("Lot nào có số lượng lỗi nhiều nhất?", "mkac")
    )

    assert source is None
    assert result is None
    assert mes_client.calls == 0
    assert mes_database.calls == []


def test_compound_highest_lot_question_is_reserved_for_sql_agent():
    mes_client = FakeMesClient()
    mes_database = FakeMesDatabase()
    pipeline = make_pipeline(mes_client, mes_database)

    source, result = asyncio.run(
        pipeline._get_mes_route(
            "Trong Lot có số lượng lỗi nhiều nhất, 3 loại lỗi nhiều nhất là gì?",
            "mes",
        )
    )

    assert source is None
    assert result is None
    assert mes_client.calls == 0
    assert mes_database.calls == []
