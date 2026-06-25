"""MES query service separated from document RAG routing."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import unicodedata
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

from src.integrations.mes_client import MesClient, MesLotError
from src.integrations.mes_database import MesDatabase, MesDatabaseError, MesDatabaseResult
from src.integrations.mes_sql_agent import MesSqlAgent, MesSqlAgentError, MesSqlQueryResult

logger = logging.getLogger(__name__)

MODEL_ROUTES = {
    "auto": "auto-model",
    "local": "local-gemma",
    "openai": "openai-model",
    "grok": "grok-model",
}

LOCAL_MODEL_ALIASES = {"local-gemma", "coding-model"}

MES_SYSTEM_PROMPT = """Bạn là trợ lý dữ liệu sản xuất bo mạch của MKAC.

Hãy trả lời bằng một câu tiếng Việt tự nhiên, ngắn gọn và trực tiếp.
Bắt buộc nêu đủ mã Lot, mã hàng và tổng số lỗi của Lot có số lỗi cao nhất.
Chỉ sử dụng dữ liệu MES được cung cấp, không suy đoán nguyên nhân lỗi và không thêm dữ liệu khác.
Định dạng số lượng lỗi theo cách đọc tiếng Việt, dùng dấu chấm phân cách hàng nghìn.
Giữ nguyên mã và số lượng ở dạng chữ số; tuyệt đối không viết số lượng lỗi bằng chữ.
Nếu có nhiều Lot đồng hạng, phải nêu đầy đủ tất cả các Lot đó."""

MES_DATABASE_SYSTEM_PROMPT = """Bạn là trợ lý phân tích dữ liệu sản xuất bo mạch MKAC.

Chỉ trả lời từ JSON MES snapshot được cung cấp. Không tự viết SQL, không suy đoán
nguyên nhân lỗi và không bổ sung dữ liệu bên ngoài JSON.

Quy tắc:
1. Trả lời bằng tiếng Việt tự nhiên, trực tiếp và ngắn gọn.
2. Giữ nguyên mã Lot, mã hàng, mã lỗi, công đoạn và các con số.
3. Dùng dấu chấm phân cách hàng nghìn khi trình bày số lượng.
4. total_error_qty là tổng số lượng lỗi; error_record_count là số lần ghi nhận,
   hai đại lượng này không được đánh đồng.
5. Tên lỗi rỗng nghĩa là *Lỗi chưa rõ tên*; không được tự đặt tên lỗi.
6. Nói rõ đây là dữ liệu MES snapshot khi kết luận có thể bị hiểu là dữ liệu
   thời gian thực.
7. Dữ liệu đã được lọc theo chính sách hiển thị trong trường filters của JSON.
8. Tuyệt đối không để lộ tên field JSON/SQL như total_error_qty,
   error_record_count, lot_count, top_errors, error_qty hoặc các tên kỹ thuật tương tự.
9. Nếu JSON chứa danh sách lỗi chi tiết (top_errors) của một Lot, hãy tự động trình bày thêm danh sách đó (nếu câu trả lời chính chưa liệt kê). Ví dụ:
   trong đó 3 lỗi có số lượng lỗi lớn nhất là:
   1. B114D - Thừa đồng: 4.293
   2. 0002 - *Lỗi chưa rõ tên*: 2.000
10. Chỉ trả lời đúng thông tin người dùng hỏi; không liệt kê thêm chỉ số không
    cần thiết."""

MES_UNSUPPORTED_ANSWER = (
    "Chưa nhận diện được truy vấn MES này. Bạn có thể hỏi về thông tin một Lot, "
    "chi tiết lỗi theo Lot, tên mã lỗi hoặc thống kê lỗi theo mã hàng."
)


class MesQueryService:
    """Route and answer MES questions without touching document retrieval."""

    def __init__(
        self,
        *,
        mes_client: MesClient | None = None,
        mes_database: MesDatabase | None = None,
        mes_sql_agent: MesSqlAgent | None = None,
        openai_client: AsyncOpenAI | None = None,
    ):
        self.mes_client = mes_client if mes_client is not None else MesClient.from_env()
        self.mes_database = (
            mes_database if mes_database is not None else MesDatabase.from_env()
        )
        self.mes_sql_agent = (
            mes_sql_agent if mes_sql_agent is not None else MesSqlAgent.from_env()
        )
        self.openai_client = openai_client or AsyncOpenAI(
            api_key=os.getenv("LITELLM_MASTER_KEY", "sk-local"),
            base_url=os.getenv("LITELLM_URL", "http://localhost:4000/v1"),
        )

    async def query(self, question: str, model: str = "openai") -> tuple[str, list, str, str]:
        mes_source, mes_data = await self.get_route(question)
        if mes_source == "mes":
            mes_lots = mes_data
            assert isinstance(mes_lots, list)
            routed_model = self.resolve_model(model)
            answer = await self._generate_live_api_answer(question, mes_lots, routed_model)
            return answer, [], routed_model, "mes"

        if mes_source == "mes_database":
            assert isinstance(mes_data, MesDatabaseResult)
            answer, routed_model = await self._generate_database_answer(
                question,
                mes_data,
                model,
            )
            return answer, [], routed_model, "mes_database"

        sql_answer = await self._generate_sql_answer(question, model)
        if sql_answer is not None:
            answer, routed_model = sql_answer
            return answer, [], routed_model, "mes_database"

        return MES_UNSUPPORTED_ANSWER, [], self.resolve_model(model), "mes_database"

    async def query_stream(
        self,
        question: str,
        model: str = "openai",
    ) -> tuple[AsyncGenerator[str, None], list, str, str]:
        answer, results, routed_model, answer_scope = await self.query(question, model)

        async def token_generator():
            yield answer

        return token_generator(), results, routed_model, answer_scope

    async def get_route(
        self,
        question: str,
    ) -> tuple[str | None, list[MesLotError] | MesDatabaseResult | None]:
        highest_lot_question = self.is_highest_lot_error_question(question)
        if highest_lot_question and self.is_compound_mes_question(question):
            return None, None
        explicit_snapshot = bool(
            self.mes_database
            and self.mes_database.is_snapshot_question(question)
        )

        if highest_lot_question:
            snapshot_result = await self._query_database(
                question,
                allow_highest_lot=True,
            )
            if snapshot_result is not None:
                logger.info("Routing highest Lot error question to local MES snapshot.")
                return "mes_database", snapshot_result

            api_error: Exception | None = None
            if self.mes_client is not None:
                try:
                    logger.info("Routing highest Lot error question to live MES API.")
                    return "mes", await self.mes_client.get_lots_with_highest_error()
                except Exception as exc:
                    api_error = exc
                    logger.warning(
                        "MES API failed; trying local snapshot fallback: %s",
                        exc,
                    )

            if api_error is not None:
                raise api_error
            raise RuntimeError("MES API và MES snapshot chưa được cấu hình.")

        snapshot_result = await self._query_database(
            question,
            allow_highest_lot=explicit_snapshot,
        )
        if snapshot_result is not None:
            return "mes_database", snapshot_result
        return None, None

    async def _generate_live_api_answer(
        self,
        question: str,
        lots: list[MesLotError],
        routed_model: str,
    ) -> str:
        fallback_answer = self.format_live_api_fallback(lots)
        try:
            response = await self.openai_client.chat.completions.create(
                model=routed_model,
                messages=self.live_api_messages(question, lots),
                temperature=0.1,
                max_tokens=240,
                **self.provider_options(routed_model),
            )
            candidate = response.choices[0].message.content or ""
            return (
                candidate
                if self.live_api_answer_has_required_fields(candidate, lots)
                else fallback_answer
            )
        except Exception as exc:
            logger.warning("LLM MES live answer generation failed: %s", exc)
            return fallback_answer

    async def _generate_database_answer(
        self,
        question: str,
        result: MesDatabaseResult,
        model: str,
    ) -> tuple[str, str]:
        routed_model = self.resolve_model(model)
        try:
            response = await self.openai_client.chat.completions.create(
                model=routed_model,
                messages=self.database_messages(question, result),
                temperature=0.1,
                max_tokens=600,
                **self.provider_options(routed_model),
            )
            candidate = response.choices[0].message.content or ""
            answer = (
                candidate
                if self.database_answer_has_required_terms(candidate, result)
                else result.fallback_answer
            )
        except Exception as exc:
            logger.warning("LLM MES snapshot answer generation failed: %s", exc)
            answer = result.fallback_answer
        return answer, routed_model

    async def _generate_sql_answer(
        self,
        question: str,
        model: str,
    ) -> tuple[str, str] | None:
        if self.mes_sql_agent is None or not self.mes_sql_agent.available:
            return None

        routed_model = self.resolve_model(model)
        previous_error = ""
        max_attempts = max(1, min(int(os.getenv("MES_SQL_AGENT_MAX_ATTEMPTS", "2")), 3))
        for attempt in range(max_attempts):
            try:
                response = await self.openai_client.chat.completions.create(
                    model=routed_model,
                    messages=self.mes_sql_agent.planner_messages(
                        question,
                        previous_error=previous_error,
                    ),
                    temperature=0,
                    max_tokens=1200,
                    **self.provider_options(routed_model),
                )
                content = response.choices[0].message.content or ""
                plan = self.mes_sql_agent.parse_plan(content)
                if not plan.can_answer:
                    logger.info("MES SQL planner cannot answer: %s", plan.reason)
                    return None
                result = await asyncio.to_thread(self.mes_sql_agent.execute, plan.sql)
                logger.info(
                    "MES SQL agent executed attempt=%s rows=%s",
                    attempt + 1,
                    len(result.rows),
                )
                answer_response = await self.openai_client.chat.completions.create(
                    model=routed_model,
                    messages=self.mes_sql_agent.answer_messages(question, result),
                    temperature=0.1,
                    max_tokens=800,
                    **self.provider_options(routed_model),
                )
                candidate = answer_response.choices[0].message.content or ""
                candidate = self.normalize_sql_answer(candidate)
                if self.sql_answer_is_natural(candidate) and self.sql_answer_matches_result(
                    candidate,
                    result,
                ):
                    return candidate, routed_model
                return self.mes_sql_agent.fallback_answer(result), routed_model
            except MesSqlAgentError as exc:
                previous_error = str(exc)
                logger.warning(
                    "MES SQL plan rejected attempt=%s: %s",
                    attempt + 1,
                    exc,
                )
            except Exception as exc:
                logger.warning("MES SQL agent failed: %s", exc)
                return None
        return None

    async def _query_database(
        self,
        question: str,
        *,
        allow_highest_lot: bool,
    ) -> MesDatabaseResult | None:
        if self.mes_database is None or not self.mes_database.available:
            return None
        try:
            result = await asyncio.to_thread(
                self.mes_database.query_question,
                question,
                allow_highest_lot=allow_highest_lot,
            )
            if result is not None:
                logger.info("Routing MES question to snapshot intent=%s", result.intent)
            return result
        except MesDatabaseError as exc:
            logger.warning("MES snapshot query failed: %s", exc)
            return None

    @staticmethod
    def live_api_messages(
        question: str,
        lots: list[MesLotError],
    ) -> list[dict[str, str]]:
        rows = "\n".join(
            (
                f"- Lot_Id={lot.lot_id}; Product_Id={lot.product_id}; "
                f"Total_Error_Qty={lot.total_error_qty}"
            )
            for lot in lots
        )
        return [
            {"role": "system", "content": MES_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Câu hỏi: {question}\n\nDữ liệu MES đã xác thực:\n{rows}",
            },
        ]

    @staticmethod
    def database_messages(
        question: str,
        result: MesDatabaseResult,
    ) -> list[dict[str, str]]:
        payload = json.dumps(
            result.prompt_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return [
            {"role": "system", "content": MES_DATABASE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Câu hỏi: {question}\n\n"
                    f"Dữ liệu MES snapshot:\n{payload}\n\n"
                    f"Câu trả lời kiểm chứng để tham khảo: {result.fallback_answer}\n"
                    "Hãy diễn đạt tự nhiên, không nhắc tên field nội bộ."
                ),
            },
        ]

    @staticmethod
    def format_live_api_fallback(lots: list[MesLotError]) -> str:
        def describe(lot: MesLotError) -> str:
            quantity = f"{lot.total_error_qty:,}".replace(",", ".")
            return (
                f"Lot {lot.lot_id}, mã hàng {lot.product_id}, "
                f"với tổng cộng {quantity} lỗi"
            )

        if len(lots) == 1:
            return f"{describe(lots[0])} là Lot có số lượng lỗi cao nhất."
        return "Các Lot có số lượng lỗi cao nhất là: " + "; ".join(
            describe(lot) for lot in lots
        ) + "."

    @staticmethod
    def live_api_answer_has_required_fields(
        answer: str,
        lots: list[MesLotError],
    ) -> bool:
        normalized_quantity = answer.replace(".", "").replace(",", "")
        return bool(answer.strip()) and all(
            lot.lot_id in answer
            and lot.product_id in answer
            and str(lot.total_error_qty) in normalized_quantity
            for lot in lots
        )

    @staticmethod
    def database_answer_has_required_terms(
        answer: str,
        result: MesDatabaseResult,
    ) -> bool:
        if not answer.strip():
            return False
        forbidden_fields = (
            "total_error_qty",
            "error_record_count",
            "distinct_error_count",
            "unmapped_error_record_count",
            "lot_count",
            "product_id",
            "lot_id",
            "error_id",
            "process_id",
        )
        if any(field in answer.lower() for field in forbidden_fields):
            return False
        normalized_answer = answer.replace(".", "").replace(",", "")
        return all(
            not term
            or term in answer
            or term.replace(".", "").replace(",", "") in normalized_answer
            for term in result.required_terms
        )

    @staticmethod
    def normalize_sql_answer(answer: str) -> str:
        text = (answer or "").strip()
        if not text:
            return ""
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return text
            if isinstance(payload, dict):
                nested_answer = payload.get("answer")
                if isinstance(nested_answer, str):
                    return nested_answer.strip()
        return text

    @staticmethod
    def sql_answer_matches_result(answer: str, result: MesSqlQueryResult) -> bool:
        normalized = answer.lower()
        for row in result.rows[:5]:
            for key in ("lot_id", "product_id", "error_id"):
                value = row.get(key)
                if value and str(value).lower() not in normalized:
                    return False
            error_name = row.get("error_name")
            if error_name and str(error_name).strip():
                if str(error_name).lower() not in normalized:
                    return False
            elif "error_name" in row and "chưa rõ tên" not in normalized:
                return False
        return True

    @staticmethod
    def sql_answer_is_natural(answer: str) -> bool:
        if not answer.strip():
            return False
        normalized = answer.lower()
        forbidden = (
            "select ",
            " from ",
            "total_error_qty",
            "error_record_count",
            "distinct_error_count",
            "```sql",
            "{\"answer\"",
        )
        return not any(marker in normalized for marker in forbidden)

    @staticmethod
    def is_highest_lot_error_question(question: str) -> bool:
        normalized = unicodedata.normalize(
            "NFD",
            question.lower().replace("đ", "d"),
        )
        normalized = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()

        has_lot = bool(re.search(r"\b(lot|lo|lo san xuat)\b", normalized))
        has_error = bool(re.search(r"\bng\b", normalized)) or any(
            marker in normalized
            for marker in ("loi", "error", "defect", "hang loi", "san pham loi")
        )
        has_maximum = bool(
            re.search(r"\b(nhieu|cao|lon)\b(?:\s+\w+){0,3}\s+nhat\b", normalized)
        ) or any(
            marker in normalized
            for marker in (
                "nhieu nhat",
                "cao nhat",
                "lon nhat",
                "toi da",
                "top 1",
                "top loi",
                "dung dau",
                "max",
                "maximum",
                "most",
            )
        )
        return has_lot and has_error and has_maximum

    @classmethod
    def is_compound_mes_question(cls, question: str) -> bool:
        normalized = unicodedata.normalize(
            "NFD",
            question.lower().replace("đ", "d"),
        )
        normalized = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
        return bool(
            re.search(r"\b\d+\s+(?:loai\s+)?loi\b", normalized)
            or re.search(r"\btop\s*\d+\s+(?:loai\s+)?loi\b", normalized)
            or any(
                marker in normalized
                for marker in ("cac loi nhieu nhat", "nhung loi nhieu nhat")
            )
        )

    @staticmethod
    def resolve_model(model: str) -> str:
        if model == "grok":
            return "openai-model"
        try:
            return MODEL_ROUTES[model]
        except KeyError as exc:
            raise ValueError(f"Unsupported model option: {model}") from exc

    @staticmethod
    def provider_options(routed_model: str) -> dict[str, Any]:
        if routed_model in LOCAL_MODEL_ALIASES:
            return {"extra_body": {"stream_options": {"include_usage": False}}}
        return {}
