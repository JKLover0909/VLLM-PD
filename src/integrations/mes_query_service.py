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
    "local": "local-qwen-chat",
    "openai": "openai-model",
    "grok": "grok-model",
}

LOCAL_CHAT_MODEL_ALIASES = {"local-gemma", "local-qwen-chat"}
LOCAL_MODEL_ALIASES = LOCAL_CHAT_MODEL_ALIASES | {"local-qwen-coder", "coding-model"}

MES_SYSTEM_PROMPT = """Bạn là trợ lý dữ liệu sản xuất bo mạch của MKAC.

Hãy trả lời bằng một câu tiếng Việt tự nhiên, ngắn gọn và trực tiếp.
Bắt buộc nêu đủ mã Lot, mã hàng và tổng số lỗi của Lot có số lỗi cao nhất.
Chỉ sử dụng dữ liệu MES được cung cấp, không suy đoán nguyên nhân lỗi và không thêm dữ liệu khác.
Định dạng số lượng lỗi theo cách đọc tiếng Việt, dùng dấu chấm phân cách hàng nghìn.
Giữ nguyên mã và số lượng ở dạng chữ số; tuyệt đối không viết số lượng lỗi bằng chữ.
Nếu có nhiều Lot đồng hạng, phải nêu đầy đủ tất cả các Lot đó."""

MES_DATABASE_SYSTEM_PROMPT = """Bạn là trợ lý phân tích dữ liệu sản xuất bo mạch MKAC.

Chỉ trả lời từ dữ liệu MES snapshot được cung cấp. Không tự viết SQL, không suy
đoán nguyên nhân lỗi và không bổ sung dữ liệu bên ngoài.

Quy tắc:
1. Trả lời bằng tiếng Việt tự nhiên, trực tiếp và ngắn gọn.
2. Giữ nguyên mã Lot, mã hàng, mã lỗi, công đoạn và các con số.
3. Dùng dấu chấm phân cách hàng nghìn khi trình bày số lượng.
4. Tổng số lượng lỗi và số lần ghi nhận lỗi là hai đại lượng khác nhau, không
   được đánh đồng.
5. Tên lỗi rỗng nghĩa là *Lỗi chưa rõ tên*; không được tự đặt tên lỗi.
6. Nói rõ đây là dữ liệu MES snapshot khi kết luận có thể bị hiểu là dữ liệu
   thời gian thực.
7. Không nhắc tới JSON, SQL, filters, chính sách hiển thị hoặc cơ chế nội bộ.
8. Tuyệt đối không để lộ tên trường kỹ thuật trong dữ liệu đầu vào.
9. Nếu dữ liệu chứa danh sách lỗi chi tiết của một Lot, hãy tự động trình bày
   thêm danh sách đó (nếu câu trả lời chính chưa liệt kê). Ví dụ:
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

        deterministic_sql_answer = await self._generate_time_sql_answer(
            question,
            model,
        )
        if deterministic_sql_answer is not None:
            answer, routed_model = deterministic_sql_answer
            return answer, [], routed_model, "mes_database"

        compound_highest_lot_answer = await self._generate_compound_highest_lot_answer(
            question,
            model,
        )
        if compound_highest_lot_answer is not None:
            answer, routed_model = compound_highest_lot_answer
            return answer, [], routed_model, "mes_database"

        sql_answer = await self._generate_sql_answer(question, model)
        if sql_answer is not None:
            answer, routed_model = sql_answer
            return answer, [], routed_model, "mes_database"

        if (
            self.is_highest_lot_error_question(question)
            and self.is_compound_mes_question(question)
            and not self.is_time_related_mes_question(question)
        ):
            snapshot_result = await self._query_database(
                question,
                allow_highest_lot=True,
            )
            if snapshot_result is not None:
                logger.info(
                    "Falling back compound highest Lot question to snapshot summary."
                )
                answer, routed_model = await self._generate_database_answer(
                    question,
                    snapshot_result,
                    model,
                )
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
        time_related_question = self.is_time_related_mes_question(question)
        if highest_lot_question and (
            self.is_compound_mes_question(question) or time_related_question
        ):
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

        routed_model = self.resolve_sql_agent_model(model)
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

    async def _generate_compound_highest_lot_answer(
        self,
        question: str,
        model: str,
    ) -> tuple[str, str] | None:
        if self.mes_sql_agent is None or not self.mes_sql_agent.available:
            return None
        if (
            not self.is_highest_lot_error_question(question)
            or not self.is_compound_mes_question(question)
            or self.is_time_related_mes_question(question)
        ):
            return None

        normalized = self.normalized_text(question)
        limit = self.extract_top_limit(normalized, default=3, maximum=20)
        sql = f"""
            WITH max_lot AS (
                SELECT MAX(total_error_qty) AS max_total_error_qty
                FROM v_lot_error_summary
            ),
            top_lot AS (
                SELECT s.lot_id, s.product_id,
                       s.total_error_qty AS lot_total_error_qty
                FROM v_lot_error_summary AS s
                JOIN max_lot AS m
                  ON s.total_error_qty = m.max_total_error_qty
            )
            SELECT b.lot_id, b.product_id,
                   t.lot_total_error_qty,
                   b.error_id, b.error_name,
                   SUM(b.total_error_qty) AS total_error_qty
            FROM v_lot_error_breakdown AS b
            JOIN top_lot AS t
              ON b.lot_id = t.lot_id
             AND b.product_id = t.product_id
            GROUP BY b.lot_id, b.product_id, t.lot_total_error_qty,
                     b.error_id, b.error_name
            ORDER BY total_error_qty DESC, b.error_id
            LIMIT {limit}
        """
        routed_model = self.resolve_model(model)
        try:
            result = await asyncio.to_thread(self.mes_sql_agent.execute, sql)
            logger.info(
                "MES deterministic compound highest Lot SQL executed rows=%s",
                len(result.rows),
            )
        except MesSqlAgentError as exc:
            logger.warning("MES deterministic compound SQL rejected: %s", exc)
            return None

        if routed_model in LOCAL_MODEL_ALIASES:
            return self.mes_sql_agent.fallback_answer(result), routed_model
        try:
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
        except Exception as exc:
            logger.warning("LLM deterministic compound MES answer failed: %s", exc)
        return self.mes_sql_agent.fallback_answer(result), routed_model

    async def _generate_time_sql_answer(
        self,
        question: str,
        model: str,
    ) -> tuple[str, str] | None:
        if self.mes_sql_agent is None or not self.mes_sql_agent.available:
            return None
        sql = self.time_sql_for_question(question)
        if not sql:
            return None
        routed_model = self.resolve_model(model)
        try:
            result = await asyncio.to_thread(self.mes_sql_agent.execute, sql)
            logger.info("MES deterministic time SQL executed rows=%s", len(result.rows))
        except MesSqlAgentError as exc:
            logger.warning("MES deterministic time SQL rejected: %s", exc)
            return None
        if routed_model in LOCAL_MODEL_ALIASES:
            return self.mes_sql_agent.fallback_answer(result), routed_model
        try:
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
        except Exception as exc:
            logger.warning("LLM deterministic MES time answer failed: %s", exc)
        return self.mes_sql_agent.fallback_answer(result), routed_model

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
                    f"Thông tin MES để trả lời:\n{payload}\n\n"
                    f"Câu trả lời kiểm chứng để tham khảo: {result.fallback_answer}\n"
                    "Hãy diễn đạt tự nhiên, không nhắc tới cấu trúc dữ liệu nội bộ."
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
            "json",
            "sql",
            "filters",
            "filter",
            "chính sách hiển thị",
            "chinh sach hien thi",
            "表示ポリシー",
            "フィルタ",
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
        normalized_numbers = answer.replace(".", "").replace(",", "")
        for row in result.rows[:5]:
            checked_row_value = False
            for key in ("lot_id", "product_id", "error_id"):
                value = row.get(key)
                checked_row_value = checked_row_value or bool(value)
                if value and str(value).lower() not in normalized:
                    return False
            error_name = row.get("error_name")
            checked_row_value = checked_row_value or bool(error_name)
            if error_name and str(error_name).strip():
                if str(error_name).lower() not in normalized:
                    return False
            elif "error_name" in row and "chưa rõ tên" not in normalized:
                return False
            if checked_row_value:
                continue

            for key, value in row.items():
                if value is None or value == "":
                    continue
                key_lower = key.lower()
                if any(
                    marker in key_lower
                    for marker in ("date", "time", "month", "day", "period")
                ):
                    if str(value).lower() not in normalized:
                        return False
                elif isinstance(value, (int, float)) and any(
                    marker in key_lower
                    for marker in (
                        "total",
                        "qty",
                        "quantity",
                        "count",
                        "sum",
                    )
                ):
                    compact_value = str(int(value) if float(value).is_integer() else value)
                    if compact_value not in normalized_numbers:
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

        has_lot = bool(re.search(r"\b(lot|lots|lo|lo san xuat)\b", normalized))
        has_error = bool(re.search(r"\bng\b", normalized)) or any(
            marker in normalized
            for marker in (
                "loi",
                "error",
                "errors",
                "defect",
                "defects",
                "hang loi",
                "san pham loi",
            )
        )
        has_maximum = bool(
            re.search(r"\b(nhieu|cao|lon)\b(?:\s+\w+){0,3}\s+nhat\b", normalized)
            or re.search(
                r"\btop\s*(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)?\b",
                normalized,
            )
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
                "highest",
                "largest",
                "greatest",
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
        number_token = (
            r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        )
        return bool(
            re.search(rf"\b{number_token}\s+(?:ma\s+|loai\s+)?loi\b", normalized)
            or re.search(
                rf"\btop\s*{number_token}\s+(?:ma\s+|loai\s+)?loi\b",
                normalized,
            )
            or re.search(
                rf"\btop\s*{number_token}\s+(?:error|defect)\s+(?:code|codes|type|types)\b",
                normalized,
            )
            or re.search(
                rf"\b{number_token}\s+(?:error|defect)\s+(?:code|codes|type|types)\b",
                normalized,
            )
            or any(
                marker in normalized
                for marker in ("cac loi nhieu nhat", "nhung loi nhieu nhat")
            )
            or (
                ("error code" in normalized or "error codes" in normalized)
                and any(marker in normalized for marker in ("top", "highest", "most"))
            )
        )

    @staticmethod
    def is_time_related_mes_question(question: str) -> bool:
        original = (question or "").lower()
        normalized = unicodedata.normalize(
            "NFD",
            original.replace("đ", "d"),
        )
        normalized = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
        has_explicit_date_value = bool(
            re.search(r"\b20\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?\b", original)
        )
        if has_explicit_date_value:
            return True
        return any(
            marker in normalized
            for marker in (
                "ngay",
                "hom nay",
                "hom qua",
                "theo ngay",
                "moi ngay",
                "thang",
                "theo thang",
                "moi thang",
                "nam",
                "tuan",
                "khoang thoi gian",
                "tu ngay",
                "den ngay",
                "gan day",
                "moi nhat",
                "date",
                "day",
                "daily",
                "today",
                "yesterday",
                "month",
                "monthly",
                "year",
                "yearly",
                "week",
                "weekly",
                "between",
                "from",
                "recent",
                "latest",
            )
        ) or any(
            marker in original
            for marker in ("今日", "昨日", "日", "月", "年", "期間", "いつ")
        )

    @classmethod
    def time_sql_for_question(cls, question: str) -> str:
        original = (question or "").lower()
        normalized = cls.normalized_text(question)
        if not cls.is_time_related_mes_question(question):
            return ""

        explicit_month = cls.extract_month(question)
        explicit_date = cls.extract_date(question)
        limit = cls.extract_top_limit(normalized)
        has_lot = bool(re.search(r"\b(lot|lots|lo|lo san xuat)\b", normalized))
        has_error = bool(
            re.search(r"\b(ng|loi|error|errors|defect|defects)\b", normalized)
            or any(marker in original for marker in ("エラー", "不良", "欠陥"))
        )
        asks_error_type = any(
            marker in normalized
            for marker in (
                "ma loi",
                "loai loi",
                "loi pho bien",
                "pho bien nhat",
                "error code",
                "error codes",
                "defect code",
                "defect codes",
                "top error",
                "top errors",
            )
        )
        asks_top = cls.has_top_marker(normalized) or any(
            marker in original for marker in ("上位", "最も", "多い", "最大")
        )
        asks_month = "thang" in normalized or "month" in normalized or "月" in original
        asks_day = "ngay" in normalized or "day" in normalized or "日" in original

        if explicit_date and has_lot and has_error and asks_top:
            return f"""
                SELECT lot_id, product_id, SUM(quantity) AS total_error_qty
                FROM v_error_details
                WHERE error_time >= '{explicit_date}'
                  AND error_time < date('{explicit_date}', '+1 day')
                GROUP BY lot_id, product_id
                ORDER BY total_error_qty DESC, lot_id
                LIMIT {limit}
            """

        if explicit_month and has_lot and has_error and asks_top:
            month_start = f"{explicit_month}-01"
            return f"""
                SELECT lot_id, product_id, SUM(quantity) AS total_error_qty
                FROM v_error_details
                WHERE error_time >= '{month_start}'
                  AND error_time < date('{month_start}', '+1 month')
                GROUP BY lot_id, product_id
                ORDER BY total_error_qty DESC, lot_id
                LIMIT {limit}
            """

        if explicit_month and asks_error_type and has_error:
            month_start = f"{explicit_month}-01"
            return f"""
                SELECT error_id, error_name, SUM(quantity) AS total_error_qty
                FROM v_error_details
                WHERE error_time >= '{month_start}'
                  AND error_time < date('{month_start}', '+1 month')
                GROUP BY error_id, error_name
                ORDER BY total_error_qty DESC, error_id
                LIMIT {limit}
            """

        if asks_month and asks_error_type and asks_top and has_error:
            return f"""
                WITH top_month AS (
                    SELECT strftime('%Y-%m', error_time) AS error_month
                    FROM v_error_details
                    WHERE error_time IS NOT NULL
                    GROUP BY strftime('%Y-%m', error_time)
                    ORDER BY SUM(quantity) DESC
                    LIMIT 1
                )
                SELECT t.error_month, e.error_id, e.error_name,
                       SUM(e.quantity) AS total_error_qty
                FROM v_error_details AS e
                JOIN top_month AS t
                  ON strftime('%Y-%m', e.error_time) = t.error_month
                GROUP BY t.error_month, e.error_id, e.error_name
                ORDER BY total_error_qty DESC, e.error_id
                LIMIT {limit}
            """

        if asks_month and has_error and asks_top and not has_lot:
            return """
                SELECT strftime('%Y-%m', error_time) AS error_month,
                       SUM(quantity) AS total_error_qty,
                       COUNT(*) AS error_record_count
                FROM v_error_details
                WHERE error_time IS NOT NULL
                GROUP BY strftime('%Y-%m', error_time)
                ORDER BY total_error_qty DESC
                LIMIT 1
            """

        if asks_day and has_error and asks_top and not has_lot:
            return """
                SELECT date(error_time) AS error_date,
                       SUM(quantity) AS total_error_qty,
                       COUNT(*) AS error_record_count
                FROM v_error_details
                WHERE error_time IS NOT NULL
                GROUP BY date(error_time)
                ORDER BY total_error_qty DESC
                LIMIT 1
            """
        return ""

    @staticmethod
    def normalized_text(question: str) -> str:
        normalized = unicodedata.normalize(
            "NFD",
            (question or "").lower().replace("đ", "d"),
        )
        normalized = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        return re.sub(r"[^a-z0-9]+", " ", normalized).strip()

    @staticmethod
    def extract_date(question: str) -> str:
        original = question or ""
        match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", original)
        if not match:
            return ""
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    @staticmethod
    def extract_month(question: str) -> str:
        original = question or ""
        match = re.search(r"\b(20\d{2})[-/](\d{1,2})(?:[-/]\d{1,2})?\b", original)
        if match:
            year, month = match.groups()
            return f"{year}-{int(month):02d}"
        normalized = MesQueryService.normalized_text(question)
        match = re.search(r"\bthang\s+(\d{1,2})\s+nam\s+(20\d{2})\b", normalized)
        if match:
            month, year = match.groups()
            return f"{year}-{int(month):02d}"
        match = re.search(r"\b(20\d{2})\s+nam\s+thang\s+(\d{1,2})\b", normalized)
        if match:
            year, month = match.groups()
            return f"{year}-{int(month):02d}"
        japanese_match = re.search(r"(20\d{2})年\s*(\d{1,2})月", original)
        if japanese_match:
            year, month = japanese_match.groups()
            return f"{year}-{int(month):02d}"
        return ""

    @staticmethod
    def extract_top_limit(normalized: str, default: int = 5, maximum: int = 50) -> int:
        match = re.search(r"\btop\s*(\d+)\b", normalized)
        if not match:
            match = re.search(
                r"\b(\d+)\s+(?:lot|lots|lo|ma loi|loai loi|error|errors)\b",
                normalized,
            )
        if not match:
            return default
        return max(1, min(maximum, int(match.group(1))))

    @staticmethod
    def has_top_marker(normalized: str) -> bool:
        return bool(
            re.search(r"\b(nhieu|cao|lon|pho bien)\b(?:\s+\w+){0,3}\s+nhat\b", normalized)
            or re.search(r"\btop\s*\d*\b", normalized)
        ) or any(
            marker in normalized
            for marker in (
                "nhieu nhat",
                "cao nhat",
                "lon nhat",
                "pho bien nhat",
                "dung dau",
                "max",
                "maximum",
                "most",
                "highest",
                "largest",
                "greatest",
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
    def resolve_sql_agent_model(model: str) -> str:
        forced_model = os.getenv("MES_SQL_AGENT_MODEL", "local-qwen-coder").strip()
        if forced_model:
            return MODEL_ROUTES.get(forced_model, forced_model)
        return MesQueryService.resolve_model(model)

    @staticmethod
    def provider_options(routed_model: str) -> dict[str, Any]:
        if routed_model in LOCAL_CHAT_MODEL_ALIASES:
            num_ctx = int(os.getenv("LOCAL_CHAT_NUM_CTX", "16384"))
            return {"extra_body": {"think": False, "num_ctx": num_ctx}}
        return {}
