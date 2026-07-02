"""MES query service separated from document RAG routing.

Phần điều phối (routing + gọi LLM) nằm ở lớp ``MesQueryService`` dưới đây. Các
hàm phụ trợ thuần túy đã được tách sang các module chuyên biệt để dễ phát triển:

* ``mes_intent``        – nhận diện ý định và sinh SQL tất định theo thời gian.
* ``mes_prompts``       – system prompt và message builder.
* ``mes_answer_format`` – định dạng fallback và kiểm chứng đầu ra của model.
* ``mes_config``        – định tuyến model và ngân sách token.

Để giữ nguyên API công khai (``MesQueryService.<helper>(...)`` mà các test và
đoạn code khác đang dùng), lớp này gắn lại các hàm module thành ``staticmethod``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

from src.integrations.mes_client import MesClient, MesLotError
from src.integrations.mes_database import MesDatabase, MesDatabaseError, MesDatabaseResult
from src.integrations.mes_sql_agent import MesSqlAgent, MesSqlAgentError, MesSqlQueryResult
from src.integrations import mes_answer_format, mes_config, mes_intent, mes_prompts
from src.integrations.mes_config import (
    LOCAL_CHAT_MODEL_ALIASES,
    LOCAL_MODEL_ALIASES,
    MODEL_ROUTES,
    env_int,
)
from src.integrations.mes_prompts import (
    MES_DATABASE_SYSTEM_PROMPT,
    MES_GENERAL_FALLBACK_ANSWER,
    MES_GENERAL_SYSTEM_PROMPT,
    MES_SYSTEM_PROMPT,
    MES_UNSUPPORTED_ANSWER,
)

logger = logging.getLogger(__name__)


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

        if self.should_use_sql_agent(question):
            sql_answer = await self._generate_sql_answer(question, model)
            if sql_answer is not None:
                answer, routed_model = sql_answer
                return answer, [], routed_model, "mes_database"
        else:
            logger.info("Skipping MES SQL agent for non-data question.")
            answer, routed_model = await self._generate_general_mes_answer(
                question,
                model,
            )
            return answer, [], routed_model, "mes"

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

    async def _generate_general_mes_answer(
        self,
        question: str,
        model: str,
    ) -> tuple[str, str]:
        routed_model = self.resolve_model(model)
        fallback_answer = MES_GENERAL_FALLBACK_ANSWER
        if routed_model in LOCAL_MODEL_ALIASES:
            return fallback_answer, routed_model
        try:
            response = await self.openai_client.chat.completions.create(
                model=routed_model,
                messages=[
                    {"role": "system", "content": MES_GENERAL_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.1,
                max_tokens=self.general_answer_max_tokens(),
                **self.provider_options(routed_model),
            )
            candidate = (response.choices[0].message.content or "").strip()
            return candidate or fallback_answer, routed_model
        except Exception as exc:
            logger.warning("MES general answer generation failed: %s", exc)
            return fallback_answer, routed_model

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
        if self.prefer_template_answers(routed_model):
            return fallback_answer
        try:
            response = await self.openai_client.chat.completions.create(
                model=routed_model,
                messages=self.live_api_messages(question, lots),
                temperature=0.1,
                max_tokens=self.live_api_answer_max_tokens(),
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
        if self.prefer_template_answers(routed_model):
            return result.fallback_answer, routed_model
        try:
            response = await self.openai_client.chat.completions.create(
                model=routed_model,
                messages=self.database_messages(question, result),
                temperature=0.1,
                max_tokens=self.database_answer_max_tokens(result),
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
                    max_tokens=self.sql_planner_max_tokens(),
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
                    max_tokens=self.sql_answer_max_tokens(result),
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
                max_tokens=self.sql_answer_max_tokens(result),
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
                max_tokens=self.sql_answer_max_tokens(result),
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

    # ──────────────────────────────────────────────────────────────────
    # Delegators giữ nguyên API công khai. Logic thật nằm ở các module
    # mes_prompts / mes_answer_format / mes_intent / mes_config.
    # ──────────────────────────────────────────────────────────────────

    # mes_prompts
    live_api_messages = staticmethod(mes_prompts.live_api_messages)
    database_messages = staticmethod(mes_prompts.database_messages)

    # mes_answer_format
    format_live_api_fallback = staticmethod(mes_answer_format.format_live_api_fallback)
    live_api_answer_has_required_fields = staticmethod(
        mes_answer_format.live_api_answer_has_required_fields
    )
    database_answer_has_required_terms = staticmethod(
        mes_answer_format.database_answer_has_required_terms
    )
    normalize_sql_answer = staticmethod(mes_answer_format.normalize_sql_answer)
    sql_answer_matches_result = staticmethod(mes_answer_format.sql_answer_matches_result)
    sql_answer_is_natural = staticmethod(mes_answer_format.sql_answer_is_natural)

    # mes_intent
    is_highest_lot_error_question = staticmethod(mes_intent.is_highest_lot_error_question)
    is_compound_mes_question = staticmethod(mes_intent.is_compound_mes_question)
    is_time_related_mes_question = staticmethod(mes_intent.is_time_related_mes_question)
    time_sql_for_question = staticmethod(mes_intent.time_sql_for_question)
    normalized_text = staticmethod(mes_intent.normalized_text)
    should_use_sql_agent = staticmethod(mes_intent.should_use_sql_agent)
    extract_date = staticmethod(mes_intent.extract_date)
    extract_month = staticmethod(mes_intent.extract_month)
    extract_top_limit = staticmethod(mes_intent.extract_top_limit)
    has_top_marker = staticmethod(mes_intent.has_top_marker)

    # mes_config
    resolve_model = staticmethod(mes_config.resolve_model)
    resolve_sql_agent_model = staticmethod(mes_config.resolve_sql_agent_model)
    provider_options = staticmethod(mes_config.provider_options)
    general_answer_max_tokens = staticmethod(mes_config.general_answer_max_tokens)
    live_api_answer_max_tokens = staticmethod(mes_config.live_api_answer_max_tokens)
    database_answer_max_tokens = staticmethod(mes_config.database_answer_max_tokens)
    sql_planner_max_tokens = staticmethod(mes_config.sql_planner_max_tokens)
    sql_answer_max_tokens = staticmethod(mes_config.sql_answer_max_tokens)
    prefer_template_answers = staticmethod(mes_config.prefer_template_answers)
