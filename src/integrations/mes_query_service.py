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
from dataclasses import dataclass
import inspect
import logging
import os
from typing import AsyncGenerator

from openai import AsyncOpenAI

from src.integrations.mes_client import MesClient, MesLotError
from src.integrations.mes_database import MesDatabase, MesDatabaseError, MesDatabaseResult
from src.integrations.mes_sql_agent import MesSqlAgent, MesSqlAgentError
from src.integrations.mes_wms_database import (
    MesWmsDatabase,
    MesWmsDatabaseError,
    MesWmsDatabaseResult,
)
from src.integrations.mes_wms_contract import (
    REASON_SNAPSHOT_QUERY_ERROR,
    REASON_SNAPSHOT_UNAVAILABLE,
    REASON_WMS_DISABLED,
)
from src.integrations.wms_sql_agent import (
    WmsSqlAgent,
    WmsSqlAgentError,
    WmsSqlQueryResult,
)
from src.integrations import mes_answer_format, mes_config, mes_intent, mes_prompts
from src.integrations.mes_config import (
    LOCAL_MODEL_ALIASES,
)
from src.integrations.mes_prompts import (
    MES_GENERAL_FALLBACK_ANSWER,
    MES_GENERAL_SYSTEM_PROMPT,
    MES_UNSUPPORTED_ANSWER,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MesQueryOutcome:
    answer: str
    results: list
    routed_model: str
    answer_scope: str
    wms_metadata: dict | None = None

    def as_tuple(self) -> tuple[str, list, str, str]:
        return self.answer, self.results, self.routed_model, self.answer_scope


@dataclass(frozen=True)
class MesQueryStreamOutcome:
    token_stream: AsyncGenerator[tuple[str, str], None]
    results: list
    routed_model: str
    answer_scope: str
    wms_metadata: dict | None = None

    def as_tuple(
        self,
    ) -> tuple[AsyncGenerator[tuple[str, str], None], list, str, str]:
        return (
            self.token_stream,
            self.results,
            self.routed_model,
            self.answer_scope,
        )


class MesQueryService:
    """Route and answer MES questions without touching document retrieval."""

    def __init__(
        self,
        *,
        mes_client: MesClient | None = None,
        mes_database: MesDatabase | None = None,
        mes_sql_agent: MesSqlAgent | None = None,
        mes_wms_database: MesWmsDatabase | None = None,
        wms_sql_agent: WmsSqlAgent | None = None,
        openai_client: AsyncOpenAI | None = None,
    ):
        self.mes_client = mes_client if mes_client is not None else MesClient.from_env()
        self.mes_database = (
            mes_database if mes_database is not None else MesDatabase.from_env()
        )
        self.mes_wms_database = (
            mes_wms_database
            if mes_wms_database is not None
            else MesWmsDatabase.from_env()
        )
        self.mes_sql_agent = (
            mes_sql_agent if mes_sql_agent is not None else MesSqlAgent.from_env()
        )
        self.wms_sql_agent = (
            wms_sql_agent if wms_sql_agent is not None else WmsSqlAgent.from_env()
        )
        self.openai_client = openai_client or AsyncOpenAI(
            api_key=os.getenv("LITELLM_MASTER_KEY", "sk-local"),
            base_url=os.getenv("LITELLM_URL", "http://localhost:4000/v1"),
        )

    async def query(
        self,
        question: str,
        model: str = "openai",
        language: str = "vi",
    ) -> tuple[str, list, str, str]:
        return (
            await self.query_outcome(question, model, language=language)
        ).as_tuple()

    async def query_outcome(
        self,
        question: str,
        model: str = "openai",
        language: str = "vi",
    ) -> MesQueryOutcome:
        mes_source, mes_data = await self.get_route(question)
        if mes_source == "mes":
            mes_lots = mes_data
            assert isinstance(mes_lots, list)
            routed_model = self.resolve_model(model)
            answer = await self._generate_live_api_answer(
                question, mes_lots, routed_model
            )
            return MesQueryOutcome(answer, [], routed_model, "mes")

        if mes_source == "mes_database":
            assert isinstance(mes_data, MesDatabaseResult)
            answer, routed_model = await self._generate_database_answer(
                question, mes_data, model
            )
            return MesQueryOutcome(answer, [], routed_model, "mes_database")

        deterministic_sql_answer = await self._generate_time_sql_answer(
            question, model
        )
        if deterministic_sql_answer is not None:
            answer, routed_model = deterministic_sql_answer
            return MesQueryOutcome(answer, [], routed_model, "mes_database")

        compound = await self._generate_compound_highest_lot_answer(
            question, model
        )
        if compound is not None:
            answer, routed_model = compound
            return MesQueryOutcome(answer, [], routed_model, "mes_database")

        if self.should_use_sql_agent(question):
            sql_answer = await self._generate_sql_answer(question, model)
            if sql_answer is not None:
                answer, routed_model = sql_answer
                return MesQueryOutcome(answer, [], routed_model, "mes_database")
        else:
            logger.info("Skipping MES SQL agent for non-data question.")
            answer, routed_model = await self._generate_general_mes_answer(
                question, model
            )
            return MesQueryOutcome(answer, [], routed_model, "mes")

        if (
            self.is_highest_lot_error_question(question)
            and self.is_compound_mes_question(question)
            and not self.is_time_related_mes_question(question)
        ):
            snapshot_result = await self._query_database(
                question, allow_highest_lot=True
            )
            if snapshot_result is not None:
                logger.info(
                    "Falling back compound highest Lot question to snapshot summary."
                )
                answer, routed_model = await self._generate_database_answer(
                    question, snapshot_result, model
                )
                return MesQueryOutcome(
                    answer, [], routed_model, "mes_database"
                )

        return MesQueryOutcome(
            MES_UNSUPPORTED_ANSWER,
            [],
            self.resolve_model(model),
            "mes_database",
        )

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
        language: str = "vi",
    ) -> tuple[AsyncGenerator[tuple[str, str], None], list, str, str]:
        return (
            await self.query_stream_outcome(question, model, language=language)
        ).as_tuple()

    async def query_wms_outcome(
        self,
        question: str,
        model: str = "openai",
        language: str = "vi",
    ) -> MesQueryOutcome:
        """Answer from deterministic WMS routes, then the WMS SQL agent."""
        wms_result = await self._query_wms_database(
            question,
            language=language,
            force=True,
        )
        assert wms_result is not None
        routed_model = self.resolve_model(model)
        sql_fallback_intents = {
            "wms_scope_clarification",
            "wms_cross_item_aggregate_suppressed",
            "wms_completed_movements_suppressed",
            "wms_wip_ambiguity",
            "wms_current_lot_lookup_suppressed",
            "wms_cross_era_presence_unobserved",
        }
        if (
            wms_result.intent in sql_fallback_intents
            or wms_result.intent.endswith("_suppressed")
        ):
            sql_result = await self._generate_wms_sql_answer(
                question,
                model,
                language=language,
            )
            if sql_result is not None:
                answer, routed_model, query_result = sql_result
                assert self.mes_wms_database is not None
                wms_result = self.mes_wms_database.sql_agent_result(
                    query_result.rows,
                    answer,
                    domain=query_result.domain,
                    evidence_domains=query_result.evidence_domains,
                )
        return MesQueryOutcome(
            answer=wms_result.fallback_answer,
            results=[],
            routed_model=routed_model,
            answer_scope="wms_database",
            wms_metadata=wms_result.metadata_payload(),
        )

    async def query_wms_stream_outcome(
        self,
        question: str,
        model: str = "openai",
        language: str = "vi",
    ) -> MesQueryStreamOutcome:
        outcome = await self.query_wms_outcome(
            question,
            model,
            language=language,
        )

        async def token_generator():
            yield ("token", outcome.answer)

        return MesQueryStreamOutcome(
            token_stream=token_generator(),
            results=outcome.results,
            routed_model=outcome.routed_model,
            answer_scope=outcome.answer_scope,
            wms_metadata=outcome.wms_metadata,
        )

    async def query_stream_outcome(
        self,
        question: str,
        model: str = "openai",
        language: str = "vi",
    ) -> MesQueryStreamOutcome:
        outcome = await self.query_outcome(question, model, language=language)

        async def token_generator():
            yield ("token", outcome.answer)

        return MesQueryStreamOutcome(
            token_stream=token_generator(),
            results=outcome.results,
            routed_model=outcome.routed_model,
            answer_scope=outcome.answer_scope,
            wms_metadata=outcome.wms_metadata,
        )

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

    async def _generate_wms_sql_answer(
        self,
        question: str,
        model: str,
        *,
        language: str,
    ) -> tuple[str, str, WmsSqlQueryResult] | None:
        if self.wms_sql_agent is None or not self.wms_sql_agent.available:
            return None

        routed_model = self.resolve_wms_sql_agent_model(model)
        previous_error = ""
        max_attempts = max(
            1,
            min(int(os.getenv("WMS_SQL_AGENT_MAX_ATTEMPTS", "2")), 3),
        )
        for attempt in range(max_attempts):
            try:
                response = await self.openai_client.chat.completions.create(
                    model=routed_model,
                    messages=self.wms_sql_agent.planner_messages(
                        question,
                        previous_error=previous_error,
                    ),
                    temperature=0,
                    max_tokens=self.wms_sql_planner_max_tokens(),
                    **self.provider_options(routed_model),
                )
                content = response.choices[0].message.content or ""
                plan = self.wms_sql_agent.parse_plan(content)
                if not plan.can_answer:
                    logger.info("WMS SQL planner cannot answer: %s", plan.reason)
                    return None
                result = await asyncio.to_thread(
                    self.wms_sql_agent.execute,
                    plan.sql,
                )
                logger.info(
                    "WMS SQL agent executed attempt=%s rows=%s domain=%s",
                    attempt + 1,
                    len(result.rows),
                    result.domain,
                )
                if result.is_empty():
                    answer = self.wms_sql_agent.fallback_answer(
                        result,
                        language=language,
                    )
                    return answer, routed_model, result

                answer_response = await self.openai_client.chat.completions.create(
                    model=routed_model,
                    messages=self.wms_sql_agent.answer_messages(
                        question,
                        result,
                        language=language,
                    ),
                    temperature=0.1,
                    max_tokens=self.wms_sql_answer_max_tokens(result),
                    **self.provider_options(routed_model),
                )
                candidate = self.normalize_sql_answer(
                    answer_response.choices[0].message.content or ""
                )
                if self.wms_sql_agent.answer_is_natural(
                    candidate
                ) and self.wms_sql_agent.answer_matches_result(candidate, result):
                    return candidate, routed_model, result
                answer = self.wms_sql_agent.fallback_answer(
                    result,
                    language=language,
                )
                return answer, routed_model, result
            except WmsSqlAgentError as exc:
                previous_error = str(exc)
                logger.warning(
                    "WMS SQL plan rejected attempt=%s: %s",
                    attempt + 1,
                    exc,
                )
            except Exception as exc:
                logger.warning("WMS SQL agent failed: %s", exc)
                return None
        return None

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
                if result.is_effectively_empty():
                    return self.mes_sql_agent.fallback_answer(result), routed_model
                if self.prefer_confident_template() and (
                    self.mes_sql_agent.has_confident_template(result)
                ):
                    logger.info(
                        "MES SQL answer served from confident template "
                        "(skip LLM reword)."
                    )
                    return self.mes_sql_agent.fallback_answer(result), routed_model
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

        if result.is_effectively_empty():
            return self.mes_sql_agent.fallback_answer(result), routed_model
        if routed_model in LOCAL_MODEL_ALIASES or (
            self.prefer_confident_template()
            and self.mes_sql_agent.has_confident_template(result)
        ):
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
        if result.is_effectively_empty():
            return self.mes_sql_agent.fallback_answer(result), routed_model
        if routed_model in LOCAL_MODEL_ALIASES or (
            self.prefer_confident_template()
            and self.mes_sql_agent.has_confident_template(result)
        ):
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

    async def _query_wms_database(
        self,
        question: str,
        *,
        language: str,
        force: bool = False,
    ) -> MesWmsDatabaseResult | None:
        if not force and not MesWmsDatabase.is_wms_question(question):
            return None
        if self.mes_wms_database is None:
            enabled = os.getenv(
                "MES_WMS_DATABASE_ENABLED", "false"
            ).lower() in {"1", "true", "yes", "on"}
            reason_code = (
                REASON_SNAPSHOT_UNAVAILABLE
                if enabled
                else REASON_WMS_DISABLED
            )
            answer = (
                "WMS機能が無効、またはスナップショットが利用できないため、"
                "他のMESデータから在庫数量を推測しません。"
                if language == "ja"
                else (
                    "WMS đang tắt hoặc snapshot chưa sẵn sàng nên tôi không suy đoán "
                    "số liệu tồn kho từ MES, SQL Agent hay nguồn khác."
                )
            )
            return MesWmsDatabaseResult(
                intent="wms_disabled" if not enabled else "wms_unavailable",
                rows=[],
                imported_at="",
                source_as_of="",
                fallback_answer=answer,
                status="SUPPRESSED",
                reason_codes=(reason_code,),
                domain="SUPPRESSED",
            )
        try:
            query_kwargs = {"language": language}
            parameters = inspect.signature(
                self.mes_wms_database.query_question
            ).parameters
            if "assume_wms" in parameters:
                query_kwargs["assume_wms"] = force
            result = await asyncio.to_thread(
                self.mes_wms_database.query_question,
                question,
                **query_kwargs,
            )
            if result is not None:
                logger.info(
                    "Routing MES question to deterministic WMS intent=%s status=%s",
                    result.intent,
                    result.status,
                )
            return result
        except MesWmsDatabaseError as exc:
            logger.warning("WMS snapshot query failed closed: %s", exc)
            return MesWmsDatabaseResult(
                intent="wms_query_error",
                rows=[],
                imported_at="",
                source_as_of="",
                fallback_answer=(
                    "WMSスナップショットを検証できないため、在庫数量は回答しません。"
                    if language == "ja"
                    else (
                        "Không thể xác minh WMS snapshot nên tôi không trả số liệu tồn kho."
                    )
                ),
                status="SUPPRESSED",
                reason_codes=(REASON_SNAPSHOT_QUERY_ERROR,),
                domain="SUPPRESSED",
            )

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
    resolve_wms_sql_agent_model = staticmethod(mes_config.resolve_wms_sql_agent_model)
    provider_options = staticmethod(mes_config.provider_options)
    general_answer_max_tokens = staticmethod(mes_config.general_answer_max_tokens)
    live_api_answer_max_tokens = staticmethod(mes_config.live_api_answer_max_tokens)
    database_answer_max_tokens = staticmethod(mes_config.database_answer_max_tokens)
    sql_planner_max_tokens = staticmethod(mes_config.sql_planner_max_tokens)
    sql_answer_max_tokens = staticmethod(mes_config.sql_answer_max_tokens)
    wms_sql_planner_max_tokens = staticmethod(mes_config.wms_sql_planner_max_tokens)
    wms_sql_answer_max_tokens = staticmethod(mes_config.wms_sql_answer_max_tokens)
    prefer_template_answers = staticmethod(mes_config.prefer_template_answers)
    prefer_confident_template = staticmethod(mes_config.prefer_confident_template)
