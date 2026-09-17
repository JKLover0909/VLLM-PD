"""Service layer for CCTVAI deterministic replica queries.

Independent from MesQueryService — CCTVAI uses a different database engine
(PostgreSQL) and a different availability model (circuit breaker + health probe).

Mirrors the ``query_wms_outcome`` / ``query_wms_stream_outcome`` pattern from
``mes_query_service.py`` so that ``main.py`` can wire both in the same way.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, TYPE_CHECKING

from openai import AsyncOpenAI

from . import cctvai_contract as contract
from . import mes_answer_format
from .cctvai_database import CctvaiDatabase
from .cctvai_sql_agent import CctvaiSqlAgent, CctvaiSqlAgentError, CctvaiSqlQueryResult
from .mes_query_service import MesQueryOutcome, MesQueryStreamOutcome

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Intents that hand off to the SQL agent when one is enabled. Wired in Giai
# đoạn 3: the deterministic router only reaches cctvai_scope_clarification
# when no fixed template matched the question (fail-closed refusals like
# camera-credential/identity/open-event-duration questions are a SEPARATE
# per-intent reason code, not this one — they stay refused even when the SQL
# agent is enabled, because the semantic model excludes those columns/tables
# entirely, so the agent could not answer them either).
_SQL_FALLBACK_INTENTS = frozenset(
    {
        "cctvai_scope_clarification",
    }
)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read a bounded integer environment setting."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


class CctvaiQueryService:
    """Route and answer CCTVAI questions via deterministic DB templates.

    Dependency-injection pattern mirrors ``MesQueryService.__init__``: pass
    fake objects in tests; let ``from_env()`` build real ones in production.
    """

    def __init__(
        self,
        *,
        cctvai_database: CctvaiDatabase | None = None,
        cctvai_sql_agent: CctvaiSqlAgent | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        self.cctvai_database = (
            cctvai_database
            if cctvai_database is not None
            else CctvaiDatabase.from_env()
        )
        self.cctvai_sql_agent = (
            cctvai_sql_agent
            if cctvai_sql_agent is not None
            else CctvaiSqlAgent.from_env()
        )
        self.openai_client = openai_client or AsyncOpenAI(
            api_key=os.getenv("LITELLM_MASTER_KEY", "sk-local"),
            base_url=os.getenv("LITELLM_URL", "http://localhost:4000/v1"),
        )

    @classmethod
    def from_env(cls) -> "CctvaiQueryService | None":
        """Return None when CCTVAI_DATABASE_ENABLED is not truthy.

        Only reads env; never opens a connection. Safe to call at import time.
        """
        enabled = os.getenv("CCTVAI_DATABASE_ENABLED", "false").lower() in {
            "1", "true", "yes", "on",
        }
        if not enabled:
            return None
        return cls()

    @property
    def available(self) -> bool:
        """True only when the underlying database probe has succeeded."""
        return (
            self.cctvai_database is not None
            and self.cctvai_database.available
        )

    def status(self) -> dict:
        """Delegate to database status for /health; zero I/O."""
        if self.cctvai_database is None:
            return {"enabled": False, "available": False}
        return self.cctvai_database.status()

    async def query_cctvai_outcome(
        self,
        question: str,
        model: str = "openai",
        language: str = "vi",
    ) -> MesQueryOutcome:
        """Answer from deterministic CCTVAI templates.

        Returns a ``MesQueryOutcome`` with ``answer_scope="cctvai_database"``
        and ``cctvai_metadata`` populated so ``main.py`` can pass it through
        to ``QueryResponse`` without a new response type.
        """
        if self.cctvai_database is None:
            return MesQueryOutcome(
                answer=(
                    "Chức năng CCTVAI chưa được kích hoạt."
                    if language != "ja"
                    else "CCTVAI機能は有効化されていません。"
                ),
                results=[],
                routed_model=model,
                answer_scope="cctvai_database",
                cctvai_metadata={"enabled": False, "available": False},
            )

        db_result = await self.cctvai_database.query(question, language=language)

        if db_result.intent in _SQL_FALLBACK_INTENTS:
            sql_answer = await self._generate_cctvai_sql_answer(
                question, language=language
            )
            if sql_answer is not None:
                answer, metadata = sql_answer
                return MesQueryOutcome(
                    answer=answer,
                    results=[],
                    routed_model=model,
                    answer_scope="cctvai_database",
                    cctvai_metadata=metadata,
                )

        return MesQueryOutcome(
            answer=db_result.fallback_answer,
            results=[],
            routed_model=model,
            answer_scope="cctvai_database",
            cctvai_metadata=db_result.metadata_payload(),
        )

    async def _generate_cctvai_sql_answer(
        self,
        question: str,
        *,
        language: str,
    ) -> tuple[str, dict[str, Any]] | None:
        """Attempt an LLM-planned SQL answer when no deterministic intent matched.

        Mirrors ``MesQueryService._generate_wms_sql_answer``: bounded retry
        with the previous validation error fed back to the planner, then a
        natural-language answer pass verified against the result before use.
        Returns ``None`` when the agent/database is unavailable, the planner
        declines, or a plan/execution attempt fails — the caller falls back
        to the deterministic refusal in ``db_result.fallback_answer`` in that
        case. Only ``CctvaiSqlAgentError`` (a rejected plan) retries; any
        other exception (LLM/network/DB failure) gives up immediately, since
        a different SQL plan will not fix an infrastructure problem.
        """
        if (
            self.cctvai_sql_agent is None
            or not self.cctvai_sql_agent.available
            or self.cctvai_database is None
            or not self.cctvai_database.available
        ):
            return None

        routed_model = (
            os.getenv("CCTVAI_SQL_AGENT_MODEL", "local-qwen-coder").strip()
            or "local-qwen-coder"
        )
        max_attempts = _env_int(
            "CCTVAI_SQL_AGENT_MAX_ATTEMPTS", 2, minimum=1, maximum=3
        )
        max_rows = _env_int("CCTVAI_SQL_AGENT_MAX_ROWS", 50, minimum=1, maximum=200)
        previous_error = ""

        for attempt in range(max_attempts):
            try:
                response = await self.openai_client.chat.completions.create(
                    model=routed_model,
                    messages=self.cctvai_sql_agent.planner_messages(
                        question, previous_error=previous_error
                    ),
                    temperature=0,
                    max_tokens=_env_int(
                        "CCTVAI_SQL_PLANNER_MAX_TOKENS",
                        1200,
                        minimum=512,
                        maximum=1600,
                    ),
                )
                content = response.choices[0].message.content or ""
                plan = self.cctvai_sql_agent.parse_plan(content)
                if not plan.can_answer:
                    log.info("CCTVAI SQL planner cannot answer: %s", plan.reason)
                    return None

                safe_sql, tables, reason_codes = self.cctvai_sql_agent.validate_sql(
                    plan.sql
                )
                rows = await asyncio.to_thread(
                    self.cctvai_database.execute_ad_hoc, safe_sql
                )
                log.info(
                    "CCTVAI SQL agent executed attempt=%s rows=%s tables=%s",
                    attempt + 1,
                    len(rows),
                    tables,
                )
                result = CctvaiSqlQueryResult(
                    columns=list(rows[0].keys()) if rows else [],
                    rows=rows,
                    truncated=len(rows) >= max_rows,
                    latest_event_at=self.cctvai_database.status().get(
                        "latest_event_at", ""
                    ),
                    tables=tables,
                    reason_codes=reason_codes,
                )

                if result.is_empty():
                    answer = self.cctvai_sql_agent.fallback_answer(
                        result, language=language
                    )
                    return answer, self._sql_agent_metadata(result)

                answer_response = await self.openai_client.chat.completions.create(
                    model=routed_model,
                    messages=self.cctvai_sql_agent.answer_messages(
                        question, result, language=language
                    ),
                    temperature=0.1,
                    max_tokens=_env_int(
                        "CCTVAI_SQL_ANSWER_MAX_TOKENS",
                        512 if result.truncated or len(result.rows) > 10 else 384,
                        minimum=192,
                        maximum=800,
                    ),
                )
                candidate = mes_answer_format.normalize_sql_answer(
                    answer_response.choices[0].message.content or ""
                )
                if self.cctvai_sql_agent.answer_is_natural(
                    candidate
                ) and self.cctvai_sql_agent.answer_matches_result(candidate, result):
                    answer = candidate
                else:
                    answer = self.cctvai_sql_agent.fallback_answer(
                        result, language=language
                    )
                return answer, self._sql_agent_metadata(result)
            except CctvaiSqlAgentError as exc:
                previous_error = str(exc)
                log.warning(
                    "CCTVAI SQL plan rejected attempt=%s: %s", attempt + 1, exc
                )
            except Exception as exc:
                log.warning("CCTVAI SQL agent failed: %s", exc)
                return None
        return None

    @staticmethod
    def _sql_agent_metadata(result: CctvaiSqlQueryResult) -> dict[str, Any]:
        reason_codes = list(
            dict.fromkeys(
                [*result.reason_codes, contract.REASON_SQL_AGENT_UNVERIFIED]
            )
        )
        return {
            "intent": "cctvai_sql_agent",
            "domain": contract.DOMAIN,
            "status": "PARTIAL",
            "reason_codes": reason_codes,
            "latest_event_at": result.latest_event_at,
            "replica_lag_state": contract.REPLICA_LAG_STATE_UNVERIFIED,
            "grain": "sql_agent",
            "schema_version": contract.SCHEMA_VERSION,
            "data_contract_version": contract.DATA_CONTRACT_VERSION,
            "semantic_contract_version": contract.SEMANTIC_CONTRACT_VERSION,
            "source_system": contract.SOURCE_SYSTEM,
        }

    async def query_cctvai_stream_outcome(
        self,
        question: str,
        model: str = "openai",
        language: str = "vi",
    ) -> MesQueryStreamOutcome:
        """Wrap the non-streaming outcome in a one-token generator.

        Same pattern as ``query_wms_stream_outcome`` (mes_query_service.py:290-311).
        The SSE event_generator unwraps the token stream correctly for all modes.
        """
        outcome = await self.query_cctvai_outcome(question, model, language=language)

        async def token_generator():
            yield ("token", outcome.answer)

        return MesQueryStreamOutcome(
            token_stream=token_generator(),
            results=outcome.results,
            routed_model=outcome.routed_model,
            answer_scope=outcome.answer_scope,
            cctvai_metadata=outcome.cctvai_metadata,
        )
