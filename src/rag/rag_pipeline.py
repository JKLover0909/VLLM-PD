"""
src/rag/rag_pipeline.py
-----------------------
Điều phối toàn bộ RAG pipeline cho Máy 2:
Nhận câu hỏi -> Embed -> Tìm kiếm ngữ nghĩa trong Qdrant -> Tạo prompt đa phương thức -> Gọi LiteLLM.
"""

import asyncio
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import AsyncGenerator, Tuple, List, Dict, Any
from openai import AsyncOpenAI

from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore, SearchResult
from src.rag.web_search import WebSearcher
from src.integrations.mes_client import MesClient, MesLotError
from src.integrations.mes_database import (
    MesDatabase,
    MesDatabaseResult,
    MesDatabaseError,
)
from src.integrations.mes_sql_agent import (
    MesSqlAgent,
    MesSqlAgentError,
    MesSqlQueryResult,
)
from src.integrations.mes_query_service import MesQueryService
from src.rag.prompts import (
    GENERAL_SYSTEM_PROMPT,
    MES_DATABASE_SYSTEM_PROMPT,
    MES_SYSTEM_PROMPT,
    build_rag_prompt,
)

logger = logging.getLogger(__name__)

MES_UNSUPPORTED_ANSWER = (
    "Chưa nhận diện được truy vấn MES này. Bạn có thể hỏi về thông tin một Lot, "
    "chi tiết lỗi theo Lot, tên mã lỗi hoặc thống kê lỗi theo mã hàng."
)


def env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 4096) -> int:
    """Read a bounded integer environment setting."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


MODEL_ROUTES = {
    "auto": "auto-model",
    "local": "local-qwen-chat",
    "openai": "openai-model",
    "grok": "grok-model",
}

LOCAL_CHAT_MODEL_ALIASES = {"auto-model", "local-gemma", "local-qwen-chat"}
LOCAL_MODEL_ALIASES = LOCAL_CHAT_MODEL_ALIASES | {"local-qwen-coder", "coding-model"}
PRIMARY_CHAT_MODELS = {"auto-model", "local-qwen-chat"}
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "openai-model").strip() or "openai-model"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class RAGPipeline:
    """
    Điều phối RAG: Nhận câu hỏi -> Embed -> Tìm kiếm ngữ nghĩa -> Gọi LiteLLM.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        mkac_vector_store: VectorStore | None = None,
        docjp_vector_store: VectorStore | None = None,
        web_searcher: WebSearcher | None = None,
        mes_client: MesClient | None = None,
        mes_database: MesDatabase | None = None,
        mes_sql_agent: MesSqlAgent | None = None,
        top_k: int = 5,
        score_threshold: float = 0.25,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.mkac_vector_store = mkac_vector_store
        self.docjp_vector_store = docjp_vector_store
        self.web_searcher = web_searcher
        self.mes_client = mes_client if mes_client is not None else MesClient.from_env()
        self.mes_database = (
            mes_database if mes_database is not None else MesDatabase.from_env()
        )
        self.mes_sql_agent = (
            mes_sql_agent if mes_sql_agent is not None else MesSqlAgent.from_env()
        )
        self.mes_query_service = MesQueryService(
            mes_client=self.mes_client,
            mes_database=self.mes_database,
            mes_sql_agent=self.mes_sql_agent,
            openai_client=None,
        )
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mkac_score_threshold = float(
            os.getenv("MKAC_SCORE_THRESHOLD", "0.38")
        )

        # Kết nối tới LiteLLM Proxy
        proxy_url = os.getenv("LITELLM_URL", "http://localhost:4000/v1")
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("LITELLM_MASTER_KEY", "sk-local"),
            base_url=proxy_url
        )

    async def query(
        self,
        session_id: str,
        question: str,
        model: str = "openai",
        mode: str = "mkac",
        current_user: Dict[str, Any] | None = None,
        conversation_context: List[Dict[str, Any]] | None = None,
        research_topic: str | None = None,
        research_scope: str | None = None,
    ) -> Tuple[str, List[SearchResult], str, str]:
        """
        Non-streaming RAG query.
        """
        if mode == "mes":
            return await self.mes_query_service.query(question, model)

        mes_source, mes_data = await self._get_mes_route(question, mode)
        if mes_source == "mes":
            mes_lots = mes_data
            assert isinstance(mes_lots, list)
            routed_model = self._resolve_model(model, mode=mode)
            fallback_answer = self._format_mes_fallback(mes_lots)
            try:
                response = await self.openai_client.chat.completions.create(
                    model=routed_model,
                    messages=self._mes_messages(question, mes_lots),
                    temperature=0.1,
                    max_tokens=self._mes_live_api_max_tokens(),
                    **self._provider_options(routed_model),
                )
                candidate = response.choices[0].message.content or ""
                answer = (
                    candidate
                    if self._mes_answer_has_required_fields(candidate, mes_lots)
                    else fallback_answer
                )
            except Exception as exc:
                logger.warning("LLM MES answer generation failed: %s", exc)
                answer = fallback_answer
            return answer, [], routed_model, "mes"
        if mes_source == "mes_database":
            assert isinstance(mes_data, MesDatabaseResult)
            answer, routed_model = await self._generate_mes_database_answer(
                question,
                mes_data,
                model,
                mode,
            )
            return answer, [], routed_model, "mes_database"
        if mode == "mes":
            sql_answer = await self._generate_mes_sql_answer(question, model, mode)
            if sql_answer is not None:
                answer, routed_model = sql_answer
                return answer, [], routed_model, "mes_database"
            return (
                MES_UNSUPPORTED_ANSWER,
                [],
                self._resolve_model(model, mode=mode),
                "mes_database",
            )

        search_results, image_paths, answer_scope = await asyncio.to_thread(
            self._prepare_query_context,
            session_id,
            question,
            mode,
            research_topic,
            research_scope,
        )
        if mode == "mkac" and current_user:
            if self._is_current_company_question(question):
                answer_scope = "mkac"
                search_results = []
                image_paths = []
            elif self._has_employee_directory_context(current_user):
                answer_scope = "mkac"
                search_results = []
                image_paths = []
            elif self._is_current_user_question(question):
                if answer_scope in {"general", "web"}:
                    answer_scope = "mkac"
                    search_results = []
                    image_paths = []

        messages = (
            build_rag_prompt(
                question,
                search_results,
                mode=mode,
                image_paths=image_paths,
                answer_scope=answer_scope,
                current_user=current_user,
                conversation_context=conversation_context,
            )
            if answer_scope != "general" or current_user
            else self._general_messages(question)
        )
        routed_model = self._resolve_model(
            model,
            has_images=bool(image_paths),
            mode=mode,
        )

        try:
            max_tokens = self._rag_answer_max_tokens(
                question=question,
                mode=mode,
                search_results=search_results,
                answer_scope=answer_scope,
                has_images=bool(image_paths),
            )
            response = await self.openai_client.chat.completions.create(
                model=routed_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                **self._provider_options(routed_model),
            )
            answer = self._clean_model_answer(response.choices[0].message.content or "")
            if not answer and self._should_retry_with_openai(routed_model):
                logger.warning(
                    "Model %s returned an empty answer; retrying with %s.",
                    routed_model,
                    OPENAI_FALLBACK_MODEL,
                )
                response = await self.openai_client.chat.completions.create(
                    model=OPENAI_FALLBACK_MODEL,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    **self._provider_options(OPENAI_FALLBACK_MODEL),
                )
                routed_model = OPENAI_FALLBACK_MODEL
                answer = self._clean_model_answer(response.choices[0].message.content or "")
            return answer, search_results, routed_model, answer_scope
        except Exception as e:
            logger.error(f"Error in RAG generation: {e}")
            raise e

    async def query_stream(
        self,
        session_id: str,
        question: str,
        model: str = "openai",
        mode: str = "mkac",
        current_user: Dict[str, Any] | None = None,
        conversation_context: List[Dict[str, Any]] | None = None,
        research_topic: str | None = None,
        research_scope: str | None = None,
    ) -> Tuple[AsyncGenerator[Tuple[str, str], None], List[SearchResult], str, str]:
        """
        Streaming RAG query.
        """
        if mode == "mes":
            return await self.mes_query_service.query_stream(question, model)

        mes_source, mes_data = await self._get_mes_route(question, mode)
        if mes_source == "mes":
            mes_lots = mes_data
            assert isinstance(mes_lots, list)
            routed_model = self._resolve_model(model, mode=mode)
            fallback_answer = self._format_mes_fallback(mes_lots)
            try:
                response = await self.openai_client.chat.completions.create(
                    model=routed_model,
                    messages=self._mes_messages(question, mes_lots),
                    temperature=0.1,
                    max_tokens=self._mes_live_api_max_tokens(),
                    **self._provider_options(routed_model),
                )
                candidate = response.choices[0].message.content or ""
                answer = (
                    candidate
                    if self._mes_answer_has_required_fields(candidate, mes_lots)
                    else fallback_answer
                )

                async def mes_token_generator():
                    yield ("token", answer)

                return mes_token_generator(), [], routed_model, "mes"
            except Exception as exc:
                logger.warning("Streaming LLM MES answer generation failed: %s", exc)

                async def fallback_token_generator():
                    yield ("token", fallback_answer)

                return fallback_token_generator(), [], routed_model, "mes"
        if mes_source == "mes_database":
            assert isinstance(mes_data, MesDatabaseResult)
            answer, routed_model = await self._generate_mes_database_answer(
                question,
                mes_data,
                model,
                mode,
            )

            async def mes_database_token_generator():
                yield ("token", answer)

            return (
                mes_database_token_generator(),
                [],
                routed_model,
                "mes_database",
            )
        if mode == "mes":
            sql_answer = await self._generate_mes_sql_answer(question, model, mode)
            if sql_answer is not None:
                answer, routed_model = sql_answer

                async def mes_sql_token_generator():
                    yield ("token", answer)

                return (
                    mes_sql_token_generator(),
                    [],
                    routed_model,
                    "mes_database",
                )

            routed_model = self._resolve_model(model, mode=mode)

            async def unsupported_mes_token_generator():
                yield ("token", MES_UNSUPPORTED_ANSWER)

            return (
                unsupported_mes_token_generator(),
                [],
                routed_model,
                "mes_database",
            )

        search_results, image_paths, answer_scope = await asyncio.to_thread(
            self._prepare_query_context,
            session_id,
            question,
            mode,
            research_topic,
            research_scope,
        )
        if mode == "mkac" and current_user:
            if self._is_current_company_question(question):
                answer_scope = "mkac"
                search_results = []
                image_paths = []
            elif self._has_employee_directory_context(current_user):
                answer_scope = "mkac"
                search_results = []
                image_paths = []
            elif self._is_current_user_question(question):
                if answer_scope in {"general", "web"}:
                    answer_scope = "mkac"
                    search_results = []
                    image_paths = []

        messages = (
            build_rag_prompt(
                question,
                search_results,
                mode=mode,
                image_paths=image_paths,
                answer_scope=answer_scope,
                current_user=current_user,
                conversation_context=conversation_context,
            )
            if answer_scope != "general" or current_user
            else self._general_messages(question)
        )
        routed_model = self._resolve_model(
            model,
            has_images=bool(image_paths),
            mode=mode,
        )

        try:
            max_tokens = self._rag_answer_max_tokens(
                question=question,
                mode=mode,
                search_results=search_results,
                answer_scope=answer_scope,
                has_images=bool(image_paths),
            )
            response = await self.openai_client.chat.completions.create(
                model=routed_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                stream=True,
                **self._provider_options(routed_model),
            )

            async def token_generator():
                # Stream từng delta ngay khi model sinh để người dùng thấy chữ
                # xuất hiện dần (first-paint sớm) thay vì chờ trọn câu.
                parts: list[str] = []
                emitted = ""
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        parts.append(content)
                        emitted += content
                        yield ("token", content)
                # Hậu xử lý cần toàn bộ câu (bỏ <think>, cắt marker, chặn lặp).
                # Chỉ phát 'replace' khi bản sạch khác bản đã stream — ca thường
                # (model ngoan) trùng nhau nên không có replace, stream thuần.
                cleaned = self._clean_model_answer("".join(parts))
                if not cleaned and self._should_retry_with_openai(routed_model):
                    logger.warning(
                        "Model %s streamed an empty answer; retrying with %s.",
                        routed_model,
                        OPENAI_FALLBACK_MODEL,
                    )
                    fallback_response = await self.openai_client.chat.completions.create(
                        model=OPENAI_FALLBACK_MODEL,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=max_tokens,
                        **self._provider_options(OPENAI_FALLBACK_MODEL),
                    )
                    fallback_answer = self._clean_model_answer(
                        fallback_response.choices[0].message.content or ""
                    )
                    if fallback_answer:
                        yield ("token", fallback_answer)
                    return
                if cleaned != emitted.strip():
                    yield ("replace", cleaned)

            return token_generator(), search_results, routed_model, answer_scope
        except Exception as e:
            logger.error(f"Error in streaming RAG generation: {e}")
            raise e

    async def _get_mes_lots(
        self,
        question: str,
        mode: str,
    ) -> list[MesLotError] | None:
        if mode != "mes" or not self._is_highest_lot_error_question(question):
            return None
        if self.mes_client is None:
            raise RuntimeError("MES integration chưa được cấu hình.")

        logger.info("Routing highest Lot error question to MES API.")
        return await self.mes_client.get_lots_with_highest_error()

    async def _get_mes_route(
        self,
        question: str,
        mode: str,
    ) -> tuple[
        str | None,
        list[MesLotError] | MesDatabaseResult | None,
    ]:
        if mode != "mes":
            return None, None

        highest_lot_question = self._is_highest_lot_error_question(question)
        time_related_question = self._is_time_related_mes_question(question)
        if highest_lot_question and (
            self._is_compound_mes_question(question) or time_related_question
        ):
            return None, None
        explicit_snapshot = bool(
            self.mes_database
            and self.mes_database.is_snapshot_question(question)
        )

        if highest_lot_question:
            snapshot_result = await self._query_mes_database(
                question,
                allow_highest_lot=True,
            )
            if snapshot_result is not None:
                logger.info("Routing highest Lot error question to local MES snapshot.")
                return "mes_database", snapshot_result

            api_error: Exception | None = None
            if self.mes_client is not None:
                try:
                    logger.info("Routing highest Lot error question to MES API.")
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

        snapshot_result = await self._query_mes_database(
            question,
            allow_highest_lot=explicit_snapshot,
        )
        if snapshot_result is not None:
            return "mes_database", snapshot_result
        return None, None

    async def _generate_mes_sql_answer(
        self,
        question: str,
        model: str,
        mode: str,
    ) -> tuple[str, str] | None:
        if self.mes_sql_agent is None or not self.mes_sql_agent.available:
            return None

        routed_model = self._resolve_mes_sql_model(model, mode=mode)
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
                    max_tokens=self._mes_sql_planner_max_tokens(),
                    **self._provider_options(routed_model),
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
                    max_tokens=self._mes_sql_answer_max_tokens(result),
                    **self._provider_options(routed_model),
                )
                candidate = answer_response.choices[0].message.content or ""
                candidate = self._normalize_mes_sql_answer(candidate)
                if self._mes_sql_answer_is_natural(
                    candidate
                ) and self._mes_sql_answer_matches_result(candidate, result):
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

    @staticmethod
    def _normalize_mes_sql_answer(answer: str) -> str:
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
    def _mes_sql_answer_matches_result(answer: str, result: MesSqlQueryResult) -> bool:
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
    def _mes_sql_answer_is_natural(answer: str) -> bool:
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

    @classmethod
    def _is_compound_mes_question(cls, question: str) -> bool:
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
    def _is_time_related_mes_question(question: str) -> bool:
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

    async def _query_mes_database(
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

    async def _generate_mes_database_answer(
        self,
        question: str,
        result: MesDatabaseResult,
        model: str,
        mode: str,
    ) -> tuple[str, str]:
        routed_model = self._resolve_model(model, mode=mode)
        try:
            response = await self.openai_client.chat.completions.create(
                model=routed_model,
                messages=self._mes_database_messages(question, result),
                temperature=0.1,
                max_tokens=self._mes_database_max_tokens(result),
                **self._provider_options(routed_model),
            )
            candidate = response.choices[0].message.content or ""
            answer = (
                candidate
                if self._mes_database_answer_has_required_terms(candidate, result)
                else result.fallback_answer
            )
        except Exception as exc:
            logger.warning("LLM MES snapshot answer generation failed: %s", exc)
            answer = result.fallback_answer
        return answer, routed_model

    @staticmethod
    def _mes_database_messages(
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
    def _mes_database_answer_has_required_terms(
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
    def _is_highest_lot_error_question(question: str) -> bool:
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

    @staticmethod
    def _mes_messages(
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
    def _format_mes_fallback(lots: list[MesLotError]) -> str:
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
    def _mes_answer_has_required_fields(
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

    def _retrieve(
        self,
        session_id: str,
        question: str,
        top_k: int | None = None,
    ) -> List[SearchResult]:
        """
        Tìm kiếm ngữ nghĩa từ Vector Store.
        """
        query_embedding = self.embedder.embed_query(question)
        results = self.vector_store.search(
            session_id=session_id,
            query_embedding=query_embedding,
            top_k=top_k or self.top_k,
            score_threshold=self.score_threshold
        )
        logger.info(f"Retrieved {len(results)} chunks for query: '{question[:50]}'")
        return results

    def _prepare_query_context(
        self,
        session_id: str,
        question: str,
        mode: str,
        research_topic: str | None = None,
        research_scope: str | None = None,
    ) -> Tuple[List[SearchResult], List[Path], str]:
        resolved_research_scope = research_scope or (
            "topic" if research_topic else "upload"
        )
        if mode == "research" and resolved_research_scope == "topic" and research_topic:
            return self._prepare_research_query_context(
                session_id=session_id,
                question=question,
                research_topic=research_topic,
            )
        if mode == "mkac":
            if self.mkac_vector_store is None:
                logger.warning("MKAC vector store is not configured.")
                return self._web_or_general(question)
            retrieval_question = self._mkac_retrieval_question(question)
            retrieval_threshold = self._mkac_retrieval_threshold(question)
            query_embedding = self.embedder.embed_query(retrieval_question)
            results = self.mkac_vector_store.search(
                session_id="mkac",
                query_embedding=query_embedding,
                top_k=self.top_k,
                score_threshold=retrieval_threshold,
            )
            results = [
                result
                for result in results
                if result.score >= retrieval_threshold
            ]
            results = self._filter_mkac_results_by_intent(question, results)
            results = self._filter_relative_results(results)
            logger.info(
                "Retrieved %s MKAC chunks for query '%s' with scores=%s",
                len(results),
                retrieval_question[:50],
                [round(result.score, 4) for result in results],
            )
            if not results:
                return self._web_or_general(question)
            images = (
                self._result_image_paths(results)
                if self._question_needs_vision(question)
                else []
            )
            return results, images, "mkac"

        results = self._retrieve(
            session_id,
            question,
            env_int("RESEARCH_TOP_K", 5, minimum=3, maximum=8),
        )
        images = self._session_image_paths(session_id)
        if self._question_needs_vision(question):
            images = list(
                dict.fromkeys([*images, *self._result_image_paths(results)])
            )[:2]
        return results, images, "research"

    def _prepare_research_query_context(
        self,
        *,
        session_id: str,
        question: str,
        research_topic: str,
    ) -> Tuple[List[SearchResult], List[Path], str]:
        """Retrieve từ kho DocJP theo nhóm chủ đề đã chọn.

        ``research_topic`` là topic id đã được validate ở tầng API. Session của
        request chỉ là phiên hội thoại; retrieval luôn dùng session cố định của
        collection DocJP, thu hẹp thêm bằng ``metadata.category`` trừ khi chọn
        ``all``.
        """
        from src.api.research_topics import research_topic_category

        store = self.docjp_vector_store
        if store is None:
            logger.warning(
                "DocJP vector store is not configured; research topic %r "
                "falls back to session documents.",
                research_topic,
            )
            results = self._retrieve(
                session_id,
                question,
                env_int("RESEARCH_TOP_K", 5, minimum=3, maximum=8),
            )
            return results, [], "research"

        category = research_topic_category(research_topic)
        metadata_filters = {"category": category} if category else None
        docjp_session_id = os.getenv("DOCJP_SESSION_ID", "docjp")
        query_embedding = self.embedder.embed_query(question)
        results = store.search(
            session_id=docjp_session_id,
            query_embedding=query_embedding,
            top_k=env_int("RESEARCH_TOP_K", 6, minimum=3, maximum=12),
            score_threshold=float(os.getenv("RESEARCH_SCORE_THRESHOLD", "0.35")),
            metadata_filters=metadata_filters,
        )
        logger.info(
            "Retrieved %s DocJP chunks for topic=%s query '%s' with scores=%s",
            len(results),
            research_topic,
            question[:50],
            [round(result.score, 4) for result in results],
        )
        return results, [], "research"

    def _web_or_general(
        self,
        question: str,
    ) -> Tuple[List[SearchResult], List[Path], str]:
        if self.web_searcher is not None:
            web_results = self.web_searcher.search(question)
            if web_results:
                return web_results, [], "web"
        return [], [], "general"

    @staticmethod
    def _mkac_retrieval_question(question: str) -> str:
        """Keep company identity terms only when the question is about identity."""
        normalized = question.lower()
        if RAGPipeline._is_employee_statistics_question(question):
            return (
                f"{question}\n"
                "Thống kê nhân sự MKAC, danh sách khám sức khỏe 2026, "
                "số nhân sự có mã ID, số phòng ban, mỗi phòng ban có bao nhiêu người, "
                "thông tin lãnh đạo, giám đốc, phó tổng giám đốc, tổng giám đốc."
            )
        identity_keywords = {
            "viết tắt",
            "tên công ty",
            "tên pháp lý",
            "tên doanh nghiệp",
            "mã số doanh nghiệp",
            "mã số thuế",
            "enterprise id",
            "legal name",
            "abbreviation",
        }
        if any(keyword in normalized for keyword in identity_keywords):
            return question

        cleaned = re.sub(
            r"\b(công ty cổ phần meiko automation|meiko automation joint stock company|mkac)\b",
            " ",
            question,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
        return cleaned or question

    def _mkac_retrieval_threshold(self, question: str) -> float:
        if self._is_employee_statistics_question(question):
            return min(self.mkac_score_threshold, 0.34)
        if self._is_company_profile_question(question):
            return 0.42
        return self.mkac_score_threshold

    @classmethod
    def _filter_mkac_results_by_intent(
        cls,
        question: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        if cls._is_employee_statistics_question(question):
            employee_categories = {"employee_statistics", "employee_directory"}
            employee_results = [
                result
                for result in results
                if (result.chunk.metadata or {}).get("category") in employee_categories
            ]
            return employee_results or results
        if not cls._is_company_profile_question(question):
            return results
        legal_categories = {"corporate_identity", "investment_registration"}
        legal_results = [
            result
            for result in results
            if (result.chunk.metadata or {}).get("category") in legal_categories
        ]
        return legal_results or results

    @staticmethod
    def _is_company_profile_question(question: str) -> bool:
        normalized = question.lower()
        keywords = {
            "lĩnh vực hoạt động",
            "ngành nghề",
            "hồ sơ đăng ký",
            "đăng ký đầu tư",
            "dự án đầu tư",
            "business activities",
            "business lines",
        }
        return any(keyword in normalized for keyword in keywords)

    @staticmethod
    def _is_employee_statistics_question(question: str) -> bool:
        normalized = question.lower()
        keywords = {
            "nhân sự",
            "bao nhiêu nhân viên",
            "số nhân viên",
            "danh sách nhân viên",
            "bao nhiêu người",
            "số người",
            "phòng ban",
            "bộ phận",
            "mỗi phòng",
            "mỗi phòng ban",
            "trưởng phòng",
            "phó phòng",
            "giám đốc",
            "tổng giám đốc",
            "phó tổng giám đốc",
            "mã nhân viên",
            "employee count",
            "employee list",
            "department",
            "director",
        }
        return any(keyword in normalized for keyword in keywords)

    @staticmethod
    def _is_current_user_question(question: str) -> bool:
        normalized = question.lower()
        personal_markers = {
            "tôi",
            "mình",
            "của tôi",
            "của mình",
            "tên tôi",
            "tôi tên",
            "tôi là ai",
            "mã nhân viên của tôi",
            "bộ phận của tôi",
            "phòng ban của tôi",
            "tôi làm bộ phận",
            "tôi thuộc bộ phận",
            "tôi làm phòng",
            "công ty của tôi",
            "công ty của mình",
            "my name",
            "my company",
            "my department",
        }
        if not any(marker in normalized for marker in personal_markers):
            return False
        topics = {
            "tên",
            "họ tên",
            "bộ phận",
            "phòng ban",
            "chức danh",
            "vị trí",
            "trưởng phòng",
            "phó phòng",
            "bao nhiêu người",
            "số người",
            "name",
            "department",
            "position",
            "manager",
        }
        return any(topic in normalized for topic in topics)

    @staticmethod
    def _is_current_company_question(question: str) -> bool:
        normalized = question.lower()
        personal_company_markers = {
            "công ty của tôi",
            "công ty của mình",
            "tên công ty tôi",
            "tên công ty của tôi",
            "tôi làm công ty nào",
            "tôi thuộc công ty nào",
            "my company",
        }
        name_markers = {
            "tên gì",
            "tên là gì",
            "tên công ty",
            "công ty nào",
            "company name",
            "what company",
        }
        return any(marker in normalized for marker in personal_company_markers) and any(
            marker in normalized for marker in name_markers
        )

    @staticmethod
    def _has_employee_directory_context(current_user: Dict[str, Any]) -> bool:
        return bool(
            current_user.get("queried_departments")
            or current_user.get("queried_people")
        )

    @staticmethod
    def _general_messages(question: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    @staticmethod
    def _question_needs_vision(question: str) -> bool:
        normalized = question.lower()
        keywords = {
            "ảnh",
            "hình",
            "sơ đồ",
            "biểu đồ",
            "bảng",
            "chart",
            "image",
            "diagram",
            "table",
        }
        return any(keyword in normalized for keyword in keywords)

    @staticmethod
    def _filter_relative_results(
        results: List[SearchResult],
        relative_floor: float = 0.85,
    ) -> List[SearchResult]:
        """Drop weak tail matches that are far below the best MKAC result."""
        if not results:
            return []
        minimum = results[0].score * relative_floor
        return [result for result in results if result.score >= minimum]

    @staticmethod
    def _result_image_paths(results: List[SearchResult]) -> List[Path]:
        paths: List[Path] = []
        seen = set()
        for result in results:
            image_path = (result.chunk.metadata or {}).get("image_path")
            if not image_path or image_path in seen:
                continue
            path = Path(image_path)
            if path.exists():
                paths.append(path)
                seen.add(image_path)
            if len(paths) >= 2:
                break
        return paths

    def _resolve_model(
        self,
        model: str,
        has_images: bool = False,
        mode: str = "mkac",
    ) -> str:
        """Ánh xạ lựa chọn từ UI sang model logic của LiteLLM."""
        if has_images and model in {"auto", "grok"}:
            return "grok-model"
        if has_images:
            return "openai-model"
        try:
            return MODEL_ROUTES[model]
        except KeyError as exc:
            raise ValueError(f"Unsupported model option: {model}") from exc

    def _resolve_mes_sql_model(self, model: str, mode: str = "mkac") -> str:
        """Resolve the dedicated SQL planner model before falling back to UI choice."""
        forced_model = os.getenv("MES_SQL_AGENT_MODEL", "local-qwen-coder").strip()
        if forced_model:
            return MODEL_ROUTES.get(forced_model, forced_model)
        return self._resolve_model(model, mode=mode)

    def _rag_answer_max_tokens(
        self,
        *,
        question: str,
        mode: str,
        search_results: List[SearchResult],
        answer_scope: str,
        has_images: bool,
    ) -> int:
        if mode == "research":
            if self._needs_extended_research_answer(question, search_results):
                return env_int("RESEARCH_MAX_TOKENS", 1800, minimum=384, maximum=2048)
            return env_int(
                "RESEARCH_SIMPLE_MAX_TOKENS", 640, minimum=256, maximum=1024
            )
        if mode != "mkac":
            return self.max_tokens
        if answer_scope == "general":
            return env_int("MKAC_GENERAL_MAX_TOKENS", 256, minimum=128, maximum=512)
        if has_images or self._needs_extended_mkac_answer(question, search_results):
            return env_int("MKAC_EXTENDED_MAX_TOKENS", 768, minimum=384, maximum=1024)
        return env_int("MKAC_SIMPLE_MAX_TOKENS", 512, minimum=256, maximum=768)

    @staticmethod
    def _needs_extended_mkac_answer(
        question: str,
        search_results: List[SearchResult],
    ) -> bool:
        normalized = RAGPipeline._normalize_question_text(question)
        if len(question or "") >= 140 or len(search_results) >= 4:
            return True
        extended_markers = (
            "quy trinh",
            "quy dinh",
            "noi quy",
            "chinh sach",
            "phuc loi",
            "che do",
            "huong dan",
            "cac buoc",
            "danh sach",
            "liet ke",
            "so sanh",
            "phan tich",
            "chi tiet",
            "bao gom",
            "nhung gi",
        )
        return any(marker in normalized for marker in extended_markers)

    @staticmethod
    def _needs_extended_research_answer(
        question: str,
        search_results: List[SearchResult],
    ) -> bool:
        """Câu hỏi Research dài/liệt kê/nhiều bước mới cần budget token cao.

        Khác MKAC, Research nhận nhiều câu hỏi tiếng Nhật nên không thể chuẩn
        hóa dấu tiếng Việt (``_normalize_question_text`` bỏ dấu qua NFD chỉ hợp
        với chữ Latin) — so khớp trực tiếp trên bản gốc (lowercase cho phần
        Latin) và không phân biệt hoa/thường với ký tự Latin.
        """
        text = (question or "").strip()
        if len(text) >= 60 or len(search_results) >= 5:
            return True
        normalized = text.lower()
        extended_markers_vi = (
            "quy trinh",
            "quy trình",
            "cac buoc",
            "các bước",
            "danh sach",
            "danh sách",
            "liet ke",
            "liệt kê",
            "so sanh",
            "so sánh",
            "phan tich",
            "phân tích",
            "tat ca",
            "tất cả",
            "bao gom",
            "bao gồm",
            "huong dan",
            "hướng dẫn chi tiết",
        )
        # Tiếng Nhật không có khái niệm "dấu" nên so khớp trực tiếp nguyên văn.
        extended_markers_ja = (
            "手順",  # quy trình/các bước
            "一覧",  # danh sách
            "すべて",  # tất cả
            "比較",  # so sánh
            "詳しく",  # chi tiết hơn
            "違い",  # sự khác biệt
        )
        if any(marker in normalized for marker in extended_markers_vi):
            return True
        return any(marker in text for marker in extended_markers_ja)

    @staticmethod
    def _normalize_question_text(question: str) -> str:
        normalized = unicodedata.normalize(
            "NFD",
            (question or "").lower().replace("đ", "d"),
        )
        normalized = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        return re.sub(r"[^a-z0-9_-]+", " ", normalized).strip()

    @staticmethod
    def _mes_live_api_max_tokens() -> int:
        return env_int("MES_LIVE_API_MAX_TOKENS", 192, minimum=96, maximum=384)

    @staticmethod
    def _mes_database_max_tokens(result: MesDatabaseResult) -> int:
        default = 384 if len(result.rows) > 3 else 256
        return env_int("MES_DATABASE_MAX_TOKENS", default, minimum=128, maximum=768)

    @staticmethod
    def _mes_sql_planner_max_tokens() -> int:
        return env_int("MES_SQL_PLANNER_MAX_TOKENS", 1200, minimum=512, maximum=1600)

    @staticmethod
    def _mes_sql_answer_max_tokens(result: MesSqlQueryResult) -> int:
        default = 512 if result.truncated or len(result.rows) > 10 else 384
        return env_int("MES_SQL_ANSWER_MAX_TOKENS", default, minimum=192, maximum=800)

    def _session_image_paths(self, session_id: str) -> List[Path]:
        """Return uploaded image paths for a session, if any."""
        info = self.vector_store.get_session_info(session_id)
        if not info:
            return []

        upload_dir = Path(os.getenv("UPLOAD_DIR", "./uploads"))
        session_dir = upload_dir / session_id
        paths = []
        for filename in info.get("files", []):
            path = session_dir / Path(filename).name
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.exists():
                paths.append(path.resolve())
        return paths

    @staticmethod
    def _provider_options(routed_model: str) -> Dict[str, Any]:
        """Provider-specific safeguards for LiteLLM upstream models."""
        if routed_model in LOCAL_CHAT_MODEL_ALIASES:
            # Gemma4 on Ollama may spend the whole generation budget in
            # message.thinking, leaving message.content empty or truncated.
            num_ctx = int(os.getenv("LOCAL_CHAT_NUM_CTX", "16384"))
            return {"extra_body": {"think": False, "num_ctx": num_ctx}}
        return {}

    @staticmethod
    def _should_retry_with_openai(routed_model: str) -> bool:
        return routed_model in PRIMARY_CHAT_MODELS and routed_model != OPENAI_FALLBACK_MODEL

    @staticmethod
    def _clean_model_answer(answer: str) -> str:
        """Remove local reasoning traces that some Qwen/Ollama responses leak."""
        text = (answer or "").strip()
        if not text:
            return ""

        text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        text = re.sub(r"\[img-\d+\]", "\n\n", text).strip()

        repeated_phrases = (
            "không dùng các từ ngữ không cần thiết",
            "không dùng markdown",
            "trả lời bằng tiếng việt",
        )
        lowered_text = text.lower()
        if any(lowered_text.count(phrase) >= 4 for phrase in repeated_phrases):
            return (
                "Chưa tạo được câu trả lời ổn định từ model local. "
                "Vui lòng thử lại với câu hỏi ngắn gọn hơn."
            )
        if RAGPipeline._has_repetitive_loop(text):
            return (
                "Chưa tạo được câu trả lời ổn định từ model local. "
                "Vui lòng thử lại với câu hỏi ngắn gọn hơn."
            )

        direct_patterns = (
            r"MKAC\s+có\s+[^.\n。]*\b16\b[^.\n。]*(?:phòng\s+ban|phong\s+ban|bộ\s+phận|bo\s+phan|nhóm|nhom)[^.\n。]*[.\n。]?",
            r"MKACには[^.\n。]*\b16\b[^.\n。]*(?:部門|部署|グループ)[^.\n。]*[.\n。]?",
        )
        for pattern in direct_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0).strip().rstrip()

        final_markers = [
            "Câu trả lời:",
            "Trả lời ngắn gọn:",
            "Kết luận:",
            "Tóm lại:",
            "Đáp án:",
            "簡潔な回答:",
            "回答:",
            "結論:",
        ]
        lowered = text.lower()
        marker_positions = [
            (lowered.rfind(marker.lower()), marker)
            for marker in final_markers
            if lowered.rfind(marker.lower()) != -1
        ]
        if marker_positions:
            pos, marker = max(marker_positions, key=lambda item: item[0])
            text = text[pos + len(marker) :].strip()

        reasoning_markers = [
            "trả lời bằng",
            "đảm bảo câu trả lời",
            "hãy trả lời",
            "Okay, let's see.",
            "Okay, let me",
            "Let me think",
            "Looking at",
            "In Đoạn",
            "In Doan",
            "I should count",
            "Let's list",
            "I need to",
            "The user is asking",
            "The answer should",
            "Let me count",
            "The first excerpt",
            "Đoạn 1",
            "Đoạn 2",
            "以下のように",
            "見てみます",
            "数えてみます",
        ]
        if any(marker.lower() in text[:500].lower() for marker in reasoning_markers):
            paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
            useful = [
                part
                for part in paragraphs
                if not any(marker.lower() in part[:300].lower() for marker in reasoning_markers)
            ]
            if useful:
                text = "\n\n".join(useful).strip()

        return text

    @staticmethod
    def _has_repetitive_loop(text: str) -> bool:
        words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(words) < 80:
            return False
        for size, threshold in ((4, 10), (6, 7), (8, 5)):
            counts: dict[tuple[str, ...], int] = {}
            for index in range(0, len(words) - size + 1):
                key = tuple(words[index : index + size])
                counts[key] = counts.get(key, 0) + 1
                if counts[key] >= threshold:
                    return True
        return False

    def format_sources(
        self,
        results: List[SearchResult],
        *,
        research_scope: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Định dạng nguồn trích dẫn trả về API.
        """
        from src.rag.media_paths import resolve_processed_image_path

        sources = []
        for r in results:
            stored_image_path = (r.chunk.metadata or {}).get("image_path")
            image_path = resolve_processed_image_path(stored_image_path)
            sources.append(
                {
                    "file": r.chunk.source_file,
                    "page": r.chunk.page_number,
                    "score": round(r.score, 4),
                    "type": r.chunk.content_type,
                    "preview": (
                        r.chunk.text[:200] + "..."
                        if len(r.chunk.text) > 200
                        else r.chunk.text
                    ),
                    "title": r.chunk.metadata.get("title"),
                    "category": r.chunk.metadata.get("category"),
                    "effective_date": r.chunk.metadata.get("effective_date"),
                    "url": r.chunk.metadata.get("url"),
                    "has_page_preview": bool(
                        image_path and Path(image_path).is_file()
                    ),
                    "source_scope": research_scope,
                }
            )
        return sources
