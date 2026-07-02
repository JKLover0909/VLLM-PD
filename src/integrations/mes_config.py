"""Cấu hình định tuyến model và ngân sách token cho phân hệ MES.

Tách riêng khỏi ``mes_query_service`` để phần định tuyến/định lượng token nằm gọn
một chỗ, dễ chỉnh khi thêm model mới hoặc thay đổi giới hạn. Giữ nguyên hành vi
so với bản gốc.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - chỉ dùng cho type hint
    from src.integrations.mes_database import MesDatabaseResult
    from src.integrations.mes_sql_agent import MesSqlQueryResult

MODEL_ROUTES = {
    "auto": "auto-model",
    "local": "local-qwen-chat",
    "openai": "openai-model",
    "grok": "grok-model",
}

LOCAL_CHAT_MODEL_ALIASES = {"auto-model", "local-gemma", "local-qwen-chat"}
LOCAL_MODEL_ALIASES = LOCAL_CHAT_MODEL_ALIASES | {"local-qwen-coder", "coding-model"}


def env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 4096) -> int:
    """Read a bounded integer environment setting."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def resolve_model(model: str) -> str:
    if model == "grok":
        return "openai-model"
    try:
        return MODEL_ROUTES[model]
    except KeyError as exc:
        raise ValueError(f"Unsupported model option: {model}") from exc


def resolve_sql_agent_model(model: str) -> str:
    forced_model = os.getenv("MES_SQL_AGENT_MODEL", "local-qwen-coder").strip()
    if forced_model:
        return MODEL_ROUTES.get(forced_model, forced_model)
    return resolve_model(model)


def provider_options(routed_model: str) -> dict[str, Any]:
    if routed_model in LOCAL_CHAT_MODEL_ALIASES:
        num_ctx = int(os.getenv("LOCAL_CHAT_NUM_CTX", "16384"))
        return {"extra_body": {"think": False, "num_ctx": num_ctx}}
    return {}


def general_answer_max_tokens() -> int:
    return env_int("MES_GENERAL_MAX_TOKENS", 256, minimum=96, maximum=512)


def live_api_answer_max_tokens() -> int:
    return env_int("MES_LIVE_API_MAX_TOKENS", 192, minimum=96, maximum=384)


def database_answer_max_tokens(result: "MesDatabaseResult") -> int:
    default = 384 if len(result.rows) > 3 else 256
    return env_int("MES_DATABASE_MAX_TOKENS", default, minimum=128, maximum=768)


def sql_planner_max_tokens() -> int:
    # Planner cần đủ không gian để trả JSON/SQL hợp lệ; không giảm quá thấp.
    return env_int("MES_SQL_PLANNER_MAX_TOKENS", 1200, minimum=512, maximum=1600)


def sql_answer_max_tokens(result: "MesSqlQueryResult") -> int:
    default = 512 if result.truncated or len(result.rows) > 10 else 384
    return env_int("MES_SQL_ANSWER_MAX_TOKENS", default, minimum=192, maximum=800)


def prefer_template_answers(routed_model: str) -> bool:
    """Prefer deterministic MES wording for local models to reduce latency."""
    return (
        os.getenv("MES_TEMPLATE_ANSWERS_FOR_LOCAL", "true").lower()
        in {"1", "true", "yes", "on"}
        and routed_model in LOCAL_MODEL_ALIASES
    )
