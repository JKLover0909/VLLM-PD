"""Conservative deterministic scope checks shared by all query modes."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.actions.report_intent import matched_domain_contexts
from src.integrations.mes_database import MesDatabase
from src.integrations.mes_wms_database import MesWmsDatabase
from src.integrations.mes_intent import normalized_text


@dataclass(frozen=True)
class ScopeDecision:
    kind: str
    target_mode: str = ""


_DENY_MARKERS = (
    "bitcoin", "crypto price", "stock price", "share price", "weather",
    "forecast", "cook", "recipe", "poem", "write a poem", "football",
    "basketball", "anime", "celebrity", "giá bitcoin", "tiền ảo",
    "chứng khoán", "thời tiết", "dự báo thời tiết", "nấu phở", "công thức nấu",
    "làm thơ", "bài thơ", "bóng đá", "bóng rổ", "người nổi tiếng",
    "ビットコイン", "仮想通貨", "株価", "天気", "料理", "レシピ",
    "詩を書いて", "サッカー", "野球", "有名人",
)
_QA_MARKERS = (
    "qa", "quality", "5m1e", "checksheet", "check sheet", "inspection",
    "machine", "equipment", "máy", "thiết bị", "phiếu kiểm tra",
    "chất lượng", "品質", "設備", "検査", "チェックシート",
)
_QA_CODE = re.compile(
    r"(?<![A-Z0-9])(?:[A-Z]{2,6}\s*)?[A-Z]\d{2,4}-\d{2,5}(?![A-Z0-9])",
    re.IGNORECASE,
)
_PRODUCT_CODE = re.compile(r"(?<![A-Z0-9])\d{4}-\d{4}(?![A-Z0-9])", re.I)
_FOLLOW_UP_MARKERS = (
    "còn", "vậy", "cái đó", "nó", "lot đó", "tháng trước", "người đó",
    "that one", "what about", "and then", "それ", "前の", "先月", "そのロット",
)


def _matches_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _mode_contexts(question: str) -> set[str]:
    contexts = matched_domain_contexts(question)
    original = question or ""
    normalized = normalized_text(original)
    if MesDatabase.LOT_PATTERN.search(original) or _PRODUCT_CODE.search(original):
        contexts.add("mes")
    if MesWmsDatabase.is_wms_question(original):
        contexts.add("wms")
    if _QA_CODE.search(original) or _matches_any(normalized, _QA_MARKERS):
        contexts.add("qa")
    return contexts


def classify_scope(
    question: str,
    mode: str,
    history: list[dict] | None = None,
) -> ScopeDecision | None:
    """Return a block decision only for clear off-topic or wrong-mode cases."""
    original = question or ""
    normalized = normalized_text(original)
    if not normalized and not original.strip():
        return None

    contexts = _mode_contexts(original)
    current_context = mode in contexts
    if history and _matches_any(normalized, tuple(
        normalized_text(marker) for marker in _FOLLOW_UP_MARKERS
    )):
        return None

    if mode == "qa":
        if not contexts.intersection({"qa", "mes", "wms", "mkac", "cctvai"}) and _matches_any(
            normalized,
            tuple(normalized_text(marker) for marker in _DENY_MARKERS),
        ):
            return ScopeDecision("off_topic")
        return None

    wrong_targets = contexts - {mode}
    if wrong_targets and not current_context:
        priority = ("mes", "wms", "cctvai", "mkac", "qa")
        target = next((item for item in priority if item in wrong_targets), "")
        if target:
            return ScopeDecision("wrong_mode", target)

    if (
        not contexts
        and _matches_any(
            normalized,
            tuple(normalized_text(marker) for marker in _DENY_MARKERS),
        )
    ):
        return ScopeDecision("off_topic")
    return None


_MODE_LABELS = {
    "vi": {
        "mkac": "MKAC/nhân sự",
        "mes": "MES",
        "wms": "WMS",
        "cctvai": "CCTVAI",
        "qa": "QA",
        "research": "Nghiên cứu tài liệu",
    },
    "ja": {
        "mkac": "MKAC・人事",
        "mes": "MES",
        "wms": "WMS",
        "cctvai": "CCTVAI",
        "qa": "QA",
        "research": "資料調査",
    },
}


def scope_guard_response(req, decision: ScopeDecision):
    """Build a non-security guard response in the requested UI language."""
    language = "ja" if req.ui_language == "ja" else "vi"
    labels = _MODE_LABELS[language]
    if decision.kind == "wrong_mode":
        answer = (
            f"この質問は{labels.get(decision.target_mode, '適切なモード')}でお尋ねください。"
            if language == "ja"
            else (
                "Câu hỏi này có vẻ thuộc chế độ "
                f"{labels.get(decision.target_mode, 'phù hợp')}. "
                "Vui lòng chuyển sang chế độ đó để được hỗ trợ."
            )
        )
    else:
        answer = (
            "この質問は選択中の業務モードの対象外です。"
            if language == "ja"
            else "Câu hỏi này nằm ngoài phạm vi của chế độ đang chọn."
        )

    from src.api.schemas import QueryResponse

    return QueryResponse(
        answer=answer,
        sources=[],
        session_id=req.session_id,
        model="auto-model",
        mode=req.mode,
        answer_scope=decision.kind,
    )
