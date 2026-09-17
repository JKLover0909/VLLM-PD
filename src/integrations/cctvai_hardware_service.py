"""Service layer for CCTVAI host hardware-metrics questions.

Talks to the independent ``tools/host_metrics`` HTTPS service on the CCTVAI
Ubuntu server via ``CctvaiHardwareClient``. Deliberately independent of the
PostgreSQL-backed replica (``cctvai_query_service.py`` / ``cctvai_database.py``):
a different transport, a different availability model, and it must keep
answering hardware questions even when RAG/MES/the replica are all down (see
AGENTS.md section 4.2b and the approved plan).

Design constraints carried over from the plan:
    * The answer is built from a deterministic, bilingual rendering of the
      validated snapshot first. A local-only LLM (``CCTVAI_HARDWARE_MODEL``,
      never in ``litellm_config.yaml``'s ``fallbacks:``) may only rephrase
      those same facts — its output is rejected if it introduces a number
      that was not already in the deterministic text, and any LLM failure
      falls back to the deterministic answer.
    * If the snapshot fetch itself fails, the LLM is never called — the
      service must not let a model guess at hardware state it cannot see.
    * Nothing here performs remote shell, restart, or any other control
      action; this module can only read ``GET /metrics``.
"""

from __future__ import annotations

import logging
import os
import re
import time

from openai import AsyncOpenAI

from tools.host_metrics.schemas import ComponentState, HostMetricsSnapshot

from .cctvai_hardware_client import (
    CctvaiHardwareClient,
    CctvaiHardwareClientError,
    HardwareSnapshotResult,
)
from .mes_query_service import MesQueryOutcome, MesQueryStreamOutcome

log = logging.getLogger(__name__)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


# ---------------------------------------------------------------------------
# classify_hardware_question — pure, bilingual, deliberately narrow so the
# existing CCTVAI-replica intents (events/cameras/violations) are never
# shadowed by a stray keyword.
# ---------------------------------------------------------------------------

_HARDWARE_KEYWORDS = (
    # English
    "cpu", "ram", "memory", "gpu", "vram", "disk", "storage", "network",
    "bandwidth", "uptime", "docker", "container", "hardware", "processor",
    "filesystem",
    # Vietnamese
    "bộ nhớ", "ổ đĩa", "ổ cứng", "dung lượng", "phần cứng", "băng thông",
    "thời gian hoạt động", "tiến trình", "vi xử lý", "card đồ họa",
    "nhiệt độ",
    # Japanese
    "メモリ", "ディスク", "ハードウェア", "ネットワーク", "稼働時間",
    "コンテナ", "プロセス", "温度",
)


def classify_hardware_question(question: str) -> bool:
    """True when ``question`` plainly asks about the CCTVAI host's own
    resource usage (CPU/RAM/GPU/disk/network/uptime/Docker/processes).

    Pure keyword match, no I/O. Only meaningful inside ``mode="cctvai"`` —
    the caller in ``main.py`` gates on the mode separately.
    """
    if not question or not question.strip():
        return False
    normalized = question.lower()
    return any(keyword in normalized for keyword in _HARDWARE_KEYWORDS)


# ---------------------------------------------------------------------------
# Deterministic bilingual rendering — the single source of truth for the
# numbers in the answer. The LLM pass (if used) may only rephrase this text.
# ---------------------------------------------------------------------------

def _gb(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value / (1024 ** 3):.1f}GB"


def _rate(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f}MB/s"
    if value >= 1024:
        return f"{value / 1024:.1f}KB/s"
    return f"{value:.0f}B/s"


def _unmeasured(is_ja: bool) -> str:
    return "情報なし。" if is_ja else "không lấy được thông tin."


def _warming_up(is_ja: bool) -> str:
    return "データ収集中。" if is_ja else "đang thu thập dữ liệu."


def _render_cpu(snapshot: HostMetricsSnapshot, *, is_ja: bool) -> str | None:
    status = snapshot.components.cpu
    cpu = snapshot.cpu
    if status.status in (ComponentState.UNAVAILABLE, ComponentState.ERROR):
        return "- **CPU:** " + _unmeasured(is_ja)
    if status.status == ComponentState.WARMING_UP or cpu is None or cpu.utilization_percent is None:
        return "- **CPU:** " + _warming_up(is_ja)
    cores = cpu.core_count_logical
    if is_ja:
        core_txt = f"（論理{cores}コア）" if cores else ""
        return f"- **CPU:** 使用率{cpu.utilization_percent:.0f}%{core_txt}"
    core_txt = f" ({cores} lõi logic)" if cores else ""
    return f"- **CPU:** {cpu.utilization_percent:.0f}%{core_txt}"


def _render_memory(snapshot: HostMetricsSnapshot, *, is_ja: bool) -> str | None:
    status = snapshot.components.memory
    mem = snapshot.memory
    if status.status in (ComponentState.UNAVAILABLE, ComponentState.ERROR):
        return "- **RAM:** " + _unmeasured(is_ja)
    if status.status == ComponentState.WARMING_UP or mem is None or mem.percent is None:
        return "- **RAM:** " + _warming_up(is_ja)
    used, total = _gb(mem.used_bytes), _gb(mem.total_bytes)
    if is_ja:
        return f"- **RAM (メモリ):** 使用率{mem.percent:.0f}% ({used}/{total})"
    return f"- **RAM:** {mem.percent:.0f}% ({used}/{total})"


def _render_gpus(snapshot: HostMetricsSnapshot, *, is_ja: bool) -> str | None:
    status = snapshot.components.gpu
    if status.status in (ComponentState.UNAVAILABLE, ComponentState.ERROR):
        return "- **GPU:** " + _unmeasured(is_ja)
    if status.status == ComponentState.WARMING_UP or not snapshot.gpus:
        return "- **GPU:** " + _warming_up(is_ja)
    parts = []
    for gpu in snapshot.gpus[:4]:
        name = gpu.name or f"GPU{gpu.index}"
        util = f"{gpu.utilization_percent:.0f}%" if gpu.utilization_percent is not None else "N/A"
        vram = (
            f"{_gb(gpu.memory_used_bytes)}/{_gb(gpu.memory_total_bytes)}"
            if gpu.memory_total_bytes is not None
            else "N/A"
        )
        temp = f"{gpu.temperature_celsius:.0f}°C" if gpu.temperature_celsius is not None else "N/A"
        parts.append(f"{name} {util}, VRAM {vram}, {temp}")
    joined = "; ".join(parts)
    return f"- **GPU:** {joined}"


def _render_disks(snapshot: HostMetricsSnapshot, *, is_ja: bool) -> str | None:
    status = snapshot.components.disk
    prefix = "- **ディスク (Ổ đĩa):** " if is_ja else "- **Ổ đĩa:** "
    if status.status in (ComponentState.UNAVAILABLE, ComponentState.ERROR):
        return prefix + _unmeasured(is_ja)
    if status.status == ComponentState.WARMING_UP or not snapshot.disks:
        return prefix + _warming_up(is_ja)
    disks_sorted = sorted(
        snapshot.disks,
        key=lambda d: d.percent if d.percent is not None else -1.0,
        reverse=True,
    )
    # Lọc các mountpoint trùng dung lượng với root / (như /tmp, /var/tmp, /var/lib/...) để tránh lặp dư thừa
    seen_sizes = set()
    unique_disks = []
    for disk in disks_sorted:
        size_key = (disk.total_bytes, disk.used_bytes)
        if size_key in seen_sizes and disk.mountpoint != "/":
            continue
        seen_sizes.add(size_key)
        unique_disks.append(disk)
        if len(unique_disks) >= 5:
            break
    parts = [
        f"`{disk.mountpoint}` "
        f"{(f'{disk.percent:.0f}%' if disk.percent is not None else 'N/A')} "
        f"({_gb(disk.used_bytes)}/{_gb(disk.total_bytes)})"
        for disk in unique_disks
    ]
    joined = "; ".join(parts)
    return f"{prefix}{joined}"


def _render_network(snapshot: HostMetricsSnapshot, *, is_ja: bool) -> str | None:
    status = snapshot.components.network
    if status.status in (ComponentState.UNAVAILABLE, ComponentState.ERROR):
        return None
    if status.status == ComponentState.WARMING_UP or not snapshot.network:
        return None
    parts = [
        f"`{iface.interface}` ↓{_rate(iface.rx_bytes_per_sec)} ↑{_rate(iface.tx_bytes_per_sec)}"
        for iface in snapshot.network[:4]
    ]
    joined = "; ".join(parts)
    prefix = "- **ネットワーク:** " if is_ja else "- **Mạng:** "
    return f"{prefix}{joined}"


def _render_uptime(snapshot: HostMetricsSnapshot, *, is_ja: bool) -> str | None:
    status = snapshot.components.uptime
    if status.status in (ComponentState.UNAVAILABLE, ComponentState.ERROR):
        return None
    if snapshot.uptime_seconds is None:
        return None
    days, rem = divmod(int(snapshot.uptime_seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if is_ja:
        return f"- **稼働時間 (Uptime):** {days}日 {hours}時間 {minutes}分"
    return f"- **Uptime:** {days} ngày {hours} giờ {minutes} phút"


def _render_docker(snapshot: HostMetricsSnapshot, *, is_ja: bool) -> str | None:
    status = snapshot.components.docker
    if status.status in (ComponentState.UNAVAILABLE, ComponentState.ERROR):
        return None
    if status.status == ComponentState.WARMING_UP or not snapshot.containers:
        return None
    total = len(snapshot.containers)
    running = sum(1 for c in snapshot.containers if c.state.lower() == "running")
    unhealthy = [
        c.name
        for c in snapshot.containers
        if c.health and c.health.lower() not in ("healthy", "none", "")
    ]
    if is_ja:
        text = f"- **Docker:** {running}/{total} container 実行中"
        if unhealthy:
            text += f" (⚠️ 異常あり: {', '.join(unhealthy[:5])})"
        return text
    text = f"- **Docker:** {running}/{total} container đang chạy"
    if unhealthy:
        text += f" (⚠️ Cảnh báo sức khỏe: {', '.join(unhealthy[:5])})"
    return text


def render_deterministic_answer(result: HardwareSnapshotResult, *, language: str) -> str:
    """Build the tất định (fully-computed) bilingual answer from a validated
    snapshot. Trình bày dưới dạng danh sách gạch đầu dòng Markdown rõ ràng,
    xuống dòng cách đoạn ngăn nắp."""
    is_ja = language == "ja"
    snapshot = result.snapshot
    header = (
        "📊 **CCTVAI サーバーの稼働状態:**"
        if is_ja
        else "📊 **Tình trạng phần cứng máy chủ CCTV AI:**"
    )
    segments = [
        _render_cpu(snapshot, is_ja=is_ja),
        _render_memory(snapshot, is_ja=is_ja),
        _render_gpus(snapshot, is_ja=is_ja),
        _render_disks(snapshot, is_ja=is_ja),
        _render_network(snapshot, is_ja=is_ja),
        _render_docker(snapshot, is_ja=is_ja),
        _render_uptime(snapshot, is_ja=is_ja),
    ]
    body_lines = [segment for segment in segments if segment]
    body = "\n".join(body_lines)

    notes = []
    if result.is_truncated:
        section_label = ", ".join(result.truncated_sections)
        notes.append(
            f"*(一部のみ表示: {section_label})*" if is_ja
            else f"*(chỉ hiển thị một phần: {section_label})*"
        )
    age = result.current_age_seconds()
    if age > 15:
        notes.append(
            f"*(データ取得から約{age:.0f}秒経過)*" if is_ja
            else f"*(dữ liệu cách đây khoảng {age:.0f}s)*"
        )
    parts = [header, body]
    if notes:
        parts.append(" ".join(notes))
    return "\n\n".join(parts).strip()


def _unavailable_answer(language: str, exc: CctvaiHardwareClientError) -> str:
    """Bilingual refusal when the snapshot itself could not be fetched.

    Only the exception *class name* is exposed — never the message, which
    may embed a hostname, path, or other internal detail.
    """
    reason = exc.__class__.__name__
    if language == "ja":
        return (
            "現在CCTVAIサーバーのハードウェア状態を取得できません"
            f"（{reason}）。しばらくしてからもう一度お試しください。"
        )
    return (
        "Hiện chưa lấy được dữ liệu phần cứng của server CCTV AI "
        f"({reason}). Vui lòng thử lại sau."
    )


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _answer_numbers_are_grounded(candidate: str, deterministic: str) -> bool:
    """Reject an LLM paraphrase that invents a number not already present
    in the deterministic text it was asked to rephrase."""
    candidate_numbers = _NUMBER_RE.findall(candidate)
    if not candidate_numbers:
        return True
    deterministic_numbers = set(_NUMBER_RE.findall(deterministic))
    return all(number in deterministic_numbers for number in candidate_numbers)


def _paraphrase_messages(deterministic: str, *, language: str) -> list[dict]:
    is_ja = language == "ja"
    system = (
        "あなたはサーバー監視アシスタントです。与えられた数値だけを使って、"
        "見やすく整理されたMarkdown形式（箇条書き）で状態を分かりやすく説明してください。"
        "新しい数値を作ったり、原因を推測したりしないでください。"
        if is_ja
        else (
            "Bạn là trợ lý giám sát máy chủ. Hãy trình bày tình trạng máy chủ "
            "bằng định dạng Markdown rõ ràng, chia từng mục gạch đầu dòng (- **Mục:** ...) "
            "xuống dòng ngăn nắp, dễ đọc. Chỉ dùng đúng các số liệu được cung cấp bên dưới, "
            "không được bịa thêm số liệu hoặc suy đoán nguyên nhân."
        )
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": deterministic},
    ]


class CctvaiHardwareService:
    """Answer CCTVAI host hardware-metrics questions.

    Dependency-injection pattern mirrors ``CctvaiQueryService``: pass a fake
    client/openai_client in tests; ``from_env()`` builds the real ones.
    """

    HEALTH_TTL = 30.0

    def __init__(
        self,
        *,
        client: CctvaiHardwareClient | None = None,
        openai_client: AsyncOpenAI | None = None,
        model: str | None = None,
        answer_max_tokens: int | None = None,
    ) -> None:
        self._client = client if client is not None else CctvaiHardwareClient.from_env()
        self.openai_client = openai_client or AsyncOpenAI(
            api_key=os.getenv("LITELLM_MASTER_KEY", "sk-local"),
            base_url=os.getenv("LITELLM_URL", "http://localhost:4000/v1"),
        )
        # Fixed, local-only model — deliberately not taken from the
        # caller's ``model``/``req.model`` so a user's cloud model choice
        # can never route hardware data off-box.
        self._model_name = (
            model
            or os.getenv("CCTVAI_HARDWARE_MODEL", "local-hardware-summary").strip()
            or "local-hardware-summary"
        )
        self._answer_max_tokens = (
            answer_max_tokens
            if answer_max_tokens is not None
            else _env_int(
                "CCTVAI_HARDWARE_ANSWER_MAX_TOKENS", 384, minimum=128, maximum=800
            )
        )
        self._last_health_ok: bool | None = None
        self._last_health_at: float | None = None

    @classmethod
    def from_env(cls) -> "CctvaiHardwareService | None":
        """Return ``None`` when hardware monitoring is disabled/unconfigured.

        Only reads env via ``CctvaiHardwareClient.from_env()`` — zero I/O,
        safe to call at import time.
        """
        client = CctvaiHardwareClient.from_env()
        if client is None:
            return None
        return cls(client=client)

    @property
    def available(self) -> bool:
        """Cached freshness of the last ``refresh_health()`` probe.

        Zero I/O. Mirrors ``CctvaiDatabase.available``: a probe older than
        ``HEALTH_TTL`` is treated as stale/unavailable rather than trusted
        indefinitely.
        """
        if self._last_health_at is None:
            return False
        if (time.monotonic() - self._last_health_at) >= self.HEALTH_TTL:
            return False
        return bool(self._last_health_ok)

    def status(self) -> dict:
        """Zero-I/O status for ``/health``."""
        if self._client is None:
            return {"enabled": False, "available": False}
        return {"enabled": True, "available": self.available}

    async def refresh_health(self) -> None:
        """Do the actual I/O probe. Call only from a background task (the
        lifespan's ``_hardware_health_loop``), never the hot request path."""
        if self._client is None:
            return
        try:
            await self._client.get_snapshot(force_refresh=True)
            self._last_health_ok = True
        except CctvaiHardwareClientError as exc:
            log.warning(
                "CCTVAI hardware health probe failed: %s", exc.__class__.__name__
            )
            self._last_health_ok = False
        self._last_health_at = time.monotonic()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def query_hardware_outcome(
        self,
        *,
        question: str,
        model: str,
        language: str = "vi",
    ) -> MesQueryOutcome:
        """Answer a hardware question from the live snapshot.

        ``model``/``question`` are accepted for interface symmetry with the
        other query services, but the actual model used is always the
        fixed, local-only ``self._model_name`` — never the caller's model
        choice — and the answer always covers the full snapshot rather than
        trying to guess a narrower scope from ``question`` wording.
        """
        del question  # answer always covers the full snapshot, see above
        del model  # local-only model is fixed; see self._model_name
        if self._client is None:
            return MesQueryOutcome(
                answer=(
                    "CCTVAI機能は有効化されていません。"
                    if language == "ja"
                    else "Chức năng giám sát phần cứng CCTVAI chưa được kích hoạt."
                ),
                results=[],
                routed_model=self._model_name,
                answer_scope="cctvai_hardware",
            )

        try:
            snapshot_result = await self._client.get_snapshot()
        except CctvaiHardwareClientError as exc:
            log.warning(
                "CCTVAI hardware snapshot fetch failed: %s", exc.__class__.__name__
            )
            return MesQueryOutcome(
                answer=_unavailable_answer(language, exc),
                results=[],
                routed_model=self._model_name,
                answer_scope="cctvai_hardware",
            )

        deterministic = render_deterministic_answer(snapshot_result, language=language)
        answer = await self._maybe_paraphrase(deterministic, language=language)
        return MesQueryOutcome(
            answer=answer,
            results=[],
            routed_model=self._model_name,
            answer_scope="cctvai_hardware",
        )

    async def _maybe_paraphrase(self, deterministic: str, *, language: str) -> str:
        """Ask the local model to rephrase the deterministic facts.

        Falls back to the deterministic text verbatim on any LLM failure,
        an empty response, or a response that introduces a number not
        already present in the deterministic text (grounding check).
        """
        try:
            response = await self.openai_client.chat.completions.create(
                model=self._model_name,
                messages=_paraphrase_messages(deterministic, language=language),
                temperature=0.2,
                max_tokens=self._answer_max_tokens,
            )
            candidate = (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - any LLM/network failure falls back
            log.warning(
                "CCTVAI hardware LLM paraphrase failed: %s", exc.__class__.__name__
            )
            return deterministic
        if not candidate or not _answer_numbers_are_grounded(candidate, deterministic):
            return deterministic
        return candidate

    async def query_hardware_stream_outcome(
        self,
        *,
        question: str,
        model: str,
        language: str = "vi",
    ) -> MesQueryStreamOutcome:
        """Wrap the non-streaming outcome in a one-token generator.

        Same pattern as ``query_wms_stream_outcome`` /
        ``query_cctvai_stream_outcome``: compute the full answer first, then
        expose it as a single-yield async generator so the SSE
        ``event_generator`` in ``main.py`` unwraps it like any other mode.
        """
        outcome = await self.query_hardware_outcome(
            question=question, model=model, language=language
        )

        async def token_generator():
            yield ("token", outcome.answer)

        return MesQueryStreamOutcome(
            token_stream=token_generator(),
            results=outcome.results,
            routed_model=outcome.routed_model,
            answer_scope=outcome.answer_scope,
        )
