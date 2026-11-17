"""Service layer for CCTVAI host hardware-metrics questions.

Talks to the independent ``tools/host_metrics`` HTTPS service on the CCTVAI
Ubuntu server via ``CctvaiHardwareClient``. Hardware snapshots are rendered
locally and deterministically; they are never sent to any LLM or translation
service. The service is independent of the PostgreSQL-backed CCTVAI replica.
"""

from __future__ import annotations

import logging
import re
import time

from tools.host_metrics.schemas import ComponentState, HostMetricsSnapshot

from .cctvai_hardware_client import (
    CctvaiHardwareClient,
    CctvaiHardwareClientError,
    HardwareSnapshotResult,
)
from .mes_query_service import MesQueryOutcome, MesQueryStreamOutcome

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# classify_hardware_question — pure, bilingual, deliberately narrow so the
# existing CCTVAI-replica intents (events/cameras/violations) are never
# shadowed by a stray keyword.
# ---------------------------------------------------------------------------

# Latin keywords must match as whole words: a plain substring test let
# "ram" fire on "program" and returned the full hardware snapshot. ASCII-only
# lookarounds (not ``\\b``) because Japanese characters are ``\\w`` too, so
# "サーバーのCPU" has no ``\\b`` before "CPU".
_HARDWARE_LATIN_KEYWORDS = (
    "cpu", "ram", "memory", "gpu", "vram", "disk", "storage", "network",
    "bandwidth", "uptime", "docker", "container", "hardware", "processor",
    "filesystem",
)
_HARDWARE_LATIN_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(_HARDWARE_LATIN_KEYWORDS) + r")(?![a-z0-9])"
)
_HARDWARE_PHRASES = (
    "bộ nhớ", "ổ đĩa", "ổ cứng", "dung lượng", "phần cứng", "băng thông",
    "thời gian hoạt động", "tiến trình", "vi xử lý", "card đồ họa",
    "メモリ", "ディスク", "ハードウェア", "ネットワーク", "稼働時間",
    "コンテナ", "プロセス",
)
_TEMPERATURE_PHRASES = ("nhiệt độ", "温度", "temperature")
_HOST_CONTEXT_PHRASES = ("server", "máy chủ", "host", "サーバ", "ホスト")


def classify_hardware_question(question: str) -> bool:
    """True when ``question`` plainly asks about CCTVAI host resources."""
    if not question or not question.strip():
        return False
    normalized = question.lower()
    if _HARDWARE_LATIN_PATTERN.search(normalized):
        return True
    if any(phrase in normalized for phrase in _HARDWARE_PHRASES):
        return True
    return any(phrase in normalized for phrase in _TEMPERATURE_PHRASES) and any(
        phrase in normalized for phrase in _HOST_CONTEXT_PHRASES
    )


# ---------------------------------------------------------------------------
# Deterministic bilingual rendering — no model sees the snapshot.
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
    return f"- **GPU:** {'; '.join(parts)}"


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
    return f"{prefix}{'; '.join(parts)}"


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
    prefix = "- **ネットワーク:** " if is_ja else "- **Mạng:** "
    return prefix + "; ".join(parts)


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
    """Render a validated snapshot as a bilingual, fully computed answer."""
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
    body = "\n".join(segment for segment in segments if segment)
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
    return "\n\n".join(part for part in [header, body, " ".join(notes)] if part).strip()


def _unavailable_answer(language: str, exc: CctvaiHardwareClientError) -> str:
    """Bilingual refusal exposing only the exception class, never its message."""
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


class CctvaiHardwareService:
    """Answer hardware questions using a deterministic renderer only."""

    HEALTH_TTL = 30.0

    def __init__(self, *, client: CctvaiHardwareClient | None = None) -> None:
        self._client = client if client is not None else CctvaiHardwareClient.from_env()
        self._last_health_ok: bool | None = None
        self._last_health_at: float | None = None

    @classmethod
    def from_env(cls) -> "CctvaiHardwareService | None":
        """Return ``None`` when hardware monitoring is disabled/unconfigured."""
        client = CctvaiHardwareClient.from_env()
        if client is None:
            return None
        return cls(client=client)

    @property
    def available(self) -> bool:
        """Cached freshness of the last ``refresh_health()`` probe."""
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
        """Probe the internal host-metrics service in the background."""
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
        """Render metrics directly; the question/model never leave this service."""
        del question, model
        if self._client is None:
            answer = (
                "CCTVAI機能は有効化されていません。"
                if language == "ja"
                else "Chức năng giám sát phần cứng CCTVAI chưa được kích hoạt."
            )
        else:
            try:
                snapshot_result = await self._client.get_snapshot()
            except CctvaiHardwareClientError as exc:
                log.warning(
                    "CCTVAI hardware snapshot fetch failed: %s", exc.__class__.__name__
                )
                answer = _unavailable_answer(language, exc)
            else:
                answer = render_deterministic_answer(snapshot_result, language=language)
        return MesQueryOutcome(
            answer=answer,
            results=[],
            routed_model="hardware-deterministic",
            answer_scope="cctvai_hardware",
        )

    async def query_hardware_stream_outcome(
        self,
        *,
        question: str,
        model: str,
        language: str = "vi",
    ) -> MesQueryStreamOutcome:
        """Wrap the deterministic answer in a one-token async generator."""
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
