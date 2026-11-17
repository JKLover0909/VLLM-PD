"""Unit tests for CCTVAI hardware metrics deterministic rendering and service."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import pytest

from tools.host_metrics.schemas import (
    ComponentState,
    Components,
    ComponentStatus,
    CpuStats,
    DiskStats,
    GpuStats,
    HostMetricsSnapshot,
    MemoryStats,
    NetworkStats,
    TruncationFlags,
)
from src.integrations.cctvai_hardware_client import (
    CctvaiHardwareClientError,
    HardwareSnapshotResult,
)
from src.integrations.cctvai_hardware_service import (
    CctvaiHardwareService,
    render_deterministic_answer,
)


def _ok() -> ComponentStatus:
    return ComponentStatus(status=ComponentState.OK)


def _sample_snapshot() -> HostMetricsSnapshot:
    return HostMetricsSnapshot(
        schema_version=1,
        collected_at=datetime.now(timezone.utc),
        sample_interval_seconds=10.0,
        cpu=CpuStats(
            utilization_percent=11.2,
            core_count_logical=48,
            core_count_physical=24,
            load_1m=1.5,
            load_5m=2.0,
            load_15m=2.5,
        ),
        memory=MemoryStats(
            total_bytes=124 * (1024**3),
            used_bytes=40 * (1024**3),
            available_bytes=84 * (1024**3),
            percent=32.2,
        ),
        gpus=[
            GpuStats(
                index=0,
                name="NVIDIA RTX PRO 5000",
                utilization_percent=43.0,
                memory_total_bytes=48 * (1024**3),
                memory_used_bytes=15 * (1024**3),
                temperature_celsius=71.0,
            )
        ],
        disks=[
            DiskStats(
                mountpoint="/", total_bytes=1864 * (1024**3),
                used_bytes=140 * (1024**3), percent=7.5,
            ),
            DiskStats(
                mountpoint="/tmp", total_bytes=1864 * (1024**3),
                used_bytes=140 * (1024**3), percent=7.5,
            ),
            DiskStats(
                mountpoint="/boot/efi", total_bytes=1 * (1024**3),
                used_bytes=50 * (1024**2), percent=5.0,
            ),
        ],
        network=[
            NetworkStats(
                interface="enp66s0f0",
                rx_bytes_per_sec=3.3 * 1024 * 1024,
                tx_bytes_per_sec=2.6 * 1024 * 1024,
            )
        ],
        uptime_seconds=7 * 86400 + 22 * 3600 + 120,
        processes=[],
        containers=[],
        components=Components(
            cpu=_ok(), memory=_ok(), gpu=_ok(), disk=_ok(), network=_ok(),
            uptime=_ok(), processes=_ok(), docker=_ok(),
        ),
        truncated=TruncationFlags(),
    )


class FakeHardwareClient:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.calls = 0
        self.closed = False

    async def get_snapshot(self, force_refresh=False):
        self.calls += 1
        if self.failure:
            raise self.failure
        return HardwareSnapshotResult(
            snapshot=_sample_snapshot(), age_seconds=2.0,
            fetched_monotonic=time.monotonic(),
        )

    async def aclose(self):
        self.closed = True


def test_render_deterministic_answer_is_structured_markdown():
    result = HardwareSnapshotResult(
        snapshot=_sample_snapshot(), age_seconds=2.0,
        fetched_monotonic=time.monotonic(),
    )
    answer = render_deterministic_answer(result, language="vi")
    assert "\n" in answer
    assert "📊 **Tình trạng phần cứng máy chủ CCTV AI:**" in answer
    assert "- **CPU:** 11% (48 lõi logic)" in answer
    assert "- **RAM:** 32%" in answer
    assert "- **GPU:** NVIDIA RTX PRO 5000 43%" in answer
    assert "- **Ổ đĩa:**" in answer
    assert "`/`" in answer
    assert "`/boot/efi`" in answer
    assert "`/tmp`" not in answer
    assert "- **Mạng:** `enp66s0f0`" in answer
    assert "- **Uptime:** 7 ngày 22 giờ 2 phút" in answer


def test_render_deterministic_answer_japanese():
    result = HardwareSnapshotResult(
        snapshot=_sample_snapshot(), age_seconds=2.0,
        fetched_monotonic=time.monotonic(),
    )
    answer = render_deterministic_answer(result, language="ja")
    assert "📊 **CCTVAI サーバーの稼働状態:**" in answer
    assert "- **CPU:** 使用率11%（論理48コア）" in answer
    assert "- **RAM (メモリ):** 使用率32%" in answer
    assert "- **稼働時間 (Uptime):** 7日 22時間 2分" in answer


@pytest.mark.parametrize("language", ["vi", "ja"])
def test_service_renders_verbatim_without_any_llm_client(language):
    client = FakeHardwareClient()
    service = CctvaiHardwareService(client=client)
    outcome = asyncio.run(
        service.query_hardware_outcome(
            question="GPU?", model="remote-model-must-be-ignored", language=language
        )
    )
    expected = render_deterministic_answer(
        HardwareSnapshotResult(
            snapshot=_sample_snapshot(), age_seconds=2.0,
            fetched_monotonic=time.monotonic(),
        ),
        language=language,
    )
    assert outcome.answer == expected
    assert outcome.routed_model == "hardware-deterministic"
    assert outcome.answer_scope == "cctvai_hardware"
    assert client.calls == 1
    assert not hasattr(service, "openai_client")
    assert not hasattr(service, "_model_name")


def test_service_handles_unavailable_snapshot_without_llm():
    client = FakeHardwareClient(failure=CctvaiHardwareClientError("hidden detail"))
    outcome = asyncio.run(
        CctvaiHardwareService(client=client).query_hardware_outcome(
            question="CPU?", model="local", language="ja"
        )
    )
    assert "CctvaiHardwareClientError" in outcome.answer
    assert "hidden detail" not in outcome.answer
    assert outcome.routed_model == "hardware-deterministic"
    assert client.calls == 1


@pytest.mark.parametrize("language", ["vi", "ja"])
def test_service_stream_yields_deterministic_answer_without_llm(language):
    client = FakeHardwareClient()
    outcome = asyncio.run(
        CctvaiHardwareService(client=client).query_hardware_stream_outcome(
            question="RAM?", model="remote-model-must-be-ignored", language=language
        )
    )
    assert outcome.routed_model == "hardware-deterministic"
    assert outcome.answer_scope == "cctvai_hardware"
    assert client.calls == 1

    async def collect():
        return [item async for item in outcome.token_stream]

    tokens = asyncio.run(collect())
    assert len(tokens) == 1
    assert tokens[0][0] == "token"
    assert ("CCTVAI サーバー" if language == "ja" else "Tình trạng") in tokens[0][1]
    assert "- **RAM" in tokens[0][1]
