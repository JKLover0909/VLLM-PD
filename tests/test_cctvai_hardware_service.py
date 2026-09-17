"""Unit tests for CCTVAI hardware metrics rendering and service logic.

Tests the deterministic markdown rendering and number-grounding validation
without making actual network calls.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

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
from src.integrations.cctvai_hardware_client import HardwareSnapshotResult
from src.integrations.cctvai_hardware_service import (
    _answer_numbers_are_grounded,
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
                mountpoint="/",
                total_bytes=1864 * (1024**3),
                used_bytes=140 * (1024**3),
                percent=7.5,
            ),
            # Trùng dung lượng với / (bind mounts /tmp, /var/tmp)
            DiskStats(
                mountpoint="/tmp",
                total_bytes=1864 * (1024**3),
                used_bytes=140 * (1024**3),
                percent=7.5,
            ),
            DiskStats(
                mountpoint="/boot/efi",
                total_bytes=1 * (1024**3),
                used_bytes=50 * (1024**2),
                percent=5.0,
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
            cpu=_ok(),
            memory=_ok(),
            gpu=_ok(),
            disk=_ok(),
            network=_ok(),
            uptime=_ok(),
            processes=_ok(),
            docker=_ok(),
        ),
        truncated=TruncationFlags(),
    )


def test_render_deterministic_answer_is_structured_markdown():
    snapshot = _sample_snapshot()
    result = HardwareSnapshotResult(
        snapshot=snapshot, age_seconds=2.0, fetched_monotonic=time.monotonic()
    )
    answer = render_deterministic_answer(result, language="vi")

    # Kiểm tra cấu trúc phân đoạn và gạch đầu dòng rõ ràng
    assert "\n" in answer
    assert "📊 **Tình trạng phần cứng máy chủ CCTV AI:**" in answer
    assert "- **CPU:** 11% (48 lõi logic)" in answer
    assert "- **RAM:** 32%" in answer
    assert "- **GPU:** NVIDIA RTX PRO 5000 43%" in answer
    assert "- **Ổ đĩa:**" in answer
    # Kiểm tra mountpoint / và /boot/efi xuất hiện, nhưng /tmp trùng size bị lọc
    assert "`/`" in answer
    assert "`/boot/efi`" in answer
    assert "`/tmp`" not in answer  # deduplicated!
    assert "- **Mạng:** `enp66s0f0`" in answer
    assert "- **Uptime:** 7 ngày 22 giờ 2 phút" in answer


def test_render_deterministic_answer_japanese():
    snapshot = _sample_snapshot()
    result = HardwareSnapshotResult(
        snapshot=snapshot, age_seconds=2.0, fetched_monotonic=time.monotonic()
    )
    answer = render_deterministic_answer(result, language="ja")

    assert "📊 **CCTVAI サーバーの稼働状態:**" in answer
    assert "- **CPU:** 使用率11%（論理48コア）" in answer
    assert "- **RAM (メモリ):** 使用率32%" in answer
    assert "- **稼働時間 (Uptime):** 7日 22時間 2分" in answer


def test_answer_numbers_are_grounded_with_markdown():
    deterministic = (
        "📊 **Tình trạng phần cứng:**\n\n"
        "- **CPU:** 15% (24 lõi logic)\n"
        "- **RAM:** 45% (32.0GB/64.0GB)"
    )
    # Câu trả lời Markdown hợp lệ dùng đúng các số có sẵn
    valid_paraphrase = (
        "Server đang chạy ổn định:\n"
        "- CPU ở mức 15% với 24 lõi logic.\n"
        "- Bộ nhớ RAM sử dụng 45% (32.0GB / 64.0GB)."
    )
    assert _answer_numbers_are_grounded(valid_paraphrase, deterministic)

    # Bịa số mới (99) phải bị từ chối
    hallucinated = "CPU đang quá tải ở mức 99%."
    assert not _answer_numbers_are_grounded(hallucinated, deterministic)
