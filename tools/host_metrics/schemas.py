"""Shared contract for the CCTVAI host metrics snapshot.

This module is the single source of truth for the JSON contract exchanged
between the standalone ``tools/host_metrics`` collector/server on the
Ubuntu host and any consumer (the ``server.py`` in this package, and the
Meibook ``cctvai_hardware_client`` running inside the app container).

Design constraints (see plan Markdowns handoff / AGENTS.md):
    * Pydantic v2 models only, ``extra="ignore"`` everywhere (forward
      compatible: unknown fields from a newer collector are dropped, not
      rejected).
    * Every list is bounded (``max_length``) and every numeric/string field
      has a sane bound -- a hostile or corrupted snapshot must fail
      validation instead of silently expanding memory/CPU on a consumer.
    * A value that could not be measured is ``null`` (``None``), never a
      fake ``0``. Rate-based metrics (CPU%, per-process CPU%, network
      rate) require two samples; the *first* sample after collector
      startup reports the relevant component as ``warming_up`` with the
      numeric fields left ``null`` instead of a misleading ``0``.
    * No secrets, no free-form command output, no process cmdline/env, no
      container env/logs/mounts. ``reason`` strings on components must be
      short, filtered, human-readable explanations -- never raw
      tracebacks or file paths.

This module must stay dependency-free beyond ``pydantic`` so it can be
imported from the Meibook app container without pulling in psutil/FastAPI.
Importing it must never perform I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Contract version and bounds. Both the collector/server in this package and
# any external client MUST import these constants rather than hard-coding
# duplicate limits.
# ---------------------------------------------------------------------------

SCHEMA_VERSION: Literal[1] = 1

# Bounded array sizes (plan section 4 defaults).
MAX_GPUS = 8
MAX_DISKS = 32
MAX_NETWORK_INTERFACES = 32
MAX_PROCESSES = 10
MAX_CONTAINERS = 100

# Bounded string lengths.
MAX_NAME_LENGTH = 256
MAX_SHORT_LABEL_LENGTH = 64
MAX_REASON_LENGTH = 256

# Snapshot transport bounds.
MAX_SNAPSHOT_BYTES = 256 * 1024  # 256 KiB

# Operational defaults (deployment can override via env; these are the
# documented plan defaults, kept here so client/server/collector agree).
DEFAULT_SAMPLE_INTERVAL_SECONDS = 10.0
DEFAULT_CLIENT_CACHE_SECONDS = 5.0
DEFAULT_MAX_AGE_SECONDS = 30.0
DEFAULT_MAX_CLOCK_SKEW_SECONDS = 30.0
DEFAULT_HTTP_TIMEOUT_SECONDS = 5.0


class ComponentState(str, Enum):
    """Per-component health used inside :class:`Components`."""

    OK = "ok"
    WARMING_UP = "warming_up"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ComponentStatus(BaseModel):
    """Health + short reason for a single metrics component."""

    model_config = ConfigDict(extra="ignore")

    status: ComponentState
    reason: Optional[str] = Field(default=None, max_length=MAX_REASON_LENGTH)


class Components(BaseModel):
    """Fixed, bounded set of component statuses (no arbitrary dict keys)."""

    model_config = ConfigDict(extra="ignore")

    cpu: ComponentStatus
    memory: ComponentStatus
    gpu: ComponentStatus
    disk: ComponentStatus
    network: ComponentStatus
    uptime: ComponentStatus
    processes: ComponentStatus
    docker: ComponentStatus


class TruncationFlags(BaseModel):
    """Set to True when the corresponding array was cut to its bound.

    A True flag means "this list is NOT the full list" -- callers must not
    describe a truncated array as complete.
    """

    model_config = ConfigDict(extra="ignore")

    gpus: bool = False
    disks: bool = False
    network: bool = False
    processes: bool = False
    containers: bool = False


class CpuStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    utilization_percent: Optional[float] = Field(default=None, ge=0, le=100)
    load_average_1m: Optional[float] = Field(default=None, ge=0, le=10_000)
    load_average_5m: Optional[float] = Field(default=None, ge=0, le=10_000)
    load_average_15m: Optional[float] = Field(default=None, ge=0, le=10_000)
    core_count_physical: Optional[int] = Field(default=None, ge=0, le=1024)
    core_count_logical: Optional[int] = Field(default=None, ge=0, le=4096)


class MemoryStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_bytes: Optional[int] = Field(default=None, ge=0)
    used_bytes: Optional[int] = Field(default=None, ge=0)
    available_bytes: Optional[int] = Field(default=None, ge=0)
    percent: Optional[float] = Field(default=None, ge=0, le=100)


class GpuStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int = Field(ge=0, le=63)
    name: Optional[str] = Field(default=None, max_length=MAX_NAME_LENGTH)
    utilization_percent: Optional[float] = Field(default=None, ge=0, le=100)
    memory_used_bytes: Optional[int] = Field(default=None, ge=0)
    memory_total_bytes: Optional[int] = Field(default=None, ge=0)
    temperature_celsius: Optional[float] = Field(default=None, ge=-50, le=200)
    power_watts: Optional[float] = Field(default=None, ge=0, le=2000)


class DiskStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mountpoint: str = Field(max_length=MAX_NAME_LENGTH)
    device: Optional[str] = Field(default=None, max_length=MAX_NAME_LENGTH)
    fstype: Optional[str] = Field(default=None, max_length=MAX_SHORT_LABEL_LENGTH)
    total_bytes: Optional[int] = Field(default=None, ge=0)
    used_bytes: Optional[int] = Field(default=None, ge=0)
    percent: Optional[float] = Field(default=None, ge=0, le=100)


class NetworkStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    interface: str = Field(max_length=MAX_SHORT_LABEL_LENGTH)
    rx_bytes_per_sec: Optional[float] = Field(default=None, ge=0)
    tx_bytes_per_sec: Optional[float] = Field(default=None, ge=0)


class ProcessStats(BaseModel):
    """Name + CPU/RAM only -- never cmdline or environment."""

    model_config = ConfigDict(extra="ignore")

    pid: Optional[int] = Field(default=None, ge=0)
    name: str = Field(max_length=MAX_NAME_LENGTH)
    cpu_percent: Optional[float] = Field(default=None, ge=0, le=6400)
    memory_percent: Optional[float] = Field(default=None, ge=0, le=100)
    memory_bytes: Optional[int] = Field(default=None, ge=0)


class ContainerStats(BaseModel):
    """State/health/restart count only -- never env, logs, or mounts."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(max_length=MAX_NAME_LENGTH)
    state: str = Field(max_length=MAX_SHORT_LABEL_LENGTH)
    health: Optional[str] = Field(default=None, max_length=MAX_SHORT_LABEL_LENGTH)
    restart_count: Optional[int] = Field(default=None, ge=0, le=1_000_000)


class HostMetricsSnapshot(BaseModel):
    """Top-level snapshot envelope: the whole JSON contract, v1."""

    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[1] = SCHEMA_VERSION
    collected_at: datetime
    sample_interval_seconds: Optional[float] = Field(default=None, ge=0, le=3600)

    cpu: Optional[CpuStats] = None
    memory: Optional[MemoryStats] = None
    gpus: List[GpuStats] = Field(default_factory=list, max_length=MAX_GPUS)
    disks: List[DiskStats] = Field(default_factory=list, max_length=MAX_DISKS)
    network: List[NetworkStats] = Field(
        default_factory=list, max_length=MAX_NETWORK_INTERFACES
    )
    uptime_seconds: Optional[float] = Field(default=None, ge=0)
    processes: List[ProcessStats] = Field(
        default_factory=list, max_length=MAX_PROCESSES
    )
    containers: List[ContainerStats] = Field(
        default_factory=list, max_length=MAX_CONTAINERS
    )
    components: Components
    truncated: TruncationFlags = Field(default_factory=TruncationFlags)

    @field_validator("collected_at")
    @classmethod
    def _collected_at_must_be_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("collected_at must be timezone-aware (UTC)")
        return value.astimezone(timezone.utc)


__all__ = [
    "SCHEMA_VERSION",
    "MAX_GPUS",
    "MAX_DISKS",
    "MAX_NETWORK_INTERFACES",
    "MAX_PROCESSES",
    "MAX_CONTAINERS",
    "MAX_NAME_LENGTH",
    "MAX_SHORT_LABEL_LENGTH",
    "MAX_REASON_LENGTH",
    "MAX_SNAPSHOT_BYTES",
    "DEFAULT_SAMPLE_INTERVAL_SECONDS",
    "DEFAULT_CLIENT_CACHE_SECONDS",
    "DEFAULT_MAX_AGE_SECONDS",
    "DEFAULT_MAX_CLOCK_SKEW_SECONDS",
    "DEFAULT_HTTP_TIMEOUT_SECONDS",
    "ComponentState",
    "ComponentStatus",
    "Components",
    "TruncationFlags",
    "CpuStats",
    "MemoryStats",
    "GpuStats",
    "DiskStats",
    "NetworkStats",
    "ProcessStats",
    "ContainerStats",
    "HostMetricsSnapshot",
]
