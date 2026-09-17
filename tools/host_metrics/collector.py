"""CCTVAI host metrics collector.

Samples CPU/RAM/GPU/disk/network/uptime/processes/Docker containers with
``psutil`` plus two fixed, bounded subprocess calls (``nvidia-smi`` and
``docker``), and atomically writes a :class:`~tools.host_metrics.schemas.
HostMetricsSnapshot` JSON document to a snapshot file. A separate,
unprivileged process (``server.py``) reads that file -- the collector and
the API never share in-memory state or a process.

Design notes:
    * Every subprocess call uses a fixed argv, ``shell=False``, a bounded
      timeout, and captured output is truncated to a hard byte cap before
      being parsed. No user input ever reaches argv.
    * Rate-based metrics (CPU%, per-process CPU%, network rx/tx) need two
      samples. On the very first ``collect()`` call after process start,
      and whenever an OS counter goes backwards (reboot/reset), the
      affected component is reported ``warming_up``/``degraded`` with the
      numeric fields left ``None`` -- never a fake ``0``.
    * Docker/GPU absence or permission errors are reported via
      ``components.docker`` / ``components.gpu`` with a short, filtered
      reason string. Raw stack traces / file paths are never put in
      ``reason``.
    * This module intentionally has no plugin/strategy abstractions --
      each metric family is one small, direct function.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

from tools.host_metrics.schemas import (
    MAX_CONTAINERS,
    MAX_DISKS,
    MAX_GPUS,
    MAX_NETWORK_INTERFACES,
    MAX_PROCESSES,
    ComponentState,
    ComponentStatus,
    Components,
    ContainerStats,
    CpuStats,
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
    DiskStats,
    GpuStats,
    HostMetricsSnapshot,
    MemoryStats,
    NetworkStats,
    ProcessStats,
    SCHEMA_VERSION,
    TruncationFlags,
)

logger = logging.getLogger("cctvai.host_metrics.collector")

DEFAULT_SNAPSHOT_PATH = "/run/cctvai-metrics/snapshot.json"

# Pseudo filesystems excluded from disk reporting (not real storage).
_DISK_FSTYPE_BLOCKLIST = frozenset(
    {
        "autofs",
        "bpf",
        "binfmt_misc",
        "cgroup",
        "cgroup2",
        "configfs",
        "debugfs",
        "devpts",
        "devtmpfs",
        "efivarfs",
        "fusectl",
        "hugetlbfs",
        "mqueue",
        "overlay",
        "proc",
        "pstore",
        "ramfs",
        "securityfs",
        "squashfs",
        "sysfs",
        "tmpfs",
        "tracefs",
    }
)

_MAX_SUBPROCESS_OUTPUT_BYTES = 64 * 1024  # per-command cap, well under snapshot cap


def _run_command(argv: List[str], timeout: float) -> subprocess.CompletedProcess:
    """Run a fixed argv command with no shell, bounded timeout and output.

    Isolated as its own function so tests can monkeypatch it instead of
    touching the real system.
    """

    result = subprocess.run(
        argv,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.stdout and len(result.stdout) > _MAX_SUBPROCESS_OUTPUT_BYTES:
        result.stdout = result.stdout[:_MAX_SUBPROCESS_OUTPUT_BYTES]
    if result.stderr and len(result.stderr) > _MAX_SUBPROCESS_OUTPUT_BYTES:
        result.stderr = result.stderr[:_MAX_SUBPROCESS_OUTPUT_BYTES]
    return result


def _ok(reason: Optional[str] = None) -> ComponentStatus:
    return ComponentStatus(status=ComponentState.OK, reason=reason)


def _warming_up(reason: str = "first sample after startup") -> ComponentStatus:
    return ComponentStatus(status=ComponentState.WARMING_UP, reason=reason)


def _degraded(reason: str) -> ComponentStatus:
    return ComponentStatus(status=ComponentState.DEGRADED, reason=reason[:200])


def _unavailable(reason: str) -> ComponentStatus:
    return ComponentStatus(status=ComponentState.UNAVAILABLE, reason=reason[:200])


def _error(reason: str) -> ComponentStatus:
    return ComponentStatus(status=ComponentState.ERROR, reason=reason[:200])


@dataclass
class _NetCounterState:
    bytes_sent: int
    bytes_recv: int
    timestamp: float


@dataclass
class CollectorConfig:
    docker_bin: str = "docker"
    nvidia_smi_bin: str = "nvidia-smi"
    command_timeout_seconds: float = 3.0
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS
    max_gpus: int = MAX_GPUS
    max_disks: int = MAX_DISKS
    max_network_interfaces: int = MAX_NETWORK_INTERFACES
    max_processes: int = MAX_PROCESSES
    max_containers: int = MAX_CONTAINERS
    exclude_network_interfaces: Tuple[str, ...] = ("lo",)


class HostMetricsCollector:
    """Stateful sampler. One instance per collector process."""

    def __init__(self, config: Optional[CollectorConfig] = None) -> None:
        self.config = config or CollectorConfig()
        self._primed = False
        self._last_net: Dict[str, _NetCounterState] = {}
        self._proc_registry: Dict[int, psutil.Process] = {}

    # -- top level -----------------------------------------------------

    def collect(self) -> HostMetricsSnapshot:
        now = datetime.now(timezone.utc)

        cpu, cpu_component = self._collect_cpu()
        memory, memory_component = self._collect_memory()
        gpus, gpu_component, gpus_truncated = self._collect_gpus()
        disks, disk_component, disks_truncated = self._collect_disks()
        network, network_component, network_truncated = self._collect_network()
        uptime, uptime_component = self._collect_uptime()
        processes, proc_component, processes_truncated = self._collect_processes()
        containers, docker_component, containers_truncated = self._collect_containers()

        snapshot = HostMetricsSnapshot(
            schema_version=SCHEMA_VERSION,
            collected_at=now,
            sample_interval_seconds=self.config.sample_interval_seconds
            if self._primed
            else None,
            cpu=cpu,
            memory=memory,
            gpus=gpus,
            disks=disks,
            network=network,
            uptime_seconds=uptime,
            processes=processes,
            containers=containers,
            components=Components(
                cpu=cpu_component,
                memory=memory_component,
                gpu=gpu_component,
                disk=disk_component,
                network=network_component,
                uptime=uptime_component,
                processes=proc_component,
                docker=docker_component,
            ),
            truncated=TruncationFlags(
                gpus=gpus_truncated,
                disks=disks_truncated,
                network=network_truncated,
                processes=processes_truncated,
                containers=containers_truncated,
            ),
        )
        self._primed = True
        return snapshot

    # -- CPU -------------------------------------------------------------

    def _collect_cpu(self) -> Tuple[Optional[CpuStats], ComponentStatus]:
        try:
            # psutil caches the previous call internally; the first call in
            # a process's lifetime always returns 0.0 which is meaningless.
            raw_percent = psutil.cpu_percent(interval=None)
            physical = psutil.cpu_count(logical=False)
            logical = psutil.cpu_count(logical=True)
            try:
                load1, load5, load15 = psutil.getloadavg()
            except (OSError, AttributeError):
                load1 = load5 = load15 = None
        except Exception as exc:  # pragma: no cover - defensive
            return None, _error(f"cpu sampling failed: {type(exc).__name__}")

        if not self._primed:
            stats = CpuStats(
                utilization_percent=None,
                load_average_1m=load1,
                load_average_5m=load5,
                load_average_15m=load15,
                core_count_physical=physical,
                core_count_logical=logical,
            )
            return stats, _warming_up()

        stats = CpuStats(
            utilization_percent=raw_percent,
            load_average_1m=load1,
            load_average_5m=load5,
            load_average_15m=load15,
            core_count_physical=physical,
            core_count_logical=logical,
        )
        return stats, _ok()

    # -- memory ------------------------------------------------------------

    def _collect_memory(self) -> Tuple[Optional[MemoryStats], ComponentStatus]:
        try:
            vm = psutil.virtual_memory()
        except Exception as exc:  # pragma: no cover - defensive
            return None, _error(f"memory sampling failed: {type(exc).__name__}")
        stats = MemoryStats(
            total_bytes=vm.total,
            used_bytes=vm.used,
            available_bytes=vm.available,
            percent=vm.percent,
        )
        return stats, _ok()

    # -- GPU (nvidia-smi) ----------------------------------------------------

    def _collect_gpus(self) -> Tuple[List[GpuStats], ComponentStatus, bool]:
        argv = [
            self.config.nvidia_smi_bin,
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = _run_command(argv, self.config.command_timeout_seconds)
        except FileNotFoundError:
            return [], _unavailable("nvidia-smi not found"), False
        except subprocess.TimeoutExpired:
            return [], _error("nvidia-smi timed out"), False
        except OSError as exc:
            return [], _error(f"nvidia-smi failed: {type(exc).__name__}"), False

        if result.returncode != 0:
            return [], _unavailable("nvidia-smi returned an error (no GPU/driver?)"), False

        gpus: List[GpuStats] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            gpu = _parse_gpu_line(line)
            if gpu is not None:
                gpus.append(gpu)

        truncated = len(gpus) > self.config.max_gpus
        if truncated:
            gpus = gpus[: self.config.max_gpus]

        if not gpus:
            return [], _unavailable("no GPUs reported by nvidia-smi"), False
        return gpus, _ok(), truncated

    # -- disks ------------------------------------------------------------

    def _collect_disks(self) -> Tuple[List[DiskStats], ComponentStatus, bool]:
        try:
            partitions = psutil.disk_partitions(all=False)
        except Exception as exc:  # pragma: no cover - defensive
            return [], _error(f"disk enumeration failed: {type(exc).__name__}"), False

        disks: List[DiskStats] = []
        denied = 0
        for part in partitions:
            if part.fstype in _DISK_FSTYPE_BLOCKLIST:
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except PermissionError:
                denied += 1
                continue
            except OSError:
                continue
            disks.append(
                DiskStats(
                    mountpoint=part.mountpoint,
                    device=part.device or None,
                    fstype=part.fstype or None,
                    total_bytes=usage.total,
                    used_bytes=usage.used,
                    percent=usage.percent,
                )
            )

        truncated = len(disks) > self.config.max_disks
        if truncated:
            disks = disks[: self.config.max_disks]

        if not disks:
            reason = (
                "no accessible real filesystems"
                if denied == 0
                else "all real filesystems denied permission"
            )
            return [], _degraded(reason), truncated
        if denied:
            return disks, _degraded(f"{denied} filesystem(s) denied permission"), truncated
        return disks, _ok(), truncated

    # -- network ------------------------------------------------------------

    def _collect_network(self) -> Tuple[List[NetworkStats], ComponentStatus, bool]:
        try:
            counters = psutil.net_io_counters(pernic=True)
        except Exception as exc:  # pragma: no cover - defensive
            return [], _error(f"network sampling failed: {type(exc).__name__}"), False

        now = time.monotonic()
        excluded = set(self.config.exclude_network_interfaces)
        interfaces = [name for name in counters if name not in excluded]
        truncated = len(interfaces) > self.config.max_network_interfaces
        interfaces = interfaces[: self.config.max_network_interfaces]

        results: List[NetworkStats] = []
        any_ready = False
        any_warming = False
        for name in interfaces:
            snap = counters[name]
            prev = self._last_net.get(name)
            self._last_net[name] = _NetCounterState(
                bytes_sent=snap.bytes_sent, bytes_recv=snap.bytes_recv, timestamp=now
            )
            if prev is None:
                results.append(NetworkStats(interface=name, rx_bytes_per_sec=None, tx_bytes_per_sec=None))
                any_warming = True
                continue

            dt = now - prev.timestamp
            if dt <= 0 or snap.bytes_recv < prev.bytes_recv or snap.bytes_sent < prev.bytes_sent:
                # Counter reset (interface reset/reboot) or non-monotonic
                # clock -- rate is unknown, not zero.
                results.append(NetworkStats(interface=name, rx_bytes_per_sec=None, tx_bytes_per_sec=None))
                any_warming = True
                continue

            rx_rate = (snap.bytes_recv - prev.bytes_recv) / dt
            tx_rate = (snap.bytes_sent - prev.bytes_sent) / dt
            results.append(
                NetworkStats(interface=name, rx_bytes_per_sec=rx_rate, tx_bytes_per_sec=tx_rate)
            )
            any_ready = True

        if not results:
            return [], _degraded("no network interfaces reported"), truncated
        if any_warming and not any_ready:
            return results, _warming_up("network rate needs a second sample"), truncated
        if any_warming:
            return results, _degraded("some interfaces still warming up"), truncated
        return results, _ok(), truncated

    # -- uptime ------------------------------------------------------------

    def _collect_uptime(self) -> Tuple[Optional[float], ComponentStatus]:
        try:
            boot_time = psutil.boot_time()
        except Exception as exc:  # pragma: no cover - defensive
            return None, _error(f"uptime sampling failed: {type(exc).__name__}")
        uptime = max(0.0, time.time() - boot_time)
        return uptime, _ok()

    # -- processes ------------------------------------------------------------

    def _collect_processes(self) -> Tuple[List[ProcessStats], ComponentStatus, bool]:
        try:
            live_pids = set(psutil.pids())
        except Exception as exc:  # pragma: no cover - defensive
            return [], _error(f"process enumeration failed: {type(exc).__name__}"), False

        # Drop registry entries for processes that exited.
        for pid in list(self._proc_registry.keys()):
            if pid not in live_pids:
                del self._proc_registry[pid]

        candidates = []
        warming = 0
        for pid in live_pids:
            proc = self._proc_registry.get(pid)
            is_new = proc is None
            if is_new:
                try:
                    proc = psutil.Process(pid)
                    proc.cpu_percent(None)  # prime; first read is meaningless
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                self._proc_registry[pid] = proc

            try:
                name = proc.name()
                cpu_percent = None if is_new else proc.cpu_percent(None)
                mem_percent = proc.memory_percent()
                mem_bytes = proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

            if is_new:
                warming += 1
            candidates.append((pid, name, cpu_percent, mem_percent, mem_bytes))

        candidates.sort(key=lambda item: (item[2] or 0.0, item[3] or 0.0), reverse=True)
        truncated = len(candidates) > self.config.max_processes
        top = candidates[: self.config.max_processes]

        processes = [
            ProcessStats(
                pid=pid,
                name=name[: 256],
                cpu_percent=cpu_percent,
                memory_percent=mem_percent,
                memory_bytes=mem_bytes,
            )
            for pid, name, cpu_percent, mem_percent, mem_bytes in top
        ]

        if not processes:
            return [], _degraded("no processes visible"), truncated
        if not self._primed:
            return processes, _warming_up(), truncated
        return processes, _ok(), truncated

    # -- docker containers ------------------------------------------------------------

    def _collect_containers(self) -> Tuple[List[ContainerStats], ComponentStatus, bool]:
        list_argv = [
            self.config.docker_bin,
            "ps",
            "-a",
            "--no-trunc",
            "--format",
            "{{.Names}}\t{{.State}}",
        ]
        try:
            list_result = _run_command(list_argv, self.config.command_timeout_seconds)
        except FileNotFoundError:
            return [], _unavailable("docker CLI not found"), False
        except subprocess.TimeoutExpired:
            return [], _error("docker ps timed out"), False
        except OSError as exc:
            return [], _error(f"docker ps failed: {type(exc).__name__}"), False

        if list_result.returncode != 0:
            reason = "permission denied" if "permission denied" in (list_result.stderr or "").lower() else "docker ps returned an error"
            return [], _unavailable(reason), False

        entries: List[Tuple[str, str]] = []
        for line in list_result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            entries.append((parts[0], parts[1]))

        truncated = len(entries) > self.config.max_containers
        entries = entries[: self.config.max_containers]

        health_by_name: Dict[str, Tuple[Optional[str], Optional[int]]] = {}
        if entries:
            inspect_argv = [
                self.config.docker_bin,
                "inspect",
                "--format",
                "{{.Name}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}|{{.RestartCount}}",
            ] + [name for name, _state in entries]
            try:
                inspect_result = _run_command(inspect_argv, self.config.command_timeout_seconds)
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                inspect_result = None

            if inspect_result is not None and inspect_result.returncode == 0:
                for line in inspect_result.stdout.splitlines():
                    fields = line.split("|")
                    if len(fields) != 3:
                        continue
                    raw_name, health, restart_count = fields
                    name = raw_name.lstrip("/")
                    try:
                        restarts = int(restart_count)
                    except ValueError:
                        restarts = None
                    health_by_name[name] = (health or None, restarts)

        containers = []
        for name, state in entries:
            health, restarts = health_by_name.get(name, (None, None))
            containers.append(
                ContainerStats(
                    name=name[:256],
                    state=state[:64],
                    health=health,
                    restart_count=restarts,
                )
            )

        return containers, _ok(), truncated


def _parse_gpu_line(line: str) -> Optional[GpuStats]:
    fields = [f.strip() for f in line.split(",")]
    if len(fields) != 7:
        return None
    raw_index, name, util, mem_used, mem_total, temp, power = fields

    def _to_int(value: str) -> Optional[int]:
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None

    def _to_float(value: str) -> Optional[float]:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    index = _to_int(raw_index)
    if index is None:
        return None

    return GpuStats(
        index=index,
        name=(name or None) and name[:256],
        utilization_percent=_to_float(util),
        memory_used_bytes=(lambda v: v * 1024 * 1024 if v is not None else None)(_to_int(mem_used)),
        memory_total_bytes=(lambda v: v * 1024 * 1024 if v is not None else None)(_to_int(mem_total)),
        temperature_celsius=_to_float(temp),
        power_watts=_to_float(power),
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".snapshot-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, 0o640)
        os.replace(tmp_name, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def write_snapshot(collector: HostMetricsCollector, path: Path) -> HostMetricsSnapshot:
    """Collect one sample and atomically persist it. Returns the snapshot."""

    snapshot = collector.collect()
    payload = json.loads(snapshot.model_dump_json())
    _atomic_write_json(path, payload)
    return snapshot


def run_forever(
    snapshot_path: Path,
    interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    config: Optional[CollectorConfig] = None,
) -> None:
    """Blocking collection loop; exits cleanly on SIGTERM/SIGINT."""

    stop = {"flag": False}

    def _handle_signal(signum, frame):  # noqa: ANN001 - signal handler signature
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    cfg = config or CollectorConfig(sample_interval_seconds=interval_seconds)
    collector = HostMetricsCollector(cfg)

    logger.info("cctvai-host-metrics collector starting, snapshot=%s interval=%ss", snapshot_path, interval_seconds)
    while not stop["flag"]:
        started = time.monotonic()
        try:
            write_snapshot(collector, snapshot_path)
        except Exception:  # pragma: no cover - defensive top-level guard
            logger.exception("host metrics collection cycle failed")
        elapsed = time.monotonic() - started
        remaining = max(0.0, interval_seconds - elapsed)
        # Sleep in short slices so SIGTERM is honored promptly.
        end = time.monotonic() + remaining
        while not stop["flag"] and time.monotonic() < end:
            time.sleep(min(0.5, max(0.0, end - time.monotonic())))
    logger.info("cctvai-host-metrics collector stopping")


def main() -> None:  # pragma: no cover - process entrypoint
    logging.basicConfig(level=os.environ.get("HOST_METRICS_LOG_LEVEL", "INFO"))
    snapshot_path = Path(os.environ.get("HOST_METRICS_SNAPSHOT_PATH", DEFAULT_SNAPSHOT_PATH))
    interval = float(os.environ.get("HOST_METRICS_SAMPLE_INTERVAL_SECONDS", DEFAULT_SAMPLE_INTERVAL_SECONDS))
    docker_bin = os.environ.get("HOST_METRICS_DOCKER_BIN", "docker")
    nvidia_smi_bin = os.environ.get("HOST_METRICS_NVIDIA_SMI_BIN", "nvidia-smi")
    config = CollectorConfig(
        docker_bin=docker_bin,
        nvidia_smi_bin=nvidia_smi_bin,
        sample_interval_seconds=interval,
    )
    run_forever(snapshot_path, interval_seconds=interval, config=config)


if __name__ == "__main__":  # pragma: no cover
    main()
