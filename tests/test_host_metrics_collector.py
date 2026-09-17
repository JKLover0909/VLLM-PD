"""Tests for tools/host_metrics/collector.py.

All psutil/subprocess access is monkeypatched -- no real system operations
(no real nvidia-smi/docker/process enumeration) are performed by this
suite, per the plan's fixture-only testing requirement.
"""

import json
import subprocess
from datetime import datetime
from types import SimpleNamespace

import pytest

from tools.host_metrics import collector as c
from tools.host_metrics import schemas as s


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class FakeProcess:
    def __init__(self, pid, name, cpu_percent_value, memory_percent_value, memory_rss):
        self.pid = pid
        self._name = name
        self._cpu_percent_value = cpu_percent_value
        self._memory_percent_value = memory_percent_value
        self._memory_rss = memory_rss
        self.cpu_percent_calls = 0

    def cpu_percent(self, interval=None):
        self.cpu_percent_calls += 1
        return self._cpu_percent_value

    def name(self):
        return self._name

    def memory_percent(self):
        return self._memory_percent_value

    def memory_info(self):
        return SimpleNamespace(rss=self._memory_rss)


def _make_collector(**kwargs) -> c.HostMetricsCollector:
    config = c.CollectorConfig(**kwargs)
    return c.HostMetricsCollector(config)


def _fake_completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# CPU: warming up then ok
# ---------------------------------------------------------------------------


def test_cpu_first_sample_is_warming_up(monkeypatch):
    monkeypatch.setattr(c.psutil, "cpu_percent", lambda interval=None: 0.0)
    monkeypatch.setattr(c.psutil, "cpu_count", lambda logical=True: 8 if logical else 4)
    monkeypatch.setattr(c.psutil, "getloadavg", lambda: (0.1, 0.2, 0.3))
    monkeypatch.setattr(c.psutil, "virtual_memory", lambda: SimpleNamespace(total=1, used=1, available=1, percent=1.0))
    monkeypatch.setattr(c.psutil, "boot_time", lambda: 0.0)
    monkeypatch.setattr(c.psutil, "disk_partitions", lambda all=False: [])
    monkeypatch.setattr(c.psutil, "net_io_counters", lambda pernic=True: {})
    monkeypatch.setattr(c.psutil, "pids", lambda: [])
    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: (_ for _ in ()).throw(FileNotFoundError()))

    col = _make_collector()
    snap = col.collect()

    assert snap.components.cpu.status == s.ComponentState.WARMING_UP
    assert snap.cpu.utilization_percent is None
    assert snap.cpu.core_count_physical == 4
    assert snap.cpu.core_count_logical == 8


def test_cpu_second_sample_is_ok(monkeypatch):
    monkeypatch.setattr(c.psutil, "cpu_percent", lambda interval=None: 37.5)
    monkeypatch.setattr(c.psutil, "cpu_count", lambda logical=True: 8 if logical else 4)
    monkeypatch.setattr(c.psutil, "getloadavg", lambda: (0.1, 0.2, 0.3))
    monkeypatch.setattr(c.psutil, "virtual_memory", lambda: SimpleNamespace(total=1, used=1, available=1, percent=1.0))
    monkeypatch.setattr(c.psutil, "boot_time", lambda: 0.0)
    monkeypatch.setattr(c.psutil, "disk_partitions", lambda all=False: [])
    monkeypatch.setattr(c.psutil, "net_io_counters", lambda pernic=True: {})
    monkeypatch.setattr(c.psutil, "pids", lambda: [])
    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: (_ for _ in ()).throw(FileNotFoundError()))

    col = _make_collector()
    col.collect()  # prime
    snap = col.collect()

    assert snap.components.cpu.status == s.ComponentState.OK
    assert snap.cpu.utilization_percent == 37.5
    assert snap.sample_interval_seconds == col.config.sample_interval_seconds


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def _patch_common(monkeypatch, primed_cpu=True):
    monkeypatch.setattr(c.psutil, "cpu_percent", lambda interval=None: 1.0 if primed_cpu else 0.0)
    monkeypatch.setattr(c.psutil, "cpu_count", lambda logical=True: 8 if logical else 4)
    monkeypatch.setattr(c.psutil, "getloadavg", lambda: (0.1, 0.2, 0.3))
    monkeypatch.setattr(c.psutil, "virtual_memory", lambda: SimpleNamespace(total=100, used=50, available=50, percent=50.0))
    monkeypatch.setattr(c.psutil, "boot_time", lambda: 0.0)
    monkeypatch.setattr(c.psutil, "disk_partitions", lambda all=False: [])
    monkeypatch.setattr(c.psutil, "net_io_counters", lambda pernic=True: {})
    monkeypatch.setattr(c.psutil, "pids", lambda: [])


def test_gpu_present_full_fields(monkeypatch):
    _patch_common(monkeypatch)
    csv = (
        "0, NVIDIA GeForce RTX 3090, 15, 2048, 24576, 45, 120.50\n"
        "1, NVIDIA GeForce RTX 3090, 5, 512, 24576, 40, 80.00\n"
    )

    def fake_run(argv, timeout):
        if argv[0] == "nvidia-smi":
            return _fake_completed(stdout=csv)
        if argv[0] == "docker":
            return _fake_completed(stdout="")  # no containers in this test
        raise AssertionError(f"unexpected command {argv}")

    monkeypatch.setattr(c, "_run_command", fake_run)

    col = _make_collector()
    snap = col.collect()

    assert snap.components.gpu.status == s.ComponentState.OK
    assert len(snap.gpus) == 2
    gpu0 = snap.gpus[0]
    assert gpu0.index == 0
    assert gpu0.name == "NVIDIA GeForce RTX 3090"
    assert gpu0.utilization_percent == 15.0
    assert gpu0.memory_used_bytes == 2048 * 1024 * 1024
    assert gpu0.memory_total_bytes == 24576 * 1024 * 1024
    assert gpu0.temperature_celsius == 45.0
    assert gpu0.power_watts == 120.50


def test_gpu_missing_binary_reports_unavailable(monkeypatch):
    _patch_common(monkeypatch)

    def fake_run(argv, timeout):
        raise FileNotFoundError()

    monkeypatch.setattr(c, "_run_command", fake_run)

    col = _make_collector()
    snap = col.collect()

    assert snap.gpus == []
    assert snap.components.gpu.status == s.ComponentState.UNAVAILABLE
    assert snap.components.gpu.reason
    assert "traceback" not in snap.components.gpu.reason.lower()


def test_gpu_timeout_reports_error(monkeypatch):
    _patch_common(monkeypatch)

    def fake_run(argv, timeout):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)

    monkeypatch.setattr(c, "_run_command", fake_run)

    col = _make_collector()
    snap = col.collect()

    assert snap.gpus == []
    assert snap.components.gpu.status == s.ComponentState.ERROR


def test_gpu_truncated_beyond_max(monkeypatch):
    _patch_common(monkeypatch)
    lines = [f"{i}, GPU{i}, 1, 1, 1, 1, 1" for i in range(s.MAX_GPUS + 3)]
    csv = "\n".join(lines) + "\n"

    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: _fake_completed(stdout=csv))

    col = _make_collector()
    snap = col.collect()

    assert len(snap.gpus) == s.MAX_GPUS
    assert snap.truncated.gpus is True


# ---------------------------------------------------------------------------
# Docker / containers
# ---------------------------------------------------------------------------


def test_docker_permission_denied_reports_unavailable(monkeypatch):
    _patch_common(monkeypatch)

    def fake_run(argv, timeout):
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError()
        if argv[0] == "docker" and argv[1] == "ps":
            return _fake_completed(stdout="", stderr="permission denied", returncode=1)
        raise AssertionError("inspect should not be called when ps fails")

    monkeypatch.setattr(c, "_run_command", fake_run)

    col = _make_collector()
    snap = col.collect()

    assert snap.containers == []
    assert snap.components.docker.status == s.ComponentState.UNAVAILABLE
    assert "permission" in (snap.components.docker.reason or "").lower()


def test_docker_missing_binary_reports_unavailable(monkeypatch):
    _patch_common(monkeypatch)

    def fake_run(argv, timeout):
        raise FileNotFoundError()

    monkeypatch.setattr(c, "_run_command", fake_run)

    col = _make_collector()
    snap = col.collect()

    assert snap.components.docker.status == s.ComponentState.UNAVAILABLE


def test_docker_timeout_reports_error(monkeypatch):
    _patch_common(monkeypatch)

    def fake_run(argv, timeout):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)

    monkeypatch.setattr(c, "_run_command", fake_run)

    col = _make_collector()
    snap = col.collect()

    assert snap.components.docker.status == s.ComponentState.ERROR


def test_docker_containers_merge_health_and_restart_count(monkeypatch):
    _patch_common(monkeypatch)

    def fake_run(argv, timeout):
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError()
        if argv[0] == "docker" and argv[1] == "ps":
            return _fake_completed(stdout="camera-a\trunning\ncamera-b\texited\n")
        if argv[0] == "docker" and argv[1] == "inspect":
            assert "camera-a" in argv
            assert "camera-b" in argv
            return _fake_completed(
                stdout="/camera-a|healthy|2\n/camera-b||0\n"
            )
        raise AssertionError(f"unexpected command {argv}")

    monkeypatch.setattr(c, "_run_command", fake_run)

    col = _make_collector()
    snap = col.collect()

    assert snap.components.docker.status == s.ComponentState.OK
    by_name = {ct.name: ct for ct in snap.containers}
    assert by_name["camera-a"].state == "running"
    assert by_name["camera-a"].health == "healthy"
    assert by_name["camera-a"].restart_count == 2
    assert by_name["camera-b"].health is None
    assert by_name["camera-b"].restart_count == 0
    # No env/logs/mounts leak through the model at all (contract test in
    # test_host_metrics_schemas.py covers the field set exhaustively).


def test_docker_containers_truncated_beyond_max(monkeypatch):
    _patch_common(monkeypatch)
    names = [f"c{i}" for i in range(s.MAX_CONTAINERS + 5)]
    ps_stdout = "\n".join(f"{n}\trunning" for n in names) + "\n"

    def fake_run(argv, timeout):
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError()
        if argv[0] == "docker" and argv[1] == "ps":
            return _fake_completed(stdout=ps_stdout)
        if argv[0] == "docker" and argv[1] == "inspect":
            assert len(argv) - 4 == s.MAX_CONTAINERS  # only the truncated set was inspected
            return _fake_completed(stdout="")
        raise AssertionError(f"unexpected command {argv}")

    monkeypatch.setattr(c, "_run_command", fake_run)

    col = _make_collector()
    snap = col.collect()

    assert len(snap.containers) == s.MAX_CONTAINERS
    assert snap.truncated.containers is True


# ---------------------------------------------------------------------------
# Disk filtering
# ---------------------------------------------------------------------------


def test_disk_filters_pseudo_filesystems(monkeypatch):
    _patch_common(monkeypatch)

    partitions = [
        SimpleNamespace(device="/dev/sda1", mountpoint="/", fstype="ext4"),
        SimpleNamespace(device="tmpfs", mountpoint="/run", fstype="tmpfs"),
        SimpleNamespace(device="overlay", mountpoint="/var/lib/docker", fstype="overlay"),
        SimpleNamespace(device="/dev/sdb1", mountpoint="/data", fstype="xfs"),
    ]
    usage_by_mount = {
        "/": SimpleNamespace(total=100, used=40, percent=40.0),
        "/data": SimpleNamespace(total=200, used=20, percent=10.0),
    }

    monkeypatch.setattr(c.psutil, "disk_partitions", lambda all=False: partitions)
    monkeypatch.setattr(c.psutil, "disk_usage", lambda mountpoint: usage_by_mount[mountpoint])
    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: (_ for _ in ()).throw(FileNotFoundError()))

    col = _make_collector()
    snap = col.collect()

    mountpoints = {d.mountpoint for d in snap.disks}
    assert mountpoints == {"/", "/data"}
    assert snap.components.disk.status == s.ComponentState.OK


def test_disk_permission_denied_is_degraded_not_zero(monkeypatch):
    _patch_common(monkeypatch)

    partitions = [SimpleNamespace(device="/dev/sda1", mountpoint="/secure", fstype="ext4")]

    def raise_permission_error(mountpoint):
        raise PermissionError()

    monkeypatch.setattr(c.psutil, "disk_partitions", lambda all=False: partitions)
    monkeypatch.setattr(c.psutil, "disk_usage", raise_permission_error)
    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: (_ for _ in ()).throw(FileNotFoundError()))

    col = _make_collector()
    snap = col.collect()

    assert snap.disks == []
    assert snap.components.disk.status == s.ComponentState.DEGRADED


# ---------------------------------------------------------------------------
# Network: warming up + counter reset
# ---------------------------------------------------------------------------


def test_network_first_sample_is_warming_up(monkeypatch):
    _patch_common(monkeypatch)
    counters = {"eth0": SimpleNamespace(bytes_sent=1000, bytes_recv=2000)}
    monkeypatch.setattr(c.psutil, "net_io_counters", lambda pernic=True: counters)
    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: (_ for _ in ()).throw(FileNotFoundError()))

    col = _make_collector()
    snap = col.collect()

    assert snap.components.network.status == s.ComponentState.WARMING_UP
    assert snap.network[0].rx_bytes_per_sec is None
    assert snap.network[0].tx_bytes_per_sec is None


def test_network_second_sample_computes_rate(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: (_ for _ in ()).throw(FileNotFoundError()))

    times = iter([100.0, 105.0])
    monkeypatch.setattr(c.time, "monotonic", lambda: next(times))

    counters_seq = iter(
        [
            {"eth0": SimpleNamespace(bytes_sent=1000, bytes_recv=2000)},
            {"eth0": SimpleNamespace(bytes_sent=1500, bytes_recv=7000)},
        ]
    )
    monkeypatch.setattr(c.psutil, "net_io_counters", lambda pernic=True: next(counters_seq))

    col = _make_collector()
    col.collect()
    snap = col.collect()

    assert snap.components.network.status == s.ComponentState.OK
    net = snap.network[0]
    assert net.rx_bytes_per_sec == pytest.approx((7000 - 2000) / 5.0)
    assert net.tx_bytes_per_sec == pytest.approx((1500 - 1000) / 5.0)


def test_network_counter_reset_is_null_not_zero(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: (_ for _ in ()).throw(FileNotFoundError()))

    times = iter([100.0, 105.0])
    monkeypatch.setattr(c.time, "monotonic", lambda: next(times))

    counters_seq = iter(
        [
            {"eth0": SimpleNamespace(bytes_sent=1000, bytes_recv=2000)},
            # Interface reset: counters went backwards.
            {"eth0": SimpleNamespace(bytes_sent=10, bytes_recv=20)},
        ]
    )
    monkeypatch.setattr(c.psutil, "net_io_counters", lambda pernic=True: next(counters_seq))

    col = _make_collector()
    col.collect()
    snap = col.collect()

    assert snap.network[0].rx_bytes_per_sec is None
    assert snap.network[0].tx_bytes_per_sec is None
    assert snap.components.network.status in (s.ComponentState.WARMING_UP, s.ComponentState.DEGRADED)


def test_network_excludes_loopback_by_default(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: (_ for _ in ()).throw(FileNotFoundError()))
    counters = {
        "lo": SimpleNamespace(bytes_sent=1, bytes_recv=1),
        "eth0": SimpleNamespace(bytes_sent=1, bytes_recv=1),
    }
    monkeypatch.setattr(c.psutil, "net_io_counters", lambda pernic=True: counters)

    col = _make_collector()
    snap = col.collect()

    names = {n.interface for n in snap.network}
    assert names == {"eth0"}


# ---------------------------------------------------------------------------
# Process top-N filtering
# ---------------------------------------------------------------------------


def test_process_top_n_filtering_and_truncation(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: (_ for _ in ()).throw(FileNotFoundError()))

    total = s.MAX_PROCESSES + 5
    pids = list(range(total))
    fakes = {pid: FakeProcess(pid, f"proc{pid}", cpu_percent_value=float(pid), memory_percent_value=1.0, memory_rss=1024) for pid in pids}

    monkeypatch.setattr(c.psutil, "pids", lambda: pids)
    monkeypatch.setattr(c.psutil, "Process", lambda pid: fakes[pid])

    col = _make_collector()
    col.collect()  # prime: every process is "new" on first cycle
    snap = col.collect()  # second cycle: cpu_percent deltas are real

    assert len(snap.processes) == s.MAX_PROCESSES
    assert snap.truncated.processes is True
    # Highest pid has the highest fake cpu_percent_value -> must be first.
    assert snap.processes[0].cpu_percent == float(total - 1)
    assert all(snap.processes[i].cpu_percent >= snap.processes[i + 1].cpu_percent for i in range(len(snap.processes) - 1))
    for proc in snap.processes:
        assert "/" not in proc.name  # sanity: still just a bare process name


def test_process_access_denied_is_skipped_not_fatal(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: (_ for _ in ()).throw(FileNotFoundError()))

    import psutil as real_psutil

    class DeniedProcess(FakeProcess):
        def name(self):
            raise real_psutil.AccessDenied(pid=self.pid)

    fakes = {1: FakeProcess(1, "ok-proc", 5.0, 1.0, 1024), 2: DeniedProcess(2, "denied", 5.0, 1.0, 1024)}
    monkeypatch.setattr(c.psutil, "pids", lambda: [1, 2])
    monkeypatch.setattr(c.psutil, "Process", lambda pid: fakes[pid])

    col = _make_collector()
    col.collect()
    snap = col.collect()

    names = {p.name for p in snap.processes}
    assert names == {"ok-proc"}


# ---------------------------------------------------------------------------
# Atomic snapshot write
# ---------------------------------------------------------------------------


def test_write_snapshot_is_atomic_and_valid(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: (_ for _ in ()).throw(FileNotFoundError()))

    snapshot_path = tmp_path / "run" / "snapshot.json"
    col = _make_collector()

    snapshot = c.write_snapshot(col, snapshot_path)

    assert snapshot_path.exists()
    # No leftover temp files after a successful write.
    leftovers = list(snapshot_path.parent.glob(".snapshot-*"))
    assert leftovers == []

    raw = snapshot_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    restored = s.HostMetricsSnapshot.model_validate(payload)
    assert restored == snapshot
    assert restored.schema_version == 1
    assert isinstance(restored.collected_at, datetime)

    mode = snapshot_path.stat().st_mode & 0o777
    assert mode == 0o640


def test_write_snapshot_overwrites_atomically_on_second_call(tmp_path, monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(c, "_run_command", lambda argv, timeout: (_ for _ in ()).throw(FileNotFoundError()))

    snapshot_path = tmp_path / "snapshot.json"
    col = _make_collector()

    c.write_snapshot(col, snapshot_path)
    first_inode = snapshot_path.stat().st_ino
    c.write_snapshot(col, snapshot_path)
    second_inode = snapshot_path.stat().st_ino

    # os.replace() on the same filesystem always changes the inode
    # (rename semantics) -- confirms we never edit the file in place.
    assert first_inode != second_inode
    payload = json.loads(snapshot_path.read_bytes())
    assert s.HostMetricsSnapshot.model_validate(payload).schema_version == 1
