"""Contract tests for tools/host_metrics/schemas.py.

No real system operations here -- pure Pydantic model construction and
validation. These pin the exact contract that both server.py (this
package) and the Meibook cctvai_hardware_client rely on.
"""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from tools.host_metrics import schemas as s


def _components_ok() -> s.Components:
    ok = s.ComponentStatus(status=s.ComponentState.OK)
    return s.Components(
        cpu=ok, memory=ok, gpu=ok, disk=ok, network=ok, uptime=ok, processes=ok, docker=ok
    )


def _minimal_snapshot(**overrides) -> s.HostMetricsSnapshot:
    kwargs = dict(
        collected_at=datetime.now(timezone.utc),
        components=_components_ok(),
    )
    kwargs.update(overrides)
    return s.HostMetricsSnapshot(**kwargs)


def test_schema_version_constant_is_one():
    assert s.SCHEMA_VERSION == 1


def test_minimal_snapshot_has_expected_defaults():
    snap = _minimal_snapshot()
    assert snap.schema_version == 1
    assert snap.cpu is None
    assert snap.memory is None
    assert snap.gpus == []
    assert snap.disks == []
    assert snap.network == []
    assert snap.processes == []
    assert snap.containers == []
    assert snap.uptime_seconds is None
    assert snap.sample_interval_seconds is None
    assert snap.truncated == s.TruncationFlags()
    assert snap.truncated.gpus is False


def test_extra_fields_are_ignored_not_rejected():
    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "components": _components_ok().model_dump(),
        "totally_unknown_future_field": {"nested": [1, 2, 3]},
    }
    snap = s.HostMetricsSnapshot.model_validate(payload)
    assert not hasattr(snap, "totally_unknown_future_field")

    cpu_payload = {"utilization_percent": 12.5, "from_the_future": True}
    cpu = s.CpuStats.model_validate(cpu_payload)
    assert cpu.utilization_percent == 12.5


def test_schema_version_literal_rejects_other_values():
    with pytest.raises(ValidationError):
        s.HostMetricsSnapshot(
            schema_version=2,
            collected_at=datetime.now(timezone.utc),
            components=_components_ok(),
        )


def test_components_is_required():
    with pytest.raises(ValidationError):
        s.HostMetricsSnapshot(collected_at=datetime.now(timezone.utc))


def test_collected_at_requires_timezone_aware_datetime():
    with pytest.raises(ValidationError):
        s.HostMetricsSnapshot(
            collected_at=datetime.now(),  # naive
            components=_components_ok(),
        )


def test_collected_at_is_normalized_to_utc():
    tz_plus7 = timezone(timedelta(hours=7))
    local = datetime.now(tz_plus7)
    snap = _minimal_snapshot(collected_at=local)
    assert snap.collected_at.tzinfo == timezone.utc
    assert snap.collected_at == local.astimezone(timezone.utc)


@pytest.mark.parametrize(
    "field_name,limit",
    [
        ("gpus", s.MAX_GPUS),
        ("disks", s.MAX_DISKS),
        ("network", s.MAX_NETWORK_INTERFACES),
        ("processes", s.MAX_PROCESSES),
        ("containers", s.MAX_CONTAINERS),
    ],
)
def test_bounded_arrays_reject_over_limit(field_name, limit):
    factories = {
        "gpus": lambda i: s.GpuStats(index=i % 64),
        "disks": lambda i: s.DiskStats(mountpoint=f"/mnt/{i}"),
        "network": lambda i: s.NetworkStats(interface=f"eth{i}"),
        "processes": lambda i: s.ProcessStats(name=f"proc{i}"),
        "containers": lambda i: s.ContainerStats(name=f"c{i}", state="running"),
    }
    make = factories[field_name]
    too_many = [make(i) for i in range(limit + 1)]
    with pytest.raises(ValidationError):
        _minimal_snapshot(**{field_name: too_many})

    exactly_at_limit = [make(i) for i in range(limit)]
    snap = _minimal_snapshot(**{field_name: exactly_at_limit})
    assert len(getattr(snap, field_name)) == limit


def test_percent_fields_are_bounded_0_to_100():
    with pytest.raises(ValidationError):
        s.MemoryStats(percent=150.0)
    with pytest.raises(ValidationError):
        s.MemoryStats(percent=-1.0)
    ok = s.MemoryStats(percent=99.9)
    assert ok.percent == 99.9


def test_byte_fields_reject_negative():
    with pytest.raises(ValidationError):
        s.MemoryStats(total_bytes=-1)


def test_null_means_unmeasured_not_zero():
    cpu = s.CpuStats()
    assert cpu.utilization_percent is None
    memory = s.MemoryStats()
    assert memory.total_bytes is None
    net = s.NetworkStats(interface="eth0")
    assert net.rx_bytes_per_sec is None
    assert net.tx_bytes_per_sec is None


def test_component_state_enum_values():
    assert {m.value for m in s.ComponentState} == {
        "ok",
        "warming_up",
        "degraded",
        "unavailable",
        "error",
    }


def test_component_reason_is_bounded_length():
    with pytest.raises(ValidationError):
        s.ComponentStatus(status=s.ComponentState.ERROR, reason="x" * (s.MAX_REASON_LENGTH + 1))
    ok = s.ComponentStatus(status=s.ComponentState.ERROR, reason="x" * s.MAX_REASON_LENGTH)
    assert len(ok.reason) == s.MAX_REASON_LENGTH


def test_process_stats_has_no_cmdline_or_env_fields():
    fields = set(s.ProcessStats.model_fields.keys())
    assert fields == {"pid", "name", "cpu_percent", "memory_percent", "memory_bytes"}
    assert "cmdline" not in fields
    assert "env" not in fields
    assert "environ" not in fields


def test_container_stats_has_no_env_log_mount_fields():
    fields = set(s.ContainerStats.model_fields.keys())
    assert fields == {"name", "state", "health", "restart_count"}
    assert "env" not in fields
    assert "mounts" not in fields
    assert "logs" not in fields


def test_round_trip_json():
    snap = _minimal_snapshot(
        cpu=s.CpuStats(utilization_percent=42.0),
        gpus=[s.GpuStats(index=0, name="RTX 3090", utilization_percent=10.0)],
        truncated=s.TruncationFlags(processes=True),
    )
    raw = snap.model_dump_json()
    restored = s.HostMetricsSnapshot.model_validate_json(raw)
    assert restored == snap
    assert restored.truncated.processes is True


def test_components_rejects_unknown_keys_as_dict_not_allowed():
    # Components is a fixed-field model, not an arbitrary dict -- passing
    # a plain dict with unexpected keys and missing required ones fails.
    with pytest.raises(ValidationError):
        s.Components.model_validate({"cpu": {"status": "ok"}})
