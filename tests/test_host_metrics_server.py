"""Tests for tools/host_metrics/server.py.

No real TLS/network/system operations: config comes from an in-memory env
mapping and a temp snapshot file; the app is exercised via FastAPI's
TestClient over the ASGI transport (no real socket).
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tools.host_metrics import schemas as s
from tools.host_metrics import server as srv


TOKEN = "unit-test-token-do-not-use-in-prod"


def _components_ok() -> s.Components:
    ok = s.ComponentStatus(status=s.ComponentState.OK)
    return s.Components(
        cpu=ok, memory=ok, gpu=ok, disk=ok, network=ok, uptime=ok, processes=ok, docker=ok
    )


def _write_snapshot(path, *, age_seconds: float = 0.0, **overrides):
    collected_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    kwargs = dict(collected_at=collected_at, components=_components_ok())
    kwargs.update(overrides)
    snap = s.HostMetricsSnapshot(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snap.model_dump_json())
    return snap


def _make_config(tmp_path, **overrides) -> srv.HostMetricsServerConfig:
    kwargs = dict(
        token=TOKEN,
        snapshot_path=tmp_path / "snapshot.json",
        max_age_seconds=30.0,
    )
    kwargs.update(overrides)
    return srv.HostMetricsServerConfig(**kwargs)


def _client(config: srv.HostMetricsServerConfig) -> TestClient:
    app = srv.create_app(config)
    return TestClient(app)


# ---------------------------------------------------------------------------
# load_config_from_env
# ---------------------------------------------------------------------------


def test_load_config_from_env_missing_token_file_var():
    with pytest.raises(srv.ConfigError, match="HOST_METRICS_TOKEN_FILE"):
        srv.load_config_from_env(env={})


def test_load_config_from_env_token_file_does_not_exist(tmp_path):
    missing = tmp_path / "nope.token"
    with pytest.raises(srv.ConfigError):
        srv.load_config_from_env(env={"HOST_METRICS_TOKEN_FILE": str(missing)})


def test_load_config_from_env_empty_token_file(tmp_path):
    token_path = tmp_path / "token"
    token_path.write_text("   \n")
    with pytest.raises(srv.ConfigError, match="empty"):
        srv.load_config_from_env(env={"HOST_METRICS_TOKEN_FILE": str(token_path)})


def test_load_config_from_env_invalid_max_age(tmp_path):
    token_path = tmp_path / "token"
    token_path.write_text(TOKEN)
    with pytest.raises(srv.ConfigError):
        srv.load_config_from_env(
            env={
                "HOST_METRICS_TOKEN_FILE": str(token_path),
                "HOST_METRICS_MAX_AGE_SECONDS": "-5",
            }
        )
    with pytest.raises(srv.ConfigError):
        srv.load_config_from_env(
            env={
                "HOST_METRICS_TOKEN_FILE": str(token_path),
                "HOST_METRICS_MAX_AGE_SECONDS": "not-a-number",
            }
        )


def test_load_config_from_env_success(tmp_path):
    token_path = tmp_path / "token"
    token_path.write_text(f"  {TOKEN}  \n")
    snapshot_path = tmp_path / "run" / "snapshot.json"
    config = srv.load_config_from_env(
        env={
            "HOST_METRICS_TOKEN_FILE": str(token_path),
            "HOST_METRICS_SNAPSHOT_PATH": str(snapshot_path),
            "HOST_METRICS_MAX_AGE_SECONDS": "15",
        }
    )
    assert config.token == TOKEN  # trimmed
    assert config.snapshot_path == snapshot_path
    assert config.max_age_seconds == 15.0


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_missing_authorization_header_is_401(tmp_path):
    config = _make_config(tmp_path)
    _write_snapshot(config.snapshot_path)
    resp = _client(config).get("/metrics")
    assert resp.status_code == 401


def test_wrong_token_is_401(tmp_path):
    config = _make_config(tmp_path)
    _write_snapshot(config.snapshot_path)
    resp = _client(config).get("/metrics", headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 401


def test_malformed_authorization_header_is_401(tmp_path):
    config = _make_config(tmp_path)
    _write_snapshot(config.snapshot_path)
    resp = _client(config).get("/metrics", headers={"Authorization": TOKEN})  # no "Bearer " prefix
    assert resp.status_code == 401


def test_correct_token_returns_snapshot(tmp_path):
    config = _make_config(tmp_path)
    _write_snapshot(config.snapshot_path, cpu=s.CpuStats(utilization_percent=12.0))
    resp = _client(config).get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == 1
    assert body["cpu"]["utilization_percent"] == 12.0


# ---------------------------------------------------------------------------
# Snapshot freshness / availability / validity
# ---------------------------------------------------------------------------


def test_missing_snapshot_file_is_503(tmp_path):
    config = _make_config(tmp_path)  # never written
    resp = _client(config).get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 503


def test_stale_snapshot_is_503(tmp_path):
    config = _make_config(tmp_path, max_age_seconds=30.0)
    _write_snapshot(config.snapshot_path, age_seconds=45.0)
    resp = _client(config).get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 503
    assert "stale" in resp.json()["detail"].lower()


def test_fresh_snapshot_within_max_age_is_200(tmp_path):
    config = _make_config(tmp_path, max_age_seconds=30.0)
    _write_snapshot(config.snapshot_path, age_seconds=10.0)
    resp = _client(config).get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200


def test_future_timestamp_snapshot_is_502(tmp_path):
    config = _make_config(tmp_path)
    _write_snapshot(config.snapshot_path, age_seconds=-3600.0)  # 1 hour in the future
    resp = _client(config).get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 502


def test_malformed_json_snapshot_is_502(tmp_path):
    config = _make_config(tmp_path)
    config.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    config.snapshot_path.write_text("{not valid json")
    resp = _client(config).get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 502


def test_schema_invalid_snapshot_is_502(tmp_path):
    config = _make_config(tmp_path)
    config.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    # Valid JSON, but missing required "components" and wrong schema_version.
    config.snapshot_path.write_text(json.dumps({"schema_version": 1, "collected_at": "2026-01-01T00:00:00Z"}))
    resp = _client(config).get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 502


def test_oversized_snapshot_is_502(tmp_path):
    config = _make_config(tmp_path, max_snapshot_bytes=100)  # tiny cap for the test
    _write_snapshot(config.snapshot_path)
    assert config.snapshot_path.stat().st_size > 100
    resp = _client(config).get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 502


def test_unreadable_snapshot_directory_is_503(tmp_path):
    config = _make_config(tmp_path)
    # snapshot_path's parent directory does not exist at all -> read fails.
    resp = _client(config).get("/metrics", headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 503
