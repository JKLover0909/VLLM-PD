"""CCTVAI host metrics read-only API.

Serves the latest snapshot written by ``collector.py`` over HTTPS as a
single ``GET /metrics`` endpoint, protected by a constant-time-compared
bearer token loaded from a file. This process never touches Docker or
GPU tooling itself and is not a member of the ``docker`` group -- it only
reads a JSON file the collector wrote.

Fails closed:
    * Startup aborts if required config (token file, TLS cert/key when run
      as the TLS entrypoint) is missing -- this process never serves with
      an implicit/default token.
    * A missing, unreadable, oversized, malformed, or stale snapshot
      returns a clear HTTP error; the process never fabricates data.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import ValidationError

from tools.host_metrics.schemas import (
    DEFAULT_MAX_AGE_SECONDS,
    HostMetricsSnapshot,
    MAX_SNAPSHOT_BYTES,
)

DEFAULT_SNAPSHOT_PATH = "/run/cctvai-metrics/snapshot.json"

# A snapshot timestamped further in the future than this (clock skew, or a
# corrupted/forged file) is rejected rather than trusted.
MAX_FUTURE_SKEW_SECONDS = 5.0


@dataclass(frozen=True)
class HostMetricsServerConfig:
    token: str
    snapshot_path: Path = Path(DEFAULT_SNAPSHOT_PATH)
    max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS
    max_snapshot_bytes: int = MAX_SNAPSHOT_BYTES


class ConfigError(RuntimeError):
    """Raised when required server configuration is missing/invalid."""


def load_config_from_env(env: Optional[Mapping[str, str]] = None) -> HostMetricsServerConfig:
    """Build config from environment; raises ConfigError if incomplete.

    This is deliberately strict: there is no default token and no
    "auth disabled" mode. A caller (systemd unit, tests) must supply a
    complete, valid environment.
    """

    src = env if env is not None else os.environ

    token_file = src.get("HOST_METRICS_TOKEN_FILE")
    if not token_file:
        raise ConfigError("HOST_METRICS_TOKEN_FILE is required and was not set")

    token_path = Path(token_file)
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigError(f"Cannot read HOST_METRICS_TOKEN_FILE: {token_path}") from exc
    if not token:
        raise ConfigError(f"HOST_METRICS_TOKEN_FILE is empty: {token_path}")

    snapshot_path = Path(src.get("HOST_METRICS_SNAPSHOT_PATH", DEFAULT_SNAPSHOT_PATH))

    max_age_raw = src.get("HOST_METRICS_MAX_AGE_SECONDS", str(DEFAULT_MAX_AGE_SECONDS))
    try:
        max_age = float(max_age_raw)
    except ValueError as exc:
        raise ConfigError("HOST_METRICS_MAX_AGE_SECONDS must be a number") from exc
    if not (0 < max_age <= 3600):
        raise ConfigError("HOST_METRICS_MAX_AGE_SECONDS must be between 0 and 3600")

    return HostMetricsServerConfig(
        token=token,
        snapshot_path=snapshot_path,
        max_age_seconds=max_age,
    )


def _check_bearer(authorization: Optional[str], expected_token: str) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    provided = authorization[len("Bearer "):]
    # Fail closed on empty token: compare_digest still runs (no early
    # return) so this branch does not create a timing oracle either.
    if not secrets.compare_digest(provided, expected_token):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _read_snapshot(config: HostMetricsServerConfig) -> HostMetricsSnapshot:
    try:
        raw = config.snapshot_path.read_bytes()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Snapshot not available yet")
    except OSError:
        raise HTTPException(status_code=503, detail="Snapshot is unreadable")

    if len(raw) > config.max_snapshot_bytes:
        raise HTTPException(status_code=502, detail="Snapshot exceeds size limit")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=502, detail="Snapshot is not valid JSON")

    try:
        snapshot = HostMetricsSnapshot.model_validate(payload)
    except ValidationError:
        raise HTTPException(status_code=502, detail="Snapshot failed schema validation")

    now = datetime.now(timezone.utc)
    age_seconds = (now - snapshot.collected_at).total_seconds()
    if age_seconds < -MAX_FUTURE_SKEW_SECONDS:
        raise HTTPException(status_code=502, detail="Snapshot timestamp is in the future")
    if age_seconds > config.max_age_seconds:
        raise HTTPException(status_code=503, detail="Snapshot is stale")

    return snapshot


def create_app(config: HostMetricsServerConfig) -> FastAPI:
    """Build the FastAPI app for a given, already-validated config.

    Kept separate from environment/TLS wiring so tests can construct an
    app against a temp snapshot file without touching real certs/sockets.
    """

    app = FastAPI(title="CCTVAI Host Metrics", version=str(1))

    @app.get("/metrics", response_model=HostMetricsSnapshot)
    async def get_metrics(authorization: Optional[str] = Header(default=None)):
        _check_bearer(authorization, config.token)
        return _read_snapshot(config)

    return app


def main() -> None:  # pragma: no cover - process entrypoint
    config = load_config_from_env()

    certfile = os.environ.get("HOST_METRICS_TLS_CERTFILE")
    keyfile = os.environ.get("HOST_METRICS_TLS_KEYFILE")
    if not certfile or not keyfile:
        raise ConfigError(
            "HOST_METRICS_TLS_CERTFILE and HOST_METRICS_TLS_KEYFILE are required"
        )
    if not Path(certfile).is_file() or not Path(keyfile).is_file():
        raise ConfigError("TLS cert/key file configured but not found on disk")

    host = os.environ.get("HOST_METRICS_BIND_HOST", "0.0.0.0")
    port = int(os.environ.get("HOST_METRICS_BIND_PORT", "8099"))

    app = create_app(config)

    import uvicorn

    uvicorn.run(app, host=host, port=port, ssl_certfile=certfile, ssl_keyfile=keyfile)


if __name__ == "__main__":  # pragma: no cover
    main()
