"""Async HTTPS client for the CCTVAI hardware-metrics service.

Talks to the independent ``tools/host_metrics`` service running on the
CCTVAI Ubuntu server (see ``Markdowns/CCTVAI_HARDWARE.md``). Read-only:
this module only ever issues a single ``GET /metrics`` request.

Design constraints (see the approved plan, section 2/3):
* Endpoint, token, and CA come only from environment/config on the
  Meibook side — never from user input, never from a chat message.
* Zero I/O at construction time and at ``from_env()`` time — not even
  reading the token/CA files. The first disk/network access happens
  lazily on the first ``get_snapshot()`` call (or ``refresh_health()`` in
  the service layer), so a token file that is briefly unmounted at
  container startup makes the client report unavailable rather than
  crashing at import.
* Redirects are disabled and environment proxies are ignored, so a
  compromised/misconfigured host cannot redirect this client elsewhere.
* Responses are bounded in size while streaming (never buffer an
  unbounded body), then validated against the shared contract in
  ``tools.host_metrics.schemas`` — the single source of truth for field
  names, types, and bounds, shared with the collector/server side so the
  two cannot drift apart. This module never redefines that contract.
* Any list truncated server-side is reported via the contract's own
  ``truncated`` flags — never silently described as complete.
* The snapshot's ``collected_at`` must be both fresh enough and not
  implausibly in the future (bounded clock-skew tolerance).
* A short TTL cache guarded by an ``asyncio.Lock`` collapses concurrent
  chat requests landing in the same window into a single upstream fetch.
* Nothing here ever logs the bearer token, the token/CA file contents, or
  the full request/response body — only exception class names and
  bounded, non-sensitive counters.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import ValidationError

from tools.host_metrics.schemas import (
    DEFAULT_CLIENT_CACHE_SECONDS,
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    DEFAULT_MAX_AGE_SECONDS,
    DEFAULT_MAX_CLOCK_SKEW_SECONDS,
    MAX_SNAPSHOT_BYTES,
    HostMetricsSnapshot,
)

log = logging.getLogger(__name__)


class CctvaiHardwareClientError(RuntimeError):
    """Base error for any client-side failure (network/auth/schema/freshness)."""


class CctvaiHardwareConfigError(CctvaiHardwareClientError):
    """Required config (endpoint/token/CA) is missing or unreadable."""


class CctvaiHardwareTransportError(CctvaiHardwareClientError):
    """Network/HTTP-level failure talking to the metrics endpoint."""


class CctvaiHardwareAuthError(CctvaiHardwareClientError):
    """The metrics endpoint rejected the bearer token (401/403)."""


class CctvaiHardwareSchemaError(CctvaiHardwareClientError):
    """The response failed structural, size, or contract validation."""


class CctvaiHardwareStaleError(CctvaiHardwareClientError):
    """The snapshot is older than the configured max age, or timestamped
    implausibly in the future."""


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class HardwareSnapshotResult:
    """A contract-validated snapshot plus client-side freshness bookkeeping."""

    snapshot: HostMetricsSnapshot
    age_seconds: float
    fetched_monotonic: float

    def current_age_seconds(self) -> float:
        """Age extrapolated to *now*, without a new network call."""
        return self.age_seconds + max(0.0, time.monotonic() - self.fetched_monotonic)

    @property
    def truncated_sections(self) -> tuple[str, ...]:
        flags = self.snapshot.truncated
        return tuple(
            name
            for name in ("gpus", "disks", "network", "processes", "containers")
            if getattr(flags, name)
        )

    @property
    def is_truncated(self) -> bool:
        return bool(self.truncated_sections)


@dataclass
class _CacheEntry:
    result: HardwareSnapshotResult | None
    error: CctvaiHardwareClientError | None
    fetched_monotonic: float


class CctvaiHardwareClient:
    """Read-only client for the fixed CCTVAI hardware-metrics HTTPS endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        token_path: str,
        ca_path: str | None = None,
        timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        cache_seconds: float = DEFAULT_CLIENT_CACHE_SECONDS,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        max_bytes: int = MAX_SNAPSHOT_BYTES,
        max_future_skew_seconds: float = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token_path = token_path
        self._ca_path = ca_path
        self._timeout_seconds = timeout_seconds
        self._cache_seconds = cache_seconds
        self._max_age_seconds = max_age_seconds
        self._max_bytes = min(max_bytes, MAX_SNAPSHOT_BYTES)
        self._max_future_skew_seconds = max_future_skew_seconds
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._cache_lock = asyncio.Lock()
        self._cache: _CacheEntry | None = None

    @classmethod
    def from_env(cls) -> "CctvaiHardwareClient | None":
        """Build from env; ``None`` when disabled or missing required config.

        Reads only environment variables — never touches the token/CA files
        and never opens a connection. Safe to call at import/startup time.
        """
        if not _env_bool("CCTVAI_HARDWARE_ENABLED", False):
            return None

        base_url = _env_str("CCTVAI_HARDWARE_URL")
        token_path = _env_str("CCTVAI_HARDWARE_TOKEN_PATH")
        if not base_url or not token_path:
            log.warning(
                "CCTVAI_HARDWARE_ENABLED is true but URL/token path is missing; disabling."
            )
            return None
        if not base_url.lower().startswith("https://"):
            log.warning("CCTVAI_HARDWARE_URL must be https://; disabling.")
            return None

        ca_path = _env_str("CCTVAI_HARDWARE_CA_PATH") or None

        return cls(
            base_url=base_url,
            token_path=token_path,
            ca_path=ca_path,
            timeout_seconds=_env_float(
                "CCTVAI_HARDWARE_TIMEOUT_SECONDS",
                DEFAULT_HTTP_TIMEOUT_SECONDS,
                minimum=1.0,
                maximum=15.0,
            ),
            cache_seconds=_env_float(
                "CCTVAI_HARDWARE_CACHE_SECONDS",
                DEFAULT_CLIENT_CACHE_SECONDS,
                minimum=0.0,
                maximum=30.0,
            ),
            max_age_seconds=_env_float(
                "CCTVAI_HARDWARE_MAX_AGE_SECONDS",
                DEFAULT_MAX_AGE_SECONDS,
                minimum=5.0,
                maximum=300.0,
            ),
            max_bytes=_env_int(
                "CCTVAI_HARDWARE_MAX_BYTES",
                MAX_SNAPSHOT_BYTES,
                minimum=1024,
                maximum=MAX_SNAPSHOT_BYTES,
            ),
            max_future_skew_seconds=_env_float(
                "CCTVAI_HARDWARE_MAX_FUTURE_SKEW_SECONDS",
                DEFAULT_MAX_CLOCK_SKEW_SECONDS,
                minimum=1.0,
                maximum=120.0,
            ),
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _load_token(self) -> str:
        """Read the bearer token from disk. Lazy — never called from
        ``from_env()``/``__init__``. Never logs the token or its path's
        content, only the exception class name on failure."""
        try:
            content = Path(self._token_path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise CctvaiHardwareConfigError(
                f"Cannot read hardware token file: {exc.__class__.__name__}"
            ) from exc
        if not content:
            raise CctvaiHardwareConfigError("Hardware token file is empty.")
        return content

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            try:
                self._client = httpx.AsyncClient(
                    verify=self._ca_path or True,
                    timeout=self._timeout_seconds,
                    follow_redirects=False,
                    trust_env=False,  # ignore HTTP(S)_PROXY / NO_PROXY env vars
                    transport=self._transport,
                )
            except OSError as exc:
                raise CctvaiHardwareConfigError(
                    f"Cannot initialize TLS client (CA file?): {exc.__class__.__name__}"
                ) from exc
        return self._client

    async def get_snapshot(
        self, *, force_refresh: bool = False
    ) -> HardwareSnapshotResult:
        """Return a fresh (possibly cached) contract-validated snapshot.

        Raises a ``CctvaiHardwareClientError`` subclass on any failure —
        callers must never fall back to guessing metrics values.
        """
        async with self._cache_lock:
            now = time.monotonic()
            cached = self._cache
            if not force_refresh and cached is not None:
                fresh_enough = (now - cached.fetched_monotonic) < self._cache_seconds
                still_within_max_age = (
                    cached.result is None
                    or cached.result.current_age_seconds() <= self._max_age_seconds
                )
                if fresh_enough and still_within_max_age:
                    if cached.error is not None:
                        raise cached.error
                    assert cached.result is not None
                    return cached.result

            try:
                result = await self._fetch_and_validate()
            except CctvaiHardwareClientError as exc:
                self._cache = _CacheEntry(result=None, error=exc, fetched_monotonic=now)
                raise
            self._cache = _CacheEntry(result=result, error=None, fetched_monotonic=now)
            return result

    async def _fetch_and_validate(self) -> HardwareSnapshotResult:
        token = self._load_token()
        client = self._ensure_client()
        headers = {"Authorization": f"Bearer {token}"}
        url = f"{self._base_url}/metrics"
        try:
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code in (401, 403):
                    raise CctvaiHardwareAuthError(
                        "Hardware metrics endpoint rejected credentials."
                    )
                if response.status_code != 200:
                    raise CctvaiHardwareTransportError(
                        f"Hardware metrics endpoint returned HTTP {response.status_code}."
                    )
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self._max_bytes:
                        raise CctvaiHardwareSchemaError(
                            f"Snapshot payload exceeds {self._max_bytes} bytes."
                        )
                    chunks.append(chunk)
                body = b"".join(chunks)
        except httpx.TimeoutException as exc:
            raise CctvaiHardwareTransportError(
                "Hardware metrics request timed out."
            ) from exc
        except httpx.HTTPError as exc:
            raise CctvaiHardwareTransportError(
                f"Hardware metrics request failed: {exc.__class__.__name__}"
            ) from exc

        return self._validate_snapshot(body)

    def _validate_snapshot(self, body: bytes) -> HardwareSnapshotResult:
        try:
            snapshot = HostMetricsSnapshot.model_validate_json(body)
        except ValidationError as exc:
            log.warning("CCTVAI hardware snapshot failed schema validation: %s", exc)
            raise CctvaiHardwareSchemaError(
                "Snapshot failed schema validation."
            ) from exc

        age_seconds = self._collected_at_age_seconds(snapshot.collected_at)
        return HardwareSnapshotResult(
            snapshot=snapshot,
            age_seconds=age_seconds,
            fetched_monotonic=time.monotonic(),
        )

    def _collected_at_age_seconds(self, collected_at: datetime) -> float:
        now = datetime.now(timezone.utc)
        age = (now - collected_at).total_seconds()
        if age < -self._max_future_skew_seconds:
            raise CctvaiHardwareStaleError(
                "Snapshot collected_at is in the future beyond allowed clock skew."
            )
        if age > self._max_age_seconds:
            raise CctvaiHardwareStaleError(
                f"Snapshot is stale: age={age:.1f}s exceeds max_age={self._max_age_seconds:.1f}s."
            )
        return max(age, 0.0)
