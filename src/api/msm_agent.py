"""Proxy from meibook to this repo's agent API v1 for the MSM chat mode.

The browser never sees the MSM admin credential: this module logs in once,
caches the JWT, and re-logs in when the agent answers 401.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Mapping

import httpx

LOGIN_PATH = "/agent/login"


class MsmAgentUnavailable(RuntimeError):
    """The MSM agent is not configured or refused the admin login."""


@dataclass(frozen=True)
class MsmAgentConfig:
    base_url: str
    username: str
    password: str = field(repr=False)
    timeout_s: float


def load_config(env: Mapping[str, str] | None = None) -> MsmAgentConfig:
    """Read the agent connection settings; fail closed when any is missing."""
    source = os.environ if env is None else env
    missing = [
        name
        for name in ("MSM_AGENT_BASE", "MSM_AGENT_USER", "MSM_AGENT_PASS")
        if not source.get(name)
    ]
    if missing:
        raise MsmAgentUnavailable(
            "MSM agent is not configured: missing " + ", ".join(missing)
        )
    try:
        timeout_s = float(source.get("MSM_AGENT_TIMEOUT_S") or 90)
    except ValueError:
        timeout_s = 90.0
    return MsmAgentConfig(
        base_url=source["MSM_AGENT_BASE"].rstrip("/"),
        username=source["MSM_AGENT_USER"],
        password=source["MSM_AGENT_PASS"],
        timeout_s=timeout_s,
    )


class MsmTokenCache:
    """One admin JWT shared by every request of this process."""

    def __init__(
        self,
        config: MsmAgentConfig,
        client: httpx.AsyncClient,
        login_path: str = LOGIN_PATH,
    ) -> None:
        self._config = config
        self._client = client
        self._login_path = login_path
        self._token = ""
        self._lock = asyncio.Lock()

    def invalidate(self) -> None:
        self._token = ""

    async def token(self) -> str:
        if self._token:
            return self._token
        async with self._lock:
            if self._token:
                return self._token
            token = await self._login()
            self._token = token
            return token

    async def _login(self) -> str:
        try:
            response = await self._client.post(
                self._config.base_url + self._login_path,
                json={
                    "username": self._config.username,
                    "password": self._config.password,
                },
                timeout=self._config.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise MsmAgentUnavailable("MSM agent login failed") from exc
        if response.status_code != 200:
            raise MsmAgentUnavailable(
                f"MSM agent login failed: HTTP {response.status_code}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise MsmAgentUnavailable("MSM agent login returned invalid JSON") from exc
        token = (body or {}).get("token") or ""
        if not token:
            raise MsmAgentUnavailable("MSM agent login returned no token")
        return token


def to_meibook_event(event: dict) -> dict | None:
    """Translate one agent SSE event into the vocabulary meibook renders."""
    kind = event.get("type")
    if kind == "answer":
        return {"type": "token", "content": event["text"]}
    if kind == "done":
        return {"type": "done", "result": event.get("result", {})}
    if kind == "error":
        return {
            "type": "error",
            "code": event.get("code", "INTERNAL"),
            "message": event.get("message", ""),
        }
    if event.get("label"):
        return {"type": "meta", "label": event["label"]}
    return None
