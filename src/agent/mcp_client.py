"""
src/agent/mcp_client.py
-----------------------
Load filesystem and git MCP tools for the Coding Agent.
"""

import asyncio
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Coroutine, List, TypeVar

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _mcp_tool_load_timeout_seconds() -> float:
    """Return the bounded per-server MCP discovery timeout."""
    try:
        timeout_seconds = float(
            os.getenv("MCP_TOOL_LOAD_TIMEOUT_SECONDS", "8")
        )
    except ValueError:
        timeout_seconds = 8.0
    return max(1.0, timeout_seconds)


def _resolve_command(
    candidates: list[str | None],
    env_path: str,
) -> str | None:
    """Resolve the first executable command from the provided candidates."""
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate, path=env_path)
        if resolved:
            return resolved
        candidate_path = Path(candidate)
        if (
            candidate_path.is_absolute()
            and candidate_path.is_file()
            and os.access(candidate_path, os.X_OK)
        ):
            return str(candidate_path)
    return None


try:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    HAS_MCP_ADAPTER = True
except ImportError:
    HAS_MCP_ADAPTER = False
    logger.warning(
        "langchain-mcp-adapters is unavailable; using local tools."
    )


def _run_async(coroutine: Coroutine[None, None, T]) -> T:
    """Run MCP discovery safely, including when imported from an active loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


async def _load_mcp_tools(
    workspace_dir: Path,
    repository_dir: Path,
) -> List[BaseTool]:
    env = dict(os.environ)
    runtime_bin = str(Path(sys.executable).parent)
    env["PATH"] = os.pathsep.join(
        filter(
            None,
            [runtime_bin, str(Path.home() / ".local" / "bin"), env.get("PATH")],
        )
    )

    npx_path = _resolve_command(
        [str(Path(runtime_bin) / "npx"), "npx", "/usr/bin/npx"],
        env["PATH"],
    )
    uvx_path = _resolve_command(
        ["uvx", str(Path.home() / ".local" / "bin" / "uvx")],
        env["PATH"],
    )
    node_path = _resolve_command(["node"], env["PATH"])

    connections = []
    if npx_path:
        connections.append(
            (
                "filesystem",
                {
                    "transport": "stdio",
                    "command": npx_path,
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(workspace_dir),
                    ],
                    "env": env,
                },
            )
        )
    else:
        logger.warning(
            "Skipping filesystem MCP tools because npx is unavailable."
        )

    if uvx_path:
        connections.append(
            (
                "git",
                {
                    "transport": "stdio",
                    "command": uvx_path,
                    "args": [
                        "mcp-server-git",
                        "--repository",
                        str(repository_dir),
                    ],
                    "env": env,
                },
            )
        )
    else:
        logger.warning("Skipping git MCP tools because uvx is unavailable.")

    if node_path:
        connections.append(
            (
                "google-calendar",
                {
                    "transport": "stdio",
                    "command": node_path,
                    "args": [
                        str(Path(__file__).parent / "run_calendar_mcp.js"),
                    ],
                    "env": env,
                },
            )
        )
    else:
        logger.warning(
            "Skipping Google Calendar MCP tools because node is unavailable."
        )

    tools: List[BaseTool] = []
    timeout_seconds = _mcp_tool_load_timeout_seconds()
    for server_name, connection in connections:
        try:
            client = MultiServerMCPClient({server_name: connection})
            server_tools = await asyncio.wait_for(
                client.get_tools(), timeout=timeout_seconds
            )
            tools.extend(server_tools)
            logger.info(
                "Loaded %s tools from the %s MCP server.",
                len(server_tools),
                server_name,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Timed out loading %s MCP tools after %.1f seconds.",
                server_name,
                timeout_seconds,
            )
        except Exception as exc:
            logger.error("Unable to load %s MCP tools: %s", server_name, exc)
    return tools


@lru_cache(maxsize=1)
def get_mcp_tools() -> List[BaseTool]:
    """Return cached MCP tools, with workspace-confined local fallbacks."""
    workspace_dir = Path(
        os.getenv("WORKSPACE_DIR", "/home/jkl0909/Code/llm")
    ).expanduser().resolve()
    repository_dir = Path(
        os.getenv("AGENT_REPOSITORY_DIR", Path.cwd())
    ).expanduser().resolve()

    if HAS_MCP_ADAPTER:
        tools = _run_async(_load_mcp_tools(workspace_dir, repository_dir))
        if tools:
            return tools

    logger.warning("No MCP tools loaded; using workspace-confined local tools.")
    return _get_fallback_local_tools(workspace_dir)


def _get_fallback_local_tools(workspace_dir: Path) -> List[BaseTool]:
    """Provide minimal file tools while enforcing the configured workspace."""
    from langchain_core.tools import tool

    def resolve_path(raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = workspace_dir / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(workspace_dir)
        except ValueError as exc:
            raise ValueError(
                f"Path must stay inside workspace: {workspace_dir}"
            ) from exc
        return resolved

    @tool
    def read_file(file_path: str) -> str:
        """Read a UTF-8 text file inside the configured workspace."""
        try:
            return resolve_path(file_path).read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading file: {exc}"

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Write a UTF-8 text file inside the configured workspace."""
        try:
            target = resolve_path(file_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote to {target}"
        except Exception as exc:
            return f"Error writing file: {exc}"

    @tool
    def list_dir(directory: str = ".") -> str:
        """List a directory inside the configured workspace."""
        try:
            names = sorted(item.name for item in resolve_path(directory).iterdir())
            return "\n".join(names)
        except Exception as exc:
            return f"Error listing directory: {exc}"

    return [read_file, write_file, list_dir]
