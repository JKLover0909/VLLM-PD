"""Workspace-confined read-only tools for Coding Agent evaluation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


class ReadOnlyToolError(ValueError):
    """Raised when a tool request violates a read-only boundary."""


class ReadOnlyToolbox:
    MAX_FILE_BYTES = 256 * 1024
    MAX_OUTPUT_CHARS = 24_000
    MAX_DIRECTORY_ENTRIES = 200
    MAX_SEARCH_RESULTS = 100

    SENSITIVE_NAMES = {
        ".env",
        ".env.local",
        ".env.production",
        "credentials",
        "credentials.json",
        "id_rsa",
        "id_ed25519",
        "known_hosts",
        "secrets.json",
    }
    BLOCKED_PARTS = {
        ".git",
        ".ssh",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
    }

    def __init__(self, workspace_dir: Path, repository_dir: Path | None = None):
        self.workspace_dir = workspace_dir.expanduser().resolve()
        self.repository_dir = (
            repository_dir.expanduser().resolve()
            if repository_dir is not None
            else self.workspace_dir
        )
        if not self.workspace_dir.is_dir():
            raise ReadOnlyToolError(
                f"Workspace does not exist: {self.workspace_dir}"
            )
        self._workspace_repo_path = self._relative_to_repository(
            self.workspace_dir
        )

    @staticmethod
    def tool_schemas() -> list[dict[str, Any]]:
        path_property = {
            "type": "string",
            "description": "Path relative to the configured workspace.",
        }
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": (
                        "List files and directories inside the read-only workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                **path_property,
                                "default": ".",
                            }
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a UTF-8 text file inside the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": path_property},
                        "required": ["file_path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_text",
                    "description": (
                        "Search for literal text in workspace files and return "
                        "path, line, column, and matching content."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "directory": {
                                **path_property,
                                "default": ".",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_file_metadata",
                    "description": (
                        "Get safe metadata for a file or directory in the workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": path_property},
                        "required": ["file_path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": (
                        "Show Git status only for the configured workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_diff",
                    "description": (
                        "Show the current Git diff only for the configured workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_log",
                    "description": (
                        "Show recent Git commits affecting the configured workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_count": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "default": 5,
                            }
                        },
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "list_directory": lambda: self.list_directory(
                arguments.get("directory") or "."
            ),
            "read_file": lambda: self.read_file(arguments["file_path"]),
            "search_text": lambda: self.search_text(
                arguments["query"], arguments.get("directory") or "."
            ),
            "get_file_metadata": lambda: self.get_file_metadata(
                arguments["file_path"]
            ),
            "git_status": self.git_status,
            "git_diff": self.git_diff,
            "git_log": lambda: self.git_log(arguments.get("max_count", 5)),
        }
        if tool_name not in handlers:
            return {
                "ok": False,
                "error": f"Tool is not available in read-only mode: {tool_name}",
            }
        try:
            return {"ok": True, "result": handlers[tool_name]()}
        except (KeyError, ReadOnlyToolError, OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "error": str(exc)}

    def list_directory(self, directory: str = ".") -> dict[str, Any]:
        target = self._resolve_path(directory)
        if not target.is_dir():
            raise ReadOnlyToolError(f"Not a directory: {directory}")

        entries = []
        for item in sorted(target.iterdir(), key=lambda value: value.name):
            if self._is_blocked_name(item.name):
                continue
            entries.append(
                {
                    "name": item.name,
                    "path": self._display_path(item),
                    "type": "directory" if item.is_dir() else "file",
                }
            )
            if len(entries) >= self.MAX_DIRECTORY_ENTRIES:
                break
        return {
            "directory": self._display_path(target),
            "entries": entries,
            "truncated": len(entries) >= self.MAX_DIRECTORY_ENTRIES,
        }

    def read_file(self, file_path: str) -> dict[str, Any]:
        target = self._resolve_path(file_path)
        if not target.is_file():
            raise ReadOnlyToolError(f"Not a file: {file_path}")
        size = target.stat().st_size
        if size > self.MAX_FILE_BYTES:
            raise ReadOnlyToolError(
                f"File exceeds {self.MAX_FILE_BYTES} bytes: {file_path}"
            )
        if b"\x00" in target.read_bytes()[:4096]:
            raise ReadOnlyToolError(f"Binary files are not readable: {file_path}")

        content = target.read_text(encoding="utf-8")
        truncated = len(content) > self.MAX_OUTPUT_CHARS
        return {
            "path": self._display_path(target),
            "size_bytes": size,
            "content": content[: self.MAX_OUTPUT_CHARS],
            "truncated": truncated,
            "read_only": True,
        }

    def search_text(self, query: str, directory: str = ".") -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ReadOnlyToolError("Search query must not be empty")
        target = self._resolve_path(directory)
        if not target.is_dir():
            raise ReadOnlyToolError(f"Not a directory: {directory}")

        rg_path = shutil.which("rg")
        if not rg_path:
            raise ReadOnlyToolError("ripgrep (rg) is not available")

        command = [
            rg_path,
            "--fixed-strings",
            "--line-number",
            "--column",
            "--no-heading",
            "--color",
            "never",
            "--glob",
            "!.git/**",
            "--glob",
            "!node_modules/**",
            "--glob",
            "!__pycache__/**",
            "--glob",
            "!.pytest_cache/**",
            "--",
            query,
            str(target),
        ]
        process = subprocess.run(
            command,
            cwd=self.workspace_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env={"PATH": os.environ.get("PATH", "")},
        )
        if process.returncode not in (0, 1):
            raise ReadOnlyToolError(process.stderr.strip() or "Search failed")

        matches = []
        for raw_line in process.stdout.splitlines():
            parts = raw_line.split(":", 3)
            if len(parts) != 4:
                continue
            raw_path, line, column, text = parts
            match_path = Path(raw_path).resolve()
            self._assert_allowed_path(match_path)
            matches.append(
                {
                    "path": self._display_path(match_path),
                    "line": int(line),
                    "column": int(column),
                    "text": text[:1000],
                }
            )
            if len(matches) >= self.MAX_SEARCH_RESULTS:
                break
        return {
            "query": query,
            "directory": self._display_path(target),
            "matches": matches,
            "truncated": len(matches) >= self.MAX_SEARCH_RESULTS,
        }

    def get_file_metadata(self, file_path: str) -> dict[str, Any]:
        target = self._resolve_path(file_path)
        stat = target.stat()
        return {
            "path": self._display_path(target),
            "type": "directory" if target.is_dir() else "file",
            "size_bytes": stat.st_size,
            "modified_timestamp": stat.st_mtime,
            "read_only": True,
        }

    def git_status(self) -> dict[str, Any]:
        output = self._run_git(
            ["status", "--short", "--untracked-files=all", "--", self._workspace_repo_path]
        )
        return {"output": output}

    def git_diff(self) -> dict[str, Any]:
        output = self._run_git(["diff", "--", self._workspace_repo_path])
        return {"output": output}

    def git_log(self, max_count: int = 5) -> dict[str, Any]:
        if not isinstance(max_count, int) or not 1 <= max_count <= 20:
            raise ReadOnlyToolError("max_count must be between 1 and 20")
        output = self._run_git(
            [
                "log",
                f"--max-count={max_count}",
                "--oneline",
                "--",
                self._workspace_repo_path,
            ]
        )
        return {"output": output}

    def _resolve_path(self, raw_path: str) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ReadOnlyToolError("Path must not be empty")
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_dir / candidate
        resolved = candidate.resolve()
        self._assert_allowed_path(resolved)
        if not resolved.exists():
            raise ReadOnlyToolError(f"Path does not exist: {raw_path}")
        return resolved

    def _assert_allowed_path(self, path: Path) -> None:
        try:
            relative = path.relative_to(self.workspace_dir)
        except ValueError as exc:
            raise ReadOnlyToolError(
                f"Path must stay inside workspace: {self.workspace_dir}"
            ) from exc
        for part in relative.parts:
            if self._is_blocked_name(part):
                raise ReadOnlyToolError(f"Sensitive or blocked path: {part}")

    def _is_blocked_name(self, name: str) -> bool:
        lower_name = name.lower()
        return (
            lower_name in self.SENSITIVE_NAMES
            or lower_name in self.BLOCKED_PARTS
            or lower_name.startswith(".env.")
            or lower_name.endswith((".pem", ".key", ".p12", ".pfx"))
        )

    def _display_path(self, path: Path) -> str:
        relative = path.relative_to(self.workspace_dir)
        return "." if not relative.parts else relative.as_posix()

    def _relative_to_repository(self, path: Path) -> str:
        try:
            return path.relative_to(self.repository_dir).as_posix()
        except ValueError as exc:
            raise ReadOnlyToolError(
                "Workspace must be inside repository for Git tools"
            ) from exc

    def _run_git(self, arguments: list[str]) -> str:
        process = subprocess.run(
            ["git", *arguments],
            cwd=self.repository_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if process.returncode != 0:
            raise ReadOnlyToolError(process.stderr.strip() or "Git command failed")
        return process.stdout[: self.MAX_OUTPUT_CHARS]


def tool_result_json(result: dict[str, Any]) -> str:
    """Serialize a tool result for an OpenAI-compatible tool message."""
    return json.dumps(result, ensure_ascii=False)
