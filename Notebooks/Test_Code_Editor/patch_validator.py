"""Validate model-proposed unified diffs without changing the real workspace."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from read_only_tools import ReadOnlyToolError, ReadOnlyToolbox


class PatchValidationError(ValueError):
    """Raised when a proposed patch is unsafe or cannot be applied."""


class PatchValidator:
    MAX_PATCH_CHARS = 30_000
    MAX_CHANGED_FILES = 1
    DIFF_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")
    OLD_HEADER = re.compile(r"^--- a/(.+)$")
    NEW_HEADER = re.compile(r"^\+\+\+ b/(.+)$")

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir.expanduser().resolve()
        if not self.workspace_dir.is_dir():
            raise PatchValidationError(
                f"Workspace does not exist: {self.workspace_dir}"
            )
        self.path_guard = ReadOnlyToolbox(self.workspace_dir, self.workspace_dir)

    @staticmethod
    def tool_schema() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "propose_patch",
                "description": (
                    "Propose a unified diff for one existing text file. "
                    "The patch is validated and tested only in a temporary copy; "
                    "it is never applied to the real workspace."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative file path.",
                        },
                        "patch": {
                            "type": "string",
                            "description": (
                                "Complete unified diff with diff --git, ---, +++, "
                                "and @@ headers."
                            ),
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Short explanation of the proposed change.",
                        },
                    },
                    "required": ["path", "patch", "explanation"],
                    "additionalProperties": False,
                },
            },
        }

    def validate(
        self,
        path: str,
        patch: str,
        explanation: str,
        run_tests: bool = True,
    ) -> dict[str, Any]:
        normalized_path = self._validate_declared_path(path)
        normalized_patch = self._normalize_patch(patch)
        changed_paths = self._extract_changed_paths(normalized_patch)

        if changed_paths != [normalized_path]:
            raise PatchValidationError(
                "Patch paths must exactly match the declared path: "
                f"declared={normalized_path}, patch={changed_paths}"
            )
        if not isinstance(explanation, str) or not explanation.strip():
            raise PatchValidationError("Explanation must not be empty")

        snapshot_before = self.workspace_snapshot()
        with tempfile.TemporaryDirectory(prefix="vllm-pd-patch-") as temp_dir:
            temp_workspace = Path(temp_dir) / "workspace"
            shutil.copytree(
                self.workspace_dir,
                temp_workspace,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    ".pytest_cache",
                    ".git",
                ),
            )
            check_process = self._run_git_apply(
                temp_workspace, normalized_patch, check_only=True
            )
            if check_process.returncode != 0:
                raise PatchValidationError(
                    check_process.stderr.strip()
                    or check_process.stdout.strip()
                    or "git apply --check failed"
                )

            apply_process = self._run_git_apply(
                temp_workspace, normalized_patch, check_only=False
            )
            if apply_process.returncode != 0:
                raise PatchValidationError(
                    apply_process.stderr.strip()
                    or apply_process.stdout.strip()
                    or "git apply failed in temporary workspace"
                )

            changed_in_copy = self._changed_files_between(
                self.workspace_dir, temp_workspace
            )
            if changed_in_copy != [normalized_path]:
                raise PatchValidationError(
                    f"Patch changed unexpected files: {changed_in_copy}"
                )

            patched_file = temp_workspace / normalized_path
            patched_content = patched_file.read_text(encoding="utf-8")
            test_result = (
                self._run_tests(temp_workspace)
                if run_tests
                else {"ran": False, "passed": None}
            )

        snapshot_after = self.workspace_snapshot()
        workspace_unchanged = snapshot_before == snapshot_after
        if not workspace_unchanged:
            raise PatchValidationError("Real workspace changed during dry-run")

        return {
            "valid": True,
            "declared_path": normalized_path,
            "changed_paths": changed_paths,
            "normalized_patch": normalized_patch,
            "explanation": explanation.strip(),
            "patched_content": patched_content,
            "test_result": test_result,
            "workspace_unchanged": workspace_unchanged,
        }

    def safe_validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return {
                "ok": True,
                "result": self.validate(
                    path=arguments["path"],
                    patch=arguments["patch"],
                    explanation=arguments["explanation"],
                ),
            }
        except (
            KeyError,
            OSError,
            UnicodeError,
            ReadOnlyToolError,
            PatchValidationError,
            subprocess.SubprocessError,
        ) as exc:
            return {"ok": False, "error": str(exc)}

    def workspace_snapshot(self) -> dict[str, str]:
        snapshot = {}
        for file_path in sorted(self.workspace_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if any(
                part in {"__pycache__", ".pytest_cache", ".git"}
                for part in file_path.parts
            ):
                continue
            relative = file_path.relative_to(self.workspace_dir).as_posix()
            snapshot[relative] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        return snapshot

    def _validate_declared_path(self, raw_path: str) -> str:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise PatchValidationError("Path must not be empty")
        path = PurePosixPath(raw_path.strip())
        if path.is_absolute() or ".." in path.parts:
            raise PatchValidationError("Patch path must stay inside workspace")
        normalized = path.as_posix()
        target = (self.workspace_dir / normalized).resolve()
        try:
            target.relative_to(self.workspace_dir)
        except ValueError as exc:
            raise PatchValidationError(
                "Patch path must stay inside workspace"
            ) from exc
        if not target.is_file():
            raise PatchValidationError(
                f"Stage 3 only accepts existing text files: {normalized}"
            )
        self.path_guard._assert_allowed_path(target)
        if b"\x00" in target.read_bytes()[:4096]:
            raise PatchValidationError(f"Binary file is not allowed: {normalized}")
        return normalized

    def _normalize_patch(self, patch: str) -> str:
        if not isinstance(patch, str) or not patch.strip():
            raise PatchValidationError("Patch must not be empty")
        if len(patch) > self.MAX_PATCH_CHARS:
            raise PatchValidationError(
                f"Patch exceeds {self.MAX_PATCH_CHARS} characters"
            )
        normalized = patch.replace("\r\n", "\n").replace("\r", "\n").strip()
        if "*** Begin Patch" in normalized or "*** End Patch" in normalized:
            raise PatchValidationError(
                "Use standard unified diff, not apply_patch wrapper syntax"
            )
        return normalized + "\n"

    def _extract_changed_paths(self, patch: str) -> list[str]:
        diff_paths = []
        old_paths = []
        new_paths = []
        for line in patch.splitlines():
            diff_match = self.DIFF_HEADER.match(line)
            if diff_match:
                if diff_match.group(1) != diff_match.group(2):
                    raise PatchValidationError("Rename patches are not allowed")
                diff_paths.append(diff_match.group(1))
                continue
            old_match = self.OLD_HEADER.match(line)
            if old_match:
                old_paths.append(old_match.group(1))
                continue
            new_match = self.NEW_HEADER.match(line)
            if new_match:
                new_paths.append(new_match.group(1))

        if not diff_paths or not old_paths or not new_paths:
            raise PatchValidationError(
                "Patch must include diff --git, ---, and +++ headers"
            )
        if not (diff_paths == old_paths == new_paths):
            raise PatchValidationError("Patch headers refer to different files")
        unique_paths = list(dict.fromkeys(diff_paths))
        if len(unique_paths) > self.MAX_CHANGED_FILES:
            raise PatchValidationError(
                f"Stage 3 allows at most {self.MAX_CHANGED_FILES} changed file"
            )
        for path in unique_paths:
            self._validate_declared_path(path)
        return unique_paths

    @staticmethod
    def _run_git_apply(
        workspace: Path, patch: str, check_only: bool
    ) -> subprocess.CompletedProcess[str]:
        # LLMs often miscount hunk lengths. --recount repairs only hunk metadata;
        # Git still requires the file context and all guarded paths to match.
        command = ["git", "apply", "--recount", "--whitespace=error"]
        if check_only:
            command.append("--check")
        return subprocess.run(
            command,
            cwd=workspace,
            input=patch,
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )

    @staticmethod
    def _changed_files_between(original: Path, changed: Path) -> list[str]:
        paths = set()
        for root in (original, changed):
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if any(
                    part in {"__pycache__", ".pytest_cache", ".git"}
                    for part in path.parts
                ):
                    continue
                paths.add(path.relative_to(root).as_posix())

        changed_paths = []
        for relative in sorted(paths):
            original_file = original / relative
            changed_file = changed / relative
            if not original_file.exists() or not changed_file.exists():
                changed_paths.append(relative)
                continue
            if original_file.read_bytes() != changed_file.read_bytes():
                changed_paths.append(relative)
        return changed_paths

    @staticmethod
    def _run_tests(workspace: Path) -> dict[str, Any]:
        process = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {
            "ran": True,
            "passed": process.returncode == 0,
            "exit_code": process.returncode,
            "stdout": process.stdout[-8_000:],
            "stderr": process.stderr[-4_000:],
        }
