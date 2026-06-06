from pathlib import Path

import pytest

from read_only_tools import ReadOnlyToolbox


@pytest.fixture()
def toolbox() -> ReadOnlyToolbox:
    test_root = Path(__file__).resolve().parents[1]
    repo_root = test_root.parents[1]
    workspace = test_root / "fixtures" / "sample_project"
    return ReadOnlyToolbox(workspace, repo_root)


def test_list_directory_hides_runtime_cache(toolbox):
    result = toolbox.execute("list_directory", {"directory": "."})

    assert result["ok"] is True
    names = {entry["name"] for entry in result["result"]["entries"]}
    assert {"calculator.py", "README.md", "tests"} <= names
    assert "__pycache__" not in names
    assert ".pytest_cache" not in names


def test_read_file_returns_fixture_content(toolbox):
    result = toolbox.execute("read_file", {"file_path": "calculator.py"})

    assert result["ok"] is True
    assert "def divide" in result["result"]["content"]
    assert result["result"]["read_only"] is True


def test_search_text_finds_definition_and_test_usage(toolbox):
    result = toolbox.execute("search_text", {"query": "divide", "directory": "."})

    assert result["ok"] is True
    paths = {match["path"] for match in result["result"]["matches"]}
    assert "calculator.py" in paths
    assert "tests/test_calculator.py" in paths


def test_metadata_is_marked_read_only(toolbox):
    result = toolbox.execute(
        "get_file_metadata", {"file_path": "calculator.py"}
    )

    assert result["ok"] is True
    assert result["result"]["read_only"] is True
    assert result["result"]["type"] == "file"


@pytest.mark.parametrize(
    "path",
    [
        "../../../../.env",
        "/home/jkl0909/Code/llm/VLLM-PD/.env",
    ],
)
def test_paths_outside_workspace_are_rejected(toolbox, path):
    result = toolbox.execute("read_file", {"file_path": path})

    assert result["ok"] is False
    assert "inside workspace" in result["error"]


def test_sensitive_file_inside_workspace_is_rejected(toolbox, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("SECRET=value", encoding="utf-8")
    isolated = ReadOnlyToolbox(workspace, tmp_path)

    result = isolated.execute("read_file", {"file_path": ".env"})

    assert result["ok"] is False
    assert "blocked path" in result["error"]


def test_symlink_escape_is_rejected(toolbox, tmp_path):
    link = toolbox.workspace_dir / "outside-link"
    try:
        link.symlink_to(tmp_path)
        result = toolbox.execute("list_directory", {"directory": "outside-link"})
        assert result["ok"] is False
        assert "inside workspace" in result["error"]
    finally:
        link.unlink(missing_ok=True)


def test_unknown_write_tool_is_rejected(toolbox):
    result = toolbox.execute(
        "write_file",
        {"file_path": "calculator.py", "content": "malicious change"},
    )

    assert result["ok"] is False
    assert "not available in read-only mode" in result["error"]


def test_git_tools_are_scoped_and_read_only(toolbox):
    status = toolbox.execute("git_status", {})
    diff = toolbox.execute("git_diff", {})
    log = toolbox.execute("git_log", {"max_count": 3})

    assert status["ok"] is True
    assert diff["ok"] is True
    assert log["ok"] is True
