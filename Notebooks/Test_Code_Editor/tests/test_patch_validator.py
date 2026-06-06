from pathlib import Path

import pytest

from patch_validator import PatchValidator


VALID_PATCH = """diff --git a/calculator.py b/calculator.py
--- a/calculator.py
+++ b/calculator.py
@@ -6,5 +6,6 @@ def add(a: float, b: float) -> float:
 
 
 def divide(a: float, b: float) -> float:
-    # Intentional Milestone 1 bug: no explicit zero-division validation.
+    if b == 0:
+        raise ValueError("b must not be zero")
     return a / b
"""


@pytest.fixture()
def validator() -> PatchValidator:
    test_root = Path(__file__).resolve().parents[1]
    workspace = test_root / "fixtures" / "sample_project"
    return PatchValidator(workspace)


def test_valid_patch_applies_only_in_temporary_copy(validator):
    before = validator.workspace_snapshot()

    result = validator.validate(
        path="calculator.py",
        patch=VALID_PATCH,
        explanation="Validate zero denominator before division.",
    )

    assert result["valid"] is True
    assert result["changed_paths"] == ["calculator.py"]
    assert 'raise ValueError("b must not be zero")' in result["patched_content"]
    assert result["test_result"]["passed"] is True
    assert result["workspace_unchanged"] is True
    assert validator.workspace_snapshot() == before


def test_malformed_patch_is_rejected(validator):
    result = validator.safe_validate(
        {
            "path": "calculator.py",
            "patch": "not a unified diff",
            "explanation": "invalid",
        }
    )

    assert result["ok"] is False
    assert "headers" in result["error"]


def test_apply_patch_wrapper_is_rejected(validator):
    result = validator.safe_validate(
        {
            "path": "calculator.py",
            "patch": "*** Begin Patch\n*** End Patch",
            "explanation": "invalid format",
        }
    )

    assert result["ok"] is False
    assert "standard unified diff" in result["error"]


@pytest.mark.parametrize(
    "path",
    [
        "../.env",
        "/home/jkl0909/Code/llm/VLLM-PD/.env",
        ".env",
    ],
)
def test_sensitive_or_outside_path_is_rejected(validator, path):
    result = validator.safe_validate(
        {
            "path": path,
            "patch": VALID_PATCH,
            "explanation": "must fail",
        }
    )

    assert result["ok"] is False


def test_declared_path_must_match_diff_path(validator):
    result = validator.safe_validate(
        {
            "path": "README.md",
            "patch": VALID_PATCH,
            "explanation": "wrong declared path",
        }
    )

    assert result["ok"] is False
    assert "exactly match" in result["error"]


def test_multiple_files_are_rejected(validator):
    second_patch = VALID_PATCH + """diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1,1 +1,1 @@
-# Calculator Fixture
+# Changed Fixture
"""
    result = validator.safe_validate(
        {
            "path": "calculator.py",
            "patch": second_patch,
            "explanation": "too many files",
        }
    )

    assert result["ok"] is False
    assert "at most 1" in result["error"]


def test_patch_that_does_not_apply_is_rejected(validator):
    broken_patch = VALID_PATCH.replace(
        "def divide(a: float, b: float) -> float:",
        "def missing_function():",
    )
    result = validator.safe_validate(
        {
            "path": "calculator.py",
            "patch": broken_patch,
            "explanation": "context mismatch",
        }
    )

    assert result["ok"] is False
    assert "patch" in result["error"].lower()


def test_patch_can_be_valid_but_fail_tests(validator):
    wrong_patch = VALID_PATCH.replace(
        'raise ValueError("b must not be zero")',
        'raise ValueError("invalid denominator")',
    )
    result = validator.validate(
        path="calculator.py",
        patch=wrong_patch,
        explanation="Wrong message on purpose.",
    )

    assert result["valid"] is True
    assert result["test_result"]["passed"] is False
    assert result["workspace_unchanged"] is True
