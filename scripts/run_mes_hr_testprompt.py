#!/usr/bin/env python3
"""Run the MES/HR prompt regression suite against the local Meibook API.

This script parses ``Markdowns/TestPrompt.md``, resolves the target mode/language,
creates a session, and calls the `/query` endpoint sequentially.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

# Default mapping of test groups/IDs to API modes.
# mode "mkac" acts as the HR mode in backend config.
MODE_BY_CASE_RANGE = {
    "MES": "mes",
    "HR": "mkac",
}


def parse_markdown_tables(path: Path) -> list[dict[str, Any]]:
    """Parse test cases from markdown tables in TestPrompt.md."""
    text = path.read_text(encoding="utf-8")

    # Matches markdown table rows like | 1 | Question | Expectation |
    # Ignores header rows and alignment delimiters |---|
    row_pattern = re.compile(r"^\s*\|\s*([0-9a-zA-Z\-]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|", re.M)

    cases = []
    for match in row_pattern.finditer(text):
        case_id = match.group(1).strip()
        question = match.group(2).strip()
        expected = match.group(3).strip()

        # Skip table headers
        if case_id.lower() in ("#", "id", "id (ja)") or "---" in case_id:
            continue

        cases.append({
            "id": case_id,
            "question": question,
            "expected": expected
        })
    return cases


def get_case_mode(case_id: str) -> str:
    """Map case ID to target API mode."""
    # Handle explicit cross-domain overrides
    if case_id in ("84", "JA-084"):
        return "mes"
    if case_id in ("85", "JA-085"):
        return "mkac"
    if case_id in ("86", "JA-086"):
        return "mes"

    # Check numeric IDs
    num_part = re.sub(r"\D", "", case_id)
    if num_part.isdigit():
        val = int(num_part)
        if 1 <= val <= 42:
            return "mes"
        elif 43 <= val <= 83:
            return "mkac"
        elif 87 <= val <= 90:
            # 87: mkac, 88-89: mes, 90: mkac
            return "mkac" if val in (87, 90) else "mes"

    return "mes"


def normalize_number(text: str) -> list[str]:
    """Extract and normalize numbers from text (e.g. 29.406 -> 29406)."""
    # Find all sequences of digits, potentially separated by dot/comma
    numbers = []
    for m in re.finditer(r"\b\d+[\.,]\d+\b|\b\d+\b", text):
        val = m.group(0).replace(".", "").replace(",", "")
        numbers.append(val)
    return numbers


def score_answer(expected: str, actual: str) -> bool:
    """Tolerant scoring using entity and normalized number matching."""
    actual_lower = actual.lower()
    expected_lower = expected.lower()

    # 1. Match numbers
    expected_nums = normalize_number(expected)
    if expected_nums:
        actual_nums = normalize_number(actual)
        for num in expected_nums:
            if num not in actual_nums and num not in actual_lower:
                return False

    # 2. Extract key entities (quoted values, uppercase codes, specific departments)
    entities = re.findall(r'"([^"]+)"|`([^`]+)`|\b[A-Z]{2,}\b|\b[A-Z]\d+[-a-zA-Z0-9]*\b', expected)
    entities = [e[0] or e[1] or e[2] or e[3] for e in entities if any(e)]

    # Fallback to key terms if no regex entities found
    if not entities:
        # Get words longer than 3 chars excluding basic connectors
        words = [w for w in re.findall(r"\w+", expected_lower) if len(w) > 3]
        # Filter out common Vietnamese/Japanese stop words if any
        stop_words = {"hoặc", "dưới", "dạng", "bảng", "trên", "dữ", "liệu", "bằng", "chất"}
        entities = [w for w in words if w not in stop_words][:3]

    for entity in entities:
        if entity.lower() not in actual_lower:
            return False

    return True


def create_session(base_url: str) -> str:
    """Create a fresh session UUID via the API."""
    url = f"{base_url}/sessions"
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.load(response)
            return data["session_id"]
    except Exception as e:
        # Fallback to local UUID generation if server session endpoint fails
        print(f"Warn: session endpoint failed ({e}), using local UUID")
        return str(uuid.uuid4())


def run_query(base_url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Execute a single sync query call."""
    url = f"{base_url}/query"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="MES/HR TestPrompt runner")
    parser.add_argument("--file", default="Markdowns/TestPrompt.md", help="Test file path")
    parser.add_argument("--base-url", default="http://localhost:8002", help="API base URL")
    parser.add_argument("--output", default="/tmp/meibook_testprompt_results.json", help="JSON result file")
    parser.add_argument("--delay", type=float, default=4.1, help="Delay between requests")
    parser.add_argument("--timeout", type=int, default=150, help="Request timeout")
    parser.add_argument("--retries", type=int, default=4, help="HTTP 429 retries")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases")
    parser.add_argument("--run-unsafe", action="store_true", help="Run unsafe email/destructive prompts")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: test file not found at {file_path}")
        raise SystemExit(1)

    print(f"Parsing cases from {file_path}...")
    all_cases = parse_markdown_tables(file_path)
    print(f"Found {len(all_cases)} total rows in tables")

    # Filter cases
    cases = []
    for c in all_cases:
        case_id = c["id"]
        # Skip unsafe email cases unless explicitly allowed
        if case_id in ("80", "JA-080") and not args.run_unsafe:
            print(f"Skipping unsafe case {case_id} (email action)")
            continue
        cases.append(c)

    if args.limit > 0:
        cases = cases[:args.limit]

    print(f"Running {len(cases)} test cases...")

    results = []
    passed_count = 0
    failed_cases = []

    for idx, case in enumerate(cases):
        case_id = case["id"]
        question = case["question"]
        expected = case["expected"]
        mode = get_case_mode(case_id)
        ui_lang = "ja" if (case_id.startswith("JA-") or "ja" in case_id.lower()) else "vi"

        print(f"\n[{idx+1}/{len(cases)}] Case {case_id} ({mode}/{ui_lang})")
        print(f"  Q: {question}")

        # Check for multi-turn questions
        # Pattern: (1) "..." -> (2) "..."
        turns = re.findall(r"\(\d+\)\s*\"([^\"]+)\"", question)
        if not turns:
            turns = [question]

        session_id = create_session(args.base_url)
        context = []
        last_answer = ""
        actual_model = "unknown"
        actual_scope = "unknown"
        latency = 0.0
        http_ok = True
        error_msg = ""

        for turn_idx, turn_text in enumerate(turns):
            if turn_idx > 0:
                # Add previous turn to context
                context.append({"role": "user", "content": turns[turn_idx-1]})
                context.append({"role": "assistant", "content": last_answer})
                # Add brief delay between turns
                time.sleep(1.0)

            payload = {
                "session_id": session_id,
                "question": turn_text,
                "stream": False,
                "model": "auto",
                "mode": mode,
                "ui_language": ui_lang,
                "employee_id": "000000",  # Guest account
                "conversation_context": context
            }

            # Request execution with 429 retry loop
            for attempt in range(args.retries + 1):
                try:
                    t0 = time.monotonic()
                    response = run_query(args.base_url, payload, args.timeout)
                    latency = time.monotonic() - t0
                    last_answer = response.get("answer", "")
                    meta = response.get("meta", {})
                    actual_model = meta.get("model", "unknown")
                    actual_scope = meta.get("answer_scope", "unknown")
                    http_ok = True
                    break
                except urllib.error.HTTPError as err:
                    latency = 0.0
                    http_ok = False
                    error_msg = f"HTTP {err.code}: {err.reason}"
                    if err.code == 429:
                        retry_after = int(err.headers.get("Retry-After", 5))
                        print(f"  Got HTTP 429. Backing off for {retry_after}s (attempt {attempt+1}/{args.retries})")
                        time.sleep(retry_after)
                        continue
                    break
                except Exception as err:
                    latency = 0.0
                    http_ok = False
                    error_msg = f"NetworkError: {err}"
                    break

            if not http_ok:
                break

        # Score final turn
        is_ok = False
        if http_ok:
            is_ok = score_answer(expected, last_answer)
            print(f"  A: {last_answer[:120]}...")
            print(f"  Score: {'PASS' if is_ok else 'FAIL'} | Scope: {actual_scope} | Model: {actual_model} | {latency:.2f}s")
        else:
            print(f"  ERROR: {error_msg}")

        case_res = {
            "id": case_id,
            "question": question,
            "expected": expected,
            "actual": last_answer if http_ok else "",
            "ok": is_ok and http_ok,
            "http_ok": http_ok,
            "error": error_msg if not http_ok else None,
            "latency": latency,
            "model": actual_model,
            "answer_scope": actual_scope
        }
        results.append(case_res)

        if case_res["ok"]:
            passed_count += 1
        else:
            failed_cases.append(case_id)

        # Write incremental output
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({
                    "summary": {
                        "total": len(cases),
                        "ran": len(results),
                        "passed": passed_count,
                        "failed": len(results) - passed_count,
                        "failed_ids": failed_cases
                    },
                    "results": results
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to write incremental output ({e})")

        # Rate limit delay
        if idx < len(cases) - 1:
            time.sleep(args.delay)

    print("\n=== TEST RUN SUMMARY ===")
    print(f"Total: {len(cases)} | Passed: {passed_count} | Failed: {len(cases) - passed_count}")
    if failed_cases:
        print(f"Failed case IDs: {', '.join(failed_cases)}")
        raise SystemExit(1)
    else:
        print("All tested cases passed successfully!")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
