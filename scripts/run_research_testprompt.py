"""Run the Research prompt regression suite against the local Meibook API.

Script này đọc ``Markdowns/TestPrompt_Research.md``, gọi ``POST /query`` với
``mode=research`` và ``research_topic`` tương ứng từng nhóm test. Đây là runner
nhẹ để kiểm tra routing/retrieval cơ bản, không thay thế manual review chất
lượng câu trả lời.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


TOPIC_BY_GROUP = {
    "INFO": "information_systems",
    "LEGAL": "legal_compliance",
    "ACC": "accounting",
    "GA": "general_affairs",
}


def section(block: str, heading: str) -> str:
    pattern = (
        rf"- \*\*{re.escape(heading)}:\*\*\s*\n"
        rf"(?P<body>.*?)(?=\n- \*\*|\n### TC-|\Z)"
    )
    match = re.search(pattern, block, flags=re.S)
    return match.group("body").strip() if match else ""


def inline_section(block: str, heading: str) -> str:
    match = re.search(rf"- \*\*{re.escape(heading)}:\*\*\s*(.+)", block)
    return match.group(1).strip() if match else ""


def parse_sources(source_text: str) -> list[str]:
    values = re.findall(r"`([^`]+)`", source_text or "")
    if not values and source_text:
        values = [item.strip() for item in source_text.split(",") if item.strip()]
    return values


def normalize_source_name(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"\.(md|pdf|docx|xlsx|pptx)$", "", value)
    return re.sub(r"[_\s\-]+", "", value)


def source_matches(expected_sources: list[str], actual_sources: list[dict[str, Any]]) -> bool:
    if not expected_sources:
        return False

    actual_names: list[str] = []
    for source in actual_sources or []:
        actual_names.extend([source.get("file", ""), source.get("title", "")])
    actual_norms = [normalize_source_name(name) for name in actual_names if name]

    for expected in expected_sources:
        normalized_expected = normalize_source_name(expected)
        if any(
            normalized_expected
            and (normalized_expected in actual or actual in normalized_expected)
            for actual in actual_norms
        ):
            return True
    return False


def parse_cases(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"(?=^### TC-)", text, flags=re.M)
    blocks = [block.strip() for block in blocks if block.strip().startswith("### TC-")]
    if limit > 0:
        blocks = blocks[:limit]

    cases: list[dict[str, Any]] = []
    for block in blocks:
        title_line = block.splitlines()[0].strip().removeprefix("### ")
        match = re.match(r"TC-([A-Z]+)-([0-9]+):\s*(.*)", title_line)
        group = match.group(1) if match else "UNKNOWN"
        case_id = title_line.split(":", 1)[0]
        cases.append(
            {
                "id": case_id,
                "title": title_line.split(":", 1)[1].strip()
                if ":" in title_line
                else title_line,
                "group": group,
                "topic": TOPIC_BY_GROUP.get(group),
                "question": section(block, "Câu hỏi kiểm thử"),
                "expected_sources": parse_sources(inline_section(block, "Nguồn tài liệu")),
                "keywords": re.findall(
                    r"`([^`]+)`",
                    section(block, "Từ khóa truy xuất gợi ý") or "",
                ),
            }
        )
    return cases


def post_json(url: str, payload: dict[str, Any], *, timeout: int) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def summarize(results: list[dict[str, Any]], total: int, started_at: float) -> dict[str, Any]:
    latencies = [
        item["elapsed_sec"]
        for item in results
        if isinstance(item.get("elapsed_sec"), (int, float))
    ]
    summary = {
        "total": total,
        "ran": len(results),
        "ok": sum(1 for item in results if item.get("ok")),
        "scope_ok": sum(1 for item in results if item.get("answer_scope") == "research"),
        "source_match": sum(1 for item in results if item.get("source_match")),
        "category_ok": sum(1 for item in results if item.get("category_ok")),
        "errors": sum(1 for item in results if not item.get("ok")),
        "elapsed_total_sec": round(time.time() - started_at, 2),
    }
    if latencies:
        summary.update(
            {
                "latency_avg_sec": round(sum(latencies) / len(latencies), 2),
                "latency_median_sec": round(statistics.median(latencies), 2),
                "latency_max_sec": round(max(latencies), 2),
            }
        )
    return summary


def write_report(path: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="Markdowns/TestPrompt_Research.md")
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--output", default="/tmp/meibook_research_testprompt_results.json")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay giữa các case.")
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases = parse_cases(Path(args.file), limit=args.limit)
    results: list[dict[str, Any]] = []
    started_at = time.time()
    output_path = Path(args.output)
    api_url = args.base_url.rstrip("/") + "/query"
    print(f"Running {len(cases)} research cases -> {output_path}", flush=True)

    for index, case in enumerate(cases, 1):
        item = {key: case[key] for key in ("id", "title", "group", "topic", "question")}
        item["expected_sources"] = case["expected_sources"]
        payload = {
            "session_id": str(uuid.uuid4()),
            "question": case["question"],
            "stream": False,
            "model": "auto",
            "mode": "research",
            "ui_language": "vi",
            "research_topic": case["topic"],
        }
        case_started_at = time.time()

        for attempt in range(1, args.retries + 2):
            try:
                status, body = post_json(api_url, payload, timeout=args.timeout)
                sources = body.get("sources") or []
                item.update(
                    {
                        "ok": status == 200,
                        "http_status": status,
                        "elapsed_sec": round(time.time() - case_started_at, 2),
                        "answer_scope": body.get("answer_scope"),
                        "model": body.get("model"),
                        "answer": body.get("answer") or "",
                        "answer_preview": (body.get("answer") or "")[:500],
                        "sources_count": len(sources),
                        "sources": sources[:8],
                        "source_match": source_matches(case["expected_sources"], sources),
                        "category_ok": all(
                            source.get("category") == case["topic"]
                            for source in sources
                            if source.get("category")
                        )
                        if sources
                        else False,
                        "attempts": attempt,
                    }
                )
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt <= args.retries:
                    retry_after = int(exc.headers.get("Retry-After") or "5")
                    print(
                        f"[{index:03d}/{len(cases)}] {case['id']} 429, "
                        f"sleep {retry_after}s",
                        flush=True,
                    )
                    time.sleep(retry_after + max(args.delay, 0.25))
                    continue
                item.update(
                    {
                        "ok": False,
                        "http_status": exc.code,
                        "elapsed_sec": round(time.time() - case_started_at, 2),
                        "error": detail[:1000],
                        "attempts": attempt,
                    }
                )
                break
            except Exception as exc:  # pragma: no cover - runner diagnostic path
                item.update(
                    {
                        "ok": False,
                        "http_status": None,
                        "elapsed_sec": round(time.time() - case_started_at, 2),
                        "error": f"{type(exc).__name__}: {exc}",
                        "attempts": attempt,
                    }
                )
                break

        results.append(item)
        summary = summarize(results, len(cases), started_at)
        write_report(output_path, summary, results)
        print(
            f"[{index:03d}/{len(cases)}] {case['id']} "
            f"ok={item.get('ok')} scope={item.get('answer_scope')} "
            f"src={item.get('sources_count', 0)} match={item.get('source_match')} "
            f"cat={item.get('category_ok')} {item.get('elapsed_sec')}s",
            flush=True,
        )
        if args.delay > 0 and index < len(cases):
            time.sleep(args.delay)

    print(json.dumps(summarize(results, len(cases), started_at), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
