#!/usr/bin/env python3
"""Run the full 135-case Research TestPrompt and generate a report."""

import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_PROMPT_PATH = REPO_ROOT / "Markdowns" / "TestPrompt_Research.md"
REPORT_PATH = REPO_ROOT / "logs" / "research_evaluation_report.md"
API_URL = "http://localhost:8001/query"

# Mapping from natural language topic name to API topic ID
TOPIC_MAP = {
    "Công nghệ thông tin & Bảo mật": "information_systems",
    "Pháp chế & Quản lý rủi ro": "legal_compliance",
    "Kế toán": "accounting",
    "Hành chính tổng hợp": "general_affairs"
}


def parse_test_cases() -> list[dict]:
    content = TEST_PROMPT_PATH.read_text(encoding="utf-8")
    cases = []

    # Split by TC ID headers (e.g., ### TC-INFO-001: ...)
    blocks = re.split(r'\n### (TC-[A-Z]+-\d+:\s*.+)\n', content)

    for i in range(1, len(blocks), 2):
        header = blocks[i].strip()
        body = blocks[i+1]

        # Extract metadata
        topic_match = re.search(r'- \*\*Nhóm tài liệu đang test:\*\*\s*(.+)', body)
        lang_match = re.search(r'- \*\*Ngôn ngữ:\*\*\s*(.+)', body)
        question_match = re.search(r'- \*\*Câu hỏi kiểm thử:\*\*\n\s+(.+)', body)

        if not (topic_match and question_match):
            continue

        topic_raw = topic_match.group(1).strip()
        topic_id = TOPIC_MAP.get(topic_raw, "all")
        question = question_match.group(1).strip()

        # Determine language (VI/JA)
        lang = "vi"
        if lang_match and "Nhật" in lang_match.group(1):
            lang = "ja"
        elif re.search(r"[぀-ヿ㐀-鿿]", question):
            lang = "ja"

        cases.append({
            "id": header.split(":")[0],
            "title": header,
            "topic_raw": topic_raw,
            "topic_id": topic_id,
            "language": lang,
            "question": question,
            "expected_body": body.strip()
        })
    return cases


async def run_query(client: httpx.AsyncClient, session_id: str, case: dict) -> dict:
    payload = {
        "session_id": session_id,
        "question": case["question"],
        "model": "auto",
        "mode": "research",
        "stream": False,
        "ui_language": case["language"],
        "research_topic": case["topic_id"]
    }

    start_time = asyncio.get_event_loop().time()
    try:
        resp = await client.post(API_URL, json=payload, timeout=90.0)
        resp.raise_for_status()
        end_time = asyncio.get_event_loop().time()
        latency = end_time - start_time

        data = resp.json()
        return {
            "success": True,
            "latency": latency,
            "answer": data.get("answer", ""),
            "model_used": data.get("model", ""),
            "sources": [s.get("file") for s in data.get("sources", [])]
        }
    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        return {
            "success": False,
            "latency": end_time - start_time,
            "error": str(e)
        }


def is_paid_cloud_model(model_name: str) -> bool:
    """Check if the model routed by LiteLLM is a paid OpenAI/Azure model."""
    name = model_name.lower()
    # Dấu hiệu của model cloud trả phí:
    if "openai" in name or "gpt" in name or "grok" in name:
        return True
    return False


async def check_local_model_health(client: httpx.AsyncClient, session_id: str) -> bool:
    """Smoke test to ensure the system doesn't fallback to paid cloud APIs."""
    print("Running pre-flight check to verify local model availability...")
    dummy_case = {
        "question": "Kiểm tra hệ thống, trả lời ngắn gọn là OK.",
        "language": "vi",
        "topic_id": "information_systems"
    }
    result = await run_query(client, session_id, dummy_case)
    if not result["success"]:
        print(f"Pre-flight check failed to connect to API: {result.get('error')}")
        return False

    model_used = result.get("model_used", "")
    print(f"Pre-flight check routed to model: {model_used}")

    if is_paid_cloud_model(model_used):
        print(f"\n[DANGER] Mạng LLM bị fallback về model trả phí ({model_used}).")
        print("Model local (Qwen qua IP/ngrok) đang không phản hồi.")
        print("Tiến trình bị HỦY để tránh mất phí API oan uổng!")
        return False

    print("Pre-flight check OK. Local model is responding.\n")
    return True


async def main():
    cases = parse_test_cases()
    print(f"Found {len(cases)} test cases in TestPrompt_Research.md")

    if not cases:
        print("No cases parsed. Check file format.")
        return

    # Get a session ID
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://localhost:8001/sessions")
        session_id = resp.json()["session_id"]
        print(f"Using session: {session_id}")

        if not await check_local_model_health(client, session_id):
            return

        REPORT_PATH.parent.mkdir(exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("# Báo cáo kiểm thử Research Mode\n\n")

            for i, case in enumerate(cases, 1):
                print(f"[{i:03d}/{len(cases)}] Running {case['id']}... ", end="", flush=True)
                result = await run_query(client, session_id, case)

                latency_str = f"{result['latency']:.2f}s"
                if result["success"]:
                    model_used = result.get("model_used", "")
                    if is_paid_cloud_model(model_used):
                        print(f"FAIL: Fallback to paid cloud API ({model_used})")
                        print("\n[DANGER] Mạng LLM giữa chừng bị fallback về model trả phí!")
                        print("HỦY tiến trình để bảo vệ ngân sách.")
                        f.write(f"\n\n**[DANGER] HỦY KIỂM THỬ GIỮA CHỪNG.** Model bị fallback về {model_used}\n")
                        break

                    print(f"OK ({latency_str}) [Model: {model_used}]")
                else:
                    print(f"FAIL ({latency_str})")

                f.write(f"## {case['title']}\n")
                f.write(f"- **Chủ đề:** {case['topic_raw']}\n")
                f.write(f"- **Câu hỏi:** {case['question']}\n")
                f.write(f"- **Thời gian xử lý:** {latency_str}\n")
                if result.get("model_used"):
                    f.write(f"- **Model thực thi:** `{result['model_used']}`\n")
                f.write("\n")

                if result["success"]:
                    f.write("### AI Trả lời:\n")
                    f.write(f"{result['answer']}\n\n")
                    f.write("### Nguồn trích xuất:\n")
                    for src in set(result["sources"]):
                        f.write(f"- `{src}`\n")
                else:
                    f.write(f"**LỖI:** {result['error']}\n")

                f.write("\n---\n\n")
                f.flush()

                # Sleep briefly to avoid rate limiting
                await asyncio.sleep(2)

    print(f"\nDone! Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
