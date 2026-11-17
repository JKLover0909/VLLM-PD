import asyncio
import json

import pytest
from fastapi.responses import StreamingResponse

from src.api.prompt_guard import refusal_text
from src.api.query_guard import guarded_sse, guard_query_response, guarded_query
from src.api.schemas import QueryRequest, QueryResponse


SESSION_ID = "00000000-0000-4000-8000-000000000501"
DISCLOSURE = "```sql\nSELECT employee_id FROM employees\n```"


def make_request(*, mode="mkac", language="vi"):
    return QueryRequest(
        session_id=SESSION_ID,
        question="Quy định nghỉ phép là gì?",
        mode=mode,
        ui_language=language,
    )


def encode_event(event):
    return ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode()


def decode_events(chunks):
    wire = b"".join(chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks)
    text = wire.decode("utf-8")
    return [
        json.loads(frame.removeprefix("data: "))
        for frame in text.split("\n\n")
        if frame.startswith("data:")
    ]


def collect(iterator):
    async def _collect():
        return [chunk async for chunk in iterator]

    return asyncio.run(_collect())


def producer(*events):
    async def _producer():
        for event in events:
            yield encode_event(event)

    return _producer()


def test_safe_stream_buffers_answer_until_done_and_preserves_event_order():
    events = collect(
        guarded_sse(
            producer(
                {"type": "status", "message": "Đang xử lý"},
                {"type": "sources", "sources": [{"title": "Quy định nghỉ phép"}]},
                {"type": "meta", "model": "safe-model", "answer_scope": "mkac"},
                {"type": "token", "content": "Theo quy định "},
                {"type": "token", "content": "hiện hành."},
                {"type": "done"},
                {"type": "token", "content": "hidden after terminal"},
            ),
            make_request(),
        )
    )
    decoded = decode_events(events)
    assert [event["type"] for event in decoded] == [
        "status", "sources", "meta", "token", "done"
    ]
    assert next(event for event in decoded if event["type"] == "token")["content"] == "Theo quy định hiện hành."


def test_split_utf8_sse_bytes_and_split_json_frames_are_decoded_safely():
    frame = encode_event({"type": "token", "content": "安全な回答です。"})
    cut = frame.index("全".encode()) + 1

    async def fragmented():
        yield frame[:cut]
        yield frame[cut:]
        yield encode_event({"type": "done"})

    events = decode_events(collect(guarded_sse(fragmented(), make_request(language="ja"))))
    assert next(event for event in events if event["type"] == "token")["content"] == "安全な回答です。"
    assert events[-1]["type"] == "done"


def test_unsafe_fragmented_answer_is_replaced_without_leaking_original_fragments():
    events = decode_events(
        collect(
            guarded_sse(
                producer(
                    {"type": "status", "message": "working"},
                    {"type": "token", "content": "```sql\\nSELECT employee_"},
                    {"type": "token", "content": "id FROM employees\\n```"},
                    {"type": "done"},
                ),
                make_request(mode="qa"),
            )
        )
    )
    serialized = json.dumps(events)
    assert "SELECT" not in serialized
    assert DISCLOSURE not in serialized
    assert [event["type"] for event in events] == [
        "status", "sources", "meta", "replace", "done"
    ]
    assert events[3]["content"] == refusal_text("vi")
    assert events[1]["sources"] == []
    assert events[2]["answer_scope"] == "guardrail"


def test_safe_replace_event_replaces_prior_token_fragments():
    events = decode_events(
        collect(
            guarded_sse(
                producer(
                    {"type": "token", "content": "obsolete partial"},
                    {"type": "replace", "content": "Final safe answer."},
                    {"type": "done"},
                ),
                make_request(),
            )
        )
    )
    answer = next(event for event in events if event["type"] == "replace")
    assert answer["content"] == "Final safe answer."
    assert "obsolete partial" not in json.dumps(events)


def test_malformed_frame_and_incomplete_eof_return_only_safe_error():
    async def malformed():
        yield b'data: {"type":"token","content":"safe"}\n\n'
        yield b"data: {not-json}\n\n"

    malformed_events = decode_events(collect(guarded_sse(malformed(), make_request())))
    assert "safe" not in json.dumps(malformed_events)
    assert [event["type"] for event in malformed_events] == ["error"]

    eof_events = decode_events(
        collect(guarded_sse(producer({"type": "token", "content": "partial"}), make_request()))
    )
    assert "partial" not in json.dumps(eof_events)
    assert [event["type"] for event in eof_events] == ["error"]


def test_answer_size_limit_refuses_without_emitting_buffered_content(monkeypatch):
    import src.api.query_guard as query_guard

    monkeypatch.setattr(query_guard, "MAX_ANSWER_CHARS", 4)
    events = decode_events(
        collect(
            guarded_sse(
                producer({"type": "token", "content": "12345"}),
                make_request(),
            )
        )
    )
    assert "12345" not in json.dumps(events)
    assert [event["type"] for event in events] == ["sources", "meta", "token", "done"]
    assert events[2]["content"] == refusal_text("vi")


def test_frame_buffer_limit_refuses_without_emitting_buffered_content(monkeypatch):
    import src.api.query_guard as query_guard

    monkeypatch.setattr(query_guard, "MAX_BUFFER_CHARS", 80)
    events = decode_events(
        collect(
            guarded_sse(
                producer({"type": "meta", "padding": "x" * 100}),
                make_request(),
            )
        )
    )
    assert "padding" not in json.dumps(events)
    assert [event["type"] for event in events] == ["sources", "meta", "token", "done"]
    assert events[1]["answer_scope"] == "guardrail"


def test_cancellation_closes_upstream_iterator():
    class BlockingProducer:
        def __init__(self):
            self.closed = False
            self.started = asyncio.Event()
            self.wait = asyncio.Event()

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.started.set()
            await self.wait.wait()
            return encode_event({"type": "done"})

        async def aclose(self):
            self.closed = True

    async def scenario():
        source = BlockingProducer()
        guarded = guarded_sse(source, make_request())
        task = asyncio.create_task(guarded.__anext__())
        await source.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return source

    source = asyncio.run(scenario())
    assert source.closed


@pytest.mark.parametrize(
    "event",
    [
        {"type": "sources", "sources": [{"preview": DISCLOSURE}]},
        {"type": "meta", "raw_prompt": "hidden"},
        {"type": "artifact", "artifact": {"html": DISCLOSURE}},
    ],
)
def test_non_answer_payload_disclosure_blocks_entire_stream(event):
    events = decode_events(
        collect(
            guarded_sse(
                producer(event, {"type": "token", "content": "Otherwise safe."}, {"type": "done"}),
                make_request(),
            )
        )
    )
    assert DISCLOSURE not in json.dumps(events)
    assert "Otherwise safe." not in json.dumps(events)
    assert [item["type"] for item in events] == ["sources", "meta", "token", "done"]
    assert events[1]["answer_scope"] == "guardrail"


def test_safe_status_progress_is_immediate_and_unsafe_progress_is_suppressed():
    async def source():
        yield encode_event({"type": "status", "message": "safe progress"})
        yield encode_event({"type": "tool_result", "summary": DISCLOSURE})
        yield encode_event({"type": "token", "content": "safe final"})
        yield encode_event({"type": "done"})

    async def first_event():
        iterator = guarded_sse(source(), make_request())
        first = await iterator.__anext__()
        await iterator.aclose()
        return first

    first = asyncio.run(first_event())
    assert decode_events([first]) == [{"type": "status", "message": "safe progress"}]

    events = decode_events(collect(guarded_sse(source(), make_request())))
    assert not any(event.get("summary") == DISCLOSURE for event in events)
    assert events[-1]["type"] == "done"
    assert events[-2]["content"] == refusal_text("vi")


def test_qa_safe_error_contract_is_preserved_but_unsafe_error_is_generic():
    safe = decode_events(
        collect(guarded_sse(producer({"type": "error", "message": "QA service unavailable"}), make_request(mode="qa")))
    )
    assert safe == [{"type": "error", "message": "QA service unavailable"}]

    unsafe = decode_events(
        collect(guarded_sse(producer({"type": "error", "message": DISCLOSURE}), make_request(mode="qa")))
    )
    assert len(unsafe) == 1
    assert unsafe[0]["type"] == "error"
    assert DISCLOSURE not in json.dumps(unsafe)


def test_response_guard_removes_sources_metadata_and_artifact_on_payload_disclosure():
    response = QueryResponse(
        answer="A safe answer.",
        sources=[{"title": "safe", "details": DISCLOSURE}],
        session_id=SESSION_ID,
        model="safe-model",
        mode="qa",
        answer_scope="qa",
        artifact={"html": "safe artifact"},
    )
    guarded = guard_query_response(response, "ja")
    assert guarded.answer == refusal_text("ja")
    assert guarded.answer_scope == "guardrail"
    assert guarded.sources == []
    assert guarded.artifact is None
    assert guarded.qa_metadata is None


def test_query_decorator_guards_rest_and_stream_results():
    response = QueryResponse(
        answer=DISCLOSURE,
        sources=[],
        session_id=SESSION_ID,
        model="safe-model",
        mode="mkac",
        answer_scope="mkac",
    )

    async def rest_handler(req, request):
        return response

    guarded_rest = guarded_query(rest_handler)
    rest_result = asyncio.run(guarded_rest(make_request(), object()))
    assert rest_result.answer == refusal_text("vi")

    async def stream_handler(req, request):
        return StreamingResponse(producer({"type": "token", "content": DISCLOSURE}, {"type": "done"}))

    guarded_stream = guarded_query(stream_handler)
    stream_result = asyncio.run(guarded_stream(make_request(), object()))
    stream_events = decode_events(collect(stream_result.body_iterator))
    assert DISCLOSURE not in json.dumps(stream_events)
    assert stream_events[-1]["type"] == "done"
