"""Contract tests for the Agent QMS 5M1E adapter (``mode="qa"``).

Only ``httpx.MockTransport`` is used: no network and no real job. A fake clock
and sleep make deadlines deterministic. The tests pin the safety claims of the
module docstring: one POST per question, no redirect/``poll_url`` following,
the key sent only to the configured host, bounded GET retries, one overall
deadline, zero outbound requests when disabled and allowlisted citations.
"""

import asyncio
import json
import logging
import re

import httpx
import pytest

from src.api.schemas import QaAnswerMetadata
from src.integrations.qa_agent_service import (
    MAX_ANSWER_CHARS,
    MAX_CITATIONS,
    MAX_RESPONSE_BYTES,
    QA_ERRORS,
    QA_MAX_QUESTION_CHARS,
    QaAgentError,
    QaAgentService,
    qa_error_detail,
    sanitize_citations,
    validate_qa_question,
)

BASE_URL = "http://qa-agent.test:8600"
API_KEY = "test-key-not-a-real-secret"
JOB_ID = "job_0123456789abcdef"
QUESTION = "Máy CNC-01 tuần trước có bao nhiêu lỗi thuộc nhóm Machine?"
ANSWER = "Tuần trước máy CNC-01 ghi nhận 3 lỗi thuộc nhóm Machine."
CLIENT_REF_PATTERN = re.compile(r"^meibook-[0-9a-f]{16}$")

DONE_RESULT = {
    "answer": ANSWER,
    "elapsed_ms": 1234,
    "citations": [
        {
            "type": "sql",
            "views": ["V_QMS_DEFECT"],
            "keys": {"LOT_NO": ["L001", 12]},
            "sql": "SELECT * FROM SECRET_TABLE",
        }
    ],
    "sqls": ["SELECT * FROM SECRET_TABLE"],
}


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeAgent:
    """Scripted upstream.

    Each queue holds ``(request, agent) -> Response`` factories; the last entry
    repeats so "always running" scripts stay short.
    """

    def __init__(self, *, post=None, get=None, health=None):
        self.post = list(post or [])
        self.get = list(get or [])
        self.health = list(health or [])
        self.requests = []
        self.client_ref = None

    def handler(self, request):
        self.requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/v1/jobs":
            self.client_ref = json.loads(request.content)["client_ref"]
            queue = self.post
        elif request.method == "GET" and path == "/health":
            queue = self.health
        elif request.method == "GET" and path.startswith("/v1/jobs/"):
            queue = self.get
        else:  # pragma: no cover - a failing assertion explains itself
            raise AssertionError(
                f"unexpected request {request.method} {request.url}"
            )
        if not queue:  # pragma: no cover
            raise AssertionError(
                f"no scripted response for {request.method} {path}"
            )

        factory = queue.pop(0) if len(queue) > 1 else queue[0]
        return factory(request, self)

    def of(self, method, path_prefix="/"):
        return [
            r for r in self.requests
            if r.method == method and r.url.path.startswith(path_prefix)
        ]


class NoNetwork:
    """Transport handler for services that must never send a request."""

    def __init__(self):
        self.requests = []

    def handler(self, request):  # pragma: no cover - must not run
        self.requests.append(request)
        raise AssertionError(f"unexpected outbound request {request.url}")


def respond(status, payload=None, *, headers=None, content=None):
    def make(request, agent):
        if content is not None:
            return httpx.Response(status, content=content, headers=headers)
        return httpx.Response(status, json=payload, headers=headers)

    return make


def fail(exc_type):
    def make(request, agent):
        raise exc_type("scripted failure", request=request)

    return make


def accepted(job_id=JOB_ID, status="queued", **extra):
    return respond(
        202,
        {
            "job_id": job_id,
            "status": status,
            "poll_url": f"/v1/jobs/{job_id}",
            **extra,
        },
    )


def job(status, *, job_id=JOB_ID, result=None, client_ref="echo", **extra):
    def make(request, agent):
        payload = {
            "job_id": job_id,
            "status": status,
            "created_at": "2026-10-27T08:00:00+07:00",
            "started_at": "2026-10-27T08:00:01+07:00",
            "finished_at": (
                "2026-10-27T08:00:12+07:00" if status == "done" else None
            ),
            **extra,
        }
        if client_ref == "echo":
            payload["client_ref"] = agent.client_ref
        elif client_ref is not None:
            payload["client_ref"] = client_ref
        if result is not None:
            payload["result"] = result
        return httpx.Response(200, json=payload)

    return make


def make_service(agent, clock=None, **overrides):
    clock = clock or FakeClock()
    options = {
        "enabled": True,
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "request_timeout_seconds": 5.0,
        "total_timeout_seconds": 180.0,
        "poll_interval_seconds": 1.5,
        "max_inflight": 2,
        "health_ttl_seconds": 30.0,
        "transport": httpx.MockTransport(agent.handler),
        "clock": clock,
        "sleep": clock.sleep,
    }
    options.update(overrides)
    return QaAgentService(**options), clock


def ask(service, question=QUESTION, **kwargs):
    async def scenario():
        try:
            return await service.ask(question, **kwargs)
        finally:
            await service.close()

    return asyncio.run(scenario())


def ask_error(service, question=QUESTION):
    with pytest.raises(QaAgentError) as info:
        ask(service, question)
    return info.value


# --------------------------------------------------------------- happy paths
def test_queued_running_done_returns_answer_and_safe_metadata():
    agent = FakeAgent(
        post=[accepted()],
        get=[job("queued"), job("running"), job("done", result=DONE_RESULT)],
    )
    service, clock = make_service(agent)
    progress = []

    async def on_progress(status):
        progress.append(status)

    result = ask(service, on_progress=on_progress)

    assert result.answer == ANSWER
    assert [s for s in progress if s != "waiting"] == ["queued", "running"]
    assert len(agent.of("POST")) == 1
    assert len(agent.of("GET", "/v1/jobs/")) == 3
    assert clock.sleeps == [1.5, 1.5]

    metadata = QaAnswerMetadata.model_validate(result.metadata())
    assert metadata.provider == "qms_5m1e_agent"
    assert metadata.answer_language == "vi"
    # A 1.0-style result (no context) leaves the follow-up flags off.
    assert metadata.context_supported is False
    assert metadata.context_applied is False
    assert metadata.clarification is False
    assert metadata.effective_question == ""
    assert result.context is None and result.history is None
    assert CLIENT_REF_PATTERN.fullmatch(metadata.client_ref)
    assert metadata.elapsed_ms == 1234.0
    assert metadata.finished_at == "2026-10-27T08:00:12+07:00"
    citations = [
        citation.model_dump(exclude_defaults=True)
        for citation in metadata.citations
    ]
    assert citations == [
        {
            "type": "sql",
            "views": ["V_QMS_DEFECT"],
            "keys": {"LOT_NO": ["L001", 12]},
        }
    ]
    serialized = json.dumps(result.metadata(), ensure_ascii=False)
    assert "sqls" not in serialized
    assert "SECRET_TABLE" not in serialized


def test_done_on_first_poll_does_not_sleep():
    agent = FakeAgent(post=[accepted()], get=[job("done", result=DONE_RESULT)])
    service, clock = make_service(agent)

    assert ask(service).answer == ANSWER
    assert clock.sleeps == []


def test_one_shot_ask_sends_only_question_and_opaque_ref_to_the_configured_host():
    agent = FakeAgent(post=[accepted()], get=[job("done", result=DONE_RESULT)])
    service, _ = make_service(agent)

    ask(service, question=f"  {QUESTION}\n")

    post = agent.of("POST")[0]
    body = json.loads(post.content)
    assert set(body) == {"question", "client_ref"}
    assert body["question"] == QUESTION
    assert CLIENT_REF_PATTERN.fullmatch(body["client_ref"])
    assert agent.of("GET")[0].url.path == f"/v1/jobs/{JOB_ID}"
    for request in agent.requests:
        assert (request.url.host, request.url.port) == ("qa-agent.test", 8600)
        assert request.headers["X-API-Key"] == API_KEY


# ------------------------------------------------------- follow-up (API 1.1)
FOLLOW_UP_CONTEXT = {
    "equipment_codes": ["CNC-01"],
    "time_range": {"from": "2026-10-20", "to": "2026-10-26"},
}
FOLLOW_UP_HISTORY = [
    {"role": "user", "content": QUESTION},
    {"role": "assistant", "content": ANSWER},
]
FOLLOW_UP_RESULT = {
    **DONE_RESULT,
    "clarification": False,
    "effective_question": "Máy CNC-01 từ 2026-10-20 đến 2026-10-26 có lỗi Method nào?",
    "context": FOLLOW_UP_CONTEXT,
    "history": FOLLOW_UP_HISTORY,
}


def test_first_follow_up_question_sends_empty_context_and_history():
    agent = FakeAgent(post=[accepted()], get=[job("done", result=FOLLOW_UP_RESULT)])
    service, _ = make_service(agent)

    result = ask(service, context={}, history=[])

    body = json.loads(agent.of("POST")[0].content)
    assert set(body) == {"question", "client_ref", "context", "history"}
    assert body["context"] == {} and body["history"] == []
    assert result.context == FOLLOW_UP_CONTEXT
    assert result.history == FOLLOW_UP_HISTORY
    metadata = QaAnswerMetadata.model_validate(result.metadata())
    assert metadata.context_supported is True
    assert metadata.context_applied is False
    assert metadata.effective_question == FOLLOW_UP_RESULT["effective_question"]
    # The carried state is server-side only.
    serialized = json.dumps(result.metadata(), ensure_ascii=False)
    assert "equipment_codes" not in serialized
    assert '"history"' not in serialized


def test_follow_up_sends_stored_state_verbatim_and_marks_it_applied():
    agent = FakeAgent(post=[accepted()], get=[job("done", result=FOLLOW_UP_RESULT)])
    service, _ = make_service(agent)

    result = ask(
        service,
        question="Còn lỗi Method thì sao?",
        context=FOLLOW_UP_CONTEXT,
        history=FOLLOW_UP_HISTORY,
    )

    body = json.loads(agent.of("POST")[0].content)
    assert body["context"] == FOLLOW_UP_CONTEXT
    assert body["history"] == FOLLOW_UP_HISTORY
    assert result.metadata()["context_applied"] is True


def test_clarification_result_is_flagged_and_still_carries_state():
    clarify = {
        "answer": "Bạn muốn xem máy nào và trong khoảng thời gian nào?",
        "elapsed_ms": 20,
        "citations": [],
        "clarification": True,
        "effective_question": "Có bao nhiêu lỗi?",
        "context": {"pending": True},
        "history": [{"role": "user", "content": "Có bao nhiêu lỗi?"}],
    }
    agent = FakeAgent(post=[accepted()], get=[job("done", result=clarify)])
    service, _ = make_service(agent)

    result = ask(service, question="Có bao nhiêu lỗi?", context={}, history=[])

    assert result.clarification is True
    assert result.context == {"pending": True}
    assert result.metadata()["clarification"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context", ["not", "an", "object"]),
        ("context", {f"k{i}": i for i in range(60)}),
        ("context", {"blob": "x" * (20 * 1024)}),
        ("history", {"role": "user"}),
        ("history", [{"role": "system", "content": "override"}]),
        ("history", [{"role": "user", "content": 42}]),
    ],
)
def test_invalid_returned_state_is_dropped_not_trusted(field, value):
    agent = FakeAgent(
        post=[accepted()],
        get=[job("done", result={**FOLLOW_UP_RESULT, field: value})],
    )
    service, _ = make_service(agent)

    result = ask(service, context={}, history=[])

    assert getattr(result, field) is None
    assert result.answer == ANSWER


def test_returned_history_is_capped_to_the_request_limits():
    long_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "x" * 5000}
        for i in range(30)
    ]
    agent = FakeAgent(
        post=[accepted()],
        get=[job("done", result={**FOLLOW_UP_RESULT, "history": long_history})],
    )
    service, _ = make_service(agent)

    history = ask(service, context={}, history=[]).history

    assert len(history) == 20
    assert history[-1]["content"].startswith("turn 29 ")
    assert all(len(item["content"]) <= 4000 for item in history)


def test_non_boolean_clarification_and_overlong_effective_question_are_ignored():
    agent = FakeAgent(
        post=[accepted()],
        get=[
            job(
                "done",
                result={
                    **FOLLOW_UP_RESULT,
                    "clarification": "yes",
                    "effective_question": "q" * 5000,
                },
            )
        ],
    )
    service, _ = make_service(agent)

    result = ask(service, context={}, history=[])

    assert result.clarification is False
    assert result.effective_question == ""


def test_rejected_follow_up_state_maps_to_context_reset():
    agent = FakeAgent(post=[respond(422, {"detail": "invalid context"})])
    service, _ = make_service(agent)

    with pytest.raises(QaAgentError) as info:
        ask(service, context=FOLLOW_UP_CONTEXT, history=FOLLOW_UP_HISTORY)

    assert info.value.code == "QA_CONTEXT_RESET"
    assert info.value.http_status == 422
    assert len(agent.of("POST")) == 1


def test_client_refs_are_unique_per_question():
    agent = FakeAgent(post=[accepted()], get=[job("done", result=DONE_RESULT)])
    service, _ = make_service(agent)

    async def scenario():
        try:
            await service.ask(QUESTION)
            await service.ask(QUESTION)
        finally:
            await service.close()

    asyncio.run(scenario())

    refs = [json.loads(r.content)["client_ref"] for r in agent.of("POST")]
    assert len(set(refs)) == 2


def test_poll_url_and_redirects_are_never_followed():
    agent = FakeAgent(
        post=[accepted(poll_url="http://evil.test/steal")],
        get=[respond(302, headers={"Location": "http://evil.test/steal"})],
    )
    service, _ = make_service(agent)

    assert ask_error(service).code == "QA_BAD_RESPONSE"
    assert len(agent.of("GET")) == 1
    assert {r.url.host for r in agent.requests} == {"qa-agent.test"}


# ------------------------------------------------------------- submit errors
@pytest.mark.parametrize(
    ("status", "content", "code"),
    [
        (401, None, "QA_UPSTREAM_AUTH"),
        (403, None, "QA_UPSTREAM_AUTH"),
        (422, None, "QA_INVALID_QUESTION"),
        (500, b"<html>Internal Server Error</html>", "QA_SUBMIT_UNKNOWN"),
        (502, None, "QA_SUBMIT_UNKNOWN"),
        (200, None, "QA_BAD_RESPONSE"),
        (418, None, "QA_BAD_RESPONSE"),
    ],
)
def test_submit_http_errors_are_mapped_and_never_resubmitted(status, content, code):
    agent = FakeAgent(
        post=[
            respond(
                status,
                {"detail": "upstream detail"},
                content=content,
            )
        ]
    )
    service, _ = make_service(agent)

    error = ask_error(service)

    assert error.code == code
    assert len(agent.requests) == 1
    assert "upstream detail" not in error.message("vi")


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("7", 7),
        ("99999", 600),
        ("Wed, 21 Oct 2026 07:28:00 GMT", None),
        (None, None),
    ],
)
def test_submit_busy_passes_only_a_numeric_retry_hint(header, expected):
    headers = {"Retry-After": header} if header is not None else None
    agent = FakeAgent(post=[respond(503, {"detail": "busy"}, headers=headers)])
    service, _ = make_service(agent)

    error = ask_error(service)

    assert error.code == "QA_BUSY"
    assert error.retry_after == expected
    detail = qa_error_detail(error, "vi")
    assert ("retry_after" in detail) is (expected is not None)


def test_submit_connect_error_means_offline():
    agent = FakeAgent(post=[fail(httpx.ConnectError)])
    service, _ = make_service(agent)

    assert ask_error(service).code == "QA_OFFLINE"
    assert len(agent.requests) == 1


@pytest.mark.parametrize("exc_type", [httpx.ReadTimeout, httpx.RemoteProtocolError])
def test_submit_with_unknown_outcome_is_reported_not_retried(exc_type):
    agent = FakeAgent(post=[fail(exc_type)])
    service, _ = make_service(agent)

    assert ask_error(service).code == "QA_SUBMIT_UNKNOWN"
    assert len(agent.requests) == 1


@pytest.mark.parametrize(
    "factory",
    [
        accepted(job_id="../../admin"),
        accepted(job_id="short"),
        respond(202, {"job_id": 12345678, "status": "queued"}),
        accepted(status="pending"),
        respond(202, ["not", "an", "object"]),
        respond(202, content=b"not json"),
    ],
)
def test_invalid_submit_payload_is_rejected_before_polling(factory):
    agent = FakeAgent(post=[factory])
    service, _ = make_service(agent)

    assert ask_error(service).code == "QA_BAD_RESPONSE"
    assert agent.of("GET") == []


# --------------------------------------------------------------- poll errors
def test_lost_job_is_reported_and_not_resubmitted():
    agent = FakeAgent(
        post=[accepted()],
        get=[respond(404, {"detail": "not found"})],
    )
    service, _ = make_service(agent)

    assert ask_error(service).code == "QA_JOB_LOST"
    assert len(agent.of("POST")) == 1
    assert len(agent.of("GET")) == 1


def test_poll_auth_failure_is_not_retried():
    agent = FakeAgent(
        post=[accepted()],
        get=[respond(401, {"detail": "bad key"})],
    )
    service, _ = make_service(agent)

    assert ask_error(service).code == "QA_UPSTREAM_AUTH"
    assert len(agent.of("GET")) == 1


def test_agent_error_status_does_not_forward_upstream_detail():
    agent = FakeAgent(
        post=[accepted()],
        get=[job("error", error="ORA-00942: table or view does not exist")],
    )
    service, _ = make_service(agent)

    error = ask_error(service)

    assert error.code == "QA_AGENT_ERROR"
    detail = json.dumps(qa_error_detail(error, "vi"), ensure_ascii=False)
    assert "ORA-" not in detail


def test_get_retries_are_bounded_then_offline():
    agent = FakeAgent(
        post=[accepted()],
        get=[
            respond(500, content=b"oops"),
            fail(httpx.ReadTimeout),
            fail(httpx.ConnectError),
        ],
    )
    service, _ = make_service(agent)

    assert ask_error(service).code == "QA_OFFLINE"
    assert len(agent.of("GET")) == 3
    assert len(agent.of("POST")) == 1


def test_get_retry_counter_resets_after_a_good_poll():
    agent = FakeAgent(
        post=[accepted()],
        get=[
            respond(503, content=b""),
            respond(500, content=b""),
            job("running"),
            fail(httpx.ReadTimeout),
            fail(httpx.ReadTimeout),
            job("done", result=DONE_RESULT),
        ],
    )
    service, _ = make_service(agent)

    assert ask(service).answer == ANSWER
    assert len(agent.of("GET")) == 6
    assert len(agent.of("POST")) == 1


def test_one_deadline_covers_queue_processing_and_each_call_timeout():
    agent = FakeAgent(post=[accepted()], get=[job("queued"), job("running")])
    service, clock = make_service(agent, total_timeout_seconds=10.0)

    assert ask_error(service).code == "QA_TIMEOUT"

    assert sum(clock.sleeps) == pytest.approx(10.0)
    assert max(clock.sleeps) <= 1.5
    assert len(agent.of("POST")) == 1
    # Each HTTP call is bounded by min(request timeout, time left).
    post_timeout = agent.of("POST")[0].extensions["timeout"]["read"]
    final_get_timeout = agent.of("GET")[-1].extensions["timeout"]["read"]
    assert post_timeout == pytest.approx(5.0)
    assert final_get_timeout == pytest.approx(1.0)


def oversized_poll(request, agent):
    padding = b"x" * MAX_RESPONSE_BYTES
    body = (
        b'{"job_id": "'
        + JOB_ID.encode()
        + b'", "status": "done", "pad": "'
        + padding
        + b'"}'
    )
    return httpx.Response(200, content=body)


@pytest.mark.parametrize(
    "factory",
    [
        job("running", job_id="job_other_0123456789"),
        job("paused"),
        respond(200, content=b"<html>proxy page</html>"),
        job("done", result=DONE_RESULT, client_ref="meibook-ffffffffffffffff"),
        job("done"),
        job("done", result={"answer": "   "}),
        job("done", result={"answer": 42}),
        job("done", result={"answer": "x" * (MAX_ANSWER_CHARS + 1)}),
        oversized_poll,
    ],
)
def test_invalid_poll_payload_is_a_bad_response(factory):
    agent = FakeAgent(post=[accepted()], get=[factory])
    service, _ = make_service(agent)

    assert ask_error(service).code == "QA_BAD_RESPONSE"


@pytest.mark.parametrize("elapsed_literal", [b"-1", b"NaN", b'"12"', b"true"])
def test_invalid_elapsed_time_is_dropped_not_fatal(elapsed_literal):
    # Raw bytes because httpx refuses to encode NaN; Python's json accepts it.
    body = (
        b'{"job_id": "'
        + JOB_ID.encode()
        + b'", "status": "done", "result": {"answer": "ok", '
        + b'"elapsed_ms": '
        + elapsed_literal
        + b"}}"
    )
    agent = FakeAgent(post=[accepted()], get=[respond(200, content=body)])
    service, _ = make_service(agent)

    result = ask(service)

    assert result.answer == "ok"
    assert result.elapsed_ms is None


# ----------------------------------------------------------------- citations
def test_citations_are_allowlisted():
    raw = [
        {
            "type": "sql",
            "views": ["V_QMS_DEFECT"],
            "keys": {"LOT_NO": ["L001", 12, True]},
            "sql": "SELECT * FROM SECRET_TABLE",
            "rows": [{"secret": 1}],
        },
        {
            "type": "semantic",
            "views": ["V_QMS_NOTE"],
            "inspection_keys": [101, "IK-2"],
            "chunks": [
                {
                    "id": 5,
                    "source_view": "V_QMS_NOTE",
                    "source_key": "K1",
                    "inspection_key": 101,
                    "text": "raw chunk text",
                }
            ],
        },
        {"type": "shell", "cmd": "rm -rf /"},
    ]

    cleaned, truncated = sanitize_citations(raw)

    assert truncated is True
    assert cleaned == [
        {
            "type": "sql",
            "views": ["V_QMS_DEFECT"],
            "keys": {"LOT_NO": ["L001", 12]},
        },
        {
            "type": "semantic",
            "views": ["V_QMS_NOTE"],
            "keys": {},
            "inspection_keys": [101, "IK-2"],
            "chunks": [
                {
                    "id": 5,
                    "source_view": "V_QMS_NOTE",
                    "source_key": "K1",
                    "inspection_key": 101,
                }
            ],
        },
    ]
    serialized = json.dumps(cleaned)
    for leaked in ("SECRET_TABLE", "raw chunk text", "rm -rf", "secret"):
        assert leaked not in serialized
    QaAnswerMetadata.model_validate({"citations": cleaned})


def test_citations_with_unsafe_identifiers_are_dropped_and_flagged():
    cleaned, truncated = sanitize_citations(
        [
            {
                "type": "row",
                "views": ["V_OK", "V; DROP TABLE X"],
                "keys": {"bad key": [1]},
            }
        ]
    )

    assert truncated is True
    assert cleaned == [{"type": "row", "views": ["V_OK"], "keys": {}}]


def test_citations_are_capped():
    raw = [{"type": "row", "views": [f"V_{i}"]} for i in range(MAX_CITATIONS + 5)]

    cleaned, truncated = sanitize_citations(raw)

    assert len(cleaned) == MAX_CITATIONS
    assert truncated is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, ([], False)), ("x", ([], True)), ([], ([], False))],
)
def test_missing_or_malformed_citation_lists(raw, expected):
    assert sanitize_citations(raw) == expected


# ------------------------------------------------- validation and gating
@pytest.mark.parametrize(
    "question",
    [
        "a" * QA_MAX_QUESTION_CHARS,
        "品" * QA_MAX_QUESTION_CHARS,
        "😀" * QA_MAX_QUESTION_CHARS,
    ],
)
def test_question_limit_counts_unicode_code_points(question):
    assert validate_qa_question(f"  {question}  ") == question


@pytest.mark.parametrize(
    "question",
    ["", "   \n\t", "a" * (QA_MAX_QUESTION_CHARS + 1), None],
)
def test_invalid_questions_never_reach_the_agent(question):
    network = NoNetwork()
    service, _ = make_service(network)

    assert ask_error(service, question).code == "QA_INVALID_QUESTION"
    assert network.requests == []


def test_disabled_service_makes_zero_outbound_requests():
    network = NoNetwork()
    service, _ = make_service(network, enabled=False)

    assert ask_error(service).code == "QA_DISABLED"
    asyncio.run(service.refresh_health())

    assert network.requests == []
    assert service.status()["state"] == "DISABLED"
    assert service.status()["configured"] is False


@pytest.mark.parametrize(
    ("base_url", "api_key"),
    [
        ("", API_KEY),
        (BASE_URL, ""),
        ("ftp://qa-agent.test:8600", API_KEY),
        ("http://user:pw@qa-agent.test:8600", API_KEY),
        ("http://qa-agent.test:8600/?next=evil", API_KEY),
        ("not a url", API_KEY),
    ],
)
def test_misconfigured_service_makes_zero_outbound_requests(base_url, api_key):
    network = NoNetwork()
    service, _ = make_service(network, base_url=base_url, api_key=api_key)

    assert service.configured is False
    assert ask_error(service).code == "QA_MISCONFIGURED"
    asyncio.run(service.refresh_health())
    assert network.requests == []
    assert service.status()["state"] == "MISCONFIGURED"


# -------------------------------------------------------------------- health
def test_health_probe_states_and_status_never_expose_url_or_key():
    agent = FakeAgent(
        health=[
            respond(200, {"status": "ok"}),
            fail(httpx.ConnectError),
            respond(500, content=b""),
        ]
    )
    service, clock = make_service(agent)

    async def scenario():
        states = [service.state]
        await service.refresh_health()
        states.append(service.state)
        clock.now += 30.0
        states.append(service.state)
        await service.refresh_health()
        states.append(service.state)
        await service.refresh_health()
        states.append(service.state)
        await service.close()
        return states

    assert asyncio.run(scenario()) == [
        "UNKNOWN",
        "ONLINE",
        "UNKNOWN",
        "OFFLINE",
        "OFFLINE",
    ]
    assert all(r.url.path == "/health" for r in agent.requests)
    assert all("X-API-Key" not in r.headers for r in agent.requests)
    assert service.health_interval_seconds == 15.0
    assert service._http().trust_env is False
    assert service._http().follow_redirects is False

    status = service.status()
    assert set(status) == {
        "enabled",
        "configured",
        "state",
        "answer_language",
        "supports_history",
        "max_question_chars",
    }
    assert status["supports_history"] is True
    assert status["max_question_chars"] == QA_MAX_QUESTION_CHARS
    serialized = json.dumps(status)
    for secret in ("qa-agent.test", "8600", API_KEY):
        assert secret not in serialized


def test_from_env_clamps_values_and_defaults_to_disabled(monkeypatch):
    for name in (
        "QA_AGENT_ENABLED",
        "QA_AGENT_URL",
        "QA_AGENT_API_KEY",
        "QA_AGENT_HEALTH_TTL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    default = QaAgentService.from_env()
    assert default.enabled is False
    assert default.status()["state"] == "DISABLED"

    monkeypatch.setenv("QA_AGENT_ENABLED", "true")
    monkeypatch.setenv("QA_AGENT_URL", BASE_URL + "/")
    monkeypatch.setenv("QA_AGENT_API_KEY", API_KEY)
    monkeypatch.setenv("QA_AGENT_HEALTH_TTL_SECONDS", "1")
    monkeypatch.setenv("QA_AGENT_TIMEOUT_SECONDS", "not-a-number")

    configured = QaAgentService.from_env()
    assert configured.enabled is True
    assert configured.configured is True
    assert configured.status()["state"] == "UNKNOWN"
    assert configured.health_interval_seconds == 2.5


# ------------------------------------------------------ concurrency / slots
def test_busy_when_inflight_limit_is_reached_and_slot_is_released():
    agent = FakeAgent(
        post=[accepted()],
        get=[job("running"), job("done", result=DONE_RESULT)],
    )
    clock = FakeClock()

    async def scenario():
        gate = asyncio.Event()
        entered = asyncio.Event()

        async def gated_sleep(seconds):
            entered.set()
            await gate.wait()
            clock.now += seconds

        service, _ = make_service(agent, clock, max_inflight=1, sleep=gated_sleep)
        try:
            first = asyncio.create_task(service.ask(QUESTION))
            await entered.wait()
            with pytest.raises(QaAgentError) as busy:
                await service.ask("Câu hỏi thứ hai")
            posts_while_busy = len(agent.of("POST"))
            gate.set()
            first_result = await first
            second_result = await service.ask(QUESTION)
        finally:
            await service.close()
        return busy.value, posts_while_busy, first_result, second_result

    busy, posts_while_busy, first_result, second_result = asyncio.run(scenario())

    assert busy.code == "QA_BUSY"
    assert busy.retry_after is None
    assert posts_while_busy == 1
    assert first_result.answer == second_result.answer == ANSWER
    assert len(agent.of("POST")) == 2


def test_slot_is_released_after_an_error():
    agent = FakeAgent(
        post=[accepted()],
        get=[respond(404, {}), job("done", result=DONE_RESULT)],
    )
    service, _ = make_service(agent, max_inflight=1)

    async def scenario():
        try:
            with pytest.raises(QaAgentError):
                await service.ask(QUESTION)
            return await service.ask(QUESTION)
        finally:
            await service.close()

    assert asyncio.run(scenario()).answer == ANSWER


def test_cancellation_stops_waiting_releases_slot_and_never_resubmits():
    agent = FakeAgent(
        post=[accepted()],
        get=[job("running"), job("done", result=DONE_RESULT)],
    )
    clock = FakeClock()
    sleeping = []

    async def scenario():
        entered = asyncio.Event()

        async def blocking_sleep(seconds):
            sleeping.append(seconds)
            entered.set()
            if len(sleeping) == 1:
                await asyncio.Event().wait()
            clock.now += seconds

        service, _ = make_service(agent, clock, max_inflight=1, sleep=blocking_sleep)
        try:
            task = asyncio.create_task(service.ask(QUESTION))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            posts_after_cancel = len(agent.of("POST"))
            result = await service.ask(QUESTION)
        finally:
            await service.close()
        return posts_after_cancel, result

    posts_after_cancel, result = asyncio.run(scenario())

    assert posts_after_cancel == 1
    assert result.answer == ANSWER
    assert len(agent.of("POST")) == 2


# ------------------------------------------------------------------- logging
def test_logs_never_contain_question_answer_or_key(caplog):
    agent = FakeAgent(post=[accepted()], get=[job("done", result=DONE_RESULT)])
    service, _ = make_service(agent)
    failing = FakeAgent(post=[respond(503, {"detail": QUESTION})])
    failing_service, _ = make_service(failing)

    with caplog.at_level(logging.DEBUG):
        ask(service)
        ask_error(failing_service)

    text = caplog.text
    assert "qa_job_complete" in text
    assert "outcome=done" in text
    assert "outcome=QA_BUSY" in text
    for secret in (QUESTION, ANSWER, API_KEY, "SECRET_TABLE"):
        assert secret not in text


# ----------------------------------------------------------- error messages
def test_every_error_code_has_localized_messages_and_a_gateway_status():
    for code, (status, vi, ja) in QA_ERRORS.items():
        assert status in {403, 422, 502, 503, 504}, code
        assert vi and ja and vi != ja, code

    error = QaAgentError("QA_BUSY", retry_after=5)
    assert qa_error_detail(error, "ja") == {
        "code": "QA_BUSY",
        "message": QA_ERRORS["QA_BUSY"][2] + "（5秒後に再試行できます）",
        "retry_after": 5,
    }
    assert qa_error_detail(error, "vi")["message"] == (
        QA_ERRORS["QA_BUSY"][1] + " (Có thể thử lại sau 5 giây.)"
    )
    assert QaAgentError("QA_BUSY").message("vi") == QA_ERRORS["QA_BUSY"][1]
    assert QaAgentError("SOMETHING_ELSE").code == "QA_BAD_RESPONSE"
