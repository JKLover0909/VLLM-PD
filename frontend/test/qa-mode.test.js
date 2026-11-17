import assert from "node:assert/strict";
import test from "node:test";

import {
  buildQaPayload,
  countCodePoints,
  isValidQaQuestion,
  qaEffectiveQuestion,
  QA_MAX_CODE_POINTS,
} from "../src/qa-mode.js";

test("counts Unicode code points rather than UTF-16 code units", () => {
  assert.equal(countCodePoints("a😀漢"), 3);
});

test("accepts the inclusive QA question length boundary", () => {
  assert.equal(isValidQaQuestion("😀".repeat(QA_MAX_CODE_POINTS)), true);
  assert.equal(isValidQaQuestion("😀".repeat(QA_MAX_CODE_POINTS + 1)), false);
});

test("rejects blank QA questions", () => {
  assert.equal(isValidQaQuestion(" \n\t "), false);
});

test("builds a QA request that never carries browser chat history", () => {
  const payload = buildQaPayload({
    sessionId: "session-1",
    question: "  Phân tích lỗi  ",
    employeeId: "123456",
    language: "ja",
  });
  assert.equal(payload.mode, "qa");
  assert.equal(payload.question, "Phân tích lỗi");
  assert.equal(payload.ui_language, "ja");
  assert.equal(payload.employee_id, "123456");
  assert.deepEqual(payload.conversation_context, []);
  assert.equal(payload.research_topic, null);
  assert.equal(payload.research_scope, null);
  assert.equal(payload.quick_answer_id, null);
  assert.equal("history" in payload, false);
  assert.equal("context" in payload, false);
  assert.equal(payload.qa_reset_context, false);
});

test("asks the server to start a new QA thread only when requested", () => {
  const base = { sessionId: "s", question: "q", employeeId: "1", language: "vi" };
  assert.equal(buildQaPayload({ ...base, resetContext: true }).qa_reset_context, true);
  assert.equal(buildQaPayload({ ...base, resetContext: 0 }).qa_reset_context, false);
});

test("shows the effective question only when context rewrote it", () => {
  const rewritten = "Máy CNC-01 tuần trước có lỗi Method nào?";
  assert.equal(qaEffectiveQuestion(null, "q"), "");
  assert.equal(qaEffectiveQuestion({ context_applied: true, effective_question: "" }, "q"), "");
  assert.equal(
    qaEffectiveQuestion({ context_applied: true, effective_question: " Còn  lỗi Method? " }, "còn lỗi method?"),
    "",
  );
  // First question of a thread: any normalization is not a follow-up rewrite.
  assert.equal(qaEffectiveQuestion({ context_applied: false, effective_question: rewritten }, "Còn lỗi Method?"), "");
  assert.equal(qaEffectiveQuestion({ context_applied: true, effective_question: rewritten }, "Còn lỗi Method?"), rewritten);
});
