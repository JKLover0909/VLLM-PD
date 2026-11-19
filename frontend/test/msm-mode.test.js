import assert from "node:assert/strict";
import test from "node:test";

import {
  buildMsmPayload,
  initialMsmState,
  reduceMsmEvent,
} from "../src/msm-mode.js";

test("buildMsmPayload trims the question and keeps the session", () => {
  assert.deepEqual(
    buildMsmPayload({ sessionId: "s1", question: "  máy nào quá hạn  ", employeeId: "E1" }),
    { session_id: "s1", question: "máy nào quá hạn", employee_id: "E1" },
  );
});

test("reduceMsmEvent collects tokens, then pending and views from done", () => {
  let state = initialMsmState();
  state = reduceMsmEvent(state, { type: "token", content: "Có 3 máy. " });
  state = reduceMsmEvent(state, { type: "token", content: "Chi tiết bên dưới." });
  state = reduceMsmEvent(state, {
    type: "done",
    result: { pending: [{ action_id: "a1" }], views: [{ kind: "table" }] },
  });
  assert.equal(state.answer, "Có 3 máy. Chi tiết bên dưới.");
  assert.equal(state.pending.length, 1);
  assert.equal(state.views.length, 1);
  assert.equal(state.done, true);
});

test("reduceMsmEvent surfaces an error without dropping the answer", () => {
  let state = reduceMsmEvent(initialMsmState(), { type: "token", content: "một phần" });
  state = reduceMsmEvent(state, { type: "error", code: "LLM_BUSY", message: "Hệ thống đang bận" });
  assert.equal(state.answer, "một phần");
  assert.deepEqual(state.error, { code: "LLM_BUSY", message: "Hệ thống đang bận" });
});
