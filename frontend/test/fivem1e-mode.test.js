import assert from "node:assert/strict";
import test from "node:test";

import {
  buildFivem1ePayload,
  canSubmitFivem1eQuestion,
  initialFivem1eState,
  reduceFivem1eEvent,
  resolveFivem1ePresence,
  shouldApplyFivem1eEvent,
  stopFivem1eQuery,
} from "../src/fivem1e-mode.js";

test("rejects 5M1E questions without an employee for every entry path", () => {
  assert.equal(canSubmitFivem1eQuestion({ question: "starter", employeeId: "" }), false);
  assert.equal(canSubmitFivem1eQuestion({ question: "free input", employeeId: undefined }), false);
  assert.equal(canSubmitFivem1eQuestion({ question: "valid", employeeId: "E1" }), true);
});

test("builds authenticated 5M1E payload without undefined employee IDs", () => {
  assert.deepEqual(
    buildFivem1ePayload({ sessionId: "s1", question: "  tiến độ  ", employeeId: "E1" }),
    { session_id: "s1", question: "tiến độ", employee_id: "E1" },
  );
  assert.equal("employee_id" in buildFivem1ePayload({ sessionId: "s1", question: "q" }), false);
});

test("applies stream events only to their active turn and controller", () => {
  const controller = new AbortController();
  const otherController = new AbortController();
  assert.equal(shouldApplyFivem1eEvent({ turnId: "t2", activeTurnId: "t2", controller, activeController: controller }), true);
  assert.equal(shouldApplyFivem1eEvent({ turnId: "t1", activeTurnId: "t2", controller, activeController: controller }), false);
  assert.equal(shouldApplyFivem1eEvent({ turnId: "t2", activeTurnId: "t2", controller, activeController: otherController }), false);
});

test("reduces terminal 5M1E events and preserves backend error codes", () => {
  let state = reduceFivem1eEvent(initialFivem1eState(), { type: "error", code: "LLM_BUSY", message: "Busy" });
  assert.deepEqual(state.error, { code: "LLM_BUSY", message: "Busy" });
  state = reduceFivem1eEvent(state, { type: "done", result: { answer: "done" } });
  assert.equal(state.done, true);
  assert.equal(state.answer, "done");
});

test("posts 5M1E stop with session and employee without blocking UI", async () => {
  const seen = [];
  const result = await stopFivem1eQuery(
    { sessionId: "s1", employeeId: "E1" },
    async (url, options) => {
      seen.push({ url, body: JSON.parse(options.body), method: options.method });
      return { ok: true };
    },
  );
  assert.equal(result, true);
  assert.deepEqual(seen, [{
    url: "/fivem1e/query/stop",
    body: { session_id: "s1", employee_id: "E1" },
    method: "POST",
  }]);
});

test("maps 5M1E stream state to robot presence", () => { assert.equal(resolveFivem1ePresence({ state: "ready" }, { streaming: true }).state, "working"); assert.equal(resolveFivem1ePresence({ state: "ready" }, { error: true }).state, "error"); });
