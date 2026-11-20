import assert from "node:assert/strict";
import test from "node:test";

import {
  buildMsmPayload,
  initialMsmState,
  reduceMsmEvent,
  submitMsmForm,
  lookupMsmCategory,
  shouldSubmitOnEnter,
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

test("reduceMsmEvent collects forms and pills from done", () => {
  let state = initialMsmState();
  state = reduceMsmEvent(state, {
    type: "done",
    result: {
      forms: [{ kind: "create_machine", schema: { title: "Thêm máy" } }],
      pills: ["Lịch sử hiệu chuẩn May11", "Kế hoạch hiệu chuẩn May11"],
    },
  });
  assert.equal(state.forms.length, 1);
  assert.equal(state.forms[0].kind, "create_machine");
  assert.deepEqual(state.pills, ["Lịch sử hiệu chuẩn May11", "Kế hoạch hiệu chuẩn May11"]);
});

test("submitMsmForm sends form payload and returns pending result", async () => {
  const seen = [];
  const fakeFetch = async (url, options) => {
    seen.push({ url, body: JSON.parse(options.body) });
    return {
      ok: true,
      json: async () => ({ pending: [{ action_id: "act-123" }] }),
    };
  };
  const res = await submitMsmForm({
    sessionId: "s1",
    employeeId: "E1",
    kind: "create_machine",
    values: { machine_id: "TEST-01" },
  }, fakeFetch);
  assert.deepEqual(res, { pending: [{ action_id: "act-123" }] });
  assert.equal(seen[0].url, "/msm/forms/submit");
  assert.equal(seen[0].body.kind, "create_machine");
});

test("lookupMsmCategory queries lookup endpoint", async () => {
  const fakeFetch = async (url) => {
    assert.ok(url.includes("/msm/lookup?kind=users&q=test&employee_id=E1"));
    return {
      ok: true,
      json: async () => [{ value: "u1", label: "User 1" }],
    };
  };
  const res = await lookupMsmCategory({ kind: "users", q: "test", employeeId: "E1" }, fakeFetch);
  assert.deepEqual(res, [{ value: "u1", label: "User 1" }]);
});


test("reduceMsmEvent collects artifact from done event", () => {
  let state = initialMsmState();
  state = reduceMsmEvent(state, {
    type: "done",
    result: { artifact: { id: "art-1", report_type: "msm_executive_report" } }
  });
  assert.equal(state.artifact.id, "art-1");
  assert.equal(state.artifact.report_type, "msm_executive_report");
});

test("does not submit Enter while an IME composition is active", async () => {
  assert.equal(shouldSubmitOnEnter({ key: "Enter", shiftKey: false, nativeEvent: { isComposing: true } }), false);
  assert.equal(shouldSubmitOnEnter({ key: "Enter", keyCode: 229, shiftKey: false, nativeEvent: {} }), false);
  assert.equal(shouldSubmitOnEnter({ key: "Enter", shiftKey: false, nativeEvent: {} }), true);
});
