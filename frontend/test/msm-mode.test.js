import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

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

test("reduceMsmEvent stopped keeps answer, views, forms and pending", () => {
  const state = { ...initialMsmState(), answer: "partial", views: ["v"], forms: ["f"], pending: { id: "p" } };
  assert.deepEqual(reduceMsmEvent(state, { type: "stopped" }), { ...state, done: true, stopped: true });
  assert.equal(initialMsmState().stopped, false);
});

test("does not submit Enter while an IME composition is active", async () => {
  assert.equal(shouldSubmitOnEnter({ key: "Enter", shiftKey: false, nativeEvent: { isComposing: true } }), false);
  assert.equal(shouldSubmitOnEnter({ key: "Enter", keyCode: 229, shiftKey: false, nativeEvent: {} }), false);
  assert.equal(shouldSubmitOnEnter({ key: "Enter", shiftKey: false, nativeEvent: {} }), true);
});

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../src");
const componentPath = join(ROOT, "components/MsmMode.jsx");
const mainPath = join(ROOT, "main.jsx");
const crypto = await import("node:crypto");

test("MsmMode reports stream flags and guards stale streams", () => {
  const source = readFileSync(componentPath, "utf8");
  assert.match(source, /onStreamingChange/);
  assert.match(source, /onAnsweringChange/);
  assert.match(source, /onErrorChange/);
  assert.match(source, /shouldApplyStreamEvent/);
  assert.match(source, /data-latest=\{index === turns\.length - 1/);
});

test("main blocks mode switch and clear while MSM or 5M1E streams", () => {
  const source = readFileSync(mainPath, "utf8");
  assert.match(source, /msmStreaming \|\| fivem1eStreaming/);
  assert.match(source, /disabled=\{[^}]*msmStreaming/);
});

test("msm write path regions are byte-identical to 8db9719", () => {
  const current = readFileSync(componentPath, "utf8");
  const baseline = String(execFileSync("git", ["show", "8db9719:frontend/src/components/MsmMode.jsx"], { encoding: "utf8", cwd: join(ROOT, "../..") }));
  const regions = [
    ["MsmTableView", "function MsmTableView", "export default function MsmMode", "38987953bf078bc36be4af9e5bcc73fe9562f714efcd707713ad86f7e0ee8801"],
    ["handleFormSubmit", "  async function handleFormSubmit", "  const disabled", "310d3816ab2405460a8248c19fca353ab6cdf0cb261b293b3a441ff0284c3032"],
    ["forms-pending", "        {turn.views.map", "        {turn.notice", "ee8338cd7a4b656025001476866c835734291dafabf62ab9f900cc6df9ee8505"],
  ];
  for (const [name, start, end, expected] of regions) {
    const slice = (source) => source.slice(source.indexOf(start), source.indexOf(end));
    assert.equal(crypto.createHash("sha256").update(slice(current)).digest("hex"), expected, name);
    assert.equal(slice(current), slice(baseline), name);
  }
});
