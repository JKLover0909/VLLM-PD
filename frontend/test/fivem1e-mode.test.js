import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  buildFivem1ePayload,
  canSubmitFivem1eQuestion,
  initialFivem1eState,
  reduceFivem1eEvent,
  resolveFivem1ePresence,
  shouldApplyFivem1eEvent,
  stopFivem1eQuery,
} from "../src/fivem1e-mode.js";
import { shouldApplyStreamEvent } from "../src/query-stream.js";

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

test("reduceFivem1eEvent stopped keeps the partial answer and terminal fields", () => {
  const state = { ...initialFivem1eState(), answer: "partial", pills: ["p"], error: { code: "X", message: "m" } };
  assert.deepEqual(reduceFivem1eEvent(state, { type: "stopped" }), { ...state, done: true, stopped: true });
  assert.equal(initialFivem1eState().stopped, false);
});

test("shouldApplyFivem1eEvent is the shared stream guard", () => {
  assert.equal(shouldApplyFivem1eEvent, shouldApplyStreamEvent);
});

test("maps 5M1E stream state to robot presence", () => {
  assert.equal(resolveFivem1ePresence({ state: "ready" }, { streaming: true }).state, "working");
  assert.equal(resolveFivem1ePresence({ state: "ready" }, { streaming: true, answering: true }).state, "answering");
  assert.equal(resolveFivem1ePresence({ state: "ready" }, { error: true }).state, "error");
});

test("maps 5M1E answering state to robot presence", () => {
  assert.deepEqual(
    resolveFivem1ePresence({ state: "ready", detail: "" }, { streaming: true, answering: true }),
    { state: "answering", detail: "" },
  );
});

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "../src");
const componentPath = join(ROOT, "components/Fivem1eMode.jsx");
const mainPath = join(ROOT, "main.jsx");

function cleanupEffect(source) {
  const start = source.indexOf("useEffect(() => () => {");
  assert.notEqual(start, -1, "5M1E cleanup effect must exist");
  const end = source.indexOf("}, []);", start);
  assert.notEqual(end, -1, "5M1E cleanup effect must depend on [] only");
  return source.slice(start, end + "}, []);".length);
}

test("Fivem1eMode cleanup is unmount-only and reports stream end", () => {
  const source = readFileSync(componentPath, "utf8");
  const cleanup = cleanupEffect(source);
  assert.match(cleanup, /employeeRef\.current/);
  assert.match(cleanup, /controller\.abort\(\);/);
  assert.match(cleanup, /void stopFivem1eQuery\(\{/);
  assert.match(cleanup, /onStreamingChange\?\.\(false\);/);
  assert.match(cleanup, /onAnsweringChange\?\.\(false\);/);
  assert.doesNotMatch(source, /useEffect\(\(\) => \(\) => \{[\s\S]*?\}, \[employeeId\]\);/, "employeeId changes must not trigger cleanup");
  assert.match(source, /onAnsweringChange/);
});

test("Fivem1eMode reports answering only for the first non-empty token", () => {
  const source = readFileSync(componentPath, "utf8");
  assert.match(
    source,
    /event\?\.type === "token" && event\.content && !previousAnswer\) onAnsweringChange\?\.\(true\);/,
  );
});

test("Fivem1eMode projects stopped into the rendered turn", () => {
  const source = readFileSync(componentPath, "utf8");
  assert.match(source, /stopped: next\.stopped/);
  assert.match(source, /turn\.stopped \? \(/);
  assert.match(source, /language === "ja" \? "停止済み" : "Đã dừng"/);
  assert.match(source, /\{!turn\.done && !turn\.error \? \(/);
});

test("Fivem1eMode stop marks the turn before clearing identity and aborting", () => {
  const source = readFileSync(componentPath, "utf8");
  const dispatch = source.indexOf('{ type: "stopped" }');
  const abort = source.indexOf("controller.abort();");
  assert.ok(dispatch > -1, "stop must dispatch a stopped event");
  assert.ok(abort > -1, "stop must abort the active stream");
  assert.ok(dispatch < abort, "stopped must be dispatched before the abort");
  assert.match(source, /void stopFivem1eQuery\(\{ sessionId: sessionIdRef\.current, employeeId \}\)/);
  assert.match(source, /onStreamingChange\?\.\(false\);\n\s*onAnsweringChange\?\.\(false\);/);
});

test("main tracks 5M1E answering through the shared resolver and resets it", () => {
  const source = readFileSync(mainPath, "utf8");
  assert.match(source, /const \[fivem1eAnswering, setFivem1eAnswering\] = useState\(false\);/);
  assert.match(
    source,
    /resolveFivem1ePresence\(agentPresence, \{\s*streaming: fivem1eStreaming,\s*answering: fivem1eAnswering,\s*error: Boolean\(fivem1eError\),?\s*\}\)/,
  );
  assert.match(source, /onAnsweringChange=\{setFivem1eAnswering\}/);
  const clear = source.slice(
    source.indexOf("function clearConversation()"),
    source.indexOf("function sourcePreviewUrl"),
  );
  assert.match(clear, /setFivem1eAnswering\(false\)/);
});
