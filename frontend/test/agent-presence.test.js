import test from "node:test";
import assert from "node:assert/strict";
import { resolveAgentPresence, gazePosition } from "../src/agent-presence.js";

test("empty and typing states do not imply microphone or service readiness", () => {
  assert.equal(resolveAgentPresence({}).state, "ready");
  assert.equal(resolveAgentPresence({ question: "test" }).state, "composing");
  assert.equal(resolveAgentPresence({ needsSelection: true }).state, "asking");
});

test("pending request never borrows content or status from an older answer", () => {
  const p = resolveAgentPresence({ busy: true, pendingId: "new", message: { id: "old", content: "old", status: "old status" } });
  assert.equal(p.state, "working");
  assert.equal(p.detail, "");
});

test("only active backend status and running tool labels describe progress", () => {
  const message = { id: "a", status: "Checking", agentTimeline: { steps: [{ title: "Query", status: "running" }] } };
  assert.equal(resolveAgentPresence({ busy: true, pendingId: "a", message }).detail, "Query");
  assert.equal(resolveAgentPresence({ message }).detail, "");
});

test("partial output, cancellation, errors and clarification stay distinct", () => {
  assert.equal(resolveAgentPresence({ busy: true, pendingId: "a", message: { id: "a", content: "partial" } }).state, "answering");
  assert.equal(resolveAgentPresence({ message: { stopped: true, content: "partial" } }).state, "stopped");
  assert.equal(resolveAgentPresence({ error: "failed", busy: true }).state, "error");
  assert.equal(resolveAgentPresence({ message: { qaMetadata: { clarification: true } } }).state, "asking");
  assert.equal(resolveAgentPresence({ message: { content: "done" } }).state, "ready");
});

test("failed tool is not celebrated as a successful result", () => {
  const message = { agentTimeline: { steps: [{ status: "error" }] } };
  assert.equal(resolveAgentPresence({ message }).state, "error");
});

test("gaze is bounded and centered for invalid inputs", () => {
  assert.deepEqual(gazePosition(1000, -1000), { x: 1, y: -1 });
  assert.deepEqual(gazePosition(NaN, Infinity), { x: 0, y: 0 });
  assert.deepEqual(gazePosition(0, 0), { x: 0, y: 0 });
});
