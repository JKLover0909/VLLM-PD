import test from "node:test";
import assert from "node:assert/strict";
import { resolveAgentPresence } from "../src/agent-presence.js";

test("presence is ready initially and composing while typing", () => {
  assert.equal(resolveAgentPresence({}).state, "ready");
  assert.equal(resolveAgentPresence({ question: "abc" }).state, "composing");
});
