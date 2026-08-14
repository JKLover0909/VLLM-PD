import assert from "node:assert/strict";
import test from "node:test";

import { quickAnswerRequestOptions } from "../src/quick-answer-request.js";

test("preserves the ID for a server-prepared suggestion", () => {
  assert.deepEqual(
    quickAnswerRequestOptions({
      id: "wms-snapshot-explainer",
      execution: "server_prepared",
    }),
    { quickAnswerId: "wms-snapshot-explainer" },
  );
});

test("omits the ID for a query-backed suggestion", () => {
  assert.deepEqual(
    quickAnswerRequestOptions({ id: "wms-item-count", execution: "query" }),
    { quickAnswerId: null },
  );
});

test("omits malformed or legacy suggestion IDs", () => {
  for (const suggestion of [
    null,
    {},
    { execution: "server_prepared" },
    { id: "   ", execution: "server_prepared" },
    { id: "wms-item-count", execution: "unexpected" },
  ]) {
    assert.deepEqual(quickAnswerRequestOptions(suggestion), {
      quickAnswerId: null,
    });
  }
});
