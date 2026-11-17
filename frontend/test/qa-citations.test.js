import assert from "node:assert/strict";
import test from "node:test";

import {
  getQaCitationEntries,
  getQaElapsedSeconds,
  isQaCitationListTruncated,
} from "../src/qa-citations.js";

test("reports no processing time when the agent omitted elapsed_ms", () => {
  assert.equal(getQaElapsedSeconds({ elapsed_ms: null }), null);
  assert.equal(getQaElapsedSeconds({}), null);
  assert.equal(getQaElapsedSeconds(undefined), null);
  assert.equal(getQaElapsedSeconds({ elapsed_ms: "1234" }), null);
  assert.equal(getQaElapsedSeconds({ elapsed_ms: -1 }), null);
  assert.equal(getQaElapsedSeconds({ elapsed_ms: 0 }), 0);
  assert.equal(getQaElapsedSeconds({ elapsed_ms: 1234 }), 1.234);
});

test("flags truncated citation lists only on an explicit true", () => {
  assert.equal(isQaCitationListTruncated({ citations_truncated: true }), true);
  assert.equal(isQaCitationListTruncated({ citations_truncated: false }), false);
  assert.equal(isQaCitationListTruncated({ citations_truncated: "true" }), false);
  assert.equal(isQaCitationListTruncated(null), false);
});

test("keeps only bounded QA citation display fields", () => {
  const result = getQaCitationEntries({
    citations: [
      {
        type: "sql",
        views: ["v_quality", 42, { secret: true }],
        keys: { item_code: ["A-1"], ignored: [null, "B-2"] },
        inspection_keys: ["INSP-1"],
        chunks: [{ id: "chunk-1", source_view: "v_quality", source_key: "A-1" }],
        sql: "SELECT confidential_data",
        url: "https://unsafe.example",
      },
      { type: "unknown", views: ["hidden"] },
    ],
  });
  assert.deepEqual(result, [
    {
      type: "sql",
      views: ["v_quality", "42"],
      fields: [{ name: "item_code", values: ["A-1"] }, { name: "ignored", values: ["B-2"] }],
      inspectionKeys: ["INSP-1"],
      chunks: [{ id: "chunk-1", sourceView: "v_quality", sourceKey: "A-1", inspectionKey: "" }],
    },
  ]);
  assert.equal(JSON.stringify(result).includes("SELECT"), false);
  assert.equal(JSON.stringify(result).includes("unsafe.example"), false);
});

test("bounds citation count and display string length", () => {
  const citations = Array.from({ length: 30 }, (_, index) => ({
    type: "row",
    views: [`${index}-${"x".repeat(250)}`],
  }));
  const result = getQaCitationEntries({ citations });
  assert.equal(result.length, 20);
  assert.equal(result[0].views[0].length, 200);
});

test("ignores malformed citation metadata", () => {
  assert.deepEqual(getQaCitationEntries(null), []);
  assert.deepEqual(getQaCitationEntries({ citations: "not-an-array" }), []);
  assert.deepEqual(getQaCitationEntries({ citations: [{ type: "sql" }] }), []);
});
