import assert from "node:assert/strict";
import test from "node:test";

import {
  getModeTabScrollLeft,
  shouldClearEmployeeAuth,
  visibleModeKeys,
} from "../src/mode-tabs.js";

const viewport = {
  scrollLeft: 100,
  clientWidth: 200,
  scrollWidth: 600,
};

test("keeps the scroll position when the mode tab is visible", () => {
  assert.equal(
    getModeTabScrollLeft({ ...viewport, tabLeft: 140, tabWidth: 80 }),
    100,
  );
});

test("reveals a mode tab hidden to the left", () => {
  assert.equal(
    getModeTabScrollLeft({ ...viewport, tabLeft: 40, tabWidth: 80 }),
    40,
  );
});

test("reveals a mode tab hidden to the right", () => {
  assert.equal(
    getModeTabScrollLeft({ ...viewport, tabLeft: 280, tabWidth: 80 }),
    160,
  );
});

test("clamps the scroll position at both boundaries", () => {
  assert.equal(
    getModeTabScrollLeft({ ...viewport, tabLeft: -20, tabWidth: 80 }),
    0,
  );
  assert.equal(
    getModeTabScrollLeft({ ...viewport, tabLeft: 560, tabWidth: 80 }),
    400,
  );
});

test("does not scroll when content fits the mode tab viewport", () => {
  assert.equal(
    getModeTabScrollLeft({
      scrollLeft: 0,
      clientWidth: 300,
      scrollWidth: 300,
      tabLeft: 20,
      tabWidth: 80,
    }),
    0,
  );
});

test("hides WMS until its Production snapshot is available", () => {
  assert.deepEqual(visibleModeKeys(false), ["mkac", "mes", "research"]);
  assert.deepEqual(visibleModeKeys(true), ["mkac", "mes", "wms", "research"]);
});

test("clears stale employee auth for protected query modes", () => {
  for (const mode of ["mkac", "mes", "wms"]) {
    assert.equal(
      shouldClearEmployeeAuth({
        status: 403,
        errorCode: "INVALID_EMPLOYEE_ID",
        mode,
      }),
      true,
    );
  }
});

test("keeps employee auth for ownership errors and public modes", () => {
  assert.equal(
    shouldClearEmployeeAuth({
      status: 403,
      errorCode: "ARTIFACT_FORBIDDEN",
      mode: "wms",
    }),
    false,
  );
  assert.equal(
    shouldClearEmployeeAuth({
      status: 403,
      errorCode: "INVALID_EMPLOYEE_ID",
      mode: "research",
    }),
    false,
  );
});
