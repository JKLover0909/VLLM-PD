import test from "node:test";
import assert from "node:assert/strict";
import {
  formatElapsed,
  handoffTransform,
  introDuration,
  introForced,
  introVariant,
  shouldPlayIntro,
} from "../src/intro.js";

// Product decision (2026-11-18): the wake-up intro plays on every page load,
// including reloads, so users always see it. It stays skippable and never
// blocks the app; "?intro=0" turns it off for one load (kiosk, demos, tests).
test("intro plays on every load, including repeated reloads", () => {
  assert.equal(shouldPlayIntro(), true);
  assert.equal(shouldPlayIntro({ search: "" }), true);
  for (let load = 0; load < 3; load += 1) assert.equal(shouldPlayIntro({ search: "?mode=mes" }), true);
});

test("intro does not depend on storage, so blocked storage still shows it", () => {
  const throwing = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); } };
  assert.equal(shouldPlayIntro({ storage: throwing }), true);
});

test("?intro=0 skips the intro for a single load; ?intro=1 still plays", () => {
  assert.equal(shouldPlayIntro({ search: "?intro=0" }), false);
  assert.equal(shouldPlayIntro({ search: "?mode=mes&intro=0" }), false);
  assert.equal(shouldPlayIntro({ search: "?intro=1" }), true);
  assert.equal(introForced("?intro=1"), true);
  assert.equal(introForced("?mode=mes"), false);
});

test("variant honours reduced motion, save-data and very weak devices only", () => {
  assert.equal(introVariant({ reduced: true, cores: 16 }), "reduced");
  assert.equal(introVariant({ saveData: true, cores: 16 }), "lite");
  assert.equal(introVariant({ cores: 2 }), "lite");
  assert.equal(introVariant({ cores: 8 }), "full");
  // Unknown signals are not treated as weak.
  assert.equal(introVariant({}), "full");
  assert.equal(introVariant({ cores: undefined }), "full");
});

test("reduced intro is short; full intro stays under the 10 s attention limit", () => {
  assert.ok(introDuration("reduced") <= 1000);
  assert.ok(introDuration("full") <= 3000);
});

test("elapsed time is whole seconds of real wall-clock time", () => {
  assert.equal(formatElapsed(0), "0 giây");
  assert.equal(formatElapsed(12_900), "12 giây");
  assert.equal(formatElapsed(75_000), "1 phút 15 giây");
  assert.equal(formatElapsed(12_000, "ja"), "12秒");
  assert.equal(formatElapsed(75_000, "ja"), "1分15秒");
  assert.equal(formatElapsed(-5), "0 giây");
  assert.equal(formatElapsed(undefined), "0 giây");
});

test("handoff flies the intro robot onto the real robot's box", () => {
  const from = { left: 600, top: 300, width: 120, height: 120 };
  const to = { left: 100, top: 80, width: 60, height: 60 };
  // Center (660,360) -> (130,110); scale 0.5.
  assert.deepEqual(handoffTransform(from, to), { x: -530, y: -250, scale: 0.5 });
});

test("handoff is skipped when either box is missing or empty", () => {
  const box = { left: 0, top: 0, width: 10, height: 10 };
  assert.equal(handoffTransform(null, box), null);
  assert.equal(handoffTransform(box, undefined), null);
  assert.equal(handoffTransform({ ...box, width: 0 }, box), null);
  assert.equal(handoffTransform(box, { ...box, height: 0 }), null);
});

test("handoff scale is clamped so a tiny or huge target never distorts", () => {
  const from = { left: 0, top: 0, width: 100, height: 100 };
  assert.equal(handoffTransform(from, { left: 0, top: 0, width: 1, height: 1 }).scale, 0.2);
  assert.equal(handoffTransform(from, { left: 0, top: 0, width: 900, height: 900 }).scale, 2);
});
