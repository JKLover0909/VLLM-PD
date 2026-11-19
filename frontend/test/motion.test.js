import test from "node:test";
import assert from "node:assert/strict";
import { canAnimate, runViewTransition, spotlightVars, taskTransitionName } from "../src/motion.js";

test("spotlight tracks the pointer inside a card and clamps tilt", () => {
  const rect = { left: 100, top: 50, width: 200, height: 100 };
  assert.deepEqual(spotlightVars(rect, 200, 100), { "--mx": "50%", "--my": "50%", "--tilt-x": "0deg", "--tilt-y": "0deg" });
  const corner = spotlightVars(rect, 300, 150);
  assert.equal(corner["--mx"], "100%");
  assert.equal(corner["--tilt-y"], "4deg");
  assert.equal(corner["--tilt-x"], "-4deg");
  const outside = spotlightVars(rect, 900, -400);
  assert.equal(outside["--mx"], "100%");
  assert.equal(outside["--my"], "0%");
});

test("spotlight ignores empty rectangles", () => {
  assert.equal(spotlightVars({ left: 0, top: 0, width: 0, height: 0 }, 1, 1), null);
});

test("animation is allowed only when motion is on, not reduced and supported", () => {
  assert.equal(canAnimate({ motion: true, reduced: false }), true);
  assert.equal(canAnimate({ motion: false, reduced: false }), false);
  assert.equal(canAnimate({ motion: true, reduced: true }), false);
});

test("view transition names are unique per task index and absent for unknown tasks", () => {
  assert.equal(taskTransitionName(0), "task-card-0");
  assert.equal(taskTransitionName(3), "task-card-3");
  assert.equal(taskTransitionName(-1), undefined);
});

test("runViewTransition applies the update directly when transitions are unavailable", () => {
  let calls = 0;
  runViewTransition(() => { calls += 1; }, { enabled: true, doc: {} });
  runViewTransition(() => { calls += 1; }, { enabled: false, doc: { startViewTransition: () => assert.fail("must not run") } });
  assert.equal(calls, 2);
});

test("runViewTransition wraps the update when supported", () => {
  let wrapped = null;
  const doc = { startViewTransition: (fn) => { wrapped = fn; fn(); return {}; } };
  let calls = 0;
  runViewTransition(() => { calls += 1; }, { enabled: true, doc });
  assert.equal(typeof wrapped, "function");
  assert.equal(calls, 1);
});

test("runViewTransition still applies the update if the API throws", () => {
  let calls = 0;
  const doc = { startViewTransition: () => { throw new Error("busy"); } };
  runViewTransition(() => { calls += 1; }, { enabled: true, doc });
  assert.equal(calls, 1);
});

test("runViewTransition swallows skipped-transition rejections (no unhandled rejection)", async () => {
  const unhandled = [];
  const onUnhandled = (reason) => unhandled.push(reason);
  process.on("unhandledRejection", onUnhandled);
  try {
    const skipped = () => Promise.reject(new DOMException("Transition was skipped", "AbortError"));
    const doc = {
      startViewTransition: (fn) => {
        fn();
        return { ready: skipped(), finished: skipped(), updateCallbackDone: skipped() };
      },
    };
    let calls = 0;
    runViewTransition(() => { calls += 1; }, { enabled: true, doc });
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(calls, 1);
    assert.deepEqual(unhandled.map(String), []);
  } finally {
    process.off("unhandledRejection", onUnhandled);
  }
});
