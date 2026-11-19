// Small, testable helpers for UI motion. Every effect must degrade to an
// immediate state change when motion is off, reduced or unsupported.

const MAX_TILT_DEG = 4;

export function canAnimate({ motion = true, reduced = false } = {}) {
  return Boolean(motion) && !reduced;
}

export function prefersReducedMotion() {
  try {
    return globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
  } catch {
    return false;
  }
}

// Pointer position inside a card → CSS variables for a light spotlight and a
// small tilt. Values are clamped so a fast pointer never flips the card.
export function spotlightVars(rect, clientX, clientY) {
  if (!rect || !(rect.width > 0) || !(rect.height > 0)) return null;
  const clamp = (value) => Math.max(0, Math.min(1, value));
  const px = clamp((clientX - rect.left) / rect.width);
  const py = clamp((clientY - rect.top) / rect.height);
  const round = (value) => Math.round(value * 100) / 100;
  return {
    "--mx": `${round(px * 100)}%`,
    "--my": `${round(py * 100)}%`,
    "--tilt-x": `${round((0.5 - py) * 2 * MAX_TILT_DEG) || 0}deg`,
    "--tilt-y": `${round((px - 0.5) * 2 * MAX_TILT_DEG) || 0}deg`,
  };
}

export function taskTransitionName(index) {
  return Number.isInteger(index) && index >= 0 ? `task-card-${index}` : undefined;
}

// Wrap a synchronous state update in a View Transition when available.
// The update always runs exactly once, even if the API is missing or throws.
export function runViewTransition(update, { enabled = true, doc = globalThis.document } = {}) {
  if (!enabled || typeof doc?.startViewTransition !== "function") {
    update();
    return;
  }
  let applied = false;
  const apply = () => {
    if (applied) return;
    applied = true;
    update();
  };
  try {
    const transition = doc.startViewTransition(apply);
    // A transition is skipped (rejecting these promises with AbortError) when
    // another starts, the tab is hidden or an overlay covers the page. That is
    // expected, not an error: the update already ran, so just settle quietly.
    for (const key of ["ready", "finished", "updateCallbackDone"]) {
      transition?.[key]?.catch?.(() => {});
    }
  } catch {
    apply();
  }
}
