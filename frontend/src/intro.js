// "Wake-up" intro shown when the app opens: pure, testable decisions.
//
// Product decision (2026-11-18): play on every page load, reloads included,
// so users always see the mascot wake up. Nothing is remembered in storage.
// It stays short, skippable (any key/click/tap/wheel) and never blocks the
// app; reduced-motion users get the short variant. "?intro=0" turns it off
// for a single load (kiosk screens, demos, automated tests).

export const INTRO_DURATION_MS = 2600;
export const INTRO_REDUCED_MS = 600;

export function shouldPlayIntro({ search = "" } = {}) {
  try {
    return new URLSearchParams(search).get("intro") !== "0";
  } catch {
    return true;
  }
}

// Full 3D scene only when the device can afford it. Missing signals are
// "unknown", not "weak" (deviceMemory/saveData are Chromium-only).
export function introVariant({ reduced = false, cores, saveData = false } = {}) {
  if (reduced) return "reduced";
  if (saveData) return "lite";
  if (Number.isFinite(cores) && cores > 0 && cores <= 2) return "lite";
  return "full";
}

export function introDuration(variant) {
  return variant === "reduced" ? INTRO_REDUCED_MS : INTRO_DURATION_MS;
}

// "?intro=1" kept for old demo links; the intro already plays on every load.
export function introForced(search = "") {
  try {
    return new URLSearchParams(search).get("intro") === "1";
  } catch {
    return false;
  }
}

// Exit handoff: translate/scale that moves the intro robot's box onto the live
// robot's box, so the same mascot visibly "walks into" the workspace.
export function handoffTransform(from, to) {
  const valid = (box) => box && box.width > 0 && box.height > 0;
  if (!valid(from) || !valid(to)) return null;
  const center = (box) => [box.left + box.width / 2, box.top + box.height / 2];
  const [fx, fy] = center(from);
  const [tx, ty] = center(to);
  const scale = Math.min(2, Math.max(0.2, to.width / from.width));
  return { x: Math.round(tx - fx), y: Math.round(ty - fy), scale: Math.round(scale * 100) / 100 };
}

// Elapsed wait shown while the model works: honest wall-clock time only.
export function formatElapsed(ms, language = "vi") {
  const seconds = Math.max(0, Math.floor((Number(ms) || 0) / 1000));
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  if (language === "ja") return minutes ? `${minutes}分${rest}秒` : `${seconds}秒`;
  return minutes ? `${minutes} phút ${rest} giây` : `${seconds} giây`;
}
