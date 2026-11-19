// Where the roaming robot should be, derived only from real UI signals.
// Pure functions so the priority rules and geometry are unit-testable.

const ROBOT_ASPECT = 230 / 240; // SVG viewBox height / width
const EDGE = 4;

export function robotHeight(size) {
  return size * ROBOT_ASPECT;
}

/**
 * Priority (highest first): visible error → active request → task preview →
 * hovered/focused task → user typing in the composer → idle patrol → resting place.
 * Composer focus alone does not count: the app focuses it programmatically
 * after each answer, and the robot should stay with that answer.
 */
export function chooseRobotTarget({
  state = "ready",
  hasConversation = false,
  hasError = false,
  previewOpen = false,
  hoverIndex = null,
  patrolIndex = null,
  messageVisible = true,
} = {}) {
  const resting = hasConversation ? (messageVisible ? "message" : "corner") : "home";
  if (hasError) return "error";
  if (state === "error") return resting;
  if (state === "working" || state === "answering") return messageVisible ? "message" : "corner";
  if (previewOpen) return "preview";
  if (Number.isInteger(hoverIndex) && hoverIndex >= 0) return `card:${hoverIndex}`;
  if (state === "composing") return "composer";
  if (!hasConversation && Number.isInteger(patrolIndex) && patrolIndex >= 0) return `card:${patrolIndex}`;
  return resting;
}

function validRect(rect) {
  return Boolean(rect) && rect.width > 0 && rect.height > 0;
}

/** Top-left robot position (container coordinates) for a target kind. */
export function placeRobot(kind, target, container, size) {
  if (!validRect(target) || !validRect(container) || !(size > 0)) return null;
  const h = robotHeight(size);
  const right = target.left + target.width;
  let x;
  let y;
  switch (kind) {
    case "corner":
      // Top-right of the conversation viewport, used when the latest answer is scrolled away.
      x = right - size - 16;
      y = target.top + 10;
      break;
    case "home":
    case "message":
      x = target.left + target.width / 2 - size / 2;
      y = target.top + target.height / 2 - h / 2;
      if (kind === "message") y = target.top - 2;
      break;
    case "card":
      // Peeks over the card's top-right corner.
      x = right - size * 0.62;
      y = target.top - h * 0.72;
      break;
    case "preview":
      if (target.left - container.left >= size + 14) {
        x = target.left - size - 10;
        y = target.top + 4;
      } else {
        x = right - size * 0.7;
        y = target.top - h * 0.8;
      }
      break;
    case "composer":
      x = target.left + 6;
      y = target.top - h * 0.86;
      break;
    case "error":
      x = target.left - size * 0.3;
      y = target.top - h * 0.8;
      break;
    default:
      return null;
  }
  const clamp = (value, max) => Math.max(EDGE, Math.min(Math.max(EDGE, max), value));
  return {
    x: clamp(x - container.left, container.width - size - EDGE),
    y: clamp(y - container.top, container.height - h - EDGE),
  };
}

export function travelDuration(x1, y1, x2, y2) {
  const distance = Math.hypot(x2 - x1, y2 - y1);
  return Math.round(Math.max(450, Math.min(1100, 450 + distance * 0.9)));
}
