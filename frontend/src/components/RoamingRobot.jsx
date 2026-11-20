import React, { useEffect, useRef, useState } from "react";
import AgentRobot from "./AgentRobot.jsx";
import { canAnimate, prefersReducedMotion } from "../motion.js";
import { chooseRobotTarget, placeRobot, robotAnchorSelectors, robotHeight, travelDuration } from "../robot-path.js";

const SMALL_SIZE = 56;
const PHONE_SIZE = 44;
const PATROL_EVERY_MS = 6500;
const PATROL_STAY_MS = 2400;

// DOM anchors the robot can travel to, taken from the shared selector table so
// the robot and its tests agree on what the real UI renders. The first selector
// that matches wins: `.error-banner` before a scoped latest-turn alert.
function findAnchor(root, target) {
  for (const selector of robotAnchorSelectors(target)) {
    const match = root.querySelector(selector);
    if (match) return match;
  }
  return null;
}

function isVisibleIn(element, viewport) {
  if (!element || !viewport) return false;
  const a = element.getBoundingClientRect();
  const b = viewport.getBoundingClientRect();
  return a.bottom > b.top + 8 && a.top < b.bottom - 8;
}

export default function RoamingRobot({ state, motion, hasConversation, hasError, previewOpen }) {
  const ref = useRef(null);
  const last = useRef({ target: "", x: 0, y: 0, size: 0, placed: false });
  const moveTimer = useRef(0);
  const [hoverIndex, setHoverIndex] = useState(null);
  const [patrolIndex, setPatrolIndex] = useState(null);
  const [pose, setPose] = useState({ x: 0, y: 0, size: SMALL_SIZE, duration: 0, moving: false, dir: 1, target: "", visible: false });

  const animate = canAnimate({ motion, reduced: prefersReducedMotion() });

  // Real user signals: which task card is hovered/focused, whether the composer has focus.
  useEffect(() => {
    const root = ref.current?.parentElement;
    if (!root) return undefined;
    const cardIndex = (node) => {
      const card = node?.closest?.(".task-card[data-task-index]");
      return card ? Number(card.dataset.taskIndex) : null;
    };
    const over = (event) => setHoverIndex(cardIndex(event.target));
    const leave = () => setHoverIndex(null);
    const focusIn = (event) => {
      const index = cardIndex(event.target);
      if (index !== null) setHoverIndex(index);
    };
    const focusOut = (event) => {
      if (cardIndex(event.target) !== null) setHoverIndex(null);
    };
    root.addEventListener("pointerover", over);
    root.addEventListener("pointerleave", leave);
    root.addEventListener("focusin", focusIn);
    root.addEventListener("focusout", focusOut);
    return () => {
      root.removeEventListener("pointerover", over);
      root.removeEventListener("pointerleave", leave);
      root.removeEventListener("focusin", focusIn);
      root.removeEventListener("focusout", focusOut);
    };
  }, []);

  // Idle patrol on the start screen only; never while a request is running.
  useEffect(() => {
    setPatrolIndex(null);
    if (!animate || hasConversation || previewOpen || state !== "ready") return undefined;
    let stayTimer = 0;
    const tick = () => {
      if (document.hidden) return;
      const cards = ref.current?.parentElement?.querySelectorAll(".task-card[data-task-index]") || [];
      if (!cards.length) return;
      setPatrolIndex(Math.floor(Math.random() * cards.length));
      clearTimeout(stayTimer);
      stayTimer = setTimeout(() => setPatrolIndex(null), PATROL_STAY_MS);
    };
    const interval = setInterval(tick, PATROL_EVERY_MS);
    return () => {
      clearInterval(interval);
      clearTimeout(stayTimer);
    };
  }, [animate, hasConversation, previewOpen, state]);

  // Placement: recomputed on any input change, layout change, DOM change or scroll.
  useEffect(() => {
    const node = ref.current;
    const root = node?.parentElement;
    if (!root) return undefined;
    let frame = 0;

    const compute = () => {
      frame = 0;
      const scroller = root.querySelector(".conversation-scroll");
      const latest = root.querySelector('.message[data-latest="true"] .message-avatar');
      const resting = hasConversation ? "corner" : "home";
      let target = animate
        ? chooseRobotTarget({
            state, hasConversation, hasError, previewOpen, hoverIndex, patrolIndex,
            messageVisible: isVisibleIn(latest, scroller),
          })
        : resting;
      let anchor = findAnchor(root, target);
      if (!anchor) {
        target = resting;
        anchor = findAnchor(root, target);
      }
      if (!anchor) {
        setPose((current) => (current.visible ? { ...current, visible: false } : current));
        return;
      }
      const container = root.getBoundingClientRect();
      const rect = anchor.getBoundingClientRect();
      const docked = target === "home";
      const size = docked ? rect.width : container.width <= 600 ? PHONE_SIZE : SMALL_SIZE;
      const kind = target.startsWith("card:") ? "card" : target;
      const place = placeRobot(kind, rect, container, size);
      if (!place) return;

      const prev = last.current;
      const retarget = prev.target !== target;
      const same = Math.abs(prev.x - place.x) < 0.5 && Math.abs(prev.y - place.y) < 0.5 && Math.abs(prev.size - size) < 0.5;
      if (same && !retarget && prev.placed) return;
      const duration = !prev.placed || !animate ? 0 : retarget ? travelDuration(prev.x, prev.y, place.x, place.y) : 140;
      const moving = duration > 200;
      const dir = place.x < prev.x ? -1 : 1;
      last.current = { target, x: place.x, y: place.y, size, placed: true };
      root.dataset.robotAt = kind;
      setPose({ x: place.x, y: place.y, size, duration, moving, dir, target: kind, visible: true });
      clearTimeout(moveTimer.current);
      if (moving) moveTimer.current = setTimeout(() => setPose((current) => ({ ...current, moving: false })), duration);
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(compute);
    };

    schedule();
    const resize = new ResizeObserver(schedule);
    resize.observe(root);
    const mutations = new MutationObserver(schedule);
    mutations.observe(root, { childList: true, subtree: true });
    root.addEventListener("scroll", schedule, true);
    window.addEventListener("resize", schedule);
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      mutations.disconnect();
      root.removeEventListener("scroll", schedule, true);
      window.removeEventListener("resize", schedule);
      delete root.dataset.robotAt;
    };
  }, [animate, state, hasConversation, hasError, previewOpen, hoverIndex, patrolIndex]);

  useEffect(() => () => clearTimeout(moveTimer.current), []);

  const thought = state === "working" || state === "answering" ? "dots" : state === "asking" ? "?" : state === "error" ? "!" : "";

  return (
    <div
      ref={ref}
      className="roaming-robot"
      aria-hidden="true"
      data-target={pose.target}
      data-moving={pose.moving}
      data-visible={pose.visible}
      style={{
        width: `${pose.size}px`,
        height: `${robotHeight(pose.size)}px`,
        transform: `translate3d(${pose.x}px, ${pose.y}px, 0)`,
        transitionDuration: `${pose.duration}ms`,
        "--dir": pose.dir,
      }}
    >
      <div className="roaming-robot-body">
        <AgentRobot state={state} motion={motion} />
      </div>
      {thought && pose.target !== "home" && (
        <span className={`robot-thought ${thought === "dots" ? "is-dots" : ""}`}>
          {thought === "dots" ? <><i /><i /><i /></> : thought}
        </span>
      )}
    </div>
  );
}
