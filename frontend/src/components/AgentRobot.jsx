import React, { useEffect, useId, useRef } from "react";
import { gazePosition } from "../agent-presence.js";

export default function AgentRobot({ state, motion = true }) {
  const ref = useRef(null);
  const id = useId().replace(/:/g, "");
  useEffect(() => {
    const node = ref.current;
    const region = node.closest(".conversation");
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    let frame = 0;
    const reset = () => {
      cancelAnimationFrame(frame);
      node.style.setProperty("--look-x", "0");
      node.style.setProperty("--look-y", "0");
    };
    const update = (event) => {
      if (!motion || reduced.matches || document.hidden) return;
      const rect = node.getBoundingClientRect();
      const focus = event.type === "focusin" ? event.target.getBoundingClientRect() : null;
      const x = focus ? focus.left + focus.width / 2 : event.clientX;
      const y = focus ? focus.top + focus.height / 2 : event.clientY;
      const gaze = gazePosition((x - rect.left - rect.width / 2) / 280, (y - rect.top - rect.height / 2) / 220);
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        node.style.setProperty("--look-x", String(gaze.x));
        node.style.setProperty("--look-y", String(gaze.y));
      });
    };
    const visibility = () => {
      node.dataset.paused = String(document.hidden || reduced.matches || !motion);
      reset();
    };
    visibility();
    region?.addEventListener("pointermove", update, { passive: true });
    region?.addEventListener("pointerdown", update, { passive: true });
    region?.addEventListener("pointerleave", reset);
    region?.addEventListener("focusin", update);
    document.addEventListener("visibilitychange", visibility);
    reduced.addEventListener("change", visibility);
    return () => {
      reset();
      region?.removeEventListener("pointermove", update);
      region?.removeEventListener("pointerdown", update);
      region?.removeEventListener("pointerleave", reset);
      region?.removeEventListener("focusin", update);
      document.removeEventListener("visibilitychange", visibility);
      reduced.removeEventListener("change", visibility);
    };
  }, [motion]);

  return (
    <div className="agent-robot" ref={ref} data-state={state} aria-hidden="true">
      <svg viewBox="0 0 240 230" focusable="false">
        <defs>
          <linearGradient id={`${id}-shell`} x1="0" y1="0" x2="1" y2="1">
            <stop stopColor="var(--robot-shell-light)" />
            <stop offset=".55" stopColor="var(--robot-shell)" />
            <stop offset="1" stopColor="var(--robot-shell-shade)" />
          </linearGradient>
          <linearGradient id={`${id}-visor`} x1="0" y1="0" x2="0" y2="1">
            <stop stopColor="var(--robot-visor-light)" />
            <stop offset="1" stopColor="var(--robot-visor)" />
          </linearGradient>
        </defs>
        <ellipse className="robot-ground" cx="120" cy="217" rx="68" ry="9" />
        <g className="robot-floating">
          <g className="robot-body">
            <rect x="83" y="160" width="74" height="45" rx="20" fill={`url(#${id}-shell)`} />
            <rect className="robot-arm left" x="60" y="165" width="17" height="33" rx="8" fill={`url(#${id}-shell)`} />
            <rect className="robot-arm right" x="163" y="165" width="17" height="33" rx="8" fill={`url(#${id}-shell)`} />
            <rect className="robot-chest" x="103" y="177" width="34" height="7" rx="3.5" />
          </g>
          <g className="robot-head">
            <path className="robot-antenna" d="M120 43V25" />
            <circle className="robot-signal" cx="120" cy="23" r="6" />
            <rect x="33" y="87" width="17" height="36" rx="8" fill={`url(#${id}-shell)`} />
            <rect x="190" y="87" width="17" height="36" rx="8" fill={`url(#${id}-shell)`} />
            <rect x="45" y="43" width="150" height="113" rx="38" fill={`url(#${id}-shell)`} />
            <rect x="58" y="57" width="124" height="85" rx="28" fill={`url(#${id}-visor)`} />
            <path className="robot-reflection" d="M73 80Q76 68 91 68H146" />
            <g className="robot-gaze">
              <g className="robot-eyes">
                <rect x="83" y="88" width="18" height="25" rx="9" />
                <rect x="139" y="88" width="18" height="25" rx="9" />
              </g>
              <path className="robot-mouth" d="M111 123Q120 129 129 123" />
            </g>
          </g>
        </g>
      </svg>
    </div>
  );
}
