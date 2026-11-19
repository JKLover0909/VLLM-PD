import React, { useEffect, useRef, useState } from "react";
import AgentRobot from "./AgentRobot.jsx";
import { prefersReducedMotion } from "../motion.js";
import {
  handoffTransform,
  introDuration,
  introVariant,
  shouldPlayIntro,
} from "../intro.js";

const COPY = {
  vi: { kicker: "MEIBOOK", line: "Đang kết nối trợ lý AI nội bộ", ready: "Sẵn sàng làm việc", offline: "Chưa kết nối được máy chủ", skip: "Bỏ qua" },
  ja: { kicker: "MEIBOOK", line: "社内AIアシスタントに接続しています", ready: "準備ができました", offline: "サーバーに接続できません", skip: "スキップ" },
};

const PARTICLES = Array.from({ length: 18 }, (_, index) => index);
const EXIT_MS = 620;

function decideIntro() {
  if (typeof window === "undefined") return null;
  if (!shouldPlayIntro({ search: window.location.search })) return null;
  return introVariant({
    reduced: prefersReducedMotion(),
    cores: navigator.hardwareConcurrency,
    saveData: navigator.connection?.saveData === true,
  });
}

// The live robot (RoamingRobot) or, before it is placed, its welcome dock.
function liveRobotBox() {
  const live = document.querySelector('.roaming-robot[data-visible="true"]');
  const dock = document.querySelector(".agent-presence.stage [data-robot-anchor]");
  const box = (live || dock)?.getBoundingClientRect();
  return box && box.width > 0 && box.height > 0 ? box : null;
}

// Decorative overlay shown on every page load. It never gates the app:
// the workspace renders underneath and stays usable; any key, click, tap or
// wheel dismisses it. `health` is the real /health bootstrap state, so the
// status line and progress bar only claim "ready" once the server answered.
export default function IntroSplash({ language = "vi", health = "checking" }) {
  const [variant] = useState(decideIntro);
  const [phase, setPhase] = useState(variant ? "play" : "done");
  const [handoff, setHandoff] = useState(null);
  const robotRef = useRef(null);
  const exitTimer = useRef(0);
  const copy = COPY[language] || COPY.vi;

  const dismiss = () => {
    setPhase((current) => {
      if (current !== "play") return current;
      if (variant !== "reduced" && robotRef.current) {
        setHandoff(handoffTransform(robotRef.current.getBoundingClientRect(), liveRobotBox()));
      }
      exitTimer.current = window.setTimeout(() => setPhase("done"), EXIT_MS);
      return "exit";
    });
  };

  useEffect(() => {
    if (phase !== "play") return undefined;
    const timer = window.setTimeout(dismiss, introDuration(variant));
    const onInput = () => dismiss();
    window.addEventListener("keydown", onInput);
    window.addEventListener("pointerdown", onInput);
    window.addEventListener("wheel", onInput, { passive: true });
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("keydown", onInput);
      window.removeEventListener("pointerdown", onInput);
      window.removeEventListener("wheel", onInput);
    };
  }, [phase, variant]);

  useEffect(() => () => window.clearTimeout(exitTimer.current), []);

  if (phase === "done" || !variant) return null;

  const ready = health === "online";
  const line = ready ? copy.ready : health === "offline" ? copy.offline : copy.line;
  const robotStyle = handoff
    ? { "--handoff-x": `${handoff.x}px`, "--handoff-y": `${handoff.y}px`, "--handoff-scale": handoff.scale }
    : undefined;

  return (
    <div
      className={`intro-splash intro-${variant} intro-${phase}`}
      data-health={health}
      data-handoff={Boolean(handoff)}
      aria-hidden="true"
    >
      <div className="intro-aurora" />
      <div className="intro-grid" />
      <div className="intro-stage">
        {variant === "full" && (
          <div className="intro-core">
            <span className="intro-ring ring-a" />
            <span className="intro-ring ring-b" />
            <span className="intro-ring ring-c" />
            <div className="intro-particles">
              {PARTICLES.map((index) => (
                <i key={index} style={{ "--i": index }} />
              ))}
            </div>
          </div>
        )}
        <div className="intro-robot" ref={robotRef} style={robotStyle}>
          <AgentRobot state={ready ? "complete" : "working"} motion={variant === "full"} />
        </div>
      </div>
      <div className="intro-copy">
        <span className="intro-kicker">{copy.kicker}</span>
        <span className="intro-line" key={line}>{line}</span>
        <span className="intro-progress"><i /></span>
      </div>
      <span className="intro-skip">{copy.skip} ⏎</span>
    </div>
  );
}
