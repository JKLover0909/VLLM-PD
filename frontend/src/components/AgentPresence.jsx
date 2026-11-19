import React from "react";
import { Pause, Play } from "lucide-react";

const labels = {
  vi: {
    headline: "Bạn muốn làm gì?", ready: "", composing: "Bạn đang nhập câu hỏi",
    asking: "Cần bạn chọn hoặc bổ sung thông tin", working: "Đang xử lý yêu cầu",
    answering: "Đang hiển thị câu trả lời", complete: "Bạn có thể xem nguồn hoặc hỏi tiếp",
    stopped: "Đã dừng nhận câu trả lời", error: "Có vấn đề cần kiểm tra",
    motion: "Chuyển động robot", motionOn: "Tắt chuyển động robot", motionOff: "Bật chuyển động robot",
  },
  ja: {
    headline: "何をしますか？", ready: "", composing: "質問を入力中です",
    asking: "選択または追加情報が必要です", working: "リクエストを処理中",
    answering: "回答を表示中", complete: "出典を確認するか、続けて質問できます",
    stopped: "回答の受信を停止しました", error: "確認が必要な問題があります",
    motion: "ロボットの動き", motionOn: "ロボットの動きをオフ", motionOff: "ロボットの動きをオン",
  },
};

// Welcome stage only: an empty dock for the roaming robot + one headline.
// While a conversation runs, state is shown by the robot itself (thought bubble)
// and announced through <AgentStatus>; there is no separate status bar.
export default function AgentPresence({ presence, mode, language }) {
  const copy = labels[language] || labels.vi;
  return (
    <div className={`agent-presence stage agent-mode-${mode}`} data-state={presence.state}>
      {/* Empty slot: the single roaming robot (RoamingRobot.jsx) docks here. */}
      <div className="agent-stage-visual" data-robot-anchor="home" aria-hidden="true">
        <span className="agent-orbit" />
      </div>
      <div className="agent-presence-copy">
        <h1 className="agent-headline">{copy.headline}</h1>
      </div>
    </div>
  );
}

// Screen-reader announcement of the real request state (WCAG 4.1.3). Always
// mounted so changes are announced; visually hidden because the robot and the
// waiting bubble already show the state.
export function AgentStatus({ presence, language }) {
  const copy = labels[language] || labels.vi;
  const status = presence.detail || copy[presence.state] || "";
  return <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">{status}</p>;
}

// Header button: fixed accessible name, state carried by aria-pressed.
export function MotionToggle({ motion, onMotionChange, language }) {
  const copy = labels[language] || labels.vi;
  const MotionIcon = motion ? Pause : Play;
  return (
    <button
      type="button"
      className="icon-button header-tool agent-motion-toggle"
      aria-pressed={motion}
      aria-label={copy.motion}
      title={motion ? copy.motionOn : copy.motionOff}
      onClick={() => onMotionChange(!motion)}
    >
      <MotionIcon size={17} aria-hidden="true" />
    </button>
  );
}
