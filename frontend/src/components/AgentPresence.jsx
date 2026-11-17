import React from "react";
import AgentRobot from "./AgentRobot.jsx";

const labels = {
  vi: {
    headline: "Bạn muốn làm gì?", start: "Bắt đầu nhập câu hỏi", ready: "", composing: "Bạn đang nhập câu hỏi",
    asking: "Cần bạn chọn hoặc bổ sung thông tin", working: "Đang xử lý yêu cầu",
    answering: "Đang hiển thị câu trả lời", complete: "Bạn có thể xem nguồn hoặc hỏi tiếp",
    stopped: "Đã dừng nhận câu trả lời", error: "Có vấn đề cần kiểm tra",
    motion: "Chuyển động robot", motionOn: "Tắt chuyển động", motionOff: "Bật chuyển động", hint: "Chọn gợi ý bên dưới hoặc tự nhập câu hỏi.",
  },
  ja: {
    headline: "何をしますか？", start: "質問を入力して開始", ready: "", composing: "質問を入力中です",
    asking: "選択または追加情報が必要です", working: "リクエストを処理中",
    answering: "回答を表示中", complete: "出典を確認するか、続けて質問できます",
    stopped: "回答の受信を停止しました", error: "確認が必要な問題があります",
    motion: "ロボットの動き", motionOn: "動きをオフ", motionOff: "動きをオン", hint: "下の候補を選ぶか、質問を入力してください。",
  },
};

export default function AgentPresence({ presence, compact, mode, language, motion, onMotionChange, onOpenComposer, minimal = false }) {
  const copy = labels[language] || labels.vi;
  return (
    <div className={`agent-presence ${compact ? "compact" : "stage"} agent-mode-${mode}`} data-state={presence.state} data-motion={motion} data-minimal={minimal}>
      <div className="agent-stage-visual">
        <span className="agent-orbit" aria-hidden="true" />
        <AgentRobot state={presence.state} motion={motion} />
      </div>
      <div className="agent-presence-copy">
        {!minimal && <span className="agent-presence-name">MEIBOOK <span aria-hidden="true">/</span> {mode.toUpperCase()}</span>}
        {(presence.detail || (minimal ? copy.headline : copy[presence.state])) && (
          <p className="agent-presence-label" role="status" aria-live="polite" aria-atomic="true">
            {presence.detail || (minimal ? copy.headline : copy[presence.state])}
          </p>
        )}
        {!compact && !minimal && <small>{copy.hint}</small>}
        {!compact && onOpenComposer && <button type="button" className="agent-presence-start" onClick={onOpenComposer}>{copy.start}</button>}
      </div>
      {!minimal && <button type="button" className="agent-motion-toggle" aria-pressed={motion} onClick={() => onMotionChange(!motion)} title={motion ? copy.motionOn : copy.motionOff} aria-label={motion ? copy.motionOn : copy.motionOff}>
        <span aria-hidden="true">{motion ? "◉" : "○"}</span> {motion ? copy.motionOn : copy.motionOff}
      </button>}
    </div>
  );
}
