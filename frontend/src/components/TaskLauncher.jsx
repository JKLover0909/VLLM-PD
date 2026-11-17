import React from "react";
import { Activity, ArrowLeft, ArrowRight, Boxes, Building2, Check, ChevronDown, ClipboardCheck, Cctv, Edit3, FlaskConical, Search, SlidersHorizontal } from "lucide-react";

const ICONS = { mkac: Building2, mes: Activity, wms: Boxes, cctvai: Cctv, qa: ClipboardCheck, research: FlaskConical };

const COPY = {
  vi: {
    title: "Tác vụ", free: "Tự nhập câu hỏi", backToTasks: "Quay lại tác vụ", other: "Gợi ý khác", info: "Thông tin", backToFree: "Đóng ô nhập",
    instructions: "Hướng dẫn", qaInfo: "Câu đầu tiên nên có mã máy hoặc mã phiếu, thời gian và điều kiện",
    modeInfo: "Phạm vi", ready: "Sẵn sàng", run: "Thực hiện", edit: "Chỉnh câu hỏi", back: "Quay lại", preview: "Câu hỏi sẽ gửi",
  },
  ja: {
    title: "タスク", free: "質問を入力", other: "その他の候補", info: "情報", backToTasks: "タスクに戻る",
    instructions: "質問のコツ", qaInfo: "最初の質問には設備またはチェックシートのコード、期間、条件を含めてください",
    modeInfo: "範囲", ready: "準備完了", run: "実行", edit: "質問を編集", back: "戻る", preview: "送信する質問",
  },
};

export default function TaskLauncher({
  mode, language, groups, selectedTask, previewQuestion, onPreviewChange, onSelect, onRun, onPreviewQuestionChange, onEdit, onBack,
  onFreeInput, onBackToTasks, showFreeInput = false, qaGuidance, modeDescription, researchContext, infoItems = [], onToggleInfo, showInfo = false, children,
}) {
  const copy = COPY[language] || COPY.vi;
  const hasSelection = Boolean(selectedTask);

  return (
    <section className="task-launcher" aria-label={copy.title}>
      {children}
      {hasSelection ? (
        <div className="task-preview" aria-live="polite">
          <label className="task-preview-label" htmlFor="task-preview-question">{copy.preview}</label>
          <p className="task-preview-original">{selectedTask.label}</p>
          <textarea
            id="task-preview-question"
            className="task-preview-question"
            value={previewQuestion}
            maxLength={mode === "qa" ? 2000 : 4000}
            onChange={(event) => onPreviewChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") onBack();
            }}
            rows={Math.min(3, Math.max(1, Math.ceil(previewQuestion.length / 68)))}
            autoFocus
          />
          <div className="task-preview-actions">
            <button type="button" className="task-run" onClick={() => onRun(selectedTask, previewQuestion)}>
              <Check size={16} aria-hidden="true" /> {copy.run}
            </button>
            <button type="button" className="task-edit" onClick={() => onEdit(selectedTask)}>
              <Edit3 size={15} aria-hidden="true" /> {copy.edit}
            </button>
            <button type="button" className="task-back" onClick={onBack}>
              <ArrowLeft size={15} aria-hidden="true" /> {copy.back}
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="task-grid" role="group" aria-label={copy.title}>
            {groups.primary.map((task) => (
              <button type="button" className="task-card" key={task.question} onClick={() => onSelect(task)}>
                <span className="task-card-icon" aria-hidden="true">{React.createElement(ICONS[mode] || Search, { size: 17 })}</span>
                <span>{task.label}</span>
                <ArrowRight className="task-card-arrow" size={16} aria-hidden="true" />
              </button>
            ))}
          </div>
          {groups.other.length > 0 && (
            <details className="task-other">
              <summary><ChevronDown size={15} aria-hidden="true" /> {copy.other}</summary>
              <div className="task-other-list">
                {groups.other.map((task) => (
                  <button type="button" key={task.question} onClick={() => onSelect(task)}>{task.question}</button>
                ))}
              </div>
            </details>
          )}
          {researchContext && <div className="task-context">{researchContext}</div>}
          <button type="button" className="task-free-input" onClick={onFreeInput}>
            {copy.free} <span aria-hidden="true">↗</span>
          </button>
          {showFreeInput && <button type="button" className="task-back-to-list" onClick={onBackToTasks}><ArrowLeft size={14} aria-hidden="true" /> {copy.backToTasks}</button>}
          <button type="button" className="task-info-toggle" onClick={onToggleInfo} aria-expanded={showInfo} aria-controls="task-details">
            <SlidersHorizontal size={14} aria-hidden="true" /> {copy.info}
          </button>
          {showInfo && (
            <div className="task-details" id="task-details" role="region" aria-label={copy.info}>
              {qaGuidance && <p className="task-info-copy">{copy.qaInfo}</p>}
              {mode !== "research" && modeDescription && (
                <details className="task-info">
                  <summary>{copy.modeInfo}</summary>
                  <p>{modeDescription}</p>
                </details>
              )}
              {infoItems.length > 0 && (
                <details className="task-info">
                  <summary>{copy.ready}</summary>
                  <ul>{infoItems.map((item) => <li key={item}>{item}</li>)}</ul>
                </details>
              )}
              {qaGuidance && (
                <details className="task-info">
                  <summary>{copy.instructions}</summary>
                  <p>{qaGuidance}</p>
                </details>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}
