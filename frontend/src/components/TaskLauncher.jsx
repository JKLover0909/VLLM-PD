import React from "react";
import {
  Activity, AlertTriangle, ArrowLeft, Boxes, Building2, Camera, ChartPie, Check, ChevronDown, ClipboardCheck,
  Cctv, Clock, CornerDownLeft, FileBarChart, FlaskConical, Gauge, GitCompare, Hash, History, Info, Keyboard,
  LayoutDashboard, Lightbulb, ListChecks, ListOrdered, Network, Package, Percent, RefreshCw, Scale, Search,
  Server, Tags, Thermometer,
} from "lucide-react";
import { spotlightVars, taskTransitionName } from "../motion.js";

const MODE_ICONS = { mkac: Building2, mes: Activity, wms: Boxes, cctvai: Cctv, qa: ClipboardCheck, research: FlaskConical };

// Keys come from task-launcher.js (resolveIconKey); unknown keys fall back to the mode icon.
const TASK_ICONS = {
  org: Network, clock: Clock, product: Package, alert: AlertTriangle, pie: ChartPie, ranking: ListOrdered,
  report: FileBarChart, compare: GitCompare, idea: Lightbulb, checklist: ListChecks, temperature: Thermometer,
  gauge: Gauge, scale: Scale, percent: Percent, version: Tags, records: History, count: Hash, refresh: RefreshCw,
  server: Server, overview: LayoutDashboard, camera: Camera,
};

const COPY = {
  vi: {
    title: "Tác vụ", free: "Tự nhập", other: "Gợi ý khác", info: "Thông tin chế độ", backToTasks: "Đóng ô nhập",
    instructions: "Hướng dẫn", qaInfo: "Câu đầu tiên nên có mã máy hoặc mã phiếu, thời gian và điều kiện",
    modeInfo: "Phạm vi", ready: "Trạng thái", run: "Thực hiện", back: "Quay lại", preview: "Câu hỏi sẽ gửi (có thể sửa)",
    enterHint: "Enter để thực hiện, Esc để quay lại",
  },
  ja: {
    title: "タスク", free: "自由入力", other: "その他の候補", info: "モード情報", backToTasks: "入力欄を閉じる",
    instructions: "質問のコツ", qaInfo: "最初の質問には設備またはチェックシートのコード、期間、条件を含めてください",
    modeInfo: "範囲", ready: "状態", run: "実行", back: "戻る", preview: "送信する質問（編集できます）",
    enterHint: "Enterで実行、Escで戻る",
  },
};

// Pointer light + small tilt; the values are CSS variables so reduced-motion
// and motion-off rules can neutralize them without touching React state.
function trackSpotlight(event) {
  if (event.pointerType === "touch") return;
  const card = event.currentTarget;
  const vars = spotlightVars(card.getBoundingClientRect(), event.clientX, event.clientY);
  if (!vars) return;
  for (const [name, value] of Object.entries(vars)) card.style.setProperty(name, value);
}

function resetSpotlight(event) {
  const card = event.currentTarget;
  ["--mx", "--my", "--tilt-x", "--tilt-y"].forEach((name) => card.style.removeProperty(name));
}

function TaskIcon({ mode, iconKey, size }) {
  const Icon = TASK_ICONS[iconKey] || MODE_ICONS[mode] || Search;
  return <Icon size={size} aria-hidden="true" />;
}

export default function TaskLauncher({
  mode, language, groups, selectedTask, previewQuestion, onPreviewChange, onSelect, onRun, onBack,
  onFreeInput, onBackToTasks, showFreeInput = false, qaGuidance, modeDescription, researchContext, infoItems = [],
  onToggleInfo, showInfo = false, children,
}) {
  const copy = COPY[language] || COPY.vi;
  const hasSelection = Boolean(selectedTask);
  // The chosen card and the preview share one view-transition name so the
  // browser morphs one into the other.
  const selectedIndex = hasSelection ? groups.primary.findIndex((task) => task.question === selectedTask.question) : -1;
  const previewName = taskTransitionName(selectedIndex) || "task-preview";

  return (
    <section className="task-launcher" aria-label={copy.title}>
      {children}
      {hasSelection ? (
        <div className="task-preview" style={{ viewTransitionName: previewName }}>
          <div className="task-preview-head">
            <span className="task-card-icon" aria-hidden="true"><TaskIcon mode={mode} iconKey={selectedTask.iconKey} size={17} /></span>
            <strong>{selectedTask.label}</strong>
          </div>
          <label className="sr-only" htmlFor="task-preview-question">{copy.preview}</label>
          <textarea
            id="task-preview-question"
            className="task-preview-question"
            value={previewQuestion}
            maxLength={mode === "qa" ? 2000 : 4000}
            aria-describedby="task-preview-hint"
            onChange={(event) => onPreviewChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") onBack();
              // The question must stay on one line, so Enter runs instead of adding a newline.
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                onRun(selectedTask, previewQuestion);
              }
            }}
            rows={Math.min(4, Math.max(2, Math.ceil(previewQuestion.length / (language === "ja" ? 24 : 40))))}
            autoFocus
          />
          <div className="task-preview-actions">
            <button type="button" className="task-back" onClick={onBack}>
              <ArrowLeft size={15} aria-hidden="true" /> {copy.back}
            </button>
            <span className="task-preview-hint" id="task-preview-hint">
              <CornerDownLeft size={12} aria-hidden="true" /> {copy.enterHint}
            </span>
            <button type="button" className="task-run" onClick={() => onRun(selectedTask, previewQuestion)}>
              <Check size={16} aria-hidden="true" /> {copy.run}
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="task-grid" role="group" aria-label={copy.title} style={{ "--task-cols": groups.primary.length <= 4 ? Math.max(groups.primary.length, 1) : 3 }}>
            {groups.primary.map((task, index) => (
              <button
                type="button"
                className="task-card"
                data-task-index={index}
                key={task.question}
                title={task.question}
                style={{ "--task-index": index, viewTransitionName: taskTransitionName(index) }}
                onPointerMove={trackSpotlight}
                onPointerLeave={resetSpotlight}
                onClick={() => onSelect(task)}
              >
                <span className="task-card-icon" aria-hidden="true"><TaskIcon mode={mode} iconKey={task.iconKey} size={18} /></span>
                <span className="task-card-label">{task.label}</span>
              </button>
            ))}
          </div>
          {researchContext && <div className="task-context">{researchContext}</div>}
          <div className="task-secondary">
            {showFreeInput ? (
              <button type="button" className="task-link" onClick={onBackToTasks}>
                <ArrowLeft size={14} aria-hidden="true" /> {copy.backToTasks}
              </button>
            ) : (
              <button type="button" className="task-link task-free-input" onClick={onFreeInput}>
                <Keyboard size={15} aria-hidden="true" /> {copy.free}
              </button>
            )}
            {groups.other.length > 0 && (
              <details className="task-other">
                <summary className="task-link"><ChevronDown size={14} aria-hidden="true" /> {copy.other} ({groups.other.length})</summary>
                <div className="task-other-list">
                  {groups.other.map((task) => (
                    <button type="button" key={task.question} onClick={() => onSelect(task)}>
                      <TaskIcon mode={mode} iconKey={task.iconKey} size={14} />
                      <span>{task.question}</span>
                    </button>
                  ))}
                </div>
              </details>
            )}
            <button
              type="button"
              className="task-link task-info-toggle"
              onClick={onToggleInfo}
              aria-expanded={showInfo}
              aria-controls="task-details"
              aria-label={copy.info}
              title={copy.info}
            >
              <Info size={15} aria-hidden="true" />
            </button>
          </div>
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
                <details className="task-info" open>
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
