import React, { useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Bot, Send, Sparkles, Square } from "lucide-react";
import AgentPresence from "./AgentPresence.jsx";
import TaskLauncher from "./TaskLauncher.jsx";
import { buildTaskGroups } from "../task-launcher.js";
import { getMessageMarkdownOptions } from "../markdown-tables";
import {
  buildMsmPayload,
  initialMsmState,
  reduceMsmEvent,
  streamMsmQuery,
  submitMsmForm,
  lookupMsmCategory,
} from "../msm-mode";
import "../msm-mode.css";

// Cùng bộ tuỳ chọn bảng Markdown với chế độ QA để bảng rộng cuộn ngang trong
// khung hội thoại thay vì bị cắt.
const MSM_MARKDOWN_OPTIONS = getMessageMarkdownOptions({ qa: true });

const COPY = {
  emptyTitle: "Quản lý thiết bị MSM",
  emptyBody:
    "Hỏi về máy, hạn hiệu chuẩn, kế hoạch MSA hoặc trạng thái thiết bị. Thao tác ghi luôn hiện thẻ chờ để bạn xác nhận trước khi ghi vào MSM.",
  inputLabel: "Câu hỏi về thiết bị MSM",
  placeholder: "Hỏi theo mã máy hoặc bộ phận, ví dụ: máy nào quá hạn hiệu chuẩn?",
  send: "Gửi",
  stop: "Dừng",
  sending: "Đang trả lời",
  processing: "Đang xử lý",
  chainTitle: "Các bước sẽ gọi",
  expires: "Hạn xác nhận",
  confirm: "Xác nhận",
  cancel: "Huỷ",
  loginRequired: "Nhập mã nhân viên ở một chế độ hỏi đáp trước khi dùng chế độ MSM.",
  misconfigured: "Chế độ MSM chưa được cấu hình. Vui lòng liên hệ quản trị viên.",
  sessionLost: "Phiên làm việc không còn trên máy chủ. Vui lòng hỏi lại.",
  requestFailed: "Không nhận được câu trả lời. Vui lòng thử lại.",
  network: "Không kết nối được máy chủ. Vui lòng thử lại.",
  expired: "Thẻ đã hết hạn, vui lòng hỏi lại.",
  handled: "Thao tác này đã được xử lý.",
  changed: "Dữ liệu đã thay đổi, thẻ được làm mới. Vui lòng kiểm tra lại trước khi xác nhận.",
  cancelled: "Đã huỷ thẻ, chưa ghi gì vào MSM.",
  done: "Đã thực hiện và kiểm chứng xong.",
  doneUnverified: "Đã thực hiện nhưng chưa kiểm chứng được. Vui lòng kiểm tra lại trong MSM.",
  notApplied: "Thao tác chưa được thực hiện. Vui lòng thử lại.",
  confirmFailed: "Không xác nhận được thẻ này. Vui lòng thử lại.",
  formTitle: "Biểu mẫu",
  formSubmit: "Tạo thao tác",
  formSubmitting: "Đang tạo…",
  formCancel: "Đóng biểu mẫu",
  formSuccess: "Đã tạo thao tác từ biểu mẫu. Vui lòng xem thẻ xác nhận bên dưới.",
  formCancelled: "Đã đóng biểu mẫu.",
  fieldRequired: "Vui lòng nhập trường này",
};

function newId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function errorCopy(error) {
  const message = error?.message || "";
  if (message === "MSM_AGENT_NOT_CONFIGURED" || error?.status === 503) return COPY.misconfigured;
  if (message === "SESSION_NOT_FOUND") return COPY.sessionLost;
  if (error?.code || error?.status) return message || COPY.requestFailed;
  return COPY.network;
}

function formatExpiry(value) {
  if (!value) return "";
  return String(value).replace("T", " ").slice(0, 16);
}

function escapeCell(value) {
  return String(value ?? "")
    .replace(/\|/g, "\\|")
    .replace(/\r?\n/g, " ");
}

// `views` là bảng do code của agent dựng ({columns, rows}); đổi sang bảng
// Markdown để dùng lại đúng đường hiển thị bảng của chế độ QA.
function viewToMarkdown(columns, rows) {
  const head = `| ${columns.map(escapeCell).join(" | ")} |`;
  const divider = `| ${columns.map(() => "---").join(" | ")} |`;
  const body = rows.map(
    (row) => `| ${columns.map((_, index) => escapeCell(row?.[index])).join(" | ")} |`,
  );
  return [head, divider, ...body].join("\n");
}

function MsmTableView({ view }) {
  if (!view || view.kind !== "table") return null;
  const columns = Array.isArray(view.columns) ? view.columns : [];
  const rows = Array.isArray(view.rows) ? view.rows : [];
  if (!columns.length) return null;
  return (
    <figure className="msm-view">
      {view.title ? <figcaption>{view.title}</figcaption> : null}
      <ReactMarkdown {...MSM_MARKDOWN_OPTIONS}>{viewToMarkdown(columns, rows)}</ReactMarkdown>
      {Number(view.total) > rows.length ? (
        <p className="msm-view-note">
          Hiển thị {rows.length}/{view.total} dòng.
        </p>
      ) : null}
    </figure>
  );
}

function WizardLookupField({ field, value, onChange, employeeId, hasError, onKeyDown, inputRef }) {
  const [options, setOptions] = useState([]);
  const [search, setSearch] = useState("");
  const source = field?.source || field?.lookup;

  React.useEffect(() => {
    let active = true;
    if (source) {
      lookupMsmCategory({ kind: source, q: "", employeeId })
        .then((items) => {
          if (active && Array.isArray(items)) {
            setOptions(items);
          }
        })
        .catch(() => {});
    } else if (Array.isArray(field?.options)) {
      setOptions(
        field.options.map((opt) =>
          typeof opt === "object" && opt !== null
            ? opt
            : { value: opt, label: field.option_labels?.[opt] || opt }
        )
      );
    }
    return () => {
      active = false;
    };
  }, [source, employeeId, field?.options]);

  const filtered = search.trim()
    ? options.filter((opt) => {
        const text = `${opt.label || ""} ${opt.value || ""}`.toLowerCase();
        return text.includes(search.toLowerCase());
      })
    : options;

  return (
    <div className="msm-wizard-lookup">
      <input
        ref={inputRef}
        type="text"
        id={`wizard-${field.name || field.key}`}
        className={`msm-form-input msm-wizard-input ${hasError ? "has-error" : ""}`}
        value={value ?? ""}
        placeholder={field.placeholder || `Tìm hoặc chọn ${(field.title || field.label || field.name || field.key || "").toLowerCase()}...`}
        autoFocus
        onChange={(e) => {
          onChange(e.target.value);
          setSearch(e.target.value);
        }}
        onKeyDown={onKeyDown}
      />
      {filtered.length > 0 ? (
        <div className="msm-wizard-suggestions">
          <span className="msm-wizard-suggestions-label">Gợi ý nhanh:</span>
          <div className="msm-wizard-suggestion-chips">
            {filtered.slice(0, 10).map((opt) => {
              const isSelected = value === opt.value;
              return (
                <button
                  key={String(opt.value)}
                  type="button"
                  className={`msm-wizard-suggestion-chip ${isSelected ? "selected" : ""}`}
                  onClick={() => {
                    onChange(opt.value);
                    inputRef.current?.focus();
                  }}
                >
                  {opt.label || opt.value}
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function MsmForm({ form, employeeId, onSubmit, onCancel }) {
  const schema = form?.schema || {};
  const fields = Array.isArray(schema.fields) ? schema.fields : [];
  const [currentStep, setCurrentStep] = useState(0);
  const [values, setValues] = useState(() => {
    const init = { ...(form?.initial || {}) };
    for (const f of fields) {
      const k = f.name || f.key;
      if (init[k] === undefined && f.default !== null && f.default !== undefined) {
        init[k] = f.default;
      }
    }
    return init;
  });
  const [errors, setErrors] = useState(() => form?.errors || {});
  const [formError, setFormError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const inputRef = useRef(null);

  const safeStep = Math.min(Math.max(0, currentStep), Math.max(0, fields.length - 1));
  const field = fields[safeStep];
  const fieldKey = field?.name || field?.key || "";
  const fieldTitle = field?.title || field?.label || field?.name || field?.key || "";
  const isRequired = Boolean(field?.required);
  const isLastStep = safeStep >= fields.length - 1;

  React.useEffect(() => {
    inputRef.current?.focus();
  }, [safeStep]);

  function handleChange(key, val) {
    setValues((prev) => ({ ...prev, [key]: val }));
    if (errors[key]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  }

  function validateStep(idx, targetValues = values) {
    const f = fields[idx];
    if (!f) return null;
    const k = f.name || f.key;
    const val = targetValues[k];
    if (f.required && (val === undefined || val === null || String(val).trim() === "")) {
      return COPY.fieldRequired;
    }
    return null;
  }

  async function submitFormWithValues(targetValues = values) {
    const newErrors = {};
    let firstErrStep = -1;
    for (let i = 0; i < fields.length; i++) {
      const err = validateStep(i, targetValues);
      if (err) {
        const k = fields[i].name || fields[i].key;
        newErrors[k] = err;
        if (firstErrStep === -1) firstErrStep = i;
      }
    }
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      if (firstErrStep !== -1) setCurrentStep(firstErrStep);
      return;
    }
    setSubmitting(true);
    setFormError("");
    try {
      await onSubmit(form.kind, targetValues);
    } catch (err) {
      if (err.errors) {
        setErrors(err.errors);
        const firstErrKey = Object.keys(err.errors)[0];
        const errIdx = fields.findIndex((f) => (f.name || f.key) === firstErrKey);
        if (errIdx >= 0) setCurrentStep(errIdx);
      }
      setFormError(err.message || COPY.requestFailed);
    } finally {
      setSubmitting(false);
    }
  }

  function handleNext() {
    const err = validateStep(safeStep);
    if (err) {
      setErrors((prev) => ({ ...prev, [fieldKey]: err }));
      return;
    }
    if (!isLastStep) {
      setCurrentStep((s) => s + 1);
    } else {
      submitFormWithValues();
    }
  }

  function handleKeyDown(event) {
    if (event.nativeEvent?.isComposing || event.keyCode === 229) return;
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleNext();
    }
  }

  function handleChoiceClick(optValue) {
    handleChange(fieldKey, optValue);
    const nextValues = { ...values, [fieldKey]: optValue };
    if (!isLastStep) {
      setCurrentStep((s) => s + 1);
    } else {
      submitFormWithValues(nextValues);
    }
  }

  if (fields.length === 0) {
    return (
      <div className="msm-form-card msm-wizard">
        <h3 className="msm-form-title">{schema.title || form.kind || COPY.formTitle}</h3>
        {formError ? <div className="msm-form-general-error" role="alert">{formError}</div> : null}
        <div className="msm-wizard-actions">
          <button
            type="button"
            className="msm-wizard-btn msm-wizard-next"
            disabled={submitting}
            onClick={() => submitFormWithValues()}
          >
            {submitting ? COPY.formSubmitting : "Hoàn tất"}
          </button>
          <button
            type="button"
            className="msm-wizard-btn msm-wizard-cancel"
            onClick={onCancel}
            disabled={submitting}
          >
            {COPY.formCancel}
          </button>
        </div>
      </div>
    );
  }

  const isLookup = Boolean(field?.source || field?.lookup || field?.datalist || field?.type === "lookup");
  const isChoice = !isLookup && Boolean(
    field?.type === "bool" ||
    field?.type === "boolean" ||
    (Array.isArray(field?.enum) && field.enum.length > 0) ||
    (Array.isArray(field?.options) && field.options.length > 0)
  );

  let choiceOptions = [];
  if (isChoice) {
    if (field.type === "bool" || field.type === "boolean") {
      choiceOptions = [
        { value: true, label: "Có / Đạt" },
        { value: false, label: "Không" },
      ];
    } else if (Array.isArray(field.enum)) {
      choiceOptions = field.enum.map((opt) =>
        typeof opt === "object" && opt !== null ? opt : { value: opt, label: field.option_labels?.[opt] || opt }
      );
    } else if (Array.isArray(field.options)) {
      choiceOptions = field.options.map((opt) =>
        typeof opt === "object" && opt !== null ? opt : { value: opt, label: field.option_labels?.[opt] || opt }
      );
    }
  }

  const hasError = Boolean(errors[fieldKey]);

  return (
    <form
      className="msm-form-card msm-wizard"
      onSubmit={(e) => {
        e.preventDefault();
        handleNext();
      }}
      noValidate
    >
      <div className="msm-wizard-header">
        <h3 className="msm-form-title">{schema.title || form.kind || COPY.formTitle}</h3>
        <div className="msm-wizard-step">
          [Bước {safeStep + 1}/{fields.length}]: {fieldTitle}
          {isRequired ? <span className="required"> *</span> : null}
        </div>
        {fields.length > 1 ? (
          <div
            className="msm-wizard-progress"
            role="progressbar"
            aria-valuenow={safeStep + 1}
            aria-valuemin={1}
            aria-valuemax={fields.length}
          >
            <div
              className="msm-wizard-progress-bar"
              style={{ width: `${((safeStep + 1) / fields.length) * 100}%` }}
            />
          </div>
        ) : null}
      </div>

      {formError ? <div className="msm-form-general-error" role="alert">{formError}</div> : null}

      <div className="msm-wizard-body">
        {field.description ? (
          <p className="msm-wizard-field-desc">{field.description}</p>
        ) : null}

        {isChoice ? (
          <div className="msm-wizard-options" role="radiogroup" aria-label={fieldTitle}>
            {choiceOptions.map((opt) => {
              const isSelected = values[fieldKey] === opt.value;
              return (
                <button
                  key={String(opt.value)}
                  type="button"
                  className={`msm-wizard-option-btn ${isSelected ? "selected" : ""}`}
                  onClick={() => handleChoiceClick(opt.value)}
                  disabled={submitting}
                >
                  <span className="msm-wizard-option-check" aria-hidden="true" />
                  <span className="msm-wizard-option-text">{opt.label}</span>
                </button>
              );
            })}
          </div>
        ) : isLookup ? (
          <WizardLookupField
            key={fieldKey}
            field={field}
            value={values[fieldKey]}
            employeeId={employeeId}
            hasError={hasError}
            inputRef={inputRef}
            onChange={(val) => handleChange(fieldKey, val)}
            onKeyDown={handleKeyDown}
          />
        ) : (
          <input
            key={fieldKey}
            ref={inputRef}
            type={
              field.type === "date"
                ? "date"
                : field.type === "int" || field.type === "number" || field.type === "float"
                  ? "number"
                  : "text"
            }
            id={`wizard-${fieldKey}`}
            className={`msm-form-input msm-wizard-input ${hasError ? "has-error" : ""}`}
            value={values[fieldKey] ?? ""}
            readOnly={Boolean(field.readonly)}
            placeholder={field.placeholder || `Nhập ${fieldTitle.toLowerCase()}...`}
            autoFocus
            onChange={(e) => {
              const raw = e.target.value;
              const val =
                field.type === "int" || field.type === "number"
                  ? raw === "" ? "" : Number(raw)
                  : raw;
              handleChange(fieldKey, val);
            }}
            onKeyDown={handleKeyDown}
          />
        )}

        {hasError ? (
          <span className="msm-form-error" role="alert" style={{ display: "block", marginTop: "6px" }}>
            {errors[fieldKey]}
          </span>
        ) : null}
      </div>

      <div className="msm-wizard-actions">
        {safeStep > 0 ? (
          <button
            type="button"
            className="msm-wizard-btn msm-wizard-back"
            onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
            disabled={submitting}
          >
            Quay lại
          </button>
        ) : null}
        <button
          type="submit"
          className="msm-wizard-btn msm-wizard-next"
          disabled={submitting}
        >
          {submitting
            ? COPY.formSubmitting
            : isLastStep
              ? "Hoàn tất"
              : "Tiếp tục"}
        </button>
        <button
          type="button"
          className="msm-wizard-btn msm-wizard-cancel"
          onClick={onCancel}
          disabled={submitting}
        >
          {COPY.formCancel}
        </button>
      </div>
    </form>
  );
}

export default function MsmMode({ employeeId, language, presence, ReportArtifactCard, onTurnCountChange, onPreviewOpenChange }) {
  const [turns, setTurns] = useState([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [cardState, setCardState] = useState({});
  const [selectedTask, setSelectedTask] = useState(null);
  const [taskPreviewDraft, setTaskPreviewDraft] = useState("");
  const [showFreeInput, setShowFreeInput] = useState(false);
  const [taskInfoOpen, setTaskInfoOpen] = useState(false);
  const textareaRef = useRef(null);
  const sessionIdRef = useRef("");
  const abortRef = useRef(null);

  const starterPrompts = useMemo(() => {
    return language === "ja"
      ? [
          { question: "機器校正エグゼクティブレポート" },
          { question: "至急対応が必要な校正タスク一覧" },
          { question: "今後12ヶ月の校正負荷予測" },
          { question: "新規設備登録フォームを開いてください" },
          { question: "校正期限が近づいている設備はどれですか？" },
          { question: "ANNEALING部門の設備状態を確認してください" },
          { question: "本日の機器サマリーを教えてください" },
          { question: "本日の設備全体状況はどうなっていますか？" },
          { question: "校正期限が切れている設備はありますか？" },
          { question: "30日以内に校正期限を迎える設備はどれですか？" },
          { question: "今月の校正計画はどうなっていますか？" },
        ]
      : [
          { question: "Báo cáo điều hành hiệu chuẩn thiết bị" },
          { question: "Danh mục việc khẩn cấp cần xử lý ngay" },
          { question: "Dự báo tải hiệu chuẩn 12 tháng tới" },
          { question: "Chuẩn bị form thêm thiết bị mới" },
          { question: "Máy nào sắp đến hạn hiệu chuẩn?" },
          { question: "Kiểm tra trạng thái thiết bị bộ phận ANNEALING" },
          { question: "Bản tin hôm nay của tôi có gì?" },
          { question: "Tổng quan thiết bị hôm nay" },
          { question: "Máy nào đã quá hạn hiệu chuẩn?" },
          { question: "Máy nào đến hạn hiệu chuẩn trong 30 ngày tới?" },
          { question: "Kế hoạch hiệu chuẩn tháng này như thế nào?" },
        ];
  }, [language]);

  const taskGroups = useMemo(
    () => buildTaskGroups("msm", starterPrompts, language),
    [starterPrompts, language],
  );

  React.useEffect(() => {
    onTurnCountChange?.(turns.length);
  }, [turns.length, onTurnCountChange]);

  React.useEffect(() => {
    onPreviewOpenChange?.(Boolean(selectedTask));
  }, [selectedTask, onPreviewOpenChange]);

  React.useEffect(() => {
    if (showFreeInput) {
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }, [showFreeInput]);

  function patchTurn(turnId, updater) {
    setTurns((current) =>
      current.map((turn) => (turn.id === turnId ? updater(turn) : turn)),
    );
  }

  function setCard(actionId, patch) {
    setCardState((current) => ({
      ...current,
      [actionId]: { ...current[actionId], ...patch },
    }));
  }

  function dropCard(turnId, actionId, notice) {
    patchTurn(turnId, (turn) => ({
      ...turn,
      pending: turn.pending.filter((item) => item.action_id !== actionId),
      notice: notice || turn.notice,
    }));
  }

  async function submitQuestion(q) {
    const question = (q !== undefined ? q : draft).trim();
    if (!question || !employeeId || streaming) return;
    if (!sessionIdRef.current) sessionIdRef.current = newId();
    const turnId = newId();
    setTurns((current) => [
      ...current,
      { id: turnId, question, notice: "", ...initialMsmState() },
    ]);
    setDraft("");
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamMsmQuery(
        buildMsmPayload({
          sessionId: sessionIdRef.current,
          question,
          employeeId,
        }),
        (item) => patchTurn(turnId, (turn) => reduceMsmEvent(turn, item)),
        controller.signal,
      );
    } catch (error) {
      if (error?.name !== "AbortError") {
        patchTurn(turnId, (turn) =>
          reduceMsmEvent(turn, {
            type: "error",
            code: error?.code || "",
            message: errorCopy(error),
          }),
        );
      }
    } finally {
      abortRef.current = null;
      setStreaming(false);
    }
  }

  function submit(event) {
    event.preventDefault();
    submitQuestion(draft);
  }

  function onKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  async function handleFormSubmit(turnId, form, kind, values) {
    if (!sessionIdRef.current) sessionIdRef.current = newId();
    const cleanedValues = {};
    for (const [k, v] of Object.entries(values)) {
      if (v !== undefined && v !== null && v !== "") {
        cleanedValues[k] = v;
      }
    }
    const res = await submitMsmForm({
      sessionId: sessionIdRef.current,
      employeeId,
      kind,
      values: cleanedValues,
      actionId: form.action_id || null,
    });
    const newPending = Array.isArray(res.pending) ? res.pending : (res.pending ? [res.pending] : []);
    patchTurn(turnId, (turn) => ({
      ...turn,
      forms: (turn.forms || []).filter((f) => f !== form),
      pending: [...turn.pending, ...newPending],
      notice: COPY.formSuccess,
    }));
  }

  function handleFormCancel(turnId, formIndex) {
    patchTurn(turnId, (turn) => ({
      ...turn,
      forms: (turn.forms || []).filter((_, idx) => idx !== formIndex),
      notice: COPY.formCancelled,
    }));
  }

  async function confirmAction(turnId, action) {
    const actionId = action.action_id;
    setCard(actionId, { busy: true, note: "" });
    try {
      const response = await fetch(
        `/msm/actions/${encodeURIComponent(actionId)}/confirm`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionIdRef.current, employee_id: employeeId }),
        },
      );
      if (response.status === 410) {
        dropCard(turnId, actionId, COPY.expired);
        return;
      }
      if (response.status === 404) {
        dropCard(turnId, actionId, COPY.handled);
        return;
      }
      if (response.status === 409) {
        const body = await response.json().catch(() => ({}));
        patchTurn(turnId, (turn) => ({
          ...turn,
          pending: [
            ...turn.pending.filter((item) => item.action_id !== actionId),
            ...(Array.isArray(body.pending) ? body.pending : []),
          ],
          notice: COPY.changed,
        }));
        return;
      }
      if (!response.ok) {
        setCard(actionId, { busy: false, note: COPY.confirmFailed });
        return;
      }
      const body = await response.json().catch(() => ({}));
      const verified = Boolean(body?.result?.verified);
      const ok = Boolean(body?.result?.ok);
      const fallback = ok ? (verified ? COPY.done : COPY.doneUnverified) : COPY.notApplied;
      dropCard(turnId, actionId, body.answer || fallback);
    } catch {
      setCard(actionId, { busy: false, note: COPY.confirmFailed });
    }
  }

  async function cancelAction(turnId, action) {
    const actionId = action.action_id;
    setCard(actionId, { busy: true, note: "" });
    try {
      await fetch(`/msm/actions/${encodeURIComponent(actionId)}/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionIdRef.current, employee_id: employeeId }),
      });
    } catch {
      // Huỷ không ghi gì vào MSM: thẻ rời khỏi giao diện dù máy chủ không phản hồi.
    }
    dropCard(turnId, actionId, COPY.cancelled);
  }

  const disabled = !employeeId || streaming;
  const isComposerOpen = turns.length > 0 || showFreeInput;

  return (
    <div className="msm-mode" lang={language === "ja" ? "ja" : "vi"} aria-label={COPY.emptyTitle}>
      <div className="msm-scroll">
        {turns.length === 0 ? (
          <div className="empty-conversation">
            <div className="empty-copy">
              <AgentPresence presence={presence} mode="msm" language={language} />
            </div>
            <TaskLauncher
              mode="msm"
              language={language}
              groups={taskGroups}
              selectedTask={selectedTask}
              previewQuestion={taskPreviewDraft}
              onPreviewChange={setTaskPreviewDraft}
              onSelect={(task) => {
                setSelectedTask(task);
                setTaskPreviewDraft(task.question);
              }}
              onRun={(task) => {
                setSelectedTask(null);
                submitQuestion(task.question);
              }}
              onBack={() => {
                setSelectedTask(null);
                setTaskPreviewDraft("");
              }}
              onFreeInput={() => {
                setSelectedTask(null);
                setShowFreeInput(true);
              }}
              onBackToTasks={() => {
                setShowFreeInput(false);
              }}
              showFreeInput={showFreeInput}
              onToggleInfo={() => setTaskInfoOpen((o) => !o)}
              showInfo={taskInfoOpen}
              modeDescription={COPY.emptyBody}
              infoItems={
                language === "ja"
                  ? [
                      "設備、校正期限、設備状態の照会",
                      "校正計画、MSA、期限切れアラート",
                      "データ保護のための確認カード付き書き込み操作",
                    ]
                  : [
                      "Tra cứu máy, hạn hiệu chuẩn và trạng thái thiết bị",
                      "Kế hoạch hiệu chuẩn, MSA và cảnh báo quá hạn",
                      "Soạn thao tác ghi có thẻ xác nhận bảo vệ dữ liệu",
                    ]
              }
            />
          </div>
        ) : null}
        {turns.map((turn) => (
          <div className="msm-turn" key={turn.id}>
            <article className="message user">
              <div className="message-avatar">U</div>
              <div className="message-content">{turn.question}</div>
            </article>
            <article className="message assistant">
              <div className="message-avatar">
                <Bot size={18} />
              </div>
              <div className="message-content">
                {turn.answer ? (
                  <ReactMarkdown {...MSM_MARKDOWN_OPTIONS}>{turn.answer}</ReactMarkdown>
                ) : null}
                {turn.artifact && ReportArtifactCard ? (
                  <ReportArtifactCard
                    artifact={turn.artifact}
                    language={language}
                    hideEmail={true}
                    hideDownload={true}
                  />
                ) : null}
                {!turn.done && !turn.error ? (
                  <div className="msm-progress" role="status">
                    {turn.meta || COPY.processing}
                  </div>
                ) : null}
                {turn.error ? (
                  <p className="msm-error" role="alert">
                    {turn.error.message}
                  </p>
                ) : null}
                {turn.views.map((view, index) => (
                  <MsmTableView key={`${turn.id}-view-${index}`} view={view} />
                ))}
                {Array.isArray(turn.forms) && turn.forms.map((form, index) => (
                  <MsmForm
                    key={`${turn.id}-form-${index}`}
                    form={form}
                    employeeId={employeeId}
                    onSubmit={(kind, values) => handleFormSubmit(turn.id, form, kind, values)}
                    onCancel={() => handleFormCancel(turn.id, index)}
                  />
                ))}
                {turn.pending.map((action) => {
                  const state = cardState[action.action_id] || {};
                  return (
                    <div className="msm-pending" key={action.action_id}>
                      <p className="msm-pending-summary">{action.summary}</p>
                      {Array.isArray(action.api_chain) && action.api_chain.length > 0 ? (
                        <>
                          <p className="msm-pending-label">{COPY.chainTitle}</p>
                          <ul className="msm-pending-chain">
                            {action.api_chain.map((step, index) => (
                              <li key={`${action.action_id}-step-${index}`}>{step}</li>
                            ))}
                          </ul>
                        </>
                      ) : null}
                      {action.expires_at ? (
                        <p className="msm-pending-expiry">
                          {COPY.expires}: {formatExpiry(action.expires_at)}
                        </p>
                      ) : null}
                      {state.note ? <p className="msm-error">{state.note}</p> : null}
                      <div className="msm-pending-actions">
                        <button
                          type="button"
                          className="msm-confirm"
                          onClick={() => confirmAction(turn.id, action)}
                          disabled={Boolean(state.busy)}
                        >
                          {COPY.confirm}
                        </button>
                        <button
                          type="button"
                          className="msm-cancel"
                          onClick={() => cancelAction(turn.id, action)}
                          disabled={Boolean(state.busy)}
                        >
                          {COPY.cancel}
                        </button>
                      </div>
                    </div>
                  );
                })}
                {turn.notice ? <p className="msm-notice">{turn.notice}</p> : null}
                {Array.isArray(turn.pills) && turn.pills.length > 0 ? (
                  <div className="msm-pills" role="group" aria-label={language === "ja" ? "次の候補" : "Gợi ý tiếp theo"}>
                    <p className="msm-pills-label">
                      <Sparkles size={13} aria-hidden="true" />
                      <span>{language === "ja" ? "次の候補:" : "Gợi ý tiếp theo:"}</span>
                    </p>
                    <div className="msm-pills-list">
                      {turn.pills.map((pill, idx) => (
                        <button
                          key={`${turn.id}-pill-${idx}`}
                          type="button"
                          className="msm-pill"
                          onClick={() => submitQuestion(pill)}
                          disabled={streaming}
                        >
                          {pill}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            </article>
          </div>
        ))}
      </div>
      {isComposerOpen ? (
        <form className="composer" onSubmit={submit}>
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={onKeyDown}
            aria-label={COPY.inputLabel}
            placeholder={COPY.placeholder}
            rows={2}
            disabled={disabled}
          />
          <div className="composer-footer">
            <div className="composer-context" />
            <div className="composer-actions">
              <button
                className={streaming ? "send-button stopping" : "send-button"}
                type={streaming ? "button" : "submit"}
                title={streaming ? COPY.stop : COPY.send}
                onClick={streaming ? () => abortRef.current?.abort() : undefined}
                disabled={streaming ? false : !draft.trim() || disabled}
              >
                {streaming ? (
                  <Square size={16} fill="currentColor" />
                ) : (
                  <Send size={18} />
                )}
              </button>
            </div>
          </div>
        </form>
      ) : null}
      {!employeeId && isComposerOpen ? (
        <p className="msm-hint" role="status">
          {COPY.loginRequired}
        </p>
      ) : null}
    </div>
  );
}
