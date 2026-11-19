import React, { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import AgentPresence from "./AgentPresence.jsx";
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

function LookupInput({ field, value, onChange, employeeId, hasError }) {
  const [options, setOptions] = useState([]);
  const listId = `lookup-${field.name}`;

  React.useEffect(() => {
    let active = true;
    lookupMsmCategory({ kind: field.source, q: "", employeeId }).then((items) => {
      if (active && Array.isArray(items)) {
        setOptions(items);
      }
    }).catch(() => {});
    return () => {
      active = false;
    };
  }, [field.source, employeeId]);

  return (
    <>
      <input
        type="text"
        list={listId}
        id={`field-${field.name}`}
        className={`msm-form-input ${hasError ? "has-error" : ""}`}
        value={value || ""}
        placeholder={`Chọn hoặc gõ ${field.label.toLowerCase()}`}
        onChange={(e) => onChange(e.target.value)}
      />
      <datalist id={listId}>
        {options.map((opt, idx) => (
          <option key={`${opt.value}-${idx}`} value={opt.value}>
            {opt.label || opt.value}
          </option>
        ))}
      </datalist>
    </>
  );
}

function MsmForm({ form, employeeId, onSubmit, onCancel }) {
  const schema = form?.schema || {};
  const fields = Array.isArray(schema.fields) ? schema.fields : [];
  const [values, setValues] = useState(() => {
    const init = { ...(form?.initial || {}) };
    for (const f of fields) {
      if (init[f.name] === undefined && f.default !== null && f.default !== undefined) {
        init[f.name] = f.default;
      }
    }
    return init;
  });
  const [errors, setErrors] = useState(() => form?.errors || {});
  const [formError, setFormError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  function handleChange(name, val) {
    setValues((prev) => ({ ...prev, [name]: val }));
    if (errors[name]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const newErrors = {};
    for (const f of fields) {
      if (f.required && (values[f.name] === undefined || values[f.name] === null || String(values[f.name]).trim() === "")) {
        newErrors[f.name] = COPY.fieldRequired;
      }
    }
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    setSubmitting(true);
    setFormError("");
    try {
      await onSubmit(form.kind, values);
    } catch (err) {
      if (err.errors) {
        setErrors(err.errors);
      }
      setFormError(err.message || COPY.requestFailed);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form className="msm-form-card" onSubmit={handleSubmit} noValidate>
      <h3 className="msm-form-title">{schema.title || form.kind || COPY.formTitle}</h3>
      {formError ? <div className="msm-form-general-error" role="alert">{formError}</div> : null}
      <div className="msm-form-grid">
        {fields.map((field) => {
          const hasError = Boolean(errors[field.name]);
          return (
            <div className="msm-form-group" key={field.name}>
              <label htmlFor={`field-${field.name}`}>
                {field.label}
                {field.required ? <span className="required">*</span> : null}
              </label>
              {field.source ? (
                <LookupInput
                  field={field}
                  value={values[field.name]}
                  employeeId={employeeId}
                  hasError={hasError}
                  onChange={(val) => handleChange(field.name, val)}
                />
              ) : field.type === "select" && Array.isArray(field.options) ? (
                <select
                  id={`field-${field.name}`}
                  className={`msm-form-select ${hasError ? "has-error" : ""}`}
                  value={values[field.name] ?? ""}
                  onChange={(e) => handleChange(field.name, e.target.value)}
                >
                  <option value="">-- Chọn {field.label} --</option>
                  {field.options.map((opt) => (
                    <option key={opt} value={opt}>
                      {field.option_labels?.[opt] || opt}
                    </option>
                  ))}
                </select>
              ) : field.type === "bool" ? (
                <select
                  id={`field-${field.name}`}
                  className={`msm-form-select ${hasError ? "has-error" : ""}`}
                  value={values[field.name] ? "true" : "false"}
                  onChange={(e) => handleChange(field.name, e.target.value === "true")}
                >
                  <option value="true">Có / Đạt</option>
                  <option value="false">Không</option>
                </select>
              ) : field.type === "date" ? (
                <input
                  type="date"
                  id={`field-${field.name}`}
                  className={`msm-form-input ${hasError ? "has-error" : ""}`}
                  value={values[field.name] || ""}
                  onChange={(e) => handleChange(field.name, e.target.value)}
                />
              ) : field.type === "int" ? (
                <input
                  type="number"
                  id={`field-${field.name}`}
                  className={`msm-form-input ${hasError ? "has-error" : ""}`}
                  value={values[field.name] ?? ""}
                  onChange={(e) => handleChange(field.name, e.target.value === "" ? "" : Number(e.target.value))}
                />
              ) : (
                <input
                  type="text"
                  id={`field-${field.name}`}
                  className={`msm-form-input ${hasError ? "has-error" : ""}`}
                  value={values[field.name] ?? ""}
                  readOnly={Boolean(field.readonly)}
                  onChange={(e) => handleChange(field.name, e.target.value)}
                />
              )}
              {hasError ? (
                <span className="msm-form-error" role="alert">
                  {errors[field.name]}
                </span>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="msm-form-actions">
        <button
          type="submit"
          className="msm-form-submit"
          disabled={submitting}
        >
          {submitting ? COPY.formSubmitting : COPY.formSubmit}
        </button>
        <button
          type="button"
          className="msm-form-cancel"
          disabled={submitting}
          onClick={onCancel}
        >
          {COPY.formCancel}
        </button>
      </div>
    </form>
  );
}

export default function MsmMode({ employeeId, language, presence, onTurnCountChange }) {
  const [turns, setTurns] = useState([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [cardState, setCardState] = useState({});
  const sessionIdRef = useRef("");
  const abortRef = useRef(null);

  React.useEffect(() => {
    onTurnCountChange?.(turns.length);
  }, [turns.length, onTurnCountChange]);

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

  async function submit(event) {
    event.preventDefault();
    const question = draft.trim();
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

  function onKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  async function handleFormSubmit(turnId, form, kind, values) {
    if (!sessionIdRef.current) sessionIdRef.current = newId();
    const res = await submitMsmForm({
      sessionId: sessionIdRef.current,
      employeeId,
      kind,
      values,
      actionId: form.action_id || null,
    });
    patchTurn(turnId, (turn) => ({
      ...turn,
      forms: (turn.forms || []).filter((f) => f !== form),
      pending: [
        ...turn.pending,
        ...(Array.isArray(res.pending) ? res.pending : []),
      ],
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

  return (
    <div className="msm-mode" lang={language === "ja" ? "ja" : "vi"} aria-label={COPY.emptyTitle}>
      <div className="msm-scroll">
        {turns.length === 0 ? (
          <div className="empty-conversation">
            <div className="empty-copy">
              <AgentPresence presence={presence} mode="msm" language={language} />
              <strong>{COPY.emptyTitle}</strong>
              <p>{COPY.emptyBody}</p>
            </div>
          </div>
        ) : null}
        {turns.map((turn) => (
          <div className="msm-turn" key={turn.id}>
            <article className="message user">
              <div className="message-avatar">U</div>
              <div className="message-content">{turn.question}</div>
            </article>
            <article className="message assistant">
              <div className="message-avatar">MSM</div>
              <div className="message-content">
                {turn.answer ? (
                  <ReactMarkdown {...MSM_MARKDOWN_OPTIONS}>{turn.answer}</ReactMarkdown>
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
              </div>
            </article>
          </div>
        ))}
      </div>
      <form className="composer" onSubmit={submit}>
        <textarea
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
              type="submit"
              className="send-button"
              disabled={disabled || !draft.trim()}
            >
              {streaming ? COPY.sending : COPY.send}
            </button>
          </div>
        </div>
      </form>
      {!employeeId ? (
        <p className="msm-hint" role="status">
          {COPY.loginRequired}
        </p>
      ) : null}
    </div>
  );
}
