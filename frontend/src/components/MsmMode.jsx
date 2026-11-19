import React, { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import AgentPresence from "./AgentPresence.jsx";
import { getMessageMarkdownOptions } from "../markdown-tables";
import {
  buildMsmPayload,
  initialMsmState,
  reduceMsmEvent,
  streamMsmQuery,
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

export default function MsmMode({ employeeId, language, presence }) {
  const [turns, setTurns] = useState([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [cardState, setCardState] = useState({});
  const sessionIdRef = useRef("");
  const abortRef = useRef(null);

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
