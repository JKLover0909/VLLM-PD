import { streamQuery } from "./query-stream.js";

export function shouldSubmitOnEnter(event) {
  return event?.key === "Enter" && !event.shiftKey && !event.nativeEvent?.isComposing && event.keyCode !== 229;
}

// Tin nhắn gửi proxy MSM: meibook giữ session_id của chính nó, server map sang
// thread của agent. Câu hỏi cắt khoảng trắng đầu/cuối trước khi gửi.
export function buildMsmPayload({ sessionId, question, employeeId }) {
  return {
    session_id: sessionId,
    question: String(question || "").trim(),
    employee_id: employeeId,
  };
}

export function initialMsmState() {
  return {
    answer: "",
    meta: "",
    pending: [],
    views: [],
    forms: [],
    pills: [],
    artifact: null,
    error: null,
    done: false,
    stopped: false,
  };
}

// Sự kiện SSE của proxy đã được dịch về từ vựng meibook (token/meta/done/error),
// nên bộ giảm trạng thái ở đây chỉ gom lại thành một lượt trả lời.
export function reduceMsmEvent(state, event) {
  switch (event?.type) {
    case "token":
      return { ...state, answer: state.answer + (event.content || "") };
    case "meta":
      return { ...state, meta: event.label || "" };
    case "stopped":
      return { ...state, done: true, stopped: true };
    case "done": {
      const result = event.result || {};
      return {
        ...state,
        answer: state.answer || result.answer || "",
        pending: Array.isArray(result.pending) ? result.pending : [],
        views: Array.isArray(result.views) ? result.views : [],
        forms: Array.isArray(result.forms) ? result.forms : [],
        pills: Array.isArray(result.pills) ? result.pills : [],
        artifact: result.artifact || null,
        done: true,
      };
    }
    case "error":
      return {
        ...state,
        error: { code: event.code || "", message: event.message || "" },
      };
    default:
      return state;
  }
}

export async function submitMsmForm({ sessionId, employeeId, kind, values, actionId }, fetchImpl = fetch) {
  const response = await fetchImpl("/msm/forms/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      employee_id: employeeId,
      kind,
      values,
      action_id: actionId || null,
    }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail && typeof data.detail === "object" ? data.detail : {};
    const message = data.message || detail.message || (typeof data.detail === "string" ? data.detail : "") || "Lỗi nộp biểu mẫu";
    const error = new Error(message);
    error.status = response.status;
    error.errors = data.errors || detail.errors || {};
    throw error;
  }
  return data;
}

export async function lookupMsmCategory({ kind, q = "", employeeId = "" }, fetchImpl = fetch) {
  const params = new URLSearchParams({ kind, q });
  if (employeeId) params.set("employee_id", employeeId);
  const response = await fetchImpl(`/msm/lookup?${params.toString()}`);
  if (!response.ok) return [];
  return response.json().catch(() => []);
}

export async function streamMsmQuery(payload, onEvent, signal, fetchImpl = fetch) {
  return streamQuery(payload, onEvent, signal, fetchImpl, "/msm/query/stream");
}
