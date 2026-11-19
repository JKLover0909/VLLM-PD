import { streamQuery } from "./query-stream.js";

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
    error: null,
    done: false,
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
    case "done": {
      const result = event.result || {};
      return {
        ...state,
        answer: state.answer || result.answer || "",
        pending: Array.isArray(result.pending) ? result.pending : [],
        views: Array.isArray(result.views) ? result.views : [],
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

export async function streamMsmQuery(payload, onEvent, signal, fetchImpl = fetch) {
  return streamQuery(payload, onEvent, signal, fetchImpl, "/msm/query/stream");
}
