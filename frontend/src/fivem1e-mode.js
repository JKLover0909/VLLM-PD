import { streamQuery } from "./query-stream.js";

export function buildFivem1ePayload({ sessionId, question, employeeId }) {
  return {
    session_id: sessionId,
    question: String(question || "").trim(),
    employee_id: employeeId,
  };
}

export function initialFivem1eState() {
  return {
    answer: "",
    meta: "",
    pills: [],
    error: null,
    done: false,
  };
}

export function reduceFivem1eEvent(state, event) {
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
        pills: Array.isArray(result.pills) ? result.pills : [],
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

export async function streamFivem1eQuery(payload, onEvent, signal, fetchImpl = fetch) {
  return streamQuery(payload, onEvent, signal, fetchImpl, "/fivem1e/query/stream");
}
