import { shouldApplyStreamEvent, streamQuery } from "./query-stream.js";

export function buildFivem1ePayload({ sessionId, question, employeeId }) {
  const payload = { session_id: sessionId, question: String(question || "").trim() };
  if (employeeId) payload.employee_id = employeeId;
  return payload;
}

export function resolveFivem1ePresence(basePresence, { streaming = false, error = false } = {}) { if (error) return { state: "error", detail: "" }; if (streaming) return { state: "working", detail: "" }; return basePresence || { state: "ready", detail: "" }; }

export function canSubmitFivem1eQuestion({ question, employeeId }) {
  return Boolean(String(question || "").trim() && employeeId);
}

export { shouldApplyStreamEvent as shouldApplyFivem1eEvent };

export async function stopFivem1eQuery({ sessionId, employeeId }, fetchImpl = fetch) {
  if (!sessionId || !employeeId) return false;
  try {
    const response = await fetchImpl("/fivem1e/query/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, employee_id: employeeId }),
    });
    return Boolean(response?.ok);
  } catch {
    return false;
  }
}

export function initialFivem1eState() {
  return { answer: "", meta: "", pills: [], error: null, done: false, stopped: false };
}

export function reduceFivem1eEvent(state, event) {
  switch (event?.type) {
    case "token": return { ...state, answer: state.answer + (event.content || "") };
    case "meta": return { ...state, meta: event.label || "" };
    case "stopped": return { ...state, done: true, stopped: true };
    case "done": {
      const result = event.result || {};
      return { ...state, answer: state.answer || result.answer || "", pills: Array.isArray(result.pills) ? result.pills : [], done: true };
    }
    case "error": return { ...state, error: { code: event.code || "", message: event.message || "" }, done: true };
    default: return state;
  }
}

export async function streamFivem1eQuery(payload, onEvent, signal, fetchImpl = fetch) {
  return streamQuery(payload, onEvent, signal, fetchImpl, "/fivem1e/query/stream", { requireTerminal: true, incompleteCode: "FIVEM1E_INCOMPLETE_STREAM" });
}
