export const QA_MAX_CODE_POINTS = 2000;

export function countCodePoints(value) {
  return Array.from(String(value || "")).length;
}

export function isValidQaQuestion(value) {
  const length = countCodePoints(value.trim());
  return length > 0 && length <= QA_MAX_CODE_POINTS;
}

// Follow-up context for QA lives on the server, keyed by session and
// employee. The browser never sends its chat history; it only asks for a new
// thread when the QA conversation it shows is empty (new chat, clear, reload).
export function buildQaPayload({ sessionId, question, employeeId, language, resetContext = false }) {
  return {
    session_id: sessionId,
    question: question.trim(),
    stream: true,
    model: "auto",
    mode: "qa",
    ui_language: language,
    employee_id: employeeId,
    quick_answer_id: null,
    conversation_context: [],
    research_topic: null,
    research_scope: null,
    qa_reset_context: Boolean(resetContext),
  };
}

// Show what the agent actually answered when it used the earlier turns to
// rewrite a follow-up; hide it when it only echoes the user's wording.
export function qaEffectiveQuestion(metadata, question) {
  const effective = String(metadata?.effective_question || "").trim();
  if (!effective || metadata?.context_applied !== true) return "";
  const normalize = (value) => String(value || "").trim().replace(/\s+/g, " ").toLocaleLowerCase();
  return normalize(effective) === normalize(question) ? "" : effective;
}
