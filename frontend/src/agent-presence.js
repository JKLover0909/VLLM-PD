export function resolveAgentPresence({ busy = false, pendingId, message, question = "", error, needsSelection = false } = {}) {
  if (error) return { state: "error", detail: "" };
  if (busy) {
    const active = message?.id === pendingId ? message : null;
    const step = active?.agentTimeline?.steps?.find((item) => item.status === "running");
    return {
      state: active?.content ? "answering" : "working",
      detail: step?.title || active?.status || "",
    };
  }
  if (question.trim()) return { state: "composing", detail: "" };
  if (message?.stopped) return { state: "stopped", detail: "" };
  if (message?.agentTimeline?.steps?.some((step) => step.status === "error")) return { state: "error", detail: "" };
  if (needsSelection || message?.qaMetadata?.clarification === true) return { state: "asking", detail: "" };
  return { state: "ready", detail: "" };
}

export function resolveStreamingModePresence(basePresence, { streaming = false, answering = false, error = false } = {}) {
  if (error) return { state: "error", detail: "" };
  if (streaming && answering) return { state: "answering", detail: "" };
  if (streaming) return { state: "working", detail: "" };
  return basePresence || { state: "ready", detail: "" };
}

export function gazePosition(x, y) {
  const clamp = (value) => Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : 0;
  return { x: clamp(x), y: clamp(y) };
}
