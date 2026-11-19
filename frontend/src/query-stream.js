const TOKEN_DRIP_MIN_LENGTH = 60;
const TOKEN_DRIP_MAX_CHUNKS = 24;
const TOKEN_DRIP_DELAY_MS = 35;

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(() => {
      if (signal) signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    function onAbort() {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    }
    if (signal) signal.addEventListener("abort", onAbort, { once: true });
  });
}

function splitForDrip(text) {
  const chars = Array.from(text);
  if (chars.length <= 1) return [text];
  const chunkCount = Math.min(TOKEN_DRIP_MAX_CHUNKS, chars.length);
  const size = Math.ceil(chars.length / chunkCount);
  const parts = [];
  for (let i = 0; i < chars.length; i += size) {
    parts.push(chars.slice(i, i + size).join(""));
  }
  return parts;
}

function parseSseBlock(block) {
  const data = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("data:")) {
      data.push(line.slice(5).replace(/^ /, ""));
    }
  }
  if (!data.length) return null;
  try {
    return JSON.parse(data.join("\n"));
  } catch {
    return null;
  }
}

function terminalError(event) {
  const error = new Error(event.message || "The request failed.");
  error.code = event.code || "";
  error.retryAfter = event.retry_after;
  return error;
}

export async function streamQuery(payload, onEvent, signal, fetchImpl = fetch, endpoint = "/query/stream") {
  const response = await fetchImpl(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    let code = "";
    let retryAfter;
    try {
      const body = await response.json();
      const detail = body.detail && typeof body.detail === "object" ? body.detail : body;
      message = detail.message || detail.detail || message;
      code = detail.code || "";
      retryAfter = detail.retry_after;
    } catch {
      // Keep the safe HTTP status message when the error body is not JSON.
    }
    const error = new Error(message);
    error.status = response.status;
    error.code = code;
    error.retryAfter = retryAfter;
    throw error;
  }
  if (!response.body?.getReader) {
    if (payload.mode === "qa") throw new Error("QA_INCOMPLETE_STREAM");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminal = false;
  let reachedEof = false;

  async function dispatch(block) {
    const event = parseSseBlock(block);
    if (!event || typeof event !== "object") return;
    if (event.type === "error") {
      terminal = true;
      throw terminalError(event);
    }
    if (event.type === "done") terminal = true;
    if (event.type === "token" && event.content?.length >= TOKEN_DRIP_MIN_LENGTH) {
      for (const part of splitForDrip(event.content)) {
        await sleep(TOKEN_DRIP_DELAY_MS, signal);
        onEvent({ ...event, content: part });
      }
      return;
    }
    onEvent(event);
  }

  async function consumeBlocks(final = false) {
    const blocks = buffer.split(/\n\n/);
    buffer = blocks.pop() || "";
    for (const block of blocks) await dispatch(block);
    if (final && buffer) {
      await dispatch(buffer);
      buffer = "";
    }
  }

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        reachedEof = true;
        buffer += decoder.decode();
        await consumeBlocks(true);
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      // Normalize CRLF (and lone CR) before splitting SSE event blocks.
      buffer = buffer.replace(/\r\n?/g, "\n");
      await consumeBlocks();
    }
    if (payload.mode === "qa" && !terminal) {
      throw new Error("QA_INCOMPLETE_STREAM");
    }
  } finally {
    if (!reachedEof) {
      try {
        await reader.cancel();
      } catch {
        // Abort/transport failure already determines the request outcome.
      }
    }
    try {
      reader.releaseLock();
    } catch {
      // A reader may already have released its lock after a stream failure.
    }
  }
}
