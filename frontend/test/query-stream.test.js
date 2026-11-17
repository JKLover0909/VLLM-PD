import assert from "node:assert/strict";
import test from "node:test";

import { streamQuery } from "../src/query-stream.js";

function responseFromChunks(chunks, { status = 200, body } = {}) {
  const encoder = new TextEncoder();
  const encoded = chunks.map((chunk) =>
    chunk instanceof Uint8Array ? chunk : encoder.encode(chunk),
  );
  let cancelled = false;
  let released = false;
  const responseBody = {
    getReader() {
      let index = 0;
      return {
        async read() {
          if (index >= encoded.length) return { done: true, value: undefined };
          return { done: false, value: encoded[index++] };
        },
        async cancel() { cancelled = true; },
        releaseLock() { released = true; },
      };
    },
  };
  return {
    response: {
      ok: status >= 200 && status < 300,
      status,
      body: responseBody,
      async json() { return body; },
    },
    state: { get cancelled() { return cancelled; }, get released() { return released; } },
  };
}

const event = (value) => `data: ${JSON.stringify(value)}\n\n`;

test("parses split SSE frames, UTF-8, and replaces content exactly", async () => {
  const encoder = new TextEncoder();
  const full = encoder.encode(
    event({ type: "replace", content: "品質😀の回答" }) + event({ type: "done" }),
  );
  const split = full.findIndex((byte, index) =>
    byte === 0xf0 && full[index + 1] === 0x9f,
  ) + 2;
  const { response } = responseFromChunks([full.slice(0, split), full.slice(split)]);
  const events = [];
  await streamQuery({ mode: "qa" }, (item) => events.push(item), undefined, async () => response);
  assert.deepEqual(events, [
    { type: "replace", content: "品質😀の回答" },
    { type: "done" },
  ]);
});

test("rejects QA streams that end without done or error", async () => {
  const { response, state } = responseFromChunks([event({ type: "replace", content: "answer" })]);
  await assert.rejects(
    streamQuery({ mode: "qa" }, () => {}, undefined, async () => response),
    /QA_INCOMPLETE_STREAM/,
  );
  assert.equal(state.released, true);
});

test("propagates terminal SSE errors and releases the reader", async () => {
  const { response, state } = responseFromChunks([event({ type: "error", code: "QA_TIMEOUT", message: "Timed out" })]);
  await assert.rejects(
    streamQuery({ mode: "qa" }, () => {}, undefined, async () => response),
    (error) => error.code === "QA_TIMEOUT" && error.message === "Timed out",
  );
  assert.equal(state.released, true);
});

test("cancels and releases the reader when the request is aborted", async () => {
  let cancelled = false;
  let released = false;
  const controller = new AbortController();
  const response = {
    ok: true,
    body: {
      getReader: () => ({
        read: () => new Promise((resolve, reject) => {
          controller.signal.addEventListener(
            "abort",
            () => reject(new DOMException("Aborted", "AbortError")),
            { once: true },
          );
        }),
        cancel: async () => { cancelled = true; },
        releaseLock: () => { released = true; },
      }),
    },
  };
  const pending = streamQuery({ mode: "qa" }, () => {}, controller.signal, async () => response);
  await new Promise((resolve) => setImmediate(resolve));
  controller.abort();
  await assert.rejects(pending, (error) => error.name === "AbortError");
  assert.equal(cancelled, true);
  assert.equal(released, true);
});

test("uses legacy EOF behavior for non-QA modes", async () => {
  const { response } = responseFromChunks([event({ type: "replace", content: "answer" })]);
  const events = [];
  await streamQuery({ mode: "mes" }, (item) => events.push(item), undefined, async () => response);
  assert.equal(events.length, 1);
});
