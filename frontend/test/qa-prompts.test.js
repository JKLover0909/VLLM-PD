import assert from "node:assert/strict";
import test from "node:test";

import {
  getQaMessageSuggestions,
  getQaStarterPrompts,
  getQaSuggestions,
  QA_STARTER_PROMPTS,
  QA_SUGGESTION_POOL,
} from "../src/qa-prompts.js";

// Mã máy kiểu B302-01 hoặc mã phiếu kiểu MMSHB B0121-01 / C0101_01.
const CONCRETE_CODE = /\b(?:[A-Z]\d{3}-\d{2}|MMSHB [A-Z]\d{4}-\d{2}|[A-Z]\d{4}_\d{2})\b/;

test("QA prompts name a concrete machine or checksheet code, not placeholders", () => {
  assert.equal(QA_STARTER_PROMPTS.length, 3);
  for (const { question } of QA_SUGGESTION_POOL) {
    assert.equal(/[[\]]/.test(question), false, question);
    assert.match(question, CONCRETE_CODE);
  }
  assert.ok(QA_STARTER_PROMPTS.some((question) => question.includes("MMSHB B0121-01")));
});

test("QA suggestions are live questions without canned answers", () => {
  for (const item of QA_SUGGESTION_POOL) {
    assert.equal(item.live, true);
    assert.equal("answer" in item, false);
    assert.equal("id" in item, false);
  }
  assert.equal(new Set(QA_SUGGESTION_POOL.map((item) => item.question)).size, QA_SUGGESTION_POOL.length);
});

test("QA suggestions are stable per message and skip the question just asked", () => {
  const asked = QA_SUGGESTION_POOL[0].question;
  const first = getQaSuggestions("msg-1", `  ${asked.toUpperCase()} `);
  assert.equal(first.length, 3);
  assert.equal(first.some((item) => item.question === asked), false);
  assert.deepEqual(getQaSuggestions("msg-1", asked), first);
});

test("Japanese QA starter prompts have Japanese labels and Vietnamese query text", () => {
  const prompts = getQaStarterPrompts("ja");
  assert.equal(prompts.length, 3);
  for (const item of prompts) {
    assert.match(item.label, /[぀-ヿ]/);
    assert.ok(QA_SUGGESTION_POOL.some((prompt) => prompt.question === item.question));
  }
  assert.deepEqual(getQaStarterPrompts("vi").map((item) => item.label), QA_STARTER_PROMPTS);
});

test("Japanese QA follow-up labels localize but keep the Vietnamese query text", () => {
  const suggestions = getQaSuggestions("msg-ja", "", 20, "ja");
  for (const item of suggestions) {
    assert.match(item.label, /[぀-ヿ]/);
    assert.ok(QA_SUGGESTION_POOL.some((prompt) => prompt.question === item.question));
  }
});

test("only completed QA answers get follow-up suggestions", () => {
  const answer = { id: "a1", role: "assistant", content: "OK", qaMetadata: { provider: "qms" } };
  assert.equal(getQaMessageSuggestions(answer).length, 3);
  assert.equal(getQaMessageSuggestions(answer, "", "ja").length, 3);
  assert.ok(getQaMessageSuggestions(answer, "", "ja").every((item) => /[぀-ヿ]/.test(item.label)));
  assert.deepEqual(getQaMessageSuggestions({ ...answer, qaMetadata: undefined }), []);
  assert.deepEqual(getQaMessageSuggestions({ ...answer, stopped: true }), []);
  assert.deepEqual(getQaMessageSuggestions({ ...answer, role: "user" }), []);
});

test("a clarification request gets no topic-switching suggestions", () => {
  const answer = {
    id: "a2", role: "assistant", content: "Bạn muốn xem máy nào?",
    qaMetadata: { provider: "qms", clarification: true },
  };
  assert.deepEqual(getQaMessageSuggestions(answer), []);
});
