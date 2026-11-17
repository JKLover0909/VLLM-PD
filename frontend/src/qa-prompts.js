import { getSuggestions } from "./suggestions.js";

// Mã máy / mã phiếu từ bộ eval và test của Agent QMS 5M1E.
// Nhãn Nhật chỉ để hiển thị; question luôn là câu Việt gửi live tới QA v1.
export const QA_SUGGESTION_POOL = Object.freeze([
  {
    question: "Phiếu MMSHB B0121-01 phiên bản 2 có bao nhiêu hạng mục kiểm tra?",
    question_ja: "チェックシート MMSHB B0121-01 のバージョン2には、検査項目がいくつありますか？",
  },
  {
    question: "Máy I120-01 trong tháng 9/2026 có bao nhiêu lần đo nhiệt độ, trong đó bao nhiêu lần NG?",
    question_ja: "設備 I120-01 の2026年9月の温度測定は何回で、そのうちNGは何回ですか？",
  },
  {
    question: "Lần đo áp lực gần nhất của máy B302-01 là khi nào?",
    question_ja: "設備 B302-01 で最後に圧力を測定したのはいつですか？",
  },
  {
    question: "Phiếu C0101_01 đang áp dụng phiên bản nào?",
    question_ja: "チェックシート C0101_01 の現在のバージョンは何ですか？",
  },
  {
    question: "Máy B302-01 có bao nhiêu bản ghi kiểm tra trong 30 ngày gần nhất?",
    question_ja: "設備 B302-01 の直近30日間の検査記録は何件ありますか？",
  },
  {
    question: "Trong 90 ngày gần nhất, máy B302-01 có lần đo áp lực nào bị NG không?",
    question_ja: "直近90日間に、設備 B302-01 の圧力測定でNGはありましたか？",
  },
  {
    question: "Máy I102-01 có bao nhiêu lần đo nhiệt độ bị NG trong 90 ngày gần nhất?",
    question_ja: "設備 I102-01 の直近90日間の温度測定で、NGは何回ありましたか？",
  },
  {
    question: "Tỷ lệ NG của máy I120-01 là bao nhiêu phần trăm?",
    question_ja: "設備 I120-01 のNG率は何パーセントですか？",
  },
  {
    question: "Máy F102-01 trong tháng 10/2026 có bao nhiêu lần đo trọng lượng và có lần nào NG không?",
    question_ja: "設備 F102-01 の2026年10月の重量測定は何回で、NGはありましたか？",
  },
  {
    question: "Máy F102-01 có bao nhiêu tên hạng mục kiểm tra khác nhau?",
    question_ja: "設備 F102-01 の検査項目名は何種類ありますか？",
  },
].map((item) => Object.freeze({ ...item, live: true })));

export const QA_STARTER_PROMPTS = Object.freeze(
  QA_SUGGESTION_POOL.slice(0, 3).map((item) => item.question),
);

function localizePrompt(item, language) {
  return { ...item, label: language === "ja" ? item.question_ja : item.question };
}

export function getQaStarterPrompts(language = "vi") {
  return QA_SUGGESTION_POOL.slice(0, 3).map((item) => localizePrompt(item, language));
}

function normalizeQuestion(value) {
  return String(value || "").trim().replace(/\s+/g, " ").toLowerCase();
}

/** Follow-up QA prompts for one answer, excluding the question just asked. */
export function getQaSuggestions(messageId, askedQuestion = "", max = 3, language = "vi") {
  const asked = normalizeQuestion(askedQuestion);
  const pool = QA_SUGGESTION_POOL.filter((item) =>
    normalizeQuestion(item.question) !== asked && normalizeQuestion(item.question_ja) !== asked,
  );
  return getSuggestions("", "qa", messageId, { qa: pool, max })
    .map((item) => localizePrompt(item, language));
}

/** Only completed QA answers (they carry qa_metadata) get follow-up prompts. */
export function getQaMessageSuggestions(message, askedQuestion = "", language = "vi") {
  if (message?.role !== "assistant" || !message.qaMetadata || message.stopped) return [];
  // The agent asked for missing details: let the user answer, not switch topic.
  if (message.qaMetadata.clarification === true) return [];
  return getQaSuggestions(message.id, askedQuestion, 3, language);
}
