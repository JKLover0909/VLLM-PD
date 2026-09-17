export function getSuggestions(text, currentMode, msgId, config) {
  if (!config || !config[currentMode] || config[currentMode].length === 0) return [];

  // Gợi ý câu hỏi đề xuất luôn hiển thị dưới câu trả lời của trợ lý
  // (bất kể câu trả lời ngắn hay dài đã chia đề mục) để người dùng có thể tiếp tục tra cứu tiện lợi.
  const hash = (msgId || "")
    .split("")
    .reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const pool = [...config[currentMode]];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = (hash + i) % (i + 1);
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, config.max || 3);
}
