import assert from "node:assert/strict";
import test from "node:test";

import { getSuggestions } from "../src/suggestions.js";

const mockConfig = {
  max: 3,
  cctvai: [
    { question: "Kiểm tra tình trạng phần cứng server CCTV AI" },
    { question: "Thống kê vi phạm an toàn hôm nay" },
    { question: "Top 5 camera ghi nhận nhiều vi phạm nhất" },
    { question: "Danh sách camera khu vực xưởng A" },
  ],
  mkac: [
    { question: "Meiko Automation có bao nhiêu phòng ban?" },
    { question: "Quy định làm thêm giờ ở MKAC như thế nào?" },
    { question: "Quy trình đặt xe ô tô, Grab, Taxi như thế nào?" },
  ],
};

test("returns suggestions for short answer", () => {
  const shortText = "CPU 10%, RAM 30%.";
  const results = getSuggestions(shortText, "cctvai", "msg-123", mockConfig);
  assert.equal(results.length, 3);
  assert.ok(results.every((item) => typeof item.question === "string"));
});

test("returns suggestions for long formatted answer with markdown headings and lists", () => {
  const longFormattedText = `
📊 **Tình trạng phần cứng máy chủ CCTV AI:**
- **CPU:** 11% (48 lõi logic)
- **Bộ nhớ RAM:** 32% (39.8GB / 124.8GB)
- **GPU:** NVIDIA RTX PRO 5000: 43%, VRAM 15.0GB / 47.8GB, 71°C
- **Ổ đĩa:** /: 8% (140.0GB / 1864.0GB)
- **Mạng:** enp66s0f0 (↓3.3MB/s, ↑2.6MB/s)
- **Thời gian hoạt động (Uptime):** 7 ngày 22 giờ 2 phút
- **Docker:** 12/17 container đang chạy

⚠️ Lưu ý: Danh sách tiến trình đã được cắt bớt để bảo đảm an toàn.
`.repeat(5); // ~3500 chars

  const results = getSuggestions(longFormattedText, "cctvai", "msg-456", mockConfig);
  assert.equal(results.length, 3);
  assert.ok(results.every((item) => typeof item.question === "string"));
});

test("returns empty array when mode has no suggestions or config is missing", () => {
  assert.deepEqual(getSuggestions("any text", "research", "msg-789", mockConfig), []);
  assert.deepEqual(getSuggestions("any text", "cctvai", "msg-789", null), []);
  assert.deepEqual(getSuggestions("any text", "cctvai", "msg-789", {}), []);
});

test("deterministic output for the same message ID", () => {
  const text = "Any answer content";
  const run1 = getSuggestions(text, "mkac", "msg-consistent-id", mockConfig);
  const run2 = getSuggestions(text, "mkac", "msg-consistent-id", mockConfig);
  assert.deepEqual(run1, run2);
});
