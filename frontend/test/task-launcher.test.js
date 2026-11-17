import test from "node:test";
import assert from "node:assert/strict";
import { buildTaskGroups, prepareTaskRun } from "../src/task-launcher.js";

const mkac = [
  "Meiko Automation gồm những phòng ban nào?",
  "Quy định làm thêm giờ ở MKAC như thế nào?",
  "Các sản phẩm chính của MKAC là gì?",
];

test("maps MKAC VI and JA labels to the original questions", () => {
  const vi = buildTaskGroups("mkac", mkac, "vi");
  const ja = buildTaskGroups("mkac", mkac, "ja");
  assert.deepEqual(vi.primary.map((item) => item.label), ["Phòng ban", "Làm thêm giờ", "Sản phẩm"]);
  assert.deepEqual(ja.primary.map((item) => item.label), ["部署", "残業規定", "主な製品"]);
  assert.deepEqual(vi.primary.map((item) => item.question), mkac);
});

test("does not infer labels for unknown dynamic suggestions", () => {
  const groups = buildTaskGroups("wms", [{ question: "Current item inventory detail with an unrecognized qualifier" }]);
  assert.equal(groups.primary.length, 0);
  assert.equal(groups.other[0].label, "Current item inventory detail with an unrecognized qualifier");
  assert.equal(groups.other[0].mapped, false);
});

test("keeps explicit server labels and quick-answer IDs intact", () => {
  const entry = { id: "wms-snapshot-count", label: "Tổng số mã", question: "Có bao nhiêu mã vật tư?", execution: "prepared" };
  const groups = buildTaskGroups("wms", [entry], "vi");
  assert.equal(groups.primary[0].label, entry.label);
  assert.equal(groups.primary[0].id, entry.id);
  assert.equal(groups.primary[0].question, entry.question);
  assert.equal(groups.primary[0].execution, entry.execution);
});

test("deduplicates identical questions without relying on array position", () => {
  const groups = buildTaskGroups("mes", [
    "Mã Lot nào có số lượng lỗi nhiều nhất?",
    "Mã Lot nào có số lượng lỗi nhiều nhất?",
    "Trong Lot lỗi nhiều nhất, loại lỗi nào phổ biến nhất?",
  ], "vi");
  assert.equal(groups.primary.length, 2);
  assert.equal(groups.other.length, 0);
});

test("preserves Japanese question labels as entries when no explicit short label exists", () => {
  const question = "設備コードで検査結果を検索する";
  const groups = buildTaskGroups("qa", [{ question, question_ja: question, label: question, live: true }], "ja");
  assert.equal(groups.primary.length, 0);
  assert.equal(groups.other[0].question, question);
  assert.equal(groups.other[0].requestQuestion, question);
});

test("keeps WMS prepared answer tied to its canonical question and ID", () => {
  const task = buildTaskGroups("wms", [{
    id: "wms-snapshot-explainer", execution: "server_prepared",
    question: "Dữ liệu tồn kho WMS có phải thời gian thực không?",
    question_ja: "WMS在庫データはリアルタイムですか？",
  }], "ja").primary[0];
  assert.equal(task.question, "Dữ liệu tồn kho WMS có phải thời gian thực không?");
  assert.equal(task.sourceQuestion, task.question);
  assert.equal(prepareTaskRun(task, task.question).question, "Dữ liệu tồn kho WMS có phải thời gian thực không?");
  assert.equal(prepareTaskRun(task, task.question).usePreparedAnswer, true);
  assert.equal(prepareTaskRun(task, task.question).quickAnswerId, "wms-snapshot-explainer");
  assert.equal(prepareTaskRun(task, "WMSの在庫数は？").usePreparedAnswer, false);
  assert.equal(prepareTaskRun(task, "WMSの在庫数は？").quickAnswerId, null);
});

test("requires a non-empty explicit question before task execution", () => {
  assert.equal(prepareTaskRun({ question: "question" }, "  ").valid, false);
  assert.equal(prepareTaskRun({ question: "question" }, "changed").valid, true);
  assert.equal(prepareTaskRun({ question: "question" }, "multi\nline").valid, false);
});

test("maps WMS and CCTVAI short labels only from stable registry IDs", () => {
  const wms = buildTaskGroups("wms", [{ id: "wms-item-count", question: "WMS item count?" }], "vi");
  const cctvai = buildTaskGroups("cctvai", [{ id: "cctvai-severity", question: "CCTV event severity?" }], "ja");
  assert.equal(wms.primary[0].label, "Số mã vật tư");
  assert.equal(cctvai.primary[0].label, "7日間の違反");
  assert.equal(wms.primary[0].question, "WMS item count?");
});
