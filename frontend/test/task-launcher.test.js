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

test("keeps WMS prepared answer tied to its localized canonical question and ID", () => {
  const entry = {
    id: "wms-snapshot-explainer", execution: "server_prepared",
    question: "Dữ liệu tồn kho WMS có phải thời gian thực không?",
    question_ja: "WMS在庫データはリアルタイムですか？",
  };
  const task = buildTaskGroups("wms", [entry], "ja").primary[0];
  assert.equal(task.question, "WMS在庫データはリアルタイムですか？");
  assert.equal(task.canonicalQuestion, "WMS在庫データはリアルタイムですか？");
  assert.equal(prepareTaskRun(task, task.question).question, "WMS在庫データはリアルタイムですか？");
  assert.equal(prepareTaskRun(task, task.question).usePreparedAnswer, true);
  assert.equal(prepareTaskRun(task, task.question).quickAnswerId, "wms-snapshot-explainer");
  const edited = prepareTaskRun(task, "WMSの在庫数は？");
  assert.equal(edited.question, "WMSの在庫数は？");
  assert.equal(edited.usePreparedAnswer, false);
  assert.equal(edited.quickAnswerId, null);
});

test("QA Japanese display keeps its Vietnamese request question", () => {
  const task = buildTaskGroups("qa", [{
    question: "Tra cứu phiếu kiểm tra máy TEST-A",
    question_ja: "設備TEST-Aの検査票を調べる",
  }], "ja").other[0];
  assert.equal(task.question, "設備TEST-Aの検査票を調べる");
  assert.equal(prepareTaskRun(task, task.question).question, "Tra cứu phiếu kiểm tra máy TEST-A");
});

test("Vietnamese WMS prepared tasks retain their canonical question", () => {
  const entry = {
    id: "wms-snapshot-explainer", execution: "server_prepared",
    question: "Dữ liệu tồn kho WMS có phải thời gian thực không?",
    question_ja: "WMS在庫データはリアルタイムですか？",
  };
  const task = buildTaskGroups("wms", [entry], "vi").primary[0];
  const run = prepareTaskRun(task, task.question);
  assert.equal(run.question, entry.question);
  assert.equal(run.quickAnswerId, entry.id);
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

test("task icons follow the task meaning in both languages", () => {
  const vi = buildTaskGroups("mkac", [{ question: "Quy định làm thêm giờ ở MKAC như thế nào?" }], "vi").primary[0];
  const ja = buildTaskGroups("mkac", [{ question: "MKACの残業規定はどうなっていますか？" }], "ja").primary[0];
  const cctv = buildTaskGroups("cctvai", [{ id: "cctvai-cameras", question: "Camera?" }], "vi").primary[0];
  assert.equal(vi.iconKey, "clock");
  assert.equal(ja.iconKey, "clock");
  assert.equal(cctv.iconKey, "camera");
});

test("QA icons come from the measured quantity, unknown tasks fall back", () => {
  const qa = buildTaskGroups("qa", [
    { question: "Máy F102-01 trong tháng 10/2026 có bao nhiêu lần đo trọng lượng và có lần nào NG không?" },
    { question: "Câu hỏi không thuộc nhóm nào" },
  ], "ja");
  assert.equal(qa.primary[0].iconKey, "scale");
  assert.equal(qa.other[0].iconKey, "");
});

test("maps MSM executive report prompt to label and report icon in VI and JA", () => {
  const vi = buildTaskGroups("msm", ["Báo cáo điều hành hiệu chuẩn thiết bị"], "vi").primary[0];
  const ja = buildTaskGroups("msm", ["機器校正エグゼクティブレポート"], "ja").primary[0];
  assert.equal(vi.label, "Báo cáo điều hành");
  assert.equal(vi.iconKey, "report");
  assert.equal(ja.label, "エグゼクティブ");
  assert.equal(ja.iconKey, "report");
});

test("maps all 4 MSM starter prompts to labels and icons in VI and JA", () => {
  const viPrompts = [
    "Báo cáo điều hành hiệu chuẩn thiết bị",
    "Danh mục việc khẩn cấp cần xử lý ngay",
    "Dự báo tải hiệu chuẩn 12 tháng tới",
    "Chuẩn bị form thêm thiết bị mới",
  ];
  const vi = buildTaskGroups("msm", viPrompts, "vi");
  assert.deepEqual(vi.primary.map((item) => item.label), [
    "Báo cáo điều hành",
    "Việc khẩn cấp",
    "Dự báo 12 tháng",
    "Thêm thiết bị",
  ]);
  assert.deepEqual(vi.primary.map((item) => item.iconKey), [
    "report",
    "alert",
    "clock",
    "checklist",
  ]);

  const jaPrompts = [
    "機器校正エグゼクティブレポート",
    "至急対応が必要な校正タスク一覧",
    "今後12ヶ月の校正負荷予測",
    "新規設備登録フォームを開いてください",
  ];
  const ja = buildTaskGroups("msm", jaPrompts, "ja");
  assert.deepEqual(ja.primary.map((item) => item.label), [
    "エグゼクティブ",
    "緊急タスク",
    "12ヶ月予測",
    "新規登録",
  ]);
  assert.deepEqual(ja.primary.map((item) => item.iconKey), [
    "report",
    "alert",
    "clock",
    "checklist",
  ]);
});

test("maps all 4 5M1E starter prompts to labels and icons in VI and JA", () => {
  const viPrompts = [
    "Tìm báo cáo có mã quản lý HB-C04-2512-01",
    "Báo cáo PCB_5M1E_20251216_0119 đang ở bước nào và tiến độ ra sao?",
    "Có những báo cáo nào đang dừng ở bước 16?",
    "Có bao nhiêu báo cáo theo từng hạng?",
  ];
  const vi = buildTaskGroups("fivem1e", viPrompts, "vi");
  assert.deepEqual(vi.primary.map((item) => item.label), [
    "Tra mã quản lý",
    "Tiến độ 30 bước",
    "Báo cáo bước 16",
    "Thống kê theo hạng",
  ]);
  assert.deepEqual(vi.primary.map((item) => item.iconKey), [
    "search",
    "checklist",
    "clock",
    "ranking",
  ]);

  const jaPrompts = [
    "管理コードHB-C04-2512-01のレポートを検索",
    "レポートPCB_5M1E_20251216_0119の30ステップ進捗",
    "ステップ16で待機中のレポート一覧",
    "ランク別のレポート件数一覧",
  ];
  const ja = buildTaskGroups("fivem1e", jaPrompts, "ja");
  assert.deepEqual(ja.primary.map((item) => item.label), [
    "管理コード検索",
    "30ステップ進捗",
    "ステップ16待機",
    "ランク別件数",
  ]);
  assert.deepEqual(ja.primary.map((item) => item.iconKey), [
    "search",
    "checklist",
    "clock",
    "ranking",
  ]);
});
