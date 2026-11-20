const LABELS_BY_ID = {
  vi: {
    "wms-executive-overview": "Báo cáo tồn kho",
    "wms-item-count": "Số mã vật tư",
    "wms-snapshot-explainer": "Dữ liệu có realtime?",
    "cctvai-hardware-status": "Tình trạng server",
    "cctvai-overview": "Tổng quan sự kiện",
    "cctvai-severity": "Vi phạm 7 ngày",
    "cctvai-cameras": "Camera hoạt động",
    "cctvai-report": "Báo cáo tổng quan",
    "cctvai-avg-confidence": "Độ tin cậy",
  },
  ja: {
    "wms-executive-overview": "在庫レポート",
    "wms-item-count": "資材コード数",
    "wms-snapshot-explainer": "リアルタイムですか？",
    "cctvai-hardware-status": "サーバー状態",
    "cctvai-overview": "イベント概要",
    "cctvai-severity": "7日間の違反",
    "cctvai-cameras": "稼働中カメラ",
    "cctvai-report": "概要レポート",
    "cctvai-avg-confidence": "平均信頼度",
  },
};

const LABELS = {
  vi: {
    mkac: {
      "Meiko Automation gồm những phòng ban nào?": "Phòng ban",
      "Meiko Automationにはどの部署がありますか？": "Phòng ban",
      "Quy định làm thêm giờ ở MKAC như thế nào?": "Làm thêm giờ",
      "MKACの残業規定はどうなっていますか？": "Làm thêm giờ",
      "Các sản phẩm chính của MKAC là gì?": "Sản phẩm",
      "MKACの主な製品は何ですか？": "Sản phẩm",
    },
    mes: {
      "Mã Lot nào có số lượng lỗi nhiều nhất?": "Lot nhiều lỗi",
      "エラー数が最も多いLotはどれですか？": "Lot nhiều lỗi",
      "Trong Lot lỗi nhiều nhất, loại lỗi nào phổ biến nhất?": "Loại lỗi",
      "エラー数が最も多いLotで、最も多いエラー種類は何ですか？": "Loại lỗi",
      "Hãy liệt kê 10 mã hàng có số lượng lỗi nhiều nhất": "Top 10 mã hàng lỗi",
      "総エラー数が多い製品を上位10件挙げてください": "Top 10 mã hàng lỗi",
    },
    wms: {},
    cctvai: {},
    msm: {
      "Báo cáo điều hành hiệu chuẩn thiết bị": "Báo cáo điều hành",
      "機器校正エグゼクティブレポート": "Báo cáo điều hành",
      "Danh mục việc khẩn cấp cần xử lý ngay": "Việc khẩn cấp",
      "至急対応が必要な校正タスク一覧": "Việc khẩn cấp",
      "Dự báo tải hiệu chuẩn 12 tháng tới": "Dự báo 12 tháng",
      "今後12ヶ月の校正負荷予測": "Dự báo 12 tháng",
      "Chuẩn bị form thêm thiết bị mới": "Thêm thiết bị",
      "新規設備登録フォームを開いてください": "Thêm thiết bị",
    },
    fivem1e: {
      "Tìm báo cáo có mã quản lý HB-C04-2512-01": "Tra mã quản lý",
      "管理コードHB-C04-2512-01のレポートを検索": "Tra mã quản lý",
      "Báo cáo PCB_5M1E_20251216_0119 đang ở bước nào và tiến độ ra sao?": "Tiến độ 30 bước",
      "レポートPCB_5M1E_20251216_0119の30ステップ進捗": "Tiến độ 30 bước",
      "Có những báo cáo nào đang dừng ở bước 16?": "Báo cáo bước 16",
      "ステップ16で待機中のレポート一覧": "Báo cáo bước 16",
      "Có bao nhiêu báo cáo theo từng hạng?": "Thống kê theo hạng",
      "ランク別のレポート件数一覧": "Thống kê theo hạng",
    },
    research: {
      "Lập báo cáo nghiên cứu tổng hợp từ các tài liệu": "Báo cáo tổng hợp",
      "So sánh các quan điểm và chỉ ra điểm mâu thuẫn": "So sánh tài liệu",
      "Đề xuất các câu hỏi nghiên cứu tiếp theo": "Câu hỏi tiếp theo",
    },
    qa: {
      "Phiếu MMSHB B0121-01 phiên bản 2 có bao nhiêu hạng mục kiểm tra?": "Hạng mục phiếu",
      "Máy I120-01 trong tháng 9/2026 có bao nhiêu lần đo nhiệt độ, trong đó bao nhiêu lần NG?": "Nhiệt độ I120-01",
      "Lần đo áp lực gần nhất của máy B302-01 là khi nào?": "Áp lực B302-01",
    },
  },
  ja: {
    qa: {
      "Phiếu MMSHB B0121-01 phiên bản 2 có bao nhiêu hạng mục kiểm tra?": "検査項目",
      "Máy I120-01 trong tháng 9/2026 có bao nhiêu lần đo nhiệt độ, trong đó bao nhiêu lần NG?": "I120-01 温度",
      "Lần đo áp lực gần nhất của máy B302-01 là khi nào?": "B302-01 圧力",
      "Phiếu C0101_01 đang áp dụng phiên bản nào?": "C0101_01 版数",
      "Máy B302-01 có bao nhiêu bản ghi kiểm tra trong 30 ngày gần nhất?": "B302-01 30日間",
      "Trong 90 ngày gần nhất, máy B302-01 có lần đo áp lực nào bị NG không?": "B302-01 圧力NG",
      "Máy I102-01 có bao nhiêu lần đo nhiệt độ bị NG trong 90 ngày gần nhất?": "I102-01 温度NG",
      "Tỷ lệ NG của máy I120-01 là bao nhiêu phần trăm?": "I120-01 NG率",
      "Máy F102-01 trong tháng 10/2026 có bao nhiêu lần đo trọng lượng và có lần nào NG không?": "F102-01 重量",
      "Máy F102-01 có bao nhiêu tên hạng mục kiểm tra khác nhau?": "F102-01 検査項目",
    },
    mkac: {
      "Meiko Automation gồm những phòng ban nào?": "部署",
      "Meiko Automationにはどの部署がありますか？": "部署",
      "Quy định làm thêm giờ ở MKAC như thế nào?": "残業規定",
      "MKACの残業規定はどうなっていますか？": "残業規定",
      "Các sản phẩm chính của MKAC là gì?": "主な製品",
      "MKACの主な製品は何ですか？": "主な製品",
    },
    mes: {
      "Mã Lot nào có số lượng lỗi nhiều nhất?": "エラー最多Lot",
      "エラー数が最も多いLotはどれですか？": "エラー最多Lot",
      "Trong Lot lỗi nhiều nhất, loại lỗi nào phổ biến nhất?": "最多エラー種別",
      "エラー数が最も多いLotで、最も多いエラー種類は何ですか？": "最多エラー種別",
      "Hãy liệt kê 10 mã hàng có số lượng lỗi nhiều nhất": "エラー上位10品目",
      "総エラー数が多い製品を上位10件挙げてください": "エラー上位10品目",
    },
    wms: {},
    cctvai: {},
    msm: {
      "Báo cáo điều hành hiệu chuẩn thiết bị": "エグゼクティブ",
      "機器校正エグゼクティブレポート": "エグゼクティブ",
      "Danh mục việc khẩn cấp cần xử lý ngay": "緊急タスク",
      "至急対応が必要な校正タスク一覧": "緊急タスク",
      "Dự báo tải hiệu chuẩn 12 tháng tới": "12ヶ月予測",
      "今後12ヶ月の校正負荷予測": "12ヶ月予測",
      "Chuẩn bị form thêm thiết bị mới": "新規登録",
      "新規設備登録フォームを開いてください": "新規登録",
    },
    fivem1e: {
      "Tìm báo cáo có mã quản lý HB-C04-2512-01": "管理コード検索",
      "管理コードHB-C04-2512-01のレポートを検索": "管理コード検索",
      "Báo cáo PCB_5M1E_20251216_0119 đang ở bước nào và tiến độ ra sao?": "30ステップ進捗",
      "レポートPCB_5M1E_20251216_0119の30ステップ進捗": "30ステップ進捗",
      "Có những báo cáo nào đang dừng ở bước 16?": "ステップ16待機",
      "ステップ16で待機中のレポート一覧": "ステップ16待機",
      "Có bao nhiêu báo cáo theo từng hạng?": "ランク別件数",
      "ランク別のレポート件数一覧": "ランク別件数",
    },
    research: {
      "Lập báo cáo nghiên cứu tổng hợp từ các tài liệu": "総合レポート",
      "So sánh các quan điểm và chỉ ra điểm mâu thuẫn": "資料を比較",
      "Đề xuất các câu hỏi nghiên cứu tiếp theo": "次の調査質問",
    },
  },
};

// Icon keys are resolved from a stable registry ID or the Vietnamese label, so
// the same task keeps the same pictogram in both languages. The key names are
// mapped to real icon components in TaskLauncher.jsx.
const ICON_KEYS = {
  "wms-executive-overview": "report",
  "wms-item-count": "count",
  "wms-snapshot-explainer": "refresh",
  "cctvai-hardware-status": "server",
  "cctvai-overview": "overview",
  "cctvai-severity": "alert",
  "cctvai-cameras": "camera",
  "cctvai-report": "report",
  "cctvai-avg-confidence": "gauge",
  "Phòng ban": "org",
  "Làm thêm giờ": "clock",
  "Sản phẩm": "product",
  "Lot nhiều lỗi": "alert",
  "Loại lỗi": "pie",
  "Top 10 mã hàng lỗi": "ranking",
  "Báo cáo tổng hợp": "report",
  "So sánh tài liệu": "compare",
  "Câu hỏi tiếp theo": "idea",
  "Hạng mục phiếu": "checklist",
  "Nhiệt độ I120-01": "temperature",
  "Áp lực B302-01": "gauge",
  "Hạn hiệu chuẩn": "clock",
  "Thiết bị ANNEALING": "overview",
  "Thêm thiết bị": "checklist",
  "新規登録": "checklist",
  "Báo cáo điều hành": "report",
  "エグゼクティブ": "report",
  "Việc khẩn cấp": "alert",
  "緊急タスク": "alert",
  "Dự báo 12 tháng": "clock",
  "12ヶ月予測": "clock",
  "Tra mã quản lý": "search",
  "管理コード検索": "search",
  "Tiến độ 30 bước": "checklist",
  "30ステップ進捗": "checklist",
  "Báo cáo bước 16": "clock",
  "ステップ16待機": "clock",
  "Thống kê theo hạng": "ranking",
  "ランク別件数": "ranking",

};

// QA has more Japanese starters than curated Vietnamese labels; pick an icon
// from the measured quantity in the (Vietnamese) request question.
const QA_ICON_RULES = [
  [/nhiệt độ/i, "temperature"],
  [/áp lực/i, "gauge"],
  [/trọng lượng/i, "scale"],
  [/tỷ lệ/i, "percent"],
  [/phiên bản/i, "version"],
  [/hạng mục/i, "checklist"],
  [/bản ghi/i, "records"],
];

function resolveIconKey(mode, entry, question) {
  const id = typeof entry === "object" && entry?.id ? String(entry.id) : "";
  if (id && ICON_KEYS[id]) return ICON_KEYS[id];
  const viLabel = LABELS.vi[mode]?.[question];
  if (viLabel && ICON_KEYS[viLabel]) return ICON_KEYS[viLabel];
  const jaLabel = LABELS.ja?.[mode]?.[question];
  if (jaLabel && ICON_KEYS[jaLabel]) return ICON_KEYS[jaLabel];
  if (mode === "qa") {
    const rule = QA_ICON_RULES.find(([pattern]) => pattern.test(question));
    if (rule) return rule[1];
  }
  return "";
}

export function prepareTaskRun(task, editedQuestion) {
  const edited = String(editedQuestion || "").trim();
  const unchanged = edited === task?.question;
  const question = unchanged
    ? String(task?.canonicalQuestion || task?.requestQuestion || task?.question || "").trim()
    : edited;
  return {
    question,
    valid: Boolean(question) && !/[\r\n]/.test(question),
    usePreparedAnswer: Boolean(unchanged && task?.execution === "server_prepared" && task?.id),
    quickAnswerId: unchanged && task?.execution === "server_prepared" ? task.id : null,
  };
}

export function buildTaskGroups(mode, entries = [], language = "vi") {
  const labels = LABELS[language]?.[mode] || LABELS.vi[mode] || {};
  const primary = [];
  const other = [];
  const seen = new Set();

  for (const entry of entries) {
    const question = typeof entry === "string" ? entry : String(entry?.question || "");
    if (!question.trim() || seen.has(question)) continue;
    seen.add(question);
    const explicitLabel = typeof entry === "object" && entry.label && entry.label !== question
      ? String(entry.label)
      : "";
    const stableIdLabel = typeof entry === "object" && entry.id
      ? LABELS_BY_ID[language]?.[entry.id]
      : "";
    const label = explicitLabel || stableIdLabel || labels[question];
    const requestQuestion = mode === "qa" && typeof entry === "object"
      ? String(entry.question || "").trim()
      : question;
    const displayQuestion = mode === "qa" && language === "ja" && typeof entry === "object"
      ? String(entry.question_ja || question).trim()
      : question;
    const safeWmsQuestion = mode === "wms" && entry?.execution === "server_prepared"
      ? String(language === "ja" ? entry.question_ja || question : question).trim()
      : question;
    const task = {
      ...((typeof entry === "object" && entry) || {}),
      question: mode === "qa" ? displayQuestion : safeWmsQuestion,
      canonicalQuestion: mode === "wms" && language === "ja" && entry?.question_ja
        ? String(entry.question_ja).trim()
        : mode === "wms" ? String(entry?.question || question).trim() : question,
      requestQuestion,
      sourceQuestion: question,
      label: label || question,
      mapped: Boolean(label),
      iconKey: resolveIconKey(mode, entry, question),
    };
    (label ? primary : other).push(task);
  }

  return { primary, other };
}
