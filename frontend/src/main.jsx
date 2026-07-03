import { StrictMode, memo, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";

// Memo hoá theo nội dung: khi stream token, chỉ message đang thay đổi mới
// parse lại markdown; các message cũ giữ nguyên nên không re-parse (tránh
// O(n) lần parse markdown cho mỗi token khi hội thoại dài).
const MessageMarkdown = memo(function MessageMarkdown({ content }) {
  return <ReactMarkdown>{content}</ReactMarkdown>;
});
import {
  Activity,
  AlertCircle,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  Copy,
  Database,
  FileText,
  FileUp,
  FlaskConical,
  Globe2,
  Layers3,
  Loader2,
  LogOut,
  Menu,
  PanelRightClose,
  PanelRightOpen,
  Paperclip,
  RefreshCcw,
  Search,
  Send,
  Server,
  ShieldCheck,
  Sparkles,
  Square,
  Sun,
  Moon,
  Monitor,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import "./styles.css";

const QUICK_PROMPTS = {
  vi: {
    mkac: [
      "Meiko Automation có bao nhiêu phòng ban, gồm các phòng ban nào?",
      "Quy định làm thêm giờ ở MKAC như thế nào?",
      "Các sản phẩm chính của MKAC là gì?",
    ],
    mes: [
      "Mã Lot nào có số lượng lỗi nhiều nhất?",
      "Trong Lot lỗi nhiều nhất, loại lỗi nào phổ biến nhất?",
      "Hãy liệt kê 10 mã hàng có số lượng lỗi nhiều nhất",
    ],
    research: [
      "Lập báo cáo nghiên cứu tổng hợp từ các tài liệu",
      "So sánh các quan điểm và chỉ ra điểm mâu thuẫn",
      "Đề xuất các câu hỏi nghiên cứu tiếp theo",
    ],
  },
  ja: {
    mkac: [
      "Meiko Automationにはいくつの部署があり、どの部署がありますか？",
      "MKACの残業規定はどうなっていますか？",
      "MKACの主な製品は何ですか？",
    ],
    mes: [
      "エラー数が最も多いLotはどれですか？",
      "エラー数が最も多いLotで、最も多いエラー種類は何ですか？",
      "総エラー数が多い製品を上位10件挙げてください",
    ],
    research: [
      "資料から総合的な調査レポートを作成してください",
      "各見解を比較し、矛盾点を示してください",
      "次に調査すべき質問を提案してください",
    ],
  },
};

const ACTIVE_MODE_KEYS = ["mkac", "mes"];

const MODE_OPTIONS = {
  mkac: {
    icon: Database,
  },
  mes: {
    icon: Activity,
  },
  research: {
    icon: FlaskConical,
  },
};
const VISIBLE_MODE_KEYS = ACTIVE_MODE_KEYS;

const MODEL_ACCENTS = {
  auto: "accent-auto",
  local: "accent-local",
  openai: "accent-openai",
  grok: "accent-grok",
};

const LEGACY_MODE_SESSION_STORAGE_KEYS = {
  mkac: "meibook-session-mkac",
  mes: "meibook-session-mes",
  research: "meibook-session-research",
};
const SESSION_STORAGE_PREFIX = "meibook-session";
const LEGACY_SESSION_STORAGE_KEY = "meibook-session";
const SESSION_TITLE_STORAGE_KEY = "meibook-session-titles";
const THEME_STORAGE_KEY = "meibook-theme";
const LANGUAGE_STORAGE_KEY = "meibook-language";
const EMPLOYEE_STORAGE_KEY = "meibook-mkac-employee";
const GUEST_EMPLOYEE_ID = "000000";
const THEME_OPTIONS = ["system", "light", "dark"];
const LANGUAGE_OPTIONS = ["vi", "ja"];
const THEME_ICONS = {
  system: Monitor,
  light: Sun,
  dark: Moon,
};

const UI_TEXT = {
  vi: {
    modes: {
      mkac: {
        label: "Hành chính nhân sự",
        shortLabel: "HCNS",
        title: "Hỏi đáp hành chính nhân sự MKAC",
        empty: "Tra cứu quy định hành chính, thông tin nội bộ và nhân sự MKAC.",
        metric: "Tài liệu MKAC",
        authHint: "Nhập mã nhân viên để truy cập chế độ hỏi đáp nội bộ MKAC.",
        inputLabel: "Câu hỏi hành chính nhân sự MKAC",
        placeholder: "Hỏi về hành chính, quy định hoặc nhân sự MKAC...",
      },
      mes: {
        label: "Quản lý MES",
        shortLabel: "MES",
        title: "Quản lý MES",
        empty: "Tra cứu Lot, mã hàng, mã lỗi và thống kê sản xuất từ MES.",
        metric: "Lot MES",
        authHint: "Nhập mã nhân viên để truy cập dữ liệu sản xuất MES.",
        inputLabel: "Câu hỏi về MES",
        placeholder: "Hỏi về Lot, mã hàng hoặc lỗi sản xuất...",
        unavailable: "MES snapshot chưa sẵn sàng",
      },
      research: {
        label: "Nghiên cứu tài liệu",
        shortLabel: "Nghiên cứu",
        title: "Nghiên cứu tài liệu",
        emptyReady: "Sẵn sàng nghiên cứu tài liệu đã index trong phiên này.",
        empty: "Tải tài liệu lên để bắt đầu nghiên cứu.",
        metric: "Tài liệu",
        inputLabel: "Câu hỏi nghiên cứu",
        placeholderReady: "Nhập chủ đề nghiên cứu...",
        placeholderEmpty: "Hãy tải tài liệu lên trước khi đặt câu hỏi...",
        lockedModel: "Chế độ nghiên cứu đang tạm ẩn",
      },
    },
    theme: {
      system: "Theo hệ thống",
      light: "Sáng",
      dark: "Tối",
    },
    waiting: [
      "Đã nhận câu hỏi, đang suy luận...",
      "Tôi hiểu rồi, bạn chờ một chút nhé...",
      "Đang đối chiếu các nguồn phù hợp...",
      "Đang tổng hợp câu trả lời...",
      "Sắp có kết quả rồi...",
    ],
    defaultTitles: {
      mkac: "Hành chính nhân sự mới",
      mes: "Phiên MES mới",
      research: "Phiên nghiên cứu mới",
      researchDemo: "Tài liệu nghiên cứu demo",
    },
    common: {
      openDocuments: "Mở tài liệu",
      closeDocuments: "Đóng tài liệu",
      assistantName: "Trợ lý hỏi đáp nội bộ",
      close: "Đóng",
      workSession: "Phiên làm việc",
      newSession: "Tạo phiên mới",
      researchDocuments: "Tài liệu nghiên cứu",
      chooseDocument: "Chọn tài liệu",
      supportedFiles: "PDF, Office, HTML, PNG/JPG",
      files: "tệp",
      removeFile: "Bỏ {name}",
      deleteFile: "Xóa {name}",
      indexDocument: "Index tài liệu",
      indexing: "Đang index {done}/{total}",
      processing: "Đang xử lý",
      indexedSummary: "Đã index {files} tệp, {chunks} đoạn",
      noDocuments: "Chưa có tài liệu",
      onlineMachine: "Máy 2 online",
      checking: "Đang kiểm tra",
      qaModes: "Chế độ hỏi đáp",
      chooseModel: "Chọn mô hình",
      modelList: "Danh sách mô hình",
      interfaceTitle: "Giao diện: {theme}. Nhấn để chuyển chế độ.",
      interfaceAria: "Giao diện hiện tại: {theme}",
      languageTitle: "Ngôn ngữ giao diện: {language}. Nhấn để chuyển.",
      employeeLogout: "Đăng xuất mã nhân viên",
      clearConversation: "Xóa hội thoại hiện tại",
      clearConfirmTitle: "Xóa toàn bộ hội thoại?",
      clearConfirmBody: "Toàn bộ tin nhắn trong hội thoại này sẽ bị xóa. Hành động này không thể hoàn tác.",
      clearConfirmAction: "Xóa hội thoại",
      cancel: "Hủy",
      hideSources: "Ẩn nguồn",
      showSources: "Hiện nguồn",
      employeeAuth: "Xác thực nhân viên MKAC",
      employeeCode: "Mã nhân viên",
      invalidEmployee: "Mã nhân viên không hợp lệ.",
      employeeRequired: "Vui lòng nhập mã nhân viên hợp lệ trước khi tiếp tục.",
      guestWelcome: "Chào mừng đến với hệ thống Meibook,",
      languageConverting: "Đang chuyển đổi ngôn ngữ...",
      continue: "Tiếp tục",
      verifying: "Đang kiểm tra",
      chooseToStart: "Chọn tài liệu để bắt đầu",
      tryDemo: "Thử nghiệm với tài liệu mẫu",
      model: "Model",
      status: "Trạng thái",
      stopped: "Đã dừng",
      noResult: "Không có kết quả",
      webSearch: "Tìm kiếm web",
      mesData: "Dữ liệu MES",
      mesSnapshot: "MES snapshot",
      research: "Nghiên cứu",
      mkacSource: "Nguồn MKAC",
      aiInfo: "Thông tin phản hồi AI",
      copied: "Đã sao chép",
      copy: "Sao chép",
      copyAnswer: "Sao chép câu trả lời",
      copyFailed: "Không thể sao chép câu trả lời.",
      suggestions: "Gợi ý thêm:",
      autoAsk: "Nhấn để tự động hỏi và trả lời",
      addResearchDocument: "Thêm tài liệu nghiên cứu",
      send: "Gửi",
      stop: "Dừng trả lời",
      sources: "Nguồn tham chiếu",
      webSource: "Nguồn web",
      page: "Trang {page}",
      similarity: "Độ tương đồng {score}%",
      clickToViewPage: "Bấm để xem trang",
      clickToViewSnippet: "Bấm để xem đoạn trích",
      noSources: "Chưa có nguồn cho lượt trả lời này",
      previewDialog: "Xem nguồn tham chiếu",
      closePreview: "Đóng preview",
      noPreviewImage: "Chưa có ảnh preview cho trang này.",
      snippet: "Đoạn trích",
      demoNotIndexed: "Tài liệu mẫu chưa được index. Vui lòng chạy script index trước.",
      stoppedResponse: "Đã dừng phản hồi.",
      requestFailed: "Không thể hoàn tất yêu cầu: {message}",
      loading: "Đang tải",
      loadingModel: "Đang tải model",
      loadingModelList: "Đang tải danh sách model",
      internalDocuments: "tài liệu nội bộ",
      sessionDocuments: "tài liệu trong phiên",
      hello: "Xin chào, {name}",
      errorRecords: "bản ghi lỗi",
      online: "Online",
      offline: "Offline",
      ready: "Sẵn sàng",
      notReady: "Chưa sẵn sàng",
      promptLabel: "Gợi ý để bắt đầu",
    },
    answerScope: {
      web: "Tổng hợp từ nguồn web",
      mes: "Dựa trên dữ liệu MES trực tiếp",
      mes_database: "Dựa trên MES snapshot cục bộ",
      research: "Dựa trên tài liệu nghiên cứu",
      mkac: "Dựa trên kho MKAC",
      fallback: "Không có nguồn đối chiếu",
    },
    models: {
      local: {
        name: "Local Model",
        description: "Chạy model nội bộ/local cho hỏi đáp dạng text.",
      },
      openai: {
        name: "Model dự phòng",
        description: "Chỉ dùng làm tuyến dự phòng kỹ thuật.",
      },
      grok: {
        name: "Model nghiên cứu",
        description: "Chỉ dùng khi bật lại chế độ nghiên cứu tài liệu.",
      },
    },
  },
  ja: {
    modes: {
      mkac: {
        label: "人事・総務",
        shortLabel: "人事",
        title: "MKAC 人事・総務 Q&A",
        empty: "MKACの規程、社内情報、人事情報を検索します。",
        metric: "MKAC資料",
        authHint: "MKAC社内Q&Aを利用するには社員番号を入力してください。",
        inputLabel: "MKAC 人事・総務に関する質問",
        placeholder: "総務、規程、人事について質問してください...",
      },
      mes: {
        label: "MES管理",
        shortLabel: "MES",
        title: "MES管理",
        empty: "Lot、品番、エラーコード、生産統計をMESから検索します。",
        metric: "MES Lot",
        authHint: "MES生産データにアクセスするには社員番号を入力してください。",
        inputLabel: "MESに関する質問",
        placeholder: "Lot、品番、生産エラーについて質問してください...",
        unavailable: "MESスナップショットはまだ利用できません",
      },
      research: {
        label: "資料調査",
        shortLabel: "調査",
        title: "資料調査",
        emptyReady: "このセッションのインデックス済み資料を調査できます。",
        empty: "調査を始めるには資料をアップロードしてください。",
        metric: "資料",
        inputLabel: "調査質問",
        placeholderReady: "調査テーマを入力してください...",
        placeholderEmpty: "質問する前に資料をアップロードしてください...",
        lockedModel: "資料調査モードは一時的に非表示です",
      },
    },
    theme: {
      system: "システム設定",
      light: "ライト",
      dark: "ダーク",
    },
    waiting: [
      "質問を受け取りました。推論しています...",
      "承知しました。少々お待ちください...",
      "関連する情報源を照合しています...",
      "回答をまとめています...",
      "もうすぐ結果が出ます...",
    ],
    defaultTitles: {
      mkac: "新しい人事・総務セッション",
      mes: "新しいMESセッション",
      research: "新しい資料調査セッション",
      researchDemo: "サンプル調査資料",
    },
    common: {
      openDocuments: "資料を開く",
      closeDocuments: "資料を閉じる",
      assistantName: "社内Q&Aアシスタント",
      close: "閉じる",
      workSession: "作業セッション",
      newSession: "新しいセッション",
      researchDocuments: "調査資料",
      chooseDocument: "資料を選択",
      supportedFiles: "PDF、Office、HTML、PNG/JPG",
      files: "件のファイル",
      removeFile: "{name}を外す",
      deleteFile: "{name}を削除",
      indexDocument: "資料をインデックス",
      indexing: "インデックス中 {done}/{total}",
      processing: "処理中",
      indexedSummary: "{files}ファイル、{chunks}チャンクをインデックスしました",
      noDocuments: "資料はまだありません",
      onlineMachine: "マシン2 オンライン",
      checking: "確認中",
      qaModes: "Q&Aモード",
      chooseModel: "モデルを選択",
      modelList: "モデル一覧",
      interfaceTitle: "表示テーマ: {theme}。クリックして切り替えます。",
      interfaceAria: "現在の表示テーマ: {theme}",
      languageTitle: "表示言語: {language}。クリックして切り替えます。",
      employeeLogout: "社員番号をログアウト",
      clearConversation: "現在の会話を削除",
      clearConfirmTitle: "会話をすべて削除しますか？",
      clearConfirmBody: "この会話のすべてのメッセージが削除されます。この操作は取り消せません。",
      clearConfirmAction: "会話を削除",
      cancel: "キャンセル",
      hideSources: "参照元を隠す",
      showSources: "参照元を表示",
      employeeAuth: "MKAC社員認証",
      employeeCode: "社員番号",
      invalidEmployee: "社員番号が正しくありません。",
      employeeRequired: "続行する前に有効な社員番号を入力してください。",
      guestWelcome: "Meibookシステムへようこそ。",
      languageConverting: "言語を変換しています...",
      continue: "続行",
      verifying: "確認中",
      chooseToStart: "資料を選択して開始",
      tryDemo: "サンプル資料で試す",
      model: "モデル",
      status: "状態",
      stopped: "停止済み",
      noResult: "結果なし",
      webSearch: "Web検索",
      mesData: "MESデータ",
      mesSnapshot: "MESスナップショット",
      research: "調査",
      mkacSource: "MKACソース",
      aiInfo: "AI応答情報",
      copied: "コピー済み",
      copy: "コピー",
      copyAnswer: "回答をコピー",
      copyFailed: "回答をコピーできませんでした。",
      suggestions: "追加候補:",
      autoAsk: "クリックすると自動で質問・回答します",
      addResearchDocument: "調査資料を追加",
      send: "送信",
      stop: "回答を停止",
      sources: "参照元",
      webSource: "Webソース",
      page: "{page}ページ",
      similarity: "類似度 {score}%",
      clickToViewPage: "クリックしてページを表示",
      clickToViewSnippet: "クリックして抜粋を表示",
      noSources: "この回答の参照元はまだありません",
      previewDialog: "参照元を表示",
      closePreview: "プレビューを閉じる",
      noPreviewImage: "このページのプレビュー画像はありません。",
      snippet: "抜粋",
      demoNotIndexed: "サンプル資料はまだインデックスされていません。先にインデックススクリプトを実行してください。",
      stoppedResponse: "応答を停止しました。",
      requestFailed: "リクエストを完了できません: {message}",
      loading: "読み込み中",
      loadingModel: "モデルを読み込み中",
      loadingModelList: "モデル一覧を読み込み中",
      internalDocuments: "社内資料",
      sessionDocuments: "セッション内の資料",
      hello: "こんにちは、{name}",
      errorRecords: "エラー記録",
      online: "オンライン",
      offline: "オフライン",
      ready: "準備完了",
      notReady: "準備中",
      promptLabel: "質問の候補",
    },
    answerScope: {
      web: "Webソースに基づく要約",
      mes: "MESリアルタイムデータに基づく",
      mes_database: "ローカルMESスナップショットに基づく",
      research: "調査資料に基づく",
      mkac: "MKACナレッジベースに基づく",
      fallback: "照合できる参照元はありません",
    },
    models: {
      local: {
        name: "ローカルモデル",
        description: "テキストQ&A向けに社内/ローカルモデルを使用します。",
      },
      openai: {
        name: "予備モデル",
        description: "技術的な予備ルートとしてのみ使用します。",
      },
      grok: {
        name: "資料調査モデル",
        description: "資料調査モードを再度有効にした場合のみ使用します。",
      },
    },
  },
};

function storedTheme() {
  try {
    const value = localStorage.getItem(THEME_STORAGE_KEY);
    return THEME_OPTIONS.includes(value) ? value : "system";
  } catch {
    return "system";
  }
}

function storedLanguage() {
  try {
    if (storedEmployee()?.id === GUEST_EMPLOYEE_ID) return "ja";
    const value = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    return LANGUAGE_OPTIONS.includes(value) ? value : "vi";
  } catch {
    return "vi";
  }
}

function formatText(template, values = {}) {
  return String(template || "").replace(/\{(\w+)\}/g, (_, key) =>
    values[key] ?? "",
  );
}

function storedEmployee() {
  try {
    const value = JSON.parse(localStorage.getItem(EMPLOYEE_STORAGE_KEY) || "null");
    return value?.id && value?.name ? value : null;
  } catch {
    return null;
  }
}

function isGuestEmployee(employee) {
  return employee?.id === GUEST_EMPLOYEE_ID;
}

function createClientId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }

  if (globalThis.crypto?.getRandomValues) {
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (byte) =>
      byte.toString(16).padStart(2, "0"),
    );
    return [
      hex.slice(0, 4).join(""),
      hex.slice(4, 6).join(""),
      hex.slice(6, 8).join(""),
      hex.slice(8, 10).join(""),
      hex.slice(10, 16).join(""),
    ].join("-");
  }

  return `client-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 10)}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      message = body.detail || body.message || message;
    } catch {
      message = (await response.text()) || message;
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return response;
}

async function createSession() {
  const response = await api("/sessions", { method: "POST" });
  return response.json();
}

async function getResearchDemoStatus() {
  const response = await api("/research/demo");
  return response.json();
}

async function authenticateEmployee(employeeId) {
  const response = await api("/auth/employee", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ employee_id: employeeId }),
  });
  return response.json();
}

async function streamQuery(payload, onEvent, signal) {
  const response = await api("/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      for (const line of block.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        let event;
        try {
          event = JSON.parse(line.slice(6));
        } catch {
          continue;
        }
        onEvent(event);
      }
    }
  }
}

function buildConversationContext(messages, limit = 6) {
  return messages
    .filter((message) => message.role === "user" || message.role === "assistant")
    .filter((message) => message.content?.trim())
    .slice(-limit)
    .map((message) => ({
      role: message.role,
      content: message.content,
      model: message.model || "",
      mode: message.mode || "",
      answer_scope: message.answerScope || "",
    }));
}

function mergeFiles(currentFiles, incomingFiles) {
  const known = new Set(currentFiles.map((file) => `${file.name}:${file.size}`));
  const merged = [...currentFiles];
  for (const file of incomingFiles) {
    const key = `${file.name}:${file.size}`;
    if (!known.has(key)) {
      known.add(key);
      merged.push(file);
    }
  }
  return merged;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 KB";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function defaultSessionTitle(workspaceMode, language = "vi") {
  return (
    UI_TEXT[language]?.defaultTitles?.[workspaceMode] ||
    UI_TEXT.vi.defaultTitles[workspaceMode]
  );
}

function workspaceKey(workspaceMode, language = "vi") {
  return `${workspaceMode}:${language}`;
}

function sessionStorageKey(workspaceMode, language = "vi") {
  return `${SESSION_STORAGE_PREFIX}-${workspaceMode}-${language}`;
}

function createLanguageScopedState(factory) {
  return ACTIVE_MODE_KEYS.reduce((state, workspaceMode) => {
    for (const item of LANGUAGE_OPTIONS) {
      state[workspaceKey(workspaceMode, item)] = factory(workspaceMode, item);
    }
    return state;
  }, {});
}

function sessionTitleFromQuestion(question) {
  const normalized = question
    .replace(/\s+/g, " ")
    .replace(/[?.!,;:]+$/g, "")
    .trim();
  const words = normalized.split(" ");
  const title = words.slice(0, 9).join(" ");
  return words.length > 9 ? `${title}...` : title;
}

function storedSessionTitles() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_TITLE_STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

function persistSessionTitle(sessionId, title) {
  if (!sessionId || !title) return;
  try {
    const titles = storedSessionTitles();
    titles[sessionId] = title;
    localStorage.setItem(SESSION_TITLE_STORAGE_KEY, JSON.stringify(titles));
  } catch {
    // The title remains available in React state when storage is unavailable.
  }
}

function quickPromptsFor(workspaceMode, language = "vi") {
  return (
    QUICK_PROMPTS[language]?.[workspaceMode] ||
    QUICK_PROMPTS.vi[workspaceMode] ||
    []
  );
}

function localizedModelInfo(modelInfo, language = "vi") {
  if (!modelInfo) return modelInfo;
  const localized = UI_TEXT[language]?.models?.[modelInfo.id];
  if (!localized) return modelInfo;
  return {
    ...modelInfo,
    name: localized.name || modelInfo.name,
    description: localized.description || modelInfo.description,
  };
}

const EMPLOYEE_POSITION_JA = {
  "Tổng Giám đốc": "社長",
  "Phó Tổng Giám đốc": "副社長",
  "Giám đốc": "取締役",
  "Phó Giám đốc": "副取締役",
  "Trưởng phòng": "部長",
  "Phó phòng": "副部長",
  "Nhân viên": "社員",
};

function employeeGreetingFor(employee, language, t) {
  if (!employee?.name) return "";
  if (isGuestEmployee(employee)) {
    return language === "ja" ? t("common.guestWelcome") : UI_TEXT.vi.common.guestWelcome;
  }
  if (language !== "ja") {
    return employee.greeting || t("common.hello", { name: employee.name });
  }
  const position = EMPLOYEE_POSITION_JA[employee.position?.trim()];
  return position
    ? `こんにちは、${employee.name}さん（${position}）`
    : `こんにちは、${employee.name}さん`;
}

const ERROR_TRANSLATIONS_JA = [
  [/Mã nhân viên không hợp lệ/i, "社員番号が正しくありません。"],
  [/Invalid or missing agent API key/i, "Agent APIキーが無効、または未設定です。"],
  [/Gmail send chưa sẵn sàng/i, "Gmail送信機能はまだ利用できません。設定とOAuthトークンを確認してください。"],
  [/Chưa có nội dung trước đó để gửi/i, "送信できる直前の内容がありません。先に結果を取得するか、送信内容を明記してください。"],
  [/Không thể dịch nội dung cho giao diện/i, "画面表示用の翻訳を完了できませんでした。"],
  [/MES API không phản hồi trong thời gian cho phép/i, "MES APIが許可時間内に応答しませんでした。"],
  [/MES query service is not ready/i, "MES検索サービスはまだ準備できていません。"],
  [/RAG pipeline is not ready/i, "RAGパイプラインはまだ準備できていません。"],
  [/Vector store is not ready/i, "ベクトルデータベースはまだ準備できていません。"],
  [/Session not found or empty/i, "セッションが見つからないか、まだ空です。"],
  [/Preview image not found/i, "プレビュー画像が見つかりません。"],
  [/Preview path is not allowed/i, "このプレビュー画像の参照は許可されていません。"],
  [/Invalid source filename/i, "参照元ファイル名が正しくありません。"],
  [/Invalid source page/i, "参照元ページ番号が正しくありません。"],
  [/Unsupported query mode/i, "未対応の質問モードです。"],
  [/Query failed:\s*/i, "質問処理に失敗しました: "],
  [/Indexing failed:\s*/i, "インデックス処理に失敗しました: "],
  [/File exceeds/i, "ファイルサイズがアップロード上限を超えています。"],
  [/Could not extract any content from the file/i, "ファイルから内容を抽出できませんでした。"],
];

function localizeErrorMessage(message, language = "vi") {
  const text = String(message || "").trim();
  if (language !== "ja" || !text) return text;
  for (const [pattern, replacement] of ERROR_TRANSLATIONS_JA) {
    if (pattern.test(text)) {
      return text.replace(pattern, replacement);
    }
  }
  return text;
}

function App() {
  const [theme, setTheme] = useState(storedTheme);
  const [language, setLanguage] = useState(storedLanguage);
  const initialLanguageRef = useRef(storedLanguage());
  const [quickAnswersConfig, setQuickAnswersConfig] = useState({ mkac: [], mes: [], threshold: 300, max: 3 });
  const [sessionIds, setSessionIds] = useState(() =>
    createLanguageScopedState(() => ""),
  );
  const [sessionTitles, setSessionTitles] = useState(() =>
    createLanguageScopedState((workspaceMode, item) =>
      defaultSessionTitle(workspaceMode, item),
    ),
  );
  const [files, setFiles] = useState([]);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [uploadSummary, setUploadSummary] = useState(null);
  const [messagesByMode, setMessagesByMode] = useState(() =>
    createLanguageScopedState(() => []),
  );
  const [models, setModels] = useState([]);
  const [mkacStatus, setMkacStatus] = useState({
    ready: false,
    num_documents: 0,
    num_chunks: 0,
    files: [],
  });
  const [researchDemo, setResearchDemo] = useState({
    enabled: false,
    ready: false,
    session_id: "",
    files: [],
    num_chunks: 0,
  });
  const [mesStatus, setMesStatus] = useState({
    available: false,
    lots: 0,
    error_events: 0,
  });
  const [model, setModel] = useState("auto");
  const [mode, setMode] = useState("mkac");
  const [question, setQuestion] = useState("");
  const [sourcesByMode, setSourcesByMode] = useState(() =>
    createLanguageScopedState(() => []),
  );
  const [health, setHealth] = useState("checking");
  const [busy, setBusy] = useState(false);
  const [pendingAssistantId, setPendingAssistantId] = useState("");
  const [waitingMessageIndex, setWaitingMessageIndex] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({
    done: 0,
    total: 0,
    current: "",
  });
  const [copiedMessageId, setCopiedMessageId] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sourcePanelOpen, setSourcePanelOpen] = useState(false);
  const [sourcePreview, setSourcePreview] = useState(null);
  const [confirmClearOpen, setConfirmClearOpen] = useState(false);
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const [employee, setEmployee] = useState(storedEmployee);
  const [employeeCodeInput, setEmployeeCodeInput] = useState(
    () => storedEmployee()?.id || "",
  );
  const [employeeCodeError, setEmployeeCodeError] = useState("");
  const [employeeVerifying, setEmployeeVerifying] = useState(false);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const modelSelectRef = useRef(null);
  const generationControllerRef = useRef(null);
  const endRef = useRef(null);

  const text = UI_TEXT[language] || UI_TEXT.vi;
  const t = (path, values = {}) => {
    const value = path.split(".").reduce((current, key) => current?.[key], text);
    const fallback = path
      .split(".")
      .reduce((current, key) => current?.[key], UI_TEXT.vi);
    return formatText(value ?? fallback ?? path, values);
  };
  const modeText = (workspaceMode = mode) => text.modes[workspaceMode] || UI_TEXT.vi.modes[workspaceMode];

  const localizedModels = useMemo(
    () => models.map((item) => localizedModelInfo(item, language)),
    [models, language],
  );
  const selectedModel = useMemo(
    () => localizedModels.find((item) => item.id === model),
    [localizedModels, model],
  );
  const requestModel = model;
  const mkacModels = useMemo(
    () => localizedModels.filter((item) => !item.hidden_in_mkac && item.id !== "grok"),
    [localizedModels],
  );

  const currentMode = {
    ...MODE_OPTIONS[mode],
    ...modeText(mode),
  };
  const ModeIcon = currentMode.icon;
  const ThemeIcon = THEME_ICONS[theme];
  const currentWorkspaceKey = workspaceKey(mode, language);
  const sessionId = sessionIds[currentWorkspaceKey] || "";
  const messages = messagesByMode[currentWorkspaceKey] || [];
  const sources = sourcesByMode[currentWorkspaceKey] || [];
  const researchReady = true;
  const mkacAuthorized = (mode !== "mkac" && mode !== "mes") || Boolean(employee?.id && employee?.name);
  const canAsk =
    Boolean(question.trim()) &&
    !busy &&
    Boolean(sessionId) &&
    researchReady &&
    mkacAuthorized;
  const pendingTotalSize = useMemo(
    () => pendingFiles.reduce((total, file) => total + file.size, 0),
    [pendingFiles],
  );
  const latestAssistantMessage = [...messages]
    .reverse()
    .find((item) => item.role === "assistant");
  const latestSources = sources.length
    ? sources
    : latestAssistantMessage?.sources || [];

  const getSuggestions = (text, currentMode, msgId) => {
    const config = quickAnswersConfig;
    if (!config || !config[currentMode] || config[currentMode].length === 0) return [];
    if (text.length >= config.threshold) return [];
    
    const hash = msgId.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const pool = [...config[currentMode]];
    for (let i = pool.length - 1; i > 0; i--) {
      const j = (hash + i) % (i + 1);
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, config.max);
  };

  const handleQuickAnswerClick = (suggestion) => {
    if (busy) return;

    // Câu hỏi "live" (số liệu MES động) phải chạy pipeline thật để lấy dữ liệu
    // tươi, không dùng đáp án đóng hộp. Chỉ câu tĩnh mới hiện đáp án tức thì.
    if (suggestion.live || !suggestion.answer) {
      sendMessage(suggestion.question);
      return;
    }

    const userMsgId = createClientId();
    const assistantMsgId = createClientId();
    
    setBusy(true);
    setPendingAssistantId(assistantMsgId);
    
    setModeMessages(mode, (current) => [
      ...current,
      { id: userMsgId, role: "user", content: suggestion.question },
      { 
        id: assistantMsgId, 
        role: "assistant", 
        content: "",
        model: selectedModel?.name || "Cache",
        mode: mode,
        answerScope: "quick_answer",
        sources: []
      }
    ]);
    
    setTimeout(() => {
      endRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 100);

    const fullAnswer = suggestion.answer;
    let i = 0;
    
    setTimeout(() => {
      const interval = setInterval(() => {
        const charsToAppend = fullAnswer.slice(i, i + 3);
        i += 3;
        
        setModeMessages(mode, (current) =>
          current.map((item) =>
            item.id === assistantMsgId
              ? { ...item, content: item.content + charsToAppend }
              : item
          )
        );
        
        if (i >= fullAnswer.length) {
          clearInterval(interval);
          setPendingAssistantId("");
          setBusy(false);
        }
      }, 25);
    }, 1000);
  };

  useEffect(() => {
    let cancelled = false;
    const savedEmployee = storedEmployee();
    if (!savedEmployee?.id) return undefined;

    async function refreshSavedEmployee() {
      try {
        const data = await authenticateEmployee(savedEmployee.id);
        if (cancelled) return;
        setEmployee(data.employee);
        setEmployeeCodeInput(data.employee.id);
        if (isGuestEmployee(data.employee)) {
          setLanguage("ja");
        }
        localStorage.setItem(EMPLOYEE_STORAGE_KEY, JSON.stringify(data.employee));
      } catch {
        if (cancelled) return;
        setEmployee(null);
        setEmployeeCodeInput("");
        localStorage.removeItem(EMPLOYEE_STORAGE_KEY);
      }
    }

    refreshSavedEmployee();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    async function bootstrap() {
      try {
        const [
          healthResponse,
          modelResponse,
          mkacResponse,
        ] = await Promise.all([
          api("/health"),
          api(`/models?language=${encodeURIComponent(language)}`),
          api("/knowledge/mkac/status"),
        ]);
        const healthData = await healthResponse.json();
        const modelData = await modelResponse.json();
        const mkacData = await mkacResponse.json();

        setModels(modelData.models || []);
        setMkacStatus(mkacData);
        setMesStatus(healthData.mes_database || {});
        setModel((current) => {
          const nextDefault = modelData.default || "auto";
          return current === "auto" || current === "grok" ? nextDefault : current;
        });
        setHealth("online");

        const legacySession = localStorage.getItem(LEGACY_SESSION_STORAGE_KEY);
        const resolvedSessions = {};

        await Promise.all(
          ACTIVE_MODE_KEYS.flatMap((workspaceMode) =>
            LANGUAGE_OPTIONS.map(async (workspaceLanguage) => {
              const scopedKey = workspaceKey(workspaceMode, workspaceLanguage);
              const storageKey = sessionStorageKey(workspaceMode, workspaceLanguage);
              const legacyModeSession =
                workspaceLanguage === language
                  ? localStorage.getItem(LEGACY_MODE_SESSION_STORAGE_KEYS[workspaceMode])
                  : null;
              const legacySharedSession =
                workspaceMode === "mkac" && workspaceLanguage === language
                  ? legacySession
                  : null;
              const storedSession =
                localStorage.getItem(storageKey) ||
                legacyModeSession ||
                legacySharedSession;

              if (storedSession) {
                try {
                  const infoResponse = await api(`/sessions/${storedSession}`);
                  await infoResponse.json();
                  resolvedSessions[scopedKey] = storedSession;
                  return;
                } catch (sessionError) {
                  localStorage.removeItem(storageKey);
                }
              }

              const session = await createSession();
              resolvedSessions[scopedKey] = session.session_id;
            }),
          ),
        );

        setSessionIds(resolvedSessions);
        const savedTitles = storedSessionTitles();
        setSessionTitles(
          createLanguageScopedState((workspaceMode, workspaceLanguage) => {
            const scopedKey = workspaceKey(workspaceMode, workspaceLanguage);
            return (
              savedTitles[resolvedSessions[scopedKey]] ||
              defaultSessionTitle(workspaceMode, workspaceLanguage)
            );
          }),
        );
        Object.entries(resolvedSessions).forEach(([scopedKey, id]) => {
          const [workspaceMode, workspaceLanguage] = scopedKey.split(":");
          localStorage.setItem(sessionStorageKey(workspaceMode, workspaceLanguage), id);
        });
        localStorage.removeItem(LEGACY_SESSION_STORAGE_KEY);
      } catch (bootstrapError) {
        setHealth("offline");
        setError(localizeErrorMessage(bootstrapError.message, language));
      }
    }
    bootstrap();
  }, []);

  useEffect(() => {
    let cancelled = false;
    setQuickAnswersConfig({ mkac: [], mes: [], threshold: 300, max: 3 });

    async function loadQuickAnswers() {
      try {
        const [qaMkacRes, qaMesRes] = await Promise.all([
          api(`/quick-answers?mode=mkac&language=${encodeURIComponent(language)}`),
          api(`/quick-answers?mode=mes&language=${encodeURIComponent(language)}`),
        ]);
        const qaMkacData = await qaMkacRes.json();
        const qaMesData = await qaMesRes.json();
        if (cancelled) return;
        setQuickAnswersConfig({
          mkac: qaMkacData.suggestions || [],
          mes: qaMesData.suggestions || [],
          threshold: qaMkacData.short_answer_threshold || 300,
          max: qaMkacData.max_suggestions || 3,
        });
      } catch {
        if (cancelled) return;
        setQuickAnswersConfig({ mkac: [], mes: [], threshold: 300, max: 3 });
      }
    }

    loadQuickAnswers();
    return () => {
      cancelled = true;
    };
  }, [language]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const applyTheme = () => {
      const resolvedTheme =
        theme === "system" ? (mediaQuery.matches ? "dark" : "light") : theme;
      document.documentElement.dataset.theme = resolvedTheme;
      document.documentElement.style.colorScheme = resolvedTheme;
      document
        .querySelector('meta[name="theme-color"]')
        ?.setAttribute("content", resolvedTheme === "dark" ? "#15191f" : "#f4f6f8");
    };

    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Theme still works for the current page when storage is unavailable.
    }
    applyTheme();
    mediaQuery.addEventListener("change", applyTheme);
    return () => mediaQuery.removeEventListener("change", applyTheme);
  }, [theme]);

  useEffect(() => {
    try {
      localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
    } catch {
      // The in-memory language state still works when storage is unavailable.
    }
    document.documentElement.lang = language === "ja" ? "ja" : "vi";
  }, [language]);

  useEffect(() => {
    setSessionTitles((current) => {
      const next = { ...current };
      for (const workspaceMode of VISIBLE_MODE_KEYS) {
        for (const item of LANGUAGE_OPTIONS) {
          const scopedKey = workspaceKey(workspaceMode, item);
          const defaultTitles = LANGUAGE_OPTIONS.map(
            (candidate) => UI_TEXT[candidate].defaultTitles[workspaceMode],
          );
          if (defaultTitles.includes(current[scopedKey])) {
            next[scopedKey] = defaultSessionTitle(workspaceMode, item);
          }
        }
      }
      return next;
    });
  }, [language]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 190)}px`;
  }, [question]);

  useEffect(() => {
    if (!modelMenuOpen) return undefined;

    function closeModelMenu(event) {
      if (event.key === "Escape") {
        setModelMenuOpen(false);
        modelSelectRef.current?.querySelector(".model-select-trigger")?.focus();
      } else if (
        event.type === "pointerdown" &&
        !modelSelectRef.current?.contains(event.target)
      ) {
        setModelMenuOpen(false);
      }
    }

    document.addEventListener("pointerdown", closeModelMenu);
    document.addEventListener("keydown", closeModelMenu);
    return () => {
      document.removeEventListener("pointerdown", closeModelMenu);
      document.removeEventListener("keydown", closeModelMenu);
    };
  }, [modelMenuOpen]);

  useEffect(() => {
    if (!busy || !pendingAssistantId) {
      setWaitingMessageIndex(0);
      return undefined;
    }

    const timer = window.setInterval(() => {
      setWaitingMessageIndex(
        (current) => (current + 1) % text.waiting.length,
      );
    }, 2400);

    return () => window.clearInterval(timer);
  }, [busy, pendingAssistantId, text.waiting.length]);

  useEffect(() => {
    if (!sourcePreview) return undefined;
    function closePreview(event) {
      if (event.key === "Escape") setSourcePreview(null);
    }
    document.addEventListener("keydown", closePreview);
    return () => document.removeEventListener("keydown", closePreview);
  }, [sourcePreview]);

  useEffect(() => {
    if (!confirmClearOpen) return undefined;
    function closeConfirm(event) {
      if (event.key === "Escape") setConfirmClearOpen(false);
    }
    document.addEventListener("keydown", closeConfirm);
    return () => document.removeEventListener("keydown", closeConfirm);
  }, [confirmClearOpen]);

  function setModeMessages(workspaceMode, updater, uiLanguage = language) {
    const scopedKey = workspaceKey(workspaceMode, uiLanguage);
    setMessagesByMode((current) => ({
      ...current,
      [scopedKey]:
        typeof updater === "function"
          ? updater(current[scopedKey] || [])
          : updater,
    }));
  }

  function setModeSources(workspaceMode, updater, uiLanguage = language) {
    const scopedKey = workspaceKey(workspaceMode, uiLanguage);
    setSourcesByMode((current) => ({
      ...current,
      [scopedKey]:
        typeof updater === "function"
          ? updater(current[scopedKey] || [])
          : updater,
    }));
  }

  async function resetSession(workspaceMode = mode, uiLanguage = language) {
    setError("");
    const data = await createSession();
    const scopedKey = workspaceKey(workspaceMode, uiLanguage);
    setSessionIds((current) => ({
      ...current,
      [scopedKey]: data.session_id,
    }));
    localStorage.setItem(sessionStorageKey(workspaceMode, uiLanguage), data.session_id);
    setSessionTitles((current) => ({
      ...current,
      [scopedKey]: defaultSessionTitle(workspaceMode, uiLanguage),
    }));
    setModeMessages(workspaceMode, [], uiLanguage);
    setModeSources(workspaceMode, [], uiLanguage);
    setSidebarOpen(false);
  }

  function useResearchDemoSession() {
    setError(t("common.demoNotIndexed"));
  }

  function switchMode(nextMode) {
    if (!VISIBLE_MODE_KEYS.includes(nextMode)) return;
    if (nextMode === mode || busy || uploading) return;
    setMode(nextMode);
    setQuestion("");
    setError("");
    setSidebarOpen(false);
    setSourcePanelOpen(false);
    setPendingAssistantId("");
  }

  function clearConversation() {
    setModeMessages(mode, [], language);
    setModeSources(mode, [], language);
    setQuestion("");
    setError("");
    setSourcePanelOpen(false);
    setSourcePreview(null);
    setConfirmClearOpen(false);
  }

  function sourcePreviewUrl(source, sourceMode = mode) {
    const previewSessionId =
      sessionIds[workspaceKey(sourceMode, language)] || sessionId;
    const params = new URLSearchParams({
      session_id: previewSessionId,
      mode: sourceMode,
      file: source.file || "",
      page: String(source.page || 1),
      language,
    });
    return `/sources/preview?${params.toString()}`;
  }

  function openSourcePreview(source, sourceMode = mode) {
    if (source.url) {
      window.open(source.url, "_blank", "noopener,noreferrer");
      return;
    }
    if (
      typeof window !== "undefined" &&
      window.matchMedia("(max-width: 900px)").matches
    ) {
      setSourcePanelOpen(false);
    }
    setSourcePreview({
      source,
      mode: sourceMode,
      imageFailed: false,
    });
  }

  function onModeTabKeyDown(event, currentModeKey) {
    const modeKeys = VISIBLE_MODE_KEYS;
    const currentIndex = modeKeys.indexOf(currentModeKey);
    let nextIndex = currentIndex;

    if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % modeKeys.length;
    else if (event.key === "ArrowLeft") {
      nextIndex = (currentIndex - 1 + modeKeys.length) % modeKeys.length;
    } else if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = modeKeys.length - 1;
    else return;

    event.preventDefault();
    const nextMode = modeKeys[nextIndex];
    switchMode(nextMode);
    document.querySelector(`[data-mode="${nextMode}"]`)?.focus();
  }

  async function copyAnswer(message) {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopiedMessageId(message.id);
      window.setTimeout(() => {
        setCopiedMessageId((current) => (current === message.id ? "" : current));
      }, 1800);
    } catch {
      setError(t("common.copyFailed"));
    }
  }

  function stopGeneration() {
    generationControllerRef.current?.abort();
  }

  function cycleTheme() {
    setTheme((current) => {
      const currentIndex = THEME_OPTIONS.indexOf(current);
      return THEME_OPTIONS[(currentIndex + 1) % THEME_OPTIONS.length];
    });
  }

  function cycleLanguage() {
    if (busy || uploading) return;
    setQuestion("");
    setError("");
    setSourcePanelOpen(false);
    setSourcePreview(null);
    setPendingAssistantId("");
    setLanguage((current) => (current === "vi" ? "ja" : "vi"));
  }

  function addPendingFiles(fileList) {
    const incoming = Array.from(fileList || []);
    if (incoming.length === 0) return;
    setPendingFiles((current) => mergeFiles(current, incoming));
    setUploadSummary(null);
    setSidebarOpen(true);
  }

  function removePendingFile(indexToRemove) {
    setPendingFiles((current) =>
      current.filter((_, index) => index !== indexToRemove),
    );
  }

  async function uploadDocuments() {
    if (!sessionId || pendingFiles.length === 0) return;
    setUploading(true);
    setError("");
    setUploadSummary(null);
    setUploadProgress({ done: 0, total: pendingFiles.length, current: "" });
    const uploaded = [];
    let totalChunks = 0;

    try {
      for (const [index, file] of pendingFiles.entries()) {
        setUploadProgress({
          done: index,
          total: pendingFiles.length,
          current: file.name,
        });
        const formData = new FormData();
        formData.append("file", file);
        const response = await api(`/sessions/${sessionId}/upload`, {
          method: "POST",
          body: formData,
        });
        const result = await response.json();
        uploaded.push(result.filename);
        totalChunks += Number(result.num_chunks || 0);
        setUploadProgress({
          done: index + 1,
          total: pendingFiles.length,
          current: file.name,
        });
      }
      setFiles((current) => [...new Set([...current, ...uploaded])]);
      setPendingFiles([]);
      setUploadSummary({
        files: uploaded.length,
        chunks: totalChunks,
      });
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (uploadError) {
      setError(localizeErrorMessage(uploadError.message, language));
    } finally {
      setUploading(false);
    }
  }

  async function removeFile(filename) {
    setError("");
    try {
      await api(`/sessions/${sessionId}/files/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      setFiles((current) => current.filter((item) => item !== filename));
    } catch (removeError) {
      setError(localizeErrorMessage(removeError.message, language));
    }
  }

  async function sendMessage(prompt = question) {
    const cleanQuestion = prompt.trim();
    if ((mode === "mkac" || mode === "mes") && !mkacAuthorized) {
      setEmployeeCodeError(t("common.employeeRequired"));
      return;
    }
    if (!cleanQuestion || busy || !sessionId) return;

    const requestMode = mode;
    const requestLanguage = language;
    const requestWorkspaceKey = workspaceKey(requestMode, requestLanguage);
    const requestSessionId = sessionId;
    const conversationContext = buildConversationContext(messages);
    if (messages.length === 0) {
      const title = sessionTitleFromQuestion(cleanQuestion);
      setSessionTitles((current) => ({ ...current, [requestWorkspaceKey]: title }));
      persistSessionTitle(requestSessionId, title);
    }
    const assistantId = createClientId();
    const controller = new AbortController();
    generationControllerRef.current = controller;
    setQuestion("");
    setError("");
    setModeSources(requestMode, [], requestLanguage);
    setSourcePanelOpen(false);
    setModeMessages(requestMode, (current) => [
      ...current,
      { id: createClientId(), role: "user", content: cleanQuestion },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        status: "",
        model: selectedModel?.name || model,
        mode: requestMode,
        answerScope:
          requestMode === "mkac"
            ? "mkac"
            : requestMode === "mes"
              ? "mes_database"
              : "research",
        sources: [],
      },
    ], requestLanguage);
    setWaitingMessageIndex(0);
    setPendingAssistantId(assistantId);
    setBusy(true);

    try {
      await streamQuery(
        {
          session_id: requestSessionId,
          question: cleanQuestion,
          stream: true,
          model: requestModel,
          mode: requestMode,
          ui_language: requestLanguage,
          employee_id: (requestMode === "mkac" || requestMode === "mes") ? employee?.id : undefined,
          conversation_context: conversationContext,
        },
        (event) => {
          if (event.type === "status") {
            setModeMessages(
              requestMode,
              (current) =>
                current.map((item) =>
                  item.id === assistantId
                    ? { ...item, status: event.message || item.status }
                    : item,
                ),
              requestLanguage,
            );
          }
          if (event.type === "sources") {
            setModeSources(requestMode, event.sources || [], requestLanguage);
            setModeMessages(
              requestMode,
              (current) =>
                current.map((item) =>
                  item.id === assistantId
                    ? { ...item, sources: event.sources || [] }
                    : item,
                ),
              requestLanguage,
            );
          }
          if (event.type === "meta") {
            setModeMessages(
              requestMode,
              (current) =>
                current.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        model: event.model || item.model,
                        mode: event.mode || item.mode,
                        answerScope: event.answer_scope || item.answerScope,
                      }
                    : item,
                ),
              requestLanguage,
            );
          }
          if (event.type === "token") {
            setPendingAssistantId("");
            setModeMessages(
              requestMode,
              (current) =>
                current.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        content: item.content + (event.content || ""),
                        status: "",
                      }
                    : item,
                ),
              requestLanguage,
            );
          }
          if (event.type === "replace") {
            // Bản đã hậu xử lý (bỏ <think>, cắt marker...) — thay toàn bộ nội
            // dung đã stream. Chỉ đến khi cleanup thực sự đổi nội dung.
            setPendingAssistantId("");
            setModeMessages(
              requestMode,
              (current) =>
                current.map((item) =>
                  item.id === assistantId
                    ? { ...item, content: event.content || "", status: "" }
                    : item,
                ),
              requestLanguage,
            );
          }
          if (event.type === "error") {
            throw new Error(
              localizeErrorMessage(
                event.message || t("common.requestFailed", { message: "" }),
                requestLanguage,
              ),
            );
          }
        },
        controller.signal,
      );
    } catch (queryError) {
      const wasStopped = queryError.name === "AbortError";
      if (queryError.status === 403 && requestMode === "mkac") {
        setEmployee(null);
        setEmployeeCodeInput("");
        try {
          localStorage.removeItem(EMPLOYEE_STORAGE_KEY);
        } catch {
          // Ignore storage failures.
        }
      }
      const localizedMessage = localizeErrorMessage(queryError.message, requestLanguage);
      if (!wasStopped) setError(localizedMessage);
      setModeMessages(
        requestMode,
        (current) =>
          current.map((item) =>
            item.id === assistantId
              ? {
                  ...item,
                  content:
                    item.content ||
                    (wasStopped
                      ? t("common.stoppedResponse")
                      : t("common.requestFailed", { message: localizedMessage })),
                  stopped: wasStopped,
                }
              : item,
          ),
        requestLanguage,
      );
    } finally {
      generationControllerRef.current = null;
      setPendingAssistantId("");
      setBusy(false);
    }
  }

  async function verifyEmployeeCode(event) {
    event.preventDefault();
    const normalizedCode = employeeCodeInput.trim();

    if (!normalizedCode) {
      setEmployeeCodeError(t("common.invalidEmployee"));
      return;
    }

    setEmployeeVerifying(true);
    setEmployeeCodeError("");
    try {
      const data = await authenticateEmployee(normalizedCode);
      setEmployee(data.employee);
      if (isGuestEmployee(data.employee)) {
        setLanguage("ja");
      }
      try {
        localStorage.setItem(EMPLOYEE_STORAGE_KEY, JSON.stringify(data.employee));
      } catch {
        // localStorage can be blocked in private browsing; the in-memory state still works.
      }
    } catch {
      setEmployee(null);
      setEmployeeCodeError(t("common.invalidEmployee"));
      try {
        localStorage.removeItem(EMPLOYEE_STORAGE_KEY);
      } catch {
        // Ignore storage failures.
      }
    } finally {
      setEmployeeVerifying(false);
    }
  }

  function updateEmployeeCodeInput(value) {
    setEmployeeCodeInput(value.replace(/\D/g, "").slice(0, 6));
    setEmployeeCodeError("");
  }

  function logoutEmployee() {
    setEmployee(null);
    setEmployeeCodeInput("");
    setEmployeeCodeError("");
    setQuestion("");
    for (const item of LANGUAGE_OPTIONS) {
      setModeMessages("mkac", [], item);
      setModeSources("mkac", [], item);
    }
    try {
      localStorage.removeItem(EMPLOYEE_STORAGE_KEY);
      localStorage.removeItem("meibook-mkac-employee-code");
    } catch {
      // Ignore storage failures.
    }
  }

  function onSubmit(event) {
    event.preventDefault();
    sendMessage(textareaRef.current?.value || question);
  }

  function onDrop(event) {
    event.preventDefault();
    setDragActive(false);
    addPendingFiles(event.dataTransfer.files);
  }

  return (
    <div className={`app-shell ${mode !== "research" ? "mkac-layout" : ""}`}>
      {mode === "research" && (
        <>
          <button
            className="mobile-menu icon-button"
            type="button"
            title={t("common.openDocuments")}
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={20} />
          </button>

          {sidebarOpen && (
            <button
              className="sidebar-backdrop"
              type="button"
              aria-label={t("common.closeDocuments")}
              onClick={() => setSidebarOpen(false)}
            />
          )}

          <aside className={`document-sidebar ${sidebarOpen ? "open" : ""}`}>
            <div className="brand-row">
              <div className="brand-mark logo-mark">
                <img src="/mkac-logo.png" alt="MKAC" />
              </div>
              <div>
                <strong>MKAC</strong>
                <span>{t("common.assistantName")}</span>
              </div>
              <button
                className="icon-button close-sidebar"
                type="button"
                title={t("common.close")}
                onClick={() => setSidebarOpen(false)}
              >
                <X size={18} />
              </button>
            </div>

            <div className="session-strip">
              <div>
                <span>{t("common.workSession")}</span>
                <strong className="session-title" title={sessionTitles.research}>
                  {sessionTitles.research}
                </strong>
              </div>
              <button
                className="icon-button"
                type="button"
                title={t("common.newSession")}
                onClick={() => resetSession("research")}
                disabled={busy || uploading}
              >
                <RefreshCcw size={17} />
              </button>
            </div>

            <section className="sidebar-section">
              <div className="section-heading">
                <span>{t("common.researchDocuments")}</span>
                <span className="count-badge">{files.length}</span>
              </div>

              <input
                ref={fileInputRef}
                className="hidden-input"
                type="file"
                multiple
                accept=".pdf,.docx,.xlsx,.pptx,.html,.png,.jpg,.jpeg"
                aria-label={t("common.chooseDocument")}
                onChange={(event) => addPendingFiles(event.target.files)}
              />

              <button
                className={`upload-zone ${dragActive ? "dragging" : ""}`}
                type="button"
                onClick={() => fileInputRef.current?.click()}
                onDragEnter={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setDragActive(false)}
                onDrop={onDrop}
              >
                <UploadCloud size={23} />
                <span>{t("common.chooseDocument")}</span>
                <small>{t("common.supportedFiles")}</small>
              </button>

              {pendingFiles.length > 0 && (
                <div className="pending-panel">
                  <div className="pending-header">
                    <span>{pendingFiles.length} {t("common.files")}</span>
                    <span>{formatBytes(pendingTotalSize)}</span>
                  </div>
                  <div className="pending-list">
                    {pendingFiles.map((file, index) => (
                      <div className="pending-file" key={`${file.name}-${file.size}`}>
                        <FileUp size={15} />
                        <span title={file.name}>{file.name}</span>
                        <button
                          className="icon-button subtle"
                          type="button"
                          title={t("common.removeFile", { name: file.name })}
                          onClick={() => removePendingFile(index)}
                          disabled={uploading}
                        >
                          <X size={14} />
                        </button>
                      </div>
                    ))}
                  </div>
                  <button
                    className="index-button"
                    type="button"
                    onClick={uploadDocuments}
                    disabled={uploading}
                  >
                    {uploading ? <Loader2 className="spin" size={16} /> : <Database size={16} />}
                    {uploading
                      ? t("common.indexing", {
                          done: uploadProgress.done,
                          total: uploadProgress.total,
                        })
                      : t("common.indexDocument")}
                  </button>
                  {uploading && (
                    <div className="upload-progress" role="status" aria-live="polite">
                      <div className="upload-progress-copy">
                        <span>{t("common.processing")}</span>
                        <strong title={uploadProgress.current}>
                          {uploadProgress.current}
                        </strong>
                      </div>
                      <div className="upload-progress-track" aria-hidden="true">
                        <span />
                      </div>
                    </div>
                  )}
                </div>
              )}

              {uploadSummary && (
                <div className="upload-summary" role="status" aria-live="polite">
                  <CheckCircle2 size={15} />
                  <span>
                    {t("common.indexedSummary", {
                      files: uploadSummary.files,
                      chunks: uploadSummary.chunks,
                    })}
                  </span>
                </div>
              )}

              <div className="file-list">
                {files.map((filename) => (
                  <div className="file-item" key={filename}>
                    <FileText size={17} />
                    <span title={filename}>{filename}</span>
                    <button
                      className="icon-button subtle danger"
                      type="button"
                      title={t("common.deleteFile", { name: filename })}
                      onClick={() => removeFile(filename)}
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                ))}
                {files.length === 0 && (
                  <div className="empty-files">
                    <FileText size={20} />
                    <span>{t("common.noDocuments")}</span>
                  </div>
                )}
              </div>
            </section>

            <div className="sidebar-footer">
              <div className={`service-state ${health}`}>
                {health === "online" ? <CheckCircle2 size={15} /> : <Server size={15} />}
                <span>{health === "online" ? t("common.onlineMachine") : t("common.checking")}</span>
              </div>
              <span className="security-chip">
                <ShieldCheck size={14} />
                RAG
              </span>
            </div>
          </aside>
        </>
      )}

      <main className={`workspace ${sourcePanelOpen ? "" : "sources-collapsed"}`}>
        <header className="workspace-header">
          <div className="header-title">
            <div
              className={`mode-mark ${
                mode === "mkac" || mode === "mes" ? "logo-mark" : mode
              }`}
            >
              {mode === "mkac" || mode === "mes" ? (
                <img src="/mkac-logo.png" alt="MKAC" />
              ) : (
                <ModeIcon size={20} />
              )}
            </div>
            <div>
              <strong>{currentMode.title}</strong>
              <span>
                {mode === "mkac"
                  ? `${mkacStatus.num_documents} ${t("common.internalDocuments")}`
                  : mode === "mes"
                    ? mesStatus.available
                      ? `${mesStatus.lots || 0} Lot · ${mesStatus.error_events || 0} ${t("common.errorRecords")}`
                      : modeText("mes").unavailable
                    : `${files.length} ${t("common.sessionDocuments")}`}
              </span>
            </div>
          </div>

          <div className="header-actions">
            <div className="mode-tabs" role="tablist" aria-label={t("common.qaModes")}>
              {VISIBLE_MODE_KEYS.map((key) => {
                const option = MODE_OPTIONS[key];
                const Icon = option.icon;
                const optionText = modeText(key);
                return (
                  <button
                    key={key}
                    type="button"
                    data-mode={key}
                    className={mode === key ? `active ${key}` : ""}
                    onClick={() => switchMode(key)}
                    onKeyDown={(event) => onModeTabKeyDown(event, key)}
                    disabled={busy || uploading}
                    role="tab"
                    aria-selected={mode === key}
                    aria-controls={`${key}-conversation`}
                    tabIndex={mode === key ? 0 : -1}
                    title={optionText.title}
                  >
                    <Icon size={17} />
                    <span className="mode-label-full">{optionText.label}</span>
                    <span className="mode-label-short">{optionText.shortLabel}</span>
                  </button>
                );
              })}
            </div>

            {mode === "research" && (
              <div className="model-select locked accent-grok" title={modeText("research").lockedModel}>
                <Bot size={17} />
                <span className="model-select-trigger">
                  <span>{selectedModel?.name || modeText("research").lockedModel}</span>
                </span>
              </div>
            )}

            <button
              className="icon-button header-tool theme-toggle"
              type="button"
              title={t("common.interfaceTitle", { theme: t(`theme.${theme}`) })}
              aria-label={t("common.interfaceAria", { theme: t(`theme.${theme}`) })}
              onClick={cycleTheme}
            >
              <ThemeIcon size={18} />
            </button>

            <button
              className="language-toggle"
              type="button"
              title={t("common.languageTitle", {
                language: language === "vi" ? "VN" : "JP",
              })}
              aria-label={t("common.languageTitle", {
                language: language === "vi" ? "VN" : "JP",
              })}
              onClick={cycleLanguage}
              disabled={busy || uploading}
            >
              <span className={language === "vi" ? "active" : ""}>VN</span>
              <span className={language === "ja" ? "active" : ""}>JP</span>
            </button>

            <button
              className="icon-button panel-toggle"
              type="button"
              title={sourcePanelOpen ? t("common.hideSources") : t("common.showSources")}
              aria-label={sourcePanelOpen ? t("common.hideSources") : t("common.showSources")}
              onClick={() => setSourcePanelOpen((open) => !open)}
              disabled={latestSources.length === 0}
              aria-controls="source-panel"
              aria-expanded={sourcePanelOpen}
            >
              {sourcePanelOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
              {latestSources.length > 0 && (
                <span className="panel-count">{latestSources.length}</span>
              )}
            </button>

            <span className="header-divider" role="separator" aria-hidden="true" />

            {mode === "mkac" && employee && (
              <button
                className="icon-button header-tool"
                type="button"
                title={t("common.employeeLogout")}
                aria-label={t("common.employeeLogout")}
                onClick={logoutEmployee}
                disabled={busy}
              >
                <LogOut size={18} />
              </button>
            )}

            <button
              className="icon-button header-tool danger"
              type="button"
              title={t("common.clearConversation")}
              aria-label={t("common.clearConversation")}
              onClick={() => setConfirmClearOpen(true)}
              disabled={busy || messages.length === 0}
            >
              <Trash2 size={18} />
            </button>
          </div>
        </header>

        <div className="workspace-body">
          <section
            id={`${mode}-conversation`}
            className="conversation"
            role="tabpanel"
            aria-label={currentMode.title}
            aria-busy={busy}
          >
            <div className="conversation-scroll">
              {(mode === "mkac" || mode === "mes") && !mkacAuthorized ? (
                <div className="employee-gate">
                  <form className="employee-card" onSubmit={verifyEmployeeCode}>
                    <div className="employee-logo">
                      <img src="/mkac-logo.png" alt="MKAC" />
                    </div>
                    <h1>{t("common.employeeAuth")}</h1>
                    <p>{modeText(mode).authHint}</p>
                    <label className="employee-field">
                      <span>{t("common.employeeCode")}</span>
                      <input
                        value={employeeCodeInput}
                        onChange={(event) =>
                          updateEmployeeCodeInput(event.target.value)
                        }
                        inputMode="numeric"
                        autoComplete="off"
                        maxLength={6}
                        placeholder={t("common.employeeCode")}
                        disabled={employeeVerifying}
                        autoFocus
                      />
                    </label>
                    {employeeCodeError && (
                      <span className="employee-error" role="alert">
                        {employeeCodeError}
                      </span>
                    )}
                    <button
                      className="employee-submit"
                      type="submit"
                      disabled={employeeVerifying}
                    >
                      {employeeVerifying && <Loader2 className="spin" size={16} />}
                      {employeeVerifying ? t("common.verifying") : t("common.continue")}
                    </button>
                  </form>
                </div>
              ) : messages.length === 0 ? (
                <div className="empty-conversation">
                  <div className="empty-copy">
                    <div className={`empty-icon ${mode}`}>
                      <ModeIcon size={30} />
                    </div>
                    <h1>
                      {mode === "mkac" && employee?.name
                        ? employeeGreetingFor(employee, language, t)
                        : currentMode.title}
                    </h1>
                    <p>
                      {mode === "mkac"
                        ? modeText("mkac").empty
                        : mode === "mes"
                          ? modeText("mes").empty
                          : files.length > 0
                            ? modeText("research").emptyReady
                            : modeText("research").empty}
                    </p>
                    {mode === "research" && files.length === 0 && (
                      <div className="research-start-actions">
                        <button
                          className="empty-upload-button"
                          type="button"
                          onClick={() => fileInputRef.current?.click()}
                        >
                          <UploadCloud size={17} />
                          {t("common.chooseToStart")}
                        </button>
                        <button
                          className="empty-upload-button secondary"
                          type="button"
                          onClick={useResearchDemoSession}
                          disabled={!researchDemo.ready}
                        >
                          <FlaskConical size={17} />
                          {t("common.tryDemo")}
                        </button>
                      </div>
                    )}
                  </div>

                  {mode !== "research" || files.length > 0 ? (
                    <div className="prompt-section">
                      <p className="prompt-label">{t("common.promptLabel")}</p>
                      <div className="prompt-grid">
                        {quickPromptsFor(mode, language).map((prompt) => (
                          <button
                            key={prompt}
                            type="button"
                            onClick={() => sendMessage(prompt)}
                            disabled={!researchReady || !mkacAuthorized}
                          >
                            <Search size={16} />
                            <span>{prompt}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  <div className="empty-status">
                    <span
                      className={`status-dot ${health === "online" ? "ready" : "offline"}`}
                    />
                    <span>
                      {health === "online" ? t("common.ready") : t("common.notReady")}
                    </span>
                    <span className="status-sep" aria-hidden="true">·</span>
                    <span>
                      {mode === "mkac"
                        ? `${mkacStatus.num_documents} ${modeText("mkac").metric}`
                        : mode === "mes"
                          ? `${mesStatus.lots || 0} ${modeText("mes").metric}`
                          : `${files.length} ${modeText("research").metric}`}
                    </span>
                  </div>
                </div>
              ) : (
                <div
                  className="message-list"
                  role="log"
                  aria-live="polite"
                  aria-relevant="additions text"
                >
                  {messages.map((message) => (
                    <article className={`message ${message.role}`} key={message.id}>
                      <div className="message-avatar">
                        {message.role === "user" ? "U" : <Bot size={18} />}
                      </div>
                      <div className="message-content">
                        {message.role === "assistant" && (
                          <div className="message-meta">
                            <span>{message.model}</span>
                            {message.stopped && <span>{t("common.stopped")}</span>}
                            <span>
                              {message.answerScope === "general"
                                ? t("common.noResult")
                                : message.answerScope === "web"
                                  ? t("common.webSearch")
                                : message.answerScope === "mes"
                                  ? t("common.mesData")
                                : message.answerScope === "mes_database"
                                  ? t("common.mesSnapshot")
                                : message.mode === "research"
                                  ? t("common.research")
                                  : t("common.mkacSource")}
                            </span>
                          </div>
                        )}
                        {message.content ? (
                          <div className="message-body">
                            <MessageMarkdown content={message.content} />
                            {busy &&
                              message.role === "assistant" &&
                              message.id === latestAssistantMessage?.id && (
                                <span
                                  className="stream-cursor"
                                  aria-hidden="true"
                                />
                              )}
                          </div>
                        ) : message.id === pendingAssistantId ? (
                          <div
                            className="waiting-status"
                            role="status"
                            aria-live="polite"
                          >
                            <div className="waiting-head">
                              <Loader2 className="spin" size={17} />
                              <span className="waiting-copy">
                                <span key={message.status || waitingMessageIndex}>
                                  {message.status || text.waiting[waitingMessageIndex]}
                                </span>
                                {language === "ja" && (
                                  <small>{t("common.languageConverting")}</small>
                                )}
                              </span>
                            </div>
                            <div className="skeleton" aria-hidden="true">
                              <span className="skeleton-line" />
                              <span className="skeleton-line" />
                              <span className="skeleton-line short" />
                            </div>
                          </div>
                        ) : null}
                        {message.role === "assistant" && message.content && (
                          <div className="message-actions">
                            <details className="ai-disclosure">
                              <summary title={t("common.aiInfo")}>
                                <Sparkles size={14} />
                                <span>AI</span>
                              </summary>
                              <div>
                                <strong>{message.model}</strong>
                                <span>
                                  {message.answerScope === "web"
                                    ? t("answerScope.web")
                                    : message.answerScope === "mes"
                                      ? t("answerScope.mes")
                                    : message.answerScope === "mes_database"
                                      ? t("answerScope.mes_database")
                                    : message.answerScope === "research"
                                      ? t("answerScope.research")
                                      : message.answerScope === "mkac"
                                        ? t("answerScope.mkac")
                                        : t("answerScope.fallback")}
                                </span>
                              </div>
                            </details>
                            <button
                              className="message-action-button"
                              type="button"
                              title={t("common.copyAnswer")}
                              onClick={() => copyAnswer(message)}
                            >
                              {copiedMessageId === message.id ? (
                                <Check size={14} />
                              ) : (
                                <Copy size={14} />
                              )}
                              <span>
                                {copiedMessageId === message.id ? t("common.copied") : t("common.copy")}
                              </span>
                            </button>
                          </div>
                        )}
                        {message.role === "assistant" && message.sources?.length > 0 && (
                          <details className="message-sources">
                            <summary>{message.sources.length} {t("common.sources")}</summary>
                            {message.sources.map((source, index) => (
                              <div key={`${source.file}-${source.page}-${index}`}>
                                {source.url ? (
                                  <a href={source.url} target="_blank" rel="noreferrer">
                                    <strong>{source.file}</strong>
                                  </a>
                                ) : (
                                  <button
                                    className="inline-source-button"
                                    type="button"
                                    onClick={() => openSourcePreview(source, message.mode)}
                                  >
                                    <strong>{source.file}</strong>
                                  </button>
                                )}
                                <span>
                                  {source.url
                                    ? t("common.webSource")
                                    : t("common.page", { page: source.page })}
                                </span>
                              </div>
                            ))}
                          </details>
                        )}
                        {message.role === "assistant" &&
                          message.id !== pendingAssistantId &&
                          message.content &&
                          getSuggestions(message.content, mode, message.id).length > 0 && (
                            <div className="message-suggestions">
                              <p>{t("common.suggestions")}</p>
                              <ul className="suggestion-list">
                                {getSuggestions(message.content, mode, message.id).map(
                                  (suggestion, i) => (
                                    <li key={i}>
                                      <button
                                        type="button"
                                        onClick={() => handleQuickAnswerClick(suggestion)}
                                        title={t("common.autoAsk")}
                                      >
                                        {suggestion.question}
                                      </button>
                                    </li>
                                  )
                                )}
                              </ul>
                            </div>
                          )}
                      </div>
                    </article>
                  ))}
                  <div ref={endRef} />
                </div>
              )}
            </div>

            {error && (
              <div className="error-banner" role="alert">
                <AlertCircle size={17} />
                <span>{error}</span>
                <button
                  className="icon-button"
                  type="button"
                  title={t("common.close")}
                  onClick={() => setError("")}
                >
                  <X size={16} />
                </button>
              </div>
            )}

            {mkacAuthorized && (
              <form className="composer" onSubmit={onSubmit}>
                {mode === "research" && (
                  <button
                    className="composer-attach icon-button"
                    type="button"
                    title={t("common.addResearchDocument")}
                    onClick={() => fileInputRef.current?.click()}
                    disabled={busy || uploading}
                  >
                    <Paperclip size={18} />
                  </button>
                )}
                <textarea
                  ref={textareaRef}
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      sendMessage(event.currentTarget.value);
                    }
                  }}
                  rows={2}
                  maxLength={4000}
                  aria-label={
                    mode === "research"
                      ? modeText("research").inputLabel
                      : mode === "mes"
                        ? modeText("mes").inputLabel
                        : modeText("mkac").inputLabel
                  }
                  placeholder={
                    mode === "research"
                      ? researchReady
                        ? modeText("research").placeholderReady
                        : modeText("research").placeholderEmpty
                      : mode === "mes"
                        ? modeText("mes").placeholder
                        : modeText("mkac").placeholder
                  }
                  disabled={busy || !researchReady}
                />
                <div className="composer-footer">
                  <div className="composer-context">
                  <span className={`model-dot ${health === "online" ? "ready" : "offline"}`} />
                  <span>
                    {health === "online" ? t("common.ready") : t("common.notReady")}
                    </span>
                  </div>
                  <div className="composer-actions">
                    <span className="char-count">{question.length}/4000</span>
                    <button
                      className={busy ? "send-button stopping" : "send-button"}
                      type={busy ? "button" : "submit"}
                      title={busy ? t("common.stop") : t("common.send")}
                      onClick={busy ? stopGeneration : undefined}
                      disabled={busy ? false : !canAsk}
                    >
                      {busy ? (
                        <Square size={16} fill="currentColor" />
                      ) : (
                        <Send size={18} />
                      )}
                    </button>
                  </div>
                </div>
              </form>
            )}
          </section>

          <aside
            id="source-panel"
            className="source-panel"
            aria-label={t("common.sources")}
          >
            <div className="source-header">
              <span>{t("common.sources")}</span>
              <span>{latestSources.length}</span>
            </div>
            <div className="source-list">
              {latestSources.map((source, index) => (
                <article
                  className={`source-item ${source.url ? "" : "clickable"}`}
                  key={`${source.file}-${source.page}-${index}`}
                  role={source.url ? undefined : "button"}
                  tabIndex={source.url ? undefined : 0}
                  onClick={
                    source.url ? undefined : () => openSourcePreview(source, mode)
                  }
                  onKeyDown={(event) => {
                    if (source.url) return;
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      openSourcePreview(source, mode);
                    }
                  }}
                >
                  <div className="source-title">
                    {source.url ? <Globe2 size={16} /> : <FileText size={16} />}
                    {source.url ? (
                      <a href={source.url} target="_blank" rel="noreferrer">
                        <strong>{source.file}</strong>
                      </a>
                    ) : (
                      <strong>{source.file}</strong>
                    )}
                  </div>
                  <div className="source-meta">
                    <span>
                      {source.url
                        ? t("common.webSource")
                        : t("common.page", { page: source.page || "?" })}
                    </span>
                    <span>
                      {t("common.similarity", {
                        score: Math.round((source.score || 0) * 100),
                      })}
                    </span>
                  </div>
                  {!source.url && (
                    <div className="source-preview-state">
                      {source.has_page_preview
                        ? t("common.clickToViewPage")
                        : t("common.clickToViewSnippet")}
                    </div>
                  )}
                  <p>{source.preview}</p>
                </article>
              ))}
              {latestSources.length === 0 && (
                <div className="empty-sources">
                  <Search size={22} />
                  <span>{t("common.noSources")}</span>
                </div>
              )}
            </div>
          </aside>
        </div>
      </main>
      {sourcePreview && (
        <div
          className="source-preview-backdrop"
          role="presentation"
          onClick={() => setSourcePreview(null)}
        >
          <section
            className="source-preview-modal"
            role="dialog"
            aria-modal="true"
            aria-label={t("common.previewDialog")}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="source-preview-header">
              <div>
                <span>{t("common.sources")}</span>
                <strong title={sourcePreview.source.file}>
                  {sourcePreview.source.file}
                </strong>
              </div>
              <button
                className="icon-button"
                type="button"
                title={t("common.closePreview")}
                onClick={() => setSourcePreview(null)}
              >
                <X size={18} />
              </button>
            </div>
            <div className="source-preview-meta">
              <span>{t("common.page", { page: sourcePreview.source.page || "?" })}</span>
              <span>
                {t("common.similarity", {
                  score: Math.round((sourcePreview.source.score || 0) * 100),
                })}
              </span>
            </div>
            {sourcePreview.source.has_page_preview && !sourcePreview.imageFailed ? (
              <div className="source-preview-image-wrap">
                <img
                  src={sourcePreviewUrl(sourcePreview.source, sourcePreview.mode)}
                  alt={`${sourcePreview.source.file} ${t("common.page", {
                    page: sourcePreview.source.page || "?",
                  })}`}
                  onError={() =>
                    setSourcePreview((current) =>
                      current ? { ...current, imageFailed: true } : current,
                    )
                  }
                />
              </div>
            ) : (
              <div className="source-preview-placeholder">
                <FileText size={28} />
                <span>{t("common.noPreviewImage")}</span>
              </div>
            )}
            <div className="source-preview-text">
              <strong>{t("common.snippet")}</strong>
              <p>{sourcePreview.source.preview}</p>
            </div>
          </section>
        </div>
      )}

      {confirmClearOpen && (
        <div
          className="confirm-backdrop"
          role="presentation"
          onClick={() => setConfirmClearOpen(false)}
        >
          <section
            className="confirm-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="confirm-clear-title"
            aria-describedby="confirm-clear-body"
            onClick={(event) => event.stopPropagation()}
          >
            <h2 id="confirm-clear-title">{t("common.clearConfirmTitle")}</h2>
            <p id="confirm-clear-body">{t("common.clearConfirmBody")}</p>
            <div className="confirm-actions">
              <button
                type="button"
                className="confirm-cancel"
                onClick={() => setConfirmClearOpen(false)}
                autoFocus
              >
                {t("common.cancel")}
              </button>
              <button
                type="button"
                className="confirm-danger"
                onClick={clearConversation}
              >
                {t("common.clearConfirmAction")}
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
