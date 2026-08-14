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
  Boxes,
  Check,
  CheckCircle2,
  ChevronDown,
  Copy,
  Database,
  Download,
  FileCode,
  FileImage,
  FileSpreadsheet,
  FileText,
  FileUp,
  FlaskConical,
  Globe2,
  Layers3,
  Loader2,
  LogOut,
  Mail,
  Menu,
  PanelRightClose,
  PanelRightOpen,
  Paperclip,
  Presentation,
  RefreshCcw,
  Search,
  Send,
  Server,
  ShieldCheck,
  Sparkles,
  TableProperties,
  Square,
  Sun,
  Moon,
  Monitor,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import { EmployeeLogin } from "./components/EmployeeLogin";
import { SourcePreviewDialog } from "./components/SourcePreviewDialog";
import { ResearchSidebar } from "./components/ResearchSidebar";
import { MessageList } from "./components/MessageList";
import { ChatInput, pushStoredPromptHistory } from "./components/ChatInput";
import { quickAnswerRequestOptions } from "./quick-answer-request";
import {
  getModeTabScrollLeft,
  shouldClearEmployeeAuth,
  visibleModeKeys,
} from "./mode-tabs";
import "./styles.css";

// Icon theo loại file để người dùng phân biệt nhanh PDF/Word/Excel/ảnh trong
// danh sách tài liệu (trước đây mọi file dùng chung một icon FileText).
const FILE_TYPE_ICONS = {
  pdf: { Icon: FileText, className: "ext-pdf" },
  doc: { Icon: FileText, className: "ext-doc" },
  docx: { Icon: FileText, className: "ext-doc" },
  xls: { Icon: FileSpreadsheet, className: "ext-xls" },
  xlsx: { Icon: FileSpreadsheet, className: "ext-xls" },
  ppt: { Icon: Presentation, className: "ext-ppt" },
  pptx: { Icon: Presentation, className: "ext-ppt" },
  png: { Icon: FileImage, className: "ext-img" },
  jpg: { Icon: FileImage, className: "ext-img" },
  jpeg: { Icon: FileImage, className: "ext-img" },
  html: { Icon: FileCode, className: "ext-html" },
  htm: { Icon: FileCode, className: "ext-html" },
};

function AgentTimeline({ timeline, language }) {
  if (!timeline?.steps?.length) return null;
  const complete = timeline.status === "done";
  const cancelled = timeline.status === "cancelled";
  const isWmsVerification = timeline.workflow === "wms_verification";
  const labels = language === "ja"
    ? {
        title: isWmsVerification ? "WMS検証ステップ" : "Agentの実行手順",
        completed: "完了",
        running: "実行中",
        queued: "待機中",
        cancelled: "中止",
      }
    : {
        title: isWmsVerification ? "Các bước kiểm chứng WMS" : "Các bước Report Agent",
        completed: "Hoàn tất",
        running: "Đang chạy",
        queued: "Chờ thực hiện",
        cancelled: "Đã dừng",
      };

  return (
    <details
      className={`agent-timeline${isWmsVerification ? " wms-verification" : ""}`}
      open={!complete}
      aria-live="off"
    >
      <summary>
        <span className="agent-timeline-heading">
          <Sparkles size={15} aria-hidden="true" />
          {labels.title}
        </span>
        <span
          className={`agent-timeline-state ${
            complete ? "done" : cancelled ? "cancelled" : "running"
          }`}
        >
          {complete ? labels.completed : cancelled ? labels.cancelled : labels.running}
        </span>
      </summary>
      <ol>
        {timeline.steps.map((step) => {
          const state = step.status || "queued";
          return (
            <li className={`agent-step ${state}`} key={step.id}>
              <span className="agent-step-icon" aria-hidden="true">
                {state === "running" ? (
                  <Loader2 className="spin" size={14} />
                ) : state === "done" ? (
                  <Check size={14} />
                ) : state === "error" ? (
                  <AlertCircle size={14} />
                ) : state === "cancelled" ? (
                  <X size={14} />
                ) : (
                  <span className="agent-step-dot" />
                )}
              </span>
              <span className="agent-step-copy">
                <strong>{step.title}</strong>
                {step.summary && <small>{step.summary}</small>}
              </span>
              <span className="sr-only">
                {state === "done"
                  ? labels.completed
                  : state === "running"
                    ? labels.running
                    : state === "cancelled"
                      ? labels.cancelled
                      : labels.queued}
              </span>
            </li>
          );
        })}
      </ol>
    </details>
  );
}

function formatReportValue(value, language) {
  if (typeof value === "number") {
    return new Intl.NumberFormat(language === "ja" ? "ja-JP" : "vi-VN", {
      maximumFractionDigits: 2,
    }).format(value);
  }
  return value ?? "—";
}

// Các phần của thẻ báo cáo, theo đúng thứ tự người đọc quét mắt. Hook bên dưới
// mở dần từng phần để dữ liệu deterministic không bật ra cùng một khung hình.
const REPORT_SECTION_ORDER = [
  "header",
  "kpis",
  "charts",
  "matrices",
  "governance",
  "observations",
  "tables",
  "limitations",
  "actions",
];
const REPORT_SECTION_REVEAL_MS = 260;

function useStagedReportReveal(artifactId, enabled, onStageChange) {
  const [stage, setStage] = useState(enabled ? 0 : REPORT_SECTION_ORDER.length);

  useEffect(() => {
    if (!enabled) {
      setStage(REPORT_SECTION_ORDER.length);
      return undefined;
    }
    if (
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
    ) {
      setStage(REPORT_SECTION_ORDER.length);
      return undefined;
    }
    setStage(0);
    const timers = REPORT_SECTION_ORDER.map((_, index) =>
      window.setTimeout(() => {
        setStage((current) => {
          const next = Math.max(current, index + 1);
          onStageChange?.(next);
          return next;
        });
      }, index * REPORT_SECTION_REVEAL_MS),
    );
    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [artifactId, enabled, onStageChange]);

  return (section) => stage > REPORT_SECTION_ORDER.indexOf(section);
}

function ReportArtifactCard({ artifact, language, onEmail, onRevealStage, animate = false }) {
  const revealed = useStagedReportReveal(artifact?.id, Boolean(artifact) && animate, onRevealStage);
  if (!artifact) return null;
  const isWms = artifact.report_type === "wms_executive_report";
  const isHr = artifact.report_type === "hr_executive_report";
  const labels = language === "ja"
    ? {
        badge: isWms ? "WMSレポート" : isHr ? "HRレポート" : "MESレポート",
        governance: "ガバナンス上の注意",
        observations: "数値からの所見",
        limitations: "データの制限",
        details: "データ表を表示",
        download: "HTMLをダウンロード",
        emailAction: "メールで共有",
      }
    : {
        badge: isWms ? "Báo cáo WMS" : isHr ? "Báo cáo HR" : "Báo cáo MES",
        governance: "Quy tắc Governance & Data Contract",
        observations: "Nhận xét từ số liệu",
        limitations: "Giới hạn dữ liệu",
        details: "Xem các bảng số liệu",
        download: "Tải báo cáo HTML",
        emailAction: "Gửi qua email",
      };
  const charts = Array.isArray(artifact.charts)
    ? artifact.charts.filter((chart) => Array.isArray(chart.rows) && chart.rows.length > 0)
    : [];
  const matrices = Array.isArray(artifact.matrices)
    ? artifact.matrices.filter((matrix) => Array.isArray(matrix.rows) && matrix.rows.length > 0)
    : [];

  return (
    <section className="report-artifact" aria-labelledby={`report-title-${artifact.id}`}>
      {revealed("header") && (
      <header className="report-artifact-header">
        <span className="report-artifact-icon" aria-hidden="true">
          <TableProperties size={18} />
        </span>
        <div>
          <span className="report-artifact-badge">{labels.badge}</span>
          <h3 id={`report-title-${artifact.id}`}>{artifact.title}</h3>
          <p>{artifact.generated_at} · {artifact.period_label}</p>
        </div>
      </header>
      )}

      {revealed("kpis") && artifact.kpis?.length > 0 && (
        <div className="report-kpis" aria-label={labels.badge}>
          {artifact.kpis.map((item) => (
            <div className="report-kpi" key={item.key}>
              <strong>{formatReportValue(item.value, language)}</strong>
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      )}

      {revealed("charts") && charts.length > 0 && (
        <div className="report-charts" aria-label="Report charts">
          {charts.map((chart) => {
            const labelKey = chart.label_key || "label";
            const valueKey = chart.value_key || "value";
            const maxValue = Math.max(...chart.rows.map((row) => Number(row[valueKey]) || 0), 1);
            return (
              <figure className="report-chart" key={chart.id || chart.title}>
                <figcaption>{chart.title}</figcaption>
                <div className="report-bars">
                  {chart.rows.map((row, index) => {
                    const value = Number(row[valueKey]) || 0;
                    const width = value > 0
                      ? `${Math.max(2, Math.round((value * 100) / maxValue))}%`
                      : "0%";
                    const label = row[labelKey] || "—";
                    return (
                      <div className="report-bar-row" key={`${label}-${index}`}>
                        <span className="report-bar-label" title={label}>{label}</span>
                        <span
                          className="report-bar-track"
                          role="img"
                          aria-label={`${label}: ${formatReportValue(value, language)}`}
                        >
                          <span className="report-bar-fill" style={{ width }} />
                        </span>
                        <strong>{formatReportValue(value, language)}</strong>
                      </div>
                    );
                  })}
                </div>
              </figure>
            );
          })}
        </div>
      )}

      {revealed("matrices") && matrices.length > 0 && (
        <details className="report-matrices" open>
          <summary>{language === "ja" ? "マトリクス" : "Ma trận điều hành"}</summary>
          {matrices.map((matrix) => {
            const columns = Array.isArray(matrix.columns) ? matrix.columns : [];
            const maxValue = Number(matrix.max_value) || 0;
            const useHeatmap = matrix.heatmap !== false;
            return (
              <div className="report-table-block" key={matrix.id || matrix.title}>
                <h4>{matrix.title}</h4>
                <div className="report-table-scroll">
                  <table>
                    <caption>{matrix.title}</caption>
                    <thead>
                      <tr>
                        <th scope="col">{matrix.row_label || ""}</th>
                        {columns.map((column) => <th scope="col" key={column}>{column}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {matrix.rows.map((row, rowIndex) => {
                        const values = Array.isArray(row.values) ? row.values : [];
                        return (
                          <tr key={rowIndex}>
                            <th scope="row">{row.label}</th>
                            {columns.map((column, valueIndex) => {
                              const value = values[valueIndex];
                              const ratio = useHeatmap && maxValue && typeof value === "number"
                                ? Math.min(1, Math.max(0.12, value / maxValue))
                                : 0;
                              const className = ratio ? "matrix-heat-cell" : undefined;
                              const style = ratio ? { "--heat": ratio } : undefined;
                              return <td key={column} className={className} style={style}>{formatReportValue(value, language)}</td>;
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}
        </details>
      )}

      {revealed("governance") && artifact.governance?.length > 0 && (
        <div className="report-insights report-governance">
          <strong>{labels.governance}</strong>
          <ul>{artifact.governance.map((note, index) => <li key={index}>{note}</li>)}</ul>
        </div>
      )}

      {revealed("observations") && artifact.observations?.length > 0 && (
        <div className="report-insights">
          <strong>{labels.observations}</strong>
          <ul>{artifact.observations.map((note, index) => <li key={index}>{note}</li>)}</ul>
        </div>
      )}

      {revealed("tables") && artifact.sections?.some((section) => section.rows?.length > 0) && (
        <details className="report-tables">
          <summary>{labels.details}</summary>
          {artifact.sections.map((section) => section.rows?.length > 0 && (
            <div className="report-table-block" key={section.id}>
              <h4>{section.title}</h4>
              <div className="report-table-scroll">
                <table>
                  <caption>{section.title}</caption>
                  <thead>
                    <tr>{section.columns.map((column) => <th scope="col" key={column}>{column}</th>)}</tr>
                  </thead>
                  <tbody>
                    {section.rows.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {section.columns.map((column) => (
                          <td key={column}>{formatReportValue(row[column], language)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </details>
      )}

      {revealed("limitations") && artifact.limitations?.length > 0 && (
        <details className="report-limitations">
          <summary>{labels.limitations}</summary>
          <ul>{artifact.limitations.map((note, index) => <li key={index}>{note}</li>)}</ul>
        </details>
      )}

      {revealed("actions") && (
      <div className="report-actions">
        {artifact.download_url && (
          <a className="report-download" href={artifact.download_url} download>
            <Download size={16} aria-hidden="true" />
            {labels.download}
          </a>
        )}
        <button
          className="report-email-action"
          type="button"
          onClick={() => onEmail?.(
            language === "ja"
              ? "このレポートを メールで送信: "
              : "Gửi báo cáo này cho ",
          )}
        >
          <Mail size={16} aria-hidden="true" />
          {labels.emailAction}
        </button>
      </div>
      )}
    </section>
  );
}

function FileTypeIcon({ filename, size = 17 }) {
  const ext = (String(filename).split(".").pop() || "").toLowerCase();
  const { Icon, className } =
    FILE_TYPE_ICONS[ext] || { Icon: FileText, className: "ext-generic" };
  return <Icon size={size} className={`file-type-icon ${className}`} aria-hidden="true" />;
}

const QUICK_PROMPTS = {
  vi: {
    mkac: [
      "Meiko Automation gồm những phòng ban nào?",
      "Quy định làm thêm giờ ở MKAC như thế nào?",
      "Các sản phẩm chính của MKAC là gì?",
    ],
    mes: [
      "Mã Lot nào có số lượng lỗi nhiều nhất?",
      "Trong Lot lỗi nhiều nhất, loại lỗi nào phổ biến nhất?",
      "Hãy liệt kê 10 mã hàng có số lượng lỗi nhiều nhất",
    ],
    wms: [
      "Kho công đoạn WMS hiện có bao nhiêu mã vật tư?",
      "Mã vật tư nào có số lượng tồn kho nhiều nhất?",
      "Tình hình tồn kho WMS hiện tại thế nào?",
    ],
    research: [
      "Lập báo cáo nghiên cứu tổng hợp từ các tài liệu",
      "So sánh các quan điểm và chỉ ra điểm mâu thuẫn",
      "Đề xuất các câu hỏi nghiên cứu tiếp theo",
    ],
  },
  ja: {
    mkac: [
      "Meiko Automationにはどの部署がありますか？",
      "MKACの残業規定はどうなっていますか？",
      "MKACの主な製品は何ですか？",
    ],
    mes: [
      "エラー数が最も多いLotはどれですか？",
      "エラー数が最も多いLotで、最も多いエラー種類は何ですか？",
      "総エラー数が多い製品を上位10件挙げてください",
    ],
    wms: [
      "WMS工程倉庫には現在いくつの資材コードがありますか？",
      "在庫数量が最も多い資材コードは何ですか？",
      "現在のWMS在庫状況はどうなっていますか？",
    ],
    research: [
      "資料から総合的な調査レポートを作成してください",
      "各見解を比較し、矛盾点を示してください",
      "次に調査すべき質問を提案してください",
    ],
  },
};

const ACTIVE_MODE_KEYS = ["mkac", "mes", "wms", "research"];

const MODE_OPTIONS = {
  mkac: {
    icon: Database,
  },
  mes: {
    icon: Activity,
  },
  wms: {
    icon: Boxes,
  },
  research: {
    icon: FlaskConical,
  },
};

const MODEL_ACCENTS = {
  auto: "accent-auto",
  local: "accent-local",
  openai: "accent-openai",
  grok: "accent-grok",
};

const LEGACY_MODE_SESSION_STORAGE_KEYS = {
  mkac: "meibook-session-mkac",
  mes: "meibook-session-mes",
  wms: "meibook-session-wms",
  research: "meibook-session-research",
};
const SESSION_STORAGE_PREFIX = "meibook-session";
const LEGACY_SESSION_STORAGE_KEY = "meibook-session";
const RESEARCH_TOPIC_STORAGE_PREFIX = "meibook-research-topic";
const RESEARCH_SCOPE_STORAGE_PREFIX = "meibook-research-scope";
const LEGACY_RESEARCH_TOPIC_STORAGE_KEY = "meibook-research-topic";
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
        shortLabel: "Nhân sự",
        title: "Hỏi đáp hành chính nhân sự MKAC",
        empty: "Tra cứu quy định hành chính, thông tin nội bộ và nhân sự MKAC.",
        metric: "Tài liệu MKAC",
        authHint: "Nhập mã nhân viên để truy cập chế độ hỏi đáp nội bộ MKAC.",
        inputLabel: "Câu hỏi hành chính nhân sự MKAC",
        placeholder: "Hỏi về hành chính, quy định hoặc nhân sự MKAC...",
      },
      mes: {
        label: "Quản lý MES",
        shortLabel: "Sản xuất",
        title: "Quản lý MES",
        empty: "Tra cứu Lot, mã hàng và lỗi sản xuất MES.",
        metric: "Dữ liệu MES",
        authHint: "Nhập mã nhân viên để truy cập dữ liệu MES.",
        inputLabel: "Câu hỏi về MES",
        placeholder: "Hỏi về Lot, mã hàng hoặc lỗi sản xuất MES...",
        unavailable: "MES snapshot chưa sẵn sàng",
      },
      wms: {
        label: "Quản lý Kho (WMS)",
        shortLabel: "Kho",
        title: "Quản lý Kho WMS MKHC",
        empty: "Tra cứu tồn kho nguyên vật liệu tại kho công đoạn WMS.",
        metric: "Dữ liệu WMS",
        itemCount: "{count} mã vật tư",
        processCount: "{count} mã công đoạn",
        authHint: "Nhập mã nhân viên để truy cập dữ liệu WMS.",
        inputLabel: "Câu hỏi về WMS",
        placeholder: "Hỏi về tồn kho vật tư, công đoạn WMS...",
        unavailable: "WMS snapshot chưa sẵn sàng",
      },
      research: {
        label: "Nghiên cứu tài liệu",
        shortLabel: "Nghiên cứu",
        title: "Nghiên cứu tài liệu nội bộ",
        emptyReady: "Sẵn sàng nghiên cứu trong nhóm tài liệu đã chọn.",
        empty: "Chọn nhóm tài liệu bạn muốn tra cứu. Câu trả lời sẽ chỉ dựa trên các tài liệu trong nhóm đã chọn.",
        metric: "Tài liệu",
        inputLabel: "Câu hỏi nghiên cứu",
        placeholderReady: "Nhập câu hỏi nghiên cứu...",
        placeholderEmpty: "Chọn nhóm tài liệu trước khi đặt câu hỏi...",
        lockedModel: "Local Model",
        chooseTopic: "Chọn nhóm tài liệu",
        changeTopic: "Đổi nhóm tài liệu",
        scopeLabel: "Phạm vi: {topic}",
        allTopics: "Tất cả tài liệu",
        allTopicsDescription: "Tìm kiếm trên toàn bộ kho tài liệu DocJP, không giới hạn nhóm.",
        topicNotReady: "Nhóm này chưa sẵn sàng để tra cứu.",
        documentsCount: "{count} tài liệu",
        documentsSummary: "{count} tài liệu · {topics} nhóm",
        allDocsNote: "Hệ thống sẽ tìm trong toàn bộ {count} tài liệu của nhóm này.",
        browseDocuments: "Xem danh sách tài liệu",
        searchFiles: "Tìm tài liệu theo tên...",
        noMatchingFiles: "Không tìm thấy tài liệu phù hợp",
        chooseTopicHint: "Chọn nhóm tài liệu ở màn hình chính để bắt đầu tra cứu.",
        sourceTopic: "Kho tài liệu có sẵn",
        sourceUpload: "Tài liệu tải lên",
        sourceTopicDescription: "Tra cứu theo 4 nhóm tài liệu nội bộ đã sẵn sàng.",
        sourceUploadDescription: "Tải tài liệu riêng và chỉ nghiên cứu trong phiên hiện tại.",
        uploadEmpty: "Chọn và xử lý ít nhất một tài liệu để bắt đầu nghiên cứu.",
        uploadReady: "Sẵn sàng nghiên cứu các tài liệu đã tải lên.",
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
      wms: "Phiên WMS mới",
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
      deleteFileConfirmTitle: "Xóa tài liệu?",
      deleteFileConfirmBody:
        "Tài liệu \"{name}\" sẽ bị xóa khỏi phiên nghiên cứu này. Hành động này không thể hoàn tác.",
      deleteFileConfirmAction: "Xóa tài liệu",
      indexDocument: "Xử lý tài liệu",
      indexing: "Đang xử lý {done}/{total}",
      processing: "Đang xử lý",
      indexedSummary: "Đã xử lý {files} tệp, sẵn sàng hỏi đáp",
      noDocuments: "Chưa có tài liệu",
      onlineMachine: "Hệ thống đang hoạt động",
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
      guestWelcome: "Chào mừng đến với hệ thống Meibook.",
      languageConverting: "Đang chuyển đổi ngôn ngữ...",
      continue: "Tiếp tục",
      verifying: "Đang kiểm tra",
      chooseToStart: "Chọn tài liệu để bắt đầu",
      uploadRecommendationTitle: "Lưu ý trước khi tải tài liệu",
      uploadRecommendation:
        "Để model nội bộ phản hồi ổn định, nên tải tối đa 3 tài liệu mỗi phiên và ưu tiên mỗi tài liệu dưới 10 trang.",
      uploadRecommendationAction: "Tiếp tục chọn tài liệu",
      tryDemo: "Thử nghiệm với tài liệu mẫu",
      model: "Model",
      status: "Trạng thái",
      stopped: "Đã dừng",
      noResult: "Không có kết quả",
      webSearch: "Tìm kiếm web",
      mesData: "Dữ liệu MES",
      mesSnapshot: "MES snapshot",
      wmsSnapshot: "Tồn kho WMS snapshot",
      wmsHealth: {
        READY: "WMS sẵn sàng",
        INCOMPATIBLE: "WMS không tương thích",
        UNAVAILABLE: "WMS chưa khả dụng",
        QUERY_ERROR: "WMS lỗi truy vấn",
        DISABLED: "WMS đang tắt",
      },
      wmsDomain: {
        CURRENT_BALANCE: "Tồn hiện tại",
        LEGACY_ARCHIVE: "Lưu trữ lịch sử",
        RAW_TRANSACTION_AUDIT: "Nhật ký giao dịch thô",
        SUPPRESSED: "Nghiệp vụ bị khóa",
        fallback: "WMS",
      },
      wmsStatus: {
        AVAILABLE: "Khả dụng",
        PARTIAL: "Một phần",
        SUPPRESSED: "Bị khóa",
        fallback: "Chưa xác định",
      },
      wmsFreshnessValue: {
        DERIVED_UNVERIFIED: "Được suy ra, chưa xác minh",
        UNAVAILABLE: "Không khả dụng",
        fallback: "Chưa xác định",
      },
      wmsTimezoneValue: {
        unverified: "Chưa xác minh",
        fallback: "Chưa xác định",
      },
      wmsEpochValue: {
        CURRENT_POST_2026_01_15: "Current sau 15/01/2026",
        LEGACY_PRE_2026_01_15: "Legacy trước 15/01/2026",
        RAW_SOURCE_AUDIT: "Audit nguồn thô",
        fallback: "Chưa xác định",
      },
      wmsReason: {
        WMS_DISABLED: "WMS đang tắt",
        WMS_SNAPSHOT_UNAVAILABLE: "Snapshot WMS chưa khả dụng",
        WMS_SNAPSHOT_INCOMPATIBLE: "Snapshot WMS không tương thích",
        WMS_SNAPSHOT_QUERY_ERROR: "Không thể truy vấn snapshot WMS",
        WMS_SOURCE_AS_OF_UNCONFIRMED: "Mốc dữ liệu nguồn chưa xác nhận",
        UOM_MASTER_UNAVAILABLE: "Chưa có master đơn vị tính",
        MIN_STOCK_CONTRACT_UNVERIFIED: "Chưa xác minh contract tồn tối thiểu",
        EXPIRY_SOURCE_UNAVAILABLE: "Chưa có nguồn hạn dùng",
        WINDOW_TIME_SOURCE_UNAVAILABLE: "Chưa có nguồn window-time",
        SNAPSHOT_HISTORY_NOT_COMPARABLE: "Lịch sử snapshot không thể so sánh",
        PRODUCTION_WIP_SOURCE_UNAVAILABLE: "Chưa có nguồn WIP sản xuất",
        BOTTLENECK_DEFINITION_UNAVAILABLE: "Chưa có định nghĩa bottleneck",
        SNAPSHOT_DELTA_NOT_IN_SCOPE: "Không hỗ trợ delta snapshot",
        TRANSACTION_SEMANTICS_NOT_IN_SCOPE: "Chưa xác minh semantics giao dịch",
        CURRENT_GRAIN_HAS_NO_MEANINGFUL_LOT: "Current không có lot nghiệp vụ",
        CROSS_ERA_KEYS_NOT_COMPARABLE: "Không thể so sánh khóa khác thời kỳ",
        COMPLETED_MOVEMENTS_NOT_VERIFIED: "Chưa xác minh giao dịch hoàn thành",
        DATASET_NOT_OBSERVED_IN_EXPORT: "Dataset chưa được quan sát trong export",
        CURRENT_BALANCE_GRAIN_DUPLICATE: "Current balance trùng grain",
        QUANTITY_EVIDENCE_INCOMPLETE: "Bằng chứng quantity chưa đầy đủ",
        RAW_TRANSACTION_HEADER_NOT_OBSERVED: "Chưa quan sát header giao dịch thô",
        SQL_AGENT_ANSWER_UNVERIFIED: "SQL do LLM lập kế hoạch, chưa kiểm chứng deterministic",
        WMS_METADATA_VALIDATION_FAILED: "Metadata WMS không hợp lệ",
        fallback: "Lý do kỹ thuật chưa xác định",
      },
      wmsAsOf: "Mốc dữ liệu nguồn",
      wmsImportedAt: "Thời điểm nhập",
      wmsFreshnessState: "Trạng thái freshness",
      wmsBasis: "Cơ sở freshness",
      wmsTimezone: "Múi giờ nguồn",
      wmsEpoch: "Thời kỳ ngữ nghĩa",
      wmsEvidence: "Freshness theo miền",
      wmsReasons: "Mã lý do",
      wmsPagination: "Phân trang",
      wmsPerPage: "mỗi trang",
      wmsHasMore: "Còn trang sau",
      wmsNoMore: "Đã hết dữ liệu",
      wmsPage: "Trang {page} · {total} bản ghi",
      mesReport: "Báo cáo MES",
      wmsReport: "Báo cáo WMS",
      hrReport: "Báo cáo HR",
      mesReportUnsupported: "Ngoài mẫu Report Agent",
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
      demoNotIndexed: "Tài liệu mẫu chưa sẵn sàng. Vui lòng liên hệ quản trị viên.",
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
      groundedBadge: "Trả lời dựa trên tài liệu",
    },
    answerScope: {
      web: "Tổng hợp từ nguồn web",
      mes: "Dựa trên dữ liệu MES trực tiếp",
      mes_database: "Dựa trên MES snapshot cục bộ",
      wms_database: "Dựa trên tồn nguyên vật liệu tại kho công đoạn WMS MKHC",
      mes_report: "Báo cáo được tổng hợp từ MES snapshot",
      wms_executive_report: "Báo cáo được tổng hợp từ current balance WMS",
      hr_executive_report: "Báo cáo được tổng hợp từ danh bạ nhân sự MKAC",
      mes_report_unsupported: "Yêu cầu chưa thuộc các mẫu báo cáo đã được kiểm chứng",
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
        label: "MES生産管理",
        shortLabel: "生産",
        title: "MES生産管理",
        empty: "MESのLot・品番・生産エラーを検索します。",
        metric: "MESデータ",
        authHint: "MESデータにアクセスするには社員番号を入力してください。",
        inputLabel: "MESに関する質問",
        placeholder: "Lot、品番、生産エラーについて質問してください...",
        unavailable: "MESスナップショットはまだ利用できません",
      },
      wms: {
        label: "在庫管理 (WMS)",
        shortLabel: "在庫",
        title: "MKHC WMS工程倉庫",
        empty: "WMS工程倉庫の資材在庫を検索します。",
        metric: "WMSデータ",
        itemCount: "資材コード {count}件",
        processCount: "工程コード {count}件",
        authHint: "WMS在庫データにアクセスするには社員番号を入力してください。",
        inputLabel: "WMSに関する質問",
        placeholder: "WMS在庫、工程、資材について質問...",
        unavailable: "WMSスナップショットはまだ利用できません",
      },
      research: {
        label: "資料調査",
        shortLabel: "調査",
        title: "社内資料調査",
        emptyReady: "選択した資料カテゴリ内で調査できます。",
        empty: "調べたい資料カテゴリを選択してください。回答は選択したカテゴリ内の資料のみに基づきます。",
        metric: "資料",
        inputLabel: "調査質問",
        placeholderReady: "調査質問を入力してください...",
        placeholderEmpty: "質問する前に資料カテゴリを選択してください...",
        lockedModel: "ローカルモデル",
        chooseTopic: "資料カテゴリを選択",
        changeTopic: "カテゴリを変更",
        scopeLabel: "調査範囲: {topic}",
        allTopics: "すべての資料",
        allTopicsDescription: "カテゴリを限定せず、DocJP資料全体から検索します。",
        topicNotReady: "このカテゴリはまだ利用できません。",
        documentsCount: "{count}件の資料",
        documentsSummary: "{count}件の資料 · {topics}カテゴリ",
        allDocsNote: "このカテゴリの{count}件の資料\nすべてから検索します。",
        browseDocuments: "資料一覧を表示",
        searchFiles: "資料名で検索...",
        noMatchingFiles: "一致する資料がありません",
        chooseTopicHint: "メイン画面で資料カテゴリを選択して調査を開始してください。",
        sourceTopic: "登録済み資料",
        sourceUpload: "アップロード資料",
        sourceTopicDescription: "4つの社内資料カテゴリから検索します。",
        sourceUploadDescription: "資料をアップロードし、現在のセッション内だけで調査します。",
        uploadEmpty: "調査を開始するには、少なくとも1件の資料を選択して処理してください。",
        uploadReady: "アップロードした資料を調査できます。",
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
      wms: "新しいWMSセッション",
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
      deleteFileConfirmTitle: "資料を削除しますか？",
      deleteFileConfirmBody:
        "資料「{name}」はこの調査セッションから削除されます。この操作は取り消せません。",
      deleteFileConfirmAction: "資料を削除",
      indexDocument: "資料を処理",
      indexing: "処理中 {done}/{total}",
      processing: "処理中",
      indexedSummary: "{files}ファイルの処理が完了しました",
      noDocuments: "資料はまだありません",
      onlineMachine: "システム稼働中",
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
      uploadRecommendationTitle: "資料アップロード前の注意",
      uploadRecommendation:
        "ローカルモデルで安定した回答を得るため、1セッションにつき資料は3件まで、1件10ページ未満を目安にしてください。",
      uploadRecommendationAction: "資料を選択する",
      tryDemo: "サンプル資料で試す",
      model: "モデル",
      status: "状態",
      stopped: "停止済み",
      noResult: "結果なし",
      webSearch: "Web検索",
      mesData: "MESデータ",
      mesSnapshot: "MESスナップショット",
      wmsSnapshot: "WMS在庫スナップショット",
      wmsHealth: {
        READY: "WMS利用可能",
        INCOMPATIBLE: "WMS互換性なし",
        UNAVAILABLE: "WMS未利用",
        QUERY_ERROR: "WMS照会エラー",
        DISABLED: "WMS無効",
      },
      wmsDomain: {
        CURRENT_BALANCE: "現行残高",
        LEGACY_ARCHIVE: "レガシーアーカイブ",
        RAW_TRANSACTION_AUDIT: "生トランザクション監査",
        SUPPRESSED: "抑止された機能",
        fallback: "WMS",
      },
      wmsStatus: {
        AVAILABLE: "利用可能",
        PARTIAL: "一部利用",
        SUPPRESSED: "抑止中",
        fallback: "未確認",
      },
      wmsFreshnessValue: {
        DERIVED_UNVERIFIED: "算出済み・未検証",
        UNAVAILABLE: "利用不可",
        fallback: "未確認",
      },
      wmsTimezoneValue: {
        unverified: "未検証",
        fallback: "未確認",
      },
      wmsEpochValue: {
        CURRENT_POST_2026_01_15: "2026年1月15日以降の現行期間",
        LEGACY_PRE_2026_01_15: "2026年1月15日以前の旧期間",
        RAW_SOURCE_AUDIT: "生ソース監査期間",
        fallback: "未確認",
      },
      wmsReason: {
        WMS_DISABLED: "WMSは無効です",
        WMS_SNAPSHOT_UNAVAILABLE: "WMSスナップショットは利用できません",
        WMS_SNAPSHOT_INCOMPATIBLE: "WMSスナップショットに互換性がありません",
        WMS_SNAPSHOT_QUERY_ERROR: "WMSスナップショットを照会できません",
        WMS_SOURCE_AS_OF_UNCONFIRMED: "データ基準時点は未確認です",
        UOM_MASTER_UNAVAILABLE: "単位マスターがありません",
        MIN_STOCK_CONTRACT_UNVERIFIED: "最低在庫の契約は未検証です",
        EXPIRY_SOURCE_UNAVAILABLE: "期限データソースがありません",
        WINDOW_TIME_SOURCE_UNAVAILABLE: "ウィンドウタイムのデータソースがありません",
        SNAPSHOT_HISTORY_NOT_COMPARABLE: "スナップショット履歴は比較できません",
        PRODUCTION_WIP_SOURCE_UNAVAILABLE: "生産WIPデータソースがありません",
        BOTTLENECK_DEFINITION_UNAVAILABLE: "ボトルネック定義がありません",
        SNAPSHOT_DELTA_NOT_IN_SCOPE: "スナップショット差分は対象外です",
        TRANSACTION_SEMANTICS_NOT_IN_SCOPE: "トランザクション意味は未検証です",
        CURRENT_GRAIN_HAS_NO_MEANINGFUL_LOT: "現行残高に業務上のロットキーはありません",
        CROSS_ERA_KEYS_NOT_COMPARABLE: "異なる期間のキーは比較できません",
        COMPLETED_MOVEMENTS_NOT_VERIFIED: "完了済み移動は未検証です",
        DATASET_NOT_OBSERVED_IN_EXPORT: "エクスポートでデータセットを確認できません",
        CURRENT_BALANCE_GRAIN_DUPLICATE: "現行残高の粒度が重複しています",
        QUANTITY_EVIDENCE_INCOMPLETE: "数量の根拠が不完全です",
        RAW_TRANSACTION_HEADER_NOT_OBSERVED: "生トランザクションヘッダーを確認できません",
        SQL_AGENT_ANSWER_UNVERIFIED: "LLMが計画したSQLであり、決定論的検証は未実施です",
        WMS_METADATA_VALIDATION_FAILED: "WMSメタデータが無効です",
        fallback: "未確認の技術的理由",
      },
      wmsAsOf: "データ基準時点",
      wmsImportedAt: "取込日時",
      wmsFreshnessState: "鮮度の状態",
      wmsBasis: "鮮度の算出根拠",
      wmsTimezone: "ソースのタイムゾーン",
      wmsEpoch: "セマンティック期間",
      wmsEvidence: "ドメイン別の鮮度",
      wmsReasons: "理由コード",
      wmsPagination: "ページ情報",
      wmsPerPage: "件/ページ",
      wmsHasMore: "次のページあり",
      wmsNoMore: "これで全件",
      wmsPage: "{page}ページ · 全{total}件",
      mesReport: "MESレポート",
      wmsReport: "WMSレポート",
      hrReport: "HRレポート",
      mesReportUnsupported: "Report Agent対象外",
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
      demoNotIndexed: "サンプル資料はまだ準備できていません。管理者にお問い合わせください。",
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
      groundedBadge: "資料に基づいて回答",
    },
    answerScope: {
      web: "Webソースに基づく要約",
      mes: "MESリアルタイムデータに基づく",
      mes_database: "ローカルMESスナップショットに基づく",
      wms_database: "MKHC WMS工程倉庫の資材在庫に基づく",
      mes_report: "MESスナップショットから作成したレポート",
      wms_executive_report: "WMS現行残高から作成したレポート",
      hr_executive_report: "MKAC人事ディレクトリから作成したレポート",
      mes_report_unsupported: "検証済みのレポート形式には含まれていないリクエスト",
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
    let errorCode = "";
    try {
      const body = await response.json();
      if (body.detail && typeof body.detail === "object") {
        message = body.detail.message || body.message || message;
        errorCode = body.detail.code || "";
      } else {
        message = body.detail || body.message || message;
        errorCode = body.code || "";
      }
    } catch {
      message = (await response.text()) || message;
    }
    const error = new Error(message);
    error.status = response.status;
    error.code = errorCode;
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

async function getSessionInfo(sessionId) {
  const response = await api(`/sessions/${sessionId}`);
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

function buildConversationContext(messages, limit = 16) {
  return messages
    .filter((message) => message.role === "user" || message.role === "assistant")
    .filter((message) => message.content?.trim() || message.artifact?.id)
    .slice(-limit)
    .map((message) => ({
      role: message.role,
      content: message.content || "",
      model: message.model || "",
      mode: message.mode || "",
      answer_scope: message.answerScope || "",
      ...(message.artifact?.id
        ? {
            artifact_id: message.artifact.id,
            artifact_type: message.artifact.report_type || "",
            artifact_title: message.artifact.title || "",
          }
        : {}),
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

function researchTopicStorageKey(language = "vi") {
  return `${RESEARCH_TOPIC_STORAGE_PREFIX}-${language}`;
}

function researchScopeStorageKey(language = "vi") {
  return `${RESEARCH_SCOPE_STORAGE_PREFIX}-${language}`;
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
  [/Indexing failed:\s*/i, "資料の処理に失敗しました: "],
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
  const [quickAnswersConfig, setQuickAnswersConfig] = useState({
    mkac: [],
    mes: [],
    wms: [],
    threshold: 300,
    max: 3,
  });
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
  const [researchTopics, setResearchTopics] = useState({
    ready: false,
    collection: "",
    session_id: "",
    default_topic: "",
    allow_all: true,
    topics: [],
  });
  const [researchTopicId, setResearchTopicId] = useState(() => {
    try {
      const language = storedLanguage();
      return (
        localStorage.getItem(researchTopicStorageKey(language)) ||
        localStorage.getItem(LEGACY_RESEARCH_TOPIC_STORAGE_KEY) ||
        ""
      );
    } catch {
      return "";
    }
  });
  const [researchScope, setResearchScope] = useState(() => {
    try {
      const stored = localStorage.getItem(
        researchScopeStorageKey(storedLanguage()),
      );
      return stored === "upload" ? "upload" : "topic";
    } catch {
      return "topic";
    }
  });
  const [mesStatus, setMesStatus] = useState({
    available: false,
    lots: 0,
    error_events: 0,
  });
  const [wmsStatus, setWmsStatus] = useState({
    enabled: false,
    available: false,
    state: "UNAVAILABLE",
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
  const [confirmDeleteFile, setConfirmDeleteFile] = useState("");
  const [confirmResearchUploadOpen, setConfirmResearchUploadOpen] = useState(false);
  const [researchUploadNoticeAccepted, setResearchUploadNoticeAccepted] = useState(false);
  const [fileSearch, setFileSearch] = useState("");
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
  const conversationScrollRef = useRef(null);
  const modeTabsRef = useRef(null);

  function revealActiveModeTab() {
    const container = modeTabsRef.current;
    const activeTab = container?.querySelector(`[data-mode="${mode}"]`);
    if (!container || !activeTab) return;

    const containerRect = container.getBoundingClientRect();
    const tabRect = activeTab.getBoundingClientRect();
    const left = getModeTabScrollLeft({
      scrollLeft: container.scrollLeft,
      clientWidth: container.clientWidth,
      scrollWidth: container.scrollWidth,
      tabLeft: container.scrollLeft + tabRect.left - containerRect.left,
      tabWidth: tabRect.width,
    });
    if (left === container.scrollLeft) return;
    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    container.scrollTo({ left, behavior: reduceMotion ? "auto" : "smooth" });
  }

  // Cuộn trực tiếp trong container hội thoại thay vì endRef.scrollIntoView().
  // scrollIntoView() có thể leo lên các ancestor scrolling box khác (kể cả
  // những box overflow:hidden) và lệch scrollTop của document/window khi phần
  // tử đích cao hơn viewport — khiến composer bị đẩy khỏi khung nhìn lúc thẻ
  // báo cáo mở rộng dần theo từng phần.
  function scrollConversationToEnd(smooth = true) {
    const container = conversationScrollRef.current;
    if (!container) return;
    container.scrollTo({
      top: container.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
  }

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
  const visibleModes = visibleModeKeys(Boolean(wmsStatus.available));
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
  const selectedTopic = selectedResearchTopic();
  const researchReady =
    mode !== "research" ||
    (researchScope === "upload"
      ? files.length > 0
      : Boolean(researchTopicId && selectedTopic?.ready));
  const mkacAuthorized =
    (mode !== "mkac" && mode !== "mes" && mode !== "wms") ||
    Boolean(employee?.id && employee?.name);
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

  function researchQuickPrompts() {
    const topic = selectedResearchTopic();
    if (!topic) return [];
    const prompts =
      language === "ja" ? topic.quick_prompts_ja : topic.quick_prompts_vi;
    if (prompts && prompts.length > 0) return prompts;
    return quickPromptsFor("research", language);
  }

  function starterPromptEntries() {
    if (mode === "wms") return quickAnswersConfig.wms || [];
    const prompts = mode === "research"
      ? researchScope === "topic"
        ? researchQuickPrompts()
        : quickPromptsFor("research", language)
      : quickPromptsFor(mode, language);
    return prompts.map((prompt) => ({ question: prompt }));
  }

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

    // Every WMS suggestion follows the authenticated SSE route. Only
    // server-prepared entries may carry an allowlisted quick-answer ID.
    if (mode === "wms") {
      sendMessage(suggestion.question, quickAnswerRequestOptions(suggestion));
      return;
    }
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
      scrollConversationToEnd();
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
          researchDemoResponse,
          researchTopicsResponse,
        ] = await Promise.all([
          api("/health"),
          api(`/models?language=${encodeURIComponent(language)}`),
          api("/knowledge/mkac/status"),
          api("/research/demo"),
          api("/research/topics").catch(() => null),
        ]);
        const healthData = await healthResponse.json();
        const modelData = await modelResponse.json();
        const mkacData = await mkacResponse.json();
        const researchDemoData = await researchDemoResponse.json();
        const researchTopicsData = researchTopicsResponse
          ? await researchTopicsResponse.json().catch(() => null)
          : null;

        setModels(modelData.models || []);
        setMkacStatus(mkacData);
        setResearchDemo(researchDemoData);
        if (researchTopicsData) {
          setResearchTopics(researchTopicsData);
          // UI demo chỉ cho chọn 4 nhóm chuyên đề. Backend vẫn có thể hỗ trợ
          // topic "all" cho debug/admin, nhưng không giữ scope này trên web.
          const validIds = new Set(
            (researchTopicsData.topics || []).map((topic) => topic.id),
          );
          setResearchTopicId((current) =>
            current && validIds.has(current) ? current : "",
          );
        }
        setMesStatus(healthData.mes_database || {});
        setWmsStatus(healthData.mes_wms_database || {});
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
    setQuickAnswersConfig({ mkac: [], mes: [], wms: [], threshold: 300, max: 3 });

    async function loadQuickAnswers() {
      try {
        const [qaMkacRes, qaMesRes, qaWmsRes] = await Promise.all([
          api(`/quick-answers?mode=mkac&language=${encodeURIComponent(language)}`),
          api(`/quick-answers?mode=mes&language=${encodeURIComponent(language)}`),
          api(`/quick-answers?mode=wms&language=${encodeURIComponent(language)}`),
        ]);
        const qaMkacData = await qaMkacRes.json();
        const qaMesData = await qaMesRes.json();
        const qaWmsData = await qaWmsRes.json();
        if (cancelled) return;
        setQuickAnswersConfig({
          mkac: qaMkacData.suggestions || [],
          mes: qaMesData.suggestions || [],
          wms: qaWmsData.suggestions || [],
          threshold: qaMkacData.short_answer_threshold || 300,
          max: qaMkacData.max_suggestions || 3,
        });
      } catch {
        if (cancelled) return;
        setQuickAnswersConfig({ mkac: [], mes: [], wms: [], threshold: 300, max: 3 });
      }
    }

    loadQuickAnswers();
    return () => {
      cancelled = true;
    };
  }, [language]);

  useEffect(() => {
    try {
      const storedScope = localStorage.getItem(researchScopeStorageKey(language));
      setResearchScope(storedScope === "upload" ? "upload" : "topic");
    } catch {
      setResearchScope("topic");
    }
    setPendingFiles([]);
    setUploadSummary(null);
  }, [language]);

  useEffect(() => {
    scrollConversationToEnd();
  }, [messages, busy]);

  useEffect(() => {
    let cancelled = false;

    async function syncResearchFiles() {
      if (mode !== "research" || researchScope !== "upload" || !sessionId) return;
      try {
        const info = await getSessionInfo(sessionId);
        if (cancelled) return;
        setFiles(info.files || []);
      } catch {
        if (cancelled) return;
        setFiles([]);
      }
    }

    syncResearchFiles();
    return () => {
      cancelled = true;
    };
  }, [mode, researchScope, sessionId]);

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
    if (mode === "wms" && !wmsStatus.available) {
      setMode("mkac");
      return;
    }
    const frame = window.requestAnimationFrame(revealActiveModeTab);
    return () => window.cancelAnimationFrame(frame);
  }, [mode, language, wmsStatus.available]);

  useEffect(() => {
    try {
      const storedTopic = localStorage.getItem(researchTopicStorageKey(language)) || "";
      if (storedTopic === "all") {
        localStorage.removeItem(researchTopicStorageKey(language));
        setResearchTopicId("");
      } else {
        setResearchTopicId(storedTopic);
      }
    } catch {
      setResearchTopicId("");
    }
  }, [language]);

  useEffect(() => {
    setSessionTitles((current) => {
      const next = { ...current };
      for (const workspaceMode of ACTIVE_MODE_KEYS) {
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
    if (mode !== "research") return undefined;
    const frame = window.requestAnimationFrame(() => {
      conversationScrollRef.current?.scrollTo({ top: 0, behavior: "auto" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [mode, researchScope, researchTopicId, language]);

  useEffect(() => {
    if (!confirmClearOpen && !confirmDeleteFile && !confirmResearchUploadOpen) return undefined;

    function closeConfirm(event) {
      if (event.key === "Escape") {
        setConfirmClearOpen(false);
        setConfirmDeleteFile("");
        setConfirmResearchUploadOpen(false);
      }
    }

    const blockedRoots = [
      document.querySelector(".document-sidebar"),
      document.querySelector("button.mobile-menu"),
      document.querySelector("main.workspace"),
    ].filter(Boolean);
    for (const el of blockedRoots) {
      el.setAttribute("inert", "");
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    document.addEventListener("keydown", closeConfirm);
    return () => {
      document.removeEventListener("keydown", closeConfirm);
      for (const el of blockedRoots) {
        el.removeAttribute("inert");
      }
      document.body.style.overflow = previousOverflow;
    };
  }, [confirmClearOpen, confirmDeleteFile, confirmResearchUploadOpen]);

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
    if (workspaceMode === "research" && uiLanguage === language) {
      setFiles([]);
      setPendingFiles([]);
      setUploadSummary(null);
    }
    setSidebarOpen(false);
  }

  function selectedResearchTopic() {
    if (!researchTopicId) return null;
    return (
      researchTopics.topics.find((topic) => topic.id === researchTopicId) ||
      null
    );
  }

  function researchTopicsTotalFiles() {
    return researchTopics.topics.reduce(
      (total, topic) => total + (topic.num_files || 0),
      0,
    );
  }

  function filterFileNames(names) {
    const needle = fileSearch.trim().toLowerCase();
    if (!needle) return names;
    return names.filter((name) => name.toLowerCase().includes(needle));
  }

  function researchTopicLabel(topic) {
    if (!topic) return "";
    return language === "ja"
      ? topic.label_ja || topic.label_vi || topic.id
      : topic.label_vi || topic.label_ja || topic.id;
  }

  function selectResearchTopic(topicId) {
    if (busy) return;
    if (topicId === "all") return;
    const topicChanged = topicId !== researchTopicId;
    setResearchScope("topic");
    setResearchTopicId(topicId);
    try {
      localStorage.setItem(researchScopeStorageKey(language), "topic");
      localStorage.setItem(researchTopicStorageKey(language), topicId);
    } catch {
      // Topic selection still works in-memory when storage is unavailable.
    }
    // Each topic is a separate corpus. Switching topics starts a fresh
    // conversation so answers never mix context across document groups.
    if (topicChanged) {
      setModeMessages("research", [], language);
      setPendingAssistantId("");
      setSourcePreview(null);
    }
    setQuestion("");
    setError("");
    setFileSearch("");
    setModeSources("research", [], language);
    setSourcePanelOpen(false);
  }

  function selectResearchScope(nextScope) {
    if (busy || uploading || !["topic", "upload"].includes(nextScope)) return;
    if (nextScope === researchScope) return;
    setResearchScope(nextScope);
    try {
      localStorage.setItem(researchScopeStorageKey(language), nextScope);
    } catch {
      // Scope selection still works in-memory when storage is unavailable.
    }
    // The two corpora have different ownership and retrieval semantics. Start
    // a clean conversation when switching to avoid leaking context between
    // shared topic documents and private uploaded documents.
    setModeMessages("research", [], language);
    setModeSources("research", [], language);
    setQuestion("");
    setError("");
    setPendingAssistantId("");
    setSourcePanelOpen(false);
    setSourcePreview(null);
    setPendingFiles([]);
    setUploadSummary(null);
  }

  function clearResearchTopic() {
    if (busy) return;
    setResearchTopicId("");
    try {
      localStorage.removeItem(researchTopicStorageKey(language));
    } catch {
      // Ignore storage failures; in-memory state is already cleared.
    }
    setQuestion("");
    setError("");
    setFileSearch("");
  }

  function useResearchDemoSession() {
    if (!researchDemo.ready || !researchDemo.session_id) {
      setError(t("common.demoNotIndexed"));
      return;
    }
    const scopedKey = workspaceKey("research", language);
    setSessionIds((current) => ({
      ...current,
      [scopedKey]: researchDemo.session_id,
    }));
    localStorage.setItem(sessionStorageKey("research", language), researchDemo.session_id);
    setSessionTitles((current) => ({
      ...current,
      [scopedKey]: t("defaultTitles.researchDemo"),
    }));
    persistSessionTitle(researchDemo.session_id, t("defaultTitles.researchDemo"));
    setModeMessages("research", [], language);
    setModeSources("research", [], language);
    setFiles(researchDemo.files || researchDemo.source_files || []);
    setPendingFiles([]);
    setUploadSummary(null);
    setError("");
    setSourcePanelOpen(false);
    setSidebarOpen(false);
  }

  function switchMode(nextMode) {
    if (!visibleModes.includes(nextMode)) return;
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
      source_scope:
        source.source_scope ||
        (sourceMode === "research" ? researchScope : "topic"),
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
    const modeKeys = visibleModes;
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
    modeTabsRef.current?.querySelector(`[data-mode="${nextMode}"]`)?.focus();
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
    const stoppedMode = mode;
    const stoppedLanguage = language;
    const stoppedAssistantId = pendingAssistantId;
    generationControllerRef.current?.abort();
    setPendingAssistantId("");
    setBusy(false);
    if (!stoppedAssistantId) return;
    setModeMessages(
      stoppedMode,
      (current) =>
        current.map((item) =>
          item.id === stoppedAssistantId && item.agentTimeline
            ? {
                ...item,
                agentTimeline: {
                  ...item.agentTimeline,
                  status: "cancelled",
                  steps: item.agentTimeline.steps.map((step) =>
                    ["running", "queued"].includes(step.status)
                      ? { ...step, status: "cancelled" }
                      : step,
                  ),
                },
              }
            : item,
        ),
      stoppedLanguage,
    );
  }

  function prepareReportEmail(prefix) {
    if (busy) return;
    setQuestion(prefix);
    window.requestAnimationFrame(() => {
      const textarea = textareaRef.current;
      if (!textarea) return;
      textarea.focus();
      textarea.setSelectionRange(prefix.length, prefix.length);
    });
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

  function requestResearchUploadPicker() {
    if (researchUploadNoticeAccepted) {
      fileInputRef.current?.click();
      return;
    }
    setConfirmResearchUploadOpen(true);
  }

  function confirmResearchUpload() {
    setResearchUploadNoticeAccepted(true);
    setConfirmResearchUploadOpen(false);
    fileInputRef.current?.click();
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

  async function sendMessage(prompt = question, { quickAnswerId } = {}) {
    const cleanQuestion = prompt.trim();
    if (["mkac", "mes", "wms"].includes(mode) && !mkacAuthorized) {
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
    pushStoredPromptHistory(cleanQuestion);
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
              : requestMode === "wms"
                ? "wms_database"
                : "research",
        researchScope: requestMode === "research" ? researchScope : undefined,
        sources: [],
        agentTimeline: null,
        artifact: null,
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
          employee_id: ["mkac", "mes", "wms"].includes(requestMode)
            ? employee?.id
            : undefined,
          quick_answer_id: requestMode === "wms" ? quickAnswerId || null : null,
          conversation_context: conversationContext,
          research_topic:
            requestMode === "research" && researchScope === "topic"
              ? researchTopicId || null
              : null,
          research_scope: requestMode === "research" ? researchScope : null,
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
          if (event.type === "agent_plan") {
            setModeMessages(
              requestMode,
              (current) =>
                current.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        status: "",
                        agentTimeline: {
                          title: event.title || "Report Agent",
                          workflow: event.workflow || "",
                          status: "running",
                          steps: (event.steps || []).map((step) => ({
                            ...step,
                            status: "queued",
                            summary: "",
                          })),
                        },
                      }
                    : item,
                ),
              requestLanguage,
            );
          }
          if (event.type === "tool_start") {
            setModeMessages(
              requestMode,
              (current) =>
                current.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        agentTimeline: item.agentTimeline
                          ? {
                              ...item.agentTimeline,
                              steps: item.agentTimeline.steps.map((step) =>
                                step.id === event.step_id
                                  ? { ...step, status: "running" }
                                  : step,
                              ),
                            }
                          : item.agentTimeline,
                      }
                    : item,
                ),
              requestLanguage,
            );
          }
          if (event.type === "tool_result") {
            setModeMessages(
              requestMode,
              (current) =>
                current.map((item) =>
                  item.id === assistantId
                    ? {
                        ...item,
                        agentTimeline: item.agentTimeline
                          ? {
                              ...item.agentTimeline,
                              steps: item.agentTimeline.steps.map((step) =>
                                step.id === event.step_id
                                  ? {
                                      ...step,
                                      status: event.status === "error" ? "error" : "done",
                                      summary: event.summary || "",
                                    }
                                  : step,
                              ),
                            }
                          : item.agentTimeline,
                      }
                    : item,
                ),
              requestLanguage,
            );
          }
          if (
            event.type === "artifact" &&
            ["mes_report", "wms_executive_report", "hr_executive_report"].includes(event.artifact_type)
          ) {
            setModeMessages(
              requestMode,
              (current) =>
                current.map((item) =>
                  item.id === assistantId ? { ...item, artifact: event.artifact } : item,
                ),
              requestLanguage,
            );
          }
          if (event.type === "agent_done") {
            setModeMessages(
              requestMode,
              (current) =>
                current.map((item) =>
                  item.id === assistantId && item.agentTimeline
                    ? {
                        ...item,
                        agentTimeline: { ...item.agentTimeline, status: "done" },
                      }
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
                        researchScope: event.research_scope || item.researchScope,
                        workflow: event.workflow || item.workflow,
                        wmsMetadata: event.wms_metadata || item.wmsMetadata,
                        sourceKind: event.source_kind || item.sourceKind,
                        cache: event.cache ?? item.cache,
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
      if (
        shouldClearEmployeeAuth({
          status: queryError.status,
          errorCode: queryError.code,
          mode: requestMode,
        })
      ) {
        setEmployee(null);
        setEmployeeCodeInput("");
        setEmployeeCodeError(t("common.invalidEmployee"));
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
    for (const workspaceMode of ["mkac", "mes", "wms"]) {
      for (const item of LANGUAGE_OPTIONS) {
        setModeMessages(workspaceMode, [], item);
        setModeSources(workspaceMode, [], item);
      }
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
    <div className={`app-shell mode-${mode} ${mode !== "research" ? "mkac-layout" : ""}`}>
      <ResearchSidebar
        mode={mode}
        t={t}
        modeText={modeText}
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
        sessionTitles={sessionTitles}
        currentWorkspaceKey={currentWorkspaceKey}
        researchScope={researchScope}
        resetSession={resetSession}
        busy={busy}
        uploading={uploading}
        selectResearchScope={selectResearchScope}
        researchTopicId={researchTopicId}
        selectedResearchTopic={selectedResearchTopic}
        researchTopicsTotalFiles={researchTopicsTotalFiles}
        files={files}
        researchTopics={researchTopics}
        language={language}
        researchTopicLabel={researchTopicLabel}
        selectResearchTopic={selectResearchTopic}
        formatText={formatText}
        fileSearch={fileSearch}
        setFileSearch={setFileSearch}
        filterFileNames={filterFileNames}
        FileTypeIcon={FileTypeIcon}
        fileInputRef={fileInputRef}
        addPendingFiles={addPendingFiles}
        dragActive={dragActive}
        setDragActive={setDragActive}
        requestResearchUploadPicker={requestResearchUploadPicker}
        onDrop={onDrop}
        pendingFiles={pendingFiles}
        pendingTotalSize={pendingTotalSize}
        formatBytes={formatBytes}
        removePendingFile={removePendingFile}
        uploadDocuments={uploadDocuments}
        uploadProgress={uploadProgress}
        uploadSummary={uploadSummary}
        setConfirmDeleteFile={setConfirmDeleteFile}
        health={health}
      />

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
                    : mode === "wms"
                      ? wmsStatus.available
                        ? `${formatText(modeText("wms").itemCount, {
                            count: wmsStatus.distinct_items || 0,
                          })} · ${formatText(modeText("wms").processCount, {
                            count: wmsStatus.distinct_process_codes || 0,
                          })}`
                        : modeText("wms").unavailable
                      : researchScope === "upload"
                        ? `${files.length} ${t("common.sessionDocuments")}`
                        : researchTopicId
                        ? formatText(modeText("research").scopeLabel, {
                            topic: researchTopicLabel(selectedResearchTopic()),
                          })
                        : modeText("research").chooseTopic}
              </span>
            </div>
          </div>

          <div className="header-actions">
            <div
              className="mode-tabs"
              ref={modeTabsRef}
              role="tablist"
              aria-label={t("common.qaModes")}
            >
              {visibleModes.map((key) => {
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
                    aria-label={optionText.label}
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
                language: language === "vi" ? "Tiếng Việt" : "日本語",
              })}
              aria-label={t("common.languageTitle", {
                language: language === "vi" ? "Tiếng Việt" : "日本語",
              })}
              onClick={cycleLanguage}
              disabled={busy || uploading}
            >
              <span className={language === "vi" ? "lang-opt active" : "lang-opt"}>VI</span>
              <span className={language === "ja" ? "lang-opt active" : "lang-opt"}>JA</span>
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

            {["mkac", "mes", "wms"].includes(mode) && employee && (
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

        {mode === "wms" && !wmsStatus.available && (
          <div className={`wms-health-banner ${String(
            wmsStatus.state || (wmsStatus.enabled ? "UNAVAILABLE" : "DISABLED"),
          ).toLowerCase()}`} role="status">
            {t(
              `common.wmsHealth.${
                wmsStatus.state || (wmsStatus.enabled ? "UNAVAILABLE" : "DISABLED")
              }`,
            )}
          </div>
        )}

        <div className="workspace-body">
          <section
            id={`${mode}-conversation`}
            className="conversation"
            role="tabpanel"
            aria-label={currentMode.title}
            aria-busy={busy}
          >
            <div className="conversation-scroll" ref={conversationScrollRef}>
              {["mkac", "mes", "wms"].includes(mode) && !mkacAuthorized ? (
                <EmployeeLogin
                  mode={mode}
                  modeText={modeText}
                  t={t}
                  verifyEmployeeCode={verifyEmployeeCode}
                  employeeCodeInput={employeeCodeInput}
                  updateEmployeeCodeInput={updateEmployeeCodeInput}
                  employeeCodeError={employeeCodeError}
                  employeeVerifying={employeeVerifying}
                />
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
                    {mode === "research" && (
                      <div className="research-source-choice" role="tablist">
                        <button
                          type="button"
                          className={researchScope === "topic" ? "active" : ""}
                          onClick={() => selectResearchScope("topic")}
                          disabled={busy || uploading}
                          role="tab"
                          aria-selected={researchScope === "topic"}
                        >
                          <Layers3 size={17} />
                          <span>{modeText("research").sourceTopic}</span>
                        </button>
                        <button
                          type="button"
                          className={researchScope === "upload" ? "active" : ""}
                          onClick={() => selectResearchScope("upload")}
                          disabled={busy || uploading}
                          role="tab"
                          aria-selected={researchScope === "upload"}
                        >
                          <FileUp size={17} />
                          <span>{modeText("research").sourceUpload}</span>
                        </button>
                      </div>
                    )}
                    {mode !== "research" && (
                      <p>{modeText(mode).empty}</p>
                    )}
                    {mode === "research" && researchScope === "topic" && !researchTopicId && (
                      <div className="research-topic-grid">
                        {researchTopics.topics.map((topic) => (
                          <button
                            key={topic.id}
                            type="button"
                            className={`research-topic-card accent-${topic.accent || "neutral"}`}
                            onClick={() => selectResearchTopic(topic.id)}
                            disabled={!topic.ready}
                            title={
                              topic.ready
                                ? researchTopicLabel(topic)
                                : modeText("research").topicNotReady
                            }
                          >
                            <span className="research-topic-label">
                              {researchTopicLabel(topic)}
                            </span>
                            <span className="research-topic-description">
                              {language === "ja"
                                ? topic.description_ja
                                : topic.description_vi}
                            </span>
                            <span className="research-topic-meta">
                              {formatText(modeText("research").documentsCount, {
                                count: topic.num_files,
                              })}
                            </span>
                          </button>
                        ))}
                      </div>
                    )}
                    {mode === "research" && researchScope === "topic" && researchTopicId && (
                      <div className="research-scope-bar">
                        <span className="research-scope-label">
                          {formatText(modeText("research").scopeLabel, {
                            topic: researchTopicLabel(selectedResearchTopic()),
                          })}
                        </span>
                        <button
                          type="button"
                          className="research-scope-change"
                          onClick={clearResearchTopic}
                        >
                          {modeText("research").changeTopic}
                        </button>
                      </div>
                    )}
                    {mode === "research" && researchScope === "upload" && files.length === 0 && (
                      <div className="research-upload-empty-action">
                        <button
                          type="button"
                          className="research-upload-action"
                          onClick={requestResearchUploadPicker}
                          disabled={busy || uploading}
                        >
                          <UploadCloud size={18} />
                          <span>{t("common.chooseToStart")}</span>
                        </button>
                      </div>
                    )}
                  </div>

                  {mode !== "research" || researchScope === "upload" || researchTopicId ? (
                    <div className="prompt-section">
                      <div className="prompt-grid">
                        {starterPromptEntries().map((prompt) => (
                          <button
                            key={prompt.id || prompt.question}
                            type="button"
                            onClick={() =>
                              mode === "wms"
                                ? handleQuickAnswerClick(prompt)
                                : sendMessage(prompt.question)
                            }
                            disabled={!researchReady || !mkacAuthorized}
                          >
                            <Search size={16} />
                            <span>{prompt.question}</span>
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
                          : mode === "wms"
                            ? `${formatText(modeText("wms").itemCount, {
                                count: wmsStatus.distinct_items || 0,
                              })} · ${formatText(modeText("wms").processCount, {
                                count: wmsStatus.distinct_process_codes || 0,
                              })}`
                            : researchScope === "upload"
                            ? `${files.length} ${modeText("research").metric}`
                            : researchTopicId
                              ? `${
                                  selectedResearchTopic()?.num_files ?? 0
                                } ${modeText("research").metric}`
                              : formatText(
                                  modeText("research").documentsSummary,
                                  {
                                    count: researchTopicsTotalFiles(),
                                    topics: researchTopics.topics.length,
                                  },
                                )}
                    </span>
                    {mode === "wms" && (
                      <>
                        <span className="status-sep" aria-hidden="true">·</span>
                        <span
                          className={`wms-health-state ${String(
                            wmsStatus.state || (wmsStatus.enabled ? "UNAVAILABLE" : "DISABLED"),
                          ).toLowerCase()}`}
                        >
                          {t(
                            `common.wmsHealth.${
                              wmsStatus.state || (wmsStatus.enabled ? "UNAVAILABLE" : "DISABLED")
                            }`,
                          )}
                        </span>
                      </>
                    )}
                  </div>
                </div>
              ) : (
                <MessageList
                  messages={messages}
                  mode={mode}
                  busy={busy}
                  language={language}
                  t={t}
                  latestAssistantMessage={latestAssistantMessage}
                  pendingAssistantId={pendingAssistantId}
                  waitingMessageIndex={waitingMessageIndex}
                  text={text}
                  copiedMessageId={copiedMessageId}
                  copyAnswer={copyAnswer}
                  openSourcePreview={openSourcePreview}
                  getSuggestions={getSuggestions}
                  handleQuickAnswerClick={handleQuickAnswerClick}
                  endRef={endRef}
                  onScrollToEnd={scrollConversationToEnd}
                  error={error}
                  MessageMarkdown={MessageMarkdown}
                  AgentTimeline={AgentTimeline}
                  ReportArtifactCard={ReportArtifactCard}
                  onReportEmail={prepareReportEmail}
                  Bot={Bot}
                />
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

            {mkacAuthorized &&
              // Research: ẩn ô chat khi chưa chọn nhóm chủ đề hoặc chưa upload tài liệu.
              !(mode === "research" &&
                ((researchScope === "topic" && !researchTopicId) ||
                  (researchScope === "upload" && files.length === 0))) && (
              <ChatInput
              mode={mode}
              t={t}
              modeText={modeText}
              busy={busy}
              question={question}
              setQuestion={setQuestion}
              onSubmit={onSubmit}
              textareaRef={textareaRef}
              employee={employee}
              language={language}
              quickPromptsFor={quickPromptsFor}
              researchQuickPrompts={researchQuickPrompts}
              sendMessage={sendMessage}
              health={health}
              researchScope={researchScope}
              researchTopicId={researchTopicId}
              researchReady={researchTopics.ready}
              mkacAuthorized={mkacAuthorized}
              requestResearchUploadPicker={requestResearchUploadPicker}
              uploading={uploading}
              stopGeneration={stopGeneration}
              canAsk={canAsk}
              Square={Square}
              Send={Send}
              Paperclip={Paperclip}
              Search={Search}
              userPrompts={messages.filter((m) => m.role === "user").map((m) => m.content)}
            />
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
      <SourcePreviewDialog
        sourcePreview={sourcePreview}
        setSourcePreview={setSourcePreview}
        sourcePreviewUrl={sourcePreviewUrl}
        t={t}
      />

      {confirmResearchUploadOpen && (
        <div
          className="confirm-backdrop"
          role="presentation"
          onClick={() => setConfirmResearchUploadOpen(false)}
        >
          <section
            className="confirm-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="research-upload-notice-title"
            aria-describedby="research-upload-notice-body"
            onClick={(event) => event.stopPropagation()}
          >
            <h2 id="research-upload-notice-title">
              {t("common.uploadRecommendationTitle")}
            </h2>
            <p id="research-upload-notice-body">{t("common.uploadRecommendation")}</p>
            <div className="confirm-actions">
              <button
                type="button"
                className="confirm-cancel"
                onClick={() => setConfirmResearchUploadOpen(false)}
                autoFocus
              >
                {t("common.cancel")}
              </button>
              <button type="button" className="confirm-primary" onClick={confirmResearchUpload}>
                {t("common.uploadRecommendationAction")}
              </button>
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

      {confirmDeleteFile && (
        <div
          className="confirm-backdrop"
          role="presentation"
          onClick={() => setConfirmDeleteFile("")}
        >
          <section
            className="confirm-dialog"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="confirm-delete-file-title"
            aria-describedby="confirm-delete-file-body"
            onClick={(event) => event.stopPropagation()}
          >
            <h2 id="confirm-delete-file-title">
              {t("common.deleteFileConfirmTitle")}
            </h2>
            <p id="confirm-delete-file-body">
              {t("common.deleteFileConfirmBody", { name: confirmDeleteFile })}
            </p>
            <div className="confirm-actions">
              <button
                type="button"
                className="confirm-cancel"
                onClick={() => setConfirmDeleteFile("")}
                autoFocus
              >
                {t("common.cancel")}
              </button>
              <button
                type="button"
                className="confirm-danger"
                onClick={() => {
                  removeFile(confirmDeleteFile);
                  setConfirmDeleteFile("");
                }}
              >
                {t("common.deleteFileConfirmAction")}
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
