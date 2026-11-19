import { StrictMode, memo, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import ReactMarkdown from "react-markdown";
import { getMessageMarkdownOptions } from "./markdown-tables";
import AgentPresence, { AgentStatus, MotionToggle } from "./components/AgentPresence.jsx";
import RoamingRobot from "./components/RoamingRobot.jsx";
import IntroSplash from "./components/IntroSplash.jsx";
import { canAnimate, prefersReducedMotion, runViewTransition } from "./motion.js";
import { THEME_DEFAULT, THEME_MIGRATION_KEY, resolveStoredTheme } from "./theme-preference.js";
import TaskLauncher from "./components/TaskLauncher.jsx";
import MsmMode from "./components/MsmMode.jsx";
import { buildTaskGroups, prepareTaskRun } from "./task-launcher.js";
import { resolveAgentPresence } from "./agent-presence.js";

// Memo hoá theo nội dung: khi stream token, chỉ message đang thay đổi mới
// parse lại markdown; các message cũ giữ nguyên nên không re-parse (tránh
// O(n) lần parse markdown cho mỗi token khi hội thoại dài).
// Bảng Markdown chỉ bật cho câu trả lời QA để không đổi hiển thị các mode khác.
const MessageMarkdown = memo(function MessageMarkdown({ content, qa = false }) {
  return <ReactMarkdown {...getMessageMarkdownOptions({ qa })}>{content}</ReactMarkdown>;
});
import {
  Activity,
  AlertCircle,
  Bot,
  Boxes,
  Check,
  CheckCircle2,
  ChevronDown,
  ClipboardCheck,
  Copy,
  Database,
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
  Cctv,
  ExternalLink,
} from "lucide-react";
import { EmployeeLogin } from "./components/EmployeeLogin";
import { SourcePreviewDialog } from "./components/SourcePreviewDialog";
import { ResearchSidebar } from "./components/ResearchSidebar";
import { MessageList } from "./components/MessageList";
import { ChatInput, pushStoredPromptHistory } from "./components/ChatInput";
import { streamQuery } from "./query-stream";
import { buildQaPayload, isValidQaQuestion, QA_MAX_CODE_POINTS } from "./qa-mode";
import { getQaStarterPrompts, QA_STARTER_PROMPTS } from "./qa-prompts";
import { quickAnswerRequestOptions } from "./quick-answer-request";
import { getSuggestions as resolveSuggestions } from "./suggestions";
import {
  shouldClearEmployeeAuth,
  visibleModeKeys,
  isCctvaiAvailable,
  cctvaiHealthBannerVisibility,
} from "./mode-tabs";
import "./styles.css";
import "./intro.css";

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
  const isCctvai = artifact.report_type === "cctvai_report";
  const labels = language === "ja"
    ? {
        badge: isWms ? "WMSレポート" : isHr ? "HRレポート" : isCctvai ? "CCTVAIレポート" : "MESレポート",
        governance: "ガバナンス上の注意",
        observations: "数値からの所見",
        limitations: "データの制限",
        details: "データ表を表示",
        download: "レポートを開く",
        emailAction: "メールで共有",
      }
    : {
        badge: isWms ? "Báo cáo WMS" : isHr ? "Báo cáo HR" : isCctvai ? "Báo cáo CCTVAI" : "Báo cáo MES",
        governance: "Quy tắc Governance & Data Contract",
        observations: "Nhận xét từ số liệu",
        limitations: "Giới hạn dữ liệu",
        details: "Xem các bảng số liệu",
        download: "Mở báo cáo",
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
          <a className="report-download" href={artifact.download_url} target="_blank" rel="noopener noreferrer">
            <ExternalLink size={16} aria-hidden="true" />
            {labels.download}
          </a>
        )}
        <button
          className="report-email-action"
          type="button"
          onClick={() => onEmail?.({ artifactId: artifact.id, language })}
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
    cctvai: [
      "Tổng quan dữ liệu CCTVAI hiện có bao nhiêu sự kiện?",
      "7 ngày qua sự kiện vi phạm theo mức độ nghiêm trọng như thế nào?",
      "Danh sách camera đang hoạt động trong hệ thống?",
    ],
    qa: QA_STARTER_PROMPTS,
    msm: [
      "Máy nào sắp đến hạn hiệu chuẩn?",
      "Kiểm tra trạng thái thiết bị bộ phận ANNEALING",
      "Chuẩn bị form thêm thiết bị mới",
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
    cctvai: [
      "CCTVAIデータの概要：現在イベントは何件ありますか？",
      "直近7日間の違反イベントを重要度別に教えてください。",
      "システム内で稼働中のカメラ一覧を表示してください。",
    ],
    // Agent QA nhận tiếng Việt; nhãn Nhật được map riêng trước khi gửi câu Việt.
    qa: getQaStarterPrompts("ja").map((item) => item.label),
    msm: [
      "校正期限が近づいている設備はどれですか？",
      "ANNEALING部門の設備状態を確認してください",
      "新規設備登録フォームを開いてください",
    ],
    research: [
      "資料から総合的な調査レポートを作成してください",
      "各見解を比較し、矛盾点を示してください",
      "次に調査すべき質問を提案してください",
    ],
  },
};

const ACTIVE_MODE_KEYS = ["mkac", "mes", "wms", "cctvai", "qa", "research"];

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
  cctvai: {
    icon: Cctv,
  },
  qa: {
    icon: ClipboardCheck,
  },
  research: {
    icon: FlaskConical,
  },
};

const MODEL_ACCENTS = {
  auto: "accent-auto",
  local: "accent-local",
  openai: "accent-openai",
};

const LEGACY_MODE_SESSION_STORAGE_KEYS = {
  mkac: "meibook-session-mkac",
  mes: "meibook-session-mes",
  wms: "meibook-session-wms",
  cctvai: "meibook-session-cctvai",
  qa: "meibook-session-qa",
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
// Fallback khi backend không trả short_answer_threshold cho mode nào đó.
const DEFAULT_SHORT_ANSWER_THRESHOLD = 300;
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
      qa: {
        label: "Quản lý chất lượng",
        shortLabel: "QA",
        title: "Hỏi đáp Quản lý chất lượng — Agent QMS 5M1E",
        empty: "Tra cứu thông tin chất lượng từ Agent QMS 5M1E.",
        metric: "Agent QMS 5M1E",
        authHint: "Nhập mã nhân viên để truy cập dữ liệu quản lý chất lượng.",
        inputLabel: "Câu hỏi về chất lượng",
        placeholder: "Hỏi theo mã máy (vd. B302-01) hoặc mã phiếu kiểm tra (vd. MMSHB B0121-01)...",
        guidance: "Agent trả lời bằng tiếng Việt và hiểu câu hỏi nối tiếp trong cùng cuộc trò chuyện; câu đầu tiên nên ghi đủ mã máy hoặc mã phiếu kiểm tra, khoảng thời gian và điều kiện. Tạo cuộc trò chuyện mới để hỏi chủ đề khác.",
        unavailable: "Agent QMS 5M1E hiện chưa khả dụng. Vui lòng thử lại sau.",
        misconfigured: "Agent QMS chưa được cấu hình. Vui lòng liên hệ quản trị viên.",
      },
      cctvai: {
        label: "Camera CCTVAI",
        shortLabel: "Camera",
        title: "Hỏi đáp dữ liệu Camera CCTVAI",
        empty: "Tra cứu sự kiện vi phạm, camera và tuyến từ dữ liệu báo cáo CCTVAI.",
        metric: "Sự kiện CCTVAI",
        authHint: "Nhập mã nhân viên để truy cập dữ liệu camera CCTVAI.",
        inputLabel: "Câu hỏi về camera CCTVAI",
        placeholder: "Hỏi về sự kiện vi phạm, camera hoặc tuyến CCTVAI...",
        unavailable: "Dữ liệu CCTVAI chưa khả dụng",
        hardwareUnavailable: "Giám sát phần cứng server CCTVAI chưa khả dụng",
      },
      msm: {
        label: "Quản lý thiết bị MSM",
        shortLabel: "MSM",
        title: "Hỏi đáp quản lý thiết bị MSM",
        empty: "Tra cứu máy, hạn hiệu chuẩn, kế hoạch MSA và trạng thái thiết bị từ MSM.",
        metric: "Dữ liệu MSM",
        authHint: "Nhập mã nhân viên để truy cập dữ liệu thiết bị MSM.",
        inputLabel: "Câu hỏi về thiết bị MSM",
        placeholder: "Hỏi theo mã máy hoặc bộ phận, ví dụ: máy nào quá hạn hiệu chuẩn?",
        guidance: "Agent trả lời bằng tiếng Việt và hiểu câu hỏi nối tiếp trong cùng cuộc trò chuyện. Thao tác ghi chỉ được thực hiện sau khi bạn bấm Xác nhận trên thẻ chờ.",
        unavailable: "Dữ liệu MSM hiện chưa khả dụng. Vui lòng thử lại sau.",
        misconfigured: "Chế độ MSM chưa được cấu hình. Vui lòng liên hệ quản trị viên.",
        // MODE_OPTIONS không có khoá msm, header lấy biểu tượng qua modeText().
        icon: Server,
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
      cctvai: "Phiên Camera CCTVAI mới",
      qa: "Phiên QA mới",
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
      qaQuestionLimit: "Câu hỏi QA phải có từ 1 đến {limit} ký tự Unicode.",
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
      cctvaiReport: "Báo cáo CCTVAI",
      cctvaiHardware: "Phần cứng CCTVAI",
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
      qa_agent: "Câu trả lời từ Agent QMS 5M1E; nội dung tham khảo, chưa được xác minh nghiệp vụ.",
      qa_disclaimer: "Tham chiếu không đảm bảo dữ liệu chính xác hoặc còn mới; hãy đối chiếu với quy trình và nguồn QMS có thẩm quyền.",
      web: "Tổng hợp từ nguồn web",
      mes: "Dựa trên dữ liệu MES trực tiếp",
      mes_database: "Dựa trên MES snapshot cục bộ",
      wms_database: "Dựa trên tồn nguyên vật liệu tại kho công đoạn WMS MKHC",
      cctvai_database: "Dựa trên dữ liệu báo cáo camera CCTVAI",
      cctvai_hardware: "Tổng hợp tất định từ dữ liệu giám sát phần cứng CCTVAI",
      mes_report: "Báo cáo được tổng hợp từ MES snapshot",
      wms_executive_report: "Báo cáo được tổng hợp từ current balance WMS",
      hr_executive_report: "Báo cáo được tổng hợp từ danh bạ nhân sự MKAC",
      cctvai_report: "Báo cáo được tổng hợp từ replica báo cáo CCTVAI",
      mes_report_unsupported: "Yêu cầu chưa thuộc các mẫu báo cáo đã được kiểm chứng",
      research: "Dựa trên tài liệu nghiên cứu",
      mkac: "Dựa trên kho MKAC",
      fallback: "Không có nguồn đối chiếu",
    },
    models: {
      local: {
        name: "Qwen primary",
        description: "Dùng Qwen từ xa; Azure chỉ dự phòng khi có lỗi kỹ thuật.",
      },
      openai: {
        name: "Model dự phòng",
        description: "Chỉ dùng làm tuyến dự phòng kỹ thuật.",
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
      qa: {
        label: "品質管理",
        shortLabel: "QA",
        title: "品質管理Q&A — Agent QMS 5M1E",
        empty: "Agent QMS 5M1Eから品質情報を検索します。",
        metric: "Agent QMS 5M1E",
        authHint: "品質管理データにアクセスするには社員番号を入力してください。",
        inputLabel: "品質に関する質問",
        placeholder: "設備コード（例: B302-01）またはチェックシートコード（例: MMSHB B0121-01）で質問してください...",
        guidance: "Agentはベトナム語で回答し、同じ会話内の続けての質問を理解します。最初の質問には設備コードまたはチェックシートコード、期間、条件を含めてください。別の話題は新しい会話で質問してください。",
        unavailable: "Agent QMS 5M1Eは現在利用できません。後でもう一度お試しください。",
        misconfigured: "Agent QMSが設定されていません。管理者にお問い合わせください。",
      },
      cctvai: {
        label: "カメラ CCTVAI",
        shortLabel: "カメラ",
        title: "CCTVAIカメラデータQ&A",
        empty: "CCTVAIレポートデータからカメラ違反イベント・カメラ・ラインを検索します。",
        metric: "CCTVAIイベント",
        authHint: "CCTVAIカメラデータにアクセスするには社員番号を入力してください。",
        inputLabel: "CCTVAIカメラに関する質問",
        placeholder: "違反イベント、カメラ、ラインについて質問...",
        unavailable: "CCTVAIデータはまだ利用できません",
        hardwareUnavailable: "CCTVAIサーバーのハードウェア監視はまだ利用できません",
      },
      msm: {
        label: "MSM設備管理",
        shortLabel: "MSM",
        title: "MSM設備管理の質問応答",
        empty: "MSMから設備、校正期限、MSA計画、設備状態を検索します。",
        metric: "MSMデータ",
        authHint: "MSM設備データにアクセスするには社員番号を入力してください。",
        inputLabel: "MSM設備に関する質問",
        placeholder: "設備コードまたは部署で質問してください（例: 校正期限を過ぎた設備は？）",
        guidance: "Agentはベトナム語で回答し、同じ会話内の続けての質問を理解します。書き込み操作は確認カードで「確認」を押した後にのみ実行されます。",
        unavailable: "MSMデータは現在利用できません。後でもう一度お試しください。",
        misconfigured: "MSMモードが設定されていません。管理者にお問い合わせください。",
        icon: Server,
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
      cctvai: "新しいCCTVAIカメラセッション",
      qa: "新しいQAセッション",
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
      qaQuestionLimit: "QAの質問はUnicode文字数で1〜{limit}文字にしてください。",
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
      cctvaiReport: "CCTVAIレポート",
      cctvaiHardware: "CCTVAIハードウェア",
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
      qa_agent: "Agent QMS 5M1Eによる回答です。業務上の正確性は未検証です。",
      qa_disclaimer: "参照情報はデータの正確性や最新性を保証しません。正式なQMS情報と照合してください。",
      web: "Webソースに基づく要約",
      mes: "MESリアルタイムデータに基づく",
      mes_database: "ローカルMESスナップショットに基づく",
      wms_database: "MKHC WMS工程倉庫の資材在庫に基づく",
      cctvai_database: "CCTVAIカメラレポートデータに基づく",
      cctvai_hardware: "CCTVAIハードウェアデータの決定的な集計",
      mes_report: "MESスナップショットから作成したレポート",
      wms_executive_report: "WMS現行残高から作成したレポート",
      hr_executive_report: "MKAC人事ディレクトリから作成したレポート",
      cctvai_report: "CCTVAIレポートレプリカから作成したレポート",
      mes_report_unsupported: "検証済みのレポート形式には含まれていないリクエスト",
      research: "調査資料に基づく",
      mkac: "MKACナレッジベースに基づく",
      fallback: "照合できる参照元はありません",
    },
    models: {
      local: {
        name: "Qwen メインモデル",
        description: "Qwenリモートモデルを使い、Azureは障害時のみ使用します。",
      },
      openai: {
        name: "予備モデル",
        description: "技術的な予備ルートとしてのみ使用します。",
      },
    },
  },
};

function storedTheme() {
  try {
    const { theme, migrate } = resolveStoredTheme(
      localStorage.getItem(THEME_STORAGE_KEY),
      localStorage.getItem(THEME_MIGRATION_KEY),
    );
    if (migrate) {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
      localStorage.setItem(THEME_MIGRATION_KEY, "1");
    }
    return THEME_OPTIONS.includes(theme) ? theme : THEME_DEFAULT;
  } catch {
    return THEME_DEFAULT;
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


function buildConversationContext(messages, limit = 16) {
  const excludedHardwareTurns = new Set();
  messages.forEach((message, index) => {
    if (message.answerScope === "cctvai_hardware") {
      excludedHardwareTurns.add(index);
      if (index > 0 && messages[index - 1]?.role === "user") {
        excludedHardwareTurns.add(index - 1);
      }
    }
  });
  return messages
    .filter((message, index) => !excludedHardwareTurns.has(index))
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
  // Backend /models trả id "auto" cho model local duy nhất hiện có, nhưng
  // bảng dịch UI_TEXT.models dùng key "local". Không map thì tra cứu luôn
  // miss và badge model bị đóng băng ở ngôn ngữ lúc /models fetch lần đầu
  // (bootstrap effect chỉ chạy 1 lần, không refetch khi đổi ngôn ngữ UI).
  const textKey = modelInfo.id === "auto" ? "local" : modelInfo.id;
  const localized = UI_TEXT[language]?.models?.[textKey];
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

/**
 * Popup nhập địa chỉ email để gửi báo cáo.
 * Mở khi nhấn nút "Gửi qua email" trên ReportArtifactCard.
 * Nhấn Enter hoặc nút Gửi → gọi onSend(email).
 * Gõ câu lệnh trực tiếp vào ô chat vẫn hoạt động như cũ.
 */
function EmailDialog({ language, onClose, onSend }) {
  const [value, setValue] = useState("");
  const inputRef = useRef(null);
  const isJa = language === "ja";

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  function handleKeyDown(e) {
    if (e.key === "Escape") { onClose(); return; }
    if (e.key === "Enter") { submit(); }
  }

  function submit() {
    const email = value.trim();
    if (!email) return;
    onSend(email);
  }

  return (
    <div
      className="confirm-backdrop"
      role="presentation"
      onClick={onClose}
    >
      <section
        className="confirm-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="email-dialog-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="email-dialog-title">
          {isJa ? "レポートをメールで送信" : "Gửi báo cáo qua email"}
        </h2>
        <p>
          {isJa
            ? "送信先のメールアドレスを入力してください。"
            : "Nhập địa chỉ email người nhận, sau đó nhấn Enter hoặc Gửi."}
        </p>
        <input
          ref={inputRef}
          className="email-dialog-input"
          type="email"
          placeholder={isJa ? "例: tanaka@example.com" : "Ví dụ: 12wuu115@gmail.com"}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          autoComplete="email"
        />
        <div className="confirm-actions">
          <button type="button" className="confirm-cancel" onClick={onClose}>
            {isJa ? "キャンセル" : "Hủy"}
          </button>
          <button
            type="button"
            className="confirm-primary"
            onClick={submit}
            disabled={!value.trim()}
          >
            {isJa ? "送信" : "Gửi"}
          </button>
        </div>
      </section>
    </div>
  );
}

function App() {
  const [theme, setTheme] = useState(storedTheme);
  const [language, setLanguage] = useState(storedLanguage);
  const initialLanguageRef = useRef(storedLanguage());
  const [quickAnswersConfig, setQuickAnswersConfig] = useState({
    mkac: [],
    mes: [],
    wms: [],
    cctvai: [],
    threshold: {},
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
  const [msmTurnCount, setMsmTurnCount] = useState(0);
  const [msmResetEpoch, setMsmResetEpoch] = useState(0);
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
  const [cctvaiStatus, setCctvaiStatus] = useState({
    enabled: false,
    available: false,
    state: "UNAVAILABLE",
  });
  // Trạng thái giám sát phần cứng server CCTVAI — nguồn độc lập với database
  // report replica ở trên. Health load riêng để mất một bên không kéo theo
  // ẩn tab hoặc hiểu nhầm là bên còn lại cũng lỗi.
  const [cctvaiHardwareStatus, setCctvaiHardwareStatus] = useState({
    enabled: false,
    available: false,
    state: "UNAVAILABLE",
  });
  const [qaStatus, setQaStatus] = useState({
    enabled: false,
    configured: false,
    state: "UNKNOWN",
    max_question_chars: QA_MAX_CODE_POINTS,
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
  // { open: bool, artifactId: string|null, language: "vi"|"ja" }
  const [emailDialog, setEmailDialog] = useState({ open: false, artifactId: null, language: "vi" });
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
  const cctvaiAvailable = isCctvaiAvailable({
    database: cctvaiStatus,
    hardware: cctvaiHardwareStatus,
  });
  // DB và hardware cảnh báo riêng biệt: mất một bên không được hiển thị lẫn
  // sang cảnh báo của bên còn lại đang hoạt động bình thường.
  const cctvaiBannerVisibility = cctvaiHealthBannerVisibility({
    database: cctvaiStatus,
    hardware: cctvaiHardwareStatus,
  });
  const visibleModes = visibleModeKeys({
    wms: Boolean(wmsStatus.available),
    cctvai: cctvaiAvailable,
    qa: Boolean(qaStatus.enabled),
  });
  const requestModel = mode === "qa" ? "auto" : model;
  const mkacModels = useMemo(
    () => localizedModels.filter((item) => !item.hidden_in_mkac),
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
  const employeeProtectedMode = ["mkac", "mes", "wms", "cctvai", "qa"].includes(mode);
  const qaAuthorized = Boolean(employee?.id && employee?.name && !isGuestEmployee(employee));
  const mkacAuthorized = !employeeProtectedMode || (mode === "qa" ? qaAuthorized : Boolean(employee?.id && employee?.name));
  const qaQuestionValid = mode !== "qa" || isValidQaQuestion(question);
  const canAsk =
    Boolean(question.trim()) &&
    qaQuestionValid &&
    !busy &&
    Boolean(sessionId) &&
    researchReady &&
    mkacAuthorized;
  const pendingTotalSize = useMemo(
    () => pendingFiles.reduce((total, file) => total + file.size, 0),
    [pendingFiles],
  );
  const [agentMotion, setAgentMotion] = useState(true);
  const [selectedTask, setSelectedTask] = useState(null);
  const [taskInfoOpen, setTaskInfoOpen] = useState(false);
  const [composerOpen, setComposerOpen] = useState(false);
  const [taskPreviewDraft, setTaskPreviewDraft] = useState("");
  const [freeComposer, setFreeComposer] = useState(false);
  const composerDraftRef = useRef("");
  const taskGroups = useMemo(
    () => buildTaskGroups(mode, starterPromptEntries(), language),
    [mode, language, quickAnswersConfig, researchTopics, researchTopicId, researchScope],
  );
  useEffect(() => {
    setSelectedTask(null);
    setTaskInfoOpen(false);
    setComposerOpen(false);
    setQuestion("");
    setTaskPreviewDraft("");
    setFreeComposer(false);
    composerDraftRef.current = "";
  }, [mode, language, currentWorkspaceKey, researchTopicId, researchScope]);
  useEffect(() => {
    if (!messages.length) return;
    setSelectedTask(null);
    setComposerOpen(true);
  }, [messages.length]);

  useEffect(() => {
    if (question) composerDraftRef.current = question;
  }, [question]);

  useEffect(() => {
    if (composerOpen) {
      requestAnimationFrame(() => textareaRef.current?.focus());
    }
  }, [composerOpen]);
  const latestAssistantMessage = [...messages]
    .reverse()
    .find((item) => item.role === "assistant");
  const agentPresence = resolveAgentPresence({
    busy, pendingId: pendingAssistantId, message: latestAssistantMessage,
    question, error, needsSelection: !researchReady,
  });
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
    if (mode === "qa") return getQaStarterPrompts(language);
    if (mode === "wms") return quickAnswersConfig.wms || [];
    if (mode === "cctvai") return quickAnswersConfig.cctvai || [];
    const prompts = mode === "research"
      ? researchScope === "topic"
        ? researchQuickPrompts()
        : quickPromptsFor("research", language)
      : quickPromptsFor(mode, language);
    return prompts.map((prompt) => ({ question: prompt }));
  }

  const getSuggestions = (text, currentMode, msgId) =>
    resolveSuggestions(text, currentMode, msgId, quickAnswersConfig);

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
        const results = await Promise.allSettled([
          api("/health").then((response) => response.json()),
          api(`/models?language=${encodeURIComponent(language)}`).then((response) => response.json()),
          api("/knowledge/mkac/status").then((response) => response.json()),
          api("/research/demo").then((response) => response.json()),
          api("/research/topics").then((response) => response.json()),
        ]);
        const [healthResult, modelResult, mkacResult, demoResult, topicsResult] = results;
        const healthData = healthResult.status === "fulfilled" ? healthResult.value : null;
        const modelData = modelResult.status === "fulfilled" ? modelResult.value : null;
        const mkacData = mkacResult.status === "fulfilled" ? mkacResult.value : null;
        const researchDemoData = demoResult.status === "fulfilled" ? demoResult.value : null;
        const researchTopicsData = topicsResult.status === "fulfilled" ? topicsResult.value : null;

        if (healthData) {
          setMesStatus(healthData.mes_database || {});
          setWmsStatus(healthData.mes_wms_database || {});
          setCctvaiStatus(healthData.cctvai_database || {});
          setCctvaiHardwareStatus(healthData.cctvai_hardware || {});
          setQaStatus(healthData.qa || { enabled: false, configured: false, state: "UNKNOWN" });
          setHealth("online");
        } else {
          setHealth("offline");
        }
        if (modelData) {
          setModels(modelData.models || []);
          setModel((current) => {
            const nextDefault = modelData.default || "auto";
            return current === "auto" ? nextDefault : current;
          });
        }
        if (mkacData) setMkacStatus(mkacData);
        if (researchDemoData) setResearchDemo(researchDemoData);
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
    setQuickAnswersConfig({ mkac: [], mes: [], wms: [], cctvai: [], threshold: {}, max: 3 });

    async function loadQuickAnswers() {
      try {
        const [qaMkacRes, qaMesRes, qaWmsRes, qaCctvaiRes] = await Promise.all([
          api(`/quick-answers?mode=mkac&language=${encodeURIComponent(language)}`),
          api(`/quick-answers?mode=mes&language=${encodeURIComponent(language)}`),
          api(`/quick-answers?mode=wms&language=${encodeURIComponent(language)}`),
          api(`/quick-answers?mode=cctvai&language=${encodeURIComponent(language)}`),
        ]);
        const qaMkacData = await qaMkacRes.json();
        const qaMesData = await qaMesRes.json();
        const qaWmsData = await qaWmsRes.json();
        const qaCctvaiData = await qaCctvaiRes.json();
        if (cancelled) return;
        // Mỗi mode có ngưỡng riêng: cctvai luôn gắn thêm disclaimer freshness cố
        // định vào cuối câu trả lời nên cần ngưỡng cao hơn mkac/mes/wms.
        setQuickAnswersConfig({
          mkac: qaMkacData.suggestions || [],
          mes: qaMesData.suggestions || [],
          wms: qaWmsData.suggestions || [],
          cctvai: qaCctvaiData.suggestions || [],
          threshold: {
            mkac: qaMkacData.short_answer_threshold || DEFAULT_SHORT_ANSWER_THRESHOLD,
            mes: qaMesData.short_answer_threshold || DEFAULT_SHORT_ANSWER_THRESHOLD,
            wms: qaWmsData.short_answer_threshold || DEFAULT_SHORT_ANSWER_THRESHOLD,
            cctvai: qaCctvaiData.short_answer_threshold || DEFAULT_SHORT_ANSWER_THRESHOLD,
          },
          max: qaMkacData.max_suggestions || 3,
        });
      } catch {
        if (cancelled) return;
        setQuickAnswersConfig({ mkac: [], mes: [], wms: [], cctvai: [], threshold: {}, max: 3 });
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
        ?.setAttribute("content", resolvedTheme === "dark" ? "#0b1220" : "#f7f9fc");
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
    if (mode === "cctvai" && !cctvaiAvailable) {
      setMode("mkac");
    }
  }, [mode, wmsStatus.available, cctvaiAvailable]);

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
    if (workspaceKey(workspaceMode, uiLanguage) === currentWorkspaceKey) {
      setSelectedTask(null);
      setComposerOpen(false);
      setTaskPreviewDraft("");
      setQuestion("");
      composerDraftRef.current = "";
    }
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
    setSelectedTask(null);
    setTaskInfoOpen(false);
    setComposerOpen(false);
    setQuestion("");
    composerDraftRef.current = "";
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
    setSelectedTask(null);
    setTaskInfoOpen(false);
    setComposerOpen(false);
    setQuestion("");
    composerDraftRef.current = "";
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
    setSelectedTask(null);
    setTaskInfoOpen(false);
    setComposerOpen(false);
    setQuestion("");
    composerDraftRef.current = "";
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
    if (mode === "msm") {
      setMsmResetEpoch((e) => e + 1);
      setMsmTurnCount(0);
    }
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

  function openEmailDialog({ artifactId, language: lang } = {}) {
    if (busy) return;
    setEmailDialog({ open: true, artifactId: artifactId ?? null, language: lang ?? language });
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
    composerDraftRef.current = "";
    setSelectedTask(null);
    setTaskInfoOpen(false);
    setComposerOpen(false);
    setTaskPreviewDraft("");
    setFreeComposer(false);
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
    if (["mkac", "mes", "wms", "cctvai", "qa"].includes(mode) && !mkacAuthorized) {
      setEmployeeCodeError(t("common.employeeRequired"));
      return;
    }
    if (mode === "qa" && !isValidQaQuestion(cleanQuestion)) {
      setError(t("common.qaQuestionLimit", { limit: QA_MAX_CODE_POINTS }));
      return;
    }
    if (!cleanQuestion || busy || !sessionId) return;

    const requestMode = mode;
    const requestLanguage = language;
    const requestWorkspaceKey = workspaceKey(requestMode, requestLanguage);
    const requestSessionId = sessionId;
    const conversationContext = requestMode === "qa" ? [] : buildConversationContext(messages);
    if (messages.length === 0) {
      const title = sessionTitleFromQuestion(cleanQuestion);
      setSessionTitles((current) => ({ ...current, [requestWorkspaceKey]: title }));
      persistSessionTitle(requestSessionId, title);
    }
    const assistantId = createClientId();
    const requestQuestion = cleanQuestion;
    const controller = new AbortController();
    generationControllerRef.current = controller;
    if (requestMode !== "qa") pushStoredPromptHistory(cleanQuestion);
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
          requestMode === "qa"
            ? "qa_agent"
            : requestMode === "mkac"
              ? "mkac"
              : requestMode === "mes"
                ? "mes_database"
                : requestMode === "wms"
                  ? "wms_database"
                  : requestMode === "cctvai"
                    ? "cctvai_database"
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
    const requestPayload = requestMode === "qa"
      ? buildQaPayload({
          sessionId: requestSessionId,
          question: cleanQuestion,
          employeeId: employee?.id,
          language: requestLanguage,
          // An empty QA screen (new chat, clear, reload) starts a new agent
          // thread instead of continuing the one kept by the server.
          resetContext: messages.length === 0,
        })
      : {
          session_id: requestSessionId,
          question: cleanQuestion,
          stream: true,
          model: requestModel,
          mode: requestMode,
          ui_language: requestLanguage,
          employee_id: ["mkac", "mes", "wms", "cctvai"].includes(requestMode)
            ? employee?.id
            : undefined,
          quick_answer_id: requestMode === "wms" ? quickAnswerId || null : null,
          conversation_context: conversationContext,
          research_topic:
            requestMode === "research" && researchScope === "topic"
              ? researchTopicId || null
              : null,
          research_scope: requestMode === "research" ? researchScope : null,
        };

    try {
      await streamQuery(
        requestPayload,
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
            ["mes_report", "wms_executive_report", "hr_executive_report", "cctvai_report"].includes(event.artifact_type)
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
                        cctvaiMetadata: event.cctvai_metadata || item.cctvaiMetadata,
                        qaMetadata: event.qa_metadata || item.qaMetadata,
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
      const isCurrentRequest = generationControllerRef.current === controller;
      if (!wasStopped && isCurrentRequest) setError(localizedMessage);
      if (requestMode === "qa" && isCurrentRequest) setQuestion(requestQuestion);
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
      if (generationControllerRef.current === controller) {
        generationControllerRef.current = null;
        setPendingAssistantId("");
        setBusy(false);
      }
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
    for (const workspaceMode of ["mkac", "mes", "wms", "cctvai", "qa"]) {
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

  function taskInfoItems() {
    if (mode === "mkac") return [`${mkacStatus.num_documents} ${modeText("mkac").metric}`];
    if (mode === "mes") return [`${mesStatus.lots || 0} ${modeText("mes").metric}`];
    if (mode === "wms") return [
      `${formatText(modeText("wms").itemCount, { count: wmsStatus.distinct_items || 0 })} · ${formatText(modeText("wms").processCount, { count: wmsStatus.distinct_process_codes || 0 })}`,
      t(`common.wmsHealth.${wmsStatus.state || (wmsStatus.enabled ? "UNAVAILABLE" : "DISABLED")}`),
    ];
    if (mode === "cctvai") return [
      cctvaiStatus.available ? modeText("cctvai").metric : modeText("cctvai").unavailable,
      ...(!cctvaiHardwareStatus.available && cctvaiHardwareStatus.enabled ? [modeText("cctvai").hardwareUnavailable] : []),
    ];
    if (mode === "qa") return [
      qaStatus.configured ? (qaStatus.state === "ONLINE" ? modeText("qa").metric : modeText("qa").unavailable) : modeText("qa").misconfigured,
    ];
    if (researchScope === "upload") return [`${files.length} ${modeText("research").metric}`];
    return researchTopicId
      ? [`${selectedResearchTopic()?.num_files ?? 0} ${modeText("research").metric}`]
      : [formatText(modeText("research").documentsSummary, { count: researchTopicsTotalFiles(), topics: researchTopics.topics.length })];
  }

  // Card ↔ preview swaps morph with the View Transitions API when motion is
  // allowed; otherwise the state changes instantly.
  function withTaskTransition(update) {
    runViewTransition(() => flushSync(update), {
      enabled: canAnimate({ motion: agentMotion, reduced: prefersReducedMotion() }),
    });
  }

  function chooseTask(task) {
    if (busy || uploading || !mkacAuthorized || !researchReady) return;
    withTaskTransition(() => {
      setSelectedTask(task);
      setTaskPreviewDraft(task.question);
      setComposerOpen(false);
    });
  }

  function closeTaskPreview() {
    withTaskTransition(() => {
      setSelectedTask(null);
      setTaskPreviewDraft("");
    });
  }

  function runSelectedTask(task, editedQuestion) {
    if (!task || busy || uploading || !mkacAuthorized || !researchReady) return;
    const { question: questionToRun, valid, usePreparedAnswer, quickAnswerId } = prepareTaskRun(task, editedQuestion);
    if (!valid || (mode === "qa" && !isValidQaQuestion(questionToRun))) {
      setError(mode === "qa"
        ? t("common.qaQuestionLimit", { limit: QA_MAX_CODE_POINTS })
        : language === "ja" ? "質問を1行で入力してください。" : "Vui lòng nhập câu hỏi trên một dòng.");
      return;
    }
    setSelectedTask(null);
    setTaskPreviewDraft("");
    if (mode === "wms" && usePreparedAnswer) {
      handleQuickAnswerClick({ ...task, id: quickAnswerId, question: questionToRun });
      return;
    }
    sendMessage(questionToRun);
  }

  function openFreeComposer() {
    if (busy || uploading || !mkacAuthorized) return;
    setSelectedTask(null);
    setFreeComposer(true);
    setQuestion(composerDraftRef.current);
    setComposerOpen(true);
  }

  function closeFreeComposer() {
    if (!composerOpen || messages.length > 0) return;
    composerDraftRef.current = question;
    setComposerOpen(false);
    setFreeComposer(false);
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
      <IntroSplash language={language} health={health} />
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
        <header className={`workspace-header ${messages.length === 0 ? "is-welcome" : ""}`}>
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
                      : mode === "cctvai"
                        ? cctvaiStatus.available
                          ? modeText("cctvai").metric
                          : modeText("cctvai").unavailable
                        : mode === "qa"
                          ? modeText("qa").metric
                          : mode === "msm"
                            ? modeText("msm").metric
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
            <div className={`mode-selector-wrap ${mode}`}>
              <ModeIcon className="mode-selector-icon" size={17} aria-hidden="true" />
              <select
                className="mode-selector"
                value={mode}
                onChange={(event) => switchMode(event.target.value)}
                disabled={busy || uploading}
                aria-label={t("common.qaModes")}
                title={currentMode.label}
              >
                {visibleModes.map((key) => (
                  <option key={key} value={key}>
                    {modeText(key).label}
                  </option>
                ))}
              </select>
              <ChevronDown className="mode-selector-chevron" size={15} aria-hidden="true" />
            </div>

            <MotionToggle motion={agentMotion} onMotionChange={setAgentMotion} language={language} />

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
                <span className="panel-count" key={latestSources.length}>{latestSources.length}</span>
              )}
            </button>

            <span className="header-divider" role="separator" aria-hidden="true" />

            {["mkac", "mes", "wms", "cctvai", "qa"].includes(mode) && employee && (
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
              disabled={busy || (mode === "msm" ? msmTurnCount === 0 : messages.length === 0)}
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

        {mode === "cctvai" && cctvaiBannerVisibility.showDatabaseBanner && (
          <div className={`wms-health-banner ${String(
            cctvaiStatus.state || (cctvaiStatus.enabled ? "UNAVAILABLE" : "DISABLED"),
          ).toLowerCase()}`} role="status">
            {modeText("cctvai").unavailable}
          </div>
        )}

        {mode === "cctvai" && cctvaiBannerVisibility.showHardwareBanner && (
          <div className={`wms-health-banner ${String(
            cctvaiHardwareStatus.state ||
              (cctvaiHardwareStatus.enabled ? "UNAVAILABLE" : "DISABLED"),
          ).toLowerCase()}`} role="status">
            {modeText("cctvai").hardwareUnavailable}
          </div>
        )}

        {mode === "qa" && qaStatus.enabled && !qaStatus.configured && (
          <div className="wms-health-banner disabled" role="status">
            {modeText("qa").misconfigured}
          </div>
        )}
        {mode === "qa" && qaStatus.enabled && qaStatus.configured && qaStatus.state !== "ONLINE" && (
          <div className="wms-health-banner unavailable" role="status">
            {modeText("qa").unavailable}
          </div>
        )}

        <div className="workspace-body">
          {mode === "msm" ? (
            <section
              id="msm-conversation"
              className={`conversation agent-workspace ${msmTurnCount > 0 ? "has-conversation" : "is-welcome"}`}
              data-motion={agentMotion}
              data-agent-state={agentPresence.state}
              role="tabpanel"
              aria-label={currentMode.title}
            >
              <AgentStatus presence={agentPresence} language={language} />
              <RoamingRobot
                state={agentPresence.state}
                motion={agentMotion}
                hasConversation={msmTurnCount > 0}
                hasError={false}
                previewOpen={false}
              />
              <MsmMode
                key={msmResetEpoch}
                employeeId={employee?.id}
                language={language}
                presence={agentPresence}
                onTurnCountChange={setMsmTurnCount}
              />
            </section>
          ) : (
          <>
          <section
            id={`${mode}-conversation`}
            className={`conversation agent-workspace ${messages.length ? "has-conversation" : "is-welcome"}`}
            data-motion={agentMotion}
            data-agent-state={agentPresence.state}
            role="tabpanel"
            aria-label={currentMode.title}
            aria-busy={busy}
          >
            <AgentStatus presence={agentPresence} language={language} />
            {mkacAuthorized && (
              <RoamingRobot
                state={agentPresence.state}
                motion={agentMotion}
                hasConversation={messages.length > 0}
                hasError={Boolean(error)}
                previewOpen={Boolean(selectedTask)}
              />
            )}
            <div className="conversation-scroll" ref={conversationScrollRef}>
              {["mkac", "mes", "wms", "cctvai", "qa"].includes(mode) && !mkacAuthorized ? (
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
                    <AgentPresence presence={agentPresence} mode={mode} language={language} />
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
                    <TaskLauncher
                      mode={mode}
                      language={language}
                      groups={taskGroups}
                      selectedTask={selectedTask}
                      previewQuestion={taskPreviewDraft}
                      onPreviewChange={setTaskPreviewDraft}
                      onSelect={chooseTask}
                      onRun={runSelectedTask}
                      onBack={closeTaskPreview}
                      onFreeInput={openFreeComposer}
                      onBackToTasks={closeFreeComposer}
                      showFreeInput={freeComposer}
                      onToggleInfo={() => setTaskInfoOpen((open) => !open)}
                      showInfo={taskInfoOpen}
                      qaGuidance={mode === "qa" ? modeText("qa").guidance : ""}
                      modeDescription={modeText(mode).empty}
                      researchContext={mode === "research" && researchScope === "topic" && researchTopicId
                        ? formatText(modeText("research").scopeLabel, { topic: researchTopicLabel(selectedResearchTopic()) })
                        : mode === "research" && researchScope === "upload" ? `${files.length} ${modeText("research").metric}` : ""}
                      infoItems={taskInfoItems()}
                    />
                  ) : null}
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
                  onReportEmail={openEmailDialog}
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
              researchReady={researchReady}
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
              isOpen={messages.length > 0 || composerOpen}
            />
            )}
          </section>
          </>
          )}

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

      {emailDialog.open && (
        <EmailDialog
          language={emailDialog.language}
          onClose={() => setEmailDialog({ open: false, artifactId: null, language: "vi" })}
          onSend={(toEmail) => {
            setEmailDialog({ open: false, artifactId: null, language: "vi" });
            const cmd = emailDialog.language === "ja"
              ? `このレポートを メールで送信: ${toEmail}`
              : `Gửi báo cáo này cho ${toEmail}`;
            sendMessage(cmd);
          }}
        />
      )}
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
