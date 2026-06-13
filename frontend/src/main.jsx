import { StrictMode, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
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
  mkac: [
    "Meiko Automation có bao nhiêu phòng ban, gồm các phòng ban nào?",
    "Quy định làm thêm giờ tại MKAC như thế nào?",
    "Các sản phẩm chính của MKAC là gì?",
  ],
  research: [
    "Lập báo cáo nghiên cứu tổng hợp từ các tài liệu",
    "So sánh các quan điểm và chỉ ra điểm mâu thuẫn",
    "Đề xuất các câu hỏi nghiên cứu tiếp theo",
  ],
};

const MODE_OPTIONS = {
  mkac: {
    label: "Hỏi đáp MKAC",
    title: "Hỏi đáp về MKAC",
    icon: Database,
  },
  research: {
    label: "Nghiên cứu",
    title: "Nghiên cứu tài liệu",
    icon: FlaskConical,
  },
};

const MODEL_ACCENTS = {
  auto: "accent-auto",
  local: "accent-local",
  openai: "accent-openai",
  grok: "accent-grok",
};

const WAITING_MESSAGES = [
  "Đã nhận câu hỏi, đang suy luận...",
  "Tôi hiểu rồi, bạn chờ một chút nhé...",
  "Đang đối chiếu các nguồn phù hợp...",
  "Đang tổng hợp câu trả lời...",
  "Sắp có kết quả rồi...",
];

const SESSION_STORAGE_KEYS = {
  mkac: "vllm-pd-session-mkac",
  research: "vllm-pd-session-research",
};
const LEGACY_SESSION_STORAGE_KEY = "vllm-pd-session";
const SESSION_TITLE_STORAGE_KEY = "vllm-pd-session-titles";
const THEME_STORAGE_KEY = "vllm-pd-theme";
const EMPLOYEE_STORAGE_KEY = "vllm-pd-mkac-employee";
const THEME_OPTIONS = ["system", "light", "dark"];
const THEME_META = {
  system: {
    label: "Theo hệ thống",
    icon: Monitor,
  },
  light: {
    label: "Sáng",
    icon: Sun,
  },
  dark: {
    label: "Tối",
    icon: Moon,
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

function storedEmployee() {
  try {
    const value = JSON.parse(localStorage.getItem(EMPLOYEE_STORAGE_KEY) || "null");
    return value?.id && value?.name ? value : null;
  } catch {
    return null;
  }
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

function defaultSessionTitle(workspaceMode) {
  return workspaceMode === "research"
    ? "Phiên nghiên cứu mới"
    : "Hỏi đáp MKAC mới";
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

function App() {
  const [theme, setTheme] = useState(storedTheme);
  const [sessionIds, setSessionIds] = useState({ mkac: "", research: "" });
  const [sessionTitles, setSessionTitles] = useState({
    mkac: defaultSessionTitle("mkac"),
    research: defaultSessionTitle("research"),
  });
  const [files, setFiles] = useState([]);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [uploadSummary, setUploadSummary] = useState(null);
  const [messagesByMode, setMessagesByMode] = useState({
    mkac: [],
    research: [],
  });
  const [models, setModels] = useState([]);
  const [mkacStatus, setMkacStatus] = useState({
    ready: false,
    num_documents: 0,
    num_chunks: 0,
    files: [],
  });
  const [model, setModel] = useState("openai");
  const [mode, setMode] = useState("mkac");
  const [question, setQuestion] = useState("");
  const [sourcesByMode, setSourcesByMode] = useState({
    mkac: [],
    research: [],
  });
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

  const selectedModel = useMemo(
    () => models.find((item) => item.id === (mode === "research" ? "grok" : model)),
    [models, mode, model],
  );
  const requestModel = mode === "research" ? "grok" : model;
  const mkacModels = useMemo(
    () => models.filter((item) => !item.hidden_in_mkac && item.id !== "grok"),
    [models],
  );

  const currentMode = MODE_OPTIONS[mode];
  const ModeIcon = currentMode.icon;
  const ThemeIcon = THEME_META[theme].icon;
  const sessionId = sessionIds[mode];
  const messages = messagesByMode[mode];
  const sources = sourcesByMode[mode];
  const researchReady = mode !== "research" || files.length > 0;
  const mkacAuthorized = mode !== "mkac" || Boolean(employee?.id && employee?.name);
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
        const [healthResponse, modelResponse, mkacResponse] = await Promise.all([
          api("/health"),
          api("/models"),
          api("/knowledge/mkac/status"),
        ]);
        await healthResponse.json();
        const modelData = await modelResponse.json();
        const mkacData = await mkacResponse.json();
        setModels(modelData.models || []);
        setMkacStatus(mkacData);
        setModel((current) => {
          const nextDefault = modelData.default || "openai";
          return current === "auto" || current === "grok" ? nextDefault : current;
        });
        setHealth("online");

        const legacySession = localStorage.getItem(LEGACY_SESSION_STORAGE_KEY);
        const storedSessions = {
          mkac: localStorage.getItem(SESSION_STORAGE_KEYS.mkac),
          research:
            localStorage.getItem(SESSION_STORAGE_KEYS.research) || legacySession,
        };
        const resolvedSessions = {};

        await Promise.all(
          Object.keys(MODE_OPTIONS).map(async (workspaceMode) => {
            const storedSession = storedSessions[workspaceMode];
            if (storedSession) {
              try {
                const infoResponse = await api(`/sessions/${storedSession}`);
                const info = await infoResponse.json();
                resolvedSessions[workspaceMode] = storedSession;
                if (workspaceMode === "research") {
                  setFiles(info.files || []);
                }
                return;
              } catch (sessionError) {
                if (sessionError.status === 404) {
                  resolvedSessions[workspaceMode] = storedSession;
                  return;
                }
                localStorage.removeItem(SESSION_STORAGE_KEYS[workspaceMode]);
              }
            }

            const session = await createSession();
            resolvedSessions[workspaceMode] = session.session_id;
          }),
        );

        setSessionIds(resolvedSessions);
        const savedTitles = storedSessionTitles();
        setSessionTitles({
          mkac:
            savedTitles[resolvedSessions.mkac] || defaultSessionTitle("mkac"),
          research:
            savedTitles[resolvedSessions.research] ||
            defaultSessionTitle("research"),
        });
        Object.entries(resolvedSessions).forEach(([workspaceMode, id]) => {
          localStorage.setItem(SESSION_STORAGE_KEYS[workspaceMode], id);
        });
        localStorage.removeItem(LEGACY_SESSION_STORAGE_KEY);
      } catch (bootstrapError) {
        setHealth("offline");
        setError(bootstrapError.message);
      }
    }
    bootstrap();
  }, []);

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
        (current) => (current + 1) % WAITING_MESSAGES.length,
      );
    }, 2400);

    return () => window.clearInterval(timer);
  }, [busy, pendingAssistantId]);

  function setModeMessages(workspaceMode, updater) {
    setMessagesByMode((current) => ({
      ...current,
      [workspaceMode]:
        typeof updater === "function"
          ? updater(current[workspaceMode])
          : updater,
    }));
  }

  function setModeSources(workspaceMode, updater) {
    setSourcesByMode((current) => ({
      ...current,
      [workspaceMode]:
        typeof updater === "function"
          ? updater(current[workspaceMode])
          : updater,
    }));
  }

  async function resetSession(workspaceMode = mode) {
    setError("");
    const data = await createSession();
    setSessionIds((current) => ({
      ...current,
      [workspaceMode]: data.session_id,
    }));
    localStorage.setItem(SESSION_STORAGE_KEYS[workspaceMode], data.session_id);
    setSessionTitles((current) => ({
      ...current,
      [workspaceMode]: defaultSessionTitle(workspaceMode),
    }));
    if (workspaceMode === "research") {
      setFiles([]);
      setPendingFiles([]);
      setUploadSummary(null);
    }
    setModeMessages(workspaceMode, []);
    setModeSources(workspaceMode, []);
    setSidebarOpen(false);
  }

  function switchMode(nextMode) {
    if (nextMode === mode || busy || uploading) return;
    setMode(nextMode);
    setQuestion("");
    setError("");
    setSidebarOpen(false);
    setSourcePanelOpen(false);
    setPendingAssistantId("");
  }

  function clearConversation() {
    setModeMessages(mode, []);
    setModeSources(mode, []);
    setQuestion("");
    setError("");
    setSourcePanelOpen(false);
  }

  function onModeTabKeyDown(event, currentModeKey) {
    const modeKeys = Object.keys(MODE_OPTIONS);
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
      setError("Không thể sao chép câu trả lời.");
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
      setError(uploadError.message);
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
      setError(removeError.message);
    }
  }

  async function sendMessage(prompt = question) {
    const cleanQuestion = prompt.trim();
    if (mode === "mkac" && !mkacAuthorized) {
      setEmployeeCodeError("Vui lòng nhập mã nhân viên hợp lệ trước khi hỏi MKAC.");
      return;
    }
    if (!cleanQuestion || busy || !sessionId) return;

    const requestMode = mode;
    const requestSessionId = sessionId;
    if (messages.length === 0) {
      const title = sessionTitleFromQuestion(cleanQuestion);
      setSessionTitles((current) => ({ ...current, [requestMode]: title }));
      persistSessionTitle(requestSessionId, title);
    }
    const assistantId = crypto.randomUUID();
    const controller = new AbortController();
    generationControllerRef.current = controller;
    setQuestion("");
    setError("");
    setModeSources(requestMode, []);
    setSourcePanelOpen(false);
    setModeMessages(requestMode, (current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", content: cleanQuestion },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        model: selectedModel?.name || model,
        mode: requestMode,
        answerScope: requestMode === "mkac" ? "mkac" : "research",
        sources: [],
      },
    ]);
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
          employee_id: requestMode === "mkac" ? employee?.id : undefined,
        },
        (event) => {
          if (event.type === "sources") {
            setModeSources(requestMode, event.sources || []);
            setModeMessages(requestMode, (current) =>
              current.map((item) =>
                item.id === assistantId
                  ? { ...item, sources: event.sources || [] }
                  : item,
              ),
            );
          }
          if (event.type === "meta") {
            setModeMessages(requestMode, (current) =>
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
            );
          }
          if (event.type === "token") {
            setPendingAssistantId("");
            setModeMessages(requestMode, (current) =>
              current.map((item) =>
                item.id === assistantId
                  ? { ...item, content: item.content + (event.content || "") }
                  : item,
              ),
            );
          }
          if (event.type === "error") {
            throw new Error(event.message || "Không thể tạo phản hồi.");
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
      if (!wasStopped) setError(queryError.message);
      setModeMessages(requestMode, (current) =>
        current.map((item) =>
          item.id === assistantId
            ? {
                ...item,
                content:
                  item.content ||
                  (wasStopped
                    ? "Đã dừng phản hồi."
                    : `Không thể hoàn tất yêu cầu: ${queryError.message}`),
                stopped: wasStopped,
              }
            : item,
        ),
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
      setEmployeeCodeError("Mã nhân viên không hợp lệ.");
      return;
    }

    setEmployeeVerifying(true);
    setEmployeeCodeError("");
    try {
      const data = await authenticateEmployee(normalizedCode);
      setEmployee(data.employee);
      try {
        localStorage.setItem(EMPLOYEE_STORAGE_KEY, JSON.stringify(data.employee));
      } catch {
        // localStorage can be blocked in private browsing; the in-memory state still works.
      }
    } catch {
      setEmployee(null);
      setEmployeeCodeError("Mã nhân viên không hợp lệ.");
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
    setModeMessages("mkac", []);
    setModeSources("mkac", []);
    try {
      localStorage.removeItem(EMPLOYEE_STORAGE_KEY);
      localStorage.removeItem("vllm-pd-mkac-employee-code");
    } catch {
      // Ignore storage failures.
    }
  }

  function onSubmit(event) {
    event.preventDefault();
    sendMessage();
  }

  function onDrop(event) {
    event.preventDefault();
    setDragActive(false);
    addPendingFiles(event.dataTransfer.files);
  }

  return (
    <div className={`app-shell ${mode === "mkac" ? "mkac-layout" : ""}`}>
      {mode === "research" && (
        <>
          <button
            className="mobile-menu icon-button"
            type="button"
            title="Mở tài liệu"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={20} />
          </button>

          {sidebarOpen && (
            <button
              className="sidebar-backdrop"
              type="button"
              aria-label="Đóng tài liệu"
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
                <span>Trợ lý hỏi đáp nội bộ</span>
              </div>
              <button
                className="icon-button close-sidebar"
                type="button"
                title="Đóng"
                onClick={() => setSidebarOpen(false)}
              >
                <X size={18} />
              </button>
            </div>

            <div className="session-strip">
              <div>
                <span>Phiên làm việc</span>
                <strong className="session-title" title={sessionTitles.research}>
                  {sessionTitles.research}
                </strong>
              </div>
              <button
                className="icon-button"
                type="button"
                title="Tạo phiên mới"
                onClick={() => resetSession("research")}
                disabled={busy || uploading}
              >
                <RefreshCcw size={17} />
              </button>
            </div>

            <section className="sidebar-section">
              <div className="section-heading">
                <span>Tài liệu nghiên cứu</span>
                <span className="count-badge">{files.length}</span>
              </div>

              <input
                ref={fileInputRef}
                className="hidden-input"
                type="file"
                multiple
                accept=".pdf,.docx,.xlsx,.pptx,.html,.png,.jpg,.jpeg"
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
                <span>Chọn tài liệu</span>
                <small>PDF, Office, HTML, PNG/JPG</small>
              </button>

              {pendingFiles.length > 0 && (
                <div className="pending-panel">
                  <div className="pending-header">
                    <span>{pendingFiles.length} tệp</span>
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
                          title={`Bỏ ${file.name}`}
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
                      ? `Đang index ${uploadProgress.done}/${uploadProgress.total}`
                      : "Index tài liệu"}
                  </button>
                  {uploading && (
                    <div className="upload-progress" role="status" aria-live="polite">
                      <div className="upload-progress-copy">
                        <span>Đang xử lý</span>
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
                    Đã index {uploadSummary.files} tệp, {uploadSummary.chunks} đoạn
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
                      title={`Xóa ${filename}`}
                      onClick={() => removeFile(filename)}
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                ))}
                {files.length === 0 && (
                  <div className="empty-files">
                    <FileText size={20} />
                    <span>Chưa có tài liệu</span>
                  </div>
                )}
              </div>
            </section>

            <div className="sidebar-footer">
              <div className={`service-state ${health}`}>
                {health === "online" ? <CheckCircle2 size={15} /> : <Server size={15} />}
                <span>{health === "online" ? "Máy 2 online" : "Đang kiểm tra"}</span>
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
            <div className={`mode-mark ${mode === "mkac" ? "logo-mark" : mode}`}>
              {mode === "mkac" ? (
                <img src="/mkac-logo.png" alt="MKAC" />
              ) : (
                <ModeIcon size={20} />
              )}
            </div>
            <div>
              <strong>{currentMode.title}</strong>
              <span>
                {mode === "mkac"
                  ? employee?.name
                    ? employee.greeting || `Xin chào, ${employee.name}`
                    : `${mkacStatus.num_documents} tài liệu nội bộ`
                  : `${files.length} tài liệu trong phiên`}
              </span>
            </div>
          </div>

          <div className="header-actions">
            <div className="mode-tabs" role="tablist" aria-label="Chế độ hỏi đáp">
              {Object.entries(MODE_OPTIONS).map(([key, option]) => {
                const Icon = option.icon;
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
                  >
                    <Icon size={17} />
                    {option.label}
                  </button>
                );
              })}
            </div>

            {mode === "mkac" ? (
              <div
                ref={modelSelectRef}
                className={`model-select ${MODEL_ACCENTS[model] || ""}`}
              >
                <Bot size={17} />
                <button
                  className="model-select-trigger"
                  type="button"
                  aria-label="Chọn mô hình"
                  aria-haspopup="listbox"
                  aria-expanded={modelMenuOpen}
                  onClick={() => setModelMenuOpen((open) => !open)}
                >
                  <span>{selectedModel?.name || "Đang tải model"}</span>
                  <ChevronDown
                    className={modelMenuOpen ? "model-chevron open" : "model-chevron"}
                    size={15}
                  />
                </button>
                {modelMenuOpen && (
                  <div className="model-menu" role="listbox" aria-label="Danh sách mô hình">
                    {mkacModels.map((item) => (
                      <button
                        key={item.id}
                        className={model === item.id ? "selected" : ""}
                        type="button"
                        role="option"
                        aria-selected={model === item.id}
                        onClick={() => {
                          setModel(item.id);
                          setModelMenuOpen(false);
                        }}
                      >
                        <span className={`model-dot ${MODEL_ACCENTS[item.id] || ""}`} />
                        <span className="model-option-copy">
                          <strong>{item.name}</strong>
                          <small>{item.description}</small>
                        </span>
                        {model === item.id && <Check size={16} />}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="model-select locked accent-grok" title="Chế độ nghiên cứu luôn dùng Grok">
                <Bot size={17} />
                <span className="model-select-trigger">
                  <span>{selectedModel?.name || "Grok 4.20 Reasoning"}</span>
                </span>
              </div>
            )}

            <button
              className="icon-button header-tool theme-toggle"
              type="button"
              title={`Giao diện: ${THEME_META[theme].label}. Nhấn để chuyển chế độ.`}
              aria-label={`Giao diện hiện tại: ${THEME_META[theme].label}`}
              onClick={cycleTheme}
            >
              <ThemeIcon size={18} />
            </button>

            {mode === "mkac" && employee && (
              <button
                className="icon-button header-tool"
                type="button"
                title="Đăng xuất mã nhân viên"
                aria-label="Đăng xuất mã nhân viên"
                onClick={logoutEmployee}
                disabled={busy}
              >
                <LogOut size={18} />
              </button>
            )}

            <button
              className="icon-button header-tool"
              type="button"
              title="Xóa hội thoại hiện tại"
              onClick={clearConversation}
              disabled={busy || messages.length === 0}
            >
              <Trash2 size={18} />
            </button>

            <button
              className="icon-button panel-toggle"
              type="button"
              title={sourcePanelOpen ? "Ẩn nguồn" : "Hiện nguồn"}
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
              {mode === "mkac" && !mkacAuthorized ? (
                <div className="employee-gate">
                  <form className="employee-card" onSubmit={verifyEmployeeCode}>
                    <div className="employee-logo">
                      <img src="/mkac-logo.png" alt="MKAC" />
                    </div>
                    <h1>Xác thực nhân viên MKAC</h1>
                    <p>
                      Nhập mã nhân viên để truy cập chế độ hỏi đáp nội bộ MKAC.
                    </p>
                    <label className="employee-field">
                      <span>Mã nhân viên</span>
                      <input
                        value={employeeCodeInput}
                        onChange={(event) =>
                          updateEmployeeCodeInput(event.target.value)
                        }
                        inputMode="numeric"
                        autoComplete="off"
                        maxLength={6}
                        placeholder="Mã nhân viên"
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
                      {employeeVerifying ? "Đang kiểm tra" : "Tiếp tục"}
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
                        ? employee.greeting || `Xin chào, ${employee.name}`
                        : currentMode.title}
                    </h1>
                    <p>
                      {mode === "mkac"
                        ? "Bạn có thể tra cứu các quy định và thông tin nội bộ MKAC."
                        : files.length > 0
                          ? "Sẵn sàng nghiên cứu tài liệu đã index trong phiên này."
                          : "Tải tài liệu lên để bắt đầu nghiên cứu."}
                    </p>
                    {mode === "research" && files.length === 0 && (
                      <button
                        className="empty-upload-button"
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                      >
                        <UploadCloud size={17} />
                        Chọn tài liệu để bắt đầu
                      </button>
                    )}
                  </div>

                  <div className="prompt-grid">
                    {QUICK_PROMPTS[mode].map((prompt) => (
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

                  <div className="empty-metrics">
                    <div>
                      <Layers3 size={16} />
                      <strong>
                        {mode === "mkac" ? mkacStatus.num_documents : files.length}
                      </strong>
                      <span>{mode === "mkac" ? "Tài liệu MKAC" : "Tài liệu"}</span>
                    </div>
                    <div>
                      <Bot size={16} />
                      <strong>{selectedModel?.name || "Đang tải"}</strong>
                      <span>Model</span>
                    </div>
                    <div>
                      <Activity size={16} />
                      <strong>{health === "online" ? "Online" : "Offline"}</strong>
                      <span>Trạng thái</span>
                    </div>
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
                            {message.stopped && <span>Đã dừng</span>}
                            <span>
                              {message.answerScope === "general"
                                ? "Không có kết quả"
                                : message.answerScope === "web"
                                  ? "Tìm kiếm web"
                                : message.mode === "research"
                                  ? "Nghiên cứu"
                                  : "Nguồn MKAC"}
                            </span>
                          </div>
                        )}
                        {message.content ? (
                          <ReactMarkdown>{message.content}</ReactMarkdown>
                        ) : message.id === pendingAssistantId ? (
                          <div
                            className="waiting-status"
                            role="status"
                            aria-live="polite"
                          >
                            <Loader2 className="spin" size={17} />
                            <span key={waitingMessageIndex}>
                              {WAITING_MESSAGES[waitingMessageIndex]}
                            </span>
                          </div>
                        ) : null}
                        {message.role === "assistant" && message.content && (
                          <div className="message-actions">
                            <details className="ai-disclosure">
                              <summary title="Thông tin phản hồi AI">
                                <Sparkles size={14} />
                                <span>AI</span>
                              </summary>
                              <div>
                                <strong>{message.model}</strong>
                                <span>
                                  {message.answerScope === "web"
                                    ? "Tổng hợp từ nguồn web"
                                    : message.answerScope === "research"
                                      ? "Dựa trên tài liệu nghiên cứu"
                                      : message.answerScope === "mkac"
                                        ? "Dựa trên kho MKAC"
                                        : "Không có nguồn đối chiếu"}
                                </span>
                              </div>
                            </details>
                            <button
                              className="message-action-button"
                              type="button"
                              title="Sao chép câu trả lời"
                              onClick={() => copyAnswer(message)}
                            >
                              {copiedMessageId === message.id ? (
                                <Check size={14} />
                              ) : (
                                <Copy size={14} />
                              )}
                              <span>
                                {copiedMessageId === message.id ? "Đã sao chép" : "Sao chép"}
                              </span>
                            </button>
                          </div>
                        )}
                        {message.role === "assistant" && message.sources?.length > 0 && (
                          <details className="message-sources">
                            <summary>{message.sources.length} nguồn tham chiếu</summary>
                            {message.sources.map((source, index) => (
                              <div key={`${source.file}-${source.page}-${index}`}>
                                {source.url ? (
                                  <a href={source.url} target="_blank" rel="noreferrer">
                                    <strong>{source.file}</strong>
                                  </a>
                                ) : (
                                  <strong>{source.file}</strong>
                                )}
                                <span>
                                  {source.url ? "Nguồn web" : `Trang ${source.page}`}
                                </span>
                              </div>
                            ))}
                          </details>
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
                  title="Đóng"
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
                    title="Thêm tài liệu nghiên cứu"
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
                      sendMessage();
                    }
                  }}
                  rows={2}
                  maxLength={4000}
                  aria-label={
                    mode === "research"
                      ? "Câu hỏi nghiên cứu"
                      : "Câu hỏi về MKAC"
                  }
                  placeholder={
                    mode === "research"
                      ? researchReady
                        ? "Nhập chủ đề nghiên cứu..."
                        : "Hãy tải tài liệu lên trước khi đặt câu hỏi..."
                      : "Đặt câu hỏi về MKAC..."
                  }
                  disabled={busy || !researchReady}
                />
                <div className="composer-footer">
                  <div className="composer-context">
                  <span className={`model-dot ${MODEL_ACCENTS[requestModel] || ""}`} />
                  <span>
                    {selectedModel?.description || "Đang tải danh sách model"}
                    </span>
                  </div>
                  <div className="composer-actions">
                    <span className="char-count">{question.length}/4000</span>
                    <button
                      className="send-button"
                      type={busy ? "button" : "submit"}
                      title={busy ? "Dừng trả lời" : "Gửi"}
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
            aria-label="Nguồn tham chiếu"
          >
            <div className="source-header">
              <span>Nguồn tham chiếu</span>
              <span>{latestSources.length}</span>
            </div>
            <div className="source-list">
              {latestSources.map((source, index) => (
                <article
                  className="source-item"
                  key={`${source.file}-${source.page}-${index}`}
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
                    <span>{source.url ? "Nguồn web" : `Trang ${source.page || "?"}`}</span>
                    <span>{Math.round((source.score || 0) * 100)}%</span>
                  </div>
                  <p>{source.preview}</p>
                </article>
              ))}
              {latestSources.length === 0 && (
                <div className="empty-sources">
                  <Search size={22} />
                  <span>Chưa có nguồn cho lượt trả lời này</span>
                </div>
              )}
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
