import { StrictMode, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import {
  Activity,
  AlertCircle,
  BookOpen,
  Bot,
  CheckCircle2,
  ChevronDown,
  Database,
  FileText,
  FileUp,
  FlaskConical,
  Layers3,
  Loader2,
  Menu,
  MessageSquare,
  PanelRightClose,
  PanelRightOpen,
  Paperclip,
  Plus,
  RefreshCcw,
  Search,
  Send,
  Server,
  ShieldCheck,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import "./styles.css";

const QUICK_PROMPTS = {
  chat: [
    "Tóm tắt tài liệu theo các ý chính",
    "Liệt kê các số liệu và kết luận quan trọng",
    "Thông tin nào trong tài liệu còn chưa rõ?",
  ],
  research: [
    "Lập báo cáo nghiên cứu tổng hợp từ các tài liệu",
    "So sánh các quan điểm và chỉ ra điểm mâu thuẫn",
    "Đề xuất các câu hỏi nghiên cứu tiếp theo",
  ],
};

const MODE_OPTIONS = {
  chat: {
    label: "Hỏi đáp",
    title: "Hỏi đáp tài liệu",
    icon: MessageSquare,
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
  mimo: "accent-mimo",
  openai: "accent-openai",
  grok: "accent-grok",
};

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

async function streamQuery(payload, onEvent) {
  const response = await api("/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
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

function shortSession(sessionId) {
  if (!sessionId) return "no-session";
  return sessionId.slice(0, 8);
}

function App() {
  const [sessionId, setSessionId] = useState("");
  const [files, setFiles] = useState([]);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [uploadSummary, setUploadSummary] = useState(null);
  const [messages, setMessages] = useState([]);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("auto");
  const [mode, setMode] = useState("chat");
  const [question, setQuestion] = useState("");
  const [sources, setSources] = useState([]);
  const [health, setHealth] = useState("checking");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({ done: 0, total: 0 });
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sourcePanelOpen, setSourcePanelOpen] = useState(true);
  const fileInputRef = useRef(null);
  const endRef = useRef(null);

  const selectedModel = useMemo(
    () => models.find((item) => item.id === model),
    [models, model],
  );

  const currentMode = MODE_OPTIONS[mode];
  const ModeIcon = currentMode.icon;
  const canAsk = Boolean(question.trim()) && !busy && Boolean(sessionId);
  const pendingTotalSize = useMemo(
    () => pendingFiles.reduce((total, file) => total + file.size, 0),
    [pendingFiles],
  );
  const latestSources = sources.length
    ? sources
    : [...messages].reverse().find((item) => item.sources?.length)?.sources || [];

  useEffect(() => {
    async function bootstrap() {
      try {
        const [healthResponse, modelResponse] = await Promise.all([
          api("/health"),
          api("/models"),
        ]);
        await healthResponse.json();
        const modelData = await modelResponse.json();
        setModels(modelData.models || []);
        setModel(modelData.default || "auto");
        setHealth("online");

        const stored = localStorage.getItem("vllm-pd-session");
        if (stored) {
          try {
            const infoResponse = await api(`/sessions/${stored}`);
            const info = await infoResponse.json();
            setSessionId(stored);
            setFiles(info.files || []);
            return;
          } catch (sessionError) {
            if (sessionError.status === 404) {
              setSessionId(stored);
              setFiles([]);
              return;
            }
            localStorage.removeItem("vllm-pd-session");
          }
        }
        await resetSession();
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

  async function resetSession() {
    setError("");
    const data = await createSession();
    setSessionId(data.session_id);
    localStorage.setItem("vllm-pd-session", data.session_id);
    setFiles([]);
    setPendingFiles([]);
    setUploadSummary(null);
    setMessages([]);
    setSources([]);
    setSidebarOpen(false);
  }

  function addPendingFiles(fileList) {
    const incoming = Array.from(fileList || []);
    if (incoming.length === 0) return;
    setPendingFiles((current) => mergeFiles(current, incoming));
    setUploadSummary(null);
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
    setUploadProgress({ done: 0, total: pendingFiles.length });
    const uploaded = [];
    let totalChunks = 0;

    try {
      for (const [index, file] of pendingFiles.entries()) {
        const formData = new FormData();
        formData.append("file", file);
        const response = await api(`/sessions/${sessionId}/upload`, {
          method: "POST",
          body: formData,
        });
        const result = await response.json();
        uploaded.push(result.filename);
        totalChunks += Number(result.num_chunks || 0);
        setUploadProgress({ done: index + 1, total: pendingFiles.length });
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
    if (!cleanQuestion || busy || !sessionId) return;

    const assistantId = crypto.randomUUID();
    setQuestion("");
    setError("");
    setSources([]);
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", content: cleanQuestion },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        model: selectedModel?.name || model,
        mode,
        sources: [],
      },
    ]);
    setBusy(true);

    try {
      await streamQuery(
        {
          session_id: sessionId,
          question: cleanQuestion,
          stream: true,
          model,
          mode,
        },
        (event) => {
          if (event.type === "sources") {
            setSources(event.sources || []);
            setMessages((current) =>
              current.map((item) =>
                item.id === assistantId
                  ? { ...item, sources: event.sources || [] }
                  : item,
              ),
            );
          }
          if (event.type === "meta") {
            setMessages((current) =>
              current.map((item) =>
                item.id === assistantId
                  ? {
                      ...item,
                      model: event.model || item.model,
                      mode: event.mode || item.mode,
                    }
                  : item,
              ),
            );
          }
          if (event.type === "token") {
            setMessages((current) =>
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
      );
    } catch (queryError) {
      setError(queryError.message);
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantId && !item.content
            ? {
                ...item,
                content: `Không thể hoàn tất yêu cầu: ${queryError.message}`,
              }
            : item,
        ),
      );
    } finally {
      setBusy(false);
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
    <div className="app-shell">
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
          <div className="brand-mark">
            <BookOpen size={21} />
          </div>
          <div>
            <strong>VLLM-PD</strong>
            <span>Document intelligence</span>
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
            <code>{shortSession(sessionId)}</code>
          </div>
          <button
            className="icon-button"
            type="button"
            title="Tạo phiên mới"
            onClick={resetSession}
          >
            <RefreshCcw size={17} />
          </button>
        </div>

        <section className="sidebar-section">
          <div className="section-heading">
            <span>Tài liệu</span>
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
            </div>
          )}

          {uploadSummary && (
            <div className="upload-summary">
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

      <main className={`workspace ${sourcePanelOpen ? "" : "sources-collapsed"}`}>
        <header className="workspace-header">
          <div className="header-title">
            <div className={`mode-mark ${mode}`}>
              <ModeIcon size={20} />
            </div>
            <div>
              <strong>{currentMode.title}</strong>
              <span>{files.length} tài liệu trong phiên</span>
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
                    className={mode === key ? `active ${key}` : ""}
                    onClick={() => setMode(key)}
                  >
                    <Icon size={17} />
                    {option.label}
                  </button>
                );
              })}
            </div>

            <label className={`model-select ${MODEL_ACCENTS[model] || ""}`}>
              <Bot size={17} />
              <select value={model} onChange={(event) => setModel(event.target.value)}>
                {models.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
              <ChevronDown size={15} />
            </label>

            <button
              className="icon-button panel-toggle"
              type="button"
              title={sourcePanelOpen ? "Ẩn nguồn" : "Hiện nguồn"}
              onClick={() => setSourcePanelOpen((open) => !open)}
            >
              {sourcePanelOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
            </button>
          </div>
        </header>

        <div className="workspace-body">
          <section className="conversation">
            <div className="conversation-scroll">
              {messages.length === 0 ? (
                <div className="empty-conversation">
                  <div className="empty-copy">
                    <div className={`empty-icon ${mode}`}>
                      <ModeIcon size={30} />
                    </div>
                    <h1>{currentMode.title}</h1>
                    <p>
                      {files.length > 0
                        ? "Sẵn sàng truy vấn tài liệu đã index trong phiên này."
                        : "Tải tài liệu lên hoặc hỏi trực tiếp để bắt đầu phiên làm việc."}
                    </p>
                  </div>

                  <div className="prompt-grid">
                    {QUICK_PROMPTS[mode].map((prompt) => (
                      <button
                        key={prompt}
                        type="button"
                        onClick={() => sendMessage(prompt)}
                      >
                        <Search size={16} />
                        <span>{prompt}</span>
                      </button>
                    ))}
                  </div>

                  <div className="empty-metrics">
                    <div>
                      <Layers3 size={16} />
                      <strong>{files.length}</strong>
                      <span>Tài liệu</span>
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
                <div className="message-list">
                  {messages.map((message) => (
                    <article className={`message ${message.role}`} key={message.id}>
                      <div className="message-avatar">
                        {message.role === "user" ? "U" : <Bot size={18} />}
                      </div>
                      <div className="message-content">
                        {message.role === "assistant" && (
                          <div className="message-meta">
                            <span>{message.model}</span>
                            <span>
                              {message.mode === "research" ? "Nghiên cứu" : "Hỏi đáp"}
                            </span>
                          </div>
                        )}
                        <ReactMarkdown>
                          {message.content || (busy ? "Đang phân tích tài liệu..." : "")}
                        </ReactMarkdown>
                        {message.role === "assistant" && message.sources?.length > 0 && (
                          <details className="message-sources">
                            <summary>{message.sources.length} nguồn tham chiếu</summary>
                            {message.sources.map((source, index) => (
                              <div key={`${source.file}-${source.page}-${index}`}>
                                <strong>{source.file}</strong>
                                <span>Trang {source.page}</span>
                              </div>
                            ))}
                          </details>
                        )}
                        {message.role === "assistant" && busy && !message.content && (
                          <Loader2 className="spin inline-loader" size={17} />
                        )}
                      </div>
                    </article>
                  ))}
                  <div ref={endRef} />
                </div>
              )}
            </div>

            {error && (
              <div className="error-banner">
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

            <form className="composer" onSubmit={onSubmit}>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                  }
                }}
                rows={2}
                placeholder={
                  mode === "research"
                    ? "Nhập chủ đề nghiên cứu..."
                    : "Đặt câu hỏi về tài liệu..."
                }
                disabled={busy}
              />
              <div className="composer-footer">
                <div className="composer-context">
                  <span className={`model-dot ${MODEL_ACCENTS[model] || ""}`} />
                  <span>{selectedModel?.description || "Đang tải danh sách model"}</span>
                </div>
                <div className="composer-actions">
                  <span className="char-count">{question.length}/4000</span>
                  <button
                    className="send-button"
                    type="submit"
                    title="Gửi"
                    disabled={!canAsk}
                  >
                    {busy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
                  </button>
                </div>
              </div>
            </form>
          </section>

          <aside className="source-panel">
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
                    <FileText size={16} />
                    <strong>{source.file}</strong>
                  </div>
                  <div className="source-meta">
                    <span>Trang {source.page || "?"}</span>
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
