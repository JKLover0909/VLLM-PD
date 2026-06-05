import { StrictMode, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import {
  AlertCircle,
  BookOpen,
  Bot,
  CheckCircle2,
  ChevronDown,
  FileText,
  FlaskConical,
  Loader2,
  Menu,
  MessageSquare,
  Paperclip,
  Plus,
  Search,
  Send,
  Server,
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

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      message = body.detail || body.message || message;
    } catch {
      message = await response.text() || message;
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
          // Ignore malformed SSE fragments and continue the stream.
          continue;
        }
        onEvent(event);
      }
    }
  }
}

function App() {
  const [sessionId, setSessionId] = useState("");
  const [files, setFiles] = useState([]);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [messages, setMessages] = useState([]);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("auto");
  const [mode, setMode] = useState("chat");
  const [question, setQuestion] = useState("");
  const [sources, setSources] = useState([]);
  const [health, setHealth] = useState("checking");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const fileInputRef = useRef(null);
  const endRef = useRef(null);

  const selectedModel = useMemo(
    () => models.find((item) => item.id === model),
    [models, model],
  );

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
    setMessages([]);
    setSources([]);
    setSidebarOpen(false);
  }

  async function uploadDocuments() {
    if (!sessionId || pendingFiles.length === 0) return;
    setUploading(true);
    setError("");
    const uploaded = [];

    try {
      for (const file of pendingFiles) {
        const formData = new FormData();
        formData.append("file", file);
        const response = await api(`/sessions/${sessionId}/upload`, {
          method: "POST",
          body: formData,
        });
        const result = await response.json();
        uploaded.push(result.filename);
      }
      setFiles((current) => [...new Set([...current, ...uploaded])]);
      setPendingFiles([]);
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
            ? { ...item, content: `Không thể hoàn tất yêu cầu: ${queryError.message}` }
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

  return (
    <div className="app-shell">
      <button
        className="mobile-menu icon-button"
        type="button"
        title="Mở danh sách tài liệu"
        onClick={() => setSidebarOpen(true)}
      >
        <Menu size={20} />
      </button>

      {sidebarOpen && (
        <button
          className="sidebar-backdrop"
          type="button"
          aria-label="Đóng danh sách tài liệu"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`document-sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark"><BookOpen size={21} /></div>
          <div>
            <strong>VLLM-PD</strong>
            <span>Research Workspace</span>
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

        <button className="new-session-button" type="button" onClick={resetSession}>
          <Plus size={17} />
          Phiên mới
        </button>

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
            onChange={(event) => setPendingFiles(Array.from(event.target.files || []))}
          />
          <button
            className="upload-zone"
            type="button"
            onClick={() => fileInputRef.current?.click()}
          >
            <UploadCloud size={22} />
            <span>Chọn tài liệu</span>
            <small>PDF, Office, HTML, ảnh</small>
          </button>

          {pendingFiles.length > 0 && (
            <div className="pending-upload">
              <span>{pendingFiles.length} tệp đã chọn</span>
              <button type="button" onClick={uploadDocuments} disabled={uploading}>
                {uploading ? <Loader2 className="spin" size={16} /> : <Paperclip size={16} />}
                {uploading ? "Đang index" : "Index"}
              </button>
            </div>
          )}

          <div className="file-list">
            {files.map((filename) => (
              <div className="file-item" key={filename}>
                <FileText size={17} />
                <span title={filename}>{filename}</span>
                <button
                  className="icon-button subtle"
                  type="button"
                  title={`Xóa ${filename}`}
                  onClick={() => removeFile(filename)}
                >
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
            {files.length === 0 && (
              <div className="empty-files">Chưa có tài liệu trong phiên</div>
            )}
          </div>
        </section>

        <div className="sidebar-footer">
          <div className={`service-state ${health}`}>
            {health === "online" ? <CheckCircle2 size={15} /> : <Server size={15} />}
            <span>{health === "online" ? "Máy 2 online" : "Đang kiểm tra"}</span>
          </div>
          <code>{sessionId ? sessionId.slice(0, 8) : "no-session"}</code>
        </div>
      </aside>

      <main className="workspace">
        <header className="workspace-header">
          <div className="mode-tabs" role="tablist">
            <button
              type="button"
              className={mode === "chat" ? "active" : ""}
              onClick={() => setMode("chat")}
            >
              <MessageSquare size={17} />
              Hỏi đáp
            </button>
            <button
              type="button"
              className={mode === "research" ? "active research" : ""}
              onClick={() => setMode("research")}
            >
              <FlaskConical size={17} />
              Nghiên cứu
            </button>
          </div>

          <label className="model-select">
            <Bot size={17} />
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {models.map((item) => (
                <option key={item.id} value={item.id}>{item.name}</option>
              ))}
            </select>
            <ChevronDown size={15} />
          </label>
        </header>

        <div className="workspace-body">
          <section className="conversation">
            <div className="conversation-scroll">
              {messages.length === 0 ? (
                <div className="empty-conversation">
                  <div className={`empty-icon ${mode}`}>
                    {mode === "chat" ? <MessageSquare size={30} /> : <FlaskConical size={30} />}
                  </div>
                  <h1>{mode === "chat" ? "Hỏi tài liệu" : "Nghiên cứu tài liệu"}</h1>
                  <div className="prompt-grid">
                    {QUICK_PROMPTS[mode].map((prompt) => (
                      <button
                        key={prompt}
                        type="button"
                        onClick={() => sendMessage(prompt)}
                      >
                        <Search size={16} />
                        {prompt}
                      </button>
                    ))}
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
                            <span>{message.mode === "research" ? "Nghiên cứu" : "Hỏi đáp"}</span>
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
                <button className="icon-button" type="button" title="Đóng" onClick={() => setError("")}>
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
                placeholder={mode === "research" ? "Nhập chủ đề nghiên cứu..." : "Đặt câu hỏi về tài liệu..."}
                disabled={busy}
              />
              <div className="composer-footer">
                <span>{selectedModel?.description || "Đang tải danh sách model"}</span>
                <button
                  className="send-button"
                  type="submit"
                  title="Gửi"
                  disabled={busy || !question.trim()}
                >
                  {busy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
                </button>
              </div>
            </form>
          </section>

          <aside className="source-panel">
            <div className="source-header">
              <span>Nguồn tham chiếu</span>
              <span>{sources.length}</span>
            </div>
            <div className="source-list">
              {sources.map((source, index) => (
                <article className="source-item" key={`${source.file}-${source.page}-${index}`}>
                  <div className="source-title">
                    <FileText size={16} />
                    <strong>{source.file}</strong>
                  </div>
                  <div className="source-meta">
                    <span>Trang {source.page}</span>
                    <span>{Math.round((source.score || 0) * 100)}%</span>
                  </div>
                  <p>{source.preview}</p>
                </article>
              ))}
              {sources.length === 0 && (
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
