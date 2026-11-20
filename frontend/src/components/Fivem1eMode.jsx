import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { getMessageMarkdownOptions } from "../markdown-tables";
import {
  AlertCircle,
  ArrowUp,
  RotateCcw,
  Square,
  User,
  ListChecks,
} from "lucide-react";
import TaskLauncher from "./TaskLauncher.jsx";
import { buildTaskGroups } from "../task-launcher.js";
import "../msm-mode.css";
import {
  buildFivem1ePayload,
  initialFivem1eState,
  reduceFivem1eEvent,
  streamFivem1eQuery,
} from "../fivem1e-mode.js";

const FIVEM1E_MARKDOWN_OPTIONS = getMessageMarkdownOptions({ qa: true });

const STARTER_PROMPTS = {
  vi: [
    "Tìm báo cáo có mã quản lý HB-C04-2512-01",
    "Báo cáo PCB_5M1E_20251216_0119 đang ở bước nào và tiến độ ra sao?",
    "Có những báo cáo nào đang dừng ở bước 16?",
    "Có bao nhiêu báo cáo theo từng hạng?",
  ],
  ja: [
    "管理コードHB-C04-2512-01のレポートを検索",
    "レポートPCB_5M1E_20251216_0119の30ステップ進捗",
    "ステップ16で待機中のレポート一覧",
    "ランク別のレポート件数一覧",
  ],
};

const COPY = {
  vi: {
    welcomeSubtitle: "Tra cứu báo cáo thay đổi 5M1E và theo dõi tiến độ 30 bước",
    placeholder: "Nhập mã quản lý, mã báo cáo hoặc câu hỏi tra cứu 5M1E...",
    send: "Gửi",
    stop: "Dừng",
    clear: "Làm mới",
    clearConfirm: "Bắt đầu cuộc trò chuyện 5M1E mới?",
    emptySession: "Chưa có lượt trao đổi nào.",
    errorPrefix: "Lỗi:",
  },
  ja: {
    welcomeSubtitle: "5M1E変更レポートの検索および30ステップ進捗管理",
    placeholder: "管理コード、レポートコード、または5M1Eの質問を入力...",
    send: "送信",
    stop: "停止",
    clear: "クリア",
    clearConfirm: "新しい5M1Eの会話を開始しますか？",
    emptySession: "履歴はありません。",
    errorPrefix: "エラー:",
  },
};

export function Fivem1eMode({
  employeeId,
  language = "vi",
  onLogout,
  onClearChat,
  onResetConversation,
}) {
  const [turns, setTurns] = useState([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [showFreeInput, setShowFreeInput] = useState(false);
  const [taskInfoOpen, setTaskInfoOpen] = useState(false);
  const [selectedTask, setSelectedTask] = useState(null);
  const [taskPreviewDraft, setTaskPreviewDraft] = useState("");

  const sessionIdRef = useRef("");
  if (!sessionIdRef.current) {
    sessionIdRef.current =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `fivem1e-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }
  const abortRef = useRef(null);
  const textareaRef = useRef(null);
  const turnsEndRef = useRef(null);

  const t = COPY[language] || COPY.vi;
  const starterPrompts = STARTER_PROMPTS[language] || STARTER_PROMPTS.vi;
  const taskGroups = useMemo(() => buildTaskGroups("fivem1e", starterPrompts, language), [starterPrompts, language]);

  const scrollToBottom = useCallback(() => {
    turnsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [turns, scrollToBottom]);

  const cancelActiveStream = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setStreaming(false);
  }, []);

  const clearConversation = useCallback(() => {
    cancelActiveStream();
    setTurns([]);
    sessionIdRef.current =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `fivem1e-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setSelectedTask(null);
    setTaskPreviewDraft("");
    setShowFreeInput(false);
    if (onClearChat) onClearChat();
    if (onResetConversation) onResetConversation();
  }, [cancelActiveStream, onClearChat, onResetConversation]);

  const submitQuestion = useCallback(
    async (rawQuestion) => {
      const q = String(rawQuestion || "").trim();
      if (!q || streaming) return;

      cancelActiveStream();

      const newTurn = {
        question: q,
        answer: "",
        meta: "",
        pills: [],
        error: null,
        done: false,
        timestamp: new Date().toISOString(),
      };

      setTurns((prev) => [...prev, newTurn]);
      setDraft("");
      setSelectedTask(null);
      setTaskPreviewDraft("");
      setShowFreeInput(false);
      setStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      let currentState = initialFivem1eState();

      const onEvent = (event) => {
        currentState = reduceFivem1eEvent(currentState, event);
        setTurns((prev) => {
          if (!prev.length) return prev;
          const lastIdx = prev.length - 1;
          const updated = [...prev];
          updated[lastIdx] = {
            ...updated[lastIdx],
            answer: currentState.answer,
            meta: currentState.meta,
            pills: currentState.pills,
            error: currentState.error,
            done: currentState.done,
          };
          return updated;
        });
      };

      try {
        const payload = buildFivem1ePayload({
          sessionId: sessionIdRef.current,
          question: q,
          employeeId,
        });
        await streamFivem1eQuery(payload, onEvent, controller.signal);
      } catch (err) {
        if (err.name !== "AbortError") {
          setTurns((prev) => {
            if (!prev.length) return prev;
            const lastIdx = prev.length - 1;
            const updated = [...prev];
            updated[lastIdx] = {
              ...updated[lastIdx],
              error: { code: "ERROR", message: err.message || "Lỗi kết nối" },
              done: true,
            };
            return updated;
          });
        }
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [cancelActiveStream, employeeId, streaming]
  );

  const handleKeyDown = (e) => {
    if (e.nativeEvent?.isComposing) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuestion(draft);
    }
  };

  return (
    <div className="msm-workspace">
      <div className="msm-header">
        <div className="msm-header-title">
          <ListChecks size={20} className="msm-header-icon" />
          <span>5M1E Change Report</span>
        </div>
        <div className="msm-header-actions">
          {turns.length > 0 && (
            <button
              type="button"
              className="msm-btn msm-btn-ghost"
              onClick={clearConversation}
              title={t.clear}
            >
              <RotateCcw size={16} />
              <span>{t.clear}</span>
            </button>
          )}
        </div>
      </div>

      <div className="msm-body">
        {turns.length === 0 ? (
          <div className="msm-welcome">
            <div className="msm-welcome-hero">
              <h2 className="msm-welcome-title">5M1E Agent</h2>
              <p className="msm-welcome-subtitle">{t.welcomeSubtitle}</p>
            </div>
            <div className="msm-launcher-wrap">
              <TaskLauncher
                mode="fivem1e"
                starterPrompts={starterPrompts}
                language={language}
                selectedTask={selectedTask}
                taskPreviewDraft={taskPreviewDraft}
                showFreeInput={showFreeInput}
                taskInfoOpen={taskInfoOpen}
                streaming={streaming}
                onSelect={(task) => {
                  if (!task) return;
                  submitQuestion(task.question);
                }}
                onPreviewChange={setTaskPreviewDraft}
                onRun={submitQuestion}
                onBack={() => {
                  setSelectedTask(null);
                  setTaskPreviewDraft("");
                }}
                onToggleFreeInput={() => {
                  setShowFreeInput((prev) => !prev);
                  setTimeout(() => textareaRef.current?.focus(), 50);
                }}
                onToggleTaskInfo={() => setTaskInfoOpen((prev) => !prev)}
              />
            </div>
          </div>
        ) : (
          <div className="msm-turns">
            {turns.map((turn, idx) => (
              <div key={idx} className="msm-turn">
                <div className="msm-user-row">
                  <div className="msm-bubble msm-bubble-user">
                    <span>{turn.question}</span>
                  </div>
                  <div className="msm-avatar msm-avatar-user">
                    <User size={16} />
                  </div>
                </div>

                <div className="msm-bot-row">
                  <div className="msm-avatar msm-avatar-bot">
                    <ListChecks size={18} />
                  </div>
                  <div className="msm-bubble msm-bubble-bot">
                    {turn.meta && !turn.done && (
                      <div className="msm-meta-status">
                        <span className="msm-pulse-dot" />
                        <span>{turn.meta}</span>
                      </div>
                    )}

                    {turn.answer && (
                      <div className="msm-markdown-answer">
                        <ReactMarkdown {...FIVEM1E_MARKDOWN_OPTIONS}>{turn.answer}</ReactMarkdown>
                      </div>
                    )}

                    {turn.error && (
                      <div className="msm-error-box">
                        <AlertCircle size={16} />
                        <span>
                          {t.errorPrefix} {turn.error.message}
                        </span>
                      </div>
                    )}

                    {turn.pills && turn.pills.length > 0 && (
                      <div className="msm-pills">
                        {turn.pills.map((pill, pIdx) => (
                          <button
                            key={pIdx}
                            type="button"
                            className="msm-pill"
                            disabled={streaming}
                            onClick={() => submitQuestion(pill)}
                          >
                            <span>{pill}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            <div ref={turnsEndRef} />
          </div>
        )}
      </div>

      <div className="msm-footer">
        <div className="msm-composer">
          <textarea
            ref={textareaRef}
            className="msm-composer-input"
            rows={1}
            value={draft}
            placeholder={t.placeholder}
            disabled={streaming}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          {streaming ? (
            <button
              type="button"
              className="msm-send-btn msm-stop-btn"
              onClick={cancelActiveStream}
              title={t.stop}
            >
              <Square size={16} />
            </button>
          ) : (
            <button
              type="button"
              className="msm-send-btn"
              disabled={!draft.trim()}
              onClick={() => submitQuestion(draft)}
              title={t.send}
            >
              <ArrowUp size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
