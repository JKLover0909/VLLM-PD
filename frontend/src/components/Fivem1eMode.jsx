import React, { useMemo, useRef, useState, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { Bot, Send, Square } from "lucide-react";
import AgentPresence from "./AgentPresence.jsx";
import TaskLauncher from "./TaskLauncher.jsx";
import { buildTaskGroups } from "../task-launcher.js";
import { getMessageMarkdownOptions } from "../markdown-tables";
import {
  buildFivem1ePayload,
  initialFivem1eState,
  reduceFivem1eEvent, canSubmitFivem1eQuestion, shouldApplyFivem1eEvent, stopFivem1eQuery,
  streamFivem1eQuery,
} from "../fivem1e-mode.js";
import "../msm-mode.css";

const FIVEM1E_MARKDOWN_OPTIONS = getMessageMarkdownOptions({ qa: true });

const STARTER_PROMPTS = {
  vi: [
    { label: "Tra mã quản lý", question: "Tìm báo cáo có mã quản lý HB-C04-2512-01" },
    { label: "Tiến độ 30 bước", question: "Báo cáo PCB_5M1E_20251216_0119 đang ở bước nào và tiến độ ra sao?" },
    { label: "Báo cáo bước 16", question: "Có những báo cáo nào đang dừng ở bước 16?" },
    { label: "Thống kê theo hạng", question: "Có bao nhiêu báo cáo theo từng hạng?" },
  ],
  ja: [
    { label: "管理コード検索", question: "管理コードHB-C04-2512-01のレポートを検索" },
    { label: "30ステップ進捗", question: "レポートPCB_5M1E_20251216_0119の30ステップ進捗" },
    { label: "ステップ16待機", question: "ステップ16で待機中のレポート一覧" },
    { label: "ランク別件数", question: "ランク別のレポート件数一覧" },
  ],
};

const COPY = {
  vi: {
    emptyTitle: "Báo cáo thay đổi 5M1E",
    emptyBody:
      "Tra cứu báo cáo thay đổi 5M1E, kiểm tra tiến độ 30 bước, lọc theo bước chờ hoặc xem thống kê theo hạng.",
    inputLabel: "Câu hỏi về 5M1E",
    placeholder: "Hỏi theo mã quản lý hoặc tiến độ bước, ví dụ: báo cáo nào ở bước 16?",
    send: "Gửi",
    stop: "Dừng",
    sending: "Đang trả lời",
    processing: "Đang xử lý",
    loginRequired: "Vui lòng nhập mã nhân viên ở góc trên để gửi câu hỏi.",
    nextCandidate: "Gợi ý tiếp theo:",
  },
  ja: {
    emptyTitle: "5M1E変更管理",
    emptyBody:
      "5M1E変更レポートの検索、30ステップ進捗確認、待機ステップ別絞り込み、ランク別集計を行います。",
    inputLabel: "5M1Eに関する質問",
    placeholder: "管理コードまたは進捗ステップで質問（例: ステップ16で待機中のレポートは？）",
    send: "送信",
    stop: "停止",
    sending: "回答中",
    processing: "処理中",
    loginRequired: "質問を送信するには上部で社員番号を入力してください。",
    nextCandidate: "次の候補:",
  },
};

export function Fivem1eMode({
  employeeId,
  language = "vi",
  presence,
  onTurnCountChange,
  onPreviewOpenChange, onStreamingChange, onAnsweringChange, onErrorChange,
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
  const activeTurnRef = useRef(null);
  const employeeRef = useRef(employeeId);
  employeeRef.current = employeeId;
  const textareaRef = useRef(null);
  const scrollRef = useRef(null);

  const t = COPY[language] || COPY.vi;
  const starterPrompts = useMemo(() => {
    return STARTER_PROMPTS[language] || STARTER_PROMPTS.vi;
  }, [language]);

  const taskGroups = useMemo(
    () => buildTaskGroups("fivem1e", starterPrompts, language),
    [starterPrompts, language]
  );

  useEffect(() => {
    onTurnCountChange?.(turns.length);
  }, [turns.length, onTurnCountChange]);

  useEffect(() => {
    onPreviewOpenChange?.(Boolean(selectedTask));
  }, [selectedTask, onPreviewOpenChange]);

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [turns, scrollToBottom]);

  const cancelActiveStream = useCallback(() => {
    const turnId = activeTurnRef.current;
    const controller = abortRef.current;
    if (turnId && controller) {
      setTurns((prev) => prev.map((turn) => (
        turn.id === turnId ? reduceFivem1eEvent(turn, { type: "stopped" }) : turn
      )));
      activeTurnRef.current = null;
      abortRef.current = null;
      controller.abort();
      void stopFivem1eQuery({ sessionId: sessionIdRef.current, employeeId });
    }
    setStreaming(false);
    onStreamingChange?.(false);
    onAnsweringChange?.(false);
  }, [employeeId, onAnsweringChange, onStreamingChange]);

  useEffect(() => () => {
    const controller = abortRef.current;
    if (!controller) return;
    activeTurnRef.current = null;
    abortRef.current = null;
    controller.abort();
    void stopFivem1eQuery({ sessionId: sessionIdRef.current, employeeId: employeeRef.current });
    onStreamingChange?.(false);
    onAnsweringChange?.(false);
  }, []);

  const submitQuestion = useCallback(
    async (rawQuestion) => {
      const q = String(rawQuestion || "").trim();
      if (!canSubmitFivem1eQuestion({ question: q, employeeId }) || streaming) return;

      cancelActiveStream();

      const newTurn = {
        id: `turn-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        question: q,
        answer: "",
        meta: "",
        pills: [],
        error: null,
        done: false,
        stopped: false,
      };
      activeTurnRef.current = newTurn.id;
      onErrorChange?.("");
      onAnsweringChange?.(false);

      setTurns((prev) => [...prev, newTurn]);
      setDraft("");
      setSelectedTask(null);
      setTaskPreviewDraft("");
      setShowFreeInput(false);
      setStreaming(true);
      onStreamingChange?.(true);

      const controller = new AbortController();
      abortRef.current = controller;

      let currentState = initialFivem1eState();

      const onEvent = (event) => {
        if (!shouldApplyFivem1eEvent({ turnId: newTurn.id, activeTurnId: activeTurnRef.current, controller, activeController: abortRef.current })) return;
        const previousAnswer = currentState.answer;
        const next = reduceFivem1eEvent(currentState, event);
        currentState = next;
        if (event?.type === "token" && event.content && !previousAnswer) onAnsweringChange?.(true);
        if (event?.type === "error") onErrorChange?.(event.code || "");
        setTurns((prev) => {
          return prev.map((turn) => turn.id === newTurn.id ? {
            ...turn,
            answer: next.answer,
            meta: next.meta,
            pills: next.pills,
            error: next.error,
            done: next.done,
            stopped: next.stopped,
          } : turn);
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
        if (err.name !== "AbortError" && shouldApplyFivem1eEvent({ turnId: newTurn.id, activeTurnId: activeTurnRef.current, controller, activeController: abortRef.current })) {
          onErrorChange?.(err.code || "ERROR");
          setTurns((prev) => prev.map((turn) => turn.id === newTurn.id ? { ...turn, error: { code: err.code || "ERROR", message: err.message || "Lỗi kết nối" }, done: true } : turn));
        }
      } finally {
        if (activeTurnRef.current === newTurn.id && abortRef.current === controller) {
          setStreaming(false);
          onStreamingChange?.(false);
          onAnsweringChange?.(false);
          activeTurnRef.current = null;
          abortRef.current = null;
        }
      }
    },
    [cancelActiveStream, employeeId, onAnsweringChange, onErrorChange, onStreamingChange, streaming]
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!canSubmitFivem1eQuestion({ question: draft, employeeId }) || streaming) return;
    submitQuestion(draft);
  };

  const handleKeyDown = (e) => {
    if (e.nativeEvent?.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!canSubmitFivem1eQuestion({ question: draft, employeeId }) || streaming) return;
      submitQuestion(draft);
    }
  };

  const isComposerOpen = turns.length > 0 || showFreeInput;

  return (
    <div className="msm-mode" lang={language === "ja" ? "ja" : "vi"} aria-label={t.emptyTitle}>
      <div className="msm-scroll conversation-scroll" ref={scrollRef}>
        {turns.length === 0 ? (
          <div className="empty-conversation">
            <div className="empty-copy">
              <AgentPresence presence={presence} mode="fivem1e" language={language} />
            </div>
            <TaskLauncher
              mode="fivem1e"
              language={language}
              groups={taskGroups}
              selectedTask={selectedTask}
              previewQuestion={taskPreviewDraft}
              onPreviewChange={setTaskPreviewDraft}
              onSelect={(task) => {
                setSelectedTask(null);
                submitQuestion(task.question);
              }}
              onRun={(task, edited) => {
                const q = typeof edited === "string" && edited.trim() ? edited.trim() : task?.question || task;
                if (!q) return;
                setSelectedTask(null);
                setTaskPreviewDraft("");
                submitQuestion(q);
              }}
              onBack={() => {
                setSelectedTask(null);
                setTaskPreviewDraft("");
              }}
              onFreeInput={() => {
                setSelectedTask(null);
                setShowFreeInput(true);
                setTimeout(() => textareaRef.current?.focus(), 50);
              }}
              onBackToTasks={() => {
                setShowFreeInput(false);
              }}
              showFreeInput={showFreeInput}
              onToggleInfo={() => setTaskInfoOpen((o) => !o)}
              showInfo={taskInfoOpen}
              modeDescription={t.emptyBody}
              infoItems={
                language === "ja"
                  ? [
                      "5M1E変更レポートの照会および進捗トラッキング",
                      "30ステップ進捗、待機ステップ、ランク別集計",
                      "レポートコードまたは管理コードによる即時検索",
                    ]
                  : [
                      "Tra cứu báo cáo thay đổi 5M1E và theo dõi tiến độ",
                      "Kiểm tra 30 bước, lọc bước đang chờ, thống kê theo hạng",
                      "Tìm kiếm tức thì theo mã quản lý hoặc mã báo cáo",
                    ]
              }
            />
          </div>
        ) : null}

        {turns.map((turn, index) => (
          <div className="msm-turn" key={turn.id}>
            <article className="message user">
              <div className="message-avatar">U</div>
              <div className="message-content">{turn.question}</div>
            </article>
            <article className="message assistant" data-latest={index === turns.length - 1 ? "true" : undefined}>
              <div className="message-avatar">
                <Bot size={18} />
              </div>
              <div className="message-content">
                {turn.answer ? (
                  <ReactMarkdown {...FIVEM1E_MARKDOWN_OPTIONS}>{turn.answer}</ReactMarkdown>
                ) : null}
                {!turn.done && !turn.error ? (
                  <div className="msm-progress" role="status">
                    {turn.meta || t.processing}
                  </div>
                ) : null}
                {turn.stopped ? (
                  <p className="msm-progress" role="status">
                    {language === "ja" ? "停止済み" : "Đã dừng"}
                  </p>
                ) : null}
                {turn.error ? (
                  <p className="msm-error" role="alert">
                    {turn.error.message}
                  </p>
                ) : null}
                {Array.isArray(turn.pills) && turn.pills.length > 0 ? (
                  <div className="msm-pills" role="group" aria-label={t.nextCandidate}>
                    <p className="msm-pills-label">
                      <span>{t.nextCandidate}</span>
                    </p>
                    <div className="msm-pills-list">
                      {turn.pills.map((pill, idx) => (
                        <button
                          key={`${turn.id}-pill-${idx}`}
                          type="button"
                          className="msm-pill"
                          onClick={() => submitQuestion(pill)}
                          disabled={streaming}
                        >
                          {pill}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            </article>
          </div>
        ))}
      </div>

      {isComposerOpen ? (
        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={handleKeyDown}
            aria-label={t.inputLabel}
            placeholder={t.placeholder}
            rows={2}
            disabled={streaming || !employeeId}
          />
          <div className="composer-footer">
            <div className="composer-context" />
            <div className="composer-actions">
              <button
                className={streaming ? "send-button stopping" : "send-button"}
                type={streaming ? "button" : "submit"}
                title={streaming ? t.stop : t.send}
                onClick={streaming ? cancelActiveStream : undefined}
                disabled={streaming ? false : !draft.trim() || !employeeId}
              >
                {streaming ? (
                  <Square size={16} fill="currentColor" />
                ) : (
                  <Send size={18} />
                )}
              </button>
            </div>
          </div>
        </form>
      ) : null}

      {!employeeId && isComposerOpen ? (
        <p className="msm-hint" role="status">
          {t.loginRequired}
        </p>
      ) : null}
    </div>
  );
}
export default Fivem1eMode;
