import React, { useRef, useState } from 'react';

const PROMPT_HISTORY_KEY = "meibook_prompt_history";

export function getStoredPromptHistory() {
  try {
    const raw = sessionStorage.getItem(PROMPT_HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function pushStoredPromptHistory(prompt) {
  if (!prompt || typeof prompt !== "string") return;
  const clean = prompt.trim();
  if (!clean) return;
  try {
    const history = getStoredPromptHistory();
    if (history[history.length - 1] !== clean) {
      const nextHistory = [...history, clean].slice(-50);
      sessionStorage.setItem(PROMPT_HISTORY_KEY, JSON.stringify(nextHistory));
    }
  } catch {
    // Ignore storage errors
  }
}

export function ChatInput({
  mode,
  t,
  modeText,
  busy,
  question,
  setQuestion,
  onSubmit,
  textareaRef,
  employee,
  language,
  quickPromptsFor,
  researchQuickPrompts,
  sendMessage,
  health,
  researchScope,
  researchTopicId,
  researchReady,
  mkacAuthorized,
  requestResearchUploadPicker,
  uploading,
  stopGeneration,
  canAsk,
  Square,
  Send,
  Paperclip,
  Search,
  userPrompts = [],
}) {
  const [historyIndex, setHistoryIndex] = useState(-1);
  const draftRef = useRef("");
  const historyRef = useRef([]);

  const refreshHistory = () => {
    const stored = getStoredPromptHistory();
    const combined = [];
    for (const item of [...stored, ...userPrompts]) {
      if (item && (combined.length === 0 || combined[combined.length - 1] !== item)) {
        combined.push(item);
      }
    }
    historyRef.current = combined;
    return combined;
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      setHistoryIndex(-1);
      draftRef.current = "";
      sendMessage(event.currentTarget.value);
      return;
    }

    if (event.key === "ArrowUp") {
      const textarea = event.currentTarget;
      const isAtStart = textarea.selectionStart === 0 && textarea.selectionEnd === 0;
      const history = historyIndex === -1 ? refreshHistory() : historyRef.current;

      if (history.length > 0 && (historyIndex !== -1 || isAtStart)) {
        event.preventDefault();

        let nextIndex;
        if (historyIndex === -1) {
          draftRef.current = question;
          nextIndex = history.length - 1;
        } else {
          nextIndex = Math.max(0, historyIndex - 1);
        }

        setHistoryIndex(nextIndex);
        const targetValue = history[nextIndex] || "";
        setQuestion(targetValue);

        setTimeout(() => {
          if (textareaRef.current) {
            textareaRef.current.selectionStart = targetValue.length;
            textareaRef.current.selectionEnd = targetValue.length;
          }
        }, 0);
      }
    } else if (event.key === "ArrowDown") {
      const history = historyRef.current;

      if (historyIndex !== -1) {
        event.preventDefault();

        if (historyIndex < history.length - 1) {
          const nextIndex = historyIndex + 1;
          setHistoryIndex(nextIndex);
          const targetValue = history[nextIndex] || "";
          setQuestion(targetValue);
          setTimeout(() => {
            if (textareaRef.current) {
              textareaRef.current.selectionStart = targetValue.length;
              textareaRef.current.selectionEnd = targetValue.length;
            }
          }, 0);
        } else {
          setHistoryIndex(-1);
          const targetValue = draftRef.current || "";
          setQuestion(targetValue);
          setTimeout(() => {
            if (textareaRef.current) {
              textareaRef.current.selectionStart = targetValue.length;
              textareaRef.current.selectionEnd = targetValue.length;
            }
          }, 0);
        }
      }
    }
  };

  const handleChange = (event) => {
    setQuestion(event.target.value);
    if (historyIndex !== -1) {
      setHistoryIndex(-1);
    }
  };

  return (
    <>
      <form className="composer" onSubmit={onSubmit}>
        {mode === "research" && researchScope === "upload" && (
          <button
            className="composer-attach icon-button"
            type="button"
            title={t("common.addResearchDocument")}
            onClick={requestResearchUploadPicker}
            disabled={busy || uploading}
          >
            <Paperclip size={18} />
          </button>
        )}
        <textarea
          ref={textareaRef}
          value={question}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={modeText(mode).inputPlaceholder}
          disabled={
            (["mkac", "mes", "wms"].includes(mode) && !mkacAuthorized) ||
            (mode === "research" && !researchReady)
          }
        />
        <div className="composer-footer">
          <div className="composer-context">
            <span className={`model-dot ${health === "online" ? "ready" : "offline"}`} />
            <span>
              {health === "online" ? t("common.ready") : t("common.notReady")}
            </span>
          </div>
          <div className="composer-actions">
            {question.length > 0 && (
              <span className="char-count">{question.length}/4000</span>
            )}
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
    </>
  );
}
