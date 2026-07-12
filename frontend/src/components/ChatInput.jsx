import React from 'react';

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
  Search
}) {
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
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              sendMessage(event.currentTarget.value);
            }
          }}
          placeholder={modeText(mode).inputPlaceholder}
          disabled={
            (mode === "mkac" && !mkacAuthorized) ||
            (mode === "mes" && !mkacAuthorized) ||
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
