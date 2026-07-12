import React from 'react';
import { Loader2, Sparkles, Check, Copy, AlertCircle, Trash2, X } from "lucide-react";

export function MessageList({
  messages,
  mode,
  busy,
  language,
  t,
  latestAssistantMessage,
  pendingAssistantId,
  waitingMessageIndex,
  text,
  copiedMessageId,
  copyAnswer,
  openSourcePreview,
  getSuggestions,
  handleQuickAnswerClick,
  endRef,
  error,
  MessageMarkdown,
  AgentTimeline,
  ReportArtifactCard,
  Bot
}) {
  return (
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
                    : message.answerScope === "mes_report"
                      ? t("common.mesReport")
                    : message.answerScope === "mes_report_unsupported"
                      ? t("common.mesReportUnsupported")
                    : message.mode === "research"
                      ? t("common.research")
                      : t("common.mkacSource")}
                </span>
              </div>
            )}
            {message.role === "assistant" && message.agentTimeline && (
              <AgentTimeline
                timeline={message.agentTimeline}
                language={language}
              />
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
            {message.role === "assistant" && message.artifact && (
              <ReportArtifactCard artifact={message.artifact} language={language} />
            )}
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
                        : message.answerScope === "mes_report"
                          ? t("answerScope.mes_report")
                        : message.answerScope === "mes_report_unsupported"
                          ? t("answerScope.mes_report_unsupported")
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
  );
}
