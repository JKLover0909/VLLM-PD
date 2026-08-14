import React from 'react';
import { Loader2, Sparkles, Check, Copy, AlertCircle, Trash2, X } from "lucide-react";

function wmsTranslation(t, group, value, fallback) {
  const normalized = String(value || "").trim();
  if (!normalized) return t(`common.${group}.${fallback}`);
  const key = `common.${group}.${normalized}`;
  const translated = t(key);
  return translated === key ? t(`common.${group}.${fallback}`) : translated;
}

function wmsReasonList(t, values) {
  return (values || [])
    .map((value) => wmsTranslation(t, "wmsReason", value, "fallback"))
    .join(", ");
}

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
  onScrollToEnd,
  error,
  MessageMarkdown,
  AgentTimeline,
  ReportArtifactCard,
  onReportEmail,
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
        <article
          className={`message ${message.role}${message.artifact ? " has-report-artifact" : ""}`}
          key={message.id}
        >
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
                    : message.answerScope === "wms_database"
                      ? t("common.wmsSnapshot")
                    : message.answerScope === "mes_report"
                      ? t("common.mesReport")
                    : message.answerScope === "wms_executive_report"
                      ? t("common.wmsReport")
                    : message.answerScope === "hr_executive_report"
                      ? t("common.hrReport")
                    : message.answerScope === "mes_report_unsupported"
                      ? t("common.mesReportUnsupported")
                    : message.mode === "research"
                      ? t("common.research")
                      : t("common.mkacSource")}
                </span>
              </div>
            )}
            {message.role === "assistant" && message.wmsMetadata && (
              <div className="wms-meta-chips" aria-label={t("common.wmsSnapshot")}>
                <span className="wms-meta-chip domain">
                  {wmsTranslation(t, "wmsDomain", message.wmsMetadata.domain, "fallback")}
                </span>
                <span
                  className={`wms-meta-chip status-${String(
                    message.wmsMetadata.status || "fallback",
                  ).toLowerCase()}`}
                >
                  {wmsTranslation(t, "wmsStatus", message.wmsMetadata.status, "fallback")}
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
              <ReportArtifactCard
                artifact={message.artifact}
                language={language}
                onEmail={onReportEmail}
                onRevealStage={() => onScrollToEnd?.(true)}
              />
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
                        : message.answerScope === "wms_database"
                          ? t("answerScope.wms_database")
                        : message.answerScope === "mes_report"
                          ? t("answerScope.mes_report")
                        : message.answerScope === "wms_executive_report"
                          ? t("answerScope.wms_executive_report")
                        : message.answerScope === "hr_executive_report"
                          ? t("answerScope.hr_executive_report")
                        : message.answerScope === "mes_report_unsupported"
                          ? t("answerScope.mes_report_unsupported")
                        : message.answerScope === "research"
                          ? t("answerScope.research")
                          : message.answerScope === "mkac"
                            ? t("answerScope.mkac")
                            : t("answerScope.fallback")}
                    </span>
                    {message.wmsMetadata && (
                      <dl className="wms-meta-details">
                        <div>
                          <dt>{t("common.wmsAsOf")}</dt>
                          <dd>{message.wmsMetadata.source_as_of || "—"}</dd>
                        </div>
                        <div>
                          <dt>{t("common.wmsImportedAt")}</dt>
                          <dd>{message.wmsMetadata.imported_at || "—"}</dd>
                        </div>
                        <div>
                          <dt>{t("common.wmsFreshnessState")}</dt>
                          <dd>
                            {wmsTranslation(
                              t,
                              "wmsFreshnessValue",
                              message.wmsMetadata.source_as_of_state,
                              "fallback",
                            )}
                          </dd>
                        </div>
                        <div>
                          <dt>{t("common.wmsBasis")}</dt>
                          <dd>{message.wmsMetadata.source_as_of_basis || "—"}</dd>
                        </div>
                        <div>
                          <dt>{t("common.wmsTimezone")}</dt>
                          <dd>
                            {wmsTranslation(
                              t,
                              "wmsTimezoneValue",
                              message.wmsMetadata.source_timezone,
                              "fallback",
                            )}
                          </dd>
                        </div>
                        <div>
                          <dt>{t("common.wmsEpoch")}</dt>
                          <dd>
                            {wmsTranslation(
                              t,
                              "wmsEpochValue",
                              message.wmsMetadata.semantic_epoch,
                              "fallback",
                            )}
                          </dd>
                        </div>
                        {message.wmsMetadata.dataset_evidence?.length > 0 && (
                          <div>
                            <dt>{t("common.wmsEvidence")}</dt>
                            <dd className="wms-evidence-list">
                              {message.wmsMetadata.dataset_evidence.map((evidence) => (
                                <span key={evidence.dataset}>
                                  <strong>
                                    {wmsTranslation(
                                      t,
                                      "wmsDomain",
                                      evidence.dataset,
                                      "fallback",
                                    )}
                                  </strong>
                                  {`: ${evidence.source_as_of || "—"} · ${wmsTranslation(
                                    t,
                                    "wmsFreshnessValue",
                                    evidence.source_as_of_state,
                                    "fallback",
                                  )}`}
                                </span>
                              ))}
                            </dd>
                          </div>
                        )}
                        {message.wmsMetadata.reason_codes?.length > 0 && (
                          <div>
                            <dt>{t("common.wmsReasons")}</dt>
                            <dd>{wmsReasonList(t, message.wmsMetadata.reason_codes)}</dd>
                          </div>
                        )}
                        {message.wmsMetadata.pagination && (
                          <div>
                            <dt>{t("common.wmsPagination")}</dt>
                            <dd>
                              {t("common.wmsPage", {
                                page: message.wmsMetadata.pagination.page,
                                total: message.wmsMetadata.pagination.total_count,
                              })}
                              {` · ${message.wmsMetadata.pagination.page_size || "—"} ${
                                t("common.wmsPerPage")
                              } · ${
                                message.wmsMetadata.pagination.has_more
                                  ? t("common.wmsHasMore")
                                  : t("common.wmsNoMore")
                              }`}
                            </dd>
                          </div>
                        )}
                      </dl>
                    )}
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
                        aria-label={`${t("common.sources")}: ${source.file}`}
                        title={source.file}
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
