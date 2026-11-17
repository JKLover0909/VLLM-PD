import React from 'react';
import { Loader2, Sparkles, Check, Copy, AlertCircle, Trash2, X } from "lucide-react";
import {
  getQaCitationEntries,
  getQaElapsedSeconds,
  isQaCitationListTruncated,
} from "../qa-citations";
import { qaEffectiveQuestion } from "../qa-mode";
import { getQaMessageSuggestions } from "../qa-prompts";

function qaCitationTypeLabel(type, language) {
  const labels = language === "ja"
    ? { sql: "SQL参照", row: "レコード参照", semantic: "意味検索参照" }
    : { sql: "Tham chiếu truy vấn", row: "Tham chiếu bản ghi", semantic: "Tham chiếu ngữ nghĩa" };
  return labels[type] || labels.semantic;
}

function QaCitationList({ metadata, language }) {
  const citations = getQaCitationEntries(metadata);
  const truncated = isQaCitationListTruncated(metadata);
  if (!citations.length && !truncated) return null;
  const labels = language === "ja"
    ? {
        title: "Agent QMS 5M1E — 参照情報",
        views: "ビュー",
        keys: "キー",
        inspections: "検査キー",
        chunks: "関連チャンク",
        truncatedNote: "参照情報の一部は件数上限または形式不正のため表示されていません。",
        sourceNote: "参照情報は照合の手がかりであり、業務データの正確性を保証するものではありません。",
      }
    : {
        title: "Nguồn tham chiếu — Agent QMS 5M1E",
        views: "View",
        keys: "Khóa bản ghi",
        inspections: "Khóa kiểm tra",
        chunks: "Đoạn ngữ nghĩa",
        truncatedNote: "Một phần tham chiếu đã được lược bớt do vượt giới hạn hoặc sai định dạng.",
        sourceNote: "Các tham chiếu hỗ trợ đối chiếu, không chứng minh số liệu đã được xác minh nghiệp vụ.",
      };

  return (
    <section className="qa-citations" aria-label={labels.title}>
      <h4>{labels.title}</h4>
      {citations.map((citation, index) => (
        <article className="qa-citation" key={`${citation.type}-${index}`}>
          <strong className="qa-citation-type">{qaCitationTypeLabel(citation.type, language)}</strong>
          {citation.views.length > 0 && (
            <p><span>{labels.views}:</span> {citation.views.join(", ")}</p>
          )}
          {citation.fields.map((field) => (
            <p key={field.name}><span>{field.name}:</span> {field.values.join(", ")}</p>
          ))}
          {citation.inspectionKeys.length > 0 && (
            <p><span>{labels.inspections}:</span> {citation.inspectionKeys.join(", ")}</p>
          )}
          {citation.chunks.length > 0 && (
            <p>
              <span>{labels.chunks}:</span>{" "}
              {citation.chunks.map((chunk, chunkIndex) => (
                <span className="qa-citation-chunk" key={`${chunk.id}-${chunkIndex}`}>
                  {[chunk.id, chunk.sourceView, chunk.sourceKey, chunk.inspectionKey].filter(Boolean).join(" · ")}
                </span>
              ))}
            </p>
          )}
        </article>
      ))}
      {truncated && <p className="qa-citation-note">{labels.truncatedNote}</p>}
      <p className="qa-citation-note">{labels.sourceNote}</p>
    </section>
  );
}

function answerScopeLabel(message, t) {
  if (message.answerScope === "qa_agent") return t("answerScope.qa_agent");
  if (message.answerScope === "general") return t("common.noResult");
  if (message.answerScope === "web") return t("common.webSearch");
  if (message.answerScope === "mes") return t("common.mesData");
  if (message.answerScope === "mes_database") return t("common.mesSnapshot");
  if (message.answerScope === "wms_database") return t("common.wmsSnapshot");
  if (message.answerScope === "cctvai_database") return "CCTVAI";
  if (message.answerScope === "cctvai_hardware") return t("common.cctvaiHardware");
  if (message.answerScope === "mes_report") return t("common.mesReport");
  if (message.answerScope === "wms_executive_report") return t("common.wmsReport");
  if (message.answerScope === "hr_executive_report") return t("common.hrReport");
  if (message.answerScope === "cctvai_report") return t("common.cctvaiReport");
  if (message.answerScope === "mes_report_unsupported") return t("common.mesReportUnsupported");
  if (message.mode === "research") return t("common.research");
  return t("common.mkacSource");
}

function answerScopeDescription(message, t) {
  if (message.answerScope === "qa_agent") return t("answerScope.qa_agent");
  if (message.answerScope === "web") return t("answerScope.web");
  if (message.answerScope === "mes") return t("answerScope.mes");
  if (message.answerScope === "mes_database") return t("answerScope.mes_database");
  if (message.answerScope === "wms_database") return t("answerScope.wms_database");
  if (message.answerScope === "cctvai_database") return t("answerScope.cctvai_database");
  if (message.answerScope === "cctvai_hardware") return t("answerScope.cctvai_hardware");
  if (message.answerScope === "mes_report") return t("answerScope.mes_report");
  if (message.answerScope === "wms_executive_report") return t("answerScope.wms_executive_report");
  if (message.answerScope === "hr_executive_report") return t("answerScope.hr_executive_report");
  if (message.answerScope === "cctvai_report") return t("answerScope.cctvai_report");
  if (message.answerScope === "mes_report_unsupported") return t("answerScope.mes_report_unsupported");
  if (message.answerScope === "research") return t("answerScope.research");
  if (message.answerScope === "mkac") return t("answerScope.mkac");
  return t("answerScope.fallback");
}

function QaAnswerDetails({ metadata, language }) {
  const seconds = getQaElapsedSeconds(metadata);
  const clarification = metadata?.clarification === true;
  if (seconds === null && !clarification) return null;
  const labels = language === "ja"
    ? { time: "処理時間", clarification: "追加情報が必要" }
    : { time: "Thời gian xử lý", clarification: "Agent cần thêm thông tin" };
  return (
    <>
      {clarification && <span className="qa-clarification">{labels.clarification}</span>}
      {seconds !== null && (
        <span className="qa-processing-time">{labels.time}: {seconds.toFixed(1)}s</span>
      )}
    </>
  );
}

function QaEffectiveQuestion({ metadata, question, language }) {
  const effective = qaEffectiveQuestion(metadata, question);
  if (!effective) return null;
  const label = language === "ja" ? "前の質問を踏まえた解釈" : "Agent hiểu câu hỏi nối tiếp là";
  return (
    <p className="qa-effective-question">
      <span>{label}:</span> {effective}
    </p>
  );
}

const EMPTY_SUGGESTIONS = [];
const NOOP = () => {};
const EmptyReportArtifactCard = () => null;
const EmptyAgentTimeline = () => null;
const EmptyBot = () => null;
const EmptyMessageMarkdown = ({ content }) => content;
const EmptyCopyAnswer = () => {};
const EmptyScrollToEnd = () => {};

export function getMessageSuggestions(message, mode, getSuggestions, askedQuestion = "", language = "vi") {
  if (mode === "qa" || message.mode === "qa" || message.answerScope === "qa_agent") {
    return getQaMessageSuggestions(message, askedQuestion, language);
  }
  return getSuggestions(message.content, mode, message.id);
}

export function MessageList({
  messages = [],
  mode = "mkac",
  busy = false,
  language = "vi",
  t = (key) => key,
  latestAssistantMessage,
  pendingAssistantId = "",
  waitingMessageIndex = 0,
  text = { waiting: [] },
  copiedMessageId = "",
  copyAnswer = EmptyCopyAnswer,
  openSourcePreview = NOOP,
  getSuggestions = () => EMPTY_SUGGESTIONS,
  handleQuickAnswerClick = NOOP,
  endRef,
  onScrollToEnd = EmptyScrollToEnd,
  error = "",
  MessageMarkdown = EmptyMessageMarkdown,
  AgentTimeline = EmptyAgentTimeline,
  ReportArtifactCard = EmptyReportArtifactCard,
  onReportEmail = NOOP,
  Bot = EmptyBot,
}) {
  return (
    <div
      className="message-list"
      role="log"
      aria-live="polite"
      aria-relevant="additions text"
    >
      {messages.map((message, index) => {
        const askedQuestion = messages[index - 1]?.role === "user" ? messages[index - 1].content : "";
        const suggestions = message.role === "assistant" &&
          message.id !== pendingAssistantId &&
          message.content
          ? getMessageSuggestions(message, mode, getSuggestions, askedQuestion, language)
          : EMPTY_SUGGESTIONS;
        const isQa = message.mode === "qa" || message.answerScope === "qa_agent";
        const citations = isQa ? getQaCitationEntries(message.qaMetadata) : [];
        return (
        <article
          className={`message ${message.role}${message.artifact ? " has-report-artifact" : ""}${isQa ? " qa-message" : ""}`}
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
                <span>{answerScopeLabel(message, t)}</span>
                {isQa && <QaAnswerDetails metadata={message.qaMetadata} language={language} />}
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
            {isQa && message.role === "assistant" && (
              <QaEffectiveQuestion
                metadata={message.qaMetadata}
                question={askedQuestion}
                language={language}
              />
            )}
            {message.content ? (
              <div className="message-body">
                <MessageMarkdown content={message.content} qa={isQa && message.role === "assistant"} />
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
                    {language === "ja" && !isQa && (
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
            {isQa && message.role === "assistant" && message.content && (
              <QaCitationList metadata={message.qaMetadata} language={language} />
            )}
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
                    <span>{answerScopeDescription(message, t)}</span>
                    {isQa && <span>{t("answerScope.qa_disclaimer")}</span>}
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
            {message.role === "assistant" && !isQa && message.sources?.length > 0 && (
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
            {suggestions.length > 0 && (
              <div className="message-suggestions">
                <p>{t("common.suggestions")}</p>
                <ul className="suggestion-list">
                  {suggestions.map((suggestion, i) => (
                    <li key={i}>
                      <button
                        type="button"
                        onClick={() => handleQuickAnswerClick(suggestion)}
                        title={t("common.autoAsk")}
                      >
                        {suggestion.label || suggestion.question}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </article>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}

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
