import React from 'react';
import {
  X, RefreshCcw, Layers3, FileUp, FileText, ChevronDown,
  Search, UploadCloud, Menu, Trash2, CheckCircle2,
  Database, Loader2, Server, ShieldCheck
} from "lucide-react";

export function ResearchSidebar({
  mode,
  t,
  modeText,
  sidebarOpen,
  setSidebarOpen,
  sessionTitles,
  currentWorkspaceKey,
  researchScope,
  resetSession,
  busy,
  uploading,
  selectResearchScope,
  researchTopicId,
  selectedResearchTopic,
  researchTopicsTotalFiles,
  files,
  researchTopics,
  language,
  researchTopicLabel,
  selectResearchTopic,
  formatText,
  fileSearch,
  setFileSearch,
  filterFileNames,
  FileTypeIcon,
  fileInputRef,
  addPendingFiles,
  dragActive,
  setDragActive,
  requestResearchUploadPicker,
  onDrop,
  pendingFiles,
  pendingTotalSize,
  formatBytes,
  removePendingFile,
  uploadDocuments,
  uploadProgress,
  uploadSummary,
  setConfirmDeleteFile,
  health
}) {
  if (mode !== "research") return null;

  return (
    <>
      <button
        className="mobile-menu icon-button"
        type="button"
        title={t("common.openDocuments")}
        onClick={() => setSidebarOpen(true)}
      >
        <Menu size={20} />
      </button>

      {sidebarOpen && (
        <button
          className="sidebar-backdrop"
          type="button"
          aria-label={t("common.closeDocuments")}
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
            <span>{t("common.assistantName")}</span>
          </div>
          <button
            className="icon-button close-sidebar"
            type="button"
            title={t("common.close")}
            onClick={() => setSidebarOpen(false)}
          >
            <X size={18} />
          </button>
        </div>

        <div className="session-strip">
          <div>
            <span>{t("common.workSession")}</span>
            <strong
              className="session-title"
              title={sessionTitles[currentWorkspaceKey]}
            >
              {sessionTitles[currentWorkspaceKey]}
            </strong>
          </div>
          {researchScope === "upload" && (
            <button
              className="icon-button"
              type="button"
              title={t("common.newSession")}
              onClick={() => resetSession("research")}
              disabled={busy || uploading}
            >
              <RefreshCcw size={17} />
            </button>
          )}
        </div>

        <div className="research-source-tabs" role="tablist" aria-label={modeText("research").chooseTopic}>
          <button
            type="button"
            className={researchScope === "topic" ? "active" : ""}
            onClick={() => selectResearchScope("topic")}
            disabled={busy || uploading}
            role="tab"
            aria-selected={researchScope === "topic"}
            title={modeText("research").sourceTopic}
          >
            <Layers3 size={16} />
            <span>{modeText("research").sourceTopic}</span>
          </button>
          <button
            type="button"
            className={researchScope === "upload" ? "active" : ""}
            onClick={() => selectResearchScope("upload")}
            disabled={busy || uploading}
            role="tab"
            aria-selected={researchScope === "upload"}
            title={modeText("research").sourceUpload}
          >
            <FileUp size={16} />
            <span>{modeText("research").sourceUpload}</span>
          </button>
        </div>

        <section className="sidebar-section">
          <div className="section-heading">
            <span>{t("common.researchDocuments")}</span>
            <span className="count-badge">
              {researchScope === "topic"
                ? researchTopicId
                  ? selectedResearchTopic()?.num_files ?? 0
                  : researchTopicsTotalFiles()
                : files.length}
            </span>
          </div>

          {researchScope === "topic" && researchTopics.ready && (
            <div className="sidebar-topic-panel">
              {researchTopicId ? (
                <>
                  <div className="sidebar-topic-selected">
                    <div className="topic-info-col">
                      <span className="topic-overline">
                        {language === "ja" ? "選択中のカテゴリ" : "Nhóm tài liệu đang chọn"}
                      </span>
                      {/* Dropdown đổi nhóm chủ đề: chọn nhóm khác sẽ mở phiên chat mới. */}
                      <div className="topic-switcher">
                        <select
                          className="topic-switcher-select"
                          value={researchTopicId}
                          onChange={(event) => selectResearchTopic(event.target.value)}
                          disabled={busy}
                          aria-label={modeText("research").changeTopic}
                          title={modeText("research").changeTopic}
                        >
                          {researchTopics.topics.map((topic) => (
                            <option key={topic.id} value={topic.id} disabled={!topic.ready}>
                              {researchTopicLabel(topic)}
                            </option>
                          ))}
                        </select>
                        <ChevronDown className="topic-switcher-chevron" size={15} aria-hidden="true" />
                      </div>
                    </div>
                  </div>
                  <p className="scope-note">
                    {formatText(modeText("research").allDocsNote, {
                      count: selectedResearchTopic()?.num_files ?? 0,
                    })}
                  </p>
                  <details className="sidebar-file-browser" key={researchTopicId}>
                    <summary>
                      <span>
                        <FileText size={15} aria-hidden="true" />
                        {modeText("research").browseDocuments}
                      </span>
                      <span className="file-browser-count">
                        {selectedResearchTopic()?.num_files ?? 0}
                      </span>
                      <ChevronDown className="file-browser-chevron" size={15} />
                    </summary>
                    <div className="file-browser-content">
                      <div className="file-search">
                        <Search size={14} aria-hidden="true" />
                        <input
                          type="search"
                          value={fileSearch}
                          onChange={(event) => setFileSearch(event.target.value)}
                          placeholder={modeText("research").searchFiles}
                          aria-label={modeText("research").searchFiles}
                        />
                      </div>
                      <div className="file-list">
                        {filterFileNames(selectedResearchTopic()?.files || []).map(
                          (filename) => (
                            <div className="file-item readonly" key={filename}>
                              <FileTypeIcon filename={filename} />
                              <span title={filename}>{filename}</span>
                            </div>
                          ),
                        )}
                        {filterFileNames(selectedResearchTopic()?.files || [])
                          .length === 0 && (
                          <div className="empty-files">
                            <Search size={20} />
                            <span>{modeText("research").noMatchingFiles}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </details>
                </>
              ) : (
                <p className="sidebar-topic-hint">
                  {modeText("research").chooseTopicHint}
                </p>
              )}
            </div>
          )}

          <input
            ref={fileInputRef}
            className="hidden-input"
            type="file"
            multiple
            accept=".pdf,.docx,.xlsx,.pptx,.html,.htm,.png,.jpg,.jpeg"
            aria-label={t("common.chooseDocument")}
            onChange={(event) => addPendingFiles(event.target.files)}
          />

          {researchScope === "upload" && (
            <button
              className={`upload-zone ${dragActive ? "dragging" : ""}`}
              type="button"
              onClick={requestResearchUploadPicker}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setDragActive(false)}
              onDrop={onDrop}
            >
              <UploadCloud size={23} />
              <span>{t("common.chooseDocument")}</span>
              <small>{t("common.supportedFiles")}</small>
            </button>
          )}

          {pendingFiles.length > 0 && (
            <div className="pending-panel">
              <div className="pending-header">
                <span>{pendingFiles.length} {t("common.files")}</span>
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
                      title={t("common.removeFile", { name: file.name })}
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
                  ? t("common.indexing", {
                      done: uploadProgress.done,
                      total: uploadProgress.total,
                    })
                  : t("common.indexDocument")}
              </button>
              {uploading && (
                <div className="upload-progress" role="status" aria-live="polite">
                  <div className="upload-progress-copy">
                    <span>{t("common.processing")}</span>
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
                {t("common.indexedSummary", {
                  files: uploadSummary.files,
                  chunks: uploadSummary.chunks,
                })}
              </span>
            </div>
          )}

          {researchScope === "upload" && (
            <div className="file-list">
              {files.map((filename) => (
                <div className="file-item" key={filename}>
                  <FileTypeIcon filename={filename} />
                  <span title={filename}>{filename}</span>
                  <button
                    className="icon-button subtle danger"
                    type="button"
                    title={t("common.deleteFile", { name: filename })}
                    onClick={() => setConfirmDeleteFile(filename)}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              ))}
              {files.length === 0 && (
                <div className="empty-files">
                  <FileText size={20} />
                  <span>{t("common.noDocuments")}</span>
                </div>
              )}
            </div>
          )}
        </section>

        <div className="sidebar-footer">
          <div className={`service-state ${health}`}>
            {health === "online" ? <CheckCircle2 size={15} /> : <Server size={15} />}
            <span>{health === "online" ? t("common.onlineMachine") : t("common.checking")}</span>
          </div>
          <span className="security-chip">
            <ShieldCheck size={14} />
            {t("common.groundedBadge")}
          </span>
        </div>
      </aside>
    </>
  );
}
