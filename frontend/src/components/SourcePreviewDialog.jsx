import React from 'react';
import { X, FileText } from "lucide-react";

export function SourcePreviewDialog({
  sourcePreview,
  setSourcePreview,
  sourcePreviewUrl,
  t
}) {
  if (!sourcePreview) return null;

  return (
    <div
      className="source-preview-backdrop"
      role="presentation"
      onClick={() => setSourcePreview(null)}
    >
      <section
        className="source-preview-modal"
        role="dialog"
        aria-modal="true"
        aria-label={t("common.previewDialog")}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="source-preview-header">
          <div>
            <span>{t("common.sources")}</span>
            <strong title={sourcePreview.source.file}>
              {sourcePreview.source.file}
            </strong>
          </div>
          <button
            className="icon-button"
            type="button"
            title={t("common.closePreview")}
            onClick={() => setSourcePreview(null)}
          >
            <X size={18} />
          </button>
        </div>
        <div className="source-preview-meta">
          <span>{t("common.page", { page: sourcePreview.source.page || "?" })}</span>
          <span>
            {t("common.similarity", {
              score: Math.round((sourcePreview.source.score || 0) * 100),
            })}
          </span>
        </div>
        {sourcePreview.source.has_page_preview && !sourcePreview.imageFailed ? (
          <div className="source-preview-image-wrap">
            <img
              src={sourcePreviewUrl(sourcePreview.source, sourcePreview.mode)}
              alt={`${sourcePreview.source.file} ${t("common.page", {
                page: sourcePreview.source.page || "?",
              })}`}
              onError={() =>
                setSourcePreview((current) =>
                  current ? { ...current, imageFailed: true } : current,
                )
              }
            />
          </div>
        ) : (
          <div className="source-preview-placeholder">
            <FileText size={28} />
            <span>{t("common.noPreviewImage")}</span>
          </div>
        )}
        <div className="source-preview-text">
          <strong>{t("common.snippet")}</strong>
          <p>{sourcePreview.source.preview}</p>
        </div>
      </section>
    </div>
  );
}
