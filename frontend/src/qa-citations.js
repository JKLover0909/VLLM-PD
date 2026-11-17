const QA_CITATION_TYPES = new Set(["sql", "row", "semantic"]);
const MAX_CITATIONS = 20;
const MAX_FIELDS = 10;
const MAX_VALUES = 20;
const MAX_DISPLAY_CHARS = 200;

function displayValue(value) {
  if (typeof value !== "string" && typeof value !== "number") return "";
  return String(value)
    .replace(/[\u0000-\u001f\u007f]/g, "")
    .trim()
    .slice(0, MAX_DISPLAY_CHARS);
}

function displayValues(values) {
  if (!Array.isArray(values)) return [];
  return values.slice(0, MAX_VALUES).map(displayValue).filter(Boolean);
}

/** Agent processing time in seconds, or null when the agent reported none. */
export function getQaElapsedSeconds(metadata) {
  const elapsed = metadata?.elapsed_ms;
  if (typeof elapsed !== "number" || !Number.isFinite(elapsed) || elapsed < 0) return null;
  return elapsed / 1000;
}

/** True when the gateway dropped or capped part of the agent's references. */
export function isQaCitationListTruncated(metadata) {
  return metadata?.citations_truncated === true;
}

/** Return only the allowlisted QA reference fields needed by the UI. */
export function getQaCitationEntries(metadata) {
  if (!Array.isArray(metadata?.citations)) return [];
  return metadata.citations
    .slice(0, MAX_CITATIONS)
    .filter((citation) => citation && QA_CITATION_TYPES.has(citation.type))
    .map((citation) => {
      const views = displayValues(citation.views);
      const fields = citation.keys && typeof citation.keys === "object"
        ? Object.entries(citation.keys)
            .slice(0, MAX_FIELDS)
            .map(([name, values]) => ({ name: displayValue(name), values: displayValues(values) }))
            .filter((field) => field.name && field.values.length > 0)
        : [];
      const inspectionKeys = displayValues(citation.inspection_keys);
      const chunks = Array.isArray(citation.chunks)
        ? citation.chunks.slice(0, MAX_VALUES).map((chunk) => ({
            id: displayValue(chunk?.id),
            sourceView: displayValue(chunk?.source_view),
            sourceKey: displayValue(chunk?.source_key),
            inspectionKey: displayValue(chunk?.inspection_key),
          })).filter((chunk) => chunk.id || chunk.sourceView || chunk.sourceKey || chunk.inspectionKey)
        : [];
      return { type: citation.type, views, fields, inspectionKeys, chunks };
    })
    .filter((citation) => citation.views.length || citation.fields.length || citation.inspectionKeys.length || citation.chunks.length);
}
