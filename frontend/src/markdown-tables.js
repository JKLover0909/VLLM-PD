import { gfmTableFromMarkdown } from "mdast-util-gfm-table";
import { gfmTable } from "micromark-extension-gfm-table";

/**
 * Remark plugin that parses GFM pipe tables only. Unlike remark-gfm it adds no
 * autolink literals, strikethrough ("10~20" stays text), footnotes or tasks.
 */
export function remarkGfmTables() {
  const data = this.data();
  (data.micromarkExtensions ||= []).push(gfmTable());
  (data.fromMarkdownExtensions ||= []).push(gfmTableFromMarkdown());
}

const DEFAULT_MARKDOWN_OPTIONS = Object.freeze({});
const QA_MARKDOWN_OPTIONS = Object.freeze({
  remarkPlugins: [remarkGfmTables],
  // Agent answers must not make the browser fetch remote images.
  disallowedElements: ["img"],
});

/** Stable react-markdown props; tables are enabled for QA answers only. */
export function getMessageMarkdownOptions({ qa = false } = {}) {
  return qa ? QA_MARKDOWN_OPTIONS : DEFAULT_MARKDOWN_OPTIONS;
}
