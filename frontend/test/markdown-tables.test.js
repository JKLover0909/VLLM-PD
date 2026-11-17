import assert from "node:assert/strict";
import test from "node:test";

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";

import { getMessageMarkdownOptions } from "../src/markdown-tables.js";

const TABLE = "| Lot | Lỗi |\n|---|---:|\n| A1 | 3 |\n| B2 | 10~20 |";

function render(content, qa) {
  return renderToStaticMarkup(
    createElement(ReactMarkdown, getMessageMarkdownOptions({ qa }), content),
  );
}

test("renders GFM pipe tables in QA answers", () => {
  const html = render(TABLE, true);
  assert.match(html, /<table>/);
  assert.match(html, /<th>Lot<\/th>/);
  assert.match(html, /<td style="text-align:right">3<\/td>/);
  assert.equal(html.includes("| Lot |"), false);
});

test("keeps other modes on the existing CommonMark rendering", () => {
  const html = render(TABLE, false);
  assert.equal(html.includes("<table>"), false);
  assert.match(html, /\| Lot \| Lỗi \|/);
});

test("enables tables only, not other GFM syntax", () => {
  const html = render("Khoảng ~10~20~ ngày, xem https://example.com hoặc [^1]", true);
  assert.equal(html.includes("<del>"), false);
  assert.equal(html.includes("<a"), false);
  assert.match(html, /~10~20~/);
});

test("drops Markdown images and keeps raw HTML as escaped text in QA answers", () => {
  const html = render('![x](http://192.0.2.1/a.png)\n\n<img src="http://192.0.2.1/b.png">', true);
  assert.equal(html.includes("<img"), false);
  assert.equal(html.includes("a.png"), false);
  assert.match(html, /&lt;img src=&quot;http:\/\/192\.0\.2\.1\/b\.png&quot;&gt;/);
});

test("returns stable option objects so memoized messages do not re-parse", () => {
  assert.equal(getMessageMarkdownOptions({ qa: true }), getMessageMarkdownOptions({ qa: true }));
  assert.equal(getMessageMarkdownOptions(), getMessageMarkdownOptions({ qa: false }));
});
