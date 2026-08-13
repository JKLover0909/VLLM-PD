---
version: alpha
name: Meibook Development UI
description: Implementation-derived visual identity for the Dev MKAC, MES, WMS, and Research interface.
colors:
  canvas-light: "#f4f6f8"
  surface-light: "#ffffff"
  surface-soft-light: "#f6f8fa"
  text-light: "#23272f"
  muted-light: "#66707d"
  sidebar: "#181a20"
  sidebar-text: "#eef2f0"
  primary: "#0d7f73"
  accent-soft: "#66d2c5"
  accent-dark: "#8ce2d8"
  mes: "#185b91"
  research: "#8a5a0a"
  wms: "#b45309"
  success: "#285e43"
  danger: "#b42318"
  canvas-dark: "#15191f"
  surface-dark: "#20242c"
  text-dark: "#edf2f7"
typography:
  body-desktop:
    fontFamily: "Merriweather, serif"
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.6
  body-mobile:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: 15px
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Merriweather, serif"
    fontSize: 13px
    fontWeight: 800
    lineHeight: 1.3
  heading:
    fontFamily: "Merriweather, serif"
    fontSize: 24px
    fontWeight: 800
    lineHeight: 1.3
  code:
    fontFamily: "SFMono-Regular, Consolas, monospace"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
rounded:
  xs: 3px
  sm: 4px
  md: 8px
  lg: 12px
  full: 999px
spacing:
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 24px
  2xl: 32px
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface-light}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: 12px
    height: 44px
  button-secondary:
    backgroundColor: "{colors.surface-light}"
    textColor: "{colors.text-light}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: 12px
    height: 44px
  mode-tab-active:
    backgroundColor: "{colors.surface-light}"
    textColor: "{colors.primary}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: 12px
    height: 44px
  mode-tab-wms:
    backgroundColor: "#fef3c7"
    textColor: "{colors.wms}"
    typography: "{typography.label}"
    rounded: "{rounded.md}"
    padding: 12px
    height: 44px
  card:
    backgroundColor: "{colors.surface-light}"
    textColor: "{colors.text-light}"
    rounded: "{rounded.md}"
    padding: 16px
  composer:
    backgroundColor: "{colors.surface-light}"
    textColor: "{colors.text-light}"
    rounded: "{rounded.lg}"
    padding: 12px
  sidebar:
    backgroundColor: "{colors.sidebar}"
    textColor: "{colors.sidebar-text}"
    width: 304px
  report-card:
    backgroundColor: "{colors.surface-light}"
    textColor: "{colors.text-light}"
    rounded: "{rounded.md}"
    padding: 16px
  status-success:
    backgroundColor: "{colors.surface-light}"
    textColor: "{colors.success}"
    rounded: "{rounded.full}"
    padding: 8px
  status-error:
    backgroundColor: "{colors.danger}"
    textColor: "{colors.surface-light}"
    rounded: "{rounded.md}"
    padding: 8px
  dialog:
    backgroundColor: "{colors.surface-light}"
    textColor: "{colors.text-light}"
    rounded: "{rounded.lg}"
    padding: 24px
---

# Meibook Development UI

## Overview

Meibook Dev is the implementation and verification surface for the bilingual MKAC internal assistant. It currently includes MKAC/HR, MES, WMS, and Research. The interface is intentionally restrained: neutral work surfaces, a dark document sidebar, teal default interactions, and domain accents that clarify context without overpowering operational data.

This is an implementation-derived reference. `frontend/src/styles.css` remains authoritative, and many selectors still use literal values or dark-mode overrides. The tokens above describe stable roles; they do not claim that the CSS is fully tokenized or authorize a broad visual refactor.

Dev is ahead of Production. WMS, advanced report artifacts, staged reveal, report email actions, and related accessibility states are Dev capabilities until deliberately reviewed and promoted.

## Colors

Use teal for default/MKAC interactions, blue for MES, amber for Research, and orange/amber for WMS. Use success and danger colors only for semantic state. WMS availability, freshness, suppression, and verification must remain distinguishable by text and structure as well as color.

Every addition must support light and dark themes. Dark mode combines root variables with selector-level overrides, so a light-only implementation is incomplete. Forced-colors support already exists for selected Dev controls and must not be broken.

The token foreground/background pairs are representative. Run contrast and visual checks on rendered states; this file is not a declaration of complete WCAG compliance.

## Typography

Desktop uses Merriweather with the implemented optical sizing and width variation. Mobile switches to a system sans-serif stack. Technical identifiers, SQL-like values, Lot IDs, process IDs, and filenames use the existing monospace treatment where appropriate.

Keep Vietnamese diacritics and Japanese glyphs readable. Do not translate or visually transform technical codes in ways that change their meaning.

## Layout

The desktop Research shell combines a `304px` document sidebar, fluid workspace, and optional `320px` source panel under a `72px` header.

Responsive boundaries are implementation contracts:

- `1140px`: simplify dense grids and supporting panels.
- `900px`: document sidebar becomes off-canvas; sources become an overlay.
- `760px` and `680px`: Dev report layouts progressively simplify.
- `700px`: compact header, short but descriptive mode labels, horizontally scrollable mode tabs, single-column prompts, and mobile typography.
- `420px`: narrow spacing and secondary content.

Support approximately `320px` width. Preserve `44px` touch targets. Active mobile mode tabs must be revealed inside their own horizontal container; do not use page-level `scrollIntoView` for this behavior.

## Elevation & Depth

Use light borders and restrained shadows for cards and the composer, stronger separation for menus and dialogs, and reduced shadow in dark mode. Report artifacts use component-scoped `--report-*` surface, border, text, muted, and accent roles.

Depth communicates containment or temporary overlay only. Avoid decorative glass effects and stacked shadows.

## Shapes

The common radius is `8px`; use `12px` for prominent dialog/composer surfaces, smaller radii for compact metadata, and full radius for chips/status indicators.

Matrices, charts, KPI groups, and governance panels should inherit established report geometry rather than introducing unrelated card shapes.

## Components

Reuse existing component boundaries:

- `EmployeeLogin` for employee access.
- `ResearchSidebar` for topic/upload scope and documents.
- `ChatInput` for compose, history, send, stop, and attachments.
- `MessageList` for streaming messages, WMS metadata, citations, and suggestions.
- `SourcePreviewDialog` for cited source inspection.

Dev-specific patterns include WMS mode and health states, WMS metadata chips, verification timeline, report KPI groups, chart rows, matrix/heatmap tables, governance and limitation sections, staged artifact reveal, report email action, and cancelled timeline states.

Use semantic HTML first. Preserve visible focus, ARIA tab semantics, polite live regions, labelled dialogs, table captions and row/column scopes, decorative-icon hiding, keyboard navigation, Escape behavior, reduced motion, and forced-colors treatment. These conventions record current practice, not full accessibility certification.

## Do's and Don'ts

**Do**

- Verify light/dark, VI/JA, desktop/mobile, keyboard, reduced-motion, and forced-colors states.
- Reuse established components and `--report-*` roles.
- Keep WMS status, reason, freshness, and limitations explicit in text.
- Preserve operational table readability and source-grounded metadata.
- Mark Dev-only patterns clearly when documenting or promoting them.

**Don't**

- Do not invent colors, radii, shadows, or an icon family per feature.
- Do not use color or animation as the only state signal.
- Do not make light-only changes.
- Do not treat this file as authorization to replace the CSS architecture.
- Do not present WMS or advanced reports as Production capability before reviewed promotion.
