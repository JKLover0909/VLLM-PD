---
version: alpha
name: Meibook Production UI
description: Implementation-derived visual identity for the Production MKAC, MES, and Research interface.
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

# Meibook Production UI

## Overview

Meibook is a restrained bilingual internal-tool interface for MKAC. The Production UI supports MKAC/HR, MES, and Research. It favors dense, verifiable information over decorative presentation: a pale neutral canvas, white working surfaces, a dark document sidebar, and teal as the default interaction accent.

This file records a stable, implementation-derived subset of the visual system. It does not replace `frontend/src/styles.css`, which remains the source of truth. Many selectors still use literal colors and component-specific dark overrides, so the token inventory above must not be treated as proof that the CSS is fully tokenized.

Use the MKAC logo from `frontend/public/mkac-logo.png` and Lucide React for interface icons. Do not introduce a second icon family without an explicit design decision.

## Colors

Use teal for default actions and MKAC context, blue for MES, and amber for Research. Reserve green and red for semantic success and error/destructive states. Do not use status colors as decoration.

Light mode uses the neutral canvas and white surfaces. Dark mode uses the dark canvas and component-level surface overrides. Every UI change must be checked in both themes; changing a light selector alone is incomplete.

The documented component foreground/background pairs are representative, not a claim of complete WCAG conformance. Validate contrast and state distinctions in the rendered interface.

## Typography

Desktop uses Merriweather with optical sizing and the existing width variation. At the mobile breakpoint, the implementation switches to a system sans-serif stack for compact readability. Code and technical identifiers use the existing monospace stack.

Keep headings clear but compact. Preserve readable line height for Vietnamese diacritics and Japanese text. Do not force identifiers, Lot codes, filenames, or process codes through decorative typography.

## Layout

The desktop Research shell combines a `304px` document sidebar, a fluid workspace, and an optional `320px` source panel under a `72px` header.

Responsive behavior is structural:

- At `1140px`, simplify dense grids and supporting panels.
- At `900px`, move the document sidebar off-canvas and show sources as an overlay.
- At `700px`, compact the header, use short mode labels, switch prompts to one column, and use mobile typography.
- At `420px`, reduce spacing and secondary metrics further.

Support viewports down to approximately `320px`. Preserve at least `44px` touch targets where the current mobile controls establish them. Tables may scroll horizontally; never squeeze operational columns until their values become ambiguous.

## Elevation & Depth

Use thin borders and restrained shadows. Cards and the composer receive subtle lift; menus and dialogs receive stronger separation. Dark mode should reduce or remove bright shadows and rely more on borders and surface contrast.

Avoid stacked shadows, glass effects, and ornamental depth. Elevation communicates containment, focus, or temporary overlay state only.

## Shapes

The dominant radius is `8px`. Use `12px` for prominent dialogs/composer surfaces, smaller radii for compact metadata, and the full radius only for pills, chips, and status dots.

Keep shapes practical and consistent. Do not introduce arbitrary radii when an existing scale value fits.

## Components

Reuse the established React boundaries before creating a parallel pattern:

- `EmployeeLogin` for the employee-code gate.
- `ResearchSidebar` for topic/upload scope and document operations.
- `ChatInput` for compose, send, stop, and Research attachment actions.
- `MessageList` for streaming messages, metadata, citations, and suggestions.
- `SourcePreviewDialog` for source inspection.

Shared patterns also include mode tabs, model selection, source panel, confirmation dialogs, agent timelines, report artifact cards, loading skeletons, upload progress, error banners, and empty states.

Use native controls and semantic elements first. Keep the shared `:focus-visible` outline. Mark decorative icons as hidden from assistive technology, label icon-only controls, expose asynchronous state through the existing polite live regions, and preserve Escape-to-close behavior. These are implementation conventions, not a declaration of complete accessibility compliance.

## Do's and Don'ts

**Do**

- Verify light, dark, Vietnamese, Japanese, desktop, and mobile states.
- Reuse existing components and semantic status treatments.
- Preserve reduced-motion behavior for loading and streaming feedback.
- Add named semantic CSS variables when a visual role becomes reusable.
- Keep citations, tables, metadata, and operational codes legible.

**Don't**

- Do not invent a new palette, icon family, radius, or shadow per feature.
- Do not use animation as the only indication of progress or state.
- Do not make light-only styling changes.
- Do not interpret this document as authorization to refactor all CSS tokens.
- Do not describe Dev-only WMS or advanced report UI as Production capability.
