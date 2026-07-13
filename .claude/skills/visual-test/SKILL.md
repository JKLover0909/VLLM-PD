---
name: visual-test
description: Visually verify Meibook's React/Vite interface through the real browser surface, including responsive layouts, dark mode, keyboard access, SSE progress, Report Agent timeline/artifacts, and console errors. Use after frontend changes or when asked to inspect the running UI.
license: MIT
metadata:
  author: Meibook project, adapted from Microsoft Fluent UI visual-test
  version: "1.0.0"
---

# Meibook Visual Test

Verify the running application as a user sees it. Prefer observable browser
evidence over source-only claims.

## Safety boundary

- Do not start destructive services, reindex documents, clear Qdrant, mutate MES
  or HR databases, send email, or create Calendar events.
- Use guest/test identities and synthetic prompts only; never expose real HR/MES
  data in screenshots or logs.
- Do not install Playwright, browser binaries, npm packages, or other dependencies
  unless the user explicitly approves.
- Reuse an already-running safe app when available. Before starting anything,
  inspect the repository's run instructions and active ports.
- Use `/tmp` for temporary scripts, screenshots, downloads, and browser profiles.
- Do not publish or upload screenshots without explicit approval.

## Prerequisites

1. Read `CLAUDE.md` and `AGENTS.md`.
2. Read `frontend/package.json` and the relevant API/UI files.
3. Check whether a compatible Playwright installation or browser-driving tool is
   already available. If not, report `BLOCKED` rather than installing it.
4. Build only when needed using the repository command:

   ```bash
   npm --prefix frontend run build
   ```

5. If the app must be launched, prefer the reviewed helper in
   `../webapp-testing/scripts/with_server.py`, run `--help` first, and pass only
   commands that were verified for this repository. Starting Docker Compose or
   any service with production mounts requires user approval.

## Core flows

Choose the smallest flow that executes the changed UI.

### General chat

- Authenticate with an approved guest/test identity.
- Switch between MKAC, MES, and Research only as needed.
- Submit a synthetic question.
- Observe status, response streaming, sources, stop/copy actions, and errors.

### Report Agent

Use a synthetic supported prompt such as:

```text
Lập báo cáo top 3 lỗi sản xuất tháng 6/2026
```

When a safe fixture backend is available, verify:

1. Report Agent status appears.
2. Timeline steps move queued → running → done/error.
3. Artifact card renders KPI, sections, observations, and limitations.
4. Download points to `/reports/{uuid}` and returns escaped HTML.
5. Unsupported dynamic report prompts produce a normal refusal with no timeline,
   SQL tools, artifact card, or error banner.

Do not drive this flow against production data merely to obtain screenshots.

### Source preview

- Open a synthetic/indexed test citation.
- Verify keyboard focus, dialog close behavior, loading/error states, and image or
  snippet fallback.

## Viewports and themes

Capture at least the viewport relevant to the change. For broad UI work use:

- Mobile: 390 × 844
- Tablet: 768 × 1024
- Desktop: 1440 × 900

Check both light and dark mode when colors, borders, elevation, or contrast change.
Ensure there is no unintended horizontal scrolling.

## Accessibility probes

- Tab through all controls in logical order.
- Confirm visible focus indicators.
- Activate controls with Enter/Space.
- Close dialogs with Escape and verify focus returns to the trigger.
- Inspect accessible names for icon-only buttons.
- Respect `prefers-reduced-motion` where animations are involved.
- Use the `accessibility` skill for a full WCAG review.

## Browser evidence

Capture:

- One screenshot showing the main changed state.
- Any adjacent failure or empty state relevant to the change.
- Browser console errors and failed network requests.
- The exact viewport, theme, language, mode, and synthetic input used.

Do not treat a successful frontend build as visual verification.

## Report format

```markdown
## Visual verification: <surface>

**Verdict:** PASS | FAIL | BLOCKED
**Environment:** <URL, viewport, theme, language, mode>

1. ✅ Happy path → <observed result>
2. 🔍 Adjacent/error probe → <observed result>
3. ♿ Keyboard/accessibility probe → <observed result>

**Evidence:** <screenshot paths and concise console/network capture>
**Findings:** <issues, friction, or none>
```

A PASS requires execution through the real browser surface. If no safe running
surface or browser driver is available, report BLOCKED with the exact missing
prerequisite.
