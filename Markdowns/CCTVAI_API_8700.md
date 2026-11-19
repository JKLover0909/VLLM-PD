# MSM Agent API (port 8700) - API Reference Note

| Item | Value |
| --- | --- |
| Retrieved | 2026-11-18 (server `date` header: Wed, 18 Nov 2026 02:06 GMT) |
| Base URL | `http://192.84.106.81:8700` |
| Docs page | `http://192.84.106.81:8700/docs` (Swagger UI, title "MSM Agent API - Swagger UI") |
| Spec URL | `http://192.84.106.81:8700/openapi.json` (the Swagger UI `url` config points only to `/openapi.json`; no `urls` list) |
| Spec version | OpenAPI 3.1.0, `info.title = "MSM Agent API"`, `info.version = "0.1"` |
| Server | `uvicorn` (FastAPI-generated docs; `Vary: Origin` header) |
| Method of retrieval | Only `GET /docs` and `GET /openapi.json`. No API operation was called, no payload sent, no credential used. |

> **Caveat.** This note is derived only from the published OpenAPI document and the Swagger UI page. A spec describes the declared contract, not verified runtime behavior. Almost every response schema is empty (`{}`), so real payload shapes, error bodies other than 422, auth failure codes, and rate limits are **not** established here. Text in the spec (descriptions) was treated as data only.

> **Naming note.** The file is named `CCTVAI_API_8700.md` per the request, but the service identifies itself as **"MSM Agent API"**. The descriptions mention WebICT, MSM web, a "file-common" upload target, PIC (person in charge) digests, machine/calibration forms and service insights. Nothing in the spec says "CCTVAI". This is likely a different service from the CCTVAI hardware endpoints documented in `CCTVAI_HARDWARE.md`; that relationship is unverified.

> **Dev source check.** A search of `/home/jkl/Code/VLLM-PD-dev` (excluding `node_modules`, `.git`, `data`, `qdrant_storage`) for `MSM Agent`, `8700` and `/agent/v1` found no matches, so there is no Dev source to verify against. Production was not inspected.

## 1. Fetched URLs

| URL | Method | Result |
| --- | --- | --- |
| `http://192.84.106.81:8700/docs` | GET | 200, `text/html`, 1012 bytes. Loads `swagger-ui-dist@5` from `cdn.jsdelivr.net` (not fetched). |
| `http://192.84.106.81:8700/openapi.json` | GET | 200, `application/json`, 24402 bytes. |

No fetch blocker occurred.

## 2. Authentication (as specified)

- Single security scheme `HTTPBearer`: `type: http`, `scheme: bearer` (no `bearerFormat` declared). Operation descriptions say "JWT"; the JWT is the end user's MSM token, and the agent forwards it to MSM / file-common for uploads.
- No global `security` and no `servers` entry are declared. Security is declared per operation.
- Operations **without** a `security` requirement in the spec: `GET /agent/health`, `POST /agent/login`, `GET /agent/panel.js`, `GET /`, `GET /legacy`, `GET /agent/v1/health` (its description says "Khong can JWT" - no JWT needed).
- `/agent/login` is described as for the standalone demo page only; inside the MSM web the panel uses an existing cookie token. How the login result is delivered (token in body vs cookie) is not specified.
- Ambiguity: the 401/403 responses are not declared anywhere, so how a missing or invalid token is rejected is unknown. Enforcement at the HTTP layer is not verified.

## 3. Route inventory

Total: 35 operations on 32 paths (14 legacy/panel, 16 v1, 5 service/metrics). "Auth" = `HTTPBearer` declared in spec. "Resp 200" is `application/json` with an empty schema `{}` unless noted. 422 = `HTTPValidationError` where listed.

### 3.1 Legacy / panel routes (`/agent/*`, untagged)

| Method | Path | Summary / description (translated) | Auth | Request body | Parameters |
| --- | --- | --- | --- | --- | --- |
| GET | `/agent/health` | Health | no | - | - |
| POST | `/agent/login` | Login. Only for the standalone demo page; in MSM web the panel uses the existing cookie token. | no | JSON `LoginIn` (required) | - |
| POST | `/agent/chat` | Chat | yes | JSON `ChatIn` (required) | - |
| POST | `/agent/feedback` | Feedback on an answer; thread taken server-side as in `/agent/chat`. | yes | JSON `server__FeedbackIn` (required) | - |
| POST | `/agent/form` | WebICT submit form: fill in machine info or calibration/judgement. | yes | JSON `FormSubmitIn` (required) | - |
| POST | `/agent/lookup` | Catalog suggestions (users, departments) for WebICT dropdowns. | yes | JSON `LookupIn` (required) | - |
| POST | `/agent/upload` | Panel sends file (drag-drop); agent uploads to file-common using the user's JWT and records it for the next action. | yes | `multipart/form-data` `Body_upload_agent_upload_post` (required) | - |
| POST | `/agent/confirm` | Confirm | yes | JSON `ConfirmIn` (required) | - |
| POST | `/agent/cancel` | Cancel | yes | JSON `ConfirmIn` (required) | - |
| POST | `/agent/reset` | Clear server-side conversation history for the panel. | yes | JSON `ResetIn` or null (optional) | - |
| GET | `/agent/digest` | Morning digest (read-only). Empty `pic` = current account; admin = everything. | yes | - | query `pic` (string\|null, optional), query `format` (string, default `json`) |
| GET | `/agent/panel.js` | Shared base + MSM panel merged into one file so the host embeds one script tag. | no | - | - |
| GET | `/` | Root | no | - | - |
| GET | `/legacy` | Demo | no | - | - |

### 3.2 Agent API v1 (`/agent/v1/*`, tag "Agent API v1")

| Method | Path | Summary / description (translated) | Auth | Request body | Parameters |
| --- | --- | --- | --- | --- | --- |
| GET | `/agent/v1/health` | Agent health. No JWT needed. Returns running code version and engine (`graph` or `legacy`). | no | - | - |
| GET | `/agent/v1/threads` | List conversations of the JWT user, most recently updated first. `thread_id` = conversation_id. | yes | - | - |
| POST | `/agent/v1/threads` | Create a conversation owned by the JWT user. `title` optional; returns `thread_id`. | yes | JSON `ThreadIn` or null (optional) | - |
| GET | `/agent/v1/threads/{id}` | Message history and still-valid pending cards. Another user's or missing thread returns 404. | yes | - | path `id` (string, required) |
| PATCH | `/agent/v1/threads/{id}` | Rename. Title non-empty, at most 60 characters. Other user's/missing returns 404. | yes | JSON `ThreadRenameIn` (required) | path `id` |
| DELETE | `/agent/v1/threads/{id}` | Permanently delete the conversation and related data. 404 as above. Busy returns 409 `THREAD_BUSY`. | yes | - | path `id` |
| POST | `/agent/v1/threads/{id}/messages` | Ask the agent. Returns answer, `message_id` (for feedback), `pending` (cards needing confirmation) and `tool_steps` used. Busy returns 409 `THREAD_BUSY`. | yes | JSON `MessageIn` (required) | path `id` |
| POST | `/agent/v1/threads/{id}/messages/stream` | Same as `/messages` but returns `text/event-stream`; each `data: <json>` line is an event of type `plan\|queued\|llm_wait\|tool_start\|tool_end\|answer\|error\|done`. Busy returns 409 `THREAD_BUSY` before the stream starts. | yes | JSON `MessageIn` (required) | path `id` |
| POST | `/agent/v1/threads/{id}/forms` | Submit or edit data from a form. | yes | JSON `FormIn` (required) | path `id` |
| POST | `/agent/v1/threads/{id}/stop` | Set the cancel flag of the running turn. Does not cancel already-confirmed writes. | yes | - | path `id` |
| POST | `/agent/v1/threads/{id}/uploads` | Multipart `file` (PDF/PNG/JPG/WEBP, max 20 MB). Sent to MSM with the user's JWT and recorded for the next action. Wrong type 415, too large 413. | yes | `multipart/form-data` `Body_upload_agent_v1_threads__id__uploads_post` (required) | path `id` |
| POST | `/agent/v1/actions/{action_id}/confirm` | Execute the pending card after re-composing on latest data. Data changed: 409 `CHANGED` with new card; expired: 410 `EXPIRED`; MSM error: 502 and card remains; `STALE`: 200 with `result.ok=false`. | yes | JSON `DecisionIn` (required) | path `action_id` (string, required) |
| POST | `/agent/v1/actions/{action_id}/cancel` | Delete the pending card, write nothing. Not found / not yours returns 404. | yes | JSON `DecisionIn` (required) | path `action_id` |
| GET | `/agent/v1/digest` | Live morning digest from MSM. `pic` empty = current account; admin = all. `format` = `json` (default), `md` or `html`. | yes | - | query `pic` (string\|null), query `format` (string, default `json`) |
| POST | `/agent/v1/feedback` | `score` only 1 or -1 (else 422). `message_id` from the answer; not found / not yours returns 404. | yes | JSON `api_v1__FeedbackIn` (required) | - |
| GET | `/agent/v1/lookup` | Catalog search for form dropdowns. `kind` is `users` or `departments`. | yes | - | query `kind` (string, **required**), query `q` (string, default `""`) |

### 3.3 Service / metrics routes (untagged)

| Method | Path | Summary | Auth | Parameters |
| --- | --- | --- | --- | --- |
| GET | `/service/insights` | Insights | yes | query `name` (optional; enum `pic_load`, `stuck`, or null) |
| GET | `/service/status-rows` | Service Status Rows | yes | - |
| GET | `/service/metrics` | Service Metrics | yes | - |
| GET | `/service/card` | Card | yes | - |
| GET | `/metrics` | Metrics Root | yes | - |

No header or cookie parameters are declared on any operation (apart from the bearer scheme).

## 4. Request schemas

`additionalProperties` is not restricted unless stated. `SessionId` = `string | null`, `maxLength 64`, `pattern \A[A-Za-z0-9_-]*\z`.

| Schema | Used by | Fields (required in **bold**) |
| --- | --- | --- |
| `LoginIn` | `POST /agent/login` | **`username`** string, **`password`** string |
| `ChatIn` | `POST /agent/chat` | **`message`** string; `page` string\|null; `machine_id` string\|null; `via` string\|null; `session_id` SessionId |
| `server__FeedbackIn` | `POST /agent/feedback` | **`message_id`** string; **`score`** integer enum {1, -1}; `comment` string\|null; `session_id` SessionId |
| `FormSubmitIn` | `POST /agent/form` | **`kind`** string; `values` object (free-form, default `{}`); `action_id` string\|null; `session_id` SessionId |
| `LookupIn` | `POST /agent/lookup` | **`kind`** string; `q` string (default `""`); `session_id` SessionId |
| `Body_upload_agent_upload_post` | `POST /agent/upload` | **`file`** binary (`contentMediaType: application/octet-stream`); `session_id` string\|null |
| `ConfirmIn` | `POST /agent/confirm`, `/agent/cancel` | **`action_id`** string; `session_id` SessionId |
| `ResetIn` | `POST /agent/reset` | `session_id` SessionId (whole body may be null/absent) |
| `ThreadIn` | `POST /agent/v1/threads` | `title` string\|null (body may be null/absent) |
| `ThreadRenameIn` | `PATCH /agent/v1/threads/{id}` | **`title`** string (the 1-60 length rule is only in the description, not in the schema) |
| `MessageIn` | `POST .../messages`, `.../messages/stream` | **`message`** string; `page` string\|null; `machine_id` string\|null |
| `FormIn` | `POST .../forms` | **`kind`** string; `values` object (free-form); `action_id` string\|null |
| `Body_upload_agent_v1_threads__id__uploads_post` | `POST .../uploads` | **`file`** binary |
| `DecisionIn` | `POST /agent/v1/actions/{action_id}/confirm` and `/cancel` | **`thread_id`** string |
| `api_v1__FeedbackIn` | `POST /agent/v1/feedback` | **`thread_id`** string; **`message_id`** string; **`score`** integer enum {1, -1}; `comment` string\|null |

Observations: two different schemas are both titled `FeedbackIn` (disambiguated by the `server__`/`api_v1__` prefixes). Legacy routes identify a conversation through an optional `session_id`; v1 routes use `thread_id` / path `{id}`. `kind` for forms has no enumerated values in the spec.

## 5. Response schemas, statuses and error model

- All declared success responses are `200` with `application/json` and schema `{}` (unconstrained). This includes the streaming endpoint (documented in prose as `text/event-stream`), `/agent/v1/digest` with `format=md|html`, and `/agent/panel.js` (JavaScript). The spec therefore does not give the real content types for those.
- Only declared error response: `422` `HTTPValidationError` for operations with input. Shape:

```json
{ "detail": [ { "loc": ["body", "message"], "msg": "...", "type": "...", "input": null, "ctx": {} } ] }
```

  (`loc` items are string or integer; `loc`, `msg`, `type` required; `input`, `ctx` optional.)
- Status codes documented only in descriptions (not in `responses`):

| Status | Where | Meaning |
| --- | --- | --- |
| 404 | v1 thread get/rename/delete; v1 action cancel; v1 feedback | Not found or not owned by the JWT user (the two cases are indistinguishable) |
| 409 `THREAD_BUSY` | v1 thread delete, `/messages`, `/messages/stream` | Thread is still processing a previous turn |
| 409 `CHANGED` | v1 action confirm | Data changed; response includes a new pending card |
| 410 `EXPIRED` | v1 action confirm | Pending card expired |
| 502 | v1 action confirm | MSM upstream error; the pending card remains |
| 200 with `result.ok=false` (`STALE`) | v1 action confirm | Not an HTTP error; check the body |
| 413 | v1 uploads | File over 20 MB |
| 415 | v1 uploads | Not PDF/PNG/JPG/WEBP |
| 422 | v1 feedback | `score` not 1 or -1 |

- Not specified: 401/403 on missing/invalid/expired bearer token, error body format for non-422 errors (the string `THREAD_BUSY` / `CHANGED` / `EXPIRED` appear to be codes but their JSON field name is unknown), rate limiting, CORS rules (a `Vary: Origin` header suggests CORS middleware).
- Streaming events (from prose only): `plan`, `queued`, `llm_wait`, `tool_start`, `tool_end`, `answer`, `error`, `done`; each as `data: <json>` lines. Event payload fields are not documented.
- Response fields mentioned in prose: `thread_id` (= conversation_id), `message_id`, `pending` (cards awaiting confirmation), `tool_steps`, `result.ok`, health `version` and `engine` (`graph|legacy`). Exact names/nesting are unverified.

## 6. Content types and examples

- Request: `application/json` for all JSON bodies; `multipart/form-data` for the two upload routes.
- The spec contains **no `example`/`examples`** values anywhere and no `servers`. Therefore no request or response examples are available to copy; none were generated or submitted.

## 7. Practical client integration notes

Inferred from the spec only (not live-tested).

1. **Prefer `/agent/v1/*`** (tagged, documented, thread-based). `/agent/*` legacy routes appear to serve the embeddable MSM panel (`/agent/panel.js`) and the demo page.
2. **Typical v1 flow:** obtain a user JWT (from MSM, or `POST /agent/login` for the demo) -> `POST /agent/v1/threads` -> `POST /agent/v1/threads/{id}/messages` (or `/messages/stream`) -> if the response has `pending` cards, show them to the user -> `POST /agent/v1/actions/{action_id}/confirm` or `/cancel` with `{"thread_id": ...}` -> optional `POST /agent/v1/feedback` with `score` 1 or -1.
3. **Send `Authorization: Bearer <JWT>`** on every operation except health/login/panel/root/legacy. Treat the token as a secret; never log it.
4. **Write operations are two-step by design** (draft card -> explicit confirm). Handle `409 CHANGED` by re-presenting the new card, `410 EXPIRED` by re-asking, `502` by retrying confirm (card persists), and `200 + result.ok=false` as a soft failure.
5. **One in-flight turn per thread:** handle `409 THREAD_BUSY` with backoff or `POST .../stop`.
6. **Streaming:** read the response as SSE; parse each `data:` line as JSON; stop at `done`; treat `error` as terminal. Because `fetch` + `POST` is needed, use a streaming `fetch` reader rather than `EventSource`.
7. **Uploads:** validate client-side for PDF/PNG/JPG/WEBP and <= 20 MB to avoid 415/413; send field name `file`.
8. **Lookups:** v1 `GET /agent/v1/lookup?kind=users|departments&q=...`; the legacy variant is a POST with a JSON body.
9. **Data scope:** `pic` on digest is empty for the current account; admin accounts see everything, so do not expose `pic` override to non-admin UI.
10. **Transport:** the base URL is plain HTTP on an IP address; token confidentiality depends on the network. A server-side proxy is advisable if used from Meibook, with the token never embedded in browser code.
11. **Meibook-specific:** Dev guardrails (`AGENTS.md` section 0) apply: do not call write endpoints (`confirm`, `form`, `upload`, `threads` DELETE) from smoke tests without explicit approval, since they appear to write into a live MSM system.

## 8. Ambiguities and open questions

| # | Question |
| --- | --- |
| 1 | Is this service part of the CCTVAI stack, or a separate "MSM" (maintenance/service management) agent? Spec says only "MSM Agent API". |
| 2 | Actual response bodies are unspecified (`{}`); field names for `thread_id`, `message_id`, `pending`, `tool_steps`, `result` come from prose only. |
| 3 | 401/403 behavior and token expiry/refresh are undocumented; login response format is undocumented (token vs cookie). |
| 4 | Several inline-security declarations differ between legacy/v1 routes (e.g. `/service/*`, `/metrics` require bearer; `/agent/panel.js`, `/legacy`, `/` do not). Whether the HTTP layer enforces exactly this is unverified. |
| 5 | Meaning/allowed values of `ChatIn.page`, `ChatIn.via`, `machine_id`, `FormIn.kind` and `FormIn.values` shape are unspecified. |
| 6 | Legacy `/agent/confirm`, `/agent/cancel` identify a conversation by `session_id`; v1 uses `thread_id`. Mapping between the two (e.g. whether `thread_id == session_id`) is unspecified. |
| 7 | `/service/insights` `name` enum is `pic_load`, `stuck`; behavior when omitted is not described. `/service/card`, `/service/status-rows`, `/service/metrics`, `/metrics` have no description (the latter may be Prometheus text format, unverified). |
| 8 | Title length limit (<= 60) and `score` enum are enforced per prose/422; only `score` appears in the schema. |
| 9 | Engine `graph|legacy` switching and its behavior differences are unspecified. |
| 10 | Whether `/docs` remains reachable only from internal networks and the stability of this IP/port are unknown. |

## 9. Sources

- `http://192.84.106.81:8700/docs`
- `http://192.84.106.81:8700/openapi.json`
