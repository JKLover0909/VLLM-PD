# Meibook Development Context

## Purpose and scope

Meibook is MKAC's bilingual Vietnamese/Japanese internal assistant. This Dev checkout is the implementation and verification surface for four user-facing modes:

- `mkac`: administrative, policy, and employee-directory questions.
- `mes`: production-quality questions over a validated MES snapshot.
- `wms`: fail-closed process-warehouse/current-balance questions over a dedicated WMS snapshot.
- `research`: source-grounded research over selected internal document topics or session uploads.

The primary path is React/Vite SPA → FastAPI/SSE → domain routing → SQLite or Qdrant → LiteLLM/model only where required. Deterministic and source-grounded paths take precedence over generative inference.

WMS and Dev's advanced report interface are in-progress Dev capabilities. Their presence here is not a Production guarantee.

## Mandatory environment boundary

| Property | Development | Production |
|---|---|---|
| Checkout | `/home/jkl/Code/VLLM-PD-dev` | `/home/jkl/Code/VLLM-PD` |
| Expected branch | `dev` | `main` |
| Compose | `docker-compose.dev.yml` | `docker-compose.web.yml` |
| Default web/LiteLLM/Qdrant ports | `8002 / 4001 / 6334` | `8001 / 4000 / 6333` |
| Containers | `-dev` suffix | no `-dev` suffix |
| Runtime data | isolated Dev lifecycle | Production lifecycle |

Before any state-changing Docker, import, index, migration, or data command, verify the actual working directory, branch, Compose file/project, resolved ports, target container, labels, and bind mounts. Stop when preflight does not match; never fall back to Production Compose.

Only reviewed code, config, schema, and documentation may be promoted. Never merge or copy Dev SQLite databases, Qdrant storage, uploads, documents, previews, logs, credentials, OAuth tokens, or generated artifacts into Production. Data migration is a separate, explicitly authorized lifecycle.

The stacks can still share a Docker daemon, GPU, caches, and upstream model hosts. Separate ports/containers/data do not mean complete physical isolation.

## Domain modes and routing

### MKAC / HR (`mode=mkac`)

Employee-directory questions use structured SQLite data when possible. Other internal questions use `mkac_knowledge` retrieval and citations. Public web search, when enabled, is a fallback and must not be presented as authoritative internal policy.

Japanese UI requests may be translated to Vietnamese before Vietnamese-first processing and translated back. Preserve names, employee identifiers, and quoted source facts.

### MES (`mode=mes`)

MES reads `data/mes.sqlite`, an imported snapshot rather than realtime MES. Prefer deterministic/template queries. The SQL Agent is a constrained fallback through the semantic model, read-only validation, allowlisted views, row limits, and timeout.

Do not invent quantities, interpret unknown status/process/route codes, equate error record count with total error quantity, or infer realtime production state. A missing business-code master means the meaning is unknown, not that the code is invalid.

### WMS (`mode=wms`, Dev extension)

WMS reads only `data/mes_wms.sqlite`; it must never fall back to MES SQLite, the MES SQL Agent, live MES API, or RAG. Deterministic intents remain the first route. When they cannot answer or suppress a question because semantics are unverified, the dedicated read-only `wms_sql_agent` may generate SQL over allowlisted WMS views.

Current balance is authoritative at grain `(process_id, item_code)`. A current Lot is not a meaningful business key. Legacy archive and raw transaction audit have separate domains, freshness, and semantics.

The operator has explicitly accepted that LLM-generated WMS SQL may aggregate quantities across unknown UOMs, compare incompatible current/legacy eras, analyze raw transactions as movement proxies, or derive unverified trend/min-stock/WIP/bottleneck proxies. Such queries are permitted but remain read-only, view-allowlisted, row-limited, and time-limited. Their answers must carry `SQL_AGENT_ANSWER_UNVERIFIED`, expose every queried evidence domain, and describe these values as unverified calculations or proxies rather than contract-verified business facts. Disabled, unavailable, incompatible, or query-error snapshots still fail closed before any LLM call.

Public WMS metadata is allowlisted and aggregate. Do not expose raw rows, SQL, internal paths, sensitive identifiers, or chain-of-thought. Report artifacts remain private to the authorized session/employee.

Keep Japanese WMS questions in the original language where translation could distort item/process identifiers.

### Research (`mode=research`)

The primary shared corpus is `docjp_knowledge`, filtered by selected topic/category. Session upload/demo retrieval is a separate scope and collection lifecycle. Do not mix context or sources across topic and upload scopes.

Japanese source documents are retrieved in their original language; avoid unnecessary double translation. Answers must cite the selected source scope and explicitly say when support is insufficient.

## Core domain language

- **Employee gate**: `employee_id` authorization for internal MKAC/MES/WMS access. Guest behavior is constrained and not full employee identity.
- **Lot**: MES production Lot identifier.
- **Product / item code**: production or material identifier; preserve exact spelling.
- **Error event**: one recorded error row, not automatically total error quantity.
- **Error catalog**: error-name mapping keyed by the full business key, including process and error type where required.
- **Process**: recorded production or warehouse process code; do not infer a name without verified mapping.
- **Snapshot**: imported point-in-time SQLite data, never realtime by default.
- **WMS domain/evidence**: current balance, legacy archive, and raw audit are separate contracts.
- **Research topic/session**: retrieval boundaries that must remain isolated.
- **Source/citation preview**: indexed source metadata and processed page assets used to verify an answer.
- **`conversation_context`**: short client-supplied context for resolvers; not durable memory or a data authority.

## Non-negotiable invariants

- Internal policy and HR claims must be grounded in structured data or cited internal sources. If evidence is absent, say so.
- MES/WMS answers must be derived from validated snapshot rows; never let an LLM fabricate operational data.
- Prefer deterministic, read-only behavior before model-generated planning or SQL.
- Preserve Lot, product, process, file, topic, and employee identifiers across VI/JA handling.
- Do not log or place real HR/MES/WMS PII, credentials, OAuth tokens, raw Qdrant payloads, or private source rows in tests, prompts, reports, or documentation.
- `data/`, runtime SQLite, Qdrant storage, uploads, logs, previews, credentials, and generated index output are runtime artifacts, not source code.
- API or payload changes require inspection of backend schemas and frontend consumers together.
- Source preview paths must remain inside allowlisted processed roots and respect host/container path mapping.
- WMS status events expose completed deterministic milestones, not model reasoning.

## Dev UI extensions

Dev includes UI behavior ahead of Production: WMS mode/status/freshness metadata, WMS verification timeline, horizontally revealed mobile mode tabs, report charts and matrices, governance/limitation sections, staged artifact reveal, report email action, cancelled states, and additional forced-colors/reduced-motion handling.

Treat these as Dev implementation details until corresponding source, contract, and tests are deliberately promoted.

## Repository map

- `frontend/src/`: React UI, behavior, and CSS source of truth.
- `src/api/`: FastAPI routes, schemas, configuration, auth gates, SSE contracts.
- `src/auth/`: employee-directory and structured HR behavior.
- `src/integrations/`: MES/WMS contracts and queries, constrained SQL agent, external actions.
- `src/rag/`: parsing, retrieval, vector storage, prompts, and media paths.
- `src/actions/`: report and action workflows.
- `config/`: semantic models, manifests, topics, and quick-answer configuration.
- `database/schema/`: committed database contracts; generated databases remain runtime data.
- `scripts/`: import, index, evaluation, and operational entry points.
- `tests/`: behavioral contracts and regressions.

## Authoritative sources

Use this priority when documents disagree:

1. `AGENTS.md` and `CLAUDE.md` for working rules and environment safety.
2. Actual checkout/branch, resolved Compose config, containers, ports, and mounts for target identity.
3. `src/api/schemas.py` and endpoint code for public contracts.
4. Domain contract/code/schema/semantic model and tests for behavior.
5. `frontend/src/styles.css`, React components, and `DESIGN.md` for implemented UI.
6. `README.md` and `Markdowns/` as maps/runbooks that may contain historical or volatile details.

For WMS specifically, verify `src/integrations/mes_wms_contract.py`, `src/integrations/mes_wms_database.py`, `database/schema/mes_wms.sql`, importer validation, and WMS tests. Their current uncommitted/local status means they are not yet a Production commitment.

## Change workflow

Implement and verify here before proposing Production promotion. Use `scripts/meibook-python` for host Python/test/lint commands. Run the most relevant tests, lint, frontend build, and safe Dev runtime smoke checks. Do not commit, push, deploy, import, migrate, reindex, or operate Production unless explicitly requested.

This context deliberately excludes live counts, model/IP/tunnel availability, snapshot dates, container health, credentials, and generated artifact IDs. Verify those from the actual runtime when needed.
