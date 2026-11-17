# CLAUDE.md

@AGENTS.md

## Claude Code Specific Rules

- Follow all repository instructions in `AGENTS.md`, especially section 0 about the Dev/Production boundary.
- Determine the environment from the actual working directory, branch, available Compose file, resolved Compose config, container labels/mounts, and host ports; never assume the local topology solely because this documentation was merged or copied.
- In the documented local topology, `/home/jkl/Code/VLLM-PD-dev` on `dev` uses `docker-compose.dev.yml`, ports `8002/4001/6334`, and containers suffixed `-dev`. If that file is absent/invalid or the preflight does not match, stop and report instead of falling back to Production Compose.
- Before any state-changing Docker, import/index, migration, or data command, verify the working directory, current branch, Compose file/project, target container, resolved host port, and bind-mount source.
- Treat `/home/jkl/Code/VLLM-PD`, `docker-compose.web.yml`, containers without `-dev`, and the host ports resolved from Production Compose (defaults `8001/4000/6333`) as Production targets; do not operate on them unless the user explicitly requests a Production action.
- Never merge or copy Dev runtime data (SQLite, Qdrant storage, uploads, logs, documents, previews, or credentials) into Production. Code/config/schema changes and data migrations have separate lifecycles.
- Respond to the user in Vietnamese unless explicitly asked otherwise.
- For multi-step coding tasks, inspect related files before editing.
- Use a clear todo/checklist internally for complex tasks.
- Before changing API schemas or payloads, inspect both backend and frontend usages.
- After code changes, run the most relevant test, lint, build, or smoke check.
- If verification cannot be run, explain why and provide exact commands for the user to run.
- Do not commit, push, reset, delete data, reindex all documents, clear vector collections, or perform destructive operations unless explicitly requested.
- Summarize changed files, verification results, and remaining risks at the end.

## Session Harness

Files: `progress.md` (session handoff log), `feature_list.json` (feature status), `scripts/harness-check.sh` (read-only validation), `.claude/hooks/` (guards). The SessionStart hook prints a short state summary; it does not replace the steps below.

### Session start

1. Run the read-only part of the AGENTS.md §0.3 preflight: `pwd`, `git branch --show-current`, `git status --short`.
2. Read `progress.md` → "Current state" and the newest session entry.
3. Look up the feature you will touch in `feature_list.json` (status, entry points, tests).
4. Run that feature's tests BEFORE editing. If they already fail, record that in `progress.md` first.

### During work

- One feature/fix per session. The Dev worktree carries uncommitted work from earlier sessions: never revert, reformat, or stage files you did not change.
- A behaviour change needs a test that fails before and passes after, or a written reason why not.
- Never use bare `git stash` / `git stash pop`; the stash stack is shared with the Production checkout and other sessions.
- Do not invoke scripts through `bash -c` / `sh -c` (blocked by the Python guard hook); run `scripts/harness-check.sh` directly.

### Validation matrix

| Change touches | Minimum check |
| --- | --- |
| `src/**/*.py` | targeted `scripts/meibook-python -m pytest <test> -q` + `scripts/harness-check.sh quick` |
| `src/api/schemas.py` / SSE payload | + `tests/test_query_transport_compatibility.py`, `tests/test_query_guard_stream.py`, `npm --prefix frontend test` |
| `frontend/src/**` | `npm --prefix frontend test`; run `npm run build` ONLY when you intend to change the live Dev UI (`frontend/dist` is bind-mounted into `meibook-web-dev`) |
| `litellm_config.yaml` | `tests/test_litellm_routing_config.py`, `tests/test_model_routes.py` |
| `docker-compose*.yml`, env examples | `docker compose -f docker-compose.dev.yml config -q` |
| docs only | `git diff --check` |
| before handoff / review / merge | `scripts/harness-check.sh full` |

### Hooks (safety net, not a security boundary)

- `require-meibook-python.sh`: denies host Python/pip/pytest/pyflakes outside `scripts/meibook-python`, nested shells, `sudo`, `xargs`.
- `guard-environment.sh`: denies volume deletion, `git clean -f`, bare `git stash` / `stash pop` / `stash clear`, printing the full resolved Compose config, and reading/copying/staging secret files; asks before Production targets (`docker-compose.web.yml`, legacy `docker-compose.yml`, containers without `-dev`, host ports 8001/4002/6333, Production scripts, writes to the Production checkout), bare `docker compose`, git commit/push/pull/merge/rebase/reset --hard/checkout/restore, broad `git add`, import/index runs without `--dry-run`, and OAuth init.
- `guard-files.sh`: denies Read/Edit/Write on secret files (`*.example` templates stay readable); asks before writing runtime data (`data/`, `qdrant_storage/`, `uploads/`, `logs/`, `documents/`, `*_processed/`, `frontend/dist/`, `node_modules/`) or the Production checkout.
- `lib-paths.sh`: shared path classifiers sourced by the hooks above.
- `post-edit-check.sh`: reports pyflakes / JSON / whitespace problems in the file just edited.
- An "ask" prompt means the user must approve that specific action; do not reword the command to avoid the prompt.

### Session end (handoff)

- Prepend a session entry to `progress.md` and refresh "Current state".
- Update `feature_list.json` `status` / `verified` only for checks that actually ran.
- Never write secrets, employee IDs/names, or real HR/MES values into `progress.md` or `feature_list.json`.
