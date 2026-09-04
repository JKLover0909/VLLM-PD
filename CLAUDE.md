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
- Codex will review your output once you are done.
