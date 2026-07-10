# CLAUDE.md

@AGENTS.md

## Claude Code Specific Rules

- Follow all repository instructions in `AGENTS.md`.
- Respond to the user in Vietnamese unless explicitly asked otherwise.
- For multi-step coding tasks, inspect related files before editing.
- Use a clear todo/checklist internally for complex tasks.
- Before changing API schemas or payloads, inspect both backend and frontend usages.
- After code changes, run the most relevant test, lint, build, or smoke check.
- If verification cannot be run, explain why and provide exact commands for the user to run.
- Do not commit, push, reset, delete data, reindex all documents, clear vector collections, or perform destructive operations unless explicitly requested.
- Summarize changed files, verification results, and remaining risks at the end.
