# Upstream provenance

- Repository: https://github.com/anthropics/skills
- Source path: `skills/mcp-builder`
- Commit: `9d2f1ae187231d8199c64b5b762e1bdf2244733d`
- Commit date: 2026-07-01
- License: Apache License 2.0 (`LICENSE.txt` in this directory)
- Installed: 2026-07-13

## Local policy

This directory is vendored and pinned. Meibook already uses an application MCP
loader in `src/agent/mcp_client.py`; inspect that integration before proposing a
new server. Prefer read-only tools, explicit destructive/idempotent annotations,
least privilege, pinned dependencies, and confirmation before write actions.

The bundled evaluation harness calls the Anthropic API and contains an older
default model identifier. Do not run it automatically. If evaluation is needed,
review dependencies and update the model/API usage against current official docs
before execution.
