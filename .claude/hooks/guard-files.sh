#!/usr/bin/env bash
# Claude Code PreToolUse guard for Read/Edit/Write/NotebookEdit file paths.
# Denies secret files; asks before writing runtime data or the other checkout.
set -euo pipefail

# shellcheck source=lib-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib-paths.sh"

payload="$(cat)"
tool="$(printf '%s' "$payload" | jq -r '.tool_name // ""')"
file_path="$(printf '%s' "$payload" | jq -r '.tool_input.file_path // .tool_input.notebook_path // ""')"
[[ -z "$file_path" ]] && exit 0

repo_root="$(meibook_repo_root)"
case "$repo_root" in
  "$MEIBOOK_PROD_ROOT"|"$MEIBOOK_DEV_ROOT") ;;
  *) exit 0 ;;
esac

respond() {
  local decision="$1" reason="$2"
  jq -n --arg decision "$decision" --arg reason "$reason" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: $decision,
      permissionDecisionReason: $reason
    }
  }'
}

if is_secret_path "$file_path"; then
  respond deny "Blocked $tool on a secret-like file ($file_path). AGENTS.md 0.3/9: do not read or show .env, OAuth tokens, or credentials. Use the committed *.example templates instead."
  exit 0
fi

case "$tool" in
  Edit|Write|NotebookEdit) ;;
  *) exit 0 ;;
esac

[[ "$file_path" == /* ]] || file_path="$repo_root/$file_path"
location="$(checkout_relative_path "$file_path")"
[[ -z "$location" ]] && exit 0
checkout="${location%%:*}"
rel="${location#*:}"

if [[ "$repo_root" == "$MEIBOOK_DEV_ROOT" && "$checkout" == "prod" ]]; then
  respond ask "This session runs in the Dev checkout but $tool targets the PRODUCTION checkout ($file_path). Approve only if the user explicitly asked for a Production change."
  exit 0
fi

if is_runtime_data_rel "$rel"; then
  respond ask "$tool targets runtime data or installed dependencies ($rel). These paths are bind-mounted into the running stack or hold HR/MES data and are not source code. Approve only if the user asked for this data change."
  exit 0
fi

exit 0
