#!/usr/bin/env bash
# Claude Code SessionStart hook: print a short, read-only state summary.
# Stdout becomes session context. Never fails the session.
set -uo pipefail

repo_root="${CLAUDE_PROJECT_DIR:-}"
if [[ -z "$repo_root" ]]; then
  repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"
fi
repo_root="$(cd "$repo_root" 2>/dev/null && pwd -P)" || exit 0
cd "$repo_root" || exit 0

case "$repo_root" in
  /home/jkl/Code/VLLM-PD) env_label="PRODUCTION checkout (docker-compose.web.yml)" ;;
  /home/jkl/Code/VLLM-PD-dev) env_label="Dev checkout (docker-compose.dev.yml)" ;;
  *) exit 0 ;;
esac

branch="$(git branch --show-current 2>/dev/null || echo '?')"
head="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
porcelain="$(git status --porcelain=v1 2>/dev/null || true)"
modified="$(grep -cv '^??' <<<"$porcelain" || true)"
untracked="$(grep -c '^??' <<<"$porcelain" || true)"
[[ -z "$porcelain" ]] && modified=0 && untracked=0

echo "[Meibook session harness]"
echo "Repo: $repo_root ($env_label)"
echo "Branch: $branch @ $head; uncommitted: $modified changed, $untracked untracked"

if [[ -f feature_list.json ]] && command -v jq >/dev/null 2>&1; then
  counts="$(jq -r '[.features[].status] | group_by(.) | map("\(.[0])=\(length)") | join(", ")' feature_list.json 2>/dev/null)"
  active="$(jq -r '[.features[] | select(.status == "in_progress") | .id] | join(", ")' feature_list.json 2>/dev/null)"
  echo "feature_list.json: ${counts:-unreadable}${active:+; in_progress: $active}"
fi

if [[ -f progress.md ]]; then
  echo "--- progress.md: Current state (truncated) ---"
  awk '/^## Current state/{on=1; next} on && /^##/{exit} on{print}' progress.md \
    | sed '/^[[:space:]]*$/d' | head -n 18
fi

echo "Follow CLAUDE.md -> Session Harness. This summary does not replace the preflight."
exit 0
