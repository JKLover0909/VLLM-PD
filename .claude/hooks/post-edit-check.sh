#!/usr/bin/env bash
# Claude Code PostToolUse check for Edit/Write: report pyflakes, JSON syntax,
# and whitespace problems in the file just changed. Read-only; never edits.
set -uo pipefail

# shellcheck source=lib-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib-paths.sh"

payload="$(cat)"
file_path="$(printf '%s' "$payload" | jq -r '.tool_input.file_path // ""')"
[[ -z "$file_path" ]] && exit 0

repo_root="$(meibook_repo_root)"
case "$repo_root" in
  "$MEIBOOK_PROD_ROOT"|"$MEIBOOK_DEV_ROOT") ;;
  *) exit 0 ;;
esac

[[ "$file_path" == /* ]] || file_path="$repo_root/$file_path"
file_path="$(realpath -m -- "$file_path")"
case "$file_path" in
  "$repo_root"/*) ;;
  *) exit 0 ;;
esac
[[ -f "$file_path" ]] || exit 0
is_secret_path "$file_path" && exit 0
rel="${file_path#"$repo_root"/}"
is_runtime_data_rel "$rel" && exit 0

cd "$repo_root" || exit 0
problems=""
add_problem() { problems+="$1"$'\n'; }

case "$rel" in
  *.py)
    if out="$(scripts/meibook-python -m pyflakes "$rel" 2>&1)"; then :; else
      add_problem "pyflakes $rel:"$'\n'"$(head -n 20 <<<"$out")"
    fi ;;
  *.json)
    if out="$(jq empty "$rel" 2>&1)"; then :; else
      add_problem "Invalid JSON in $rel: $out"
    fi ;;
esac

# Tracked files: diff against the index; untracked: whole file against /dev/null.
if git ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
  if ! out="$(git diff --check -- "$rel" 2>&1)"; then
    add_problem "Whitespace problems in $rel:"$'\n'"$(head -n 20 <<<"$out")"
  fi
elif ! git check-ignore -q -- "$rel"; then
  rc=0
  out="$(git diff --no-index --check -- /dev/null "$rel" 2>&1)" || rc=$?
  if (( rc > 1 )); then
    add_problem "Whitespace problems in $rel:"$'\n'"$(head -n 20 <<<"$out")"
  fi
fi

if [[ -n "$problems" ]]; then
  jq -n --arg reason "Post-edit check found problems in $rel. Fix the ones your edit introduced; if a problem pre-dates this session, leave it and note it in progress.md instead of widening scope:"$'\n'"$problems" \
    '{decision: "block", reason: $reason}'
fi
exit 0
