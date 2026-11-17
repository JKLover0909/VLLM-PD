#!/usr/bin/env bash
# Read-only validation for agent session handoffs. No Docker state changes,
# no frontend build, no data access.
#   scripts/harness-check.sh quick   whitespace, pyflakes on src/, frontend tests, JSON
#   scripts/harness-check.sh full    quick + backend pytest suite + Dev Compose syntax
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODE="${1:-quick}"
cd "$REPO_ROOT"

case "$MODE" in
  quick|full) ;;
  *)
    echo "Usage: scripts/harness-check.sh [quick|full]" >&2
    exit 2
    ;;
esac

declare -a results=()
failed=0

run_step() {
  local name="$1"
  shift
  echo "==> $name"
  if "$@"; then
    results+=("PASS  $name")
  else
    results+=("FAIL  $name")
    failed=1
  fi
}

# git diff --check ignores untracked files; check them against /dev/null.
# --no-index exits 1 for "has differences" and >1 for whitespace problems.
check_whitespace() {
  local rc=0 status=0 file
  git diff --check || status=1
  while IFS= read -r -d '' file; do
    rc=0
    git diff --no-index --check -- /dev/null "$file" || rc=$?
    if (( rc > 1 )); then
      status=1
    fi
  done < <(git ls-files -z --others --exclude-standard)
  return "$status"
}

check_pyflakes() {
  scripts/meibook-python -m pyflakes \
    src/api/*.py src/rag/*.py src/integrations/*.py src/auth/*.py \
    src/actions/*.py src/i18n/*.py src/agent/*.py
}

check_json() {
  local status=0 file
  for file in feature_list.json .claude/settings.json; do
    if [[ -f "$file" ]]; then
      jq empty "$file" || status=1
    fi
  done
  return "$status"
}

check_pytest() {
  scripts/meibook-python -m pytest tests/ -q \
    --ignore=tests/test_mes_integration.py --ignore=tests/test_mkac_pipeline.py
}

# Syntax only; never print the resolved config (it can contain secrets).
check_dev_compose() {
  if [[ "$REPO_ROOT" != "/home/jkl/Code/VLLM-PD-dev" ]]; then
    echo "skip: not the local Dev checkout ($REPO_ROOT)"
    return 0
  fi
  docker compose -f docker-compose.dev.yml config -q
}

run_step "whitespace (git diff --check + untracked)" check_whitespace
run_step "pyflakes src/" check_pyflakes
run_step "frontend node --test" npm --prefix frontend test --silent
run_step "JSON syntax" check_json

if [[ "$MODE" == "full" ]]; then
  run_step "backend pytest (default suite)" check_pytest
  run_step "Dev Compose config -q" check_dev_compose
fi

echo
echo "Harness check ($MODE) in $REPO_ROOT:"
printf '  %s\n' "${results[@]}"
exit "$failed"
