# Shared path classifiers for the Meibook Claude Code hooks. Sourced, not run.
# These are name-based heuristics (a safety net), not an access-control boundary.

MEIBOOK_PROD_ROOT="/home/jkl/Code/VLLM-PD"
MEIBOOK_DEV_ROOT="/home/jkl/Code/VLLM-PD-dev"

# Resolve the checkout the hook runs for; empty when it cannot be resolved.
meibook_repo_root() {
  local root="${CLAUDE_PROJECT_DIR:-}"
  if [[ -z "$root" ]]; then
    root="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"
  fi
  (cd "$root" 2>/dev/null && pwd -P) || true
}

# Return 0 when the basename looks like a credential, token, or real env file.
# Committed templates (*.example, *.sample, *.template) stay readable.
is_secret_path() {
  local base="${1##*/}"
  base="${base,,}"
  [[ -z "$base" ]] && return 1
  case "$base" in
    *.example|*.example.*|*.sample|*.template) return 1 ;;
    .env|.env.*) return 0 ;;
    *.py|*.js|*.jsx|*.mjs|*.ts|*.tsx|*.md|*.sql|*.sh|*.yml|*.yaml|*.toml|*.html|*.css|*.txt|tokenizer*) return 1 ;;
    *token*|client_secret*|*credentials*.json|*credentials*.json.*|*secret*.json|*.key|*key*.pem|*.p12|*.pfx) return 0 ;;
    id_rsa|id_rsa.*|id_ed25519|id_ed25519.*|.pgpass|.netrc) return 0 ;;
  esac
  return 1
}

# Print "dev:<rel>" or "prod:<rel>" for a path inside one of the two checkouts;
# print nothing for paths outside both.
checkout_relative_path() {
  local resolved
  resolved="$(realpath -m -- "$1" 2>/dev/null)" || return 0
  case "$resolved" in
    "$MEIBOOK_DEV_ROOT"/*) printf 'dev:%s\n' "${resolved#"$MEIBOOK_DEV_ROOT"/}" ;;
    "$MEIBOOK_PROD_ROOT"/*) printf 'prod:%s\n' "${resolved#"$MEIBOOK_PROD_ROOT"/}" ;;
  esac
}

# Runtime state that must not be edited casually: bind-mounted into containers,
# holding HR/MES data, vectors, uploads or logs, or installed dependencies.
is_runtime_data_rel() {
  case "$1" in
    data/*|qdrant_storage/*|uploads/*|logs/*|documents/*|*_processed/*|frontend/dist/*) return 0 ;;
    node_modules/*|*/node_modules/*|.git/*) return 0 ;;
  esac
  return 1
}
