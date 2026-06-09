#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMAND="${1:-start}"
BUILD_FRONTEND="${2:-}"

cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

API_PORT="${MACHINE2_API_PORT:-8001}"
NGROK_API_URL="${NGROK_API_URL:-http://localhost:4040/api/tunnels}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/vllm-pd.sh start [--build]
  ./scripts/vllm-pd.sh restart [--build]
  ./scripts/vllm-pd.sh stop
  ./scripts/vllm-pd.sh status
  ./scripts/vllm-pd.sh logs
EOF
}

build_frontend_if_needed() {
  if [[ "$BUILD_FRONTEND" == "--build" || ! -f "$REPO_ROOT/frontend/dist/index.html" ]]; then
    echo "Building React frontend..."
    npm --prefix "$REPO_ROOT/frontend" install
    npm --prefix "$REPO_ROOT/frontend" run build
  fi
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local tries="${3:-30}"

  for ((i = 1; i <= tries; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name is healthy: $url"
      return 0
    fi
    sleep 2
  done

  echo "$name did not become healthy in time: $url" >&2
  return 1
}

get_ngrok_url() {
  local tries="${1:-30}"
  local public_url=""

  for ((i = 1; i <= tries; i++)); do
    public_url="$(
      curl -fsS "$NGROK_API_URL" 2>/dev/null \
        | jq -r '.tunnels[]? | select(.proto == "https") | .public_url' \
        | head -n 1 \
        || true
    )"
    if [[ -n "$public_url" && "$public_url" != "null" ]]; then
      printf '%s\n' "$public_url"
      return 0
    fi
    sleep 2
  done

  return 1
}

start_system() {
  build_frontend_if_needed

  echo "Starting Docker services..."
  docker compose up -d

  echo "Starting user services..."
  systemctl --user daemon-reload
  systemctl --user enable --now vllm-pd-api.service
  systemctl --user enable --now vllm-pd-ngrok.service

  wait_for_url "FastAPI/Web" "http://localhost:${API_PORT}/health" 45
  wait_for_url "LiteLLM" "http://localhost:4000/health/liveliness" 20 || true
  wait_for_url "Qdrant" "http://localhost:6333/healthz" 20 || true

  local public_url=""
  if public_url="$(get_ngrok_url 30)"; then
    wait_for_url "Public ngrok API" "${public_url%/}/health" 30 || true
  else
    echo "Ngrok did not publish a URL in time." >&2
  fi

  echo
  echo "System started."
  echo "Local web:  http://localhost:${API_PORT}"
  if [[ -n "$public_url" ]]; then
    echo "Public web: ${public_url}"
  fi
}

stop_system() {
  echo "Stopping user services..."
  systemctl --user stop vllm-pd-ngrok.service vllm-pd-api.service || true

  echo "Stopping Docker services..."
  docker compose stop
}

status_system() {
  docker compose ps
  echo
  systemctl --user --no-pager --full status vllm-pd-api.service vllm-pd-ngrok.service || true
}

logs_system() {
  journalctl --user -u vllm-pd-api.service -u vllm-pd-ngrok.service -n 160 --no-pager
}

case "$COMMAND" in
  start)
    start_system
    ;;
  restart)
    stop_system
    start_system
    ;;
  stop)
    stop_system
    ;;
  status)
    status_system
    ;;
  logs)
    logs_system
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
