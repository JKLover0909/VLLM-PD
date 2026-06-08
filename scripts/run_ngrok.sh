#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

NGROK_BIN="${NGROK_BIN:-/snap/bin/ngrok}"
MACHINE2_API_PORT="${MACHINE2_API_PORT:-8001}"
NGROK_RESERVED_DOMAIN="${NGROK_RESERVED_DOMAIN:-}"

if [[ ! -x "$NGROK_BIN" ]]; then
  echo "ngrok binary not found or not executable: $NGROK_BIN" >&2
  exit 1
fi

if [[ -n "$NGROK_RESERVED_DOMAIN" ]]; then
  exec "$NGROK_BIN" http "$MACHINE2_API_PORT" \
    --url "$NGROK_RESERVED_DOMAIN" \
    --log stdout \
    --log-format logfmt
fi

exec "$NGROK_BIN" http "$MACHINE2_API_PORT" \
  --log stdout \
  --log-format logfmt
