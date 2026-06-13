#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/docker-compose.web.yml"
ENV_FILE="$REPO_ROOT/.env.docker"

cd "$REPO_ROOT"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing .env.docker. Create it first:" >&2
  echo "  cp .env.docker.example .env.docker" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
export VLLM_PD_ENV_FILE="$ENV_FILE"

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose -f "$COMPOSE_FILE" "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose -f "$COMPOSE_FILE" "$@"
  else
    echo "Docker Compose is not available. Install docker compose plugin or docker-compose." >&2
    exit 1
  fi
}

compose run --rm \
  -e ENABLE_AGENT=false \
  app python scripts/index_mkac_documents.py
