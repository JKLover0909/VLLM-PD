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

VLLM_PD_ENV_FILE="$ENV_FILE" \
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" run --rm \
  -e ENABLE_AGENT=false \
  app python scripts/index_mkac_documents.py
