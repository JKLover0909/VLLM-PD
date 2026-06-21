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
export MEIBOOK_ENV_FILE="$ENV_FILE"

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

compose up -d --build

port="$(sed -n 's/^MACHINE2_API_PORT=//p' "$ENV_FILE" | tail -n 1)"
port="${port:-8001}"

echo "Waiting for Meibook web/API on port ${port}..."
for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:${port}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

curl -fsS "http://localhost:${port}/health" >/dev/null

echo
echo "Meibook Docker web deployment is running."
echo "Local URL: http://localhost:${port}"

lan_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [[ -n "${lan_ip:-}" ]]; then
  echo "LAN URL:   http://${lan_ip}:${port}"
fi

echo
echo "Useful commands:"
echo "  source .env.docker && MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml ps"
echo "  source .env.docker && MEIBOOK_ENV_FILE=.env.docker docker compose -f docker-compose.web.yml logs -f app"
echo "  ./scripts/docker-index-mkac.sh"
