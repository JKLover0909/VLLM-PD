#!/usr/bin/env bash
set -euo pipefail

DOCMIND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$DOCMIND_ROOT/scripts"
BACKEND_DIR="$DOCMIND_ROOT/backend"
FRONTEND_DIR="$DOCMIND_ROOT/frontend"

ENV_NAME="${ENV_NAME:-docmind}"
LOG_DIR="$DOCMIND_ROOT/logs"
PID_DIR="$LOG_DIR/pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

VLLM_PID="$PID_DIR/vllm.pid"
BACKEND_PID="$PID_DIR/backend.pid"
FRONTEND_PID="$PID_DIR/frontend.pid"

usage() {
  cat <<EOF
Usage: $(basename "$0") <start|stop|status> [--no-frontend]

Commands:
  start          Start vLLM + backend (+ frontend by default)
  stop           Stop started services
  status         Show current process status

Options:
  --no-frontend  Do not start Streamlit frontend

Environment variables:
  ENV_NAME       Conda environment name (default: docmind)
EOF
}

init_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "[ERROR] Không tìm thấy conda. Hãy cài Miniconda/Anaconda trước." >&2
    exit 1
  fi

  conda activate "$ENV_NAME"
  export PYTHONNOUSERSITE=1
}

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_service() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  local work_dir="$4"
  local cmd="$5"

  if is_running "$pid_file"; then
    echo "[SKIP] $name đang chạy (PID $(cat "$pid_file"))"
    return
  fi

  echo "[START] $name"
  (
    cd "$work_dir"
    nohup bash -lc "$cmd" >"$log_file" 2>&1 &
    echo $! >"$pid_file"
  )
  echo "       log: $log_file"
}

stop_service() {
  local name="$1"
  local pid_file="$2"

  if ! [[ -f "$pid_file" ]]; then
    echo "[SKIP] $name chưa có PID file"
    return
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    rm -f "$pid_file"
    echo "[SKIP] $name PID file rỗng"
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "[STOP] $name (PID $pid)"
    kill "$pid" || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      echo "       force kill $pid"
      kill -9 "$pid" || true
    fi
  else
    echo "[SKIP] $name không còn chạy"
  fi

  rm -f "$pid_file"
}

show_status() {
  for item in "vLLM:$VLLM_PID" "Backend:$BACKEND_PID" "Frontend:$FRONTEND_PID"; do
    local name="${item%%:*}"
    local pid_file="${item##*:}"
    if is_running "$pid_file"; then
      echo "[RUNNING] $name (PID $(cat "$pid_file"))"
    else
      echo "[STOPPED] $name"
    fi
  done
  echo "Logs: $LOG_DIR"
}

command="${1:-}"
shift || true

start_frontend=1
if [[ "${1:-}" == "--no-frontend" ]]; then
  start_frontend=0
fi

case "$command" in
  start)
    init_conda
    start_service "vLLM" "$VLLM_PID" "$LOG_DIR/vllm.log" "$SCRIPTS_DIR" "bash start_vllm.sh"
    start_service "Backend" "$BACKEND_PID" "$LOG_DIR/backend.log" "$BACKEND_DIR" "python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload"

    if [[ "$start_frontend" -eq 1 ]]; then
      start_service "Frontend" "$FRONTEND_PID" "$LOG_DIR/frontend.log" "$FRONTEND_DIR" "streamlit run app.py --server.port 8501"
    else
      echo "[INFO] Bỏ qua frontend (--no-frontend)"
    fi

    echo "\n[SUCCESS] Hệ thống đã khởi động."
    echo "- Backend:  http://localhost:8001/docs"
    echo "- Frontend: http://localhost:8501"
    echo "- Logs:     $LOG_DIR"
    ;;

  stop)
    stop_service "Frontend" "$FRONTEND_PID"
    stop_service "Backend" "$BACKEND_PID"
    stop_service "vLLM" "$VLLM_PID"
    echo "[SUCCESS] Đã dừng các tiến trình."
    ;;

  status)
    show_status
    ;;

  *)
    usage
    exit 1
    ;;
esac
