#!/usr/bin/env bash
# Claude Code PreToolUse guard for host Python commands in this repository.
set -euo pipefail

payload="$(cat)"
command="$(printf '%s' "$payload" | jq -r '.tool_input.command // ""')"
repo_root="${CLAUDE_PROJECT_DIR:-}"
if [[ -z "$repo_root" ]]; then
  repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"
fi
repo_root="$(cd "$repo_root" 2>/dev/null && pwd -P)" || exit 0

case "$repo_root" in
  /home/jkl/Code/VLLM-PD)
    env_name="meibook"
    ;;
  /home/jkl/Code/VLLM-PD-dev)
    env_name="meibook-dev"
    ;;
  *)
    exit 0
    ;;
esac

expected_python="/home/jkl/miniconda3/envs/$env_name/bin/python"

respond_deny() {
  local reason="$1"
  jq -n --arg reason "$reason" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $reason
    }
  }'
}

# The wrapper is the only supported host-Python entrypoint because it selects
# the checkout environment and applies the CPU defaults. Reject indirect shell
# launchers rather than trying to parse nested command strings recursively.
nested_shell='(^|[;&|(`][&|]?[[:space:]]*)((/usr/bin/env|env)[[:space:]]+)?(bash|sh|zsh|dash|ksh)([[:space:]]+-[^[:space:]]+)*[[:space:]]+-[^[:space:]]*c([[:space:]]|$)'
indirect_launcher='(^|[;&|(`][&|]?[[:space:]]*)(sudo|xargs)([[:space:]]|$)'
if [[ "$command" =~ $nested_shell ]] || [[ "$command" =~ $indirect_launcher ]]; then
  respond_deny "Nested shell, sudo, and xargs launchers are not allowed in Meibook Bash commands because they can bypass the host-Python guard."
  exit 0
fi

shell_boundary='(^|[;&|(`][&|]?[[:space:]]*)'
launcher_prefix='((command|builtin|nohup|time)([[:space:]]+-[^[:space:]]+)*[[:space:]]+)*'
env_prefix='((/usr/bin/env|env)([[:space:]]+-[^[:space:]]+)*[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]+[[:space:]]+)*'
base_activate="${shell_boundary}${launcher_prefix}${env_prefix}conda[[:space:]]+activate[[:space:]]+base([[:space:];|&)\`]|$)"
conda_run="${shell_boundary}${launcher_prefix}${env_prefix}conda[[:space:]]+run([[:space:];|&)\`]|$)"
if [[ "$command" =~ $base_activate ]]; then
  respond_deny "Conda base is forbidden for Meibook host work. Use scripts/meibook-python."
  exit 0
fi
if [[ "$command" =~ $conda_run ]]; then
  respond_deny "Do not use conda run for Meibook host work. Use scripts/meibook-python so the checkout selects its required environment."
  exit 0
fi

python_entry="${shell_boundary}${launcher_prefix}${env_prefix}([^[:space:];|&()\`]+/)?(python([0-9.]*)?|pip([0-9.]*)?|pytest|pyflakes)([[:space:];|&)\`]|$)"
if [[ "$command" =~ $python_entry ]]; then
  respond_deny "Use scripts/meibook-python for all host Python, pip, pytest, or pyflakes commands. Direct interpreters can bypass the required environment or CPU defaults: $expected_python."
  exit 0
fi

# The wrapper fails closed itself, but this gives an earlier, clearer message
# without blocking unrelated Bash commands needed to bootstrap the environment.
if [[ "$command" == *"scripts/meibook-python"* ]] && [[ ! -x "$expected_python" ]]; then
  respond_deny "Missing required Conda environment '$env_name': $expected_python. Create it before running host Python."
  exit 0
fi

exit 0
