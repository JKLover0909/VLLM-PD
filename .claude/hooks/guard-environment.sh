#!/usr/bin/env bash
# Claude Code PreToolUse guard for Bash: Dev/Production boundary, destructive
# git/Docker operations, data imports, and secret files. Heuristic safety net,
# not a security boundary. Precedence: deny > ask > allow.
set -euo pipefail

# shellcheck source=lib-paths.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib-paths.sh"

payload="$(cat)"
command="$(printf '%s' "$payload" | jq -r '.tool_input.command // ""')"
[[ -z "$command" ]] && exit 0

repo_root="$(meibook_repo_root)"
case "$repo_root" in
  "$MEIBOOK_PROD_ROOT"|"$MEIBOOK_DEV_ROOT") ;;
  *) exit 0 ;;
esac

deny_reason=""
ask_reasons=()
deny() { if [[ -z "$deny_reason" ]]; then deny_reason="$1"; fi; }
ask() { ask_reasons+=("$1"); }

respond() {
  jq -n --arg decision "$1" --arg reason "$2" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: $decision,
      permissionDecisionReason: $reason
    }
  }'
}

prod_container='(^|[=/])meibook-(web|litellm|qdrant|ollama-proxy)$'
prod_port='(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]):(8001|4000|4002|6333)([^0-9]|$)'
volume_flag='[[:space:]](-v|--volumes|-[a-zA-Z]*v[a-zA-Z]*)([[:space:]=]|$)'
config_safe='[[:space:]](-q|--quiet|--services|--volumes|--profiles|--images|--networks|--hash)([[:space:]=]|$)'
dry_run_flag='[[:space:]](--dry-run|--validate-snapshot|-h|--help)([[:space:]=]|$)'
branch_mutation='[[:space:]](-[dDmMcCf]|--delete|--move|--copy|--force)([[:space:]]|$)'
read_only_bases=" ls cat head tail less grep rg find stat wc diff file tree du realpath readlink test [ "
read_only_git=" status log show diff rev-parse rev-list ls-files blame describe "
git_sub=""
readers=" cat tac nl less more head tail grep egrep fgrep rg ag awk gawk sed cut strings xxd od hexdump base64 jq yq cp mv scp rsync tee diff cmp vi vim nvim nano code sort uniq column bat dd openssl gpg zip tar curl wget "

# Strip option prefixes ("--file=x", "@x") so file arguments can be classified.
scan_token() {
  local t="$1"
  [[ "$t" == -*=* ]] && t="${t#*=}"
  printf '%s' "${t#@}"
}

check_compose() {
  local -a a=("$@") files=()
  local k=0 sub="" project=""
  while (( k < ${#a[@]} )); do
    case "${a[$k]}" in
      -f|--file) files+=("${a[$((k+1))]:-}"); k=$((k+2)) ;;
      --file=*) files+=("${a[$k]#--file=}"); k=$((k+1)) ;;
      -f?*) files+=("${a[$k]#-f}"); k=$((k+1)) ;;
      -p|--project-name) project="${a[$((k+1))]:-}"; k=$((k+2)) ;;
      --project-name=*) project="${a[$k]#*=}"; k=$((k+1)) ;;
      --env-file|--profile|--project-directory|--ansi|--progress|--parallel) k=$((k+2)) ;;
      -*) k=$((k+1)) ;;
      *) sub="${a[$k]}"; break ;;
    esac
  done
  local -a rest=("${a[@]:$((k+1))}")
  local r=" ${rest[*]} " f

  if (( ${#files[@]} == 0 )); then
    case "$sub" in
      ""|version|ls) ;;
      *) ask "docker compose without -f resolves the legacy docker-compose.yml (Production ports) or COMPOSE_FILE; use 'docker compose -f docker-compose.dev.yml ...'." ;;
    esac
  fi
  for f in "${files[@]}"; do
    case "${f##*/}" in
      docker-compose.dev.yml) ;;
      docker-compose.web.yml) ask "Production Compose file docker-compose.web.yml (AGENTS.md 0.3: only on an explicit Production request)." ;;
      docker-compose.yml) ask "Legacy docker-compose.yml publishes the Production ports 6333/4000." ;;
      *) ask "Unrecognised Compose file '$f'; verify the target environment first." ;;
    esac
  done
  if [[ -n "$project" && "$project" != "meibook-dev" ]]; then
    ask "Compose project '$project' is not meibook-dev."
  fi
  case "$sub" in
    down|rm)
      if [[ "$r" =~ $volume_flag ]]; then
        deny "docker compose $sub with -v/--volumes deletes volumes (AGENTS.md 0.3)."
      fi ;;
    config)
      if [[ ! "$r" =~ $config_safe ]]; then
        deny "Printing the resolved Compose config can expose secrets; use 'config -q' or 'config --services'."
      fi ;;
  esac
  return 0
}

check_git() {
  local -a a=("$@")
  local j=0 sub="" tok
  while (( j < ${#a[@]} )); do
    case "${a[$j]}" in
      -C|-c|--git-dir|--work-tree|--namespace) j=$((j+2)) ;;
      -*) j=$((j+1)) ;;
      *) sub="${a[$j]}"; break ;;
    esac
  done
  local -a rest=("${a[@]:$((j+1))}")
  local r=" ${rest[*]} "
  git_sub="$sub"
  case "$sub" in
    stash)
      case "${rest[0]:-}" in
        list|show|apply|create|store|branch) ;;
        pop|clear) deny "git stash ${rest[0]} is blocked: the stash stack is shared with the Production checkout and other sessions. Use 'git stash apply <sha>' and drop only your own tagged entry." ;;
        drop) ask "git stash drop: confirm the entry belongs to this session (re-find it by tag with 'git stash list')." ;;
        *)
          if [[ "$r" != *" -m "* && "$r" != *" --message"* ]]; then
            deny "Bare git stash is blocked (shared stash stack). Prefer a WIP commit, or 'git stash push -u -m <unique-tag>' and record its SHA."
          fi ;;
      esac ;;
    clean)
      if [[ "$r" =~ [[:space:]]-[a-zA-Z]*f || "$r" == *" --force "* ]]; then
        deny "git clean --force deletes untracked files; this worktree carries untracked work from earlier sessions. Preview with 'git clean -n' and ask the user."
      fi ;;
    commit|push|pull|merge|rebase|cherry-pick|revert|am)
      ask "git $sub changes history or remote state; AGENTS.md 9 requires an explicit user request." ;;
    reset)
      if [[ "$r" == *" --hard"* || "$r" == *" --merge"* || "$r" == *" --keep"* ]]; then
        ask "git reset --hard/--merge/--keep discards working-tree changes."
      fi ;;
    checkout|switch|restore)
      ask "git $sub can overwrite uncommitted work in this worktree." ;;
    branch)
      if [[ "$r" =~ $branch_mutation ]]; then ask "git branch delete/move/copy/force."; fi ;;
    worktree)
      case "${rest[0]:-}" in
        remove|prune|move) ask "git worktree ${rest[0]} changes checkouts." ;;
      esac ;;
    add)
      for tok in "${rest[@]}"; do
        if is_secret_path "$(scan_token "$tok")"; then deny "Refusing to stage a secret-like file ($tok)."; fi
      done
      case "$r" in
        *" -A "*|*" --all "*|*" . "*|*" -u "*|*" --update "*|*" -f "*|*" --force "*)
          ask "Broad or forced git add can stage other sessions' work or ignored runtime data; stage explicit paths." ;;
      esac ;;
    filter-branch|filter-repo|update-ref|gc|prune|replace)
      ask "git $sub rewrites or prunes repository objects." ;;
  esac
  return 0
}

check_docker() {
  local base="$1"
  shift
  local -a a=("$@")
  local j=0 sub="" tok
  if [[ "$base" == "docker-compose" ]]; then
    check_compose "${a[@]}"
    return 0
  fi
  while (( j < ${#a[@]} )); do
    case "${a[$j]}" in
      -H|--host|--context|-c|--config|-l|--log-level) j=$((j+2)) ;;
      -*) j=$((j+1)) ;;
      *) sub="${a[$j]}"; break ;;
    esac
  done
  local -a rest=("${a[@]:$((j+1))}")
  case "$sub" in
    compose) check_compose "${rest[@]}"; return 0 ;;
    volume)
      case "${rest[0]:-}" in
        rm|remove|prune) deny "docker volume ${rest[0]} deletes volumes; the model caches are shared with Production." ;;
      esac ;;
    system|builder|image|container|network)
      if [[ "${rest[0]:-}" == "prune" ]]; then
        if [[ " ${rest[*]} " == *" --volumes "* ]]; then
          deny "docker $sub prune --volumes deletes volumes shared with Production."
        else
          ask "docker $sub prune removes Docker objects shared with Production."
        fi
      fi ;;
    rmi) ask "docker rmi: image tags such as meibook-litellm:latest are shared by Dev and Production." ;;
  esac
  case "$sub" in
    ps|images|inspect|port|stats|top|version|info) ;;
    *)
      for tok in "${rest[@]}"; do
        if [[ "$tok" =~ $prod_container ]]; then
          ask "Targets Production container '$tok' (no -dev suffix); AGENTS.md 0.3 requires an explicit Production request."
          break
        fi
      done ;;
  esac
  return 0
}

check_segment() {
  # Only a single "<" redirects from a file; here-docs/here-strings are inline text.
  local seg="${1//<<</ __herestring__ }"
  seg="${seg//<</ __heredoc__ }"
  seg="${seg//</ < }"
  local -a words=()
  read -ra words <<< "$seg" || true
  (( ${#words[@]} )) || return 0
  local i n=${#words[@]}
  for (( i = 0; i < n; i++ )); do
    words[i]="${words[i]//[\"\']/}"
  done

  i=0
  while (( i < n )); do
    case "${words[$i]}" in
      env|command|builtin|nohup|time|exec|setsid) i=$((i+1)) ;;
      timeout) i=$((i+2)) ;;
      *)
        if [[ "${words[$i]}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then i=$((i+1)); else break; fi ;;
    esac
  done
  (( i < n )) || return 0
  local base="${words[$i]##*/}"
  local -a args=("${words[@]:$((i+1))}")
  git_sub=""
  local seg_flat=" ${words[*]} " tok tb prev=""

  # Secret files: block content reads (reader commands or "<" redirection).
  local -a scan=("${args[@]}")
  case "$base" in
    grep|egrep|fgrep|rg|ag)
      if [[ " ${args[*]} " != *" -e "* && " ${args[*]} " != *" --regexp"* ]]; then
        for i in "${!scan[@]}"; do
          [[ "${scan[$i]}" == -* ]] && continue
          unset 'scan[i]'
          break
        done
      fi ;;
  esac
  for tok in "${words[@]}"; do
    if [[ "$prev" == "<" ]] && is_secret_path "$tok"; then
      deny "Reading a secret-like file ($tok) is blocked (AGENTS.md 0.3/9). Use the committed *.example templates."
    fi
    prev="$tok"
  done
  if [[ "$readers" == *" $base "* ]]; then
    for tok in "${scan[@]}"; do
      if is_secret_path "$(scan_token "$tok")"; then
        deny "Reading or copying a secret-like file ($tok) is blocked (AGENTS.md 0.3/9). Use the committed *.example templates."
        break
      fi
    done
  fi

  case "$base" in
    git) check_git "${args[@]}" ;;
    docker|docker-compose) check_docker "$base" "${args[@]}" ;;
  esac

  # Production scripts, OAuth init, and data import/index runs.
  local runner=0
  case "$base" in
    meibook-python|python|python3|bash|sh|zsh|docker|docker-compose) runner=1 ;;
  esac
  for tok in "${words[@]}"; do
    tb="${tok##*/}"
    if (( ! runner )) && [[ "$tb" != "$base" ]]; then
      continue
    fi
    case "$tb" in
      docker-deploy.sh|docker-index-mkac.sh|meibook.sh|run_ngrok.sh)
        ask "scripts/$tb targets the Production stack (docker-compose.web.yml / port 8001)." ;;
      init_gmail_oauth.py|init_calendar_oauth.py)
        ask "$tb rewrites OAuth credentials shared with Production (AGENTS.md 0.2)." ;;
    esac
    if [[ "$tb" =~ ^(import_|index_)[A-Za-z0-9_]+\.py$ || "$tb" == "create_msm_mock_database.py" || "$tok" =~ ^scripts\.(import|index)_[A-Za-z0-9_]+$ ]]; then
      if [[ ! "$seg_flat" =~ $dry_run_flag ]]; then
        ask "Data import/index run without --dry-run (AGENTS.md 5): confirm scope, target DB/collection, and a backup."
      fi
    fi
  done

  # Host ports published by Production Compose. Inside a "-dev" container the
  # same numbers are that container's own ports, so skip docker exec there.
  if [[ "$seg_flat" =~ $prod_port ]] && ! [[ "$base" == docker && "$seg_flat" == *"-dev "* ]]; then
    ask "Production host port ${BASH_REMATCH[2]} (Dev uses 8002/4001/6334)."
  fi

  # Production checkout paths from a Dev session (plain reads are allowed).
  if [[ "$repo_root" == "$MEIBOOK_DEV_ROOT" ]] \
    && [[ "$read_only_bases" != *" $base "* ]] \
    && ! [[ "$base" == git && "$read_only_git" == *" $git_sub "* ]]; then
    local resolved
    for tok in "${words[@]}"; do
      [[ "$tok" == *VLLM-PD* ]] || continue
      resolved="$(realpath -m -- "$(scan_token "$tok")" 2>/dev/null || true)"
      if [[ "$resolved" == "$MEIBOOK_PROD_ROOT" || "$resolved" == "$MEIBOOK_PROD_ROOT"/* ]]; then
        ask "Command references the Production checkout ($tok) from the Dev session."
        break
      fi
    done
  fi
  return 0
}

segments="$command"
for sep in '&&' '||' ';' '|' '&' '(' ')' '`'; do
  segments="${segments//"$sep"/$'\n'}"
done
while IFS= read -r seg; do
  check_segment "$seg"
done <<< "$segments"

if [[ -n "$deny_reason" ]]; then
  respond deny "$deny_reason"
  exit 0
fi
if (( ${#ask_reasons[@]} )); then
  reason="$(printf '%s\n' "${ask_reasons[@]}" | awk '!seen[$0]++' | paste -sd ' ')"
  respond ask "$reason"
fi
exit 0
