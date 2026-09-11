#!/usr/bin/env bash
set -euo pipefail

# Deterministic llama.cpp canary state machine for changed upstream pins.
#
# Usage: llama-canary-agent-repair.sh [upstream-sha]
#
# The wrapper owns prepare -> build -> certify -> publish. An agent may repair
# a failed phase, but it never decides whether a gate passed and never receives
# repository-write credentials. Every agent edit sends the candidate back
# through prepare and the complete build before another certification attempt.
# The branch and PR appear once, after success or a bounded terminal failure.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_SHA="${1:-${UPSTREAM_SHA_INPUT:-latest}}"
if [[ "$UPSTREAM_SHA" == "latest" || -z "$UPSTREAM_SHA" ]]; then
  UPSTREAM_SHA="$(git ls-remote https://github.com/ggml-org/llama.cpp.git master | awk '{print $1}')"
fi
if [[ ! "$UPSTREAM_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "refusing to run the canary against a non-40-hex upstream SHA: $UPSTREAM_SHA" >&2
  exit 1
fi

cd "$ROOT"

OLD_SHA="$(tr -d '[:space:]' < third_party/llama.cpp/upstream.txt)"
PIN_FILE="$ROOT/third_party/llama.cpp/upstream.txt"
AGENT_MODEL="${CANARY_AGENT_MODEL:-zai-coding-plan/glm-5.3-flash}"
MAX_REPAIR_TURNS="${CANARY_REPAIR_MAX_TURNS:-2}"
REPAIR_BUDGET_SECONDS="${CANARY_REPAIR_BUDGET_SECONDS:-41400}"
PUBLISH_RESERVE_SECONDS="${CANARY_PUBLISH_RESERVE_SECONDS:-1800}"
REPAIR_TURN_TIMEOUT_SECONDS="${CANARY_REPAIR_TURN_TIMEOUT_SECONDS:-3600}"
REPAIR_TOTAL_BUDGET_SECONDS="${CANARY_REPAIR_TOTAL_BUDGET_SECONDS:-5400}"
RUN_ID="${GITHUB_RUN_ID:-manual-$(date +%s)}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
RUN_KEY="${RUN_ID}-${RUN_ATTEMPT}"
BRANCH="llama-canary/repair-${RUN_KEY}-${UPSTREAM_SHA:0:10}"
STATE_DIR="$ROOT/.deps/llama-canary-state-${RUN_KEY}"
TARGET_SHA_FILE="$ROOT/.deps/llama-canary-target-sha"
PREPARE_LOG="$STATE_DIR/prepare.log"
BUILD_LOG="$STATE_DIR/build.log"
CERTIFY_LOG="$STATE_DIR/certify.log"
PR_BODY="$STATE_DIR/pr-body.md"
UPSTREAM_SUMMARY="$STATE_DIR/upstream-summary.md"
GIT_ASKPASS_SCRIPT="$STATE_DIR/git-askpass.sh"
FAMILY_BATTERY_RUN_ID="${FAMILY_BATTERY_RUN_ID:-${RUN_KEY}}"
PLAN_PATH="$ROOT/target/family-battery/$FAMILY_BATTERY_RUN_ID/policy-plan.json"
STARTED_AT="$(date +%s)"
DEADLINE_AT="$((STARTED_AT + REPAIR_BUDGET_SECONDS))"
PUBLISHED_SHA=""
CERTIFIED_SHA=""
FAILED_PHASE=""
PREPARE_REPAIR_TURNS=0
BUILD_REPAIR_TURNS=0
CERTIFY_REPAIR_TURNS=0
AGENT_REPAIR_SECONDS_USED=0

if [[ ! "$MAX_REPAIR_TURNS" =~ ^[0-9]+$ ]]; then
  echo "CANARY_REPAIR_MAX_TURNS must be a non-negative integer" >&2
  exit 1
fi
if [[ ! "$REPAIR_BUDGET_SECONDS" =~ ^[0-9]+$ || ! "$PUBLISH_RESERVE_SECONDS" =~ ^[0-9]+$ ]] \
    || (( REPAIR_BUDGET_SECONDS <= PUBLISH_RESERVE_SECONDS )); then
  echo "the canary budget must be numeric and exceed the publication reserve" >&2
  exit 1
fi
if [[ ! "$REPAIR_TURN_TIMEOUT_SECONDS" =~ ^[0-9]+$ \
    || ! "$REPAIR_TOTAL_BUDGET_SECONDS" =~ ^[0-9]+$ ]] \
    || (( REPAIR_TURN_TIMEOUT_SECONDS <= 0 || REPAIR_TOTAL_BUDGET_SECONDS <= 0 )); then
  echo "the repair turn timeout and total repair budget must be positive integers" >&2
  exit 1
fi
for required_name in LLAMA_STAGE_BUILD_DIR HF_CACHE GITHUB_REPOSITORY; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "${required_name} is not set; cannot run the changed-pin canary" >&2
    exit 1
  fi
done
if [[ -z "${CANARY_REPAIR_TOKEN:-}" ]]; then
  echo "CANARY_REPAIR_TOKEN is not set; cannot publish the terminal canary PR" >&2
  exit 1
fi

mkdir -p "$STATE_DIR" "$(dirname "$PLAN_PATH")"
rm -f "$PREPARE_LOG" "$BUILD_LOG" "$CERTIFY_LOG" "$PR_BODY" "$UPSTREAM_SUMMARY"
printf '%s\n' "$UPSTREAM_SHA" > "$TARGET_SHA_FILE"
cat > "$GIT_ASKPASS_SCRIPT" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  Username*) printf '%s\n' 'x-access-token' ;;
  Password*) printf '%s\n' "${CANARY_REPAIR_TOKEN:?}" ;;
  *) exit 1 ;;
esac
EOF
chmod 700 "$GIT_ASKPASS_SCRIPT"

if ! command -v opencode >/dev/null 2>&1; then
  echo "opencode CLI not found on runner; install opencode-ai on the family-certify image" >&2
  exit 1
fi
if [[ -z "${OPENCODE_API_KEY:-}" && -z "${NEMOTRON_API_KEY:-}" ]]; then
  if [[ ! -s "${HOME}/.local/share/opencode/auth.json" ]] \
      && ! opencode auth list 2>/dev/null | grep -Eq '[1-9][0-9]* credentials'; then
    echo "no agent credentials: set OPENCODE_API_KEY/NEMOTRON_API_KEY or run 'opencode auth login' on the runner" >&2
    exit 1
  fi
fi
gh_repair() {
  GH_TOKEN="$CANARY_REPAIR_TOKEN" "$@"
}

check_repair_token_permissions() {
  local login default_branch head_sha probe_branch probe_ref
  login="$(gh_repair gh api user --jq .login 2>/dev/null)" || {
    echo "preflight: CANARY_REPAIR_TOKEN does not authenticate" >&2
    return 1
  }
  default_branch="$(gh_repair gh api "repos/${GITHUB_REPOSITORY:?}" --jq .default_branch 2>/dev/null)"
  head_sha="$(gh_repair gh api "repos/${GITHUB_REPOSITORY:?}/branches/${default_branch}" --jq .commit.sha 2>/dev/null)"
  probe_branch="canary-repair-token-preflight-${RUN_KEY}"
  probe_ref="refs/heads/${probe_branch}"
  if ! gh_repair gh api --method POST "repos/${GITHUB_REPOSITORY:?}/git/refs" \
      -f ref="$probe_ref" -f sha="$head_sha" >/dev/null 2>&1; then
    echo "preflight: identity '${login}' cannot write refs on ${GITHUB_REPOSITORY}; CANARY_REPAIR_TOKEN needs Contents: Read and write" >&2
    return 1
  fi
  if ! gh_repair gh api --method DELETE \
      "repos/${GITHUB_REPOSITORY:?}/git/refs/heads%2F${probe_branch}" >/dev/null 2>&1; then
    echo "preflight: WARNING: could not delete temporary ref ${probe_ref}" >&2
  fi
  echo "preflight: repair token identity '${login}' verified read+write on ${GITHUB_REPOSITORY}"
}

redact_token() {
  python3 -c 'import os, sys; token = os.environ["CANARY_REPAIR_TOKEN"]; sys.stdout.write(sys.stdin.read().replace(token, "***redacted***"))'
}

remaining_work_seconds() {
  local remaining
  remaining="$((DEADLINE_AT - $(date +%s) - PUBLISH_RESERVE_SECONDS))"
  (( remaining > 0 )) || return 1
  printf '%s\n' "$remaining"
}

remaining_repair_seconds() {
  local remaining
  remaining="$((REPAIR_TOTAL_BUDGET_SECONDS - AGENT_REPAIR_SECONDS_USED))"
  (( remaining > 0 )) || return 1
  printf '%s\n' "$remaining"
}

run_bounded() {
  local label="$1" seconds
  shift
  if ! seconds="$(remaining_work_seconds)"; then
    echo "$label cannot start: internal canary deadline reached; publication reserve is active" >&2
    return 124
  fi
  python3 scripts/run-command-with-timeout.py \
    --seconds "$seconds" --label "$label" -- "$@"
}

run_bounded_for() {
  local label="$1" maximum_seconds="$2" seconds
  shift 2
  if ! seconds="$(remaining_work_seconds)"; then
    echo "$label cannot start: internal canary deadline reached; publication reserve is active" >&2
    return 124
  fi
  if (( maximum_seconds < seconds )); then
    seconds="$maximum_seconds"
  fi
  python3 scripts/run-command-with-timeout.py \
    --seconds "$seconds" --label "$label" -- "$@"
}

run_logged() {
  local label="$1" log="$2"
  shift 2
  run_bounded "$label" "$@" > >(tee -a "$log") 2>&1
}

check_repair_token_permissions

# Persistent runners retain nested llama.cpp worktree registrations and /tmp
# checkouts. Remove only the known canary scratch state before agent turns.
git -C "$ROOT/.deps/llama.cpp" worktree prune >/dev/null 2>&1 || true
rm -rf /tmp/llama-old-pin /tmp/llama-repair /tmp/llama-repair-* 2>/dev/null || true

agent_turn() {
  local prompt="$1" started finished elapsed heartbeat_pid repair_remaining status
  if ! repair_remaining="$(remaining_repair_seconds)"; then
    echo "agent repair cannot start: ${REPAIR_TOTAL_BUDGET_SECONDS}s aggregate repair budget exhausted" >&2
    return 124
  fi
  if (( REPAIR_TURN_TIMEOUT_SECONDS < repair_remaining )); then
    repair_remaining="$REPAIR_TURN_TIMEOUT_SECONDS"
  fi
  started="$(date +%s)"
  set -m
  # shellcheck disable=SC2016
  env -i PATH="$PATH" bash -c '
    ROOT="$1"
    started="$2"
    while sleep 600; do
      newest="$(find "$ROOT/.deps/llama.cpp" -type f -newer "$ROOT/third_party/llama.cpp/upstream.txt" -print -quit 2>/dev/null || true)"
      printf "heartbeat: agent repair running for %dm; recent worktree activity: %s\n" \
        "$(( ($(date +%s) - started) / 60 ))" "${newest:-none observed yet}"
    done
  ' heartbeat "$ROOT" "$started" &
  heartbeat_pid=$!
  set +m
  set +e
  run_bounded_for "agent repair turn" "$repair_remaining" env \
    -u GH_TOKEN -u GITHUB_TOKEN -u CANARY_REPAIR_TOKEN \
    opencode run --auto --model "$AGENT_MODEL" "$prompt"
  status=$?
  finished="$(date +%s)"
  elapsed="$((finished - started))"
  AGENT_REPAIR_SECONDS_USED="$((AGENT_REPAIR_SECONDS_USED + elapsed))"
  set -e
  kill -- "-$heartbeat_pid" 2>/dev/null || kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  echo "agent repair turn used ${elapsed}s; aggregate ${AGENT_REPAIR_SECONDS_USED}s/${REPAIR_TOTAL_BUDGET_SECONDS}s"
  return "$status"
}

write_repair_pin() {
  scripts/update-llama-pin.sh "$UPSTREAM_SHA"
}

verify_repair_pin() {
  local pin
  pin="$(tr -d '[:space:]' < "$PIN_FILE")"
  if [[ "$pin" != "$UPSTREAM_SHA" ]]; then
    echo "candidate pin is $pin, expected $UPSTREAM_SHA" >&2
    return 1
  fi
}

run_prepare() {
  local prepared_upstream
  : > "$PREPARE_LOG"
  echo "state-machine phase: prepare" | tee -a "$PREPARE_LOG"
  write_repair_pin >>"$PREPARE_LOG" 2>&1 || return 1
  verify_repair_pin >>"$PREPARE_LOG" 2>&1 || return 1
  run_logged "apply llama.cpp patch queue" "$PREPARE_LOG" \
    scripts/prepare-llama.sh pinned || return 1
  prepared_upstream="$(tr -d '[:space:]' < "$ROOT/.deps/llama.cpp/.mesh-llm-upstream-sha")"
  if [[ "$prepared_upstream" != "$UPSTREAM_SHA" ]]; then
    echo "prepared upstream is $prepared_upstream, expected $UPSTREAM_SHA" | tee -a "$PREPARE_LOG" >&2
    return 1
  fi
}

run_full_build() {
  local archive arches
  : > "$BUILD_LOG"
  echo "state-machine phase: build" | tee -a "$BUILD_LOG"
  run_logged "complete patched llama.cpp build" "$BUILD_LOG" env \
    LLAMA_STAGE_UPSTREAM_TESTS=ON uv run --no-project --with jinja2==3.1.6 -- \
    arch -arm64 bash scripts/build-llama.sh -DCMAKE_OSX_ARCHITECTURES=arm64 \
    || return 1
  archive="$LLAMA_STAGE_BUILD_DIR/src/libllama.a"
  arches="$(lipo -archs "$archive" 2>/dev/null || true)"
  if [[ "$arches" != "arm64" ]]; then
    echo "candidate native archive must be arm64, got: ${arches:-missing}" | tee -a "$BUILD_LOG" >&2
    return 1
  fi
  run_logged "generated model-family patch check" "$BUILD_LOG" \
    scripts/check-skippy-generated-family-patch.sh || return 1
  run_logged "stage runtime crate build" "$BUILD_LOG" \
    cargo build -p skippy-runtime -p skippy-server -p skippy-model-package -p skippy-correctness \
    || return 1
  if [[ "${LLAMA_UPSTREAM_CANARY_SMOKE:-1}" != "0" \
      && "${LLAMA_UPSTREAM_CANARY_SMOKE:-1}" != "false" ]]; then
    run_logged "Skippy smoke tests" "$BUILD_LOG" scripts/skippy-ci-smoke.sh || return 1
  fi
}

run_certification() {
  : > "$CERTIFY_LOG"
  echo "state-machine phase: certify" | tee -a "$CERTIFY_LOG"
  run_logged "parity manifest validation" "$CERTIFY_LOG" \
    python3 scripts/skippy-llama-parity.py --llama-src .deps/llama.cpp validate \
    || return 1
  run_logged "full family certification plan" "$CERTIFY_LOG" \
    python3 scripts/plan-family-battery.py \
      --manifest ci/llama-canary/family-certified.json \
      --cadence llama-bump \
      --shard-count 1 \
      --check-cache \
      --cache-root "$HF_CACHE" \
      --output "$PLAN_PATH" \
    || return 1
  if [[ "${LLAMA_UPSTREAM_CANARY_SMOKE:-1}" != "0" \
      && "${LLAMA_UPSTREAM_CANARY_SMOKE:-1}" != "false" ]]; then
    run_logged "live package-v2 matrix" "$CERTIFY_LOG" env \
      FAMILY_BATTERY_RUN_ID="$FAMILY_BATTERY_RUN_ID" \
      SKIPPY_CANARY_LIVE_MATRIX_BACKEND="${SKIPPY_CANARY_LIVE_MATRIX_BACKEND:-metal}" \
      SKIPPY_CANARY_LIVE_MATRIX_ROOT="$ROOT/target/family-battery/$FAMILY_BATTERY_RUN_ID" \
      scripts/skippy-canary-live-matrix.sh --prepare || return 1
  fi
  run_logged "full supported-family certification" "$CERTIFY_LOG" env \
    FAMILY_BATTERY_RUN_ID="$FAMILY_BATTERY_RUN_ID" \
    scripts/skippy-family-battery.sh --skip-build --plan "$PLAN_PATH"
}

phase_log() {
  case "$1" in
    prepare) printf '%s\n' "$PREPARE_LOG" ;;
    build) printf '%s\n' "$BUILD_LOG" ;;
    certify) printf '%s\n' "$CERTIFY_LOG" ;;
  esac
}

phase_turns() {
  case "$1" in
    prepare) printf '%s\n' "$PREPARE_REPAIR_TURNS" ;;
    build) printf '%s\n' "$BUILD_REPAIR_TURNS" ;;
    certify) printf '%s\n' "$CERTIFY_REPAIR_TURNS" ;;
  esac
}

increment_phase_turns() {
  case "$1" in
    prepare) PREPARE_REPAIR_TURNS=$((PREPARE_REPAIR_TURNS + 1)) ;;
    build) BUILD_REPAIR_TURNS=$((BUILD_REPAIR_TURNS + 1)) ;;
    certify) CERTIFY_REPAIR_TURNS=$((CERTIFY_REPAIR_TURNS + 1)) ;;
  esac
}

repair_prompt() {
  local phase="$1" log turn
  log="$(phase_log "$phase")"
  turn="$(( $(phase_turns "$phase") + 1 ))"
  printf 'The llama.cpp canary %s phase failed at upstream %s (repair turn %s of %s for this phase).

Read ci/llama-canary/agent-repair-prompt.md and the repository instructions it names. Follow that runbook for this wrapper-owned prepare -> build -> certify -> publish state machine. Fix the root cause minimally. Do not weaken, skip, or narrow any gate. Use focused checks while repairing; the deterministic wrapper will restart at prepare, run the complete build, and run the full supported-family certification before it can publish success. Leave changes local. Do not push, open a PR, or use GitHub credentials.

Failure evidence (tail):

%s' "$phase" "$UPSTREAM_SHA" "$turn" "$MAX_REPAIR_TURNS" \
    "$(tail -n 100 "$log" 2>/dev/null || echo '(no phase output captured)')"
}

current_pr() {
  gh_repair gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number' 2>/dev/null || true
}

commit_terminal_tree() {
  local outcome="$1"
  git checkout -B "$BRANCH"
  git add -A
  if ! git diff --cached --quiet; then
    if [[ "$outcome" == "certified" ]]; then
      git commit -m "fix(llama): certify upstream ${UPSTREAM_SHA:0:10}" \
        -m "Prepare, build, and run the full supported-family certification through the deterministic canary state machine."
    else
      git commit -m "fix(llama): preserve failed canary at ${UPSTREAM_SHA:0:10}" \
        -m "Preserve the bounded terminal state from the ${FAILED_PHASE} phase for human diagnosis."
    fi
  fi
}

publish_terminal_branch() {
  local outcome="$1"
  commit_terminal_tree "$outcome"
  PUBLISHED_SHA="$(git rev-parse HEAD)"
  if [[ "$outcome" == "certified" ]]; then
    CERTIFIED_SHA="$PUBLISHED_SHA"
  fi
  if ! GIT_ASKPASS="$GIT_ASKPASS_SCRIPT" GIT_TERMINAL_PROMPT=0 \
      git push "https://github.com/${GITHUB_REPOSITORY}.git" \
      "HEAD:refs/heads/${BRANCH}" 2> >(redact_token >&2); then
    echo "ERROR: could not push ${BRANCH}; the identity behind CANARY_REPAIR_TOKEN needs Contents and pull-request write access" >&2
    return 1
  fi
}

write_upstream_summary() {
  if SKIPPY_CI_SMOKE="${LLAMA_UPSTREAM_CANARY_SMOKE:-1}" \
      scripts/summarize-llama-upstream.sh "$OLD_SHA" "$UPSTREAM_SHA" "$ROOT/.deps/llama.cpp" \
      | awk 'BEGIN { include = 1 } /^## Validation$/ { include = 0 } include { print }' \
      > "$UPSTREAM_SUMMARY"; then
    return 0
  fi
  printf '%s\n' \
    '## Upstream Summary' \
    '' \
    'The automated upstream summary was unavailable; inspect the candidate pin and patch queue directly.' \
    > "$UPSTREAM_SUMMARY"
}

write_pr_body() {
  local outcome="$1"
  write_upstream_summary
  {
    echo "Automated llama.cpp upstream canary for \`${UPSTREAM_SHA}\`."
    echo
    echo "- Previous pin: \`${OLD_SHA}\`"
    echo "- Candidate pin: \`${UPSTREAM_SHA}\`"
    echo "- Workflow run: \`${RUN_KEY}\`"
    echo "- Terminal commit: \`${PUBLISHED_SHA}\`"
    echo "- Repair turns: prepare=${PREPARE_REPAIR_TURNS}, build=${BUILD_REPAIR_TURNS}, certify=${CERTIFY_REPAIR_TURNS}"
    echo "- Agent repair wall time: ${AGENT_REPAIR_SECONDS_USED}s / ${REPAIR_TOTAL_BUDGET_SECONDS}s"
    echo
    if [[ "$outcome" == "certified" ]]; then
      echo "The wrapper applied the complete patch queue, completed the patched llama.cpp and Rust build gates, and passed the full supported-family certification on this exact commit."
    else
      echo "This draft preserves a bounded terminal failure in the **${FAILED_PHASE}** phase. It is not certified and is not eligible to merge until the failing gate is repaired and the complete state machine passes."
    fi
    echo
    echo "State machine: \`prepare -> build -> certify -> publish\`. Agent turns may edit local files, while the wrapper owns every gate and all GitHub mutations."
    echo
    cat "$UPSTREAM_SUMMARY"
  } > "$PR_BODY"
}

ensure_pr() {
  local outcome="$1" pr title created
  local -a create_args
  pr="$(current_pr)"
  if [[ -n "$pr" ]]; then
    gh_repair gh pr edit "$pr" --body-file "$PR_BODY" >/dev/null
    printf '%s\n' "$pr"
    return 0
  fi
  if [[ "$outcome" == "certified" ]]; then
    title="fix(llama): certify upstream ${UPSTREAM_SHA:0:10}"
    create_args=()
  else
    title="draft(llama): failed canary at ${UPSTREAM_SHA:0:10}"
    create_args=(--draft)
  fi
  if ! created="$(gh_repair gh pr create --base main --head "$BRANCH" "${create_args[@]}" \
      --title "$title" --body-file "$PR_BODY" 2> >(redact_token >&2))"; then
    echo "ERROR: could not create the terminal canary PR for ${BRANCH}" >&2
    return 1
  fi
  if ! pr="$(printf '%s\n' "$created" | grep -oE '[0-9]+$')"; then
    echo "ERROR: terminal canary PR creation returned no PR number for ${BRANCH}" >&2
    return 1
  fi
  printf '%s\n' "$pr"
}

verify_pr_head() {
  local expected="$1" pr remote_head attempt
  pr="$(current_pr)"
  [[ -n "$pr" ]] || { echo "terminal canary PR was not created" >&2; return 1; }
  for attempt in 1 2 3; do
    remote_head="$(gh_repair gh pr view "$pr" --json headRefOid --jq .headRefOid 2>/dev/null || true)"
    [[ "$remote_head" == "$expected" ]] && return 0
    sleep "$attempt"
  done
  echo "canary PR #${pr} head (${remote_head:-none}) does not match published commit ${expected}" >&2
  return 1
}

report_terminal() {
  local outcome="$1" pr comment
  publish_terminal_branch "$outcome"
  write_pr_body "$outcome"
  pr="$(ensure_pr "$outcome")"
  verify_pr_head "$PUBLISHED_SHA"
  if [[ "$outcome" == "certified" ]]; then
    comment="**Certified terminal state.** The exact PR head \`${CERTIFIED_SHA}\` passed prepare, the complete build, and the full supported-family certification."
  else
    comment="**Uncertified terminal state.** The internal deadline or repair-turn limit stopped the \`${FAILED_PHASE}\` phase. This draft preserves the final attempted bytes and must not merge until the complete state machine passes."
  fi
  gh_repair gh pr comment "$pr" --body "$comment" >/dev/null 2>&1 || true
  echo "terminal canary PR #${pr}: ${outcome}; branch=${BRANCH}; head=${PUBLISHED_SHA}"
}

phase="prepare"
while true; do
  phase_status=0
  # A function called on the left side of `||` inherits disabled errexit.
  # Every fallible phase command therefore has an explicit `|| return 1`;
  # the final command's status is the function status. This is load-bearing.
  case "$phase" in
    prepare)
      run_prepare || phase_status=$?
      [[ "$phase_status" -ne 0 ]] || { phase="build"; continue; }
      ;;
    build)
      run_full_build || phase_status=$?
      [[ "$phase_status" -ne 0 ]] || { phase="certify"; continue; }
      ;;
    certify)
      run_certification || phase_status=$?
      if [[ "$phase_status" -eq 0 ]]; then
        report_terminal certified
        exit 0
      fi
      ;;
  esac

  FAILED_PHASE="$phase"
  if ! remaining_work_seconds >/dev/null; then
    echo "internal canary deadline reached after ${phase} failure; reserving time for terminal publication" >&2
    report_terminal failed
    exit 1
  fi
  if (( $(phase_turns "$phase") >= MAX_REPAIR_TURNS )); then
    echo "${phase} exhausted ${MAX_REPAIR_TURNS} repair turns" >&2
    report_terminal failed
    exit 1
  fi
  if ! remaining_repair_seconds >/dev/null; then
    echo "agent repair exhausted its ${REPAIR_TOTAL_BUDGET_SECONDS}s aggregate budget" >&2
    report_terminal failed
    exit 1
  fi

  prompt="$(repair_prompt "$phase")"
  increment_phase_turns "$phase"
  agent_turn "$prompt" || echo "warning: agent repair turn exited non-zero; wrapper will retry the gates" >&2
  # Any agent edit may affect the selected pin or patch queue. Restore the
  # deterministic pin and restart at the first invalidated gate.
  phase="prepare"
done
