#!/usr/bin/env bash
set -euo pipefail

# Developer-style llama.cpp canary harness for changed upstream pins.
#
# One agent session owns the complete pin/patch/test task. The trusted wrapper
# runs deterministic gates after every coding turn and returns failures to that
# same session. A separate job repeats the gates before creating the certified
# bundle. GitHub credentials and publication live in a later workflow step, so
# an agent or verification failure cannot publish a branch or pull request.

# The persistent Apple Silicon runner service can be launched by an x86_64
# parent under Rosetta. Re-enter the complete harness as arm64 before it
# configures or executes native build artifacts.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]] \
    && [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
  exec arch -arm64 "${BASH_SOURCE[0]}" "$@"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRUSTED_ROOT="$ROOT"

# shellcheck disable=SC1091
source "$ROOT/scripts/lib/macos-deployment-target.sh"
HARNESS_MODE="${CANARY_HARNESS_MODE:-repair}"
if [[ -n "${CANARY_MESH_SOURCE:-}" ]]; then
  if [[ "$HARNESS_MODE" != pinned-build ]]; then
    echo "selected MeshLLM source requires certify-only pinned-build mode" >&2
    exit 1
  fi
  ROOT="${CANARY_SOURCE_ROOT:?selected source checkout required}"
  if [[ "$(git -C "$ROOT" rev-parse HEAD)" != "$CANARY_MESH_SOURCE" ]]; then
    echo "selected MeshLLM checkout does not match frozen revision" >&2
    exit 1
  fi
fi
UPSTREAM_SHA="${1:-${UPSTREAM_SHA_INPUT:-latest}}"
if [[ "$UPSTREAM_SHA" == "latest" || -z "$UPSTREAM_SHA" ]]; then
  UPSTREAM_SHA="$(git ls-remote https://github.com/ggml-org/llama.cpp.git master | awk '{print $1}')"
fi
if [[ ! "$UPSTREAM_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "refusing to run the canary against a non-40-hex upstream SHA: $UPSTREAM_SHA" >&2
  exit 1
fi

cd "$ROOT"

OLD_SHA="$(tr -d '[:space:]' < skippy/llama_cpp/upstream.txt)"
PIN_FILE="$ROOT/skippy/llama_cpp/upstream.txt"
AGENT_PROVIDER="${CANARY_AGENT_PROVIDER:-zai_coding_plan}"
AGENT_MODEL="${CANARY_AGENT_MODEL:-glm-5.3-flash}"
AGENT_TIMEOUT_SECONDS="${CANARY_AGENT_TIMEOUT_SECONDS:-41400}"
VERIFICATION_TIMEOUT_SECONDS="${CANARY_VERIFICATION_TIMEOUT_SECONDS:-43200}"
RUN_ID="${GITHUB_RUN_ID:-manual-$(date +%s)}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
RUN_KEY="${RUN_ID}-${RUN_ATTEMPT}"
PASS_ID="${CANARY_PASS_ID:-local}"
BRANCH="llama-canary/repair-${RUN_KEY}-${UPSTREAM_SHA:0:10}"
VERIFY_ROOT="/tmp/mesh-llm-canary-verify-${RUN_KEY}"
STATE_DIR="$ROOT/.deps/llama-canary-state-${RUN_KEY}-${PASS_ID}"
TARGET_SHA_FILE="$ROOT/.deps/llama-canary-target-sha"
AGENT_LOG="$STATE_DIR/agent.log"
AGENT_COMMAND_LOG_DIR="$STATE_DIR/agent-commands"
PREPARE_LOG="$STATE_DIR/prepare.log"
BUILD_LOG="$STATE_DIR/build.log"
CERTIFY_LOG="$STATE_DIR/certify.log"
MANIFEST_POLICY_LOG="$STATE_DIR/manifest-policy.log"
PR_BODY="$STATE_DIR/pr-body.md"
UPSTREAM_SUMMARY="$STATE_DIR/upstream-summary.md"
BUNDLE="$STATE_DIR/candidate.bundle"
EVIDENCE_DIR="$STATE_DIR/verification-evidence"
SYSTEMONE_SMOKE_DIR="$ROOT/target/skippy-system-one-smoke"
FAMILY_BATTERY_RUN_ID="${FAMILY_BATTERY_RUN_ID:-${RUN_KEY}}"
PLAN_PATH="$ROOT/target/family-battery/$FAMILY_BATTERY_RUN_ID/policy-plan.json"
BASE_HEAD="$(git rev-parse HEAD)"
BASE_REF="$(git symbolic-ref -q HEAD || true)"
CANDIDATE_BASE_HEAD="$BASE_HEAD"
GIT_CONFIG_FINGERPRINT="$(git config --list --show-origin | shasum -a 256 | awk '{print $1}')"
CERTIFIED_SHA=""
VERIFICATION_TREE=""
VERIFICATION_DEADLINE_AT=0
REPAIR_DEADLINE_AT=0
AGENT_SESSION_NAME="llama-canary-repair-${RUN_KEY}-${PASS_ID}"
AGENT_SESSION_STARTED=false

if [[ ! "$HARNESS_MODE" =~ ^(repair|verify|repair-build|verify-build|pinned-build)$ ]]; then
  echo "CANARY_HARNESS_MODE must be repair, verify, repair-build, verify-build, or pinned-build" >&2
  exit 1
fi

for timeout_name in AGENT_TIMEOUT_SECONDS VERIFICATION_TIMEOUT_SECONDS; do
  if [[ ! "${!timeout_name}" =~ ^[0-9]+$ ]] || (( ${!timeout_name} <= 0 )); then
    echo "$timeout_name must be a positive integer" >&2
    exit 1
  fi
done
for required_name in LLAMA_STAGE_BUILD_DIR HF_CACHE; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "${required_name} is not set; cannot run the changed-pin canary" >&2
    exit 1
  fi
done
if [[ -n "$(git status --porcelain)" ]]; then
  echo "changed-pin canary requires a clean trusted-main checkout" >&2
  exit 1
fi
# Legacy workload automation selection begins.
# Normal verify freezes this trusted controller before importing candidate source.
if [[ "$HARNESS_MODE" == repair || "$HARNESS_MODE" == verify ]]; then
  if [[ "${MESH_LLM_AUTOMATION_BIN+set}" == set ]]; then
    repair_workload_controller="$MESH_LLM_AUTOMATION_BIN"
  else
    repair_workload_bootstrap="$(just --justfile "$TRUSTED_ROOT/Justfile" automation-bootstrap)" || exit 1
    repair_workload_controller="$(printf '%s\n' "$repair_workload_bootstrap" | awk -F= '
      $1 == "binary_path" { count++; value=substr($0, index($0, "=") + 1) }
      END { if (count != 1 || value == "") exit 1; print value }
    ')" || { echo 'automation bootstrap must return one nonempty binary_path' >&2; exit 1; }
  fi
  if [[ "$repair_workload_controller" != /* || ! -f "$repair_workload_controller" || ! -x "$repair_workload_controller" ]]; then
    echo 'MESH_LLM_AUTOMATION_BIN or bootstrap binary_path must be an absolute executable' >&2
    exit 1
  fi
  repair_workload_controller_sha="$(shasum -a 256 "$repair_workload_controller" | awk '{print $1}')" || exit 1
  repair_workload_automation=("$repair_workload_controller")
  repair_workload_controller_unchanged() {
    local current
    current="$(shasum -a 256 "$repair_workload_controller" | awk '{print $1}')" || return 1
    if [[ "$current" != "$repair_workload_controller_sha" ]]; then
      echo 'frozen workload automation controller changed after admission' >&2
      return 1
    fi
  }
fi
# Legacy workload automation selection ends.
if [[ "$HARNESS_MODE" != pinned-build ]] && [[ -z "$(git config user.name)" || -z "$(git config user.email)" ]]; then
  echo "git user.name and user.email must be configured before canary repair" >&2
  exit 1
fi
if [[ "$HARNESS_MODE" == repair* ]] && ! command -v goose >/dev/null 2>&1; then
  echo "Goose CLI not found on runner; install it at /Users/lab/.local/bin/goose on the family-certify image" >&2
  exit 1
fi
if [[ "$HARNESS_MODE" == repair* ]]; then
  goose_check_status=0
  goose_check="$(
    GOOSE_PROVIDER="$AGENT_PROVIDER" GOOSE_MODEL="$AGENT_MODEL" \
      goose info --check 2>&1
  )" || goose_check_status=$?
  if (( goose_check_status != 0 )); then
    printf '%s\n' "$goose_check" >&2
    echo "Goose provider check failed for ${AGENT_PROVIDER}/${AGENT_MODEL}; repair the family-certify runner configuration" >&2
    exit 1
  fi
  printf '%s\n' "$goose_check"
fi

mkdir -p "$STATE_DIR" "$(dirname "$PLAN_PATH")"
mkdir -p "$AGENT_COMMAND_LOG_DIR"
rm -f "$AGENT_LOG" "$PREPARE_LOG" "$BUILD_LOG" "$CERTIFY_LOG" \
  "$MANIFEST_POLICY_LOG" \
  "$PR_BODY" "$UPSTREAM_SUMMARY" "$BUNDLE"
rm -rf "$EVIDENCE_DIR"
printf '%s\n' "$UPSTREAM_SHA" > "$TARGET_SHA_FILE"

# Persistent runners retain nested llama.cpp worktree registrations and /tmp
# checkouts. Remove only the known canary scratch state before the agent starts.
git -C "$ROOT/.deps/llama.cpp" worktree prune >/dev/null 2>&1 || true
rm -rf /tmp/llama-old-pin /tmp/llama-repair /tmp/llama-repair-* 2>/dev/null || true

run_for() {
  local label="$1" seconds="$2"
  shift 2
  local transaction_root input executable result timeout_parent
  local automation=()
  if [[ "$HARNESS_MODE" == repair || "$HARNESS_MODE" == verify ]]; then
    repair_workload_controller_unchanged || return 125
    automation=("${repair_workload_automation[@]}")
    timeout_parent="${RUNNER_TEMP:-/tmp}"
  else
    automation=("${MESH_LLM_AUTOMATION_BIN:?}")
    timeout_parent="${RUNNER_TEMP:?}"
  fi
  executable="$(command -v "$1")" || return 125
  if [[ ( "$HARNESS_MODE" == repair || "$HARNESS_MODE" == verify ) && "$executable" != /* && "$executable" == */* ]]; then
    executable="$PWD/$executable"
  fi
  if [[ "$executable" != /* ]]; then
    echo "$label requires an absolute executable" >&2
    return 125
  fi
  transaction_root="$(mktemp -d "$timeout_parent/canary-timeout.XXXXXXXX")" || return 125
  input="$transaction_root/input.json"
  shift
  jq -n --arg label "$label" --argjson seconds "$seconds" --arg cwd "$PWD" \
    --arg executable "$executable" --args \
    '{label:$label,seconds:$seconds,cwd:$cwd,executable:$executable,arguments:$ARGS.positional}' \
    -- "$@" > "$input" || { rm -rf "$transaction_root"; return 125; }
  if "${automation[@]}" automation canary-timeout --input "$input"; then
    result=0
  else
    result=$?
  fi
  rm -rf "$transaction_root" || return 125
  if [[ "$HARNESS_MODE" == repair || "$HARNESS_MODE" == verify ]]; then
    repair_workload_controller_unchanged || return 125
  fi
  return "$result"
}

record_failure_class() {
  local failure_class="$1" failure_stage="$2"
  echo "canary failure: class=$failure_class stage=$failure_stage" >&2
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "failure_class=$failure_class"
      echo "failure_stage=$failure_stage"
    } >> "$GITHUB_OUTPUT"
  fi
}

verification_source_inspection() {
  local verb="$1" log="${2:-}" transaction_root input status
  [[ "$HARNESS_MODE" == verify ]] || return 1
  case "$verb" in
    verification-source-admit|verification-manifest-policy|verification-parity-inventory|verification-split-roster-check) ;;
    *) echo "unsupported independent verification inspection" >&2; return 1 ;;
  esac
  repair_workload_controller_unchanged || return 1
  transaction_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/independent-verification.XXXXXXXX")" || return 1
  input="$transaction_root/input.json"
  if ! jq -n --arg controller_root "$TRUSTED_ROOT" --arg controller_revision "$BASE_HEAD" \
    --arg controller_sha "$repair_workload_controller_sha" --arg root "$ROOT" \
    --arg base "$CANDIDATE_BASE_HEAD" --arg candidate "$CERTIFIED_SHA" --arg tree "$VERIFICATION_TREE" \
    '{authority:{controller:{root:$controller_root,revision:$controller_revision,executable_sha256:$controller_sha},
      root:$root,base:$base,candidate:$candidate,tree:$tree}}' > "$input"; then
    rm -rf "$transaction_root"
    return 1
  fi
  if [[ -n "$log" ]]; then
    if run_verification_logged "parity manifest validation" "$log" \
      "${repair_workload_automation[@]}" automation canary-receipts "$verb" --input "$input"; then
      status=0
    else
      status=$?
    fi
  elif "${repair_workload_automation[@]}" automation canary-receipts "$verb" --input "$input"; then
    status=0
  else
    status=$?
  fi
  rm -rf "$transaction_root" || return 1
  repair_workload_controller_unchanged || return 1
  return "$status"
}

verification_candidate_unchanged() {
  if [[ "$HARNESS_MODE" == verify && -n "$CERTIFIED_SHA" ]]; then
    verification_source_inspection verification-source-admit
  fi
}

repair_family_plan_step() {
  local log="$1"
  shift
  local status
  repair_workload_controller_unchanged || return 1
  verification_candidate_unchanged || return 1
  if [[ -n "$log" ]]; then
    if run_verification_logged "full family certification plan" "$log" "$@"; then status=0; else status=$?; fi
  else
    if "$@"; then status=0; else status=$?; fi
  fi
  verification_candidate_unchanged || return 1
  repair_workload_controller_unchanged || return 1
  return "$status"
}

repair_family_plan() {
  local shards="$1" log="${2:-}" manifest="$ROOT/ci/llama-canary/family-certified.json"
  mkdir -p "$(dirname "$PLAN_PATH")" || return 1
  repair_family_plan_step "$log" "${repair_workload_automation[@]}" --repo-root "$ROOT" ci family-plan \
    --manifest "$manifest" --shard-count "$shards" --output "$PLAN_PATH" || return 1
  repair_family_plan_step "$log" "${repair_workload_automation[@]}" --repo-root "$ROOT" ci family-plan \
    --manifest "$manifest" --verify-plan "$PLAN_PATH" || return 1
  repair_family_plan_step "$log" "${repair_workload_automation[@]}" automation family-battery-policy --cache \
    "$ROOT" "$manifest" "$PLAN_PATH" "${HF_CACHE:?}" || return 1
}

check_family_cache() {
  if [[ "$HARNESS_MODE" == *-build ]]; then
    local transaction_root input source_revision
    transaction_root="$(mktemp -d "${RUNNER_TEMP:?}/canary-cache-plan.XXXXXXXX")" || return 1
    input="$transaction_root/input.json"
    source_revision="$(git rev-parse HEAD)" || return 1
    jq -n --arg controller_root "$TRUSTED_ROOT" --arg source_root "$ROOT" \
      --arg controller_revision "${CANARY_CONTROLLER_SHA:?}" --arg selected_revision "$source_revision" \
      --arg output "$transaction_root/admitted" --arg cache_root "${HF_CACHE:?}" \
      '{controller_root:$controller_root,source_root:$source_root,controller_revision:$controller_revision,selected_revision:$selected_revision,
        manifest:"ci/llama-canary/family-certified.json",output:$output,cache:{mode:"gguf_metadata",root:$cache_root}}' > "$input" || return 1
    "${MESH_LLM_AUTOMATION_BIN:?}" automation canary-receipts preflight --input "$input" || return 1
    mkdir -p "$(dirname "$PLAN_PATH")" || return 1
    cp "$transaction_root/admitted/plan.json" "$PLAN_PATH"
    return
  fi
  if [[ "$HARNESS_MODE" == repair || "$HARNESS_MODE" == verify ]]; then
    repair_family_plan 256
    return
  fi
  echo "unsupported canary cache mode: $HARNESS_MODE" >&2
  return 2
}

remaining_verification_seconds() {
  local remaining
  remaining="$((VERIFICATION_DEADLINE_AT - $(date +%s)))"
  (( remaining > 0 )) || return 1
  printf '%s\n' "$remaining"
}

remaining_repair_seconds() {
  local remaining
  remaining="$((REPAIR_DEADLINE_AT - $(date +%s)))"
  (( remaining > 0 )) || return 1
  printf '%s\n' "$remaining"
}

run_verification_logged() {
  local label="$1" log="$2" seconds
  shift 2
  if ! seconds="$(remaining_verification_seconds)"; then
    echo "$label cannot start: final verification budget exhausted" | tee -a "$log" >&2
    return 124
  fi
  run_for "$label" "$seconds" "$@" > >(tee -a "$log") 2>&1
}

write_repair_pin() {
  if [[ "$HARNESS_MODE" == pinned-build ]]; then
    verify_repair_pin
    return
  fi
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

agent_prompt() {
  printf 'Complete the llama.cpp upstream update to %s as one developer task in this checkout.

The trusted harness has already written skippy/llama_cpp/upstream.txt to the exact target and recorded it in .deps/llama-canary-target-sha. Read ci/llama-canary/agent-repair-prompt.md and every repository skill it names, then own the work end to end: reproduce the queue failure, deliberately rebase or regenerate the owned patches, fix any generated-family rewriter or Rust ABI fallout, and run the prepare, build, smoke, and focused reproductions needed to validate your repairs. Once those checks pass, return control to the trusted harness for the full supported-family battery. Do not start an additional full battery in the coding session; the wrapper and separate verifier each run all required gates.

Do not weaken, skip, or narrow a gate. Do not edit the workflow, this wrapper, its publisher, the agent runbook, or their contract tests. Do not create or switch branches, commit, push, open a pull request, or use GitHub credentials. Leave the completed changes in this working tree. The harness will independently rerun the entire verification sequence and only a green exact tree can be published.' \
    "$UPSTREAM_SHA"
  if [[ -n "${CANARY_PREVIOUS_FEEDBACK:-}" ]]; then
    printf '\n\nThis is distributed repair attempt %s. The exact prior candidate has already been restored as uncommitted changes on the frozen base. Read the digest-bound family failure summary and every failed-family directory under %s before editing. Preserve the prior repairs, fix the candidate failures demonstrated there, and use focused reproductions before returning control for a new complete family pass.\n\n' \
      "$PASS_ID" "$CANARY_PREVIOUS_FEEDBACK"
    python3 scripts/summarize-canary-feedback.py "$CANARY_PREVIOUS_FEEDBACK"
  fi
}

restore_previous_repair_candidate() {
  local bundle expected branch bundle_head protected
  if [[ -z "${CANARY_INPUT_BUNDLE:-}" ]]; then
    return 0
  fi
  if [[ "$HARNESS_MODE" != "repair-build" || -z "${CANARY_PREVIOUS_FEEDBACK:-}" ]]; then
    echo "previous repair candidate is only valid with distributed family feedback" >&2
    return 1
  fi
  bundle="$CANARY_INPUT_BUNDLE"
  expected="${CANARY_CANDIDATE_SHA:?previous candidate SHA required}"
  branch="${CANARY_CANDIDATE_BRANCH:?previous candidate branch required}"
  git bundle verify "$bundle" >/dev/null
  bundle_head="$(git bundle list-heads "$bundle" "refs/heads/${branch}" | awk '{print $1}')"
  if [[ "$bundle_head" != "$expected" ]]; then
    echo "previous candidate bundle head does not match dependency output" >&2
    return 1
  fi
  git fetch "$bundle" "refs/heads/${branch}" >/dev/null
  if [[ "$(git rev-parse FETCH_HEAD)" != "$expected" || "$(git rev-parse "${expected}^")" != "$BASE_HEAD" ]]; then
    echo "previous candidate is not a direct child of the frozen base" >&2
    return 1
  fi
  protected="$(
    git diff --name-only "$BASE_HEAD" "$expected" -- \
      .github .agents scripts .gitattributes ci/ci.md ci/llama-canary/agent-repair-prompt.md \
      | head -n 1
  )"
  if [[ -n "$protected" ]]; then
    echo "previous candidate modified protected orchestration: $protected" >&2
    return 1
  fi
  git diff --binary "$BASE_HEAD" "$expected" -- | git apply --index --binary
  if [[ "$(git write-tree)" != "$(git rev-parse "${expected}^{tree}")" ]]; then
    echo "restored previous candidate tree does not match its bundle" >&2
    return 1
  fi
}

agent_session_step() {
  local prompt="$1" started heartbeat_pid status seconds
  local -a goose_args
  if ! seconds="$(remaining_repair_seconds)"; then
    echo "agent developer task cannot continue: repair budget exhausted" >&2
    return 124
  fi
  started="$(date +%s)"
  # shellcheck disable=SC2016
  env -i PATH="$PATH" bash -c '
    root="$1"
    started="$2"
    sleeper=""
    stop_heartbeat() {
      if [[ -n "$sleeper" ]]; then
        kill "$sleeper" 2>/dev/null || true
        wait "$sleeper" 2>/dev/null || true
      fi
    }
    trap stop_heartbeat EXIT
    trap "exit 0" TERM INT
    while true; do
      sleep 600 &
      sleeper=$!
      wait "$sleeper" || break
      sleeper=""
      newest="$(find "$root/.deps/llama.cpp" -type f -newer "$root/skippy/llama_cpp/upstream.txt" -print -quit 2>/dev/null || true)"
      printf "heartbeat: agent task running for %dm; recent llama.cpp activity: %s\n" \
        "$(( ($(date +%s) - started) / 60 ))" "${newest:-none observed yet}"
    done
  ' heartbeat "$ROOT" "$started" &
  heartbeat_pid=$!
  set +e
  goose_args=(
    run
    --provider "$AGENT_PROVIDER"
    --model "$AGENT_MODEL"
    --with-builtin developer
    --no-profile
    --max-turns 1000
    --output-format text
    --name "$AGENT_SESSION_NAME"
  )
  if [[ "$AGENT_SESSION_STARTED" == "true" ]]; then
    goose_args+=(--resume)
  fi
  goose_args+=(--text "$prompt")
  run_for "agent developer task" "$seconds" env \
    -u GH_TOKEN -u GITHUB_TOKEN -u CANARY_REPAIR_TOKEN \
    CANARY_REPAIR_LOG_DIR="$AGENT_COMMAND_LOG_DIR" \
    GOOSE_MODE=auto GOOSE_DISABLE_SESSION_NAMING=true \
    goose "${goose_args[@]}" \
    > >(tee -a "$AGENT_LOG") 2>&1
  status=$?
  set -e
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  if (( status != 0 )); then
    printf 'agent developer task exited with status %s\n' "$status" \
      | tee -a "$AGENT_LOG" >&2
  fi
  if (( status == 0 )); then
    AGENT_SESSION_STARTED=true
  fi
  return "$status"
}

assert_agent_control_unchanged() {
  local changed_path
  if [[ "$(git rev-parse HEAD)" != "$BASE_HEAD" ]]; then
    echo "agent created commits; the harness requires uncommitted candidate changes" >&2
    return 1
  fi
  if [[ "$(git symbolic-ref -q HEAD || true)" != "$BASE_REF" ]]; then
    echo "agent switched branches; refusing to verify the candidate" >&2
    return 1
  fi
  if [[ "$(git config --list --show-origin | shasum -a 256 | awk '{print $1}')" != "$GIT_CONFIG_FINGERPRINT" ]]; then
    echo "agent changed Git configuration; refusing to materialize the candidate" >&2
    return 1
  fi
  changed_path="$(
    git status --porcelain=v1 --untracked-files=all -- \
      .github .agents scripts .gitattributes ci/ci.md ci/llama-canary/agent-repair-prompt.md \
      | head -n 1
  )"
  if [[ -n "$changed_path" ]]; then
    echo "agent modified protected CI or verification file: $changed_path" >&2
    return 1
  fi
}

repair_source_inspection() {
  local verb="$1" check="${2:-}" log="${3:-}" transaction_root input status
  repair_workload_controller_unchanged || return 1
  transaction_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/local-repair-inspection.XXXXXXXX")" || return 1
  input="$transaction_root/input.json"
  if ! jq -n --arg controller_root "$TRUSTED_ROOT" --arg controller_revision "$BASE_HEAD" \
    --arg controller_sha "$repair_workload_controller_sha" --arg root "$ROOT" \
    --arg base "$CANDIDATE_BASE_HEAD" --arg check "$check" \
    '{authority:{controller:{root:$controller_root,revision:$controller_revision,executable_sha256:$controller_sha},root:$root,base:$base}}
      + (if $check == "true" then {check:true} elif $check == "false" then {check:false} else {} end)' \
    > "$input"; then
    rm -rf "$transaction_root"
    return 1
  fi
  if [[ -n "$log" ]]; then
    if run_verification_logged "parity manifest validation" "$log" \
      "${repair_workload_automation[@]}" automation canary-receipts "$verb" --input "$input"; then
      status=0
    else
      status=$?
    fi
  elif "${repair_workload_automation[@]}" automation canary-receipts "$verb" --input "$input"; then
    status=0
  else
    status=$?
  fi
  rm -rf "$transaction_root" || return 1
  return "$status"
}

validate_agent_manifest_changes() {
  if [[ "$HARNESS_MODE" == *-build ]]; then
    local transaction_root input context
    transaction_root="$(mktemp -d "${RUNNER_TEMP:?}/canary-manifest-policy.XXXXXXXX")" || return 1
    input="$transaction_root/input.json"
    context="$(controller_package_context)" || return 1
    jq -n --argjson context "$context" --arg root "$ROOT" --arg base "$CANDIDATE_BASE_HEAD" \
      '{context:$context,root:$root,base:$base}' > "$input" || return 1
    : > "$MANIFEST_POLICY_LOG"
    "${MESH_LLM_AUTOMATION_BIN:?}" automation canary-receipts manifest-policy --input "$input" \
      > >(tee -a "$MANIFEST_POLICY_LOG") 2>&1
    return
  fi
  : > "$MANIFEST_POLICY_LOG"
  if [[ "$HARNESS_MODE" == repair ]]; then
    repair_source_inspection local-manifest-policy > >(tee -a "$MANIFEST_POLICY_LOG") 2>&1
    return
  fi
  if [[ "$HARNESS_MODE" == verify ]]; then
    verification_source_inspection verification-manifest-policy > >(tee -a "$MANIFEST_POLICY_LOG") 2>&1
    return
  fi
  echo "unsupported canary manifest mode: $HARNESS_MODE" >&2
  return 2
}

agent_feedback_prompt() {
  printf 'The trusted harness tested the current working tree and it is still red. Continue the same developer task in this session. Read the current failure logs at:\n\n- %s\n- %s\n- %s\n- %s\n\nFix the actual source or narrowly permitted manifest data, then rerun the affected checks. Once the known failures are fixed and their reproductions pass, return control for the trusted full gates; do not repeat the full family battery inside the coding session. Report the exact checks run and any unresolved failures, without claiming certification. The same control-file, Git, credential, and publication restrictions still apply.' \
    "$PREPARE_LOG" "$MANIFEST_POLICY_LOG" "$BUILD_LOG" "$CERTIFY_LOG"
}

snapshot_candidate_tree() {
  assert_agent_control_unchanged || return 1
  verify_repair_pin || return 1
  validate_agent_manifest_changes || return 1
  # Freeze the actual dirty-tree producer before staging changes Git identity.
  if [[ "$HARNESS_MODE" == "repair-build" ]]; then
    controller_producer_receipt || return 1
  else
    local closure="${LLAMA_STAGE_BUILD_DIR:?}-workloads"
    repair_workload_controller_unchanged || return 1
    "${repair_workload_automation[@]}" automation canary-receipts workload-manifest verify \
      "$ROOT" "$closure/cargo/debug/skippy" "$closure/native" "$closure/producer.json" || return 1
    CANARY_VERIFIED_WORKLOAD_PRODUCER="$(shasum -a 256 "$closure/producer.json" | awk '{print $1}')" || return 1
    export CANARY_VERIFIED_WORKLOAD_PRODUCER
  fi
  git add -A
  if git diff --cached --quiet; then
    echo "agent produced no candidate changes to verify" >&2
    return 1
  fi
  VERIFICATION_TREE="$(git write-tree)"
  if [[ "$HARNESS_MODE" == "repair-build" && -n "${CANARY_INPUT_BUNDLE:-}" ]] &&
      [[ "$VERIFICATION_TREE" == "$(git rev-parse "${CANARY_CANDIDATE_SHA}^{tree}")" ]]; then
    echo "agent made no changes to the restored candidate" >&2
    return 1
  fi
  CERTIFIED_SHA="$(
    printf '%s\n\n%s\n' \
      "fix(llama): certify upstream ${UPSTREAM_SHA:0:10}" \
      "Prepared upstream candidate; certification evidence is recorded separately before publication." \
      | git commit-tree "$VERIFICATION_TREE" -p "$BASE_HEAD"
  )"
}

write_candidate_bundle() {
  git -c core.hooksPath=/dev/null branch -f "$BRANCH" "$CERTIFIED_SHA"
  git bundle create "$BUNDLE" "$BRANCH" "^${BASE_HEAD}"
  git bundle verify "$BUNDLE" >/dev/null
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "branch=$BRANCH"
      echo "head=$CERTIFIED_SHA"
      echo "candidate_bundle=$BUNDLE"
    } >> "$GITHUB_OUTPUT"
  fi
  echo "agent candidate bundle: branch=$BRANCH head=$CERTIFIED_SHA"
}

load_candidate_bundle() {
  local input_bundle expected_head candidate_branch bundle_head
  input_bundle="${CANARY_INPUT_BUNDLE:?CANARY_INPUT_BUNDLE is required in verify mode}"
  expected_head="${CANARY_CANDIDATE_SHA:?CANARY_CANDIDATE_SHA is required in verify mode}"
  candidate_branch="${CANARY_CANDIDATE_BRANCH:?CANARY_CANDIDATE_BRANCH is required in verify mode}"
  if [[ ! "$expected_head" =~ ^[0-9a-f]{40}$ || ! -s "$input_bundle" ]]; then
    echo "verification requires a non-empty candidate bundle and 40-hex head" >&2
    return 1
  fi
  if [[ "$candidate_branch" != llama-canary/repair-* ]] ||
      ! git check-ref-format "refs/heads/${candidate_branch}"; then
    echo "verification requires a valid identity-bound candidate branch" >&2
    return 1
  fi
  git bundle verify "$input_bundle" >/dev/null
  bundle_head="$(git bundle list-heads "$input_bundle" "refs/heads/${candidate_branch}" | awk '{print $1}')"
  if [[ "$bundle_head" != "$expected_head" ]]; then
    echo "candidate bundle head does not match the repair job output" >&2
    return 1
  fi
  git fetch "$input_bundle" "refs/heads/${candidate_branch}"
  CERTIFIED_SHA="$expected_head"
  CANDIDATE_BASE_HEAD="$(git rev-parse "${CERTIFIED_SHA}^")"
  VERIFICATION_TREE="$(git rev-parse "${CERTIFIED_SHA}^{tree}")"
}

cleanup_verification_worktree() {
  local source
  if [[ -d "$VERIFY_ROOT" ]]; then
    mkdir -p "$EVIDENCE_DIR"
    for source in \
        "$VERIFY_ROOT/target/family-battery/$FAMILY_BATTERY_RUN_ID" \
        "$VERIFY_ROOT/target/skippy-stage-rewriter-check" \
        "$VERIFY_ROOT/target/skippy-system-one-smoke"; do
      if [[ -e "$source" ]]; then
        cp -R "$source" "$EVIDENCE_DIR/" || true
      fi
    done
  fi
  git -c core.hooksPath=/dev/null -C "$TRUSTED_ROOT" \
    worktree remove --force "$VERIFY_ROOT" >/dev/null 2>&1 || true
}

materialize_verification_tree() {
  cleanup_verification_worktree
  git -c core.hooksPath=/dev/null -C "$TRUSTED_ROOT" worktree prune
  git -c core.hooksPath=/dev/null -C "$TRUSTED_ROOT" \
    worktree add --detach "$VERIFY_ROOT" "$CERTIFIED_SHA"
  ROOT="$VERIFY_ROOT"
  PIN_FILE="$ROOT/skippy/llama_cpp/upstream.txt"
  FAMILY_BATTERY_RUN_ID="${RUN_KEY}-verification"
  PLAN_PATH="$ROOT/target/family-battery/$FAMILY_BATTERY_RUN_ID/policy-plan.json"
  LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR}-verification-${RUN_KEY}"
  LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR"
  export LLAMA_BUILD_DIR LLAMA_STAGE_BUILD_DIR FAMILY_BATTERY_RUN_ID
  rm -rf "$LLAMA_STAGE_BUILD_DIR" \
    "$ROOT/.deps/llama.cpp" \
    "$ROOT/target/family-battery/$FAMILY_BATTERY_RUN_ID" \
    "$ROOT/target/skippy-stage-rewriter-check" \
    "$ROOT/target/skippy-system-one-smoke"
  # Re-derive the System One smoke work dir under the verification ROOT: the
  # top-level assignment still points at the trusted checkout, and verifier
  # evidence must stay inside the candidate worktree that
  # cleanup_verification_worktree copies into EVIDENCE_DIR.
  SYSTEMONE_SMOKE_DIR="$ROOT/target/skippy-system-one-smoke"
  mkdir -p "$(dirname "$PLAN_PATH")"
  cd "$ROOT"
  verify_repair_pin
}

run_prepare() {
  local prepared_upstream
  : > "$PREPARE_LOG"
  echo "trusted candidate gate: prepare" | tee -a "$PREPARE_LOG"
  write_repair_pin >>"$PREPARE_LOG" 2>&1 || return 1
  verify_repair_pin >>"$PREPARE_LOG" 2>&1 || return 1
  run_verification_logged "apply llama.cpp patch queue" "$PREPARE_LOG" \
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
  echo "trusted candidate gate: build" | tee -a "$BUILD_LOG"
  run_verification_logged "complete patched llama.cpp build" "$BUILD_LOG" env \
    LLAMA_STAGE_UPSTREAM_TESTS=ON uv run --no-project --with jinja2==3.1.6 -- \
    arch -arm64 bash scripts/build-llama.sh -DCMAKE_OSX_ARCHITECTURES=arm64 -DGGML_METAL_EMBED_LIBRARY=ON \
    || return 1
  archive="$LLAMA_STAGE_BUILD_DIR/src/libllama.a"
  arches="$(lipo -archs "$archive" 2>/dev/null || true)"
  if [[ "$arches" != "arm64" ]]; then
    echo "candidate native archive must be arm64, got: ${arches:-missing}" | tee -a "$BUILD_LOG" >&2
    return 1
  fi
  run_verification_logged "generated model-family patch check" "$BUILD_LOG" \
    scripts/check-skippy-generated-family-patch.sh || return 1
  run_verification_logged "stage runtime crate build" "$BUILD_LOG" \
    cargo build -p skippy-runtime -p skippy-cli -p skippy-package-builder -p skippy-correctness -p skippy-topology --bins \
    || return 1
  run_verification_logged "Skippy smoke tests" "$BUILD_LOG" \
    scripts/skippy-ci-smoke.sh || return 1
  # Run-scoped CPU workload oracle closure for the non-chat certification
  # lanes; packed into the executable handoff for family workers.
  run_verification_logged "pinned CPU workload oracles and candidate" "$BUILD_LOG" \
    just skippy-workload-oracles-build "${LLAMA_STAGE_BUILD_DIR:?}-workloads" || return 1
  if [[ "$HARNESS_MODE" == *-build ]]; then
    # The nested shell expands its positional argument, not this shell.
    # shellcheck disable=SC2016
    run_verification_logged "build transferable multimodal test executable" "$BUILD_LOG" \
      bash -c 'cargo test -p skippy-serving --lib --no-run --message-format=json > "$1"' \
      build-mm "$STATE_DIR/mm-build.jsonl" || return 1
  fi
  # The family-certify runner is a Metal execution lane. Both real decision
  # models are mandatory here: Jev exercises the complete DiffusionGemma read
  # through the staged Metal server, and Laya exercises the static native CLI
  # on its explicit CPU device against the upstream golden fixtures. Platform
  # Laya smokes separately prove the packaged Metal runtime.
  run_verification_logged "System One smoke" "$BUILD_LOG" env \
    WORK_DIR="$SYSTEMONE_SMOKE_DIR" \
    SYSTEMONE_SMOKE_CADENCE=llama-bump \
    SYSTEMONE_SMOKE_BUILD_BACKEND=metal \
    SYSTEMONE_SMOKE_CERTIFIED_BACKENDS=metal \
    SYSTEMONE_SMOKE_REQUIRE_QUALIFIED=1 \
    scripts/skippy-system-one-smoke.sh || return 1
  run_verification_logged "Laya smoke" "$BUILD_LOG" env \
    WORK_DIR="$SYSTEMONE_SMOKE_DIR" \
    LAYA_SMOKE_CADENCE=llama-bump \
    LAYA_SMOKE_DEVICE=CPU \
    scripts/skippy-laya-smoke.sh || return 1
}

# Local CLI compatibility path. CI uses *-build modes and separate family jobs.
run_certification() {
  if [[ "$HARNESS_MODE" != repair && "$HARNESS_MODE" != verify ]]; then
    echo "local certification requires repair or verify mode" >&2
    return 2
  fi
  local setting workload_settings
  local workload_env=()
  workload_settings="$(bash scripts/skippy-workload-oracles-build.sh --print-env "${LLAMA_STAGE_BUILD_DIR:?}-workloads")" || return 1
  if [[ -z "$workload_settings" ]]; then
    echo "workload producer returned no certification environment" >&2
    return 1
  fi
  while IFS= read -r setting; do
    workload_env+=("$setting")
  done <<< "$workload_settings"
  : > "$CERTIFY_LOG"
  echo "trusted candidate gate: certify" | tee -a "$CERTIFY_LOG"
  if [[ "$HARNESS_MODE" == repair ]]; then
    repair_source_inspection local-parity-inventory "" "$CERTIFY_LOG" || return 1
  elif [[ "$HARNESS_MODE" == verify ]]; then
    verification_source_inspection verification-parity-inventory "$CERTIFY_LOG" || return 1
  fi
  repair_family_plan 1 "$CERTIFY_LOG" || return 1
  run_verification_logged "full supported-family certification" "$CERTIFY_LOG" env \
    FAMILY_BATTERY_RUN_ID="$FAMILY_BATTERY_RUN_ID" \
    "${workload_env[@]}" \
    scripts/skippy-family-battery.sh --skip-build --plan "$PLAN_PATH"
}

run_early_metal_certification() {
  local setting workload_settings mm_test_bin
  local workload_env=()
  # A cached executable is usable only when its recorded source tree (and
  # therefore pin), native stamp, and every handed-off binary still match.
  run_verification_logged "verify exact workload producer" "$CERTIFY_LOG" \
    python3 scripts/check-skippy-workload-candidate.py \
      --candidate-binary "${LLAMA_STAGE_BUILD_DIR:?}-workloads/cargo/debug/skippy" \
      --native-build-dir "${LLAMA_STAGE_BUILD_DIR:?}-workloads/native" \
      --producer-manifest "${LLAMA_STAGE_BUILD_DIR:?}-workloads/producer.json" || return 1
  workload_settings="$(bash scripts/skippy-workload-oracles-build.sh --print-env "${LLAMA_STAGE_BUILD_DIR:?}-workloads")" || return 1
  [[ -n "$workload_settings" ]] || return 1
  while IFS= read -r setting; do
    workload_env+=("$setting")
  done <<< "$workload_settings"
  mm_test_bin="$(jq -rs '[.[] | select(.reason == "compiler-artifact" and .profile.test == true and .target.name == "skippy_serving" and .executable != null) | .executable] | unique | if length == 1 then .[0] else error("expected one exact multimodal test executable") end' "$STATE_DIR/mm-build.jsonl")" || return 1
  [[ -x "$mm_test_bin" ]] || return 1
  # One exact candidate on Metal, with representatives for split parity,
  # recurrent and MoE replay, both encode-only startup classes, T5 ordering,
  # and an actual image response. The full roster remains the promotion gate.
  run_verification_logged "early real-model Metal certification" "$CERTIFY_LOG" env \
    FAMILY_BATTERY_RUN_ID="${FAMILY_BATTERY_RUN_ID}-early" \
    FAMILY_BATTERY_MM_TEST_BIN="$mm_test_bin" \
    "${workload_env[@]}" \
    scripts/skippy-family-battery.sh --skip-build \
      --families llama,mamba2,deepseek2,nomic-bert-embedding,jina-bert-v2-rerank,t5-encoder-decoder,qwen3-vl
}

run_candidate_gates() {
  local roster_mode="${1:-verify}"
  if [[ "$roster_mode" != "verify" && "$roster_mode" != "refresh" ]]; then
    echo "invalid candidate-gate roster mode: $roster_mode" >&2
    return 2
  fi
  : > "$PREPARE_LOG"
  : > "$MANIFEST_POLICY_LOG"
  : > "$BUILD_LOG"
  : > "$CERTIFY_LOG"
  run_prepare || return 1
  if [[ "$roster_mode" == "refresh" ]]; then
    # Preparation writes the candidate upstream pin. Generate only after that
    # transition so the roster recipe matches the runtime about to be built.
    # Independent verification uses the default read-only mode below.
    write_split_certification_roster || return 1
  fi
  # The prepared pin supplies the exact GGML type table. Compare tensor
  # descriptors with the manifest now, before the native and Rust builds.
  run_verification_logged "validate pinned GGUF tensor bytes before compilation" "$CERTIFY_LOG" \
    python3 scripts/plan-family-battery.py --shard-count 256 \
      --check-cache --cache-root "$HF_CACHE" \
      --gguf-constants "$ROOT/.deps/llama.cpp/gguf-py/gguf/constants.py" \
      --output "$PLAN_PATH" || return 1
  validate_agent_manifest_changes || return 1
  if [[ "$HARNESS_MODE" == *-build ]]; then
    run_verification_logged "validate family plan before compilation" "$CERTIFY_LOG" \
      "${MESH_LLM_AUTOMATION_BIN:?}" --repo-root "$ROOT" ci family-plan --shard-count 256 \
        --output "$PLAN_PATH" || return 1
  fi
  run_full_build || return 1
  if [[ "$HARNESS_MODE" == *-build ]]; then
    run_early_metal_certification || return 1
    controller_parity_inventory
  else
    run_certification
  fi
}

controller_parity_inventory() {
  local transaction_root input context
  transaction_root="$(mktemp -d "${RUNNER_TEMP:?}/canary-parity.XXXXXXXX")" || return 1
  input="$transaction_root/input.json"
  context="$(controller_package_context)" || return 1
  jq -n --argjson context "$context" --arg root "$ROOT" --arg base "$CANDIDATE_BASE_HEAD" \
    --arg source_revision "${CERTIFIED_SHA:-$BASE_HEAD}" \
    '{context:$context,root:$root,base:$base,source_revision:$source_revision}' > "$input" || return 1
  run_verification_logged "parity manifest validation" "$CERTIFY_LOG" \
    "${MESH_LLM_AUTOMATION_BIN:?}" automation canary-receipts parity-inventory --input "$input"
}

controller_split_roster() {
  local check="$1" transaction_root input context
  transaction_root="$(mktemp -d "${RUNNER_TEMP:?}/canary-roster.XXXXXXXX")" || return 1
  input="$transaction_root/input.json"
  context="$(controller_package_context)" || return 1
  jq -n --argjson context "$context" --arg root "$ROOT" --argjson check "$check" \
    '{context:$context,root:$root,check:$check}' > "$input" || return 1
  "${MESH_LLM_AUTOMATION_BIN:?}" automation canary-receipts split-roster --input "$input"
}

write_split_certification_roster() {
  if [[ "$HARNESS_MODE" == *-build ]]; then
    controller_split_roster false
  elif [[ "$HARNESS_MODE" == repair ]]; then
    repair_source_inspection local-split-roster false
  elif [[ "$HARNESS_MODE" == verify ]]; then
    echo "independent verification cannot write the split roster" >&2
    return 1
  else
    echo "unsupported canary roster mode: $HARNESS_MODE" >&2
    return 2
  fi
}

check_split_certification_roster() {
  if [[ "$HARNESS_MODE" == *-build ]]; then
    controller_split_roster true
  elif [[ "$HARNESS_MODE" == repair ]]; then
    repair_source_inspection local-split-roster true
  elif [[ "$HARNESS_MODE" == verify ]]; then
    verification_source_inspection verification-split-roster-check
  else
    echo "unsupported canary roster mode: $HARNESS_MODE" >&2
    return 2
  fi
}

repair_candidate_until_green() {
  local prompt status
  REPAIR_DEADLINE_AT="$(( $(date +%s) + AGENT_TIMEOUT_SECONDS ))"
  prompt="$(agent_prompt)"

  while remaining_repair_seconds >/dev/null; do
    if agent_session_step "$prompt"; then
      :
    else
      status=$?
      record_failure_class infrastructure agent-runtime
      return "$status"
    fi
    assert_agent_control_unchanged || return 1
    # Coding turns may start only within the repair window. Once a turn
    # returns, give its complete gate sequence a fresh bounded pass, even
    # when earlier gates have consumed most of the repair window.
    VERIFICATION_DEADLINE_AT="$(( $(date +%s) + VERIFICATION_TIMEOUT_SECONDS ))"
    if run_candidate_gates refresh; then
      assert_agent_control_unchanged || return 1
      validate_agent_manifest_changes || return 1
      return 0
    fi
    if ! remaining_repair_seconds >/dev/null; then
      echo "candidate remains red and the repair budget is exhausted" >&2
      record_failure_class candidate trusted-gates
      return 124
    fi
    echo "candidate gates remain red; returning their logs to the same agent session"
    prompt="$(agent_feedback_prompt)"
  done
  echo "candidate remains red and the repair budget is exhausted" >&2
  record_failure_class candidate trusted-gates
  return 124
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
  write_upstream_summary
  {
    echo "Automated llama.cpp upstream update for \`${UPSTREAM_SHA}\`."
    echo
    echo "- Previous pin: \`${OLD_SHA}\`"
    echo "- Candidate pin: \`${UPSTREAM_SHA}\`"
    echo "- Workflow run: \`${RUN_KEY}\`"
    echo "- Certified commit: \`${CERTIFIED_SHA}\`"
    echo
    echo "One agent completed the pin and patch-queue task. The trusted harness then independently passed prepare, the complete patched llama.cpp and Rust build, Skippy smoke tests, the System One (OpenJEV) smoke, and the full supported-family certification on this exact commit."
    echo
    cat "$UPSTREAM_SUMMARY"
  } > "$PR_BODY"
}

finalize_certified_tree() {
  verification_candidate_unchanged || return 1
  verify_repair_pin || return 1
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "verified checkout changed tracked files during final verification" >&2
    return 1
  fi
  if [[ "$(git rev-parse "${CERTIFIED_SHA}^{tree}")" != "$VERIFICATION_TREE" ]]; then
    echo "certified commit tree changed after final verification" >&2
    return 1
  fi
  write_pr_body
  git -c core.hooksPath=/dev/null -C "$TRUSTED_ROOT" \
    branch -f "$BRANCH" "$CERTIFIED_SHA"
  git -C "$TRUSTED_ROOT" bundle create "$BUNDLE" "$BRANCH" "^${CANDIDATE_BASE_HEAD}"
  git -C "$TRUSTED_ROOT" bundle verify "$BUNDLE" >/dev/null
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "branch=$BRANCH"
      echo "head=$CERTIFIED_SHA"
      echo "pr_body=$PR_BODY"
      echo "candidate_bundle=$BUNDLE"
    } >> "$GITHUB_OUTPUT"
  fi
  echo "certified local canary commit: branch=$BRANCH head=$CERTIFIED_SHA"
}

controller_package_context() {
  jq -n --arg controller_root "$TRUSTED_ROOT" --arg controller_revision "${CANARY_CONTROLLER_SHA:?}" \
    --arg selected_source "${CANARY_MESH_SOURCE:-}" --arg run_id "${GITHUB_RUN_ID:?}" \
    --arg run_attempt "${GITHUB_RUN_ATTEMPT:?}" \
    '{controller_root:$controller_root,controller_revision:$controller_revision,selected_source:$selected_source,run_id:$run_id,run_attempt:$run_attempt}'
}

controller_producer_receipt() {
  local transaction_root input result context
  transaction_root="$(mktemp -d "${RUNNER_TEMP:?}/canary-package.XXXXXXXX")" || return 1
  CANARY_PRODUCER_RECEIPT="$transaction_root/producer.json"
  input="$transaction_root/producer-input.json"
  context="$(controller_package_context)" || return 1
  jq -n --argjson context "$context" --arg root "$ROOT" \
    --arg closure "${LLAMA_STAGE_BUILD_DIR:?}-workloads" --arg output "$CANARY_PRODUCER_RECEIPT" \
    '{context:$context,root:$root,closure:$closure,output:$output}' > "$input" || return 1
  if result="$("${MESH_LLM_AUTOMATION_BIN:?protected automation required}" automation canary-receipts producer-receipt --input "$input")"; then
    CANARY_PRODUCER_RECEIPT_SHA="$(jq -er '.producer_receipt_sha256' <<< "$result")" || return 1
  else
    record_failure_class infrastructure producer-receipt
    return 1
  fi
}

export_family_inputs() {
  local destination="${CANARY_EXPORT_DIR:?CANARY_EXPORT_DIR required}"
  local context transaction_root input result admission admission_sha
  [[ -n "${CANARY_PRODUCER_RECEIPT:-}" ]] || controller_producer_receipt || return 1
  context="$(controller_package_context)" || return 1
  transaction_root="$(mktemp -d "${RUNNER_TEMP:?}/canary-admission.XXXXXXXX")" || return 1
  admission="$transaction_root/admitted"
  input="$transaction_root/plan-input.json"
  jq -n --argjson context "$context" --arg root "$ROOT" --arg base "$CANDIDATE_BASE_HEAD" \
    --arg candidate "$CERTIFIED_SHA" --arg cache_root "${HF_CACHE:?}" --arg output "$admission" \
    '{context:$context,root:$root,base:$base,candidate:$candidate,cache_root:$cache_root,output:$output}' > "$input" || return 1
  if result="$("${MESH_LLM_AUTOMATION_BIN:?}" automation canary-receipts candidate-plan --input "$input")"; then
    admission_sha="$(jq -er '.admitted_identity_sha256' <<< "$result")" || return 1
  else
    record_failure_class infrastructure candidate-plan
    return 1
  fi
  write_upstream_summary
  input="$transaction_root/pack-input.json"
  jq -n --argjson context "$context" --arg root "$ROOT" --arg output "$destination" \
    --arg candidate "$CERTIFIED_SHA" --arg base "$CANDIDATE_BASE_HEAD" --arg branch "$BRANCH" \
    --arg pass_id "$PASS_ID" --arg mode "$HARNESS_MODE" --arg test_build "$STATE_DIR/mm-build.jsonl" \
    --arg bundle "$BUNDLE" --arg summary "$UPSTREAM_SUMMARY" \
    --arg workload_oracles "${LLAMA_STAGE_BUILD_DIR:?}-workloads" --arg admitted_plan "$admission" \
    --arg admitted_identity_sha256 "$admission_sha" --arg producer_receipt "$CANARY_PRODUCER_RECEIPT" \
    --arg producer_receipt_sha256 "$CANARY_PRODUCER_RECEIPT_SHA" \
    '{context:$context,root:$root,output:$output,candidate:$candidate,base:$base,branch:$branch,pass_id:$pass_id,mode:$mode,
      test_build:$test_build,bundle:(if $mode == "pinned-build" then null else $bundle end),summary:$summary,
      workload_oracles:$workload_oracles,admitted_plan:$admitted_plan,admitted_identity_sha256:$admitted_identity_sha256,
      producer_receipt:$producer_receipt,producer_receipt_sha256:$producer_receipt_sha256}' > "$input" || return 1
  if "${MESH_LLM_AUTOMATION_BIN:?}" automation canary-receipts pack --input "$input"; then
    :
  else
    record_failure_class infrastructure artifact-export
    return 1
  fi
}

if ! check_family_cache; then
  record_failure_class infrastructure model-cache
  echo "pinned model cache is not ready; candidate source was not evaluated" >&2
  exit 125
fi

if [[ "$HARNESS_MODE" == repair* ]]; then
  restore_previous_repair_candidate
  write_repair_pin
  verify_repair_pin
  echo "starting agent repair/build gates; distributed families follow in separate jobs"
  if repair_candidate_until_green; then
    :
  else
    status=$?
    # Run the helper from the trusted base commit: a failed agent may have
    # edited its checkout's scripts. The result is diagnostic evidence only.
    if ! python3 - "$ROOT" "$STATE_DIR/recovery" "$BASE_HEAD" \
        < <(git show "$BASE_HEAD:scripts/llama-canary-recover-source.py"); then
      echo "could not capture the unverified repair source" >&2
    fi
    echo "agent task failed or timed out; no canary branch or pull request was published" >&2
    exit "$status"
  fi
  snapshot_candidate_tree
  write_candidate_bundle
  if [[ "$HARNESS_MODE" == "repair-build" ]]; then
    export_family_inputs
  fi
  exit 0
fi

if [[ "$HARNESS_MODE" == "pinned-build" ]]; then
  CERTIFIED_SHA="$BASE_HEAD"
  VERIFICATION_DEADLINE_AT="$(( $(date +%s) + VERIFICATION_TIMEOUT_SECONDS ))"
  run_candidate_gates
  check_split_certification_roster
  export_family_inputs
  exit 0
fi

load_candidate_bundle
trap cleanup_verification_worktree EXIT
materialize_verification_tree
VERIFICATION_DEADLINE_AT="$(( $(date +%s) + VERIFICATION_TIMEOUT_SECONDS ))"
if [[ "$HARNESS_MODE" == verify ]]; then
  verification_source_inspection verification-source-admit || exit 1
fi
echo "starting independent verification build of the exact candidate"
if run_candidate_gates; then
  :
else
  status=$?
  record_failure_class candidate independent-verification
  echo "final canary verification failed; no canary branch or pull request was published" >&2
  exit "$status"
fi
check_split_certification_roster
if [[ "$HARNESS_MODE" == "verify-build" ]]; then
  cp "$CANARY_INPUT_BUNDLE" "$BUNDLE"
  export_family_inputs
else
  finalize_certified_tree
fi
