#!/usr/bin/env bash
set -euo pipefail

# Developer-style llama.cpp canary harness for changed upstream pins.
#
# One agent owns the complete pin/patch/test task. After that agent exits, this
# trusted wrapper runs one fresh deterministic verification pass and creates a
# local certified commit. GitHub credentials and publication live in a later
# workflow step, so an agent or verification failure cannot publish a branch or
# pull request.

# The persistent Apple Silicon runner service can be launched by an x86_64
# parent under Rosetta. Re-enter the complete harness as arm64 before it
# configures or executes native build artifacts.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]] \
    && [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
  exec arch -arm64 "${BASH_SOURCE[0]}" "$@"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRUSTED_ROOT="$ROOT"
HARNESS_MODE="${CANARY_HARNESS_MODE:-repair}"
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
AGENT_TIMEOUT_SECONDS="${CANARY_AGENT_TIMEOUT_SECONDS:-27000}"
VERIFICATION_TIMEOUT_SECONDS="${CANARY_VERIFICATION_TIMEOUT_SECONDS:-14400}"
RUN_ID="${GITHUB_RUN_ID:-manual-$(date +%s)}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
RUN_KEY="${RUN_ID}-${RUN_ATTEMPT}"
BRANCH="llama-canary/repair-${RUN_KEY}-${UPSTREAM_SHA:0:10}"
VERIFY_ROOT="/tmp/mesh-llm-canary-verify-${RUN_KEY}"
STATE_DIR="$ROOT/.deps/llama-canary-state-${RUN_KEY}"
TARGET_SHA_FILE="$ROOT/.deps/llama-canary-target-sha"
AGENT_LOG="$STATE_DIR/agent.log"
PREPARE_LOG="$STATE_DIR/prepare.log"
BUILD_LOG="$STATE_DIR/build.log"
CERTIFY_LOG="$STATE_DIR/certify.log"
PR_BODY="$STATE_DIR/pr-body.md"
UPSTREAM_SUMMARY="$STATE_DIR/upstream-summary.md"
BUNDLE="$STATE_DIR/candidate.bundle"
EVIDENCE_DIR="$STATE_DIR/verification-evidence"
FAMILY_BATTERY_RUN_ID="${FAMILY_BATTERY_RUN_ID:-${RUN_KEY}}"
PLAN_PATH="$ROOT/target/family-battery/$FAMILY_BATTERY_RUN_ID/policy-plan.json"
BASE_HEAD="$(git rev-parse HEAD)"
BASE_REF="$(git symbolic-ref -q HEAD || true)"
GIT_CONFIG_FINGERPRINT="$(git config --list --show-origin | shasum -a 256 | awk '{print $1}')"
CERTIFIED_SHA=""
VERIFICATION_TREE=""
VERIFICATION_DEADLINE_AT=0

if [[ "$HARNESS_MODE" != "repair" && "$HARNESS_MODE" != "verify" ]]; then
  echo "CANARY_HARNESS_MODE must be repair or verify" >&2
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
if [[ -z "$(git config user.name)" || -z "$(git config user.email)" ]]; then
  echo "git user.name and user.email must be configured before canary repair" >&2
  exit 1
fi
if [[ "$HARNESS_MODE" == "repair" ]] && ! command -v opencode >/dev/null 2>&1; then
  echo "opencode CLI not found on runner; install opencode-ai on the family-certify image" >&2
  exit 1
fi
if [[ "$HARNESS_MODE" == "repair" \
    && -z "${OPENCODE_API_KEY:-}" && -z "${NEMOTRON_API_KEY:-}" ]]; then
  if [[ ! -s "${HOME}/.local/share/opencode/auth.json" ]] \
      && ! opencode auth list 2>/dev/null | grep -Eq '[1-9][0-9]* credentials'; then
    echo "no agent credentials: set OPENCODE_API_KEY/NEMOTRON_API_KEY or run 'opencode auth login' on the runner" >&2
    exit 1
  fi
fi

mkdir -p "$STATE_DIR" "$(dirname "$PLAN_PATH")"
rm -f "$AGENT_LOG" "$PREPARE_LOG" "$BUILD_LOG" "$CERTIFY_LOG" \
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
  python3 scripts/run-command-with-timeout.py \
    --seconds "$seconds" --label "$label" -- "$@"
}

remaining_verification_seconds() {
  local remaining
  remaining="$((VERIFICATION_DEADLINE_AT - $(date +%s)))"
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

The trusted harness has already written third_party/llama.cpp/upstream.txt to the exact target and recorded it in .deps/llama-canary-target-sha. Read ci/llama-canary/agent-repair-prompt.md and every repository skill it names, then own the work end to end: reproduce the queue failure, deliberately rebase or regenerate the owned patches, fix any generated-family rewriter or Rust ABI fallout, and run the canonical prepare, build, smoke, live-matrix, and full supported-family certification commands. Inspect each failure and keep iterating until every required command passes.

Do not weaken, skip, or narrow a gate. Do not edit the workflow, this wrapper, its publisher, the agent runbook, or their contract tests. Do not create or switch branches, commit, push, open a pull request, or use GitHub credentials. Leave the completed changes in this working tree. The harness will independently rerun the entire verification sequence and only a green exact tree can be published.' \
    "$UPSTREAM_SHA"
}

agent_turn() {
  local started heartbeat_pid status
  started="$(date +%s)"
  set -m
  # shellcheck disable=SC2016
  env -i PATH="$PATH" bash -c '
    root="$1"
    started="$2"
    while sleep 600; do
      newest="$(find "$root/.deps/llama.cpp" -type f -newer "$root/third_party/llama.cpp/upstream.txt" -print -quit 2>/dev/null || true)"
      printf "heartbeat: agent task running for %dm; recent llama.cpp activity: %s\n" \
        "$(( ($(date +%s) - started) / 60 ))" "${newest:-none observed yet}"
    done
  ' heartbeat "$ROOT" "$started" &
  heartbeat_pid=$!
  set +m
  set +e
  run_for "agent developer task" "$AGENT_TIMEOUT_SECONDS" env \
    -u GH_TOKEN -u GITHUB_TOKEN -u CANARY_REPAIR_TOKEN \
    opencode run --auto --model "$AGENT_MODEL" "$(agent_prompt)" \
    > >(tee -a "$AGENT_LOG") 2>&1
  status=$?
  set -e
  kill -- "-$heartbeat_pid" 2>/dev/null || kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
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

snapshot_candidate_tree() {
  assert_agent_control_unchanged || return 1
  verify_repair_pin || return 1
  git add -A
  if git diff --cached --quiet; then
    echo "agent produced no candidate changes to verify" >&2
    return 1
  fi
  VERIFICATION_TREE="$(git write-tree)"
  CERTIFIED_SHA="$(
    printf '%s\n\n%s\n' \
      "fix(llama): certify upstream ${UPSTREAM_SHA:0:10}" \
      "A single agent completed the upstream repair and the trusted harness independently passed the full changed-pin verification." \
      | git commit-tree "$VERIFICATION_TREE" -p "$BASE_HEAD"
  )"
}

write_candidate_bundle() {
  git -c core.hooksPath=/dev/null branch "$BRANCH" "$CERTIFIED_SHA"
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
  local input_bundle expected_head bundle_head
  input_bundle="${CANARY_INPUT_BUNDLE:?CANARY_INPUT_BUNDLE is required in verify mode}"
  expected_head="${CANARY_CANDIDATE_SHA:?CANARY_CANDIDATE_SHA is required in verify mode}"
  if [[ ! "$expected_head" =~ ^[0-9a-f]{40}$ || ! -s "$input_bundle" ]]; then
    echo "verification requires a non-empty candidate bundle and 40-hex head" >&2
    return 1
  fi
  git bundle verify "$input_bundle" >/dev/null
  bundle_head="$(git bundle list-heads "$input_bundle" "refs/heads/${BRANCH}" | awk '{print $1}')"
  if [[ "$bundle_head" != "$expected_head" ]]; then
    echo "candidate bundle head does not match the repair job output" >&2
    return 1
  fi
  git fetch "$input_bundle" "refs/heads/${BRANCH}"
  CERTIFIED_SHA="$expected_head"
  VERIFICATION_TREE="$(git rev-parse "${CERTIFIED_SHA}^{tree}")"
}

cleanup_verification_worktree() {
  local source
  if [[ -d "$VERIFY_ROOT" ]]; then
    mkdir -p "$EVIDENCE_DIR"
    for source in \
        "$VERIFY_ROOT/target/family-battery/$FAMILY_BATTERY_RUN_ID" \
        "$VERIFY_ROOT/target/skippy-stage-rewriter-check"; do
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
  PIN_FILE="$ROOT/third_party/llama.cpp/upstream.txt"
  FAMILY_BATTERY_RUN_ID="${RUN_KEY}-verification"
  PLAN_PATH="$ROOT/target/family-battery/$FAMILY_BATTERY_RUN_ID/policy-plan.json"
  LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR}-verification-${RUN_KEY}"
  LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR"
  export LLAMA_BUILD_DIR LLAMA_STAGE_BUILD_DIR FAMILY_BATTERY_RUN_ID
  rm -rf "$LLAMA_STAGE_BUILD_DIR" \
    "$ROOT/.deps/llama.cpp" \
    "$ROOT/target/family-battery/$FAMILY_BATTERY_RUN_ID" \
    "$ROOT/target/skippy-stage-rewriter-check"
  mkdir -p "$(dirname "$PLAN_PATH")"
  cd "$ROOT"
  verify_repair_pin
}

run_prepare() {
  local prepared_upstream
  : > "$PREPARE_LOG"
  echo "final verification: prepare" | tee -a "$PREPARE_LOG"
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
  echo "final verification: build" | tee -a "$BUILD_LOG"
  run_verification_logged "complete patched llama.cpp build" "$BUILD_LOG" env \
    LLAMA_STAGE_UPSTREAM_TESTS=ON uv run --no-project --with jinja2==3.1.6 -- \
    arch -arm64 bash scripts/build-llama.sh -DCMAKE_OSX_ARCHITECTURES=arm64 \
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
    cargo build -p skippy-runtime -p skippy-server -p skippy-model-package -p skippy-correctness \
    || return 1
  run_verification_logged "Skippy smoke tests" "$BUILD_LOG" \
    scripts/skippy-ci-smoke.sh || return 1
}

run_certification() {
  : > "$CERTIFY_LOG"
  echo "final verification: certify" | tee -a "$CERTIFY_LOG"
  run_verification_logged "parity manifest validation" "$CERTIFY_LOG" \
    python3 scripts/skippy-llama-parity.py --llama-src .deps/llama.cpp validate \
    || return 1
  run_verification_logged "full family certification plan" "$CERTIFY_LOG" \
    python3 scripts/plan-family-battery.py \
      --manifest ci/llama-canary/family-certified.json \
      --cadence llama-bump \
      --shard-count 1 \
      --check-cache \
      --cache-root "$HF_CACHE" \
      --output "$PLAN_PATH" \
    || return 1
  run_verification_logged "live package-v2 matrix" "$CERTIFY_LOG" env \
    FAMILY_BATTERY_RUN_ID="$FAMILY_BATTERY_RUN_ID" \
    SKIPPY_CANARY_LIVE_MATRIX_BACKEND="${SKIPPY_CANARY_LIVE_MATRIX_BACKEND:-metal}" \
    SKIPPY_CANARY_LIVE_MATRIX_ROOT="$ROOT/target/family-battery/$FAMILY_BATTERY_RUN_ID" \
    scripts/skippy-canary-live-matrix.sh --prepare || return 1
  run_verification_logged "full supported-family certification" "$CERTIFY_LOG" env \
    FAMILY_BATTERY_RUN_ID="$FAMILY_BATTERY_RUN_ID" \
    scripts/skippy-family-battery.sh --skip-build --plan "$PLAN_PATH"
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
    echo "One agent completed the pin and patch-queue task. The trusted harness then independently passed prepare, the complete patched llama.cpp and Rust build, Skippy smoke tests, the live package-v2 matrix, and the full supported-family certification on this exact commit."
    echo
    cat "$UPSTREAM_SUMMARY"
  } > "$PR_BODY"
}

finalize_certified_tree() {
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
  git -C "$TRUSTED_ROOT" bundle create "$BUNDLE" "$BRANCH" "^${BASE_HEAD}"
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

if [[ "$HARNESS_MODE" == "repair" ]]; then
  write_repair_pin
  verify_repair_pin
  echo "starting one agent developer task with a ${AGENT_TIMEOUT_SECONDS}s budget..."
  if ! agent_turn; then
    echo "agent task failed or timed out; no canary branch or pull request was published" >&2
    exit 1
  fi
  assert_agent_control_unchanged
  snapshot_candidate_tree
  write_candidate_bundle
  exit 0
fi

load_candidate_bundle
trap cleanup_verification_worktree EXIT
materialize_verification_tree
VERIFICATION_DEADLINE_AT="$(( $(date +%s) + VERIFICATION_TIMEOUT_SECONDS ))"
echo "starting one independent ${VERIFICATION_TIMEOUT_SECONDS}s verification pass..."
if ! run_prepare || ! run_full_build || ! run_certification; then
  echo "final canary verification failed; no canary branch or pull request was published" >&2
  exit 1
fi
finalize_certified_tree
