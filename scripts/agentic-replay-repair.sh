#!/usr/bin/env bash
# Regression repair loop for the agentic-replay nightly.
# Mirrors scripts/llama-canary-agent-repair.sh: on a gated regression, give
# Goose the run evidence and let it analyze + attempt a fix, re-run the
# benchmark, then emit a publication artifact for the trusted hosted job.
set -euo pipefail

# This script runs on the persistent runner. It never receives or publishes
# credentials; the separate hosted publication job owns that boundary.
unset CANARY_REPAIR_TOKEN GH_TOKEN GITHUB_TOKEN HF_TOKEN

OUTPUT_DIR="${1:?usage: agentic-replay-repair.sh <output-dir>}"
RUN_ID="${GITHUB_RUN_ID:-local}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
RESOLVED=0
MATRIX_FILE="${MATRIX_FILE:-ci/agentic-replay-nightly/matrix.json}"
REPLAY_PARAMS_FILE=""
REPLAY_MODELS_FILE=""
REPLAY_DATASET_PINS_FILE=""
PUBLICATION_DIR="$OUTPUT_DIR/repair-publication"
AGENT_PROVIDER="${REPLAY_AGENT_PROVIDER:-zai_coding_plan}"
AGENT_MODEL="${REPLAY_AGENT_MODEL:-glm-5.3-flash}"
AGENT_SESSION_NAME="agentic-replay-repair-${RUN_ID}-${RUN_ATTEMPT}"

# shellcheck disable=SC2329 # invoked indirectly by the EXIT trap
cleanup() {
  [[ -z "${REPLAY_PARAMS_FILE:-}" ]] || rm -f -- "$REPLAY_PARAMS_FILE"
  [[ -z "${REPLAY_MODELS_FILE:-}" ]] || rm -f -- "$REPLAY_MODELS_FILE"
  [[ -z "${REPLAY_DATASET_PINS_FILE:-}" ]] || rm -f -- "$REPLAY_DATASET_PINS_FILE"
}
trap cleanup EXIT

run_untrusted() {
  env -u CANARY_REPAIR_TOKEN -u GH_TOKEN -u GITHUB_TOKEN -u HF_TOKEN "$@"
}

# The typed owner inherits live streams and owns the entire command group.
# This handoff is built before Goose can modify any candidate source.
run_timed_untrusted() (
  local label="$1" seconds="$2" transaction_root input executable result
  shift 2
  transaction_root=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/replay-timeout.XXXXXXXX") || return 125
  trap 'rm -rf -- "$transaction_root"' EXIT
  input="$transaction_root/input.json"
  executable=$(command -v "$1") || return 125
  if [[ "$executable" != /* ]]; then
    echo "$label requires an absolute executable" >&2
    return 125
  fi
  shift
  jq -n --arg label "$label" --argjson seconds "$seconds" --arg cwd "$PWD" \
    --arg executable "$executable" --args \
    '{label:$label,seconds:$seconds,cwd:$cwd,executable:$executable,arguments:$ARGS.positional}' \
    -- "$@" > "$input" || return 125
  if run_untrusted cargo xtool automation canary-timeout --input "$input"; then
    result=0
  else
    result=$?
  fi
  exit "$result"
)

run_untrusted goose --version
run_timed_untrusted agentic-replay-goose-preflight 60 \
  env GOOSE_PROVIDER="$AGENT_PROVIDER" GOOSE_MODEL="$AGENT_MODEL" goose info --check
git config user.name "mesh-replay-bot"
git config user.email "replay-bot@meshllm.invalid"
git checkout -b "agentic-replay-nightly/repair-${RUN_ID}-${RUN_ATTEMPT}"
BASE_SHA=$(git rev-parse HEAD)

# Repair evidence and downloaded history are inputs, never source changes.
REPO_ROOT=$(git rev-parse --show-toplevel)
EXCLUDE_FILE=$(git rev-parse --git-path info/exclude)
for artifact_root in "$OUTPUT_DIR" "${HISTORY_LOCAL:-$REPO_ROOT/.replay-history-cache}"; do
  if [[ "$artifact_root" == "$REPO_ROOT/"* ]]; then
    relative_root=${artifact_root#"$REPO_ROOT/"}
    printf '/%s/\n' "${relative_root%/}" >> "$EXCLUDE_FILE"
  fi
done

# 1. Goose analyzes the regression evidence and attempts a fix. It must not
# see GitHub credentials. The same wrapper is used for every command that can
# execute repair-modified repository code.
run_timed_untrusted agentic-replay-agent 3600 \
  env GOOSE_MODE=auto GOOSE_DISABLE_SESSION_NAMING=true \
  goose run --provider "$AGENT_PROVIDER" --model "$AGENT_MODEL" \
    --with-builtin developer --no-profile --max-turns 1000 --output-format text \
    --name "$AGENT_SESSION_NAME" \
    --text "The nightly agentic replay benchmark on micstudio regressed at immutable base $BASE_SHA. Evidence: $OUTPUT_DIR/summary/history.jsonl and per-model artifacts in $OUTPUT_DIR. Analyze the regression against that exact base (git log $BASE_SHA is available) and attempt a minimal source fix. Do not edit .github, .agents, scripts, evals, or ci: the trusted harness owns verification and benchmark policy. Do not commit, change branches, push, or open PRs. Leave source changes uncommitted and return control for the full replay verification."

if [[ "$(git rev-parse HEAD)" != "$BASE_SHA" ]]; then
  echo "repair agent changed HEAD; no candidate will be published" >&2
  exit 1
fi
git add -A
if ! git diff --cached --quiet "$BASE_SHA" -- .github .agents scripts evals ci; then
  echo "repair agent changed protected verification or CI files; no candidate will be published" >&2
  exit 1
fi
if git diff --cached --quiet; then
  echo "Goose produced no changes; no candidate will be published" >&2
  exit 1
else
  git -c core.hooksPath=/dev/null commit --no-gpg-sign -m "fix: agentic replay nightly regression (run ${RUN_ID}-${RUN_ATTEMPT})

Attempted automated repair by Goose from nightly run evidence."

  # 2. Re-run the benchmark on the repaired tree with the nightly benchmark
  # shape, then re-normalize and gate the repaired summaries.
  REPLAY_PARAMS_FILE=$(mktemp "${TMPDIR:-/tmp}/agentic-replay-params.XXXXXX")
  run_untrusted cargo xtool automation replay-matrix export \
    --matrix "$MATRIX_FILE" --json-output "$REPLAY_PARAMS_FILE"
  REPLAY_DATASET_FILE="${DATASET_FILE:-${MESH_AGENTIC_REPLAY_DATASET_FILE:-}}"
  RERUN_FAILED=0
  if [[ -z "$REPLAY_DATASET_FILE" ]]; then
    echo "replay dataset file is unavailable — needs-attention" >&2
    RERUN_FAILED=1
  fi
  REPLAY_MODELS_FILE=$(mktemp "${TMPDIR:-/tmp}/agentic-replay-models.XXXXXX")
  REPLAY_DATASET_PINS_FILE=$(mktemp "${TMPDIR:-/tmp}/agentic-replay-dataset.XXXXXX")
  run_untrusted cargo xtool automation replay-matrix pins --matrix "$MATRIX_FILE" \
    --canonical skippy/evals/skippy-competitive-benchmark.json \
    --models-output "$REPLAY_MODELS_FILE" --dataset-output "$REPLAY_DATASET_PINS_FILE"
  while IFS=$'\t' read -r family _repo _revision _file expected_sha; do
    if [[ "$RERUN_FAILED" == "1" && -z "$REPLAY_DATASET_FILE" ]]; then break; fi
    model_path=""
    matches=0
    if [[ -f "${REPLAY_MODELS_LOCAL_FILE:-}" ]]; then
      while IFS=$'\t' read -r local_family local_path local_sha; do
        if [[ "$local_family" == "$family" ]]; then
          matches=$((matches + 1))
          if [[ "$local_sha" == "$expected_sha" ]]; then model_path="$local_path"; fi
        fi
      done < "$REPLAY_MODELS_LOCAL_FILE"
    fi
    if [[ "$matches" != 1 || ! -f "$model_path" ]]; then
      echo "verified local model mapping is unavailable for $family — needs-attention" >&2
      RERUN_FAILED=1
      continue
    fi
    run_untrusted cargo xtool automation replay-matrix run-family \
      --matrix "$MATRIX_FILE" --run-family "$family" --model-file "$model_path" \
      --python "${REPLAY_PYTHON:?locked replay interpreter required}" --timeout 21600 \
      --ref fixed=HEAD --ref "base=$BASE_SHA" \
      --dataset-file "$REPLAY_DATASET_FILE" \
      --output "$OUTPUT_DIR/repair/$family" || RERUN_FAILED=1
  done < "$REPLAY_MODELS_FILE"
  HISTORY_ARGS=(
    --matrix "$MATRIX_FILE"
    --replay-dir "$OUTPUT_DIR/repair"
    --label fixed
    --hardware "$OUTPUT_DIR/hardware.json"
    --source-sha "$(git rev-parse HEAD)"
    --replay "$REPLAY_PARAMS_FILE"
    --output "$OUTPUT_DIR/summary/history-repair.jsonl"
    --gate
  )
  HISTORY_RUNS="${HISTORY_LOCAL:-$REPO_ROOT/.replay-history-cache}/data/runs"
  if [[ -d "$HISTORY_RUNS" ]]; then
    HISTORY_ARGS+=(--baseline "$HISTORY_RUNS")
  fi
  if [[ "$RERUN_FAILED" == "0" ]] && run_untrusted cargo xtool automation replay-matrix history "${HISTORY_ARGS[@]}"; then
    RESOLVED=1
  fi
fi

if [[ "$RESOLVED" != "1" ]]; then
  echo "repair did not pass the complete replay gate; no candidate will be published" >&2
  exit 1
fi

# 3. Emit a publication artifact for the trusted hosted job. The persistent
# runner cannot receive publication credentials or publish the branch or PR.
if [[ ! -d "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
  echo "repair output directory must be a non-symlink directory: $OUTPUT_DIR" >&2
  exit 1
fi
rm -rf -- "$PUBLICATION_DIR"
install -d -m 700 -- "$PUBLICATION_DIR"
PATCH_FILE="$PUBLICATION_DIR/repair.patch"
BASE_SHA=$(git rev-parse HEAD^)
RESOLUTION="fix-verified"

# A format-patch artifact is data for the hosted publisher. It is never
# sourced, executed, or used as a workflow/action definition in this job.
git format-patch -1 --binary --stdout HEAD > "$PATCH_FILE"

# The trusted replay/history commands already succeeded before this data-only
# preparation. Credentials remain absent; only the hosted job publishes.
TEMPLATE_FILE="$(git rev-parse --show-toplevel)/.github/AGENTIC_REPLAY_REPAIR_PR_TEMPLATE.md"
PUBLICATION_ARGS=(
  --publication-dir "$PUBLICATION_DIR"
  --run-id "$RUN_ID" --run-attempt "$RUN_ATTEMPT" --base-sha "$BASE_SHA"
  --run-date "$(date -u +%F)" --server-url "${GITHUB_SERVER_URL:-}"
  --dataset-repo "${DATASET_REPO:-meshllm/agentic-replay-nightly}"
  --fix-summary "$(git log -1 --format=%s HEAD)"
  --files-changed "$(git diff --name-only HEAD~1..HEAD | tr '\n' ' ')"
  --regressing-cohorts "${REPAIR_REGRESSING_COHORTS:-unavailable}"
  --bootstrap-state "${REPAIR_BOOTSTRAP_STATE:-unavailable}"
  --history-outcome success --rerun-outcome success
)
if [[ -f "$TEMPLATE_FILE" ]]; then
  PUBLICATION_ARGS+=(--template "$TEMPLATE_FILE")
fi
run_untrusted cargo xtool automation replay-matrix publication-prepare "${PUBLICATION_ARGS[@]}"
echo "repair publication artifact written to $PUBLICATION_DIR ($RESOLUTION)"
exit 0 # the nightly is red; the hosted job owns publication
