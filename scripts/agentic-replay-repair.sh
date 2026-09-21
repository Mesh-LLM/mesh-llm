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
PUBLICATION_DIR="$OUTPUT_DIR/repair-publication"
AGENT_PROVIDER="${REPLAY_AGENT_PROVIDER:-zai_coding_plan}"
AGENT_MODEL="${REPLAY_AGENT_MODEL:-glm-5.3-flash}"
AGENT_SESSION_NAME="agentic-replay-repair-${RUN_ID}-${RUN_ATTEMPT}"

# shellcheck disable=SC2329 # invoked indirectly by the EXIT trap
cleanup() {
  [[ -z "${REPLAY_PARAMS_FILE:-}" ]] || rm -f -- "$REPLAY_PARAMS_FILE"
}
trap cleanup EXIT

run_untrusted() {
  env -u CANARY_REPAIR_TOKEN -u GH_TOKEN -u GITHUB_TOKEN -u HF_TOKEN "$@"
}

run_untrusted goose --version
run_untrusted python3 scripts/run-command-with-timeout.py \
  --seconds 60 --label agentic-replay-goose-preflight -- \
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
run_untrusted python3 scripts/run-command-with-timeout.py \
  --seconds 3600 --label agentic-replay-agent -- \
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
  run_untrusted python3 scripts/agentic-replay-params.py \
    --matrix "$MATRIX_FILE" --json-output "$REPLAY_PARAMS_FILE"
  REPLAY_DATASET_FILE="${DATASET_FILE:-${MESH_AGENTIC_REPLAY_DATASET_FILE:-}}"
  RERUN_FAILED=0
  if [[ -z "$REPLAY_DATASET_FILE" ]]; then
    echo "replay dataset file is unavailable — needs-attention" >&2
    RERUN_FAILED=1
  fi
  for family in $(run_untrusted python3 - "$MATRIX_FILE" <<'PY'
import json
import pathlib
import sys
print(" ".join(model["family"] for model in json.loads(pathlib.Path(sys.argv[1]).read_text())["models"]))
PY
  ); do
    if [[ "$RERUN_FAILED" == "1" && -z "$REPLAY_DATASET_FILE" ]]; then break; fi
    run_untrusted python3 scripts/agentic-replay-params.py \
      --matrix "$MATRIX_FILE" --run-family "$family" \
      --ref fixed=HEAD --ref "base=$BASE_SHA" \
      --dataset-file "$REPLAY_DATASET_FILE" \
      --output "$OUTPUT_DIR/repair/$family" || RERUN_FAILED=1
  done
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
  if [[ "$RERUN_FAILED" == "0" ]] && run_untrusted python3 scripts/agentic-replay-history.py "${HISTORY_ARGS[@]}"; then
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
BODY_FILE="$PUBLICATION_DIR/pr-body.md"
STATUS_FILE="$PUBLICATION_DIR/status.json"
BASE_SHA=$(git rev-parse HEAD^)
RESOLUTION="fix-verified"
BODY=$'The nightly agentic replay regressed; Goose analyzed the evidence and this fix passes the re-run benchmark.\n\nResults and repair logs are retained in the replay-artifacts workflow artifact.'

# A format-patch artifact is data for the hosted publisher. It is never
# sourced, executed, or used as a workflow/action definition in this job.
git format-patch -1 --binary --stdout HEAD > "$PATCH_FILE"

# Fill the PR template from the run evidence where available; fall back to
# the short body if the template or fill data is missing.
TEMPLATE_FILE="$(git rev-parse --show-toplevel)/.github/AGENTIC_REPLAY_REPAIR_PR_TEMPLATE.md"
if [[ -f "$TEMPLATE_FILE" ]]; then
  sed -e "s|{{RESOLUTION_STATUS}}|$( [[ $RESOLVED == 1 ]] && echo 'fix verified' || echo 'NEEDS ATTENTION' )|" \
      -e "s|{{RUN_URL}}|${GITHUB_SERVER_URL:-}/Mesh-LLM/mesh-llm/actions/runs/${RUN_ID}/attempts/${RUN_ATTEMPT}|g" \
      -e "s|{{RUN_DATE}}|$(date -u +%F)|g" \
      -e "s|{{RUN_ID}}|${RUN_ID}|g" \
      -e "s|{{SOURCE_SHA}}|${BASE_SHA}|g" \
      -e "s|{{DATASET_REPO}}|${DATASET_REPO:-meshllm/agentic-replay-nightly}|g" \
      -e "s|{{REGRESSING_COHORTS}}|${REPAIR_REGRESSING_COHORTS:-unavailable}|g" \
      -e "s|{{GATE_OUTPUT}}|see run artifacts|g" \
      -e "s|{{DIAGNOSIS}}|see Goose output in repair.log|g" \
      -e "s|{{FIX_SUMMARY}}|$( git log -1 --format=%s HEAD )|g" \
      -e "s|{{FILES_CHANGED}}|$( git diff --name-only HEAD~1..HEAD | tr '\n' ' ' )|g" \
      -e "s|{{RATIONALE}}|automated repair attempt|g" \
      -e "s|{{RESULT_ROWS}}|see history-repair.jsonl artifact|g" \
      -e "s|{{RERUN_GATE_RESULT}}|$( [[ $RESOLVED == 1 ]] && echo 'pass' || echo 'fail' )|g" \
      -e "s|{{BOOTSTRAP_STATE}}|${REPAIR_BOOTSTRAP_STATE:-unavailable}|g" \
      "$TEMPLATE_FILE" > "$BODY_FILE"
  printf '\n---\n%s\n' "$BODY" >> "$BODY_FILE"
else
  printf '%s\n' "$BODY" > "$BODY_FILE"
fi

python3 - "$STATUS_FILE" "$PATCH_FILE" "$BODY_FILE" "$RUN_ID" "$RUN_ATTEMPT" "$RESOLUTION" "$BASE_SHA" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

status_path = pathlib.Path(sys.argv[1])
patch_path = pathlib.Path(sys.argv[2])
body_path = pathlib.Path(sys.argv[3])
run_id, run_attempt, resolution, base_sha = sys.argv[4:]

if resolution not in {"fix-verified", "needs-attention"}:
    raise SystemExit(f"invalid repair resolution: {resolution!r}")
if run_id != "local" and not re.fullmatch(r"[0-9]+", run_id):
    raise SystemExit(f"invalid GitHub run id: {run_id!r}")
if not re.fullmatch(r"[1-9][0-9]*", run_attempt):
    raise SystemExit(f"invalid GitHub run attempt: {run_attempt!r}")
for name, value in (("base_sha", base_sha),):
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise SystemExit(f"invalid {name}: {value!r}")

def digest(path: pathlib.Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

patch_bytes = patch_path.stat().st_size
body_bytes = body_path.stat().st_size
if patch_bytes <= 0 or body_bytes <= 0:
    raise SystemExit("repair publication files must be non-empty")
metadata = {
    "base_sha": base_sha,
    "body_bytes": body_bytes,
    "body_sha256": digest(body_path),
    "patch_bytes": patch_bytes,
    "patch_sha256": digest(patch_path),
    "resolution": resolution,
    "run_id": run_id,
    "run_attempt": int(run_attempt),
    "schema_version": 1,
}
status_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
PY
echo "repair publication artifact written to $PUBLICATION_DIR ($RESOLUTION)"
exit 0 # the nightly is red; the hosted job owns publication
