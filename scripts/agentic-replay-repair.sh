#!/usr/bin/env bash
# Regression repair loop for the agentic-replay nightly.
# Mirrors scripts/llama-canary-agent-repair.sh: on a gated regression, give
# opencode the run evidence and let it analyze + attempt a fix, re-run the
# benchmark, then always open a PR — labelled either as a verified fix or as
# an unresolved regression that needs human attention.
set -euo pipefail

REPAIR_TOKEN="${CANARY_REPAIR_TOKEN:-}"
# Keep the captured secret as a shell-only value. It is passed to a child only
# for the narrowly scoped credential operations below.
export -n REPAIR_TOKEN 2>/dev/null || true
unset CANARY_REPAIR_TOKEN GH_TOKEN GITHUB_TOKEN
if [[ -z "$REPAIR_TOKEN" ]]; then
  echo "CANARY_REPAIR_TOKEN is required to publish the repair PR" >&2
  exit 1
fi

OUTPUT_DIR="${1:?usage: agentic-replay-repair.sh <output-dir>}"
BRANCH="agentic-replay-nightly/repair-${GITHUB_RUN_ID:-local}"
RESOLVED=0
MATRIX_FILE="${MATRIX_FILE:-ci/agentic-replay-nightly/matrix.json}"
ASKPASS_SCRIPT=""
REPLAY_PARAMS_FILE=""
BODY_FILE=""

# shellcheck disable=SC2329 # invoked indirectly by the EXIT trap
cleanup() {
  [[ -z "${ASKPASS_SCRIPT:-}" ]] || rm -f -- "$ASKPASS_SCRIPT"
  [[ -z "${REPLAY_PARAMS_FILE:-}" ]] || rm -f -- "$REPLAY_PARAMS_FILE"
  [[ -z "${BODY_FILE:-}" ]] || rm -f -- "$BODY_FILE"
}
trap cleanup EXIT

run_untrusted() {
  env -u CANARY_REPAIR_TOKEN -u GH_TOKEN -u GITHUB_TOKEN -u REPAIR_TOKEN "$@"
}

gh_repair() {
  GH_TOKEN="$REPAIR_TOKEN" gh "$@"
}

redact_token() {
  REDACTION_TOKEN="$REPAIR_TOKEN" python3 -c \
    'import os, sys; sys.stdout.write(sys.stdin.read().replace(os.environ["REDACTION_TOKEN"], "***redacted***"))'
}

git config user.name "mesh-replay-bot"
git config user.email "replay-bot@meshllm.invalid"
git checkout -b "$BRANCH"

# Repair evidence and downloaded history are inputs, never source changes.
REPO_ROOT=$(git rev-parse --show-toplevel)
EXCLUDE_FILE=$(git rev-parse --git-path info/exclude)
for artifact_root in "$OUTPUT_DIR" "${HISTORY_LOCAL:-$REPO_ROOT/.replay-history-cache}"; do
  if [[ "$artifact_root" == "$REPO_ROOT/"* ]]; then
    relative_root=${artifact_root#"$REPO_ROOT/"}
    printf '/%s/\n' "${relative_root%/}" >> "$EXCLUDE_FILE"
  fi
done

# 1. opencode analyzes the regression evidence and attempts a fix. It must not
# see GitHub credentials. The same wrapper is used for every command that can
# execute repair-modified repository code.
run_untrusted opencode run --mode agent \
  "The nightly agentic replay benchmark on micstudio regressed. Evidence: $OUTPUT_DIR/summary/history.jsonl and per-model artifacts in $OUTPUT_DIR. Analyze the regression, identify the offending change (git log origin/main is available), and attempt a minimal fix. Do not touch ci/agentic-replay-nightly baselines or thresholds." || true

if [[ -z "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "opencode produced no changes — needs-attention" >&2
  git commit --allow-empty -m "chore: agentic replay nightly regression needs attention (run ${GITHUB_RUN_ID:-local})

Automated repair produced no changes; PR opened for human triage with the
run evidence attached."
else
  git add -A
  git commit -m "fix: agentic replay nightly regression (run ${GITHUB_RUN_ID:-local})

Attempted automated repair by opencode from nightly run evidence.

Co-authored-by: opencode <opencode@meshllm.invalid>"

  # 2. Re-run the benchmark on the repaired tree with the nightly benchmark
  # shape, then re-normalize and gate the repaired summaries.
  REPLAY_CONFIG="$(run_untrusted python3 - "$MATRIX_FILE" <<'PY'
import json
import pathlib
import sys

replay = json.loads(pathlib.Path(sys.argv[1]).read_text()).get("replay")
if not isinstance(replay, dict):
    raise SystemExit("matrix replay block is missing")
mode = replay.get("mode")
mode_map = {"checkpoint": "checkpoints", "final": "final", "all": "all"}
if mode not in mode_map:
    raise SystemExit(f"unsupported replay mode: {mode!r}")
values = {
    "trajectories_per_framework": replay.get("trajectories_per_framework"),
    "passes": replay.get("passes"),
    "warmup_turns": replay.get("warmup_turns"),
    "max_output_tokens": replay.get("max_output_tokens"),
}
for key, value in values.items():
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit(f"{key} must be a positive integer")
print(mode_map[mode], *(values[key] for key in (
    "trajectories_per_framework", "passes", "warmup_turns", "max_output_tokens")))
PY
)"
  read -r REPLAY_MODE TRAJECTORIES_PER_FRAMEWORK PASSES WARMUP_TURNS MAX_OUTPUT_TOKENS <<< "$REPLAY_CONFIG"
  LEVELS="$(run_untrusted python3 - "$MATRIX_FILE" <<'PY'
import json
import pathlib
import sys

levels = json.loads(pathlib.Path(sys.argv[1]).read_text())["replay"].get("concurrency")
if (
    not isinstance(levels, list)
    or not levels
    or any(isinstance(level, bool) or not isinstance(level, int) or level <= 0 for level in levels)
    or len(set(levels)) != len(levels)
):
    raise SystemExit("concurrency must be a non-empty list of unique positive integers")
print(" ".join(map(str, levels)))
PY
)"
  LEVEL_ARGS=()
  for level in $LEVELS; do LEVEL_ARGS+=(--concurrency "$level"); done
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
    model_uri=$(run_untrusted python3 - "$MATRIX_FILE" "$family" <<'PY'
import json
import pathlib
import sys

family = sys.argv[2]
model = next(model for model in json.loads(pathlib.Path(sys.argv[1]).read_text())["models"] if model["family"] == family)
print(f'{model["repo"]}@{model["revision"]}/{model["file"]}')
PY
    )
    run_untrusted python3 evals/agentic-replay.py run \
      --ref fixed=HEAD \
      --ref base=origin/main \
      --model "$model_uri" \
      --backend metal \
      --replay-mode "$REPLAY_MODE" \
      --trajectories-per-framework "$TRAJECTORIES_PER_FRAMEWORK" \
      "${LEVEL_ARGS[@]}" \
      --passes "$PASSES" \
      --warmup-turns "$WARMUP_TURNS" \
      --max-output-tokens "$MAX_OUTPUT_TOKENS" \
      --dataset-file "$REPLAY_DATASET_FILE" \
      --output "$OUTPUT_DIR/repair/$family" || RERUN_FAILED=1
  done
  REPLAY_PARAMS_FILE=$(mktemp "${TMPDIR:-/tmp}/agentic-replay-params.XXXXXX")
  # Process substitution breaks on micstudio (/dev/fd is not passable to the
  # child); write the replay parameters to a plain file instead.
run_untrusted python3 - "$MATRIX_FILE" <<'PY' > "$REPLAY_PARAMS_FILE"
import json
import pathlib
import sys
replay = json.loads(pathlib.Path(sys.argv[1]).read_text())["replay"]
print(json.dumps(replay, sort_keys=True))
PY
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

# 3. Always open a PR with the evidence; flag resolution status.
LABELS="agentic-replay,nightly"
TITLE="Agentic replay nightly regression — run ${GITHUB_RUN_ID:-local}"
TEMPLATE_FILE="$(git rev-parse --show-toplevel)/.github/AGENTIC_REPLAY_REPAIR_PR_TEMPLATE.md"
if [[ "$RESOLVED" == "1" ]]; then
  TITLE="$TITLE (fix verified)"
  BODY=$'The nightly agentic replay regressed; opencode analyzed the evidence and this fix passes the re-run benchmark.\n\nResults (HF card format) are linked in the run report artifact and the dataset shard.'
else
  LABELS="$LABELS,needs-attention"
  TITLE="$TITLE — NEEDS ATTENTION"
  BODY=$'The nightly agentic replay regressed and the automated repair did not clear it. Review the evidence: history.jsonl, per-model artifacts, and the HF dataset shard for this run.'
fi

# Push with a temporary askpass helper. The token is never present in the
# remote URL or process arguments, and the helper is removed on every exit.
ASKPASS_SCRIPT=$(mktemp "${TMPDIR:-/tmp}/agentic-replay-askpass.XXXXXX")
chmod 700 "$ASKPASS_SCRIPT"
cat > "$ASKPASS_SCRIPT" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  Username*) printf '%s\n' 'x-access-token' ;;
  Password*) printf '%s\n' "${CANARY_REPAIR_TOKEN:?}" ;;
  *) exit 1 ;;
esac
EOF
if ! CANARY_REPAIR_TOKEN="$REPAIR_TOKEN" \
  GIT_ASKPASS="$ASKPASS_SCRIPT" GIT_TERMINAL_PROMPT=0 \
  git -c core.hooksPath=/dev/null -c credential.helper= \
    push "https://github.com/${GITHUB_REPOSITORY:-Mesh-LLM/mesh-llm}.git" \
  "HEAD:refs/heads/${BRANCH}" 2> >(redact_token >&2); then
  echo "repair branch push failed" >&2
  exit 1
fi
rm -f -- "$ASKPASS_SCRIPT"
ASKPASS_SCRIPT=""

# Fill the PR template from the run evidence where available; fall back to
# the short body if the template or fill data is missing.
BODY_FILE=$(mktemp)
if [[ -f "$TEMPLATE_FILE" ]]; then
  sed -e "s|{{RESOLUTION_STATUS}}|$( [[ $RESOLVED == 1 ]] && echo 'fix verified' || echo 'NEEDS ATTENTION' )|" \
      -e "s|{{RUN_URL}}|${GITHUB_SERVER_URL:-}/Mesh-LLM/mesh-llm/actions/runs/${GITHUB_RUN_ID:-local}|g" \
      -e "s|{{RUN_DATE}}|$(date -u +%F)|g" \
      -e "s|{{RUN_ID}}|${GITHUB_RUN_ID:-local}|g" \
      -e "s|{{SOURCE_SHA}}|$(git rev-parse HEAD)|g" \
      -e "s|{{DATASET_REPO}}|${DATASET_REPO:-meshllm/agentic-replay-nightly}|g" \
      -e "s|{{REGRESSING_COHORTS}}|${REPAIR_REGRESSING_COHORTS:-unavailable}|g" \
      -e "s|{{GATE_OUTPUT}}|see run artifacts|g" \
      -e "s|{{DIAGNOSIS}}|see opencode session log|g" \
      -e "s|{{FIX_SUMMARY}}|$( git log -1 --format=%s HEAD )|g" \
      -e "s|{{FILES_CHANGED}}|$( git diff --name-only HEAD~1..HEAD | tr '\n' ' ' )|g" \
      -e "s|{{RATIONALE}}|automated repair attempt|g" \
      -e "s|{{RESULT_ROWS}}|see history-repair.jsonl artifact|g" \
      -e "s|{{RERUN_GATE_RESULT}}|$( [[ $RESOLVED == 1 ]] && echo 'pass' || echo 'fail' )|g" \
      -e "s|{{BOOTSTRAP_STATE}}|${REPAIR_BOOTSTRAP_STATE:-unavailable}|g" \
      "$TEMPLATE_FILE" >> "$BODY_FILE"
  printf '\n---\n%s\n' "$BODY" >> "$BODY_FILE"
else
  printf '%s\n' "$BODY" > "$BODY_FILE"
fi
gh_repair pr create --title "$TITLE" --body-file "$BODY_FILE" --label "$LABELS" --base main --head "$BRANCH" || \
  echo "PR creation failed — evidence retained in run artifacts" >&2
exit 0 # the nightly is red; the PR is the actionable output
