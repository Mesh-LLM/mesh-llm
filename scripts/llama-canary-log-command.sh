#!/usr/bin/env bash
set -euo pipefail

# Keep the complete output of a build or test run by the repair agent. The
# trusted harness exports this directory inside its uploaded evidence tree.
if (( $# < 2 )); then
  echo "usage: llama-canary-log-command.sh <label> <command> [args...]" >&2
  exit 2
fi
label="$1"
shift
if [[ ! "$label" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "log label must contain only letters, digits, underscores, or hyphens" >&2
  exit 2
fi
log_dir="${CANARY_REPAIR_LOG_DIR:?CANARY_REPAIR_LOG_DIR is required}"
mkdir -p "$log_dir"
command_dir="$(mktemp -d "$log_dir/${label}.XXXXXX")"
log="$command_dir/output.log"
printf 'repair command log: %s\n' "$log" >&2
set +e
{
  printf 'started: %s\nlabel: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label"
  "$@"
} 2>&1 | tee "$log"
pipeline_status=("${PIPESTATUS[@]}")
status=${pipeline_status[0]}
if (( status == 0 && pipeline_status[1] != 0 )); then
  status=${pipeline_status[1]}
fi
set -e
if [[ -n "${LLAMA_BUILD_DIR:-}" && -d "$LLAMA_BUILD_DIR/Testing/Temporary" ]]; then
  ctest_dir="$command_dir/ctest"
  mkdir -p "$ctest_dir"
  for ctest_file in LastTest.log LastTestsFailed.log CostData.txt; do
    if [[ -f "$LLAMA_BUILD_DIR/Testing/Temporary/$ctest_file" ]]; then
      cp "$LLAMA_BUILD_DIR/Testing/Temporary/$ctest_file" "$ctest_dir/" || true
    fi
  done
fi
printf 'finished: %s\nexit_status: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" | tee -a "$log"
exit "$status"
