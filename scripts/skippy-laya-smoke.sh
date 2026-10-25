#!/usr/bin/env bash
set -euo pipefail

# Real-model Laya smoke for the llama.cpp upstream canary. The canary cache is
# offline and operator-owned, so this script resolves and verifies the pinned
# artifact but never downloads it.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$ROOT"
source "$ROOT/scripts/lib/automation.sh"
MANIFEST="${LAYA_SMOKE_MANIFEST:-$ROOT/ci/model-artifacts/manifests/skippy-system-one-smoke.json}"
ARTIFACT_ID="${LAYA_SMOKE_ARTIFACT_ID:-family-laya-multilingual}"
CADENCE="${LAYA_SMOKE_CADENCE:-manual}"
BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-${SKIPPY_LLAMA_BUILD_DIR:-}}"
CLI="${LAYA_SMOKE_CLI:-${BUILD_DIR:+$BUILD_DIR/bin/llama-laya-cli}}"
MODEL_PATH="${LAYA_SMOKE_MODEL_PATH:-}"
DEVICE="${LAYA_SMOKE_DEVICE:-auto}"
WORK_DIR="${WORK_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/skippy-system-one-smoke}"
REPORT="${LAYA_SMOKE_REPORT:-$WORK_DIR/reports/laya.json}"

artifact_summary() {
  mesh_automation models resolve "$MANIFEST" \
    --artifact-id "$ARTIFACT_ID" \
    --cadence "$CADENCE" \
    --require-single-file
}

cached_artifact_path() {
  local repo="$1" revision="$2" file="$3" root snapshot
  for root in "${HF_CACHE:-}/hub" "${HF_HUB_CACHE:-}"; do
    if [[ -n "$root" && -d "$root" ]]; then
      snapshot="$root/models--${repo//\//--}/snapshots/$revision/$file"
      if [[ -f "$snapshot" ]]; then
        printf '%s\n' "$snapshot"
        return 0
      fi
    fi
  done
  return 0
}

main() {
  local summary repo revision file
  summary="$(artifact_summary)"
  repo="$(jq -r '.repo' <<<"$summary")"
  revision="$(jq -r '.revision' <<<"$summary")"
  file="$(jq -r '.file' <<<"$summary")"

  if [[ "${1:-}" == "--prewarm" ]]; then
    cat <<EOF
Laya smoke: pre-warm plan for $ARTIFACT_ID

    hf download "$repo" "$file" --revision "$revision"

The canary verifies the manifest-pinned size and SHA-256 before running the
golden read battery.
EOF
    return 0
  fi

  command -v jq >/dev/null 2>&1 || {
    echo "required command not found: jq" >&2
    return 2
  }
  [[ -n "$CLI" && -x "$CLI" ]] || {
    echo "llama-laya-cli not found: ${CLI:-<unset>}" >&2
    return 2
  }
  if [[ -z "$MODEL_PATH" ]]; then
    MODEL_PATH="$(cached_artifact_path "$repo" "$revision" "$file")"
  fi
  if [[ -z "$MODEL_PATH" ]]; then
    echo "pinned Laya fixture $repo/$file is not in the offline model cache" >&2
    echo "pre-warm it with: scripts/skippy-laya-smoke.sh --prewarm" >&2
    return 1
  fi
  mesh_automation models resolve "$MANIFEST" \
    --artifact-id "$ARTIFACT_ID" \
    --cadence "$CADENCE" \
    --require-single-file \
    --verify-root "$(dirname "$MODEL_PATH")" >/dev/null

  mkdir -p "$(dirname "$REPORT")"
  mesh_automation automation laya parity \
    --cli "$CLI" \
    --gguf "$MODEL_PATH" \
    --device "$DEVICE" \
    --timeout "${LAYA_SMOKE_TIMEOUT_SECS:-600}" \
    --json-out "$REPORT"
  echo "Laya smoke passed on device $DEVICE"
  echo "Laya report: $REPORT"
}

main "$@"
