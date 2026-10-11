#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    printf 'Usage: %s <mesh-llm-binary> <bin-dir> <model-path> [mmproj-path]\n' "$0" >&2
    exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/automation.sh"
binary="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
model="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
runtime_root="${MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR:-$(dirname "$binary")/native-runtimes}"
args=(automation required-smoke run --binary "$binary" --model "$model"
    --native-runtime-root "$runtime_root" --device "${MESH_CI_DEVICE:-CPU}"
    --ctx-size "${MESH_CI_CTX_SIZE:-256}" --ready-max-wait "${MESH_CI_MAX_WAIT:-180}"
    --api-port "${MESH_CI_API_PORT:-9337}" --console-port "${MESH_CI_CONSOLE_PORT:-3131}"
    --headless-api-port "${MESH_CI_HEADLESS_API_PORT:-9338}"
    --headless-console-port "${MESH_CI_HEADLESS_CONSOLE_PORT:-3132}")
if [[ -n "${4:-}" ]]; then args+=(--mmproj "$4"); fi
if [[ -n "${MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE:-}" ]]; then
    args+=(--public-key-file "$MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE"
        --expected-attestation "${MESH_RELEASE_ATTESTATION_EXPECTED_STATUS:-valid}")
fi
if [[ -n "${MESH_CI_BATCH_SIZE:-}" || -n "${MESH_CI_UBATCH_SIZE:-}" ]]; then
    args+=(--batch-size "${MESH_CI_BATCH_SIZE:-}" --ubatch-size "${MESH_CI_UBATCH_SIZE:-}")
fi
if [[ -n "${MESH_TOKIO_STACK_SIZE:-}" ]]; then args+=(--stack constrained); fi
mesh_automation "${args[@]}"
