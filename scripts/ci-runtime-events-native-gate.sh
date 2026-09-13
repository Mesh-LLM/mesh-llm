#!/usr/bin/env bash
# Run the real native runtime-event gate against a built native runtime and
# a real model, and write its evidence file.
#
# `crates/skippy-runtime/tests/runtime_events_native.rs` is the only test
# that exercises the reporter against actual native code: install the
# process-global reporter, open a real model, observe structured production
# callbacks, exercise unload, clear the reporter. It is gated behind
# MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST so an ordinary `cargo test` never
# touches a native symbol -- which also meant nothing in CI ever ran it, so
# the whole native reporter path was covered only by whoever remembered to
# run it by hand.
#
# This script is what a CI lane calls. It fails loudly rather than skipping:
# a lane that opts into the gate and then silently passes because a
# prerequisite was missing is worse than no lane at all.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat >&2 <<'USAGE'
usage: scripts/ci-runtime-events-native-gate.sh --bundle-dir <dir> --model <path> [--evidence <file>]

  --bundle-dir  Directory containing the built native runtime bundle.
  --model       Path to a real GGUF model to open.
  --evidence    Where to write the evidence markers.
                Defaults to $PWD/runtime-events-native-evidence.txt.
USAGE
}

BUNDLE_DIR=""
MODEL_PATH=""
EVIDENCE_FILE="$PWD/runtime-events-native-evidence.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bundle-dir)
            BUNDLE_DIR="${2:-}"
            shift 2
            ;;
        --model)
            MODEL_PATH="${2:-}"
            shift 2
            ;;
        --evidence)
            EVIDENCE_FILE="${2:-}"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$BUNDLE_DIR" || -z "$MODEL_PATH" ]]; then
    usage
    exit 2
fi
if [[ ! -d "$BUNDLE_DIR" ]]; then
    echo "native runtime bundle directory does not exist: $BUNDLE_DIR" >&2
    exit 1
fi
if [[ ! -s "$MODEL_PATH" ]]; then
    echo "model is missing or empty: $MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$(dirname "$EVIDENCE_FILE")"
# Cargo may run integration-test binaries from a package directory. Resolve
# the caller's evidence path before invoking it so both the test and this
# wrapper always refer to the same file.
EVIDENCE_DIR="$(cd "$(dirname "$EVIDENCE_FILE")" && pwd)"
EVIDENCE_FILE="$EVIDENCE_DIR/$(basename "$EVIDENCE_FILE")"
# Start from an empty file so the assertion below reads THIS run's markers,
# never a previous run's left behind by a warm workspace.
: >"$EVIDENCE_FILE"

cd "$ROOT"
MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST=1 \
    MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$BUNDLE_DIR" \
    MESH_LLM_RUNTIME_EVENTS_MODEL="$MODEL_PATH" \
    MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE="$EVIDENCE_FILE" \
    cargo test \
    --locked \
    -p skippy-runtime \
    --features dynamic-native-runtime \
    --test runtime_events_native \
    -- --nocapture

# The test writes `executed` only after the reporter installed, a real model
# opened, structured production callbacks were observed, unload was
# exercised where advertised, and the reporter cleared. A green exit with no
# `executed` marker means the gate took a blocked path -- which the test
# reports as a pass by design, so the lane has to check for itself.
if ! grep -q '^executed' "$EVIDENCE_FILE"; then
    echo "native runtime-event gate did not execute; evidence follows:" >&2
    cat "$EVIDENCE_FILE" >&2
    exit 1
fi

echo "native runtime-event gate executed; evidence at $EVIDENCE_FILE"
cat "$EVIDENCE_FILE"
