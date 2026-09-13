#!/usr/bin/env bash
# Explicit producer for the CPU workload candidate and its monolithic oracles.
# Called through just; never touches the canary's Metal/native or Rust outputs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRINT_ENV=0
if [[ "${1:-}" == "--print-env" ]]; then
  PRINT_ENV=1
  shift
fi
if [[ $# != 1 || "$1" != /* || "$1" == *$'\n'* || "$1" == *$'\r'* ]]; then
  echo "usage: skippy-workload-oracles-build.sh [--print-env] ABSOLUTE_BUILD_ROOT" >&2
  exit 1
fi
BUILD_ROOT="$1"
NATIVE_DIR="$BUILD_ROOT/native"
CARGO_DIR="$BUILD_ROOT/cargo"
MANIFEST="$BUILD_ROOT/producer.json"
if (( PRINT_ENV == 1 )); then
  printf '%s\n' \
    "SKIPPY_WORKLOAD_ORACLE_SERVER=$NATIVE_DIR/bin/llama-server" \
    "SKIPPY_WORKLOAD_ORACLE_COMPLETION=$NATIVE_DIR/bin/llama-completion" \
    "SKIPPY_WORKLOAD_ORACLE_TTS=$NATIVE_DIR/bin/llama-tts" \
    "SKIPPY_WORKLOAD_NATIVE_BUILD_DIR=$NATIVE_DIR" \
    "SKIPPY_WORKLOAD_CANDIDATE_BIN_DIR=$CARGO_DIR/debug" \
    "SKIPPY_WORKLOAD_PRODUCER_MANIFEST=$MANIFEST"
  exit 0
fi
cd "$ROOT"
mkdir -p "$BUILD_ROOT"
python3 scripts/check-skippy-workload-candidate.py --write-source-snapshot "$BUILD_ROOT/source.json"
python3 scripts/llama-oracle-source.py
export LLAMA_STAGE_BACKEND=cpu LLAMA_STAGE_LINK_MODE=static
export LLAMA_BUILD_DIR="$NATIVE_DIR" LLAMA_STAGE_BUILD_DIR="$NATIVE_DIR"
export LLAMA_STAGE_WORKLOAD_ORACLE=ON LLAMA_STAGE_UPSTREAM_TESTS=OFF
export LLAMA_STAGE_FULL_REPLAY=OFF LLAMA_STAGE_BUILD_TESTS=OFF
export CARGO_TARGET_DIR="$CARGO_DIR"
native_args=()
if [[ "$(uname -s)" == Darwin ]]; then
  native_args+=(-DCMAKE_OSX_ARCHITECTURES=arm64)
fi
scripts/build-llama.sh "${native_args[@]}"
just with-lld cargo build --locked -p skippy-server -p skippy-model-package -p skippy-correctness
just with-lld cargo test --locked -p skippy-server --lib --no-run --message-format=json > "$BUILD_ROOT/test-artifacts.jsonl"
test_binary="$(jq -rs '[.[] | select(.reason == "compiler-artifact" and .profile.test == true and .target.name == "skippy_server" and .executable != null) | .executable] | unique | if length == 1 then .[0] else error("expected one skippy-server library test binary") end' "$BUILD_ROOT/test-artifacts.jsonl")"
python3 scripts/check-skippy-workload-candidate.py \
  --candidate-binary "$CARGO_DIR/debug/skippy-server" \
  --native-build-dir "$NATIVE_DIR" \
  --test-binary "$test_binary" --write-producer "$MANIFEST" --source-snapshot "$BUILD_ROOT/source.json"
