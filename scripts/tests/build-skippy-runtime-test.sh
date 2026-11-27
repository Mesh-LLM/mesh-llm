#!/usr/bin/env bash
# Reuse a validated source runtime offline and distinguish CUDA toolkit packages.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/mesh/scripts" "$work/mesh/target/debug/native-runtimes/skippy-test" "$work/bin"
cp "$root/scripts/build-skippy-runtime.sh" "$work/mesh/scripts/"
cat >"$work/mesh/scripts/skippy-source.sh" <<'EOF'
#!/bin/sh
echo 'pinned https://example.invalid/skippy.git deadbeef'
EOF
cat >"$work/mesh/scripts/skippy-pin.sh" <<'EOF'
#!/bin/sh
echo 'https://example.invalid/skippy.git deadbeef 0.80.3'
EOF
cat >"$work/mesh/scripts/fetch-skippy-runtime.sh" <<'EOF'
#!/bin/sh
echo called >>"$FETCH_LOG"
mkdir -p "$1/skippy-test"
echo '{"schema_version":2,"runtime":{"backend":{"kind":"cuda"},"release_version":"0.80.3","skippy_abi":"1"}}' >"$1/skippy-test/manifest.json"
EOF
cat >"$work/bin/nvcc" <<'EOF'
#!/bin/sh
echo "Cuda compilation tools, release $CUDA_TEST_MAJOR.0"
EOF
chmod +x "$work/mesh/scripts/"* "$work/bin/nvcc"
runtime="$work/mesh/target/debug/native-runtimes"
echo '{"schema_version":2,"runtime":{"backend":{"kind":"cuda"},"release_version":"0.80.3","skippy_abi":"1"}}' >"$runtime/skippy-test/manifest.json"
printf '%s\n' 'deadbeef cuda 12  ' >"$runtime/.built-from"
export PATH="$work/bin:$PATH" FETCH_LOG="$work/fetch.log" MESH_LLM_RUNTIME_FROM_SOURCE=0
CUDA_TEST_MAJOR=12 "$work/mesh/scripts/build-skippy-runtime.sh" --backend cuda >/dev/null
[[ ! -e "$FETCH_LOG" ]]
CUDA_TEST_MAJOR=13 "$work/mesh/scripts/build-skippy-runtime.sh" --backend cuda >/dev/null
[[ "$(wc -l <"$FETCH_LOG" | tr -d ' ')" == 1 ]]
[[ "$(cat "$runtime/.built-from")" == 'release deadbeef cuda 13' ]]
CUDA_TEST_MAJOR=13 "$work/mesh/scripts/build-skippy-runtime.sh" --backend cuda >/dev/null
[[ "$(wc -l <"$FETCH_LOG" | tr -d ' ')" == 1 ]]
echo 'skippy runtime cache: ok'
