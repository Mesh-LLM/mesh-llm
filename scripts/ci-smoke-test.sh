#!/usr/bin/env bash
# Real inference through MeshLLM on a separately built Skippy runtime.
#
#   scripts/ci-smoke-test.sh <runtime-dir> <model.gguf>
#
# <runtime-dir> holds one or more unpacked Skippy native runtimes (the --out
# directory of Skippy's scripts/package-native-runtime.sh). Runs with a
# throwaway HOME and with runtime downloads off, so only the supplied runtime
# can serve. First checks that MeshLLM refuses to serve without a runtime.
set -euo pipefail

if (($# != 2)); then
    echo "usage: $0 <runtime-dir> <model.gguf>" >&2
    exit 2
fi
runtime_dir="$(cd "$1" && pwd -P)"
model="$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")"
[[ -f "$model" ]] || { echo "model not found: $model" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin="${MESH_LLM_BIN:-$REPO_ROOT/target/debug/mesh-llm}"
[[ -x "$bin" ]] || { echo "mesh-llm not built: $bin (run: just build)" >&2; exit 2; }
port="${MESH_LLM_SMOKE_PORT:-19337}"

work="$(mktemp -d)"
pid=""
cleanup() {
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    rm -rf "$work"
}
trap cleanup EXIT

serve() { # serve <bundle-dir> <log>
    HOME="$work/home" \
        MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$1" \
        MESH_LLM_NATIVE_RUNTIME_CACHE_DIR="$work/cache" \
        MESH_LLM_NO_RUNTIME_DOWNLOAD=1 \
        "$bin" serve --local-model-only --gguf "$model" --port "$port" --log-format json >"$2" 2>&1 &
    pid=$!
}

wait_ready() { # 0 once the API answers, 1 if the server exits or times out
    for _ in $(seq 1 90); do
        curl -sf "localhost:$port/v1/models" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

mkdir -p "$work/home" "$work/empty"

echo "== no runtime: MeshLLM must refuse to serve"
serve "$work/empty" "$work/reject.log"
if wait_ready; then
    echo "FAIL: served a model with no native runtime supplied" >&2
    exit 1
fi
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
pid=""
grep -q "native runtime" "$work/reject.log" || {
    echo "FAIL: no native-runtime error in the log:" >&2
    tail -20 "$work/reject.log" >&2
    exit 1
}

echo "== inference on $runtime_dir"
serve "$runtime_dir" "$work/serve.log"
wait_ready || {
    echo "FAIL: MeshLLM did not come up:" >&2
    tail -40 "$work/serve.log" >&2
    exit 1
}
model_id="$(curl -sf "localhost:$port/v1/models" | jq -r '.data[0].id')"
reply="$(curl -sf "localhost:$port/v1/chat/completions" -H 'content-type: application/json' \
    -d "$(jq -n --arg m "$model_id" '{model:$m,max_tokens:24,messages:[{role:"user",content:"What is the capital of France?"}]}')")"
content="$(jq -r '.choices[0].message.content // ""' <<<"$reply")"
tokens="$(jq -r '.usage.completion_tokens // 0' <<<"$reply")"
if [[ -z "$content" || "$tokens" -lt 1 ]]; then
    echo "FAIL: empty completion: $reply" >&2
    exit 1
fi
echo "model:  $model_id"
echo "reply:  $content"
echo "tokens: $tokens"
echo "smoke OK"
