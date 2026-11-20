#!/usr/bin/env bash
# Real inference through MeshLLM on a separately built Skippy runtime.
#
#   scripts/ci-smoke-test.sh <runtime-dir> <model.gguf>
#
# <runtime-dir> holds one or more unpacked Skippy native runtimes (the --out
# directory of Skippy's scripts/package-native-runtime.sh).
#
# The supplied runtime is the only one MeshLLM can find: the binary runs from
# a private copy (nothing beside it), with a throwaway HOME and cache, no
# inherited runtime settings and downloads off. The script first requires
# MeshLLM to refuse to serve without a runtime, then requires a completion
# from the server it started, on the model that server loaded, with a library
# from <runtime-dir> mapped into that process.
set -euo pipefail

if (($# != 2)); then
    echo "usage: $0 <runtime-dir> <model.gguf>" >&2
    exit 2
fi
runtime_dir="$(cd "$1" && pwd -P)"
model="$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")"
[[ -f "$model" ]] || { echo "model not found: $model" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
built="${MESH_LLM_BIN:-$REPO_ROOT/target/debug/mesh-llm}"
[[ -x "$built" ]] || { echo "mesh-llm not built: $built (run: just build)" >&2; exit 2; }
port="${MESH_LLM_SMOKE_PORT:-19337}"
url="http://127.0.0.1:$port"
missing_runtime_error="no compatible MeshLLM native runtime"

work="$(mktemp -d)"
pid=""
stop() { # stop and reap the server, leaving its exit status in $status
    status=0
    [[ -n "$pid" ]] || return 0
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || status=$?
    pid=""
}
cleanup() {
    stop
    rm -rf "$work"
}
trap cleanup EXIT

fail() {
    echo "FAIL: $1" >&2
    [[ -n "${2:-}" && -f "$2" ]] && tail -40 "$2" >&2
    exit 1
}
alive() { kill -0 "$pid" 2>/dev/null; }
get() { curl -sf --max-time 10 "$@"; }

mkdir -p "$work/home" "$work/bin"
cp "$built" "$work/bin/mesh-llm"
bin="$work/bin/mesh-llm"

if get "$url/v1/models" >/dev/null 2>&1; then
    fail "something is already listening on port $port (set MESH_LLM_SMOKE_PORT)"
fi

serve() { # serve <log> [bundle-dir]
    local bundle=()
    (($# > 1)) && bundle=(MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$2")
    env -u SKIPPY_NATIVE_RUNTIME_BUNDLE_DIR \
        -u SKIPPY_NATIVE_RUNTIME_CACHE_DIR \
        -u SKIPPY_NATIVE_RUNTIME_MANIFEST_URL \
        -u MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR \
        -u MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL \
        HOME="$work/home" \
        MESH_LLM_NATIVE_RUNTIME_CACHE_DIR="$work/cache" \
        MESH_LLM_NO_RUNTIME_DOWNLOAD=1 \
        ${bundle[@]+"${bundle[@]}"} \
        "$bin" serve --local-model-only --gguf "$model" --port "$port" --log-format json >"$1" 2>&1 &
    pid=$!
}

echo "== no runtime: MeshLLM must refuse to serve"
log="$work/reject.log"
serve "$log"
for _ in $(seq 1 60); do
    alive || break
    get "$url/v1/models" >/dev/null 2>&1 && fail "served with no native runtime supplied" "$log"
    sleep 2
done
alive && fail "did not exit without a native runtime (timed out)" "$log"
status=0
wait "$pid" || status=$?
pid=""
((status != 0)) || fail "exited 0 without a native runtime" "$log"
grep -q "$missing_runtime_error" "$log" || fail "exited $status without the missing-runtime error" "$log"

echo "== inference on $runtime_dir"
log="$work/serve.log"
serve "$log" "$runtime_dir"
ready=""
for _ in $(seq 1 90); do
    alive || fail "MeshLLM exited during startup" "$log"
    if get "$url/v1/models" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 2
done
[[ -n "$ready" ]] || fail "MeshLLM did not come up (timed out)" "$log"

# The model this server says it loaded, from its own log.
model_id="$(jq -r 'select(.event == "model_loading") | .model' "$log" 2>/dev/null | head -1)"
[[ -n "$model_id" ]] || fail "the server log names no loaded model" "$log"
get "$url/v1/models" | jq -e --arg m "$model_id" 'any(.data[]; .id == $m)' >/dev/null ||
    fail "port $port does not advertise this server's model $model_id" "$log"

# A library from the supplied runtime is mapped into this server.
if [[ -r "/proc/$pid/maps" ]]; then
    maps="$(cat "/proc/$pid/maps")"
else
    maps="$(lsof -p "$pid" 2>/dev/null || true)"
fi
grep -qF "$runtime_dir/" <<<"$maps" || fail "no library from $runtime_dir is loaded" "$log"

reply="$(get --max-time 120 "$url/v1/chat/completions" -H 'content-type: application/json' \
    -d "$(jq -n --arg m "$model_id" '{model:$m,max_tokens:24,messages:[{role:"user",content:"What is the capital of France?"}]}')")" ||
    fail "the completion request failed" "$log"
alive || fail "MeshLLM exited during the completion" "$log"
[[ "$(jq -r '.model // ""' <<<"$reply")" == "$model_id" ]] || fail "completion came from another model: $reply"
content="$(jq -r '.choices[0].message.content // ""' <<<"$reply")"
tokens="$(jq -r '.usage.completion_tokens // 0' <<<"$reply")"
[[ -n "$content" && "$tokens" -ge 1 ]] || fail "empty completion: $reply"

echo "model:  $model_id"
echo "reply:  $content"
echo "tokens: $tokens"
echo "smoke OK"
