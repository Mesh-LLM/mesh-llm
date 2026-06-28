#!/usr/bin/env bash
# rc-release-smoke.sh - verify the clean release install and inference path.
#
# Usage:
#   scripts/rc-release-smoke.sh <release-tag>
#
# Useful overrides:
#   MESH_RC_RELEASE_ASSET=<asset-name>
#   MESH_RC_RELEASE_MODEL=<model-ref>
#   MESH_RC_RELEASE_API_PORT=<port>
#   MESH_RC_RELEASE_CONSOLE_PORT=<port>
#   MESH_RC_RELEASE_AUDIT_DIR=<path>

set -euo pipefail

RELEASE_TAG="${1:?usage: $0 <release-tag>}"
REPO="${MESH_RC_RELEASE_REPO:-Mesh-LLM/mesh-llm}"
MODEL_REF="${MESH_RC_RELEASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m}"
API_PORT="${MESH_RC_RELEASE_API_PORT:-19337}"
CONSOLE_PORT="${MESH_RC_RELEASE_CONSOLE_PORT:-13131}"
MAX_WAIT="${MESH_RC_RELEASE_MAX_WAIT:-240}"
AUDIT="${MESH_RC_RELEASE_AUDIT_DIR:-$(mktemp -d /tmp/mesh-llm-rc-smoke.XXXXXX)}"
HOME_DIR="$AUDIT/home"
XDG_CACHE="$AUDIT/xdg-cache"
XDG_DATA="$AUDIT/xdg-data"
LOG="$AUDIT/mesh-llm.log"
MESH_PID=""

release_url() {
    printf 'https://github.com/%s/releases/download/%s/%s\n' "$REPO" "$RELEASE_TAG" "$1"
}

canonical_arch() {
    case "$(uname -m)" in
        arm64|aarch64) printf 'aarch64\n' ;;
        amd64|x86_64) printf 'x86_64\n' ;;
        *) uname -m ;;
    esac
}

release_target() {
    case "$(uname -s)/$(canonical_arch)" in
        Linux/aarch64) printf 'aarch64-unknown-linux-gnu\n' ;;
        Linux/x86_64) printf 'x86_64-unknown-linux-gnu\n' ;;
        Darwin/aarch64) printf 'aarch64-apple-darwin\n' ;;
        *)
            echo "unsupported smoke host: $(uname -s)/$(uname -m)" >&2
            exit 1
            ;;
    esac
}

archive_ext() {
    case "$(uname -s)" in
        Darwin) printf 'zip\n' ;;
        *) printf 'tar.gz\n' ;;
    esac
}

default_asset_name() {
    printf 'mesh-llm-%s-%s.%s\n' "$RELEASE_TAG" "$(release_target)" "$(archive_ext)"
}

sha256_check() {
    local sidecar="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -c "$sidecar"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 -c "$sidecar"
    else
        echo "sha256sum or shasum is required" >&2
        exit 1
    fi
}

extract_archive() {
    local archive="$1"
    case "$archive" in
        *.tar.gz) tar -xzf "$archive" ;;
        *.zip) unzip -q "$archive" ;;
        *)
            echo "unsupported release archive: $archive" >&2
            exit 1
            ;;
    esac
}

mesh_env() {
    HOME="$HOME_DIR" XDG_CACHE_HOME="$XDG_CACHE" XDG_DATA_HOME="$XDG_DATA" "$@"
}

cleanup() {
    if [[ -n "$MESH_PID" ]]; then
        kill "$MESH_PID" 2>/dev/null || true
        pkill -P "$MESH_PID" 2>/dev/null || true
        wait "$MESH_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

ASSET="${MESH_RC_RELEASE_ASSET:-$(default_asset_name)}"
BINARY="$AUDIT/mesh-bundle/mesh-llm"

mkdir -p "$HOME_DIR" "$XDG_CACHE" "$XDG_DATA"
cd "$AUDIT"

echo "=== RC release smoke ==="
echo "  release: $RELEASE_TAG"
echo "  asset:   $ASSET"
echo "  audit:   $AUDIT"
echo "  model:   $MODEL_REF"

curl -fL -O "$(release_url "$ASSET")"
curl -fL -O "$(release_url "$ASSET.sha256")"
sha256_check "$ASSET.sha256"
extract_archive "$ASSET"

"$BINARY" --version

echo "runtime list --available"
mesh_env "$BINARY" runtime list --available --json | tee "$AUDIT/runtime-available.json"

echo "runtime install"
mesh_env "$BINARY" runtime install --json | tee "$AUDIT/runtime-install.json"

echo "runtime list"
mesh_env "$BINARY" runtime list | tee "$AUDIT/runtime-list.txt"

echo "download $MODEL_REF"
mesh_env "$BINARY" download "$MODEL_REF"

MODEL_PATH="$(
    find -L "$XDG_CACHE/huggingface/hub" -type f -name '*.gguf' | head -1
)"
if [[ -z "$MODEL_PATH" ]]; then
    echo "download did not produce a GGUF under $XDG_CACHE/huggingface/hub" >&2
    exit 1
fi

mesh_env "$BINARY" serve \
    --log-format json \
    --gguf "$MODEL_PATH" \
    --port "$API_PORT" \
    --console "$CONSOLE_PORT" \
    >"$LOG" 2>&1 &
MESH_PID=$!

echo "waiting for /v1/models"
MODEL_ID=""
for second in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$MESH_PID" 2>/dev/null; then
        echo "mesh-llm exited before readiness" >&2
        tail -120 "$LOG" >&2 || true
        exit 1
    fi
    MODELS_JSON="$(curl -fsS "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null || true)"
    MODEL_ID="$(
        printf '%s' "$MODELS_JSON" |
            python3 -c 'import json,sys; data=json.load(sys.stdin).get("data", []); print(data[0].get("id", "") if data else "")' 2>/dev/null ||
            true
    )"
    if [[ -n "$MODEL_ID" ]]; then
        echo "/v1/models ready with $MODEL_ID"
        break
    fi
    if [[ "$second" -eq "$MAX_WAIT" ]]; then
        echo "timed out waiting for /v1/models" >&2
        tail -120 "$LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

echo "checking /v1/chat/completions"
CHAT_PAYLOAD="$(
    python3 - "$MODEL_ID" <<'PY'
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "Reply with exactly: rc-ok"}],
    "max_tokens": 16,
    "temperature": 0,
}))
PY
)"
CHAT_RESPONSE="$(
    curl -fsS "http://127.0.0.1:${API_PORT}/v1/chat/completions" \
        -H 'content-type: application/json' \
        -H 'authorization: Bearer mesh' \
        -d "$CHAT_PAYLOAD"
)"
printf '%s' "$CHAT_RESPONSE" >"$AUDIT/chat-completions.json"
printf '%s' "$CHAT_RESPONSE" |
    python3 -c 'import json,sys; content=json.load(sys.stdin)["choices"][0]["message"]["content"]; raise SystemExit(0 if content.strip() == "rc-ok" else 1)'

cleanup
MESH_PID=""

echo "verify no mesh-llm process remains"
if pgrep -f "$BINARY" >/dev/null 2>&1; then
    echo "mesh-llm process still running for $BINARY" >&2
    exit 1
fi

echo "RC release smoke passed"
