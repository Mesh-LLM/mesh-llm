#!/usr/bin/env bash
set -euo pipefail

# Fetches the pinned whisper.cpp source that build-llama.sh folds into the
# patched llama.cpp build for speech recognition. whisper.cpp is built on that
# tree's ggml, so the pin must track a whisper.cpp commit whose ggml matches
# skippy/llama_cpp/upstream.txt.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WHISPER_UPSTREAM_URL="${WHISPER_UPSTREAM_URL:-https://github.com/ggml-org/whisper.cpp.git}"
WHISPER_WORKDIR="${WHISPER_WORKDIR:-$ROOT/.deps/whisper.cpp}"
PIN_FILE="${WHISPER_PIN_FILE:-$ROOT/skippy/llama_cpp/whisper-upstream.txt}"

if [[ ! -f "$PIN_FILE" ]]; then
  echo "missing whisper.cpp upstream pin: $PIN_FILE" >&2
  exit 1
fi
TARGET_SHA="$(tr -d '[:space:]' < "$PIN_FILE")"

mkdir -p "$(dirname "$WHISPER_WORKDIR")"
# Keep git discovery inside the workdir; see prepare-llama.sh.
WHISPER_WORKDIR_PARENT="$(cd "$(dirname "$WHISPER_WORKDIR")" && pwd -P)" || {
  echo "error: cannot resolve parent directory of WHISPER_WORKDIR ($WHISPER_WORKDIR)" >&2
  exit 1
}
export GIT_CEILING_DIRECTORIES="$WHISPER_WORKDIR_PARENT"

if [[ -d "$WHISPER_WORKDIR/.git" ]] &&
   [[ "$(git -C "$WHISPER_WORKDIR" rev-parse HEAD 2>/dev/null)" == "$TARGET_SHA" ]] &&
   git -C "$WHISPER_WORKDIR" diff-index --quiet HEAD --; then
  echo "whisper.cpp already at $TARGET_SHA"
  exit 0
fi

fetch_pin() {
  local attempt=1
  local max_attempts="${WHISPER_GIT_MAX_ATTEMPTS:-4}"
  local delay="${WHISPER_GIT_RETRY_DELAY_SECONDS:-10}"
  while (( attempt <= max_attempts )); do
    if git -C "$WHISPER_WORKDIR" fetch --depth 1 origin "$TARGET_SHA"; then
      return 0
    fi
    if (( attempt == max_attempts )); then
      return 1
    fi
    echo "whisper.cpp fetch failed (attempt $attempt/$max_attempts); retrying in ${delay}s" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

if [[ ! -d "$WHISPER_WORKDIR/.git" ]]; then
  rm -rf "$WHISPER_WORKDIR"
  git init -q "$WHISPER_WORKDIR"
  git -C "$WHISPER_WORKDIR" remote add origin "$WHISPER_UPSTREAM_URL"
else
  git -C "$WHISPER_WORKDIR" remote set-url origin "$WHISPER_UPSTREAM_URL"
fi
fetch_pin
git -C "$WHISPER_WORKDIR" -c advice.detachedHead=false checkout -q --force "$TARGET_SHA"
git -C "$WHISPER_WORKDIR" clean -fdxq

echo "prepared whisper.cpp"
echo "  upstream: $TARGET_SHA"
echo "  workdir:  $WHISPER_WORKDIR"
