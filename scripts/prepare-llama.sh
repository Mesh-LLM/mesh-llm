#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="${1:-pinned}"
LLAMA_UPSTREAM_URL="${LLAMA_UPSTREAM_URL:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_WORKDIR="${LLAMA_WORKDIR:-$REPO_ROOT/.deps/llama.cpp}"
PIN_FILE="${LLAMA_PIN_FILE:-$REPO_ROOT/third_party/llama.cpp/upstream.txt}"
PATCH_DIR="${LLAMA_PATCH_DIR:-$REPO_ROOT/third_party/llama.cpp/patches}"
LEGACY_LINK="${MESH_LLM_LLAMA_COMPAT_LINK:-1}"

if [[ ! -f "$PIN_FILE" ]]; then
    echo "missing llama.cpp upstream pin: $PIN_FILE" >&2
    exit 1
fi

if [[ ! -d "$PATCH_DIR" ]]; then
    echo "missing llama.cpp patch directory: $PATCH_DIR" >&2
    exit 1
fi

mkdir -p "$(dirname "$LLAMA_WORKDIR")"

if [[ ! -d "$LLAMA_WORKDIR/.git" ]]; then
    rm -rf "$LLAMA_WORKDIR"
    git clone "$LLAMA_UPSTREAM_URL" "$LLAMA_WORKDIR"
fi

git -C "$LLAMA_WORKDIR" am --abort >/dev/null 2>&1 || true
git -C "$LLAMA_WORKDIR" remote set-url origin "$LLAMA_UPSTREAM_URL"
git -C "$LLAMA_WORKDIR" fetch origin master --tags
if [[ "$(git -C "$LLAMA_WORKDIR" config --bool remote.origin.promisor || true)" == "true" ]]; then
    if git -C "$LLAMA_WORKDIR" fetch -h 2>&1 | grep -q -- "--unfilter"; then
        git -C "$LLAMA_WORKDIR" fetch --unfilter origin
    else
        git -C "$LLAMA_WORKDIR" fetch --refetch --filter=blob:limit=1g origin
    fi
fi
git -C "$LLAMA_WORKDIR" config user.name "${GIT_AUTHOR_NAME:-Mesh-LLM CI}"
git -C "$LLAMA_WORKDIR" config user.email "${GIT_AUTHOR_EMAIL:-ci@mesh-llm.local}"

case "$MODE" in
    pinned)
        TARGET_SHA="$(tr -d '[:space:]' < "$PIN_FILE")"
        ;;
    latest)
        TARGET_SHA="$(git -C "$LLAMA_WORKDIR" rev-parse origin/master)"
        ;;
    *)
        TARGET_SHA="$MODE"
        ;;
esac

git -c advice.detachedHead=false -C "$LLAMA_WORKDIR" checkout --detach --quiet "$TARGET_SHA"
git -C "$LLAMA_WORKDIR" reset --hard --quiet "$TARGET_SHA"
# Keep the CMake build directory so repeated local and CI builds can reuse
# compiler output. Build scripts own explicit clean behavior.
git -C "$LLAMA_WORKDIR" clean -fdx -e build/

printf '%s\n' "$TARGET_SHA" > "$LLAMA_WORKDIR/.git/mesh-llm-upstream-sha"

PATCHES=()
while IFS= read -r patch; do
    PATCHES+=("$patch")
done < <(find "$PATCH_DIR" -maxdepth 1 -type f -name '*.patch' | sort)

if (( ${#PATCHES[@]} > 0 )); then
    git -C "$LLAMA_WORKDIR" am --3way "${PATCHES[@]}"
fi

git -C "$LLAMA_WORKDIR" rev-parse HEAD > "$LLAMA_WORKDIR/.git/mesh-llm-patched-sha"

if [[ "$LEGACY_LINK" != "0" && "$LLAMA_WORKDIR" == "$REPO_ROOT/.deps/llama.cpp" ]]; then
    if [[ -L "$REPO_ROOT/llama.cpp" ]]; then
        ln -sfn ".deps/llama.cpp" "$REPO_ROOT/llama.cpp"
    elif [[ ! -e "$REPO_ROOT/llama.cpp" ]]; then
        ln -s ".deps/llama.cpp" "$REPO_ROOT/llama.cpp"
    else
        echo "note: $REPO_ROOT/llama.cpp already exists; not replacing it with a compatibility symlink" >&2
    fi
fi

echo "prepared llama.cpp"
echo "  upstream: $TARGET_SHA"
echo "  patched:  $(cat "$LLAMA_WORKDIR/.git/mesh-llm-patched-sha")"
echo "  workdir:  $LLAMA_WORKDIR"
