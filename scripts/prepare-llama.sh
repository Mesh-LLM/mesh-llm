#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-pinned}"

LLAMA_UPSTREAM_URL="${LLAMA_UPSTREAM_URL:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_WORKDIR="${LLAMA_WORKDIR:-$ROOT/.deps/llama.cpp}"
PIN_FILE="${LLAMA_PIN_FILE:-$ROOT/third_party/llama.cpp/upstream.txt}"
PATCH_DIR="${LLAMA_PATCH_DIR:-$ROOT/third_party/llama.cpp/patches}"

if [[ ! -f "$PIN_FILE" ]]; then
  echo "missing llama upstream pin: $PIN_FILE" >&2
  exit 1
fi

if [[ ! -d "$PATCH_DIR" ]]; then
  echo "missing llama patch directory: $PATCH_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$LLAMA_WORKDIR")"

git_retry() {
  local attempt=1
  local max_attempts="${LLAMA_GIT_MAX_ATTEMPTS:-4}"
  local delay="${LLAMA_GIT_RETRY_DELAY_SECONDS:-10}"
  local status=0

  while (( attempt <= max_attempts )); do
    if "$@"; then
      return 0
    else
      status=$?
    fi

    if (( attempt == max_attempts )); then
      return "$status"
    fi

    echo "git command failed (attempt $attempt/$max_attempts): $*" >&2
    echo "retrying in ${delay}s..." >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

clone_llama_workdir() {
  local attempt=1
  local max_attempts="${LLAMA_GIT_MAX_ATTEMPTS:-4}"
  local delay="${LLAMA_GIT_RETRY_DELAY_SECONDS:-10}"
  local status=0

  while (( attempt <= max_attempts )); do
    rm -rf "$LLAMA_WORKDIR"
    if git clone --filter=blob:none "$LLAMA_UPSTREAM_URL" "$LLAMA_WORKDIR"; then
      return 0
    else
      status=$?
    fi

    if (( attempt == max_attempts )); then
      return "$status"
    fi

    echo "llama.cpp clone failed (attempt $attempt/$max_attempts)" >&2
    echo "retrying in ${delay}s..." >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

if [[ ! -d "$LLAMA_WORKDIR/.git" ]]; then
  clone_llama_workdir
fi

case "$MODE" in
  pinned)
    TARGET_SHA="$(tr -d '[:space:]' < "$PIN_FILE")"
    ;;
  latest)
    git -C "$LLAMA_WORKDIR" remote set-url origin "$LLAMA_UPSTREAM_URL"
    git_retry git -C "$LLAMA_WORKDIR" fetch origin master --tags
    TARGET_SHA="$(git -C "$LLAMA_WORKDIR" rev-parse origin/master)"
    ;;
  *)
    TARGET_SHA="$MODE"
    ;;
esac

PATCHES=()
while IFS= read -r patch; do
  PATCHES+=("$patch")
done < <(find "$PATCH_DIR" -maxdepth 1 -type f -name '*.patch' | sort)

compute_patch_digest() {
  (
    for patch in "${PATCHES[@]}"; do
      rel="${patch#$PATCH_DIR/}"
      checksum="$(shasum -a 256 "$patch" | awk '{print $1}')"
      printf '%s\n' "$rel"
      printf '%s\n' "$checksum"
    done
  ) | shasum -a 256 | awk '{print $1}'
}

PATCH_DIGEST="$(compute_patch_digest)"

if [[ -f "$LLAMA_WORKDIR/.mesh-llm-upstream-sha" &&
      -f "$LLAMA_WORKDIR/.mesh-llm-patched-sha" &&
      -f "$LLAMA_WORKDIR/.mesh-llm-patch-digest" ]]; then
  PREPARED_UPSTREAM="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-upstream-sha")"
  PREPARED_PATCHED="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-patched-sha")"
  PREPARED_DIGEST="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-patch-digest")"
  CURRENT_HEAD="$(git -C "$LLAMA_WORKDIR" rev-parse HEAD 2>/dev/null || true)"

  if [[ "$PREPARED_UPSTREAM" == "$TARGET_SHA" &&
        "$PREPARED_PATCHED" == "$CURRENT_HEAD" &&
        "$PREPARED_DIGEST" == "$PATCH_DIGEST" &&
        ! -d "$LLAMA_WORKDIR/.git/rebase-apply" ]] &&
     git -C "$LLAMA_WORKDIR" diff-index --quiet HEAD --; then
    echo "llama.cpp already prepared"
    echo "  upstream: $TARGET_SHA"
    echo "  patched:  $PREPARED_PATCHED"
    echo "  workdir:  $LLAMA_WORKDIR"
    exit 0
  fi
fi

git -C "$LLAMA_WORKDIR" am --abort >/dev/null 2>&1 || true
git -C "$LLAMA_WORKDIR" remote set-url origin "$LLAMA_UPSTREAM_URL"
if [[ "$MODE" != "latest" ]]; then
  git_retry git -C "$LLAMA_WORKDIR" fetch origin master --tags
fi
git -C "$LLAMA_WORKDIR" config user.name "${GIT_AUTHOR_NAME:-Mesh-LLM CI}"
git -C "$LLAMA_WORKDIR" config user.email "${GIT_AUTHOR_EMAIL:-ci@mesh-llm.local}"

# The llama.cpp checkout is a generated dependency worktree. Local edits there
# should live in third_party/llama.cpp/patches, so reset before switching pins.
git -C "$LLAMA_WORKDIR" reset --hard HEAD
git -C "$LLAMA_WORKDIR" clean -fdx
git -C "$LLAMA_WORKDIR" checkout --force --detach "$TARGET_SHA"
git -C "$LLAMA_WORKDIR" reset --hard "$TARGET_SHA"
git -C "$LLAMA_WORKDIR" clean -fdx

printf '%s\n' "$TARGET_SHA" > "$LLAMA_WORKDIR/.mesh-llm-upstream-sha"

if (( ${#PATCHES[@]} > 0 )); then
  git -C "$LLAMA_WORKDIR" am --3way "${PATCHES[@]}"
fi

git -C "$LLAMA_WORKDIR" rev-parse HEAD > "$LLAMA_WORKDIR/.mesh-llm-patched-sha"
printf '%s\n' "$PATCH_DIGEST" > "$LLAMA_WORKDIR/.mesh-llm-patch-digest"

echo "prepared llama.cpp"
echo "  upstream: $TARGET_SHA"
echo "  patched:  $(cat "$LLAMA_WORKDIR/.mesh-llm-patched-sha")"
echo "  workdir:  $LLAMA_WORKDIR"
