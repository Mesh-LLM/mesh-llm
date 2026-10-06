#!/bin/bash
set -euo pipefail

# This script runs inside an HF Job container.
# It clones mesh-llm, builds the splitter, splits the model, validates, and publishes.
#
# Environment variables (set by mesh-llm skippy-model-package job spec):
#   SOURCE_REPO, SOURCE_FILE, SOURCE_QUANT, TARGET_REPO, MODEL_ID, SOURCE_REVISION
#   SOURCE_PROJECTOR_FILES — optional newline-delimited repo-relative mmproj GGUFs
#   SOURCE_PIPELINE_TAG — source model pipeline tag for the published model card
#   MESH_LLM_REF — git ref to build from (default: main)
#   CATALOG_CREATE_PR — "true" to open a PR for catalog updates (non-org members)
#   REPUBLISH — "true" to stage a replacement and atomically promote it to main
#   PACKAGE_EXPERIMENTAL — "true" to label the public package as not runtime-certified
#   HF_TOKEN — injected as a secret by HF Jobs
#
# Volumes:
#   /bucket  — writable storage bucket for script and fallback source cache

MESH_LLM_REF="${MESH_LLM_REF:-main}"
SOURCE_REVISION="${SOURCE_REVISION:-main}"
SOURCE_QUANT="${SOURCE_QUANT:-}"
REPUBLISH="${REPUBLISH:-false}"
: "${SOURCE_REPO:?SOURCE_REPO is required}"
if [ -z "$SOURCE_QUANT" ] && [[ "${MODEL_ID:-}" == *:* ]]; then
    SOURCE_QUANT="${MODEL_ID##*:}"
fi
if [ -z "$SOURCE_QUANT" ]; then
    echo "ERROR: SOURCE_QUANT is required to resolve the source GGUF without a model volume" >&2
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Layer Package Split Job                                 ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Source: ${SOURCE_REPO}/${SOURCE_FILE}"
echo "║  Quant:  ${SOURCE_QUANT}"
echo "║  Target: ${TARGET_REPO}"
echo "║  Model:  ${MODEL_ID}"
echo "║  Build:  mesh-llm @ ${MESH_LLM_REF}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Keep executable toolchains/build products on local ephemeral storage:
# HF bucket mounts can be unsuitable for dynamic loader/toolchain execution.
# Package artifacts are also written to the local work dir: the per-artifact
# upload+delete interleave keeps peak usage at one artifact plus its shard
# scratch, which fits the 50G ephemeral cap — and the upload hook re-reading
# artifacts through the writable /bucket FUSE mount surfaces I/O errors.
JOB_WORK_ROOT="${JOB_WORK_ROOT:-/bucket/job-work}"
SAFE_TARGET_REPO="$(printf '%s' "$TARGET_REPO" | tr -c '[:alnum:]._-' '_')"
LOCAL_WORK_DIR="${LOCAL_WORK_DIR:-/tmp/meshllm-layer-job-${SAFE_TARGET_REPO}-$$}"
if [ -z "${JOB_WORK_DIR:-}" ]; then
    JOB_WORK_DIR="${JOB_WORK_ROOT}/${SAFE_TARGET_REPO}-$(date +%Y%m%d%H%M%S)-$$"
    CLEANUP_JOB_WORK_DIR="${CLEANUP_JOB_WORK_DIR:-true}"
else
    CLEANUP_JOB_WORK_DIR="${CLEANUP_JOB_WORK_DIR:-false}"
fi
PACKAGE_DIR="${PACKAGE_DIR:-${LOCAL_WORK_DIR}/package}"
HF_HOME="${HF_HOME:-${JOB_WORK_DIR}/hf-home}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
# The Xet chunk cache must live on the container's local SSD: on network
# filesystems it performs poorly and surfaces I/O errors (the July convert
# wrapper learned this; the same os error 5 killed three split jobs through
# the /bucket FUSE mount on 2026-09-10).
HF_XET_CACHE="${HF_XET_CACHE:-${LOCAL_WORK_DIR}/xet-cache}"
# Route uploads through the classic HTTP path instead of Xet-CAS: HF Jobs
# containers hit sustained I/O errors (os error 5) on the Xet channel that
# do not reproduce outside the cluster, and per-layer GGUF artifacts do not
# benefit from chunk deduplication anyway. Set HF_HUB_DISABLE_XET=0 to
# restore the Xet uploader.
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
PACKAGE_DIR_ALLOW_BUCKET="${PACKAGE_DIR_ALLOW_BUCKET:-}"
JOB_TMP_DIR="${JOB_TMP_DIR:-${LOCAL_WORK_DIR}/tmp}"
BUILD_DIR="${BUILD_DIR:-${LOCAL_WORK_DIR}/build}"
TOOL_DIR="${TOOL_DIR:-${LOCAL_WORK_DIR}/tools}"
ARTIFACT_UPLOAD_HOOK="${ARTIFACT_UPLOAD_HOOK:-${LOCAL_WORK_DIR}/upload-package-artifact.sh}"
SNAPSHOT_PROMOTER="${SNAPSHOT_PROMOTER:-${TOOL_DIR}/promote-layer-package-snapshot}"
CARGO_HOME="${CARGO_HOME:-${LOCAL_WORK_DIR}/cargo-home}"
RUSTUP_HOME="${RUSTUP_HOME:-${LOCAL_WORK_DIR}/rustup-home}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${LOCAL_WORK_DIR}/cargo-target}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LOCAL_WORK_DIR}/xdg-cache}"
BUILD_TMP_DIR="${BUILD_TMP_DIR:-${LOCAL_WORK_DIR}/tmp}"
TMPDIR="$BUILD_TMP_DIR"
TEMP="$BUILD_TMP_DIR"
TMP="$BUILD_TMP_DIR"
export JOB_WORK_DIR PACKAGE_DIR HF_HOME HF_HUB_CACHE HF_XET_CACHE HF_HUB_DISABLE_XET
export TMPDIR TEMP TMP CARGO_HOME RUSTUP_HOME CARGO_TARGET_DIR XDG_CACHE_HOME

cleanup_job_work_dir() {
    if [ -n "${LOCAL_WORK_DIR:-}" ]; then
        echo "Cleaning local work dir: ${LOCAL_WORK_DIR}"
        rm -rf "$LOCAL_WORK_DIR" || true
    fi
    if [ "${CLEANUP_JOB_WORK_DIR}" = "true" ] && [ -n "${JOB_WORK_DIR:-}" ]; then
        echo "Cleaning job work dir: ${JOB_WORK_DIR}"
        rm -rf "$JOB_WORK_DIR" || true
    fi
}
trap cleanup_job_work_dir EXIT

log_storage_snapshot() {
    local label="$1"
    echo "  Storage snapshot (${label}):"
    df -h / /bucket "$PACKAGE_DIR" "$TMPDIR" 2>/dev/null || true
    echo "  Mounts (${label}):"
    mount | grep -E ' on / | on /bucket ' || true
}

on_error() {
    local status=$?
    local line=${BASH_LINENO[0]:-unknown}
    local command=${BASH_COMMAND:-unknown}
    echo "ERROR: split job command failed at line ${line} with status ${status}: ${command}" >&2
    log_storage_snapshot "error" >&2 || true
    exit "$status"
}
trap on_error ERR

start_heartbeat() {
    local label="$1"
    (
        while true; do
            sleep "${JOB_HEARTBEAT_SECONDS:-60}"
            echo "  Heartbeat (${label}) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            df -h / /bucket "$PACKAGE_DIR" "$TMPDIR" 2>/dev/null || true
            if [ -d "$PACKAGE_DIR" ]; then
                du -sh "$PACKAGE_DIR" 2>/dev/null || true
            fi
            if [ -d "$HF_HUB_CACHE" ]; then
                du -sh "$HF_HUB_CACHE" 2>/dev/null || true
            fi
        done
    ) &
    HEARTBEAT_PID=$!
}

stop_heartbeat() {
    if [ -n "${HEARTBEAT_PID:-}" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
        HEARTBEAT_PID=""
    fi
}

mkdir -p "$PACKAGE_DIR" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$JOB_TMP_DIR" "$TOOL_DIR" \
    "$CARGO_HOME" "$RUSTUP_HOME" "$CARGO_TARGET_DIR" "$XDG_CACHE_HOME" \
    "$BUILD_TMP_DIR"

format_bytes() { "$LAYER_JOB" format-bytes --bytes "$1"; }
estimate_bucket_workspace_bytes() { "$LAYER_JOB" workspace-estimate --bytes "$1"; }

# ─── Build tools ──────────────────────────────────────────────────────────
echo "=== [1/9] Installing build dependencies ==="
apt-get update -qq && apt-get install -y -qq \
    cmake git curl build-essential pkg-config libssl-dev sudo \
    > /dev/null 2>&1
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "=== [2/9] Installing Rust ==="
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y > /dev/null 2>&1
# shellcheck source=/dev/null
source "${CARGO_HOME}/env"
# The native promoter build uses the repository Just facade.
cargo install --locked just --version 1.58.0 > /dev/null 2>&1

# Fetch a raw commit SHA with retries: GitHub's ref advertisement for
# allow-any-SHA fetches ("upload-pack: not our ref") lags behind the push for
# freshly-pushed commits, which killed two jobs 90 seconds in. Poll until the
# commit is servable, up to 10 minutes.
fetch_ref_with_retry() {
    local ref="$1" attempt
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        if git fetch --depth 1 origin "$ref"; then
            return 0
        fi
        echo "  fetch of ${ref} failed (attempt ${attempt}/10); retrying in 60s..." >&2
        sleep 60
    done
    echo "ERROR: could not fetch ${ref} from origin after 10 attempts" >&2
    return 1
}

echo "=== [3/9] Cloning mesh-llm and building skippy-package-builder ==="
git clone --filter=blob:none https://github.com/Mesh-LLM/mesh-llm.git "$BUILD_DIR"
cd "$BUILD_DIR"
if git ls-remote --exit-code --heads origin "$MESH_LLM_REF" >/dev/null 2>&1 || \
   git ls-remote --exit-code --tags origin "$MESH_LLM_REF" >/dev/null 2>&1; then
    fetch_ref_with_retry "$MESH_LLM_REF"
    git checkout --detach FETCH_HEAD
elif git cat-file -e "$MESH_LLM_REF^{commit}" 2>/dev/null; then
    git checkout --detach "$MESH_LLM_REF"
else
    fetch_ref_with_retry "$MESH_LLM_REF"
    git checkout --detach FETCH_HEAD
fi

# Install the canonical compiler-cache/linker prerequisites for Just builds.
just bootstrap-build-tools

# Full clone needed for git-am patches in prepare-llama
sed -i 's/--filter=blob:none //' scripts/prepare-llama.sh
echo "  Running prepare-llama.sh..."
scripts/prepare-llama.sh pinned 2>&1 | tail -5
echo "  Running build-llama.sh..."
scripts/build-llama.sh 2>&1 | tail -5

# Locate the llama.cpp build directory (build-llama.sh puts it here)
LLAMA_BUILD_DIR=".deps/llama-build/build-stage-abi-cpu"
echo "  Verifying llama.cpp build at $LLAMA_BUILD_DIR..."
find "$LLAMA_BUILD_DIR" -name "*.a" 2>/dev/null | head -10 || echo "  WARNING: no .a files found"

# Build the splitter binary
echo "  Building skippy-package-builder..."
SKIPPY_LLAMA_BUILD_DIR="$LLAMA_BUILD_DIR" \
    cargo build --release -p skippy-package-builder 2>&1 | tail -20
SLICER="${CARGO_TARGET_DIR}/release/skippy-package-builder"
if [ ! -f "$SLICER" ]; then
    echo "ERROR: Build failed — binary not found at $SLICER"
    echo "Retrying with full output..."
    SKIPPY_LLAMA_BUILD_DIR=.deps/llama.cpp/build-stage-abi-static \
        cargo build --release -p skippy-package-builder 2>&1
    exit 1
fi
cp "$SLICER" "${TOOL_DIR}/skippy-package-builder"
just snapshot-promoter-release-build
just layer-job-helper-release-build
LAYER_JOB="${TOOL_DIR}/model-package-layer-job"
cp "${CARGO_TARGET_DIR}/release/model-package-layer-job" "$LAYER_JOB"
chmod +x "$LAYER_JOB"
cp "${CARGO_TARGET_DIR}/release/promote-layer-package-snapshot" "$SNAPSHOT_PROMOTER"
chmod +x "$SNAPSHOT_PROMOTER"
SLICER="${TOOL_DIR}/skippy-package-builder"
chmod +x "$SLICER"
cd /
rm -rf "$BUILD_DIR" "$CARGO_TARGET_DIR" "$CARGO_HOME" "$RUSTUP_HOME"
TMPDIR="$JOB_TMP_DIR"
TEMP="$JOB_TMP_DIR"
TMP="$JOB_TMP_DIR"
export TMPDIR TEMP TMP
echo "  ✓ Built: $SLICER"
echo "  Root filesystem after build cleanup:"
df -h / || true

echo "  Preparing native Hugging Face source and publication helpers..."
# Native source admission pins every optional sidecar to the resolved commit.
SOURCE_ADMISSION_DIR="${LOCAL_WORK_DIR}/source-admission"
SOURCE_CREDENTIAL="${JOB_TMP_DIR}/layer-job-credential"
(umask 077; printf '%s' "${HF_TOKEN}" > "$SOURCE_CREDENTIAL")
SOURCE_REVISION="$("$LAYER_JOB" source --repo "$SOURCE_REPO" --revision "${SOURCE_REVISION:-main}" --credential-file "$SOURCE_CREDENTIAL" --output-directory "$SOURCE_ADMISSION_DIR" --timeout-seconds 1200)"
rm -f "$SOURCE_CREDENTIAL"
export SOURCE_REVISION
echo "  Pinned source revision: $SOURCE_REVISION"
# Native explicit model repository creation preserves exist_ok and authenticates HEAD.
REPOSITORY_CREDENTIAL="${JOB_TMP_DIR}/layer-repository-credential"
(umask 077; printf '%s' "${HF_TOKEN}" > "$REPOSITORY_CREDENTIAL")
"$LAYER_JOB" ensure-repo --confirm --repo "$TARGET_REPO" \
    --credential-file "$REPOSITORY_CREDENTIAL" \
    --output-directory "${LOCAL_WORK_DIR}/repository-admission" --timeout-seconds 300
rm -f "$REPOSITORY_CREDENTIAL"
TARGET_UPLOAD_REVISION="main"
TARGET_MAIN_PARENT=""
if [ "$REPUBLISH" = "true" ]; then
    SNAPSHOT_OUTPUT="$(
        "$SNAPSHOT_PROMOTER" prepare --confirm \
            --repo "$TARGET_REPO" \
            --source-revision "$SOURCE_REVISION" \
            --token "$(date -u +%Y%m%d%H%M%S)-$$"
    )"
    mapfile -t SNAPSHOT_STATE <<< "$SNAPSHOT_OUTPUT"
    TARGET_UPLOAD_REVISION="${SNAPSHOT_STATE[0]:?missing staging revision}"
    TARGET_MAIN_PARENT="${SNAPSHOT_STATE[1]:?missing target main parent}"
    echo "  Staging replacement on ${TARGET_UPLOAD_REVISION} from ${TARGET_MAIN_PARENT}"
fi
export TARGET_UPLOAD_REVISION TARGET_MAIN_PARENT
# Native one-artifact publication. Preserve the original complete card/license fallback below.
export LAYER_JOB
cat > "$ARTIFACT_UPLOAD_HOOK" <<'BASH'
#!/bin/bash
set -euo pipefail
receipt_root="$(mktemp -d "${JOB_TMP_DIR}/layer-upload.XXXXXXXX")"
credential="${receipt_root}/credential"
(umask 077; printf '%s' "${HF_TOKEN}" > "$credential")
trap 'rm -f "$credential"' EXIT
"${LAYER_JOB}" upload --confirm \
    --repo "${TARGET_REPO}" --revision "${TARGET_UPLOAD_REVISION}" \
    --artifact "${SKIPPY_PACKAGE_ARTIFACT_PATH}" \
    --relative-path "${SKIPPY_PACKAGE_ARTIFACT_RELATIVE_PATH}" \
    --credential-file "$credential" --output-directory "${receipt_root}/result" \
    --maximum-attempts "${ARTIFACT_UPLOAD_ATTEMPTS:-8}" \
    --timeout-seconds "${ARTIFACT_UPLOAD_TIMEOUT_SECONDS:-3600}" --unlink-after-success
BASH
chmod +x "$ARTIFACT_UPLOAD_HOOK"

# ─── Split ────────────────────────────────────────────────────────────────
echo ""
echo "=== [4/9] Splitting model ==="
if [ "$SOURCE_REVISION" = "main" ]; then
    SOURCE_REF="${SOURCE_REPO}:${SOURCE_QUANT}"
else
    SOURCE_REF="${SOURCE_REPO}@${SOURCE_REVISION}:${SOURCE_QUANT}"
fi
echo "  Source ref: $SOURCE_REF"
if [ -n "${SOURCE_TOTAL_BYTES:-}" ]; then
    echo "  Source bytes: $SOURCE_TOTAL_BYTES"
    ESTIMATED_BUCKET_BYTES="$(estimate_bucket_workspace_bytes "$SOURCE_TOTAL_BYTES")"
    echo "  Estimated fallback /bucket cache needed: $(format_bytes "$ESTIMATED_BUCKET_BYTES")"
fi
MOUNTED_SOURCE_PATH="/source/${SOURCE_FILE}"
if [ -f "$MOUNTED_SOURCE_PATH" ]; then
    WRITE_PACKAGE_INPUT="$MOUNTED_SOURCE_PATH"
    WRITE_PACKAGE_IDENTITY_ARGS=(
        --model-id "$MODEL_ID"
        --source-repo "$SOURCE_REPO"
        --source-revision "$SOURCE_REVISION"
        --source-file "$SOURCE_FILE"
    )
    echo "  Source mount: $MOUNTED_SOURCE_PATH"
else
    WRITE_PACKAGE_INPUT="$SOURCE_REF"
    WRITE_PACKAGE_IDENTITY_ARGS=()
    echo "  Source mount: not available; falling back to Hugging Face cache download"
fi
WRITE_PACKAGE_PROJECTOR_ARGS=()
while IFS= read -r PROJECTOR_FILE; do
    if [ -z "$PROJECTOR_FILE" ]; then
        continue
    fi
    MOUNTED_PROJECTOR_PATH="/source/${PROJECTOR_FILE}"
    if [ -f "$MOUNTED_PROJECTOR_PATH" ]; then
        PROJECTOR_PATH="$MOUNTED_PROJECTOR_PATH"
    else
        echo "  Projector mount missing; downloading ${PROJECTOR_FILE} at ${SOURCE_REVISION}"
        PROJECTOR_RECEIPT_ROOT="$(mktemp -d "${JOB_TMP_DIR}/projector-acquisition.XXXXXXXX")"
        PROJECTOR_CREDENTIAL="${PROJECTOR_RECEIPT_ROOT}/credential"
        (umask 077; printf '%s' "${HF_TOKEN}" > "$PROJECTOR_CREDENTIAL")
        if PROJECTOR_PATH="$("$LAYER_JOB" projector --repo "$SOURCE_REPO" --revision "$SOURCE_REVISION" \
            --file "$PROJECTOR_FILE" --credential-file "$PROJECTOR_CREDENTIAL" \
            --output-directory "${PROJECTOR_RECEIPT_ROOT}/result" --timeout-seconds 3600)"; then
            rm -f "$PROJECTOR_CREDENTIAL"
        else
            PROJECTOR_STATUS=$?
            rm -f "$PROJECTOR_CREDENTIAL"
            exit "$PROJECTOR_STATUS"
        fi
    fi
    echo "  Projector: $PROJECTOR_PATH"
    WRITE_PACKAGE_PROJECTOR_ARGS+=(--projector "$PROJECTOR_PATH")
done <<< "${SOURCE_PROJECTOR_FILES:-}"
PUBLISHER_METADATA_DIR="$SOURCE_ADMISSION_DIR"
PUBLISHER_METADATA_PATHS=()
for METADATA_NAME in config.json generation_config.json tokenizer_config.json chat_template.jinja hf_quant_config.json; do
    if [ -f "$PUBLISHER_METADATA_DIR/$METADATA_NAME" ]; then
        PUBLISHER_METADATA_PATHS+=("$PUBLISHER_METADATA_DIR/$METADATA_NAME")
    fi
done
WRITE_PACKAGE_METADATA_ARGS=()
for METADATA_PATH in "${PUBLISHER_METADATA_PATHS[@]}"; do
    echo "  Publisher metadata: $METADATA_PATH"
    WRITE_PACKAGE_METADATA_ARGS+=(--publisher-metadata "$METADATA_PATH")
done
echo "  Hugging Face cache: $HF_HUB_CACHE"
echo "  Package workspace: $PACKAGE_DIR"
echo "  Temporary workspace: $TMPDIR"
log_storage_snapshot "before write-package"
# The package workspace must live on the container's local SSD. Two reasons:
# (1) HF Jobs evicts the pod once container-local ephemeral storage exceeds
#     50G, and writes through the /bucket FUSE mount count against that same
#     budget ~1:1, so staging on /bucket never avoided the wall — only the
#     per-artifact upload+delete interleave does; (2) the artifact upload hook
#     re-reads each artifact through the writable FUSE mount, which surfaces
#     I/O errors (os error 5) mid-upload. Re-create the directory here: empty
#     directories on the bucket FUSE mount are not backed by an object and can
#     disappear between the initial mkdir and this point.
mkdir -p "$PACKAGE_DIR"
if [ -n "$PACKAGE_DIR_ALLOW_BUCKET" ]; then
    echo "  NOTE: PACKAGE_DIR placement override active; /bucket EIO risk accepted." >&2
else
    ROOT_FS="$(df -P / | awk 'NR==2 {print $1}')"
    PACKAGE_FS="$(df -P "$PACKAGE_DIR" | awk 'NR==2 {print $1}')"
    if [ -n "$ROOT_FS" ] && [ "$ROOT_FS" != "$PACKAGE_FS" ]; then
        echo "ERROR: package workspace $PACKAGE_DIR is not on the container root filesystem. The /bucket FUSE mount counts writes against the same 50G ephemeral cap AND surfaces upload I/O errors; refusing to continue (unset PACKAGE_DIR_ALLOW_BUCKET to require local staging)." >&2
        exit 1
    fi
fi
if [ -n "${ESTIMATED_BUCKET_BYTES:-}" ]; then
    # This estimate covers the HF-cache fallback (full source under
    # HF_HUB_CACHE, which lives on /bucket). The local package workspace only
    # needs one artifact plus its shard scratch at a time.
    PACKAGE_AVAILABLE_BYTES="$(df -Pk /bucket | awk 'NR==2 {printf "%.0f", $4 * 1024}')"
    if [ -n "$PACKAGE_AVAILABLE_BYTES" ] && [ "$PACKAGE_AVAILABLE_BYTES" -gt 0 ] && \
        [ "$PACKAGE_AVAILABLE_BYTES" -lt "$ESTIMATED_BUCKET_BYTES" ]; then
        echo "WARNING: /bucket has $(format_bytes "$PACKAGE_AVAILABLE_BYTES") available for the source-cache fallback, below estimated need $(format_bytes "$ESTIMATED_BUCKET_BYTES")." >&2
    fi
fi
echo "  Starting write-package at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
WRITE_PACKAGE_GENERATION_ARGS=()
if [ -n "${GENERATION_DEFAULTS_JSON:-}" ]; then
    GENERATION_DEFAULTS_FILE="$TMPDIR/generation-defaults.json"
    printf '%s' "$GENERATION_DEFAULTS_JSON" > "$GENERATION_DEFAULTS_FILE"
    WRITE_PACKAGE_GENERATION_ARGS+=(--generation-defaults "$GENERATION_DEFAULTS_FILE")
    echo "  Generation defaults:"
    "$LAYER_JOB" generation-defaults --file "$GENERATION_DEFAULTS_FILE"
fi
start_heartbeat "write-package"
set +e
time "$SLICER" write-package "$WRITE_PACKAGE_INPUT" \
    --out-dir "$PACKAGE_DIR" \
    --after-artifact-command "$ARTIFACT_UPLOAD_HOOK" \
    "${WRITE_PACKAGE_PROJECTOR_ARGS[@]}" \
    "${WRITE_PACKAGE_METADATA_ARGS[@]}" \
    "${WRITE_PACKAGE_GENERATION_ARGS[@]}" \
    "${WRITE_PACKAGE_IDENTITY_ARGS[@]}"
WRITE_PACKAGE_STATUS=$?
set -e
stop_heartbeat
if [ "$WRITE_PACKAGE_STATUS" -ne 0 ]; then
    echo "ERROR: write-package failed with status $WRITE_PACKAGE_STATUS" >&2
    log_storage_snapshot "write-package failed" >&2 || true
    exit "$WRITE_PACKAGE_STATUS"
fi
echo "  Finished write-package at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_storage_snapshot "after write-package"

# The root/card projection retains full catalog/source identity; uploads are separate.
LAYER_PROJECTION_DIR="${LOCAL_WORK_DIR}/layer-projection"
LAYER_JOB_EXPERIMENTAL_ARGS=()
if [ "${PACKAGE_EXPERIMENTAL:-false}" = "true" ]; then LAYER_JOB_EXPERIMENTAL_ARGS+=(--experimental); fi
LAYER_PROJECTION_OUTPUT="$("$LAYER_JOB" project --manifest "$PACKAGE_DIR/model-package.json" \
    --source-repo "$SOURCE_REPO" --source-revision "$SOURCE_REVISION" \
    --source-admission-file "$SOURCE_ADMISSION_DIR/source.json" \
    --target-repo "$TARGET_REPO" --pipeline-tag "${SOURCE_PIPELINE_TAG:-text-generation}" \
    --output-directory "$LAYER_PROJECTION_DIR" "${LAYER_JOB_EXPERIMENTAL_ARGS[@]}")"
mapfile -t LAYER_PROJECTION_SUMMARY <<< "$LAYER_PROJECTION_OUTPUT"
if [ "${#LAYER_PROJECTION_SUMMARY[@]}" -ne 3 ]; then echo "ERROR: native projection summary refused" >&2; exit 1; fi
SOURCE_IDENTITY="${LAYER_PROJECTION_SUMMARY[0]}"
LAYER_COUNT="${LAYER_PROJECTION_SUMMARY[1]}"
TOTAL_SIZE="${LAYER_PROJECTION_SUMMARY[2]}"
TOTAL_SIZE_LABEL="$(format_bytes "$TOTAL_SIZE")"
echo "  Source identity: $SOURCE_IDENTITY"
echo "  ✓ Validated package root with $LAYER_COUNT layers; declared artifacts total $TOTAL_SIZE_LABEL"

# ─── Publish ──────────────────────────────────────────────────────────────
echo ""
echo "=== [6/9] Publishing to HuggingFace ==="
# Shared native publication entry for finite manifest/card files; no ambient token in argv.
publish_layer_file() {
    local file="$1" relative="$2" revision="$3" receipt_root credential
    receipt_root="$(mktemp -d "${JOB_TMP_DIR}/layer-file-upload.XXXXXXXX")"
    credential="${receipt_root}/credential"
    (umask 077; printf '%s' "${HF_TOKEN}" > "$credential")
    if ! "$LAYER_JOB" upload --confirm --repo "$TARGET_REPO" --revision "$revision" \
        --artifact "$file" --relative-path "$relative" --credential-file "$credential" \
        --output-directory "${receipt_root}/result" --maximum-attempts 8 --timeout-seconds 3600; then
        rm -f "$credential"
        return 1
    fi
    rm -f "$credential"
}
publish_layer_file "$PACKAGE_DIR/model-package.json" model-package.json "$TARGET_UPLOAD_REVISION"
echo "  ✓ Manifest publication verified on $TARGET_UPLOAD_REVISION"

if [ "$REPUBLISH" = "true" ]; then
    "$SNAPSHOT_PROMOTER" promote --confirm \
        --repo "$TARGET_REPO" \
        --manifest "$PACKAGE_DIR/model-package.json" \
        --staging-revision "$TARGET_UPLOAD_REVISION" \
        --parent-commit "$TARGET_MAIN_PARENT"
    echo "  ✓ Atomically promoted replacement snapshot to main"
fi

# ─── Update catalog ───────────────────────────────────────────────────────
echo ""
echo "=== [7/9] Updating meshllm/catalog ==="
CATALOG_CREDENTIAL="${JOB_TMP_DIR}/layer-catalog-credential"
(umask 077; printf '%s' "${HF_TOKEN}" > "$CATALOG_CREDENTIAL")
CATALOG_PR_ARGS=()
if [ "${CATALOG_CREATE_PR:-false}" = "true" ]; then CATALOG_PR_ARGS+=(--create-pr); fi
"$LAYER_JOB" update-catalog --confirm --manifest "$PACKAGE_DIR/model-package.json" \
    --source-repo "$SOURCE_REPO" --source-revision "$SOURCE_REVISION" --source-file "$SOURCE_FILE" \
    --target-repo "$TARGET_REPO" --credential-file "$CATALOG_CREDENTIAL" \
    --output-directory "${LOCAL_WORK_DIR}/catalog-publication" "${CATALOG_PR_ARGS[@]}"
rm -f "$CATALOG_CREDENTIAL"

# ─── Model Card ────────────────────────────────────────────────────────────
echo ""
echo "=== [8/9] Uploading model card ==="
CARD_CREDENTIAL="${JOB_TMP_DIR}/layer-card-credential"
(umask 077; printf '%s' "${HF_TOKEN}" > "$CARD_CREDENTIAL")
CARD_EXPERIMENTAL_ARGS=()
if [ "${PACKAGE_EXPERIMENTAL:-false}" = "true" ]; then CARD_EXPERIMENTAL_ARGS+=(--experimental); fi
"$LAYER_JOB" prepare-card --manifest "$PACKAGE_DIR/model-package.json" \
    --source-repo "$SOURCE_REPO" --source-revision "$SOURCE_REVISION" --source-file "$SOURCE_FILE" \
    --target-repo "$TARGET_REPO" --credential-file "$CARD_CREDENTIAL" \
    --pipeline-tag "${SOURCE_PIPELINE_TAG:-text-generation}" --mesh-llm-ref "${MESH_LLM_REF:-main}" \
    --output-directory "${LOCAL_WORK_DIR}/model-card" "${CARD_EXPERIMENTAL_ARGS[@]}"
rm -f "$CARD_CREDENTIAL"
publish_layer_file "${LOCAL_WORK_DIR}/model-card/README.md" README.md main
echo "  ✓ Model card publication verified"

# ─── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "=== [9/9] Done ==="
echo ""
echo "  Published:  https://huggingface.co/${TARGET_REPO}"
echo "  Layers:     ${LAYER_COUNT}"
echo "  Total size: ${TOTAL_SIZE_LABEL}"
echo ""
echo "  Use with mesh-llm:"
echo "    mesh-llm serve --model ${TARGET_REPO} --split"
