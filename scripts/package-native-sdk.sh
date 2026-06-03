#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD=0
OUT_DIR="$REPO_ROOT/dist/native-sdk"
BACKEND="${LLAMA_STAGE_BACKEND:-${SKIPPY_LLAMA_BACKEND:-cpu}}"
TARGET_TRIPLE="${MESH_NATIVE_SDK_TARGET:-}"
PROFILE="${MESH_NATIVE_SDK_PROFILE:-release}"
LLAMA_WORKDIR="${LLAMA_WORKDIR:-$REPO_ROOT/.deps/llama.cpp}"
LLAMA_BUILD_ROOT="${MESH_LLM_LLAMA_BUILD_ROOT:-$REPO_ROOT/.deps/llama-build}"

usage() {
    cat >&2 <<'EOF'
Usage: scripts/package-native-sdk.sh [options]

Package a backend-flavoured MeshLLM native SDK runtime artifact.

Options:
  --build             Build patched llama.cpp and mesh-llm-ffi before packaging.
  --backend NAME      cpu, metal, cuda, rocm, hip, or vulkan.
  --target TRIPLE     Rust target triple. Defaults to the host target.
  --profile PROFILE   Cargo profile to package: release or debug. Defaults to release.
  --out DIR           Output directory. Defaults to dist/native-sdk.
  -h, --help          Show this help.

Environment:
  LLAMA_STAGE_CUDA_ARCHITECTURES / SKIPPY_CUDA_ARCHITECTURES
  LLAMA_STAGE_AMDGPU_TARGETS / SKIPPY_AMDGPU_TARGETS
  LLAMA_STAGE_BUILD_DIR
  MESH_NATIVE_SDK_TARGET
  MESH_NATIVE_SDK_PROFILE
  MESH_LLM_LLAMA_PIN_SHA
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --build)
            BUILD=1
            shift
            ;;
        --backend)
            BACKEND="${2:?missing backend}"
            shift 2
            ;;
        --target)
            TARGET_TRIPLE="${2:?missing target triple}"
            shift 2
            ;;
        --profile)
            PROFILE="${2:?missing cargo profile}"
            shift 2
            ;;
        --out)
            OUT_DIR="${2:?missing output directory}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

case "$BACKEND" in
    cpu|metal|cuda|cuda-blackwell|rocm|hip|vulkan) ;;
    *)
        echo "unsupported native SDK backend: $BACKEND" >&2
        exit 1
        ;;
esac

case "$PROFILE" in
    release|debug) ;;
    *)
        echo "unsupported native SDK cargo profile: $PROFILE" >&2
        exit 1
        ;;
esac

host_os() {
    case "$(uname -s)" in
        Darwin) printf 'darwin\n' ;;
        Linux) printf 'linux\n' ;;
        MINGW*|MSYS*|CYGWIN*) printf 'windows\n' ;;
        *) uname -s | tr '[:upper:]' '[:lower:]' ;;
    esac
}

host_arch() {
    case "$(uname -m)" in
        arm64|aarch64) printf 'aarch64\n' ;;
        x86_64|amd64) printf 'x86_64\n' ;;
        *) uname -m ;;
    esac
}

default_target_triple() {
    case "$(host_os)/$(host_arch)" in
        darwin/aarch64) printf 'aarch64-apple-darwin\n' ;;
        darwin/x86_64) printf 'x86_64-apple-darwin\n' ;;
        linux/x86_64) printf 'x86_64-unknown-linux-gnu\n' ;;
        linux/aarch64) printf 'aarch64-unknown-linux-gnu\n' ;;
        windows/x86_64) printf 'x86_64-pc-windows-msvc\n' ;;
        *) printf '\n' ;;
    esac
}

target_platform() {
    case "$1" in
        aarch64-apple-darwin) printf 'darwin-aarch64\n' ;;
        x86_64-apple-darwin) printf 'darwin-x86_64\n' ;;
        x86_64-unknown-linux-gnu) printf 'linux-x86_64\n' ;;
        aarch64-unknown-linux-gnu) printf 'linux-aarch64\n' ;;
        aarch64-linux-android) printf 'android-arm64-v8a\n' ;;
        armv7-linux-androideabi) printf 'android-armeabi-v7a\n' ;;
        x86_64-linux-android) printf 'android-x86_64\n' ;;
        x86_64-pc-windows-msvc) printf 'windows-x86_64\n' ;;
        *) printf '%s\n' "$1" | tr '_' '-' ;;
    esac
}

library_extension() {
    case "$1" in
        *apple-darwin) printf 'dylib\n' ;;
        *linux*|*android*) printf 'so\n' ;;
        *windows*) printf 'dll\n' ;;
        *) echo "cannot infer dynamic library extension for target: $1" >&2; exit 1 ;;
    esac
}

library_basename() {
    case "$1" in
        dll) printf 'meshllm_ffi.dll\n' ;;
        *) printf 'libmeshllm_ffi.%s\n' "$1" ;;
    esac
}

uniffi_library_basename() {
    case "$1" in
        dll) printf 'uniffi_mesh_ffi.dll\n' ;;
        *) printf 'libuniffi_mesh_ffi.%s\n' "$1" ;;
    esac
}

sanitize_component() {
    printf '%s' "$1" | tr ';, /:' '_____' | tr -cd 'A-Za-z0-9_.-'
}

backend_flavor() {
    case "$BACKEND" in
        cuda) printf 'cuda\n' ;;
        cuda-blackwell) printf 'cuda-blackwell\n' ;;
        rocm|hip) printf 'rocm\n' ;;
        *)
            printf '%s\n' "$BACKEND"
            ;;
    esac
}

build_backend() {
    case "$BACKEND" in
        cuda-blackwell) printf 'cuda\n' ;;
        hip) printf 'rocm\n' ;;
        *) printf '%s\n' "$BACKEND" ;;
    esac
}

target_runtime_os() {
    case "$1" in
        *apple-darwin) printf 'macos\n' ;;
        *linux*|*android*) printf 'linux\n' ;;
        *windows*) printf 'windows\n' ;;
        *) echo "cannot infer runtime os for target: $1" >&2; exit 1 ;;
    esac
}

target_runtime_arch() {
    case "$1" in
        aarch64-*) printf 'aarch64\n' ;;
        x86_64-*) printf 'x86_64\n' ;;
        armv7-*) printf 'arm\n' ;;
        *) echo "cannot infer runtime arch for target: $1" >&2; exit 1 ;;
    esac
}

sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        echo "shasum or sha256sum is required" >&2
        exit 1
    fi
}

workspace_version() {
    python3 - "$REPO_ROOT/Cargo.toml" <<'PY'
import re
import sys

in_workspace_package = False
for line in open(sys.argv[1], encoding="utf-8"):
    stripped = line.strip()
    if stripped == "[workspace.package]":
        in_workspace_package = True
        continue
    if stripped.startswith("[") and stripped != "[workspace.package]":
        in_workspace_package = False
    if in_workspace_package:
        match = re.match(r'version\s*=\s*"([^"]+)"', stripped)
        if match:
            print(match.group(1))
            raise SystemExit(0)
raise SystemExit("workspace package version not found")
PY
}

if [[ -z "$TARGET_TRIPLE" ]]; then
    TARGET_TRIPLE="$(default_target_triple)"
fi
if [[ -z "$TARGET_TRIPLE" ]]; then
    echo "could not infer target triple; pass --target" >&2
    exit 1
fi

if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
    LLAMA_STAGE_BUILD_DIR="$(LLAMA_STAGE_BACKEND="$(build_backend)" "$SCRIPT_DIR/build-llama.sh" --print-build-dir)"
fi

if [[ "$BUILD" == "1" ]]; then
    "$SCRIPT_DIR/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"
    LLAMA_STAGE_BACKEND="$(build_backend)" \
        LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
        LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
        "$SCRIPT_DIR/build-llama.sh"

    cargo_args=(build -p mesh-llm-ffi --no-default-features --features host,embedded-runtime)
    if [[ "$PROFILE" == "release" ]]; then
        cargo_args+=(--release)
    fi
    if [[ "$TARGET_TRIPLE" != "$(default_target_triple)" ]]; then
        cargo_args+=(--target "$TARGET_TRIPLE")
    fi
    LLAMA_STAGE_BACKEND="$(build_backend)" \
        LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
        cargo "${cargo_args[@]}"
fi

lib_ext="$(library_extension "$TARGET_TRIPLE")"
lib_name="$(library_basename "$lib_ext")"
uniffi_lib_name="$(uniffi_library_basename "$lib_ext")"
platform="$(target_platform "$TARGET_TRIPLE")"
runtime_os="$(target_runtime_os "$TARGET_TRIPLE")"
runtime_arch="$(target_runtime_arch "$TARGET_TRIPLE")"
flavor="$(backend_flavor)"
artifact_id="meshllm-native-${platform}-${flavor}"

target_dir="$REPO_ROOT/target/$PROFILE"
if [[ "$TARGET_TRIPLE" != "$(default_target_triple)" ]]; then
    target_dir="$REPO_ROOT/target/$TARGET_TRIPLE/$PROFILE"
fi

lib_path="$target_dir/$lib_name"
if [[ ! -f "$lib_path" && -f "$target_dir/deps/$lib_name" ]]; then
    lib_path="$target_dir/deps/$lib_name"
fi
if [[ ! -f "$lib_path" && -f "$REPO_ROOT/target/$TARGET_TRIPLE/$PROFILE/$lib_name" ]]; then
    lib_path="$REPO_ROOT/target/$TARGET_TRIPLE/$PROFILE/$lib_name"
fi
if [[ ! -f "$lib_path" && -f "$REPO_ROOT/target/$TARGET_TRIPLE/$PROFILE/deps/$lib_name" ]]; then
    lib_path="$REPO_ROOT/target/$TARGET_TRIPLE/$PROFILE/deps/$lib_name"
fi

if [[ ! -f "$lib_path" ]]; then
    echo "native SDK library not found: $lib_name" >&2
    echo "looked in: $target_dir and $target_dir/deps" >&2
    echo "rerun with --build or build mesh-llm-ffi first" >&2
    exit 1
fi

stage_dir="$OUT_DIR/$artifact_id"
rm -rf "$stage_dir"
mkdir -p "$stage_dir/lib"

cp "$lib_path" "$stage_dir/lib/$lib_name"
cp "$lib_path" "$stage_dir/lib/$uniffi_lib_name"

patched_sha=""
upstream_sha=""
patch_digest=""
if [[ -f "$LLAMA_WORKDIR/.mesh-llm-patched-sha" ]]; then
    patched_sha="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-patched-sha")"
fi
if [[ -f "$LLAMA_WORKDIR/.mesh-llm-upstream-sha" ]]; then
    upstream_sha="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-upstream-sha")"
fi
if [[ -f "$LLAMA_WORKDIR/.mesh-llm-patch-digest" ]]; then
    patch_digest="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-patch-digest")"
fi

lib_sha="$(sha256_file "$stage_dir/lib/$lib_name")"
sdk_version="$(workspace_version)"

python3 - "$stage_dir/manifest.json" <<PY
import json
import os
import sys

manifest = {
    "schema_version": 1,
    "artifact_id": "$artifact_id",
    "native_runtime_id": "$artifact_id",
    "sdk_version": "$sdk_version",
    "mesh_version": "$sdk_version",
    "target_triple": "$TARGET_TRIPLE",
    "platform": "$platform",
    "os": "$runtime_os",
    "arch": "$runtime_arch",
    "backend": "$BACKEND",
    "flavor": "$flavor",
    "cargo_profile": "$PROFILE",
    "library": "lib/$lib_name",
    "library_paths": ["lib/$lib_name"],
    "uniffi_library": "lib/$uniffi_lib_name",
    "library_sha256": "$lib_sha",
    "url": None,
    "sha256": None,
    "signature": None,
    "requirements": [],
    "llama_upstream_sha": "$upstream_sha" or None,
    "llama_patched_sha": "$patched_sha" or None,
    "llama_patch_digest": "$patch_digest" or None,
    "llama_build_dir": os.path.abspath("$LLAMA_STAGE_BUILD_DIR"),
    "cuda_architectures": os.environ.get("LLAMA_STAGE_CUDA_ARCHITECTURES") or os.environ.get("SKIPPY_CUDA_ARCHITECTURES"),
    "amdgpu_targets": os.environ.get("LLAMA_STAGE_AMDGPU_TARGETS") or os.environ.get("SKIPPY_AMDGPU_TARGETS"),
    "features": [
        "mesh-inference",
        "model-management",
        "local-serving",
        "chat",
        "responses",
    ],
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\\n")
PY

cat > "$stage_dir/README.md" <<EOF
# $artifact_id

This artifact contains the MeshLLM native SDK runtime for:

- target: \`$TARGET_TRIPLE\`
- backend: \`$BACKEND\`
- flavor: \`$flavor\`

SDK loaders should read \`manifest.json\`, verify \`library_sha256\`, then load
\`$lib_name\`.

Kotlin/JVM UniFFI consumers can load \`$uniffi_lib_name\`, which is an alias of
the same library kept for UniFFI's generated JNA lookup name.
EOF

mkdir -p "$OUT_DIR"
archive="$OUT_DIR/$artifact_id.tar.gz"
tar -C "$OUT_DIR" -czf "$archive" "$artifact_id"
archive_sha="$(sha256_file "$archive")"
printf '%s  %s\n' "$archive_sha" "$(basename "$archive")" > "$archive.sha256"

echo "packaged native SDK runtime:"
echo "  artifact: $artifact_id"
echo "  library:  $stage_dir/lib/$lib_name"
echo "  archive:  $archive"
