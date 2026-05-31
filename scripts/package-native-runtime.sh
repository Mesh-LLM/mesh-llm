#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD=0
OUT_DIR="$REPO_ROOT/dist/native-runtimes"
BACKEND="${LLAMA_STAGE_BACKEND:-${SKIPPY_LLAMA_BACKEND:-cpu}}"
TARGET_TRIPLE="${MESH_NATIVE_RUNTIME_TARGET:-}"
LLAMA_WORKDIR="${LLAMA_WORKDIR:-$REPO_ROOT/.deps/llama.cpp}"
LLAMA_BUILD_ROOT="${MESH_LLM_LLAMA_BUILD_ROOT:-$REPO_ROOT/.deps/llama-build}"

usage() {
    cat >&2 <<'EOF'
Usage: scripts/package-native-runtime.sh [options]

Package a MeshLLM native runtime artifact containing the patched llama/Skippy
shared libraries selected by `mesh-llm runtime install`.

Options:
  --build             Build patched llama.cpp shared libraries before packaging.
  --backend NAME      cpu, metal, cuda, rocm, hip, vulkan, or cuda-blackwell.
  --target TRIPLE     Runtime target triple. Defaults to the host target.
  --out DIR           Output directory. Defaults to dist/native-runtimes.
  -h, --help          Show this help.

Environment:
  LLAMA_STAGE_CUDA_ARCHITECTURES / SKIPPY_CUDA_ARCHITECTURES
  LLAMA_STAGE_AMDGPU_TARGETS / SKIPPY_AMDGPU_TARGETS
  LLAMA_STAGE_BUILD_DIR
  MESH_NATIVE_RUNTIME_TARGET
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
        echo "unsupported native runtime backend: $BACKEND" >&2
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
        x86_64-pc-windows-msvc) printf 'windows-x86_64\n' ;;
        *) printf '%s\n' "$1" | tr '_' '-' ;;
    esac
}

target_runtime_os() {
    case "$1" in
        *apple-darwin) printf 'macos\n' ;;
        *linux*) printf 'linux\n' ;;
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

sanitize_component() {
    printf '%s' "$1" | tr ';, /:' '_____' | tr -cd 'A-Za-z0-9_.-'
}

backend_flavor() {
    case "$BACKEND" in
        cuda) printf 'cuda\n' ;;
        cuda-blackwell) printf 'cuda-blackwell\n' ;;
        rocm|hip) printf 'rocm\n' ;;
        *) printf '%s\n' "$BACKEND" ;;
    esac
}

build_backend() {
    case "$BACKEND" in
        cuda-blackwell) printf 'cuda\n' ;;
        hip) printf 'rocm\n' ;;
        *) printf '%s\n' "$BACKEND" ;;
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

skippy_abi_version() {
    python3 - "$REPO_ROOT/crates/skippy-ffi/src/lib.rs" <<'PY'
import re
import sys

values = {}
for line in open(sys.argv[1], encoding="utf-8"):
    match = re.match(r"pub const ABI_VERSION_(MAJOR|MINOR|PATCH): u32 = ([0-9]+);", line.strip())
    if match:
        values[match.group(1)] = match.group(2)
print("{}.{}.{}".format(values["MAJOR"], values["MINOR"], values["PATCH"]))
PY
}

library_pattern() {
    case "$TARGET_TRIPLE" in
        *apple-darwin) printf '*.dylib\n' ;;
        *windows*) printf '*.dll\n' ;;
        *) printf '*.so*\n' ;;
    esac
}

primary_library_name() {
    case "$TARGET_TRIPLE" in
        *apple-darwin) printf 'libllama.dylib\n' ;;
        *windows*) printf 'llama.dll\n' ;;
        *) printf 'libllama.so\n' ;;
    esac
}

collect_runtime_libraries() {
    local pattern primary
    pattern="$(library_pattern)"
    primary="$(primary_library_name)"
    find "$LLAMA_STAGE_BUILD_DIR" \( -type f -o -type l \) -name "$pattern" \
        ! -path '*/CMakeFiles/*' \
        | sort \
        | awk -v primary="$primary" '
            BEGIN { primary_path = "" }
            {
                name = $0
                sub(/^.*\//, "", name)
                if (name == primary) {
                    primary_path = $0
                } else {
                    print $0
                }
            }
            END {
                if (primary_path != "") print primary_path
            }
        '
}

if [[ -z "$TARGET_TRIPLE" ]]; then
    TARGET_TRIPLE="$(default_target_triple)"
fi
if [[ -z "$TARGET_TRIPLE" ]]; then
    echo "could not infer target triple; pass --target" >&2
    exit 1
fi

if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
    LLAMA_STAGE_BUILD_DIR="$(LLAMA_STAGE_LINK_MODE=dynamic LLAMA_STAGE_BACKEND="$(build_backend)" "$SCRIPT_DIR/build-llama.sh" --print-build-dir)"
fi

if [[ "$BUILD" == "1" ]]; then
    "$SCRIPT_DIR/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"
    LLAMA_STAGE_LINK_MODE=dynamic \
        LLAMA_STAGE_BACKEND="$(build_backend)" \
        LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
        LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
        "$SCRIPT_DIR/build-llama.sh"
fi

platform="$(target_platform "$TARGET_TRIPLE")"
runtime_os="$(target_runtime_os "$TARGET_TRIPLE")"
runtime_arch="$(target_runtime_arch "$TARGET_TRIPLE")"
flavor="$(backend_flavor)"
artifact_id="meshllm-native-runtime-${platform}-${flavor}"
stage_dir="$OUT_DIR/$artifact_id"

runtime_libraries=()
while IFS= read -r library; do
    runtime_libraries+=("$library")
done < <(collect_runtime_libraries)
if [[ "${#runtime_libraries[@]}" -eq 0 ]]; then
    echo "no native runtime libraries found under $LLAMA_STAGE_BUILD_DIR" >&2
    echo "rerun with --build or build patched llama.cpp with LLAMA_STAGE_LINK_MODE=dynamic" >&2
    exit 1
fi

primary_name="$(primary_library_name)"
last_index=$((${#runtime_libraries[@]} - 1))
if [[ "$(basename "${runtime_libraries[$last_index]}")" != "$primary_name" ]]; then
    echo "primary native runtime library not found: $primary_name" >&2
    exit 1
fi

rm -rf "$stage_dir"
mkdir -p "$stage_dir/lib"

library_paths=()
for library in "${runtime_libraries[@]}"; do
    name="$(basename "$library")"
    cp "$library" "$stage_dir/lib/$name"
    library_paths+=("lib/$name")
done

primary_library="lib/$primary_name"
primary_sha="$(sha256_file "$stage_dir/$primary_library")"
mesh_version="$(workspace_version)"
abi_version="$(skippy_abi_version)"

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

python3 - "$stage_dir/manifest.json" "$primary_library" "${library_paths[@]}" <<PY
import json
import os
import sys

manifest_path = sys.argv[1]
primary_library = sys.argv[2]
library_paths = sys.argv[3:]
manifest = {
    "schema_version": 1,
    "artifact_id": "$artifact_id",
    "native_runtime_id": "$artifact_id",
    "mesh_version": "$mesh_version",
    "target_triple": "$TARGET_TRIPLE",
    "platform": "$platform",
    "os": "$runtime_os",
    "arch": "$runtime_arch",
    "backend": "$BACKEND",
    "flavor": "$flavor",
    "library": primary_library,
    "library_paths": library_paths,
    "library_sha256": "$primary_sha",
    "url": None,
    "sha256": None,
    "signature": None,
    "requirements": [],
    "skippy_abi_version": "$abi_version",
    "llama_upstream_sha": "$upstream_sha" or None,
    "llama_patched_sha": "$patched_sha" or None,
    "llama_patch_digest": "$patch_digest" or None,
    "llama_build_dir": os.path.abspath("$LLAMA_STAGE_BUILD_DIR"),
    "cuda_architectures": os.environ.get("LLAMA_STAGE_CUDA_ARCHITECTURES") or os.environ.get("SKIPPY_CUDA_ARCHITECTURES"),
    "amdgpu_targets": os.environ.get("LLAMA_STAGE_AMDGPU_TARGETS") or os.environ.get("SKIPPY_AMDGPU_TARGETS"),
    "features": [
        "skippy-runtime",
        "llama.cpp",
        "native-runtime-loader",
    ],
}
with open(manifest_path, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\\n")
PY

cat > "$stage_dir/README.md" <<EOF
# $artifact_id

This artifact contains MeshLLM native runtime shared libraries for:

- target: \`$TARGET_TRIPLE\`
- backend: \`$BACKEND\`
- flavor: \`$flavor\`
- MeshLLM version: \`$mesh_version\`
- Skippy ABI: \`$abi_version\`

\`mesh-llm runtime install\` reads \`manifest.json\`, verifies the archive
checksum from \`native-runtimes.json\`, installs the artifact into the
versioned native runtime cache, and loads these libraries before Skippy starts.
EOF

mkdir -p "$OUT_DIR"
archive="$OUT_DIR/$artifact_id.tar.gz"
tar -C "$OUT_DIR" -czf "$archive" "$artifact_id"
archive_sha="$(sha256_file "$archive")"
printf '%s  %s\n' "$archive_sha" "$(basename "$archive")" > "$archive.sha256"

echo "packaged native runtime:"
echo "  artifact: $artifact_id"
echo "  primary:  $stage_dir/$primary_library"
echo "  archive:  $archive"
