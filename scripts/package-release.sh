#!/usr/bin/env bash

set -euo pipefail

RELEASE_FLAVOR="${MESH_RELEASE_FLAVOR:-}"
_STAGING_DIR=""
trap 'rm -rf "$_STAGING_DIR"' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RELEASE_BIN_DIR="$REPO_ROOT/target/release"
ATTESTATION_SIGNING_KEY_FILE="${MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE:-}"
ATTESTATION_PUBLIC_KEY_FILE="${MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE:-}"

python_bin() {
    if command -v python3 >/dev/null 2>&1; then
        echo python3
    elif command -v python >/dev/null 2>&1; then
        echo python
    else
        echo "python3 or python is required for packaging" >&2
        exit 1
    fi
}

release_os_name() {
    if [[ -n "${MESH_RELEASE_OS:-}" ]]; then
        printf '%s\n' "$MESH_RELEASE_OS"
        return 0
    fi

    uname -s
}

release_arch_name() {
    if [[ -n "${MESH_RELEASE_ARCH:-}" ]]; then
        printf '%s\n' "$MESH_RELEASE_ARCH"
        return 0
    fi

    uname -m
}

canonical_release_arch() {
    case "$(release_arch_name)" in
        x86_64|amd64)
            printf 'x86_64\n'
            ;;
        arm64|aarch64)
            printf 'aarch64\n'
            ;;
        arm|armv6l|armv6hf|armv7l|armv7hf)
            printf 'arm\n'
            ;;
        *)
            printf '%s\n' "$(release_arch_name)"
            ;;
    esac
}

flavor_suffix() {
    case "$1" in
        ""|cpu|metal)
            printf '\n'
            ;;
        *)
            printf -- '-%s\n' "$1"
            ;;
    esac
}

binary_flavor_for_release_flavor() {
    printf '%s\n' "$1"
}

bundle_bin_name() {
    local name="$1"
    if [[ "$name" == "mesh-llm" ]]; then
        echo "$name"
        return
    fi

    local binary_flavor
    binary_flavor="$(binary_flavor_for_release_flavor "$RELEASE_FLAVOR")"
    if [[ -z "$binary_flavor" ]]; then
        case "$(release_os_name)" in
            Darwin) binary_flavor="metal" ;;
            Linux) binary_flavor="cpu" ;;
        esac
    fi

    if [[ -n "$binary_flavor" ]]; then
        echo "${name}-${binary_flavor}"
    else
        echo "$name"
    fi
}

create_archive() {
    local source_dir="$1"
    local archive_path="$2"
    local archive_kind="$3"
    local py
    py="$(python_bin)"

    rm -f "$archive_path"
    mkdir -p "$(dirname "$archive_path")"

    "$py" - "$source_dir" "$archive_path" "$archive_kind" <<'PY'
import os
import sys
import tarfile
import zipfile

source_dir, archive_path, archive_kind = sys.argv[1:4]
base = os.path.basename(os.path.normpath(source_dir))
root = os.path.dirname(os.path.normpath(source_dir))

if archive_kind == "tar.gz":
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(source_dir, arcname=base)
elif archive_kind == "zip":
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for current_root, dirs, files in os.walk(source_dir):
            dirs.sort()
            files.sort()
            rel_root = os.path.relpath(current_root, root)
            if rel_root != ".":
                zf.write(current_root, rel_root)
            for filename in files:
                path = os.path.join(current_root, filename)
                rel = os.path.relpath(path, root)
                zf.write(path, rel)
else:
    raise SystemExit(f"unsupported archive kind: {archive_kind}")
PY
}

validate_attestation_env() {
    if [[ -n "$ATTESTATION_SIGNING_KEY_FILE" && -z "$ATTESTATION_PUBLIC_KEY_FILE" ]]; then
        echo "MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE is required when MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE is set" >&2
        exit 1
    fi
    if [[ -z "$ATTESTATION_SIGNING_KEY_FILE" && -n "$ATTESTATION_PUBLIC_KEY_FILE" ]]; then
        echo "MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE is required when MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE is set" >&2
        exit 1
    fi
}

stamp_bundle_binary() {
    local binary_path="$1"
    local inspect_json
    local inspect_status
    local py

    if [[ -z "$ATTESTATION_SIGNING_KEY_FILE" ]]; then
        echo "Release attestation: missing (packaged binary left unstamped)"
        return 0
    fi

    py="$(python_bin)"

    inspect_json="$(
        cd "$REPO_ROOT"
        cargo run -q -p xtask -- release-attestation stamp \
            --binary "$binary_path" \
            --signing-key-file "$ATTESTATION_SIGNING_KEY_FILE" \
            >/dev/null
        cargo run -q -p xtask -- release-attestation inspect \
            --binary "$binary_path" \
            --public-key-file "$ATTESTATION_PUBLIC_KEY_FILE" \
            --json
    )"
    printf '%s\n' "$inspect_json"
    inspect_status="$(printf '%s' "$inspect_json" | "$py" -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
    if [[ "$inspect_status" != "valid" ]]; then
        echo "release-attestation inspect reported status '$inspect_status' for $binary_path" >&2
        exit 1
    fi
}

normalized_release_platform() {
    local os_name
    local arch_name

    os_name="$(release_os_name)"
    arch_name="$(canonical_release_arch)"

    case "$os_name/$arch_name" in
        Darwin/aarch64)
            printf 'macos/aarch64\n'
            ;;
        Linux/x86_64)
            printf 'linux/x86_64\n'
            ;;
        Linux/aarch64)
            printf 'linux/aarch64\n'
            ;;
        Linux/arm)
            printf 'linux/arm\n'
            ;;
        *)
            printf 'unsupported\n'
            ;;
    esac
}

effective_release_flavor() {
    case "$(normalized_release_platform)" in
        macos/aarch64)
            printf '%s\n' "${RELEASE_FLAVOR:-metal}"
            ;;
        linux/x86_64|linux/aarch64|linux/arm)
            printf '%s\n' "${RELEASE_FLAVOR:-cpu}"
            ;;
        *)
            printf '%s\n' "$RELEASE_FLAVOR"
            ;;
    esac
}

supported_release_flavors() {
    case "$(normalized_release_platform)" in
        macos/aarch64)
            printf 'metal\n'
            ;;
        linux/x86_64)
            printf 'cpu cuda cuda-blackwell rocm vulkan\n'
            ;;
        linux/aarch64)
            printf 'cpu\n'
            ;;
        *)
            printf '\n'
            ;;
    esac
}

release_target_flavor_supported() {
    local effective_flavor
    local supported_flavor

    effective_flavor="$(effective_release_flavor)"
    for supported_flavor in $(supported_release_flavors); do
        if [[ "$supported_flavor" == "$effective_flavor" ]]; then
            return 0
        fi
    done

    return 1
}

release_target_support() {
    case "$(normalized_release_platform)" in
        linux/arm)
            printf 'recognized-unsupported\n'
            ;;
        unsupported)
            printf 'unsupported\n'
            ;;
        *)
            if release_target_flavor_supported; then
                printf 'supported\n'
            else
                printf 'unsupported\n'
            fi
            ;;
    esac
}

release_target_error_message() {
    local os_name
    local arch_name
    local normalized
    local effective_flavor
    local support

    os_name="$(release_os_name)"
    arch_name="$(release_arch_name)"
    normalized="$(normalized_release_platform)"
    effective_flavor="$(effective_release_flavor)"
    support="$(release_target_support)"

    case "$support" in
        supported)
            printf 'release target is supported: %s\n' "$normalized"
            ;;
        recognized-unsupported)
            printf 'Recognized but unsupported release target: %s/%s (normalized: %s)\n' "$os_name" "$arch_name" "$normalized"
            ;;
        *)
            if [[ "$normalized" == "unsupported" ]]; then
                printf 'Unsupported OS/arch for packaging: %s/%s\n' "$os_name" "$arch_name"
            else
                printf 'Unsupported release target/flavor for packaging: %s/%s with flavor %s (normalized: %s)\n' "$os_name" "$arch_name" "$effective_flavor" "$normalized"
            fi
            ;;
    esac
}

resolve_release_target() {
    local normalized
    local support
    local effective_flavor

    normalized="$(normalized_release_platform)"
    support="$(release_target_support)"
    effective_flavor="$(effective_release_flavor)"
    BIN_EXT=""
    ARCHIVE_EXT="tar.gz"
    LEGACY_ASSET=""

    case "$support" in
        recognized-unsupported)
            return 2
            ;;
        unsupported)
            return 1
            ;;
    esac

    case "$normalized" in
        macos/aarch64)
            TARGET_TRIPLE="aarch64-apple-darwin"
            LEGACY_ASSET="mesh-bundle.tar.gz"
            ;;
        linux/x86_64)
            TARGET_TRIPLE="x86_64-unknown-linux-gnu"
            ;;
        linux/aarch64)
            TARGET_TRIPLE="aarch64-unknown-linux-gnu"
            ;;
        *)
            return 1
            ;;
    esac

    STABLE_ASSET="$(printf 'mesh-llm-%s%s.%s\n' "$TARGET_TRIPLE" "$(flavor_suffix "$effective_flavor")" "$ARCHIVE_EXT")"
    TARGET_TRIPLE="${TARGET_TRIPLE}$(flavor_suffix "$effective_flavor")"

    return 0
}

versioned_asset_name() {
    local version="$1"

    resolve_release_target
    printf 'mesh-llm-%s-%s.%s\n' "$version" "$TARGET_TRIPLE" "$ARCHIVE_EXT"
}

usage() {
    echo "usage: scripts/package-release.sh <version> [output_dir]" >&2
}

cuda_version_check_needs_stub() {
    [[ "$(release_os_name)" == "Linux" ]] || return 1

    case "$(effective_release_flavor)" in
        cuda|cuda-blackwell)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

cuda_stub_library_path() {
    local candidate
    local candidates=()

    if [[ -n "${CUDA_HOME:-}" ]]; then
        candidates+=("$CUDA_HOME/lib64/stubs/libcuda.so")
        candidates+=("$CUDA_HOME/targets/x86_64-linux/lib/stubs/libcuda.so")
    fi
    if [[ -n "${CUDA_PATH:-}" ]]; then
        candidates+=("$CUDA_PATH/lib64/stubs/libcuda.so")
        candidates+=("$CUDA_PATH/targets/x86_64-linux/lib/stubs/libcuda.so")
    fi

    candidates+=(
        "/usr/local/cuda/lib64/stubs/libcuda.so"
        "/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

run_mesh_binary_version_check() {
    local binary="$1"
    local cuda_stub
    local cuda_stub_dir

    if ! cuda_version_check_needs_stub; then
        "$binary" --version
        return
    fi

    if ! cuda_stub="$(cuda_stub_library_path)"; then
        echo "CUDA release binary needs libcuda.so.1 for version verification, but no CUDA stub libcuda.so was found." >&2
        exit 1
    fi

    cuda_stub_dir="$_STAGING_DIR/cuda-version-stubs"
    mkdir -p "$cuda_stub_dir"
    ln -sf "$cuda_stub" "$cuda_stub_dir/libcuda.so.1"

    LD_LIBRARY_PATH="$cuda_stub_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$binary" --version
}

verify_mesh_binary_version() {
    local binary="$1"
    local expected="$2"
    local output
    local actual

    expected="${expected#v}"
    if [[ ! -x "$binary" ]]; then
        echo "Release binary is not executable: $binary" >&2
        exit 1
    fi

    output="$(run_mesh_binary_version_check "$binary")"
    actual="$(awk '{print $NF}' <<<"$output")"
    if [[ "$actual" != "$expected" ]]; then
        echo "Release binary version mismatch: expected $expected, got ${actual:-<empty>}" >&2
        echo "Binary: $binary" >&2
        echo "Output: $output" >&2
        exit 1
    fi
}

main() {
    if [[ $# -lt 1 || -z "${1:-}" ]]; then
        usage
        exit 1
    fi

    local version="$1"
    local output_dir="${2:-dist}"
    local os_name
    local bundle_dir
    local bundle_binary
    local versioned_asset

    validate_attestation_env

    if ! resolve_release_target; then
        release_target_error_message >&2
        exit 1
    fi

    versioned_asset="$(versioned_asset_name "$version")"
    os_name="$(release_os_name)"

    mkdir -p "$output_dir"
    _STAGING_DIR="$(mktemp -d)"

    bundle_dir="$_STAGING_DIR/mesh-bundle"
    mkdir -p "$bundle_dir"

bundle_binary="$bundle_dir/$(bundle_bin_name mesh-llm)"
    cp "$RELEASE_BIN_DIR/mesh-llm${BIN_EXT}" "$bundle_binary"


    if [[ "$os_name" == "Darwin" && -f "$bundle_binary" ]]; then
        install_name_tool -add_rpath @executable_path/ "$bundle_binary" 2>/dev/null || true
    fi

    stamp_bundle_binary "$bundle_binary"

    create_archive "$bundle_dir" "$output_dir/$versioned_asset" "$ARCHIVE_EXT"
    create_archive "$bundle_dir" "$output_dir/$STABLE_ASSET" "$ARCHIVE_EXT"

    if [[ -n "$LEGACY_ASSET" ]]; then
        cp "$output_dir/$STABLE_ASSET" "$output_dir/$LEGACY_ASSET"
    fi

    echo "Created release archives:"
    find "$output_dir" -maxdepth 1 -type f -print | sort
}

if [[ "${BASH_SOURCE[0]-}" == "$0" || ( -z "${BASH_SOURCE[0]-}" && "$0" == "bash" ) ]]; then
    main "$@"
fi
