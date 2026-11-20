#!/usr/bin/env bash
# Package the Mesh host into the per-platform release archives.
#
# The Skippy native runtime is not built here. MESH_LLM_NATIVE_RUNTIME_DIR must
# name one already-downloaded runtime directory (with its schema 2
# manifest.json); it is bundled under native-runtimes/ and described in
# product-manifest.json, which install.sh and self-update require.
#
# Signing: with MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE and
# MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE naming non-empty files, the host is
# stamped and the stamp verified. Setting only one is an error; empty files
# (unsecreted runs) leave the host unstamped.

set -euo pipefail

RELEASE_FLAVOR="${MESH_RELEASE_FLAVOR:-}"
_STAGING_DIR=""
trap 'rm -rf "$_STAGING_DIR"' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RELEASE_BIN_DIR="${MESH_LLM_RELEASE_BIN_DIR:-$REPO_ROOT/target/release}"
NATIVE_RUNTIME_DIR="${MESH_LLM_NATIVE_RUNTIME_DIR:-}"
ATTESTATION_SIGNING_KEY_FILE="${MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE:-}"
ATTESTATION_PUBLIC_KEY_FILE="${MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE:-}"

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
        cuda)
            # When MESH_CUDA_VERSION is set (CI matrix), include major version.
            if [[ -n "${MESH_CUDA_VERSION:-}" ]]; then
                local major="${MESH_CUDA_VERSION%%.*}"
                printf -- '-%s-%s\n' "$1" "$major"
            else
                printf -- '-%s\n' "$1"
            fi
            ;;
        *)
            printf -- '-%s\n' "$1"
            ;;
    esac
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
            printf 'cpu cuda rocm vulkan\n'
            ;;
        linux/aarch64)
            printf 'cpu cuda\n'
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

    HOST_TARGET_TRIPLE="$TARGET_TRIPLE"
    STABLE_ASSET="$(printf 'mesh-llm-%s%s.%s\n' "$TARGET_TRIPLE" "$(flavor_suffix "$effective_flavor")" "$ARCHIVE_EXT")"
    TARGET_TRIPLE="${TARGET_TRIPLE}$(flavor_suffix "$effective_flavor")"

    return 0
}

versioned_asset_name() {
    local version="$1"

    resolve_release_target
    printf 'mesh-llm-%s-%s.%s\n' "$version" "$TARGET_TRIPLE" "$ARCHIVE_EXT"
}

release_attestation() {
    (cd "$REPO_ROOT" && cargo run --locked --quiet -p mesh-llm-release-attestation -- "$@")
}

stamp_bundle_binary() {
    local binary_path="$1"
    if [[ -z "$ATTESTATION_SIGNING_KEY_FILE" && -z "$ATTESTATION_PUBLIC_KEY_FILE" ]]; then
        echo "Release attestation: missing (packaged binary left unstamped)"
        return 0
    fi
    if [[ -z "$ATTESTATION_SIGNING_KEY_FILE" || -z "$ATTESTATION_PUBLIC_KEY_FILE" ]]; then
        echo "MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE and MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE must be set together" >&2
        exit 1
    fi
    # Unsecreted (fork/PR) runs pass empty key files: package unstamped, as before.
    if [[ ! -s "$ATTESTATION_SIGNING_KEY_FILE" || ! -s "$ATTESTATION_PUBLIC_KEY_FILE" ]]; then
        echo "Release attestation: key file empty or missing; packaged binary left unstamped" >&2
        return 0
    fi
    # A signed stamp must name the commit it was built from. MESH_RELEASE_COMMIT
    # overrides the checkout's HEAD; an unknown commit is refused, not invented.
    local stamp_args=(--binary "$binary_path" --signing-key-file "$ATTESTATION_SIGNING_KEY_FILE" --require-source-commit --target-triple "$HOST_TARGET_TRIPLE")
    [[ -z "${MESH_RELEASE_COMMIT:-}" ]] || stamp_args+=(--commit "$MESH_RELEASE_COMMIT")
    release_attestation stamp "${stamp_args[@]}" >/dev/null
    local status
    status="$(release_attestation inspect --binary "$binary_path" --public-key-file "$ATTESTATION_PUBLIC_KEY_FILE" --json | jq -r '.status')"
    if [[ "$status" != "valid" ]]; then
        echo "release attestation for $binary_path is '$status', expected valid" >&2
        exit 1
    fi
    echo "Release attestation: stamped and verified"
}

# Same digest as mesh-llm-system's product_runtime_tree_sha256: over files sorted
# by relative path, u64 BE path length, path bytes, file SHA-256.
tree_sha256() {
    python3 - "$1" <<'PY'
import hashlib, os, struct, sys
root = os.path.realpath(sys.argv[1])
files = []
for directory, _, names in os.walk(root):
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            files.append((os.path.relpath(path, root).replace(os.sep, "/"), path))
tree = hashlib.sha256()
for relative, path in sorted(files):
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).digest()
    encoded = relative.encode()
    tree.update(struct.pack(">Q", len(encoded)) + encoded + digest)
print(tree.hexdigest())
PY
}

file_sha256() {
    { shasum -a 256 "$1" 2>/dev/null || sha256sum "$1"; } | awk '{print $1}'
}

# Write product-manifest.json (contract mesh-llm-product-v2) for the bundled
# host and runtime, refusing a runtime whose backend or ABI the host can't use.
write_product_manifest() {
    local bundle_dir="$1" host="$2" runtime="$3" version="${4#v}" backend="$5"
    local contract runtime_manifest="$runtime/manifest.json" expected_kind
    case "$backend" in
        cuda|cuda-blackwell) expected_kind=cuda ;;
        rocm|hip) expected_kind=rocm ;;
        *) expected_kind="$backend" ;;
    esac
    contract="$("$host" --log-format json --print-build-contract | tail -n 1)"
    jq -e --arg v "$version" '.product_version == $v' <<<"$contract" >/dev/null ||
        { echo "host build contract product version does not match $version: $contract" >&2; exit 1; }
    jq -e '.schema_version == 2' "$runtime_manifest" >/dev/null ||
        { echo "native runtime manifest requires schema_version 2: $runtime_manifest" >&2; exit 1; }
    jq -e --arg k "$expected_kind" '.runtime.backend.kind == $k' "$runtime_manifest" >/dev/null ||
        { echo "native runtime backend is $(jq -c .runtime.backend.kind "$runtime_manifest"), expected $expected_kind for $backend" >&2; exit 1; }
    jq -e --argjson c "$contract" '.runtime.skippy_abi == $c.skippy_abi' "$runtime_manifest" >/dev/null ||
        { echo "native runtime ABI $(jq -c .runtime.skippy_abi "$runtime_manifest") does not match host-required $(jq -c .skippy_abi <<<"$contract")" >&2; exit 1; }
    jq -n \
        --arg version "$version" --arg backend "$backend" --argjson contract "$contract" \
        --slurpfile runtime "$runtime_manifest" \
        --arg host_path "${host#"$bundle_dir/"}" --arg host_sha "$(file_sha256 "$host")" \
        --arg runtime_path "${runtime#"$bundle_dir/"}" --arg runtime_sha "$(tree_sha256 "$runtime")" \
        --arg manifest_sha "$(file_sha256 "$runtime_manifest")" \
        '{schema_version: 2, contract: "mesh-llm-product-v2", mesh_version: $version, backend: $backend,
          host: {path: $host_path, sha256: $host_sha, required_skippy_abi: $contract.skippy_abi},
          runtime: {id: $runtime[0].runtime.id, release_version: $runtime[0].runtime.release_version,
                    skippy_abi: $runtime[0].runtime.skippy_abi, path: $runtime_path,
                    sha256: $runtime_sha, manifest_sha256: $manifest_sha}}' \
        >"$bundle_dir/product-manifest.json"
}

create_archive() {
    local source_dir="$1" archive_path="$2"
    COPYFILE_DISABLE=1 tar -czf "$archive_path" -C "$(dirname "$source_dir")" "$(basename "$source_dir")"
    (cd "$(dirname "$archive_path")" && { shasum -a 256 "$(basename "$archive_path")" 2>/dev/null || sha256sum "$(basename "$archive_path")"; }) >"$archive_path.sha256"
}

usage() {
    echo "usage: scripts/package-release.sh <version> [output_dir]" >&2
}

verify_mesh_binary_version() {
    local binary="$1"
    local expected="${2#v}"
    local output actual
    [[ -x "$binary" ]] || { echo "Release binary is not executable: $binary" >&2; exit 1; }
    output="$("$binary" --version)"
    actual="$(awk '{print $NF}' <<<"$output")"
    if [[ "$actual" != "$expected" ]]; then
        echo "Release binary version mismatch: expected $expected, got ${actual:-<empty>} ($binary)" >&2
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
    local bundle_dir bundle_binary versioned_asset

    if ! resolve_release_target; then
        release_target_error_message >&2
        exit 1
    fi
    versioned_asset="$(versioned_asset_name "$version")"

    mkdir -p "$output_dir"
    output_dir="$(cd "$output_dir" && pwd)"
    _STAGING_DIR="$(mktemp -d)"
    bundle_dir="$_STAGING_DIR/mesh-bundle"
    mkdir -p "$bundle_dir"

    bundle_binary="$bundle_dir/mesh-llm"
    cp "$RELEASE_BIN_DIR/mesh-llm${BIN_EXT}" "$bundle_binary"
    if [[ "$(release_os_name)" == "Darwin" ]]; then
        install_name_tool -add_rpath @executable_path/ "$bundle_binary" 2>/dev/null || true
    fi
    stamp_bundle_binary "$bundle_binary"
    verify_mesh_binary_version "$bundle_binary" "$version"

    [[ -f "$NATIVE_RUNTIME_DIR/manifest.json" ]] ||
        { echo "MESH_LLM_NATIVE_RUNTIME_DIR must name a native runtime directory with manifest.json (got '${NATIVE_RUNTIME_DIR}')" >&2; exit 1; }
    local bundled_runtime
    bundled_runtime="$bundle_dir/native-runtimes/$(basename "$NATIVE_RUNTIME_DIR")"
    mkdir -p "$bundle_dir/native-runtimes"
    cp -R "$NATIVE_RUNTIME_DIR" "$bundled_runtime"
    write_product_manifest "$bundle_dir" "$bundle_binary" "$bundled_runtime" "$version" "$(effective_release_flavor)"

    create_archive "$bundle_dir" "$output_dir/$versioned_asset"
    create_archive "$bundle_dir" "$output_dir/$STABLE_ASSET"

    echo "Created release archives:"
    find "$output_dir" -maxdepth 1 -type f -print | sort
}

if [[ "${BASH_SOURCE[0]-}" == "$0" || ( -z "${BASH_SOURCE[0]-}" && "$0" == "bash" ) ]]; then
    main "$@"
fi
