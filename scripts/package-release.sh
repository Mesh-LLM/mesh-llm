#!/usr/bin/env bash

set -euo pipefail

RELEASE_FLAVOR="${MESH_RELEASE_FLAVOR:-}"
_STAGING_DIR=""
trap 'rm -rf "$_STAGING_DIR"' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RELEASE_BIN_DIR="${MESH_LLM_RELEASE_BIN_DIR:-$REPO_ROOT/target/release}"
NATIVE_RUNTIME_ROOT="${MESH_LLM_NATIVE_RUNTIME_ROOT:-$REPO_ROOT/dist/native-runtimes}"
PROVIDER_RUNTIME_ROOT="${MESH_LLM_PROVIDER_RUNTIME_ROOT:-}"
ALLOW_UNNOTARIZED_PROVIDER_RUNTIME="${MESH_RELEASE_ALLOW_UNNOTARIZED_PROVIDER_RUNTIME:-0}"
ATTESTATION_SIGNING_KEY_FILE="${MESH_RELEASE_ATTESTATION_SIGNING_KEY_FILE:-}"
ATTESTATION_PUBLIC_KEY_FILE="${MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE:-}"
PRECOMPOSED_PRODUCT_DIR="${MESH_LLM_PRECOMPOSED_PRODUCT_DIR:-}"
ATTESTATION_PREVERIFIED="${MESH_RELEASE_ATTESTATION_PREVERIFIED:-0}"
PROVIDER_RUNTIME_BUNDLES=()

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

write_checksum_sidecar() {
    local file="$1"
    local digest
    if command -v shasum >/dev/null 2>&1; then
        digest="$(shasum -a 256 "$file" | awk '{print tolower($1)}')"
    elif command -v sha256sum >/dev/null 2>&1; then
        digest="$(sha256sum "$file" | awk '{print tolower($1)}')"
    else
        echo "shasum or sha256sum is required to write checksum sidecars" >&2
        exit 1
    fi
    printf '%s  %s\n' "$digest" "$(basename "$file")" > "$file.sha256"
}

select_native_runtime_dir() {
    local platform
    local flavor
    local cuda_major
    local cuda_version
    platform="$(normalized_release_platform)"
    flavor="$(effective_release_flavor)"
    cuda_version="${MESH_CUDA_VERSION:-}"
    cuda_major="${MESH_LLM_CUDA_TOOLKIT_MAJOR:-${cuda_version%%.*}}"

    "$(python_bin)" "$SCRIPT_DIR/select-native-runtime.py" \
        --root "$NATIVE_RUNTIME_ROOT" \
        --os "${platform%%/*}" \
        --arch "${platform#*/}" \
        --backend "$flavor" \
        --cuda-major "$cuda_major"
}

write_product_manifest() {
    local bundle_dir="$1"
    local host_path="$2"
    local runtime_path="$3"
    local version="$4"
    local flavor="$5"

    local provider_args=()
    local provider_runtime
    for provider_runtime in ${PROVIDER_RUNTIME_BUNDLES[@]+"${PROVIDER_RUNTIME_BUNDLES[@]}"}; do
        provider_args+=(--provider-runtime "$provider_runtime")
    done

    "$(python_bin)" "$SCRIPT_DIR/compose-product-bundle.py" \
        --bundle "$bundle_dir" \
        --host "$host_path" \
        --runtime "$runtime_path" \
        --version "$version" \
        --backend "$flavor" \
        ${provider_args[@]+"${provider_args[@]}"}
}

provider_runtime_coordinates() {
    local manifest="$1"
    "$(python_bin)" - "$manifest" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    document = json.load(handle)
runtime = document["runtime"]
values = (runtime["provider_kind"], runtime["id"])
if any(not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value) for value in values):
    raise SystemExit("provider kind and runtime id must be safe package path components")
print("\t".join(values))
PY
}

verify_provider_runtime_contract() {
    local bundle="$1"
    just --justfile "$REPO_ROOT/Justfile" with-lld \
        cargo run --quiet -p mesh-llm-provider-runtime --example inspect -- "$bundle"
}

verify_apple_provider_release_policy() {
    local bundle="$1"
    local manifest="$bundle/provider-runtime.json"
    local entrypoint
    local declared_notarized
    local declared_team
    local declared_identifier
    local codesign_details
    local actual_team
    local actual_identifier

    read -r entrypoint declared_notarized declared_team declared_identifier < <(
        "$(python_bin)" - "$manifest" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    runtime = json.load(handle)["runtime"]
signature = runtime.get("signature") or {}
print(
    runtime["entrypoint"],
    str(signature.get("notarized") is True).lower(),
    signature.get("team_identifier") or "-",
    signature.get("signing_identifier") or "-",
)
PY
    )
    local executable="$bundle/$entrypoint"
    codesign --verify --strict --verbose=2 "$executable"
    codesign_details="$(codesign -dv --verbose=4 "$executable" 2>&1)"
    actual_team="$(printf '%s\n' "$codesign_details" | awk -F= '/^TeamIdentifier=/ {print $2; exit}')"
    actual_identifier="$(printf '%s\n' "$codesign_details" | awk -F= '/^Identifier=/ {print $2; exit}')"

    if [[ "$ALLOW_UNNOTARIZED_PROVIDER_RUNTIME" == "1" ]]; then
        echo "Apple provider release policy: local-development exception (not release eligible)"
        return 0
    fi
    if [[ "$ALLOW_UNNOTARIZED_PROVIDER_RUNTIME" != "0" ]]; then
        echo "MESH_RELEASE_ALLOW_UNNOTARIZED_PROVIDER_RUNTIME must be 0 or 1" >&2
        exit 1
    fi
    if [[ "$declared_notarized" != "true" || "$declared_team" == "-" ]]; then
        echo "release products require a Developer ID signed and notarized Apple provider runtime" >&2
        exit 1
    fi
    if [[ "$declared_team" != "$actual_team" || "$declared_identifier" != "$actual_identifier" ]]; then
        echo "Apple provider manifest signing identity does not match the executable" >&2
        exit 1
    fi
    spctl --assess --type execute --verbose=2 "$executable"
}

verify_provider_runtime_release_policy() {
    local bundle="$1"
    local provider_kind="$2"
    case "$provider_kind" in
        apple)
            if [[ "$(normalized_release_platform)" != "macos/aarch64" ]]; then
                echo "Apple provider runtimes may only be composed into macOS arm64 products" >&2
                exit 1
            fi
            verify_apple_provider_release_policy "$bundle"
            ;;
    esac
}

copy_provider_runtimes() {
    local source_root="$1"
    local bundle_dir="$2"
    local manifest
    local source_bundle
    local coordinates
    local provider_kind
    local runtime_id
    local destination
    local found=0

    [[ -d "$source_root" ]] || {
        echo "Provider runtime root does not exist: $source_root" >&2
        exit 1
    }
    while IFS= read -r manifest; do
        found=$((found + 1))
        source_bundle="$(dirname "$manifest")"
        verify_provider_runtime_contract "$source_bundle"
        coordinates="$(provider_runtime_coordinates "$manifest")"
        IFS=$'\t' read -r provider_kind runtime_id <<<"$coordinates"
        verify_provider_runtime_release_policy "$source_bundle" "$provider_kind"
        destination="$bundle_dir/provider-runtimes/$provider_kind/$runtime_id"
        [[ ! -e "$destination" ]] || {
            echo "Duplicate provider runtime destination: $destination" >&2
            exit 1
        }
        mkdir -p "$(dirname "$destination")"
        cp -R "$source_bundle" "$destination"
        PROVIDER_RUNTIME_BUNDLES+=("$destination")
    done < <(find "$source_root" -maxdepth 2 -type f -name provider-runtime.json -print | sort)
    if [[ "$found" -eq 0 ]]; then
        echo "Provider runtime root contains no provider-runtime.json: $source_root" >&2
        exit 1
    fi
}

discover_bundled_provider_runtimes() {
    local bundle_dir="$1"
    local manifest
    PROVIDER_RUNTIME_BUNDLES=()
    [[ -d "$bundle_dir/provider-runtimes" ]] || return 0
    while IFS= read -r manifest; do
        PROVIDER_RUNTIME_BUNDLES+=("$(dirname "$manifest")")
    done < <(
        find "$bundle_dir/provider-runtimes" \
            -mindepth 3 -maxdepth 3 -type f -name provider-runtime.json -print | sort
    )
}

validate_attestation_env() {
    if [[ "$ATTESTATION_PREVERIFIED" == "1" ]]; then
        if [[ "${MESH_RELEASE_HOST_PRESTAMPED:-0}" != "1" || -z "$PRECOMPOSED_PRODUCT_DIR" ]]; then
            echo "MESH_RELEASE_ATTESTATION_PREVERIFIED=1 requires a pre-stamped precomposed product" >&2
            exit 1
        fi
    elif [[ "$ATTESTATION_PREVERIFIED" != "0" ]]; then
        echo "MESH_RELEASE_ATTESTATION_PREVERIFIED must be 0 or 1" >&2
        exit 1
    fi
    if [[ "${MESH_RELEASE_HOST_PRESTAMPED:-0}" == "1" ]]; then
        if [[ -z "$ATTESTATION_PUBLIC_KEY_FILE" ]]; then
            echo "MESH_RELEASE_HOST_PRESTAMPED=1 requires MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE" >&2
            exit 1
        fi
        return 0
    fi
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

    if [[ "${MESH_RELEASE_HOST_PRESTAMPED:-0}" == "1" ]]; then
        if [[ -z "$ATTESTATION_PUBLIC_KEY_FILE" || ! -s "$ATTESTATION_PUBLIC_KEY_FILE" ]]; then
            echo "MESH_RELEASE_HOST_PRESTAMPED=1 requires a non-empty MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE" >&2
            exit 1
        fi
        if [[ "$ATTESTATION_PREVERIFIED" == "1" ]]; then
            echo "Release attestation: verified by immutable product composer"
            return 0
        fi
        inspect_json="$(
            cd "$REPO_ROOT"
            cargo run -q -p xtask -- release-attestation inspect \
                --binary "$binary_path" \
                --public-key-file "$ATTESTATION_PUBLIC_KEY_FILE" \
                --json
        )"
        inspect_status="$(printf '%s' "$inspect_json" | "$(python_bin)" -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
        if [[ "$inspect_status" != "valid" ]]; then
            echo "pre-stamped release host is not valid: $binary_path ($inspect_status)" >&2
            exit 1
        fi
        echo "Release attestation: verified pre-stamped host"
        return 0
    fi

    if [[ -z "$ATTESTATION_SIGNING_KEY_FILE" ]]; then
        echo "Release attestation: missing (packaged binary left unstamped)"
        return 0
    fi

    if [[ ! -s "$ATTESTATION_SIGNING_KEY_FILE" ]]; then
        echo "Release attestation: signing key file is empty ($ATTESTATION_SIGNING_KEY_FILE); leaving binary unstamped" >&2
        return 0
    fi

    if [[ ! -s "$ATTESTATION_PUBLIC_KEY_FILE" ]]; then
        echo "Release attestation: public key file is empty ($ATTESTATION_PUBLIC_KEY_FILE); leaving binary unstamped" >&2
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

run_mesh_binary_version_check() {
    local binary="$1"
    "$binary" --version
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

copy_and_verify_precomposed_product() {
    local source_dir="$1"
    local bundle_dir="$2"
    local version="$3"
    local flavor="$4"
    local runtime_dir=""
    local runtime_count=0
    local verification_report="$_STAGING_DIR/host-imports.verify.json"
    local bundle_binary=""
    bundle_binary="$bundle_dir/$(bundle_bin_name mesh-llm)"

    if [[ ! -d "$source_dir" ]]; then
        echo "Precomposed product directory does not exist: $source_dir" >&2
        exit 1
    fi
    test -s "$source_dir/product-manifest.json"
    mkdir -p "$bundle_dir"
    cp -a "$source_dir/." "$bundle_dir/"

    stamp_bundle_binary "$bundle_binary"
    verify_mesh_binary_version "$bundle_binary" "$version"
    "$(python_bin)" "$SCRIPT_DIR/verify-host-dependencies.py" \
        "$bundle_binary" \
        --report "$verification_report"

    while IFS= read -r manifest; do
        runtime_dir="$(dirname "$manifest")"
        runtime_count=$((runtime_count + 1))
    done < <(
        find "$bundle_dir/native-runtimes" \
            -mindepth 2 \
            -maxdepth 2 \
            -type f \
            -name manifest.json \
            -print
    )
    if [[ "$runtime_count" -ne 1 ]]; then
        echo "Precomposed product must contain exactly one native runtime; found $runtime_count" >&2
        exit 1
    fi
    "$SCRIPT_DIR/verify-native-runtime-package.sh" "$runtime_dir"
    discover_bundled_provider_runtimes "$bundle_dir"
    local provider_runtime
    local provider_kind
    for provider_runtime in ${PROVIDER_RUNTIME_BUNDLES[@]+"${PROVIDER_RUNTIME_BUNDLES[@]}"}; do
        verify_provider_runtime_contract "$provider_runtime"
        provider_kind="$(provider_runtime_coordinates "$provider_runtime/provider-runtime.json" | cut -f1)"
        verify_provider_runtime_release_policy "$provider_runtime" "$provider_kind"
    done
    local provider_args=()
    for provider_runtime in ${PROVIDER_RUNTIME_BUNDLES[@]+"${PROVIDER_RUNTIME_BUNDLES[@]}"}; do
        provider_args+=(--provider-runtime "$provider_runtime")
    done
    "$(python_bin)" "$SCRIPT_DIR/compose-product-bundle.py" \
        --bundle "$bundle_dir" \
        --host "$bundle_binary" \
        --runtime "$runtime_dir" \
        --version "$version" \
        --backend "$flavor" \
        ${provider_args[@]+"${provider_args[@]}"} \
        --check
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
    local runtime_dir
    local bundled_runtime
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
    if [[ -n "$PRECOMPOSED_PRODUCT_DIR" ]]; then
        copy_and_verify_precomposed_product \
            "$PRECOMPOSED_PRODUCT_DIR" \
            "$bundle_dir" \
            "$version" \
            "$(effective_release_flavor)"
    else
        mkdir -p "$bundle_dir"

        bundle_binary="$bundle_dir/$(bundle_bin_name mesh-llm)"
        cp "$RELEASE_BIN_DIR/mesh-llm${BIN_EXT}" "$bundle_binary"

        if [[ "$os_name" == "Darwin" && -f "$bundle_binary" ]]; then
            install_name_tool -add_rpath @executable_path/ "$bundle_binary" 2>/dev/null || true
        fi

        stamp_bundle_binary "$bundle_binary"
        verify_mesh_binary_version "$bundle_binary" "$version"
        "$(python_bin)" "$SCRIPT_DIR/verify-host-dependencies.py" \
            "$bundle_binary" \
            --report "$bundle_dir/host-imports.json"

        runtime_dir="$(select_native_runtime_dir)"
        bundled_runtime="$bundle_dir/native-runtimes/$(basename "$runtime_dir")"
        mkdir -p "$(dirname "$bundled_runtime")"
        cp -R "$runtime_dir" "$bundled_runtime"
        if [[ -n "$PROVIDER_RUNTIME_ROOT" ]]; then
            copy_provider_runtimes "$PROVIDER_RUNTIME_ROOT" "$bundle_dir"
        fi
        write_product_manifest \
            "$bundle_dir" \
            "$bundle_binary" \
            "$bundled_runtime" \
            "$version" \
            "$(effective_release_flavor)"
    fi

    create_archive "$bundle_dir" "$output_dir/$versioned_asset" "$ARCHIVE_EXT"
    write_checksum_sidecar "$output_dir/$versioned_asset"
    create_archive "$bundle_dir" "$output_dir/$STABLE_ASSET" "$ARCHIVE_EXT"
    write_checksum_sidecar "$output_dir/$STABLE_ASSET"

    echo "Created release archives:"
    find "$output_dir" -maxdepth 1 -type f -print | sort
}

if [[ "${BASH_SOURCE[0]-}" == "$0" || ( -z "${BASH_SOURCE[0]-}" && "$0" == "bash" ) ]]; then
    main "$@"
fi
