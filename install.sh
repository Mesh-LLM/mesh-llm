#!/usr/bin/env bash

set -euo pipefail

REPO="${MESH_LLM_INSTALL_REPO:-Mesh-LLM/mesh-llm}"
INSTALL_DIR="${MESH_LLM_INSTALL_DIR:-$HOME/.local/bin}"
INSTALL_FLAVOR="${MESH_LLM_INSTALL_FLAVOR:-}"
INSTALL_PRERELEASE="${MESH_LLM_INSTALL_PRERELEASE:-0}"
INSTALL_SERVICE="${MESH_LLM_INSTALL_SERVICE:-0}"
INSTALL_SERVICE_ARGS="${MESH_LLM_INSTALL_SERVICE_ARGS:-}"
INSTALL_SERVICE_START="${MESH_LLM_INSTALL_SERVICE_START:-1}"
RELEASE_URL_BASE="${MESH_LLM_INSTALL_URL_BASE:-}"
REQUIRE_CHECKSUM="${MESH_LLM_REQUIRE_CHECKSUM:-0}"
AUTO_SETUP=1
DOWNLOADED_ARCHIVE=""
DOWNLOADED_ASSET=""
SETUP_ARGS=()

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command not found: $1" >&2
        exit 1
    fi
}

warn() {
    echo "warning: $*" >&2
}

bool_is_true() {
    local value="${1:-}"
    value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    case "$value" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

path_contains_install_dir() {
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) return 0 ;;
        *) return 1 ;;
    esac
}

usage() {
    cat <<EOF
Usage: install.sh [--pre-release] [--install-dir DIR] [--no-setup]

Options:
  --pre-release              Install the latest published GitHub prerelease instead of the latest stable release.
  --install-dir DIR          Install the binary into DIR.
  --no-setup                 Do not run \
                             \
mesh-llm setup automatically after install.
  --service                  Legacy compatibility flag. Passes --service through to \
                             \
mesh-llm setup instead of installing services in shell.
  --no-start-service         Legacy compatibility flag. Ignored with a warning.
  --service-args VALUE       Legacy compatibility flag. Ignored with a warning.
  -h, --help                 Show this help text.

Environment overrides:
  MESH_LLM_INSTALL_DIR
  MESH_LLM_INSTALL_URL_BASE  Override release asset base URL for testing.
  MESH_LLM_INSTALL_PRERELEASE=1
  MESH_LLM_INSTALL_FLAVOR    Legacy compatibility variable. Ignored with a warning.
  MESH_LLM_INSTALL_SERVICE=1 Legacy compatibility variable. Passed through as --service.
  MESH_LLM_INSTALL_SERVICE_START=0 Legacy compatibility variable. Ignored with a warning.
  MESH_LLM_REQUIRE_CHECKSUM=1
EOF
}

add_setup_arg() {
    local arg="$1"
    local existing
    for existing in "${SETUP_ARGS[@]:-}"; do
        if [[ "$existing" == "$arg" ]]; then
            return 0
        fi
    done
    SETUP_ARGS+=("$arg")
}

parse_args() {
    while (($# > 0)); do
        case "$1" in
            --pre-release)
                INSTALL_PRERELEASE=1
                ;;
            --install-dir)
                shift
                if (($# == 0)); then
                    echo "error: --install-dir requires a directory" >&2
                    exit 1
                fi
                INSTALL_DIR="$1"
                ;;
            --no-setup)
                AUTO_SETUP=0
                ;;
            --service)
                warn "--service is deprecated in install.sh; forwarding it to \`mesh-llm setup --service\`."
                add_setup_arg "--service"
                ;;
            --no-start-service)
                warn "--no-start-service is deprecated in install.sh; service start policy now belongs to \`mesh-llm setup\`."
                ;;
            --service-args)
                shift
                if (($# == 0)); then
                    echo "error: --service-args requires a value" >&2
                    exit 1
                fi
                warn "--service-args is deprecated in install.sh and is ignored; configure startup behavior after \`mesh-llm setup\`."
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "error: unknown argument: $1" >&2
                echo >&2
                usage >&2
                exit 1
                ;;
        esac
        shift
    done
}

apply_legacy_env_compat() {
    if [[ -n "$INSTALL_FLAVOR" ]]; then
        warn "MESH_LLM_INSTALL_FLAVOR is deprecated in install.sh and is ignored; \`mesh-llm setup\` now owns runtime selection."
    fi
    if bool_is_true "$INSTALL_SERVICE"; then
        warn "MESH_LLM_INSTALL_SERVICE is deprecated in install.sh; forwarding it to \`mesh-llm setup --service\`."
        add_setup_arg "--service"
    fi
    if [[ -n "$INSTALL_SERVICE_ARGS" ]]; then
        warn "MESH_LLM_INSTALL_SERVICE_ARGS is deprecated in install.sh and is ignored; configure startup behavior after \`mesh-llm setup\`."
    fi
    if ! bool_is_true "$INSTALL_SERVICE_START"; then
        warn "MESH_LLM_INSTALL_SERVICE_START is deprecated in install.sh and is ignored; service start policy now belongs to \`mesh-llm setup\`."
    fi
}

platform_os() {
    if [[ -n "${MESH_LLM_TEST_UNAME_S:-}" ]]; then
        printf '%s\n' "$MESH_LLM_TEST_UNAME_S"
        return 0
    fi
    uname -s
}

platform_arch() {
    local os
    local arch
    os="$(platform_os)"
    if [[ -n "${MESH_LLM_TEST_UNAME_M:-}" ]]; then
        arch="$MESH_LLM_TEST_UNAME_M"
    else
        arch="$(uname -m)"
    fi
    case "$os/$arch" in
        Linux/amd64) printf 'x86_64\n' ;;
        Linux/arm64|Linux/aarch64) printf 'aarch64\n' ;;
        *) printf '%s\n' "$arch" ;;
    esac
}

platform_id() {
    printf '%s/%s\n' "$(platform_os)" "$(platform_arch)"
}

platform_asset_name() {
    case "$(platform_id)" in
        Darwin/arm64) printf 'mesh-llm-aarch64-apple-darwin.tar.gz\n' ;;
        Darwin/x86_64) printf 'mesh-llm-x86_64-apple-darwin.tar.gz\n' ;;
        Linux/aarch64) printf 'mesh-llm-aarch64-unknown-linux-gnu.tar.gz\n' ;;
        Linux/x86_64) printf 'mesh-llm-x86_64-unknown-linux-gnu.tar.gz\n' ;;
        *)
            echo "error: unsupported platform: $(platform_id)" >&2
            exit 1
            ;;
    esac
}

latest_prerelease_tag() {
    local api_url="https://api.github.com/repos/${REPO}/releases?per_page=20"
    local response
    local -a curl_args=(
        -fsSL
        -H 'Accept: application/vnd.github+json'
        -H 'X-GitHub-Api-Version: 2022-11-28'
    )

    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        curl_args+=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    elif [[ -n "${GH_TOKEN:-}" ]]; then
        curl_args+=(-H "Authorization: Bearer ${GH_TOKEN}")
    fi

    response="$(curl "${curl_args[@]}" "$api_url")"
    printf '%s' "$response" |
        tr -d '\n\r\t ' |
        sed 's/},{/}\
{/g' |
        awk '
            /"prerelease":true/ && !/"draft":true/ {
                if (match($0, /"tag_name":"[^"]+"/)) {
                    value = substr($0, RSTART, RLENGTH)
                    sub(/^"tag_name":"/, "", value)
                    sub(/"$/, "", value)
                    print value
                    exit
                }
            }
        '
}

release_url() {
    local asset="$1"
    if [[ -n "$RELEASE_URL_BASE" ]]; then
        printf '%s/%s\n' "${RELEASE_URL_BASE%/}" "$asset"
        return 0
    fi
    if bool_is_true "$INSTALL_PRERELEASE"; then
        local tag
        tag="$(latest_prerelease_tag)"
        printf 'https://github.com/%s/releases/download/%s/%s\n' "$REPO" "$tag" "$asset"
        return 0
    fi
    printf 'https://github.com/%s/releases/latest/download/%s\n' "$REPO" "$asset"
}

checksum_url() {
    printf '%s.sha256\n' "$1"
}

sha256_file() {
    local file="$1"
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file" | awk '{print tolower($1)}'
        return 0
    fi
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print tolower($1)}'
        return 0
    fi
    echo "error: shasum or sha256sum is required to verify release artifacts" >&2
    return 1
}

checksum_from_sidecar() {
    local sidecar="$1"
    local expected
    expected="$(awk 'match($0, /[A-Fa-f0-9]{64}/) { print substr($0, RSTART, RLENGTH); exit }' "$sidecar" | tr '[:upper:]' '[:lower:]')"
    if [[ -z "$expected" ]]; then
        echo "error: checksum sidecar did not contain a SHA-256 digest: $sidecar" >&2
        return 1
    fi
    printf '%s\n' "$expected"
}

download_checksum_sidecar() {
    local url="$1"
    local sidecar="$2"
    local status
    status="$(curl -sSL -w '%{http_code}' -o "$sidecar" "$url")" || {
        rm -f -- "$sidecar"
        echo "error: could not download checksum sidecar: $url" >&2
        return 2
    }

    case "$status" in
        2*) return 0 ;;
        000)
            if [[ -f "$sidecar" ]]; then
                return 0
            fi
            rm -f -- "$sidecar"
            return 1
            ;;
        404|410)
            rm -f -- "$sidecar"
            return 1
            ;;
        *)
            rm -f -- "$sidecar"
            echo "error: checksum sidecar download returned HTTP $status: $url" >&2
            return 2
            ;;
    esac
}

verify_downloaded_file() {
    local file="$1"
    local url="$2"
    local require_checksum="${3:-$REQUIRE_CHECKSUM}"
    local sidecar="$file.sha256"
    local sidecar_url
    local sidecar_status=0
    local expected
    local actual

    sidecar_url="$(checksum_url "$url")"
    download_checksum_sidecar "$sidecar_url" "$sidecar" || sidecar_status=$?
    if [[ "$sidecar_status" -ne 0 ]]; then
        case "$sidecar_status" in
            1)
                if bool_is_true "$require_checksum"; then
                    echo "error: checksum sidecar is required but missing: $sidecar_url" >&2
                    return 1
                fi
                echo "warning: checksum sidecar not found; continuing without archive verification: $sidecar_url" >&2
                return 0
                ;;
            *) return 1 ;;
        esac
    fi

    expected="$(checksum_from_sidecar "$sidecar")"
    actual="$(sha256_file "$file")"
    if [[ "$actual" != "$expected" ]]; then
        echo "error: checksum mismatch for $(basename "$file")" >&2
        echo "expected: $expected" >&2
        echo "actual:   $actual" >&2
        return 1
    fi
    echo "Verified checksum: $(basename "$file")"
}

download_release_archive() {
    local tmp_dir="$1"
    local preferred_asset="$2"
    local fallback_asset="mesh-bundle.tar.gz"
    local asset
    local url
    local archive

    for asset in "$preferred_asset" "$fallback_asset"; do
        url="$(release_url "$asset")"
        archive="$tmp_dir/$asset"
        if ! curl -fsSL "$url" -o "$archive"; then
            rm -f -- "$archive"
            continue
        fi
        verify_downloaded_file "$archive" "$url"
        DOWNLOADED_ASSET="$asset"
        DOWNLOADED_ARCHIVE="$archive"
        if [[ "$asset" == "$fallback_asset" ]]; then
            echo "Using runtime-enabled mesh bundle fallback: $fallback_asset"
        fi
        return 0
    done

    echo "error: could not download release archive for $preferred_asset or $fallback_asset" >&2
    return 1
}

stale_binary_names() {
    cat <<'EOF'
mesh-llm
rpc-server
llama-server
llama-moe-split
rpc-server-cpu
llama-server-cpu
rpc-server-cuda
llama-server-cuda
rpc-server-rocm
llama-server-rocm
rpc-server-vulkan
llama-server-vulkan
rpc-server-metal
llama-server-metal
EOF
}

remove_stale_binaries() {
    mkdir -p "$INSTALL_DIR"
    local name
    while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        rm -f "$INSTALL_DIR/$name"
    done < <(stale_binary_names)
}

install_bundle() {
    local bundle_dir="$1"
    remove_stale_binaries
    local file
    for file in "$bundle_dir"/*; do
        mv -f "$file" "$INSTALL_DIR/"
    done
}

shell_is_interactive() {
    if [[ -n "${MESH_LLM_TEST_INTERACTIVE:-}" ]]; then
        bool_is_true "$MESH_LLM_TEST_INTERACTIVE"
        return
    fi
    [[ -t 0 && -t 1 ]]
}

setup_command_string() {
    local -a command=("$INSTALL_DIR/mesh-llm" setup)
    local token
    local rendered=""
    command+=("${SETUP_ARGS[@]}")
    for token in "${command[@]}"; do
        printf -v rendered '%s%q ' "$rendered" "$token"
    done
    printf '%s\n' "${rendered% }"
}

run_or_print_setup() {
    local binary="$INSTALL_DIR/mesh-llm"
    local command
    command="$(setup_command_string)"

    echo
    if (( AUTO_SETUP == 1 )) && shell_is_interactive; then
        echo "Running post-install setup: $command"
        "$binary" setup "${SETUP_ARGS[@]}"
        return 0
    fi

    echo "Run this next: $command"
}

main() {
    parse_args "$@"
    apply_legacy_env_compat
    need_cmd curl
    need_cmd tar
    need_cmd mktemp

    local asset
    asset="$(platform_asset_name)"

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local tmp_dir_escaped
    printf -v tmp_dir_escaped '%q' "$tmp_dir"
    trap "rm -rf -- $tmp_dir_escaped" EXIT

    echo "Release channel: $(bool_is_true "$INSTALL_PRERELEASE" && echo prerelease || echo stable)"
    download_release_archive "$tmp_dir" "$asset"

    tar -xzf "$DOWNLOADED_ARCHIVE" -C "$tmp_dir"
    if [[ ! -d "$tmp_dir/mesh-bundle" ]]; then
        echo "error: release archive did not contain mesh-bundle/" >&2
        exit 1
    fi

    install_bundle "$tmp_dir/mesh-bundle"
    echo "Installed $DOWNLOADED_ASSET to $INSTALL_DIR"

    if ! path_contains_install_dir; then
        echo
        echo "$INSTALL_DIR is not on your PATH."
        echo "Add it with one of these commands:"
        echo
        echo "bash:"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc"
        echo "  source ~/.bashrc"
        echo
        echo "zsh:"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.zshrc"
        echo "  source ~/.zshrc"
    fi

    run_or_print_setup
}

if [[ "${BASH_SOURCE[0]-}" == "$0" || ( -z "${BASH_SOURCE[0]-}" && "$0" == "bash" ) ]]; then
    main "$@"
fi
