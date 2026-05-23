#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: scripts/publish-crates.sh [--dry-run] [--allow-dirty] [--sleep-seconds N]

Publishes the crates.io package chain in dependency order. Use --dry-run for
local and CI validation without uploading packages. --allow-dirty is accepted
only with --dry-run so local pre-commit validation can include uncommitted
manifest changes; real publishing always requires Cargo's clean-tree check.
USAGE
}

dry_run=0
allow_dirty=0
sleep_seconds=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        --allow-dirty)
            allow_dirty=1
            shift
            ;;
        --sleep-seconds)
            if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
                usage
                exit 1
            fi
            sleep_seconds="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if [[ "$allow_dirty" -eq 1 && "$dry_run" -eq 0 ]]; then
    echo "--allow-dirty is only supported together with --dry-run" >&2
    exit 1
fi

if [[ -z "$sleep_seconds" ]]; then
    if [[ "$dry_run" -eq 1 ]]; then
        sleep_seconds=0
    else
        sleep_seconds="${CRATES_IO_PUBLISH_SETTLE_SECONDS:-30}"
    fi
fi

if [[ "$dry_run" -eq 0 && -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
    echo "CARGO_REGISTRY_TOKEN is required for real crates.io publishing" >&2
    exit 1
fi

workspace_version="$(
    perl -ne '
        $in_workspace_package = 1 if /^\[workspace\.package\]/;
        $in_workspace_package = 0 if /^\[/ && !/^\[workspace\.package\]/;
        if ($in_workspace_package && /^\s*version\s*=\s*"([^"]+)"/) {
            print $1;
            exit;
        }
    ' Cargo.toml
)"

if [[ -z "$workspace_version" ]]; then
    echo "failed to read [workspace.package].version from Cargo.toml" >&2
    exit 1
fi

crate_version_published() {
    local crate="$1"
    local status
    if ! command -v curl >/dev/null 2>&1; then
        return 1
    fi
    status="$(
        curl \
            --fail \
            --silent \
            --show-error \
            --output /dev/null \
            --write-out '%{http_code}' \
            "https://crates.io/api/v1/crates/${crate}/${workspace_version}" \
            2>/dev/null || true
    )"
    [[ "$status" == "200" ]]
}

unpublished_registry_deps() {
    case "$1" in
        model-artifact)
            printf '%s\n' model-ref
            ;;
        model-hf)
            printf '%s\n' \
                model-artifact \
                model-ref
            ;;
        mesh-llm-client)
            printf '%s\n' \
                model-artifact \
                mesh-llm-identity \
                mesh-llm-protocol \
                mesh-llm-routing \
                mesh-llm-types
            ;;
        mesh-llm-api-client)
            printf '%s\n' \
                mesh-llm-client
            ;;
        mesh-llm-node)
            printf '%s\n' \
                mesh-llm-types \
                model-artifact \
                model-hf \
                model-ref
            ;;
        mesh-llm-api)
            printf '%s\n' \
                mesh-llm-api-client \
                mesh-llm-node
            ;;
    esac
}

should_skip_initial_dry_run() {
    local crate="$1"
    local dep
    while IFS= read -r dep; do
        [[ -n "$dep" ]] || continue
        if ! crate_version_published "$dep"; then
            echo "dry-run cannot verify ${crate} until ${dep}@${workspace_version} exists in crates.io"
            return 0
        fi
    done < <(unpublished_registry_deps "$crate")
    return 1
}

publish_crates=(
    model-ref
    mesh-llm-identity
    mesh-llm-protocol
    mesh-llm-routing
    mesh-llm-types
    model-artifact
    model-hf
    mesh-llm-client
    mesh-llm-api-client
    mesh-llm-node
    mesh-llm-api
)

for index in "${!publish_crates[@]}"; do
    crate="${publish_crates[$index]}"
    if [[ "$dry_run" -eq 1 ]] && should_skip_initial_dry_run "$crate"; then
        continue
    fi

    args=(publish --locked -p "$crate")
    if [[ "$dry_run" -eq 1 ]]; then
        args+=(--dry-run)
    fi
    if [[ "$allow_dirty" -eq 1 ]]; then
        args+=(--allow-dirty)
    fi

    echo "cargo ${args[*]}"
    cargo "${args[@]}"

    if [[ "$index" -lt "$((${#publish_crates[@]} - 1))" && "$sleep_seconds" -gt 0 ]]; then
        echo "waiting ${sleep_seconds}s for crates.io index propagation"
        sleep "$sleep_seconds"
    fi
done
