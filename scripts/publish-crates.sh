#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: scripts/publish-crates.sh [--dry-run] [--allow-dirty] [--sleep-seconds N]

Publishes the crates.io package chain in dependency order. Use --dry-run for
local and CI validation without uploading packages. --allow-dirty is accepted
only with --dry-run so local pre-commit validation can include uncommitted
manifest changes; real publishing always requires Cargo's clean-tree check.

Environment:
  CRATES_IO_PUBLISH_MAX_ATTEMPTS        Real-publish retry attempts for crates.io 429s (default: 6)
  CRATES_IO_PUBLISH_RETRY_BASE_SECONDS Fallback retry base when crates.io gives no timestamp (default: 60)
  CRATES_IO_PUBLISH_RETRY_MAX_SECONDS  Fallback retry cap when crates.io gives no timestamp (default: 900)
USAGE
}

log() {
    echo "publish-crates: $*"
}

warn() {
    echo "publish-crates: $*" >&2
}

require_positive_int() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer" >&2
        exit 1
    fi
}

require_nonnegative_int() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "${name} must be a non-negative integer" >&2
        exit 1
    fi
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

max_attempts="${CRATES_IO_PUBLISH_MAX_ATTEMPTS:-6}"
retry_base_seconds="${CRATES_IO_PUBLISH_RETRY_BASE_SECONDS:-60}"
retry_max_seconds="${CRATES_IO_PUBLISH_RETRY_MAX_SECONDS:-900}"

require_nonnegative_int CRATES_IO_PUBLISH_SETTLE_SECONDS "$sleep_seconds"
require_positive_int CRATES_IO_PUBLISH_MAX_ATTEMPTS "$max_attempts"
require_positive_int CRATES_IO_PUBLISH_RETRY_BASE_SECONDS "$retry_base_seconds"
require_positive_int CRATES_IO_PUBLISH_RETRY_MAX_SECONDS "$retry_max_seconds"

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

registry_version_status() {
    local crate="$1"
    local status
    if ! command -v curl >/dev/null 2>&1; then
        echo "unknown"
        return 0
    fi
    status="$(
        curl \
            --silent \
            --show-error \
            --output /dev/null \
            --write-out '%{http_code}' \
            "https://crates.io/api/v1/crates/${crate}/${workspace_version}" \
            2>/dev/null || true
    )"
    case "$status" in
        200)
            echo "published"
            ;;
        404)
            echo "missing"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

crate_version_published() {
    local crate="$1"
    [[ "$(registry_version_status "$crate")" == "published" ]]
}

publish_error_is_429() {
    local output="$1"
    [[ "$output" == *"status 429 Too Many Requests"* || "$output" == *"published too many new crates"* ]]
}

publish_error_is_already_uploaded() {
    local output="$1"
    [[ "$output" == *"already uploaded"* || "$output" == *"already exists on crates.io index"* ]]
}

print_publish_output() {
    local output="$1"
    if [[ -z "$output" ]]; then
        return 0
    fi
    if [[ -n "${CARGO_REGISTRY_TOKEN:-}" ]]; then
        output="${output//${CARGO_REGISTRY_TOKEN}/<redacted>}"
    fi
    printf '%s\n' "$output"
}

retry_after_epoch() {
    local output="$1"
    local retry_after
    retry_after="$(
        printf '%s\n' "$output" \
            | sed -nE 's/.*Please try again after ([^"]+)$/\1/p' \
            | head -n 1 \
            || true
    )"
    retry_after="${retry_after%.}"
    retry_after="${retry_after%\"}"
    if [[ -z "$retry_after" ]]; then
        return 1
    fi
    if date -u -d "$retry_after" +%s 2>/dev/null; then
        return 0
    fi
    date -u -j -f "%a, %d %b %Y %H:%M:%S %Z" "$retry_after" +%s 2>/dev/null
}

retry_delay_seconds() {
    local output="$1"
    local attempt="$2"
    local target_epoch now_epoch delay
    if target_epoch="$(retry_after_epoch "$output")" && now_epoch="$(date -u +%s 2>/dev/null)"; then
        delay=$((target_epoch - now_epoch + 5))
        if [[ "$delay" -lt 1 ]]; then
            delay=1
        fi
        echo "$delay"
        return 0
    fi

    delay="$retry_base_seconds"
    for ((step = 1; step < attempt; step++)); do
        delay=$((delay * 2))
        if [[ "$delay" -ge "$retry_max_seconds" ]]; then
            delay="$retry_max_seconds"
            break
        fi
    done
    echo "$delay"
}

last_publish_output=""

run_cargo_publish_once() {
    local crate="$1"
    local output status
    local args=(publish --locked -p "$crate")
    if [[ "$dry_run" -eq 1 ]]; then
        args+=(--dry-run)
    fi
    if [[ "$allow_dirty" -eq 1 ]]; then
        args+=(--allow-dirty)
    fi

    echo "cargo ${args[*]}"
    if output="$(cargo "${args[@]}" 2>&1)"; then
        last_publish_output="$output"
        print_publish_output "$output"
        return 0
    else
        status=$?
    fi

    last_publish_output="$output"
    print_publish_output "$output" >&2
    return "$status"
}

publish_crate_with_retry() {
    local crate="$1"
    local index="$2"
    local total="$3"
    local attempt delay

    attempt=1
    while [[ "$attempt" -le "$max_attempts" ]]; do
        if [[ "$dry_run" -eq 1 ]]; then
            log "[${index}/${total}] ${crate}@${workspace_version} dry-run"
        elif [[ "$attempt" -eq 1 ]]; then
            log "[${index}/${total}] ${crate}@${workspace_version} publish"
        else
            log "[${index}/${total}] ${crate}@${workspace_version} publish retry ${attempt}/${max_attempts}"
        fi

        if run_cargo_publish_once "$crate"; then
            return 0
        fi

        if [[ "$dry_run" -eq 0 ]] && publish_error_is_already_uploaded "$last_publish_output"; then
            log "[${index}/${total}] ${crate}@${workspace_version} already published according to cargo; continuing"
            return 0
        fi

        if [[ "$dry_run" -eq 0 ]] && publish_error_is_429 "$last_publish_output"; then
            warn "crates.io rate limit hit for ${crate}@${workspace_version} on attempt ${attempt}/${max_attempts}"
            if [[ "$attempt" -ge "$max_attempts" ]]; then
                warn "retry limit exceeded for ${crate}@${workspace_version} after ${max_attempts} attempts"
                return 101
            fi
            delay="$(retry_delay_seconds "$last_publish_output" "$attempt")"
            warn "retrying ${crate}@${workspace_version} after ${delay}s"
            sleep "$delay"
            attempt=$((attempt + 1))
            continue
        fi

        return 101
    done

    warn "retry limit exceeded for ${crate}@${workspace_version} after ${max_attempts} attempts"
    return 101
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
        mesh-llm-hardware-profile)
            printf '%s\n' \
                mesh-llm-native-runtime
            ;;
        mesh-llm-runtime-install)
            printf '%s\n' \
                mesh-llm-build-info \
                mesh-llm-hardware-profile \
                mesh-llm-native-runtime \
                skippy-ffi
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
        mesh-llm-console-server)
            printf '%s\n' \
                mesh-llm-ui
            ;;
        mesh-llm-cli)
            printf '%s\n' \
                mesh-llm-build-info \
                mesh-llm-events
            ;;
        mesh-llm-tui)
            printf '%s\n' \
                mesh-llm-events
            ;;
        mesh-llm-plugin-manager)
            printf '%s\n' \
                mesh-llm-skills
            ;;
        mesh-llm-system)
            printf '%s\n' \
                mesh-llm-build-info \
                mesh-llm-gpu-bench \
                skippy-runtime
            ;;
        mesh-llm-config)
            printf '%s\n' \
                mesh-llm-types \
                skippy-protocol
            ;;
        mesh-mixture-of-agents)
            printf '%s\n' \
                mesh-llm-guardrails
            ;;
        model-resolver)
            printf '%s\n' \
                model-artifact \
                model-ref
            ;;
        model-package)
            printf '%s\n' \
                model-hf \
                model-ref
            ;;
        openai-frontend)
            printf '%s\n' \
                mesh-llm-guardrails
            ;;
        skippy-cache)
            printf '%s\n' \
                skippy-protocol
            ;;
        skippy-runtime)
            printf '%s\n' \
                skippy-ffi
            ;;
        skippy-server)
            printf '%s\n' \
                openai-frontend \
                skippy-cache \
                skippy-metrics \
                skippy-protocol \
                skippy-runtime
            ;;
        mesh-llm-host-runtime)
            printf '%s\n' \
                mesh-llm-api-server \
                mesh-llm-build-info \
                mesh-llm-client \
                mesh-llm-config \
                mesh-llm-events \
                mesh-llm-guardrails \
                mesh-llm-identity \
                mesh-llm-native-runtime \
                mesh-llm-node \
                mesh-llm-plugin \
                mesh-llm-plugin-manager \
                mesh-llm-protocol \
                mesh-llm-routing \
                mesh-llm-runtime-install \
                mesh-llm-system \
                mesh-llm-types \
                mesh-llm-ui \
                mesh-mixture-of-agents \
                model-artifact \
                model-hf \
                model-package \
                model-ref \
                model-resolver \
                openai-frontend \
                skippy-coordinator \
                skippy-protocol \
                skippy-runtime \
                skippy-server \
                skippy-topology
            ;;
        mesh-llm-sdk)
            printf '%s\n' \
                mesh-llm-api-client \
                mesh-llm-api-server \
                mesh-llm-console-server \
                mesh-llm-embedded-runtime \
                mesh-llm-runtime-install
            ;;
        mesh-llm-embedded-runtime)
            printf '%s\n' \
                mesh-llm-host-runtime
            ;;
        mesh-llm-node)
            printf '%s\n' \
                mesh-llm-types \
                model-artifact \
                model-hf \
                model-ref
            ;;
        mesh-llm-api-server)
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
    mesh-llm-identity
    mesh-llm-protocol
    mesh-llm-routing
    mesh-llm-types
    mesh-llm-guardrails
    mesh-llm-plugin
    mesh-llm-skills
    mesh-llm-gpu-bench
    skippy-ffi
    skippy-protocol
    skippy-coordinator
    skippy-topology
    skippy-metrics
    skippy-cache
    model-ref
    model-artifact
    model-resolver
    mesh-llm-client
    mesh-llm-api-client
    mesh-llm-events
    mesh-llm-build-info
    mesh-llm-config
    mesh-llm-ui
    mesh-llm-console-server
    mesh-llm-tui
    mesh-llm-cli
    model-hf
    model-package
    mesh-llm-node
    mesh-llm-api-server
    mesh-llm-native-runtime
    mesh-llm-hardware-profile
    skippy-runtime
    openai-frontend
    skippy-server
    mesh-llm-plugin-manager
    mesh-mixture-of-agents
    mesh-llm-system
    mesh-llm-runtime-install
    mesh-llm-host-runtime
    mesh-llm-embedded-runtime
    mesh-llm-sdk
)

for index in "${!publish_crates[@]}"; do
    crate="${publish_crates[$index]}"
    if [[ "$dry_run" -eq 1 ]] && should_skip_initial_dry_run "$crate"; then
        continue
    fi
    publish_crate_with_retry "$crate" "$((index + 1))" "${#publish_crates[@]}"

    if [[ "$index" -lt "$((${#publish_crates[@]} - 1))" && "$sleep_seconds" -gt 0 ]]; then
        log "waiting ${sleep_seconds}s for crates.io index propagation"
        sleep "$sleep_seconds"
    fi
done
