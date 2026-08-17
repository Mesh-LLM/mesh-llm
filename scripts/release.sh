#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: scripts/release.sh <version|vversion> [--skip-gpu-bundles] [--canary]

Runs a synchronous release from main:
  1. verifies gh and the local git state
  2. dispatches .github/workflows/release.yml
  3. lets the workflow commit and push the canonical version bump
  4. watches the GitHub Actions run until it succeeds or fails
  5. verifies the release and origin/main version

The command requires the current branch to be main and up to date with
origin/main before it starts.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

require_command() {
    local name="$1"
    command -v "$name" >/dev/null 2>&1 || die "$name is required on PATH"
}

semver_from_arg() {
    local raw="$1"
    local version="${raw#v}"
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
        die "invalid version '$raw'; expected 0.72.1, v0.72.1, 0.72.1-rc1, etc."
    fi
    printf '%s\n' "$version"
}

ensure_version_format() {
    local version="$1"
    local context="$2"
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
        die "$context version '$version' is not a supported semantic version"
    fi
}

read_workspace_version() {
    perl -0ne 'print "$1\n" if /\[workspace\.package\][^[]*?\nversion\s*=\s*"([^"]+)"/s' Cargo.toml
}

ensure_target_version_not_older() {
    local current="$1"
    local target="$2"
    local compare_status

    if python3 - "$current" "$target" <<'PY'
import re
import sys

current, target = sys.argv[1:3]
pattern = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")


def parse(value):
    match = pattern.match(value)
    if not match:
        raise SystemExit(2)
    major, minor, patch, prerelease = match.groups()
    return (int(major), int(minor), int(patch), parse_prerelease(prerelease))


def parse_prerelease(value):
    if value is None:
        return None
    parts = []
    for part in value.split("."):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part))
    return parts


def compare_prerelease(left, right):
    if left is None and right is None:
        return 0
    if left is None:
        return 1
    if right is None:
        return -1
    for left_part, right_part in zip(left, right):
        if left_part == right_part:
            continue
        return -1 if left_part < right_part else 1
    if len(left) == len(right):
        return 0
    return -1 if len(left) < len(right) else 1


current_version = parse(current)
target_version = parse(target)

for left, right in zip(current_version[:3], target_version[:3]):
    if right > left:
        raise SystemExit(0)
    if right < left:
        raise SystemExit(1)

raise SystemExit(0 if compare_prerelease(target_version[3], current_version[3]) >= 0 else 1)
PY
    then
        compare_status=0
    else
        compare_status="$?"
    fi

    case "$compare_status" in
        0)
            ;;
        1)
            die "target version $target must not be older than current workspace version $current"
            ;;
        *)
            die "failed to compare current version $current and target version $target"
            ;;
    esac
}

ensure_clean_worktree() {
    if [[ -n "$(git status --porcelain)" ]]; then
        git status --short >&2
        die "working tree must be clean before release"
    fi
}

ensure_main_is_current() {
    local branch
    branch="$(git branch --show-current)"
    [[ "$branch" == "main" ]] || die "release must run from local main; current branch is '${branch:-detached}'"

    git fetch origin main

    local local_head
    local remote_head
    local_head="$(git rev-parse HEAD)"
    remote_head="$(git rev-parse origin/main)"
    [[ "$local_head" == "$remote_head" ]] || die "local main must match origin/main before release"
}

ensure_github_release_preflight() {
    local tag="$1"

    gh auth status >/dev/null 2>&1 || die "gh must be authenticated before release"
    gh workflow view release.yml >/dev/null 2>&1 || die "GitHub Release workflow release.yml is not available"

    if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
        die "local tag $tag already exists"
    fi
    if git ls-remote --exit-code --tags origin "refs/tags/$tag" >/dev/null 2>&1; then
        die "remote tag $tag already exists on origin"
    fi
    if gh release view "$tag" >/dev/null 2>&1; then
        die "GitHub release $tag already exists"
    fi
}

confirm_release() {
    local tag="$1"
    local version="$2"
    local current_version="$3"
    local skip_gpu_bundles="$4"
    local canary="$5"
    local answer

    cat <<EOF
This will:
  - verify the current branch is clean, local main, and equal to origin/main
  - dispatch .github/workflows/release.yml for $tag
  - for a non-canary release, let the workflow update source files from $current_version to $version
  - for a non-canary release, let the workflow commit and push "$tag: prepare release source" to origin/main
  - watch the GitHub Actions release run until it succeeds or fails
  - publish $tag with GitHub-generated release notes after all release checks pass

Options:
  - skip GPU bundles: $skip_gpu_bundles
  - canary: $canary

Continue? y/N
EOF
    read -r answer
    case "$answer" in
        y|Y|yes|YES)
            ;;
        *)
            die "release cancelled"
            ;;
    esac
}

dispatch_release_workflow() {
    local tag="$1"
    local skip_gpu_bundles="$2"
    local canary="$3"

    echo "Dispatching Release workflow for $tag..." >&2
    gh workflow run release.yml \
        --ref main \
        --raw-field "version=$tag" \
        --raw-field "skip_gpu_bundles=$skip_gpu_bundles" \
        --raw-field "canary=$canary"
}

workflow_run_id_from_url() {
    local run_url="$1"

    if [[ "$run_url" =~ /actions/runs/([0-9]+)([/?#]|$) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    fi
}

find_dispatched_release_run_id() {
    local release_sha="$1"
    local created_after="$2"
    local run_id

    echo "Waiting for GitHub to create the workflow run..." >&2
    for _ in {1..30}; do
        run_id="$(
            gh run list \
                --workflow release.yml \
                --branch main \
                --commit "$release_sha" \
                --event workflow_dispatch \
                --created ">=$created_after" \
                --json databaseId,headSha,createdAt \
                --jq 'sort_by(.createdAt) | reverse | .[0].databaseId // empty'
        )"
        if [[ -n "$run_id" ]]; then
            printf '%s\n' "$run_id"
            return
        fi
        sleep 2
    done

    die "timed out waiting for Release workflow run to appear"
}

wait_for_release() {
    local tag="$1"

    for _ in {1..60}; do
        if gh release view "$tag" >/dev/null 2>&1; then
            return
        fi
        sleep 5
    done

    die "release $tag was not visible after the workflow completed"
}

main() {
    if [[ $# -lt 1 ]]; then
        usage
        exit 1
    fi

    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage
        exit 0
    fi

    local raw_version="$1"
    shift

    local skip_gpu_bundles="false"
    local canary="false"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-gpu-bundles)
                skip_gpu_bundles="true"
                ;;
            --canary)
                canary="true"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                usage
                die "unknown argument: $1"
                ;;
        esac
        shift
    done

    require_command gh
    require_command git
    require_command perl
    require_command python3

    local version
    local tag
    version="$(semver_from_arg "$raw_version")"
    tag="v$version"

    ensure_clean_worktree
    ensure_main_is_current

    local current_version
    current_version="$(read_workspace_version)"
    [[ -n "$current_version" ]] || die "could not read workspace version from Cargo.toml"
    ensure_version_format "$current_version" "current workspace"
    ensure_target_version_not_older "$current_version" "$version"
    ensure_github_release_preflight "$tag"

    confirm_release "$tag" "$version" "$current_version" "$skip_gpu_bundles" "$canary"

    local dispatch_sha
    local dispatch_started_at
    local run_url
    local run_id
    dispatch_sha="$(git rev-parse origin/main)"
    dispatch_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    run_url="$(dispatch_release_workflow "$tag" "$skip_gpu_bundles" "$canary")"
    run_id="$(workflow_run_id_from_url "$run_url")"
    if [[ -z "$run_id" ]]; then
        run_id="$(find_dispatched_release_run_id "$dispatch_sha" "$dispatch_started_at")"
    fi

    echo "Watching Release workflow run $run_id..."
    gh run watch "$run_id" --compact --exit-status

    if [[ "$canary" == "true" ]]; then
        echo "Canary release workflow succeeded; no GitHub release was published."
        return
    fi

    wait_for_release "$tag"
    git fetch origin main

    local remote_version
    remote_version="$(git show origin/main:Cargo.toml | perl -0ne 'print "$1\n" if /\[workspace\.package\][^[]*?\nversion\s*=\s*"([^"]+)"/s')"
    [[ "$remote_version" == "$version" ]] || die "release succeeded, but origin/main reports version ${remote_version:-unknown} instead of $version"

    echo "Release complete: $tag; origin/main is version $remote_version"
    echo "Local main can now be fast-forwarded with: git pull --ff-only"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
