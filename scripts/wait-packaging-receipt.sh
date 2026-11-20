#!/usr/bin/env bash
# Waits for mesh-packaging to finish the run this release dispatched, then
# checks that run's receipt names this exact release. The contract is in
# mesh/docs/design/PACKAGING_HANDOFF.md.
#
#   scripts/wait-packaging-receipt.sh --tag v1.2.3 --sha <tag commit> \
#     --assets <sha256 of the Mesh asset list> \
#     --skippy-release <tag> --skippy-catalog <sha256 of its native-runtimes.json> \
#     --correlation <id> --since <UTC time the release run started>
#
# Needs gh (with GH_TOKEN), jq and unzip. Any failed request, any mismatch and
# any run that did not succeed ends the wait with an error.
# The jq filters below use jq variables, not shell ones.
# shellcheck disable=SC2016
set -euo pipefail

packaging="${MESH_PACKAGING_REPOSITORY:-Mesh-LLM/mesh-packaging}"
mesh="${GITHUB_REPOSITORY:-Mesh-LLM/mesh-llm}"
skippy="${SKIPPY_REPOSITORY:-Mesh-LLM/skippy}"
timeout="${PACKAGING_RECEIPT_TIMEOUT_SECONDS:-7200}"
poll="${PACKAGING_RECEIPT_POLL_SECONDS:-15}"
request_seconds="${PACKAGING_RECEIPT_REQUEST_SECONDS:-60}"
max_bytes=1048576

fail() {
    echo "wait-packaging-receipt: $*" >&2
    exit 1
}

tag="" sha="" assets="" skippy_release="" skippy_catalog="" correlation="" since=""
(($# == 14)) || { sed -n '6,9p' "$0" >&2; exit 2; }
while (($#)); do
    case "$1" in
        --tag) tag="$2" ;;
        --sha) sha="$2" ;;
        --assets) assets="$2" ;;
        --skippy-release) skippy_release="$2" ;;
        --skippy-catalog) skippy_catalog="$2" ;;
        --correlation) correlation="$2" ;;
        --since) since="$2" ;;
        *) fail "unknown argument $1" ;;
    esac
    shift 2
done
[[ -n "$tag" && -n "$skippy_release" ]] || fail "the Mesh tag and the Skippy release are required"
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || fail "--sha must be a 40-character commit"
[[ "$assets" =~ ^[0-9a-f]{64}$ && "$skippy_catalog" =~ ^[0-9a-f]{64}$ ]] || fail "--assets and --skippy-catalog must be SHA-256 digests"
[[ "$correlation" =~ ^[A-Za-z0-9._-]{1,128}$ ]] || fail "invalid correlation id"
[[ "$since" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] || fail "--since must be a UTC timestamp"
[[ "$timeout" =~ ^[0-9]+$ && "$poll" =~ ^[0-9]+$ && "$request_seconds" =~ ^[1-9][0-9]*$ ]] || fail "timeout and poll interval must be whole seconds"

title="Packaging · $correlation"
deadline=$((SECONDS + timeout))
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Each request gets at most a minute, or what is left of the wait, and at
# most 1 MiB of response. A request that stalls, fails or answers with more is
# an error; nothing truncated is ever parsed.
get() { # get <output file> <api path> [field=value ...]
    local out="$1" path="$2" args=() limit pid status=0 watchdog sleeper
    shift 2
    for field in "$@"; do args+=(-f "$field"); done
    limit=$((deadline - SECONDS))
    ((limit > 0)) || fail "timed out waiting for packaging readiness"
    ((limit <= request_seconds)) || limit="$request_seconds"
    (
        set -o pipefail
        gh api --method GET "$path" ${args[@]+"${args[@]}"} | head -c $((max_bytes + 1)) >"$out"
    ) &
    pid=$!
    # The watchdog ends the request at its limit and leaves a marker.
    rm -f "$work/stalled"
    (
        sleep "$limit"
        : >"$work/stalled"
        pkill -P "$pid" 2>/dev/null
        kill "$pid" 2>/dev/null
    ) >/dev/null 2>&1 &
    watchdog=$!
    wait "$pid" 2>/dev/null || status=$?
    # Stop the watchdog before its sleep, so it cannot leave a late marker.
    sleeper="$(pgrep -P "$watchdog" 2>/dev/null || true)"
    kill "$watchdog" 2>/dev/null || true
    wait "$watchdog" 2>/dev/null || true
    [[ -z "$sleeper" ]] || kill "$sleeper" 2>/dev/null || true
    [[ ! -e "$work/stalled" ]] || fail "GitHub request took longer than ${limit}s: $path"
    (($(wc -c <"$out") <= max_bytes)) || fail "GitHub response is larger than $max_bytes bytes: $path"
    ((status == 0)) || fail "GitHub request failed: $path"
}

# query <file> [jq args ...] <jq filter>. A filter that errors, or yields null
# or false, is a failure. Callers check its status themselves, because bash
# does not stop on errors inside $( ).
query() {
    local file="$1"
    shift
    jq -er "$@" "$file" || fail "unexpected response in ${file##*/}"
}

# Prints "<id> <attempt> <status> <conclusion>" for the one run with our
# title, or nothing if it has not appeared yet.
find_run() {
    local page rows found="" count
    for page in 1 2 3 4 5 6 7 8 9 10; do
        get "$work/runs.json" "repos/$packaging/actions/workflows/images-release.yml/runs" \
            event=repository_dispatch "created=>=$since" per_page=100 "page=$page"
        count="$(query "$work/runs.json" '.workflow_runs | arrays | length')" || exit 1
        rows="$(query "$work/runs.json" --arg title "$title" \
            '[.workflow_runs[] | select(.display_title == $title)
              | if (.id | type) == "number" and .id > 0 and (.run_attempt | type) == "number" and .run_attempt > 0
                   and (.status | type) == "string"
                then "\(.id) \(.run_attempt) \(.status) \(.conclusion // "none")"
                else error("bad run") end] | join("\n")')" || exit 1
        if [[ -n "$rows" ]]; then
            [[ -z "$found" && "$rows" != *$'\n'* ]] || fail "more than one packaging run has this correlation id"
            found="$rows"
        fi
        if ((count < 100)); then
            echo "$found"
            return
        fi
    done
    fail "too many packaging runs to search"
}

# Prints the ids of the run's receipt artifacts, one per line.
receipt_artifacts() {
    local run="$1" page ids="" more count
    for page in 1 2 3 4 5 6 7 8 9 10; do
        get "$work/artifacts.json" "repos/$packaging/actions/runs/$run/artifacts" per_page=100 "page=$page"
        count="$(query "$work/artifacts.json" '.artifacts | arrays | length')" || exit 1
        more="$(query "$work/artifacts.json" --argjson run "$run" --argjson max "$max_bytes" \
            '[.artifacts[] | select(.name == "packaging-readiness")
              | if .workflow_run.id == $run and .expired == false
                   and (.size_in_bytes | type) == "number" and .size_in_bytes > 0 and .size_in_bytes <= $max
                   and (.id | type) == "number" and .id > 0
                then "\(.id)" else error("bad artifact") end] | join("\n")')" || exit 1
        [[ -z "$more" ]] || ids+="$more"$'\n'
        if ((count < 100)); then
            printf '%s' "$ids"
            return
        fi
    done
    fail "too many packaging artifacts to search"
}

# The receipt of the current attempt must say everything succeeded for this
# exact Mesh release and this exact Skippy release.
check_receipt() {
    jq -e --arg correlation "$correlation" --argjson run "$1" --argjson attempt "$2" \
        --arg mesh "$mesh" --arg tag "$tag" --arg sha "$sha" --arg assets "$assets" \
        --arg skippy "$skippy" --arg skippy_release "$skippy_release" --arg skippy_catalog "$skippy_catalog" '
        .schema == "mesh-packaging-readiness-v2" and .status == "success"
        and .correlation_id == $correlation
        and .packaging_run_id == $run and .packaging_run_attempt == $attempt
        and .upstream == {repository: $mesh, ref: $tag, sha: $sha, assets_sha256: $assets}
        and .skippy == {repository: $skippy, release: $skippy_release, catalog_sha256: $skippy_catalog}
        and .requested == {publish_images: "true", publish_release_assets: "true", publish_npm: "true"}
        and (.results | type) == "object"
        and ([.results[("plan", "upstream", "package_images", "homebrew", "release_assembly", "npm_addons",
               "npm_package", "npm_preflight", "publish_npm", "image_index", "promote_images",
               "publish_release_assets")] == "success"] | all)
    ' "$3" >/dev/null
}

verify() { # verify <run id> <attempt>
    local run="$1" attempt="$2" ids id n=0 matched=0 members candidate kind
    ids="$(receipt_artifacts "$run")" || exit 1
    [[ -n "$ids" ]] || fail "the packaging run has no readiness receipt"
    while read -r id; do
        n=$((n + 1))
        ((n <= 16)) || fail "too many readiness receipts"
        get "$work/receipt.zip" "repos/$packaging/actions/artifacts/$id/zip"
        members="$(unzip -Z1 "$work/receipt.zip")" || fail "the receipt archive is not a zip"
        [[ "$members" == packaging-readiness.json ]] || fail "the receipt archive must hold only packaging-readiness.json"
        kind="$(unzip -Z "$work/receipt.zip" packaging-readiness.json)" || fail "could not list the receipt archive"
        [[ "$kind" == -* ]] || fail "the receipt must be a regular file"
        # A small archive can hold a huge file, so at most one byte past the
        # limit reaches the disk. A receipt cut short there is refused, not
        # parsed, and the archive's own check catches a damaged one.
        { unzip -p "$work/receipt.zip" packaging-readiness.json 2>/dev/null || true; } |
            head -c $((max_bytes + 1)) >"$work/receipt.json" || fail "could not read the receipt"
        (($(wc -c <"$work/receipt.json") <= max_bytes)) || fail "the receipt is too large"
        unzip -tq "$work/receipt.zip" >/dev/null 2>&1 || fail "the receipt archive is damaged"
        candidate="$(query "$work/receipt.json" --argjson run "$run" \
            'if .packaging_run_id == $run and (.packaging_run_attempt | type) == "number" and .packaging_run_attempt > 0
             then .packaging_run_attempt else error("bad receipt") end')" || exit 1
        ((candidate <= attempt)) || fail "a receipt is from a later attempt"
        ((candidate == attempt)) || continue
        ((matched == 0)) || fail "more than one receipt for this attempt"
        check_receipt "$run" "$attempt" "$work/receipt.json" || fail "the receipt does not match this release"
        matched=1
    done <<<"$ids"
    ((matched == 1)) || fail "no receipt for the current attempt"

    # The run must still be the same finished, successful attempt.
    get "$work/run.json" "repos/$packaging/actions/runs/$run"
    jq -e --argjson run "$run" --argjson attempt "$attempt" --arg title "$title" '
        .id == $run and .run_attempt == $attempt and .display_title == $title
        and .event == "repository_dispatch" and .status == "completed" and .conclusion == "success"
    ' "$work/run.json" >/dev/null || fail "the packaging run changed while its receipt was checked"
}

while :; do
    found="$(find_run)" || exit 1
    if [[ -n "$found" ]]; then
        read -r run attempt status conclusion <<<"$found"
        case "$status" in
            completed)
                [[ "$conclusion" == success ]] || fail "the packaging run concluded $conclusion"
                verify "$run" "$attempt"
                echo "packaging run $run attempt $attempt succeeded with a matching receipt"
                exit 0
                ;;
            queued | in_progress | waiting | pending | requested) ;;
            *) fail "unknown packaging run status: $status" ;;
        esac
    fi
    ((SECONDS + poll < deadline)) || fail "timed out waiting for packaging readiness"
    sleep "$poll"
done
