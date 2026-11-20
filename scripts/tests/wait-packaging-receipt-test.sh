#!/usr/bin/env bash
# wait-packaging-receipt.sh against a stub `gh` that serves files from a
# directory. No network. Each case starts from a release that should pass and
# changes one thing.
# shellcheck disable=SC2016
set -euo pipefail

wait_script="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/wait-packaging-receipt.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fix="$work/fix"
mkdir "$work/bin"

# gh api --method GET <path> [-f field]... -> the fixture for that path.
# The run list can differ per call: runs.1, runs.2, ... then runs.
cat >"$work/bin/gh" <<'STUB'
#!/usr/bin/env bash
path="$4"
# A request that never answers.
[[ ! -f "$FIX/stall" ]] || exec sleep 30
case "$path" in
    */workflows/images-release.yml/runs)
        n=$(($(cat "$FIX/calls" 2>/dev/null || echo 0) + 1))
        echo "$n" >"$FIX/calls"
        file="$FIX/runs.$n"
        [[ -f "$file" ]] || file="$FIX/runs"
        ;;
    */artifacts) file="$FIX/artifacts" ;;
    */zip) file="$FIX/receipt.zip" ;;
    */actions/runs/*) file="$FIX/run" ;;
    *) exit 1 ;;
esac
[[ -f "$file" ]] && cat "$file"
STUB
chmod +x "$work/bin/gh"

tag=v1.2.3
sha="$(printf 'a%.0s' {1..40})"
assets="$(printf 'b%.0s' {1..64})"
catalog="$(printf 'c%.0s' {1..64})"

run_list() { # <status> <conclusion> [title]
    jq -n --arg status "$1" --arg conclusion "$2" --arg title "${3:-Packaging · mesh-9-1}" \
        '{workflow_runs: [{id: 77, run_attempt: 1, status: $status, conclusion: $conclusion, display_title: $title}]}'
}
pack() { # zip receipt.json as the artifact
    rm -rf "$fix/z" "$fix/receipt.zip"
    mkdir "$fix/z"
    cp "$fix/receipt.json" "$fix/z/packaging-readiness.json"
    (cd "$fix/z" && zip -q ../receipt.zip ./*)
}
good() { # a fixture set that passes
    rm -rf "$fix"
    mkdir "$fix"
    run_list completed success >"$fix/runs"
    echo '{"artifacts": [{"id": 5, "name": "packaging-readiness", "expired": false, "size_in_bytes": 400, "workflow_run": {"id": 77}}]}' >"$fix/artifacts"
    jq -n --arg tag "$tag" --arg sha "$sha" --arg assets "$assets" --arg catalog "$catalog" '{
        schema: "mesh-packaging-readiness-v2", status: "success", correlation_id: "mesh-9-1",
        packaging_run_id: 77, packaging_run_attempt: 1,
        upstream: {repository: "Mesh-LLM/mesh-llm", ref: $tag, sha: $sha, assets_sha256: $assets},
        skippy: {repository: "Mesh-LLM/skippy", release: "v0.80.0", catalog_sha256: $catalog},
        requested: {publish_images: "true", publish_release_assets: "true", publish_npm: "true"},
        results: ([ "plan", "upstream", "package_images", "homebrew", "release_assembly", "npm_addons", "npm_package",
                    "npm_preflight", "publish_npm", "image_index", "promote_images", "publish_release_assets" ]
                  | map({(.): "success"}) | add)}' >"$fix/receipt.json"
    pack
    echo '{"id": 77, "run_attempt": 1, "display_title": "Packaging · mesh-9-1", "event": "repository_dispatch", "status": "completed", "conclusion": "success"}' >"$fix/run"
}
edit() { # <file> <jq filter>
    jq "$2" "$fix/$1" >"$fix/tmp" && mv "$fix/tmp" "$fix/$1"
}

fail=0
expect() { # <ok|message the error must contain> <what the case is>
    local out status=0
    out="$(env -i PATH="$work/bin:$PATH" FIX="$fix" GH_TOKEN=unused \
        PACKAGING_RECEIPT_TIMEOUT_SECONDS=6 PACKAGING_RECEIPT_POLL_SECONDS=1 PACKAGING_RECEIPT_REQUEST_SECONDS=2 \
        "$wait_script" --tag "$tag" --sha "$sha" --assets "$assets" --skippy-release v0.80.0 \
        --skippy-catalog "$catalog" --correlation mesh-9-1 --since 2026-11-20T00:00:00Z 2>&1)" || status=$?
    if [[ "$1" == ok ]]; then
        ((status == 0)) || { echo "FAIL: $2: $out" >&2; fail=1; }
    elif ((status == 0)) || [[ "$out" != *"$1"* ]]; then
        echo "FAIL: $2: wanted an error holding '$1', got status $status: $out" >&2
        fail=1
    fi
}

good
expect ok "a matching receipt"

good
run_list queued "" >"$fix/runs.1"
run_list in_progress "" >"$fix/runs.2"
expect ok "a run that finishes on the third look"

# One wrong field in the receipt is enough.
for change in \
    '.upstream.ref = "v9.9.9"' '.upstream.sha |= "d" + .[1:]' '.upstream.assets_sha256 |= "d" + .[1:]' \
    '.upstream.repository = "someone/else"' '.skippy.release = "v0.79.0"' '.skippy.catalog_sha256 |= "d" + .[1:]' \
    '.skippy.repository = "someone/else"' 'del(.skippy)' '.correlation_id = "mesh-9-2"' '.status = "failure"' \
    '.results.publish_npm = "skipped"' 'del(.results.image_index)' '.requested.publish_images = "false"' \
    '.schema = "mesh-packaging-readiness-v1"'; do
    good
    edit receipt.json "$change"
    pack
    expect "does not match this release" "receipt with $change"
done

good
run_list completed failure >"$fix/runs"
expect "concluded failure" "a failed packaging run"

good
run_list completed cancelled >"$fix/runs"
expect "concluded cancelled" "a cancelled packaging run"

good
run_list mystery "" >"$fix/runs"
expect "unknown packaging run status" "an unknown run status"

good
run_list completed success "Packaging · other" >"$fix/runs"
expect "timed out" "a run that never appears"

good
run_list queued "" >"$fix/runs"
expect "timed out" "a run that never finishes"

good
edit runs '.workflow_runs += .workflow_runs'
expect "more than one packaging run" "two runs with the same correlation id"

# A small archive holding a huge receipt is refused before it is unpacked.
good
head -c 3000000 /dev/zero | tr '\0' ' ' >"$fix/receipt.json"
pack
(($(wc -c <"$fix/receipt.zip") < 100000)) || { echo "FAIL: the oversized fixture did not compress" >&2; fail=1; }
expect "the receipt is too large" "a receipt that unpacks past the limit"

good
rm "$fix/z/packaging-readiness.json" "$fix/receipt.zip"
# The link's target text is the valid receipt, so only the file type gives it away.
ln -s "$(jq -c . "$fix/receipt.json")" "$fix/z/packaging-readiness.json"
(cd "$fix/z" && zip -qy ../receipt.zip packaging-readiness.json)
expect "must be a regular file" "a receipt that is a symlink"

good
touch "$fix/stall"
began=$SECONDS
expect "took longer than 2s" "a request that stalls"
((SECONDS - began < 6)) || { echo "FAIL: a stalled request was not cut off at its own limit" >&2; fail=1; }

good
head -c 1100000 /dev/zero | tr '\0' ' ' >>"$fix/runs" # still valid JSON, only too long
expect "larger than 1048576 bytes" "a response past the size limit"

good
head -c 1100000 /dev/zero >"$fix/receipt.zip"
expect "larger than 1048576 bytes" "a receipt archive past the size limit"

good
rm "$fix/runs"
expect "GitHub request failed" "a failed GitHub request"

good
echo '{"message": "Not Found"}' >"$fix/runs"
expect "unexpected response" "a run list that is not a run list"

good
echo '{"artifacts": []}' >"$fix/artifacts"
expect "no readiness receipt" "a run without a receipt"

good
edit artifacts '.artifacts[0].expired = true'
expect "unexpected response" "an expired receipt"

good
edit artifacts '.artifacts[0].workflow_run.id = 78'
expect "unexpected response" "a receipt artifact from another run"

good
echo extra >"$fix/z/other.txt"
(cd "$fix/z" && rm ../receipt.zip && zip -q ../receipt.zip ./*)
expect "must hold only packaging-readiness.json" "a receipt archive with a second file"

good
echo "not a zip" >"$fix/receipt.zip"
expect "not a zip" "a receipt that is not a zip"

good
edit receipt.json '.packaging_run_attempt = 2'
pack
expect "later attempt" "a receipt from a later attempt"

good
edit receipt.json '.packaging_run_id = 78'
pack
expect "unexpected response" "a receipt from another run"

good
edit run '.run_attempt = 2'
expect "changed while its receipt was checked" "a run re-run during the check"

((fail == 0)) && echo "wait-packaging-receipt-test: ok"
exit "$fail"
