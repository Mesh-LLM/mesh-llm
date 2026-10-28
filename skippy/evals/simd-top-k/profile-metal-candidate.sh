#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
    echo 'usage: profile-metal-candidate.sh RETAINED_WORKSPACE QUALIFIED_WORKSPACE MODEL_FILE ARTIFACT_DIRECTORY' >&2
    exit 2
fi
retained=$1
qualified=$2
model_file=$3
artifacts=$4
binary="$retained/.deps/simd-D/grammar-sampling"
test "$(sysctl -n machdep.cpu.brand_string)" = 'Apple M6'
test "$(shasum -a 256 "$binary" | cut -d ' ' -f 1)" = aded43ba5f769c0eabcaa552e32fba1fa592743d3df8a4e4ddf880511d631ddb
test "$(git -C "$retained/.deps/llama.cpp" rev-parse 'HEAD^{tree}')" = c81b95e77324501872f9aac8acc2488d9365df63
mkdir -p "$artifacts"
artifacts=$(cd "$artifacts" && pwd -P)
private=$(mktemp -d /tmp/skippy-highway-metal-isa.XXXXXX)
developer_directory=$(xcode-select -p)
# The prior LLDB launch blocked before executing its child. Time Profiler has
# already launched this campaign's owned workers on this same JY account.
env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin HOME="$HOME" \
    USER=gi LOGNAME=gi LC_ALL=C DEVELOPER_DIR="$developer_directory" TMPDIR="$private" \
    xcrun xctrace record --template 'Time Profiler' --no-prompt --time-limit 180s \
    --output "$private/selection.trace" --target-stdout "$artifacts/isa-replay-diagnostic.json" \
    --launch -- "$binary" "$model_file" replay "$qualified/.deps/grammar-inputs" C \
    lazy-before-tool-trigger 100 > "$artifacts/isa-record.log" 2>&1
env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME="$HOME" DEVELOPER_DIR="$developer_directory" \
    xcrun xctrace export --input "$private/selection.trace" \
    --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
    --output "$artifacts/selected-highway-profile.xml" > "$artifacts/isa-export.log" 2>&1
test -s "$artifacts/selected-highway-profile.xml"
printf '%s\n' 'Diagnostic only. Require an actual hwy::N_*::PartialSortKV64Desc frame; no profiled timing qualifies speed.' \
    > "$artifacts/isa-scope.txt"
