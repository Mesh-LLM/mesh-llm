#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
    echo 'usage: profile.sh QUALIFIED_WORKSPACE MODEL_FILE ARTIFACT_DIRECTORY' >&2
    exit 2
fi

qualified_workspace=$1
model_file=$2
artifact_directory=$3
mkdir -p "$artifact_directory"
artifact_directory=$(cd "$artifact_directory" && pwd -P)
cd "$qualified_workspace"
test "$(git rev-parse HEAD)" = a9b37b71bb9ebcff51e321e7fcedba323f7d9458
test -z "$(git status --porcelain --untracked-files=no)"
test "$(sysctl -n machdep.cpu.brand_string)" = 'Apple M6'
test "$(sysctl -n hw.memsize)" = 25769803776
shasum -a 256 -c .deps/grammar-receipts/binary-hashes.txt
shasum -a 256 -c .deps/grammar-receipts/input-hashes.txt
printf '%s  %s\n' 03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8 "$model_file" | shasum -a 256 -c -
candidate_binary="$PWD/.deps/grammar-C/grammar-sampling"
candidate_hash=$(shasum -a 256 "$candidate_binary" | cut -d ' ' -f 1)
test "$candidate_hash" = d93d1643c2675f4f33e8bd8d720c639075448225fa5f6468e7f79c72033e12f1
private_directory=$(mktemp -d /tmp/skippy-simd-profile.XXXXXX)
developer_directory=$(xcode-select -p)
profile_path=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin
{
    git rev-parse HEAD
    sysctl -n machdep.cpu.brand_string
    sysctl -n hw.memsize
    sw_vers
    shasum -a 256 "$candidate_binary" .deps/grammar-inputs/fixtures.json
    printf 'diagnostic only; profiler observations do not qualify performance\n'
} > "$artifact_directory/identity.txt"

# Trace bundles stay private to this job. Only scrubbed stack tables are exported.
for case_name in lazy-before-tool-trigger large-HTML-write-argument Unicode-and-partial-UTF8 restricted-values-and-rejection-fallback closing-delimiters-and-EOS; do
    env -i PATH="$profile_path" HOME="$HOME" USER=gi LOGNAME=gi LC_ALL=C \
        DEVELOPER_DIR="$developer_directory" TMPDIR="$private_directory" \
        xcrun xctrace record --template 'Time Profiler' --no-prompt --time-limit 180s \
        --output "$private_directory/$case_name.trace" \
        --target-stdout "$artifact_directory/$case_name-observations.json" \
        --launch -- "$candidate_binary" "$model_file" replay "$PWD/.deps/grammar-inputs" C "$case_name" 100 \
        > "$artifact_directory/$case_name-profiler.log" 2>&1
    env -i PATH="$profile_path" HOME="$HOME" DEVELOPER_DIR="$developer_directory" \
        xcrun xctrace export --input "$private_directory/$case_name.trace" \
        --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
        --output "$artifact_directory/$case_name-time-profile.xml" \
        > "$artifact_directory/$case_name-export.log" 2>&1
    test -s "$artifact_directory/$case_name-time-profile.xml"
    printf 'Exported current-C sampled stacks: %s\n' "$case_name"
done

xcrun llvm-nm --defined-only "$candidate_binary" > "$private_directory/symbols.txt"
symbols=$(awk '$3 ~ /^_Z21common_sampler_sampleP/ || $3 ~ /^_Z24skippy_chat_sample_tokenP/ || $3 ~ /partial_sort_inplace/ {print $3}' "$private_directory/symbols.txt" | paste -sd, -)
test -n "$symbols"
xcrun llvm-objdump --disassemble-symbols="$symbols" --demangle --symbolize-operands "$candidate_binary" > "$artifact_directory/current-C-disassembly.txt"
cp .deps/grammar-receipts/binary-hashes.txt "$artifact_directory/binary-hashes.txt"
cp .deps/grammar-receipts/input-hashes.txt "$artifact_directory/input-hashes.txt"
shasum -a 256 "$artifact_directory/"* > "$private_directory/artifact-hashes.txt"
cp "$private_directory/artifact-hashes.txt" "$artifact_directory/artifact-hashes.txt"
