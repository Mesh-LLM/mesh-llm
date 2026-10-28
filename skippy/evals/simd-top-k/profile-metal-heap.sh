#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 5 ]]; then
    echo 'usage: profile-metal-heap.sh RETAINED_WORKSPACE QUALIFIED_WORKSPACE MODEL_FILE ARTIFACT_DIRECTORY BINARY_SHA256' >&2
    exit 2
fi
retained=$1
qualified=$2
model_file=$3
artifacts=$4
binary_sha256=$5
binary="$retained/.deps/simd-D/grammar-sampling"
test "$(sysctl -n machdep.cpu.brand_string)" = 'Apple M6'
test "$(shasum -a 256 "$binary" | cut -d ' ' -f 1)" = "$binary_sha256"
test "$(git -C "$retained/.deps/llama.cpp" rev-parse 'HEAD^{tree}')" = e4cc2b65b42834fe2ab257a2517afd969ba7f9c5
mkdir -p "$artifacts"
artifacts=$(cd "$artifacts" && pwd -P)
nm "$binary" | awk '$2 == "t" && $3 ~ /llama_top_k.*N_NEON.*L10HeapFilter/ { print $3 }' > "$artifacts/heap-kernel-symbols.txt"
test -s "$artifacts/heap-kernel-symbols.txt"
while IFS= read -r symbol; do
    xcrun llvm-objdump --macho --disassemble --dis-symname "$symbol" "$binary"
done < "$artifacts/heap-kernel-symbols.txt" > "$artifacts/heap-kernel-disassembly.txt"
private=$(mktemp -d /tmp/skippy-heap-metal-isa.XXXXXX)
developer_directory=$(xcode-select -p)
# A compiled symbol is insufficient: retain actual sampled frames from this exact
# worker, then require vector comparison instructions in its executed target.
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
printf '%s\n' 'Diagnostic only. Require actual llama_top_k::N_NEON*::HeapFilter and production wrapper in the same admitted-binary stack, plus vector comparison/skip disassembly. No profiled timing qualifies speed.' > "$artifacts/isa-scope.txt"
