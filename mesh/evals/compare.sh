#!/bin/bash
# Compare mesh vs opus results side by side
#
# Usage: ./evals/compare.sh [scenario]
#        ./evals/compare.sh              # compare all
#        ./evals/compare.sh edit-file    # compare one

EVALS_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$EVALS_DIR/results"

if [ -n "$1" ]; then
    scenarios="$1"
else
    scenarios=$(ls "$EVALS_DIR/scenarios" 2>/dev/null)
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║              Mesh vs Opus Comparison                     ║"
echo "╠═══════════════════════════════════════════════════════════╣"

for s in $scenarios; do
    mesh_dir="$RESULTS_DIR/mesh/$s"
    opus_dir="$RESULTS_DIR/opus/$s"

    echo "║                                                           ║"
    echo "║  Scenario: $s"

    # Time comparison
    mesh_time="—"
    opus_time="—"
    [ -f "$mesh_dir/_time.txt" ] && mesh_time="$(cat "$mesh_dir/_time.txt")s"
    [ -f "$opus_dir/_time.txt" ] && opus_time="$(cat "$opus_dir/_time.txt")s"
    echo "║  ⏱  Mesh: $mesh_time  |  Opus: $opus_time"

    # Files produced
    mesh_files=$(ls "$mesh_dir" 2>/dev/null | grep -v "^_" | wc -l | tr -d ' ')
    opus_files=$(ls "$opus_dir" 2>/dev/null | grep -v "^_" | wc -l | tr -d ' ')
    echo "║  📁 Mesh: $mesh_files files  |  Opus: $opus_files files"

    # Output size (rough proxy for verbosity)
    mesh_output="—"
    opus_output="—"
    [ -f "$mesh_dir/_output.txt" ] && mesh_output="$(wc -l < "$mesh_dir/_output.txt" | tr -d ' ') lines"
    [ -f "$opus_dir/_output.txt" ] && opus_output="$(wc -l < "$opus_dir/_output.txt" | tr -d ' ') lines"
    echo "║  💬 Mesh: $mesh_output  |  Opus: $opus_output"

    # Diff the actual output files (not the _meta files)
    for f in $(ls "$mesh_dir" 2>/dev/null | grep -v "^_"); do
        if [ -f "$opus_dir/$f" ]; then
            if diff -q "$mesh_dir/$f" "$opus_dir/$f" > /dev/null 2>&1; then
                echo "║  ✅ $f — identical"
            else
                lines_diff=$(diff "$mesh_dir/$f" "$opus_dir/$f" | grep "^[<>]" | wc -l | tr -d ' ')
                echo "║  🔀 $f — $lines_diff lines differ"
            fi
        fi
    done

    echo "╠═══════════════════════════════════════════════════════════╣"
done

echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "To inspect a specific result:"
echo "  cat evals/results/mesh/<scenario>/_output.txt"
echo "  diff evals/results/mesh/<scenario>/file evals/results/opus/<scenario>/file"
