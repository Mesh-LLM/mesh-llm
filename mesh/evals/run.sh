#!/bin/bash
# A/B eval runner for pi with mesh vs opus
#
# Usage:
#   ./evals/run.sh mesh edit-file     # Run edit-file scenario with mesh
#   ./evals/run.sh opus edit-file     # Run edit-file scenario with opus
#   ./evals/run.sh mesh all           # Run all scenarios with mesh
#   ./evals/run.sh opus all           # Run all scenarios with opus
#
# Results go to evals/results/<provider>/<scenario>/

set -e

PROVIDER="${1:?Usage: $0 <mesh|opus> <scenario|all>}"
SCENARIO="${2:?Usage: $0 <mesh|opus> <scenario|all>}"

EVALS_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENARIOS_DIR="$EVALS_DIR/scenarios"
RESULTS_DIR="$EVALS_DIR/results"

run_scenario() {
    local provider="$1"
    local scenario="$2"
    local scenario_dir="$SCENARIOS_DIR/$scenario"
    local result_dir="$RESULTS_DIR/$provider/$scenario"
    local prompt_file="$scenario_dir/prompt.txt"

    if [ ! -f "$prompt_file" ]; then
        echo "ERROR: No prompt.txt in $scenario_dir"
        return 1
    fi

    # Fresh working copy for each run
    rm -rf "$result_dir"
    mkdir -p "$result_dir"

    # Copy scenario files (except prompt.txt) as working files
    for f in "$scenario_dir"/*; do
        [ "$(basename "$f")" = "prompt.txt" ] && continue
        cp "$f" "$result_dir/"
    done

    local prompt
    prompt=$(cat "$prompt_file")

    echo "═══════════════════════════════════════════════════"
    echo "  Provider: $provider | Scenario: $scenario"
    echo "═══════════════════════════════════════════════════"

    local start_time
    start_time=$(date +%s)

    # Run pi with the appropriate provider
    if [ "$provider" = "mesh" ]; then
        pi --provider mesh --model auto \
            -p "$prompt" \
            --working-dir "$result_dir" \
            2>&1 | tee "$result_dir/_output.txt"
    elif [ "$provider" = "mesh-pinned" ]; then
        # Use the biggest model directly (no routing)
        pi --provider mesh --model Qwen3-30B-A3B-Q4_K_M \
            -p "$prompt" \
            --working-dir "$result_dir" \
            2>&1 | tee "$result_dir/_output.txt"
    elif [ "$provider" = "opus" ]; then
        pi --provider anthropic --model "claude-sonnet-4-20250514" \
            -p "$prompt" \
            --working-dir "$result_dir" \
            2>&1 | tee "$result_dir/_output.txt"
    else
        echo "Unknown provider: $provider (use mesh or opus)"
        return 1
    fi

    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    echo ""
    echo "⏱  Completed in ${elapsed}s"
    echo "$elapsed" > "$result_dir/_time.txt"
    echo "$provider" > "$result_dir/_provider.txt"
    echo "$scenario" > "$result_dir/_scenario.txt"

    # Show what files were created/modified
    echo "📁 Files in result dir:"
    ls -la "$result_dir/" | grep -v "^total" | grep -v "^\." | grep -v "_output\|_time\|_provider\|_scenario"
    echo ""
}

# Collect scenarios
if [ "$SCENARIO" = "all" ]; then
    scenarios=$(ls "$SCENARIOS_DIR")
else
    scenarios="$SCENARIO"
fi

for s in $scenarios; do
    if [ ! -d "$SCENARIOS_DIR/$s" ]; then
        echo "Scenario '$s' not found in $SCENARIOS_DIR"
        exit 1
    fi
    run_scenario "$PROVIDER" "$s"
done

echo "═══════════════════════════════════════════════════"
echo "  All done! Results in $RESULTS_DIR/$PROVIDER/"
echo "═══════════════════════════════════════════════════"
