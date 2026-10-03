#!/bin/bash
# A/B test: pipeline-routing branch vs main branch
#
# Usage: ./evals/ab-test.sh [scenario]
# If no scenario given, runs all scenarios.
#
# Prerequisites:
#   /tmp/mesh-llm-pipeline  — binary from pipeline-routing branch
#   /tmp/mesh-llm-main      — binary from main branch
#   Models already downloaded: Qwen2.5-32B + Hermes-7B

set -o pipefail

EVALS_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENARIOS_DIR="$EVALS_DIR/scenarios"
RESULTS_DIR="$EVALS_DIR/results"
DEFAULT_WAIT=30
MODEL_A="Qwen2.5-32B-Instruct-Q4_K_M"
MODEL_B="Hermes-2-Pro-Mistral-7B-Q4_K_M"

if [ ! -f /tmp/mesh-llm-pipeline ] || [ ! -f /tmp/mesh-llm-main ]; then
    echo "ERROR: Need /tmp/mesh-llm-pipeline and /tmp/mesh-llm-main"
    exit 1
fi

# Which scenarios to run
if [ -n "$1" ]; then
    SCENARIOS=("$1")
else
    SCENARIOS=()
    for d in "$SCENARIOS_DIR"/*/; do
        [ -f "$d/turns.txt" ] && SCENARIOS+=("$(basename "$d")")
    done
fi

echo "═══════════════════════════════════════════════════"
echo "  A/B Test: pipeline-routing vs main"
echo "  Scenarios: ${SCENARIOS[*]}"
echo "  Models: $MODEL_A + $MODEL_B"
echo "═══════════════════════════════════════════════════"

run_variant() {
    local variant="$1"    # "pipeline" or "main"
    local binary="$2"
    local scenario="$3"
    local scenario_dir="$SCENARIOS_DIR/$scenario"
    local result_dir="$RESULTS_DIR/$variant/$scenario"
    local turns_file="$scenario_dir/turns.txt"
    local session="pi-ab-$$"
    local mesh_session="mesh-ab-$$"

    echo ""
    echo "──── $variant / $scenario ────"

    # Fresh result dir with scenario files
    rm -rf "$result_dir"
    mkdir -p "$result_dir"
    for f in "$scenario_dir"/*; do
        base="$(basename "$f")"
        [ "$base" = "turns.txt" ] && continue
        [ "$base" = "prompt.txt" ] && continue
        cp "$f" "$result_dir/"
    done

    # Kill any existing mesh processes
    pkill -9 -f mesh-llm 2>/dev/null; pkill -9 -f llama-server 2>/dev/null; pkill -9 -f rpc-server 2>/dev/null
    sleep 2

    # Start mesh-llm
    RUST_LOG=mesh_llm=info MESH_LLM_EPHEMERAL_KEY=1 "$binary" \
        --model "$MODEL_A" --model "$MODEL_B" \
        > "$result_dir/_mesh.log" 2>&1 &
    local mesh_pid=$!

    # Wait for models to load (check /v1/models)
    echo "  ⏳ Waiting for models..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:9337/v1/models > /dev/null 2>&1; then
            model_count=$(curl -s http://localhost:9337/v1/models | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))" 2>/dev/null || echo 0)
            [ "$model_count" -ge 2 ] && break
        fi
        sleep 2
    done

    model_count=$(curl -s http://localhost:9337/v1/models | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))" 2>/dev/null || echo 0)
    if [ "$model_count" -lt 2 ]; then
        echo "  ❌ Models didn't load in time"
        kill $mesh_pid 2>/dev/null; wait $mesh_pid 2>/dev/null
        return 1
    fi
    echo "  ✅ $model_count models loaded"

    # Launch pi in tmux
    local start_time=$(date +%s)
    tmux new-session -d -s "$session" -x 200 -y 50
    tmux send-keys -t "$session" "cd $result_dir && pi --provider mesh --model auto --working-dir $result_dir --no-session" Enter
    sleep 5

    # Send each turn
    local turn_num=0
    while IFS= read -r line || [ -n "$line" ]; do
        [ -z "$line" ] && continue
        [[ "$line" =~ ^#.* ]] && {
            if [[ "$line" =~ ^#wait:([0-9]+) ]]; then
                sleep "${BASH_REMATCH[1]}"
            fi
            continue
        }
        turn_num=$((turn_num + 1))
        echo "  → Turn $turn_num: ${line:0:55}..."
        tmux send-keys -t "$session" "$line" Enter
        sleep "$DEFAULT_WAIT"
        tmux capture-pane -t "$session" -p > "$result_dir/_screen_turn${turn_num}.txt" 2>/dev/null
    done < "$turns_file"

    sleep 5
    tmux capture-pane -t "$session" -p -S - > "$result_dir/_output.txt" 2>/dev/null
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    tmux kill-session -t "$session" 2>/dev/null

    echo "$elapsed" > "$result_dir/_time.txt"
    echo "$variant" > "$result_dir/_variant.txt"
    echo "$turn_num" > "$result_dir/_turns.txt"

    # Extract router/pipeline decisions
    grep -E "router:|pipeline:" "$result_dir/_mesh.log" > "$result_dir/_routing.txt" 2>/dev/null

    echo "  ⏱  ${elapsed}s, $turn_num turns"

    # Check if any files were modified
    local modified=0
    for f in "$scenario_dir"/*; do
        base="$(basename "$f")"
        [ "$base" = "turns.txt" ] || [ "$base" = "prompt.txt" ] && continue
        if [ -f "$result_dir/$base" ] && ! diff -q "$f" "$result_dir/$base" > /dev/null 2>&1; then
            modified=$((modified + 1))
        fi
    done
    echo "  📝 $modified files modified by agent"
    echo "$modified" > "$result_dir/_files_modified.txt"

    # Cleanup
    kill $mesh_pid 2>/dev/null; wait $mesh_pid 2>/dev/null
    pkill -9 -f mesh-llm 2>/dev/null; pkill -9 -f llama-server 2>/dev/null; pkill -9 -f rpc-server 2>/dev/null
    sleep 2
}

# Run each scenario on both variants
for scenario in "${SCENARIOS[@]}"; do
    run_variant "main" "/tmp/mesh-llm-main" "$scenario"
    run_variant "pipeline" "/tmp/mesh-llm-pipeline" "$scenario"
done

# Generate comparison report
echo ""
echo "═══════════════════════════════════════════════════"
echo "  COMPARISON REPORT"
echo "═══════════════════════════════════════════════════"
printf "%-20s %-8s %-8s %-8s %-8s %-12s %-12s\n" "Scenario" "M-Time" "P-Time" "M-Files" "P-Files" "M-Routing" "P-Routing"
echo "───────────────────────────────────────────────────────────────────────────────────"

for scenario in "${SCENARIOS[@]}"; do
    m_time=$(cat "$RESULTS_DIR/main/$scenario/_time.txt" 2>/dev/null || echo "?")
    p_time=$(cat "$RESULTS_DIR/pipeline/$scenario/_time.txt" 2>/dev/null || echo "?")
    m_files=$(cat "$RESULTS_DIR/main/$scenario/_files_modified.txt" 2>/dev/null || echo "?")
    p_files=$(cat "$RESULTS_DIR/pipeline/$scenario/_files_modified.txt" 2>/dev/null || echo "?")
    m_routes=$(wc -l < "$RESULTS_DIR/main/$scenario/_routing.txt" 2>/dev/null | tr -d ' ' || echo "?")
    p_routes=$(wc -l < "$RESULTS_DIR/pipeline/$scenario/_routing.txt" 2>/dev/null | tr -d ' ' || echo "?")
    p_pipelines=$(grep -c "pipeline:" "$RESULTS_DIR/pipeline/$scenario/_routing.txt" 2>/dev/null || echo "0")

    printf "%-20s %-8s %-8s %-8s %-8s %-12s %-12s\n" \
        "$scenario" "${m_time}s" "${p_time}s" "$m_files" "$p_files" \
        "${m_routes} routes" "${p_routes} (${p_pipelines} pipe)"
done

echo ""
echo "Pipeline routing details:"
for scenario in "${SCENARIOS[@]}"; do
    echo ""
    echo "── $scenario ──"
    echo "  Main routing:"
    cat "$RESULTS_DIR/main/$scenario/_routing.txt" 2>/dev/null | sed 's/.*\] /    /' | head -10
    echo "  Pipeline routing:"
    cat "$RESULTS_DIR/pipeline/$scenario/_routing.txt" 2>/dev/null | sed 's/.*\] /    /' | head -10
done

echo ""
echo "Full results in: $RESULTS_DIR/{main,pipeline}/<scenario>/"
