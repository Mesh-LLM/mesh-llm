#!/usr/bin/env bash
# Sweep remote-handoff sender runs across prefix lengths against a receiver
# started with --accept-count matching the number of runs, e.g.:
#
#   receiver$ target/release/skippy-correctness remote-handoff --role recv \
#       --listen 0.0.0.0:19081 --model M --layer-end N --ctx-size 16384 \
#       --n-gpu-layers 99 --decode-tokens 32 --accept-count 4 \
#       --allow-mismatch --report-out recv.json
#
#   sender$ scripts/remote-handoff-sweep.sh <receiver-ip>:19081 <model.gguf> \
#       <layer-end> <out-dir> [prefix counts...]
set -euo pipefail

PEER="${1:?receiver address}"
MODEL="${2:?model path}"
LAYER_END="${3:?layer end}"
OUT_DIR="${4:?output directory}"
shift 4
PREFIXES=("${@:-512 2048 4096 8192}")
if [[ $# -eq 0 ]]; then PREFIXES=(512 2048 4096 8192); fi

CTX_SIZE="${CTX_SIZE:-16384}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
BIN="${BIN:-target/release/skippy-correctness}"

mkdir -p "$OUT_DIR"
for prefix in "${PREFIXES[@]}"; do
  echo "== prefix ${prefix}"
  "$BIN" remote-handoff --role send --peer "$PEER" \
    --model "$MODEL" --layer-end "$LAYER_END" --ctx-size "$CTX_SIZE" \
    --n-gpu-layers 99 --prefix-token-count "$prefix" \
    --decode-tokens "$DECODE_TOKENS" --baseline \
    --report-out "$OUT_DIR/send-${prefix}.json" \
    > "$OUT_DIR/send-${prefix}.log" 2>&1 \
    || echo "   prefix ${prefix} FAILED (see $OUT_DIR/send-${prefix}.log)"
done

automation=(cargo xtool)
if [[ "${MESH_LLM_AUTOMATION_BIN+set}" == set ]]; then
  if [[ "$MESH_LLM_AUTOMATION_BIN" != /* || ! -f "$MESH_LLM_AUTOMATION_BIN" || ! -x "$MESH_LLM_AUTOMATION_BIN" ]]; then
    echo "MESH_LLM_AUTOMATION_BIN must be an absolute regular executable" >&2
    exit 2
  fi
  automation=("$MESH_LLM_AUTOMATION_BIN")
fi
"${automation[@]}" automation remote-handoff-summary --reports-dir "$OUT_DIR"
