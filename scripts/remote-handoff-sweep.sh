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

python3 - "$OUT_DIR" <<'EOF'
import glob, json, sys

rows = []
for path in sorted(glob.glob(f"{sys.argv[1]}/send-*.json")):
    r = json.load(open(path))
    rows.append((
        r["prompt_token_count"],
        r["state_bytes"] / 2**20,
        r["transfer_gbps"],
        r["source_prefill_ms"],
        r["state_export_ms"],
        r["transfer_ms"],
        r["receiver"]["kv_attach_ms"],
        r["ttft_disaggregated_ms"],
        r.get("ttft_local_ms"),
        r.get("ttft_speedup"),
        r["matches"],
    ))
rows.sort()
print(f"{'prefix':>7} {'MiB':>7} {'Gbps':>6} {'prefill':>8} {'export':>7} "
      f"{'xfer':>7} {'attach':>7} {'ttft-pd':>8} {'ttft-lo':>8} {'speedup':>7} match")
for r in rows:
    local = f"{r[8]:8.0f}" if r[8] is not None else "       -"
    speedup = f"{r[9]:7.2f}" if r[9] is not None else "      -"
    print(f"{r[0]:>7} {r[1]:7.1f} {r[2]:6.2f} {r[3]:8.0f} {r[4]:7.0f} "
          f"{r[5]:7.0f} {r[6]:7.0f} {r[7]:8.0f} {local} {speedup} {r[10]}")
EOF
