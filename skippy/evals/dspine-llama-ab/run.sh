#!/usr/bin/env bash
# DSpine vs DFlash vs off on llama.cpp (mesh-llm pin + patch queue) on one GPU
# node. Runs from the jy workspace root after build.sh.
#
# Arms (ARMS env, default "off dflash dspine"): one fresh llama-server each,
# same target, same prompts, greedy, 256 tokens, concurrency 1. Draft arms use
# the full 15-token block (block_size 16 minus the anchor row).
#
#   off     no draft
#   dflash  z-lab/Qwen3-8B-DFlash-b16 converted to GGUF (plain DFlash, block 16)
#   dspine  tencent/Qwen3-8B-DSpine-b16 converted to GGUF (DSpine, block 16,
#           Ajay9o9 layout detected via dflash.selector_top_k on the DFlash2 path)
#
# Both drafts were converted from the pinned upstream revisions with the
# repo's converter (conversion/qwen.py), BF16, --target-model-dir Qwen/Qwen3-8B.
#
# Env:
#   ARMS     space-separated arms            (default: "off dflash dspine")
#   LIMIT    first N prompts only (smoke)    (default: all 77)
set -euo pipefail

HERE="$PWD/mesh-llm/skippy/evals/dspine-llama-ab"
ART="${JIANYANG_ARTIFACT_DIR:?}"
BIN="$PWD/mesh-llm/.deps/llama.cpp/build-dspine-ab/bin/llama-server"
[ -x "$BIN" ] || { echo "no llama-server build at $BIN" >&2; exit 1; }

W="$HOME/.cache/dspine-llama-ab"
mkdir -p "$W/models" "$ART"

ARMS="${ARMS:-off dflash dspine}"
LIMIT="${LIMIT:-0}"
CTX=8192

# --- weights -----------------------------------------------------------------

TARGET_URL="https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/a6adef130ffb23ddaf1a62fec9dced968c9bc482/Qwen3-8B-BF16.gguf"
TARGET_SHA="5e416a2020fe63e76ea13c8979be35fc6070aaf3578f7876400c55c2f5c3eb30"
DFLASH_URL="https://huggingface.co/jamesdumay/qwen3-8b-spec-drafts-gguf/resolve/main/qwen3-8b-dflash-b16.gguf"
DFLASH_SHA="cc55453612f0e1ffc37b1a97ddc04ee8ba8a67a75be25ff9678f8d072e4fe0b6"
DSPINE_URL="https://huggingface.co/jamesdumay/qwen3-8b-spec-drafts-gguf/resolve/main/qwen3-8b-dspine-v2-b16.gguf"
DSPINE_Q8_URL="https://huggingface.co/jamesdumay/qwen3-8b-spec-drafts-gguf/resolve/main/qwen3-8b-dspine-v2-q8_0.gguf"
DSPINE_SHA="923546406a0dd417e69c2e01cc5b5cf48024e314fc768594e31ed1784134897b"
DSPINE_Q8_SHA="439db62eb16d74659bb5d543fcdc5d0a8c265aef3ec95b2cab1f1de9695ad994"

fetch() { # fetch <url> <sha256> <dest>
  local url="$1" sha="$2" dest="$3"
  if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    echo "ok cached $dest"
    return 0
  fi
  echo "downloading $url"
  curl -fL --retry 4 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
  local got
  got=$(sha256sum "$dest" | cut -d' ' -f1)
  [ "$got" = "$sha" ] || { echo "sha256 mismatch on $dest: got $got want $sha" >&2; return 1; }
  echo "ok $dest"
}

echo "=== fetch + verify weights"
fetch "$TARGET_URL" "$TARGET_SHA" "$W/models/qwen3-8b-bf16.gguf"
case " $ARMS " in *" dflash "*) fetch "$DFLASH_URL" "$DFLASH_SHA" "$W/models/qwen3-8b-dflash-b16.gguf";; esac
case " $ARMS " in *dspine*) fetch "$DSPINE_URL" "$DSPINE_SHA" "$W/models/qwen3-8b-dspine-v2-b16.gguf";; esac
case " $ARMS " in *dspine-q8*) fetch "$DSPINE_Q8_URL" "$DSPINE_Q8_SHA" "$W/models/qwen3-8b-dspine-v2-q8_0.gguf";; esac

gpu_state() { nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,pstate --format=csv,noheader 2>/dev/null | sed "s/^/gpu: $1 /" || true; }

# --- arms --------------------------------------------------------------------

PORT=8399
SRV_PID=""
stop_server() {
  [ -n "$SRV_PID" ] && { kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; SRV_PID=""; }
  return 0
}
trap stop_server EXIT

run_arm() {
  local arm="$1"; shift
  echo "=== arm $arm: $BIN $*"
  gpu_state "$arm-before"
  "$BIN" "$@" --port "$PORT" > "$ART/server-$arm.log" 2>&1 &
  SRV_PID=$!
  local t0 up=0
  t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 900 ]; do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
    kill -0 "$SRV_PID" 2>/dev/null || break
    sleep 5
  done
  [ "$up" = 1 ] || { echo "server for arm $arm did not come up; see $ART/server-$arm.log" >&2; return 1; }
  python3 "$HERE/measure.py" "$PORT" "$arm" "$HERE/prompts.jsonl" "$LIMIT" | tee "$ART/arm-$arm.log"
  stop_server
  gpu_state "$arm-after"
}

for arm in $ARMS; do
  case "$arm" in
    off)
      run_arm off -m "$W/models/qwen3-8b-bf16.gguf" -fa on -c "$CTX" --spec-type none;;
    dflash)
      # plain DFlash needs CPU sampling: backend sampling rejects the
      # multi-output noise block at this pin ("at most one output token per
      # sequence"); DSpine bypasses backend sampling on its own
      run_arm dflash -m "$W/models/qwen3-8b-bf16.gguf" -md "$W/models/qwen3-8b-dflash-b16.gguf" \
        --spec-type draft-dflash --spec-draft-n-max 15 --no-spec-draft-backend-sampling -fa on -c "$CTX";;
    dspine)
      run_arm dspine -m "$W/models/qwen3-8b-bf16.gguf" -md "$W/models/qwen3-8b-dspine-v2-b16.gguf" \
        --spec-type draft-dflash --spec-draft-n-max 15 -fa on -c "$CTX";;
    dspine16)
      # clamp smoke test: the runtime clamps n_max to block_size - 1 = 15,
      # so this arm must produce identical acceptance to dspine
      run_arm dspine16 -m "$W/models/qwen3-8b-bf16.gguf" -md "$W/models/qwen3-8b-dspine-v2-b16.gguf" \
        --spec-type draft-dflash --spec-draft-n-max 16 -fa on -c "$CTX";;
    dspine-q8)
      # quantized draft: Q8_0 weights, paper block width (1 anchor + 15 drafts)
      run_arm dspine-q8 -m "$W/models/qwen3-8b-bf16.gguf" -md "$W/models/qwen3-8b-dspine-v2-q8_0.gguf" \
        --spec-type draft-dflash --spec-draft-n-max 15 -fa on -c "$CTX";;
    dspine-p05|dspine-p10)
      # walk early-exit sweep: truncate the chain when transition confidence
      # drops below p_min
      pmin="0.${arm##dspine-p}"
      run_arm "$arm" -m "$W/models/qwen3-8b-bf16.gguf" -md "$W/models/qwen3-8b-dspine-v2-b16.gguf" \
        --spec-type draft-dflash --spec-draft-n-max 15 --spec-draft-p-min "$pmin" -fa on -c "$CTX";;
    *) echo "unknown arm: $arm" >&2; exit 1;;
  esac
done

echo "=== all arms done"
