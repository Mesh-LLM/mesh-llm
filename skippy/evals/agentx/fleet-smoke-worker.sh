#!/usr/bin/env bash
# Fleet smoke worker: one model, llama vs skippy, 4-conversation multi-turn
# probe. Managed by `cargo xtool automation agentx fleet --mode smoke`; the
# env contract matches the fleet spec generator. Never hand-edit run copies.
set -Eeuo pipefail
W=$PWD
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
if [ "$BACKEND" = "cuda" ]; then
  [ "$(uname -m)" = "aarch64" ] || { echo NO_AARCH64_HOST; exit 33; }
  nvidia-smi -L >/dev/null 2>&1 || { echo NO_CUDA_GPU; exit 31; }
fi
THREADS="${THREADS:-20}"
mkdir -p "$W/opt/kv-tools" "$W/opt/kv-runtime" "$W/parts"
# ---- kits (CDN, sha-verified) ----
fetch_chunks () { # prefix count out sha
  for i in $(seq 0 $(($2-1))); do ii=$(printf '%02d' "$i");
    curl -fsSL --retry 4 --retry-delay 3 --max-time 900 "$KIT_URL_BASE/$1$ii" -o "$W/parts/f$ii" || { echo FETCH_FAIL $1$ii; exit 41; }; done
  cat "$W/parts"/f0* "$W/parts"/f1* "$W/parts"/f2* > "$3" 2>/dev/null || cat "$W/parts"/f* > "$3"
  printf '%s  %s\n' "$4" "$3" | shasum -a256 -c - || { echo SHA_MISMATCH "$3"; exit 41; }
}
if [ "$BACKEND" = "cuda" ]; then
  fetch_chunks "$CUDA_KEY_PREFIX" "$CUDA_CHUNKS" "$W/tools.tgz" "$CUDA_TGZ_SHA"
  fetch_chunks "$LLAMA_KEY_PREFIX" "$LLAMA_CHUNKS" "$W/llama.tgz" "$LLAMA_TGZ_SHA"
  tar xzf "$W/tools.tgz" -C "$W/opt" skippy native-runtimes
  mv "$W/opt/skippy" "$W/opt/kv-tools/skippy"
  cp -R "$W/opt/native-runtimes/meshllm-native-runtime-linux-aarch64-cuda13/." "$W/opt/kv-runtime/"
  mkdir -p "$W/opt/standard/bin" "$W/opt/standard/lib"
  tar xzf "$W/llama.tgz" -C "$W/opt/standard"
  LDC=$(ldconfig -p 2>/dev/null); RT="$W/opt/kv-runtime"
  for lib in libssl.so.3 libcrypto.so.3; do
    line=$(printf '%s\n' "$LDC" | grep -m1 " $lib ") || true; [ -n "$line" ] || continue
    cp "$(printf '%s' "$line" | awk '{print $NF}')" "$RT/lib/$lib"
    "$HOME/.local/bin/patchelf" --set-rpath '$ORIGIN' "$RT/lib/$lib" 2>/dev/null || patchelf --set-rpath '$ORIGIN' "$RT/lib/$lib" || true
  done
  chmod +x "$W/opt/standard/bin/llama-server" 2>/dev/null || true
else
  # metal: single kit tarball holding kv-tools/ standard/ kv-runtime/
  curl -fsSL --retry 4 --retry-delay 3 --max-time 1200 "$KIT_URL_BASE/$METAL_KIT_KEY" -o "$W/kit.tgz" || { echo FETCH_FAIL metal; exit 41; }
  printf '%s  %s\n' "$METAL_KIT_SHA" "$W/kit.tgz" | shasum -a256 -c -
  tar xzf "$W/kit.tgz" -C "$W/opt"
  MEM=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
  [ "$MEM" -ge "$((MIN_MAC_MEM_BYTES:-200000000000))" ] || { echo NEED_BIGGER_MAC; exit 30; }
  if ! command -v python3.12 >/dev/null; then
    PYURL="https://github.com/astral-sh/python-build-standalone/releases/download/20250115/cpython-3.12.8+20250115-aarch64-apple-darwin-install_only.tar.gz"
    mkdir -p "$W/py312"; curl -fsSL --retry 3 --max-time 600 "$PYURL" -o "$W/py312.tgz" \
      && tar xzf "$W/py312.tgz" -C "$W/py312" && export PATH="$W/py312/python/bin:$PATH" \
      || { echo NO_PYTHON312; exit 31; }
  fi
fi
# ---- model (hub cache verified, else verified download) ----
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
MODEL="$HUB/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REV/$MODEL_FILE"
if [ ! -f "$MODEL" ] || ! printf '%s  %s\n' "$MODEL_SHA" "$MODEL" | shasum -a256 -c - 2>/dev/null; then
  echo "downloading $MODEL_FILE"
  mkdir -p "$W/models"; MODEL="$W/models/$MODEL_FILE"
  ok=0
  for a in 1 2 3; do rm -f "$MODEL"
    curl -fL --retry 4 --retry-all-errors --max-time 2400 "https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REV/$MODEL_FILE" -o "$MODEL" \
      && printf '%s  %s\n' "$MODEL_SHA" "$MODEL" | shasum -a256 -c - && ok=1 && break
    sleep 10
  done
  [ "$ok" = "1" ] || { echo MODEL_DOWNLOAD_FAILED; exit 45; }
fi
# ---- probe ----
cat > "$W/smoke.py" <<'PYEOF'
import concurrent.futures, json, sys, time, urllib.request, urllib.error
PORT, NCONV, TURNS, BLOCK, MAXTOK = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
def convo(i, turn):
    filler = f"Session {i} stable context block. " * 220
    history = "\n".join(f"User: {filler[:BLOCK*k]} Question {i}-{k}: name a color." for k in range(1, turn+1))
    return [{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":history+f" Q{i}-{turn}: describe a sunset in one paragraph."}]
def ask(pair):
    i, turn = pair
    body = json.dumps({"model":"m","messages":convo(i,turn),"max_tokens":MAXTOK,"temperature":0,"stream":False}).encode()
    t0=time.time(); err=None; pt=ct=comp=0
    try:
        r=urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"}),timeout=2400)
        out=json.load(r); u=out.get("usage",{}); d=u.get("prompt_tokens_details",{}) or {}
        pt, ct, comp = u.get("prompt_tokens",0), d.get("cached_tokens",0), u.get("completion_tokens",0)
    except urllib.error.HTTPError as e: err=f"HTTP {e.code}: {e.read()[:120]}"
    except Exception as e: err=f"{type(e).__name__}: {e}"
    return (i, turn, pt, ct, comp, time.time()-t0, err)
turns=[]
for turn in range(1, TURNS+1):
    t0=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=NCONV) as ex:
        results = list(ex.map(ask, [(i,turn) for i in range(NCONV)]))
    wall=time.time()-t0
    errs=[r for r in results if r[6]]
    if errs:
        for r in errs[:2]: print(f"  turn {turn} conv {r[0]}: ERROR {r[6]}", flush=True)
        continue
    toks=sum(r[4] for r in results); cached=sum(r[3] for r in results); prompts=sum(r[2] for r in results)
    print(f"  RESULT turn {turn}: {toks/wall:6.1f} tok/s | cached {100*cached/max(prompts,1):3.0f}% | wall {wall:5.1f}s", flush=True)
    turns.append(toks/wall)
if turns:
    steady=turns[1:] or turns
    print(f"  SUMMARY steady_tok_s={sum(steady)/len(steady):.1f}", flush=True)
PYEOF
PORT=64800
run_engine () { # name logfile ready_timeout cmd...
  NAME="$1"; LOG="$2"; RT="$3"; shift 3
  PORT=$((PORT+1))
  "$@" > "$W/$LOG" 2>&1 &
  PID=$!
  for i in $(seq 1 "$RT"); do sleep 5; curl -s --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; done
  echo "-- $NAME:"
  python3 "$W/smoke.py" "$PORT" "$PROBE_CONVERSATIONS" "$PROBE_TURNS" "$PROBE_BLOCK_CHARS" "$PROBE_MAX_TOKENS" || true
  kill $PID 2>/dev/null || true; sleep 8
}
echo "===== MODEL $MODEL_KEY ($BACKEND) ====="
if [ "$BACKEND" = "cuda" ]; then
  run_engine llama "serve-llama.log" 120 "$W/opt/standard/bin/llama-server" --model "$MODEL" --alias m --host 127.0.0.1 --port $((PORT+1)) --parallel "$PROBE_CONVERSATIONS" --batch-size 2048 --ubatch-size 512 --n-gpu-layers 99 --cache-type-k f16 --cache-type-v f16 --flash-attn on --jinja --chat-template-kwargs '{"enable_thinking":false}' --ctx-size $((131072*PROBE_CONVERSATIONS)) --cache-ram 8192 --no-kv-unified --threads "$THREADS" --threads-batch "$THREADS"
  run_engine skippy "serve-skippy.log" 180 "$W/opt/kv-tools/skippy" --output jsonl serve --model-path "$MODEL" --model-id m --bind-addr "127.0.0.1:$((PORT+1))" --startup-timeout-secs 900 --batch-size 2048 --microbatch-size 512 --generation-queue-capacity 256 --generation-admission-timeout-secs 1500 --n-gpu-layers 99 --kv-cache-type-k f16 --kv-cache-type-v f16 --flash-attention enabled --speculative-strategy disabled --prefix-cache on --prefix-cache-ram 8GiB --kv-cache-disk off --threads "$THREADS" --threads-batch "$THREADS" --ctx-size 131072 --runtime-bundle "$W/opt/kv-runtime"
else
  run_engine llama "serve-llama.log" 120 "$W/opt/standard/bin/llama-server" --model "$MODEL" --alias m --host 127.0.0.1 --port $((PORT+1)) --parallel "$PROBE_CONVERSATIONS" --batch-size 2048 --ubatch-size 512 --n-gpu-layers 99 --cache-type-k f16 --cache-type-v f16 --flash-attn on --jinja --chat-template-kwargs '{"enable_thinking":false}' --ctx-size $((131072*PROBE_CONVERSATIONS)) --cache-ram 8192 --no-kv-unified --threads "$THREADS" --threads-batch "$THREADS"
  run_engine skippy "serve-skippy.log" 180 "$W/opt/kv-tools/skippy" --output jsonl serve --model-path "$MODEL" --model-id m --bind-addr "127.0.0.1:$((PORT+1))" --startup-timeout-secs 900 --batch-size 2048 --microbatch-size 512 --generation-queue-capacity 256 --generation-admission-timeout-secs 1500 --n-gpu-layers -1 --kv-cache-type-k f16 --kv-cache-type-v f16 --flash-attention enabled --speculative-strategy disabled --prefix-cache on --prefix-cache-ram 8GiB --kv-cache-disk off --threads "$THREADS" --threads-batch "$THREADS" --ctx-size 131072 --runtime-bundle "$W/opt/kv-runtime"
fi
echo SMOKE_DONE
