#!/usr/bin/env bash
# Fleet worker: builds product + baseline from pinned source in-job, then runs
# the AgentX benchmark. Everything that leaves the machine travels through jy
# artifacts (the daemon uploads $JIANYANG_ARTIFACT_DIR); nothing is fetched
# from or pushed to external object storage. Managed by
# `cargo xtool automation agentx fleet`; see .agents/skills/agentx-fleet-bench.
#
# Env contract (set by the generated spec):
#   MODE (smoke|comparison), PRODUCT_COMMIT, REPO_URL,
#   BACKEND (cuda|metal), MODEL_KEY, MODEL_REPO, MODEL_REV, MODEL_FILE, MODEL_SHA,
#   TOK_REPO, TOK_REV, CORPUS_REPO, CORPUS_REV, CORPUS_SHA (comparison),
#   PROBE_CONVERSATIONS, PROBE_TURNS, PROBE_BLOCK_CHARS, PROBE_MAX_TOKENS (smoke),
#   THREADS, CAMPAIGN
# Exit codes: 30 need-bigger-mac, 31 no-python, 33 wrong-arch, 34 repo-clone,
#             46 product-build, 47 baseline-build, 48 client-env
set -Eeuo pipefail
W=$PWD
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
THREADS="${THREADS:-20}"
SMOKE_SCOPE="${SMOKE_SCOPE:-full}"

# ---- host guards (hardware placement; never node names) ----
if [ "$BACKEND" = "cuda" ]; then
  [ "$(uname -m)" = "aarch64" ] || { echo NO_AARCH64_HOST; exit 33; }
  nvidia-smi -L >/dev/null 2>&1 || { echo NO_CUDA_GPU; exit 31; }
else
  MEM=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
  [ "$MEM" -ge 200000000000 ] || { echo NEED_BIGGER_MAC; exit 30; }
fi
if ! command -v python3.12 >/dev/null; then
  PYTUPLE="aarch64-unknown-linux-gnu"; [ "$(uname -s)" = Darwin ] && PYTUPLE="aarch64-apple-darwin"
  mkdir -p "$W/py312"
  curl -fsSL --retry 3 --max-time 600 "https://github.com/astral-sh/python-build-standalone/releases/download/20250115/cpython-3.12.8+20250115-$PYTUPLE-install_only.tar.gz" -o "$W/py312.tgz" \
    && tar xzf "$W/py312.tgz" -C "$W/py312" && export PATH="$W/py312/python/bin:$PATH" \
    || { echo NO_PYTHON312; exit 31; }
fi
command -v cargo >/dev/null || curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
command -v just >/dev/null || cargo install just --locked
# Fresh machine: a cold sccache adds failure modes (CUDA cubin outputs are not
# cacheable) with no cache benefit. Use the build script's own off switch.
export LLAMA_STAGE_USE_SCCACHE=0
export SKIPPY_USE_SCCACHE=0
BUILD_PATH="$PATH"

# ---- product source at the pinned commit ----
echo "== cloning $REPO_URL @ ${PRODUCT_COMMIT:0:12}"
git clone -q "$REPO_URL" "$W/repo" || { echo REPO_CLONE_FAILED; exit 34; }
cd "$W/repo"
git checkout -q --detach "$PRODUCT_COMMIT" || { echo COMMIT_CHECKOUT_FAILED; exit 34; }
UPSTREAM_PIN="$(tr -d '[:space:]' < skippy/llama_cpp/upstream.txt)"

# ---- product build: native runtime + skippy CLI (owning recipes) ----
echo "== building product runtime (backend=$BACKEND)"
if ! PATH="$BUILD_PATH" just release-runtime-build "$BACKEND" > "$W/build-runtime.log" 2>&1; then
  tail -5 "$W/build-runtime.log"; echo PRODUCT_BUILD_FAILED; exit 46
fi
if ! PATH="$BUILD_PATH" just skippy-cli-release-build > "$W/build-cli.log" 2>&1; then
  tail -5 "$W/build-cli.log"; echo PRODUCT_BUILD_FAILED; exit 46
fi
RUNTIME_DIR="$(ls -d "$W/repo/dist/native-runtimes"/meshllm-native-runtime-* 2>/dev/null | head -1)"
NATIVE_COMMIT="$(git -C "$W/repo/.deps/llama.cpp" rev-parse HEAD 2>/dev/null || echo "$PRODUCT_COMMIT")"
[ -n "$RUNTIME_DIR" ] || { echo RUNTIME_DIR_MISSING; exit 46; }

# ---- baseline build: unpatched llama-server at the upstream pin ----
echo "== building vanilla llama-server @ ${UPSTREAM_PIN:0:12}"
git clone -q https://github.com/ggml-org/llama.cpp "$W/llama-upstream" || { echo BASELINE_CLONE_FAILED; exit 47; }
cd "$W/llama-upstream" && git checkout -q --detach "$UPSTREAM_PIN"
CMAKE_FLAGS=(-DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release)
if [ "$BACKEND" = "cuda" ]; then
  CMAKE_FLAGS+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121)
fi
PATH="$BUILD_PATH" cmake -B build -S . "${CMAKE_FLAGS[@]}" > "$W/baseline-cmake.log" 2>&1 || { tail -5 "$W/baseline-cmake.log"; echo BASELINE_BUILD_FAILED; exit 47; }
PATH="$BUILD_PATH" cmake --build build --target llama-server -j "$THREADS" > "$W/baseline-build.log" 2>&1 || { tail -5 "$W/baseline-build.log"; echo BASELINE_BUILD_FAILED; exit 47; }
mkdir -p "$W/repo/.deps/agentx-llama-build/bin" "$W/repo/.deps/agentx-llama-build/lib"
cp build/bin/llama-server "$W/repo/.deps/agentx-llama-build/bin/"
# Companion shared libs live beside the binary in build/bin and in build/lib;
# collect both so the server finds libllama-server-impl.so and friends.
for libdir in build/bin build/lib; do
  [ -d "$libdir" ] && cp "$libdir"/*.so* "$W/repo/.deps/agentx-llama-build/lib/" 2>/dev/null || true
done
"$W/repo/.deps/agentx-llama-build/bin/llama-server" --version || { echo BASELINE_SERVER_BROKEN; exit 47; }

# ---- link fixups for the pinned baseline (system ssl/crypto, $ORIGIN rpath) ----
if [ "$BACKEND" = "cuda" ]; then
  command -v patchelf >/dev/null || { mkdir -p "$W/pf" "$HOME/.local/bin"; \
    curl -fsSL --retry 3 https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-aarch64.tar.gz | tar xz -C "$W/pf"; \
    cp "$(find "$W/pf" -name patchelf -type f -print -quit)" "$HOME/.local/bin/patchelf"; export PATH="$HOME/.local/bin:$PATH"; }
  LDC=$(ldconfig -p 2>/dev/null); BLD="$W/repo/.deps/agentx-llama-build"
  for lib in libssl.so.3 libcrypto.so.3; do
    line=$(printf '%s\n' "$LDC" | grep -m1 " $lib ") || true; [ -n "$line" ] || continue
    cp "$(printf '%s' "$line" | awk '{print $NF}')" "$BLD/lib/$lib"
    patchelf --set-rpath '$ORIGIN' "$BLD/lib/$lib" || true
  done
  patchelf --set-rpath '$ORIGIN/../lib' "$BLD/bin/llama-server" 2>/dev/null || true
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

if [ "$MODE" = "smoke" ]; then
  # ---- smoke: llama vs skippy multi-turn probe ----
  cat > "$W/smoke.py" <<'PYEOF'
import concurrent.futures, json, sys, time, urllib.request, urllib.error
PORT, NCONV, TURNS, BLOCK, MAXTOK = (int(a) for a in sys.argv[1:6])
def convo(i, turn):
    filler = f"Session {i} stable context block. " * 220
    history = "\n".join(f"User: {filler[:BLOCK*k]} Question {i}-{k}: name a color." for k in range(1, turn+1))
    return [{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":history+f" Q{i}-{turn}: describe a sunset in one paragraph."}]
def ask(pair):
    i, turn = pair
    time.sleep(i)  # stagger concurrent conversation starts by 1s each
    body = json.dumps({"model":"m","messages":convo(i,turn),"max_tokens":MAXTOK,"temperature":0,"stream":False}).encode()
    t0=time.time(); err=None; pt=ct=comp=0
    # A freshly started server can refuse part of its first concurrent accept
    # burst while its listener finishes warming up; retry briefly instead of
    # failing the whole conversation.
    for attempt in range(6):
        try:
            r=urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"}),timeout=2400)
            out=json.load(r); u=out.get("usage",{}); d=u.get("prompt_tokens_details",{}) or {}
            pt, ct, comp = u.get("prompt_tokens",0), d.get("cached_tokens",0), u.get("completion_tokens",0)
            err=None; break
        except urllib.error.HTTPError as e: err=f"HTTP {e.code}: {e.read()[:120]}"; break
        except Exception as e:
            err=f"{type(e).__name__}: {e}"
            if attempt < 5: time.sleep(5)
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
  PORT=65100
  run_engine () {
    NAME="$1"; RT="$2"; shift 2
    PORT=$((PORT+1))
    "$@" > "$W/serve-$NAME.log" 2>&1 &
    PID=$!
    READY=0
    for i in $(seq 1 "$RT"); do
      sleep 5
      kill -0 $PID 2>/dev/null || break
      # Require the full API surface, not just a bound port: /v1/models
      # answers only once the server has finished loading and its slots exist.
      if curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        READY=$((READY+1)); [ "$READY" -ge 2 ] && break
      else
        READY=0
      fi
    done
    if [ "$READY" -lt 2 ]; then
      echo "-- $NAME: SERVER_START_FAILED; log tail:"
      tail -30 "$W/serve-$NAME.log" 2>/dev/null | sed 's/^/    /'
      cp "$W/serve-$NAME.log" "$JIANYANG_ARTIFACT_DIR/" 2>/dev/null || true
      kill $PID 2>/dev/null || true; sleep 8
      return 1
    fi
    echo "-- $NAME:"
    python3 "$W/smoke.py" "$PORT" "$PROBE_CONVERSATIONS" "$PROBE_TURNS" "$PROBE_BLOCK_CHARS" "$PROBE_MAX_TOKENS" || true
    kill $PID 2>/dev/null || true
    for w in $(seq 1 12); do kill -0 $PID 2>/dev/null || break; sleep 5; done
    kill -9 $PID 2>/dev/null || true
    sleep 5
  }
  echo "===== MODEL $MODEL_KEY ($BACKEND) ====="
  NGL=99; [ "$BACKEND" = "metal" ] && NGL=-1
  CTX=$((131072*PROBE_CONVERSATIONS))
  run_engine llama 120 "$W/repo/.deps/agentx-llama-build/bin/llama-server" --model "$MODEL" --alias m --host 127.0.0.1 --port $((PORT+1)) --parallel "$PROBE_CONVERSATIONS" --batch-size 2048 --ubatch-size 512 --n-gpu-layers 99 --cache-type-k f16 --cache-type-v f16 --flash-attn on --jinja --chat-template-kwargs '{"enable_thinking":false}' --ctx-size "$CTX" --cache-ram 8192 --no-kv-unified --threads "$THREADS" --threads-batch "$THREADS"
  run_engine skippy 180 "$W/repo/target/release/skippy" --output jsonl serve --model-path "$MODEL" --model-id m --bind-addr "127.0.0.1:$((PORT+1))" --startup-timeout-secs 900 --batch-size 2048 --microbatch-size 512 --generation-queue-capacity 256 --generation-admission-timeout-secs 1500 --n-gpu-layers "$NGL" --kv-cache-type-k f16 --kv-cache-type-v f16 --flash-attention enabled --speculative-strategy disabled --prefix-cache on --prefix-cache-ram 8GiB --kv-cache-disk off --threads "$THREADS" --threads-batch "$THREADS" --ctx-size 131072 --runtime-bundle "$RUNTIME_DIR"
  # ---- decode profile: streaming pass with per-token timestamps ----
  cat > "$W/profile.py" <<'PYEOF'
import json, sys, time, urllib.request
PORT, NCONV, MAXTOK = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
def convo(i):
    filler = f"Session {i} stable context block. " * 220
    return [{"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":filler + f" Q{i}: write a vivid paragraph about the ocean."}]
def profile_one(i):
    body = json.dumps({"model":"m","messages":convo(i),"max_tokens":MAXTOK,"temperature":0,"stream":True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    stamps = []
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=600)
    for line in r:
        if not line.startswith(b"data: "): continue
        payload = line[6:].strip()
        if payload == b"[DONE]": break
        try: chunk = json.loads(payload)
        except Exception: continue
        delta = chunk.get("choices",[{}])[0].get("delta",{})
        if delta.get("content"): stamps.append(time.time())
    ttft = stamps[0]-t0 if stamps else -1
    itls = [b-a for a,b in zip(stamps, stamps[1:])] if len(stamps)>1 else []
    return {"conv":i,"ttft_s":round(ttft,4),"tokens":len(stamps),
            "itl_ms_p50":round(sorted(itls)[len(itls)//2]*1000,2) if itls else -1,
            "itl_ms_mean":round(sum(itls)/len(itls)*1000,2) if itls else -1,
            "tok_s":round(len(stamps)/(stamps[-1]-t0),2) if len(stamps)>1 else -1}
import concurrent.futures
for nconc in ([1] if NCONV==1 else [1, NCONV]):
    with concurrent.futures.ThreadPoolExecutor(max_workers=nconc) as ex:
        results = list(ex.map(profile_one, range(nconc)))
    tok_s = sum(r["tok_s"] for r in results if r["tok_s"]>0)
    itl = [r["itl_ms_p50"] for r in results if r["itl_ms_p50"]>0]
    line = {"concurrency":nconc,"aggregate_tok_s":round(tok_s,1),
            "itl_p50_ms":round(sum(itl)/len(itl),2) if itl else -1,
            "per_conv":results}
    print("PROFILE " + json.dumps(line), flush=True)
PYEOF
  PORT=$((PORT+1))
  "$W/repo/.deps/agentx-llama-build/bin/llama-server" --model "$MODEL" --alias m --host 127.0.0.1 --port $PORT --parallel 4 --batch-size 2048 --ubatch-size 512 --n-gpu-layers 99 --cache-type-k f16 --cache-type-v f16 --flash-attn on --jinja --chat-template-kwargs '{"enable_thinking":false}' --ctx-size 524288 --cache-ram 8192 --no-kv-unified --threads "$THREADS" --threads-batch "$THREADS" > "$W/serve-llama-prof.log" 2>&1 &
  LP=$!
  for i in $(seq 1 120); do sleep 5; curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; done
  echo "-- llama profile:"
  python3 "$W/profile.py" "$PORT" "$PROBE_CONVERSATIONS" 256 || true
  kill $LP 2>/dev/null || true
  for w in $(seq 1 12); do kill -0 $LP 2>/dev/null || break; sleep 5; done
  kill -9 $LP 2>/dev/null || true
  sleep 5
  PORT=$((PORT+1))
  SKIPPY_TELEMETRY_STDERR=1 \
  "$W/repo/target/release/skippy" --output jsonl serve --model-path "$MODEL" --model-id m --bind-addr "127.0.0.1:$PORT" --startup-timeout-secs 900 --batch-size 2048 --microbatch-size 512 --generation-queue-capacity 256 --generation-admission-timeout-secs 1500 --n-gpu-layers 99 --kv-cache-type-k f16 --kv-cache-type-v f16 --flash-attention enabled --speculative-strategy disabled --prefix-cache on --prefix-cache-ram 8GiB --kv-cache-disk off --threads "$THREADS" --threads-batch "$THREADS" --ctx-size 131072 --runtime-bundle "$RUNTIME_DIR" > "$W/serve-skippy-prof.log" 2>&1 &
  SP=$!
  for i in $(seq 1 180); do sleep 5; curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; done
  echo "-- skippy profile:"
  python3 "$W/profile.py" "$PORT" "$PROBE_CONVERSATIONS" 256 || true
  kill $SP 2>/dev/null || true
  for w in $(seq 1 12); do kill -0 $SP 2>/dev/null || break; sleep 5; done
  kill -9 $SP 2>/dev/null || true
  sleep 5
  if [ "$SMOKE_SCOPE" != "quick" ]; then
  # ---- nsys CUDA timeline: one c1 prefill per engine ----
  echo "-- nsys profiles:"
  for ENGINE in llama skippy skippy-nocache; do
    PORT=$((PORT+1))
    EXTRA=""
    [ "$ENGINE" = "skippy-nocache" ] && EXTRA="--prefix-cache off"
    [ "$ENGINE" = "skippy" ] || [ "$ENGINE" = "skippy-nocache" ] && BIN="$W/repo/target/release/skippy" || BIN="$W/repo/.deps/agentx-llama-build/bin/llama-server"
    if [ "$ENGINE" = "llama" ]; then
      nsys profile -o "$W/nsys-$ENGINE" --trace=cuda,nvtx --force-overwrite=true \
        "$BIN" --model "$MODEL" --alias m --host 127.0.0.1 --port $PORT --parallel 1 --batch-size 2048 --ubatch-size 512 --n-gpu-layers 99 --cache-type-k f16 --cache-type-v f16 --flash-attn on --jinja --ctx-size 131072 --no-kv-unified --threads "$THREADS" --threads-batch "$THREADS" > /dev/null 2>&1 &
    else
      [ "$ENGINE" = "skippy" ] && PC="on" || PC="off"
      nsys profile -o "$W/nsys-$ENGINE" --trace=cuda,nvtx --force-overwrite=true \
        "$BIN" --output jsonl serve --model-path "$MODEL" --model-id m --bind-addr "127.0.0.1:$PORT" --startup-timeout-secs 900 --batch-size 2048 --microbatch-size 512 --generation-queue-capacity 256 --n-gpu-layers 99 --kv-cache-type-k f16 --kv-cache-type-v f16 --flash-attention enabled --speculative-strategy disabled --prefix-cache "$PC" --kv-cache-disk off --threads "$THREADS" --threads-batch "$THREADS" --ctx-size 131072 --runtime-bundle "$RUNTIME_DIR" > /dev/null 2>&1 &
    fi
    NPID=$!
    for i in $(seq 1 180); do sleep 5; curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; done
    kill -0 $NPID 2>/dev/null && python3 - "$PORT" <<'NSYSPY'
import json, sys, time, urllib.request
PORT = int(sys.argv[1])
msgs = [{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Session stable context block. "*220 + " Write one paragraph."}]
body = json.dumps({"model":"m","messages":msgs,"max_tokens":32,"temperature":0,"stream":False}).encode()
t0=time.time()
r=urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"}),timeout=300)
out=json.load(r)
print(f"  nsys {PORT}: ttft={out.get('choices',[{}])[0].get('created',0)} total={time.time()-t0:.3f}s tokens={out.get('usage',{}).get('completion_tokens',0)}")
NSYSPY
    kill $NPID 2>/dev/null || true; sleep 10; kill -9 $NPID 2>/dev/null || true; sleep 5
  done
  # Ship nsys reports
  for f in "$W"/nsys-*.nsys-rep; do [ -f "$f" ] && cp "$f" "$JIANYANG_ARTIFACT_DIR/" 2>/dev/null; done
  for f in "$W"/nsys-*.qdrep; do [ -f "$f" ] && cp "$f" "$JIANYANG_ARTIFACT_DIR/" 2>/dev/null; done

  # ---- skippy profile WITHOUT prefix cache: isolates checkpoint export cost ----
  PORT=$((PORT+1))
  "$W/repo/target/release/skippy" --output jsonl serve --model-path "$MODEL" --model-id m --bind-addr "127.0.0.1:$PORT" --startup-timeout-secs 900 --batch-size 2048 --microbatch-size 512 --generation-queue-capacity 256 --generation-admission-timeout-secs 1500 --n-gpu-layers 99 --kv-cache-type-k f16 --kv-cache-type-v f16 --flash-attention enabled --speculative-strategy disabled --prefix-cache off --kv-cache-disk off --threads "$THREADS" --threads-batch "$THREADS" --ctx-size 131072 --runtime-bundle "$RUNTIME_DIR" > "$W/serve-skippy-nocache.log" 2>&1 &
  NP=$!
  for i in $(seq 1 180); do sleep 5; curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; done
  kill -0 $NP 2>/dev/null && echo "-- skippy nocache profile:" && python3 "$W/profile.py" "$PORT" "$PROBE_CONVERSATIONS" 256 || echo "-- skippy nocache: SERVER_DIED"
  kill $NP 2>/dev/null || true
  for w in $(seq 1 12); do kill -0 $NP 2>/dev/null || break; sleep 5; done
  kill -9 $NP 2>/dev/null || true
  sleep 5
  # ---- prefill chunk sweep: TTFT vs chunk size ----
  echo "-- prefill chunk sweep:"
  for CHUNK in 64 256 512 2048; do
    PORT=$((PORT+1))
    "$W/repo/target/release/skippy" --output jsonl serve --model-path "$MODEL" --model-id m --bind-addr "127.0.0.1:$PORT" --startup-timeout-secs 900 --batch-size 2048 --microbatch-size 512 --generation-queue-capacity 256 --generation-admission-timeout-secs 1500 --n-gpu-layers 99 --kv-cache-type-k f16 --kv-cache-type-v f16 --flash-attention enabled --speculative-strategy disabled --prefix-cache on --prefix-cache-ram 8GiB --kv-cache-disk off --threads "$THREADS" --threads-batch "$THREADS" --ctx-size 131072 --prefill-chunk-size "$CHUNK" --prefill-chunk-policy fixed --runtime-bundle "$RUNTIME_DIR" > "$W/serve-sweep-$CHUNK.log" 2>&1 &
    SWPID=$!
    for i in $(seq 1 180); do sleep 5; curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; done
    if ! kill -0 $SWPID 2>/dev/null; then
      echo "  SWEEP chunk=$CHUNK SERVER_DIED"
      continue
    fi
    python3 - "$PORT" "$CHUNK" <<'SWEEPPY'
import json, sys, time, urllib.request
PORT, CHUNK = int(sys.argv[1]), sys.argv[2]
filler = "Session sweep stable context block. " * 220
msgs = [{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":filler + " Write one paragraph about mountains."}]
body = json.dumps({"model":"m","messages":msgs,"max_tokens":64,"temperature":0,"stream":True}).encode()
ttfts = []
for trial in range(3):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t0 = time.time(); first = None; n = 0
    r = urllib.request.urlopen(req, timeout=120)
    for line in r:
        if not line.startswith(b"data: "): continue
        p = line[6:].strip()
        if p == b"[DONE]": break
        try: c = json.loads(p)
        except Exception: continue
        if c.get("choices",[{}])[0].get("delta",{}).get("content"):
            if first is None: first = time.time()
            n += 1
    if first: ttfts.append(first - t0)
avg = sum(ttfts)/len(ttfts) if ttfts else -1
print(f"  SWEEP chunk={CHUNK} ttft_avg={avg:.3f}s (trials={len(ttfts)})", flush=True)
SWEEPPY
    kill $SWPID 2>/dev/null || true
    for w in $(seq 1 12); do kill -0 $SWPID 2>/dev/null || break; sleep 5; done
    kill -9 $SWPID 2>/dev/null || true
    sleep 5
  done
  fi
  cp "$W"/serve-*.log "$JIANYANG_ARTIFACT_DIR/" 2>/dev/null || true
  echo SMOKE_DONE
  exit 0
fi

# ---- comparison: full provenance-pinned campaign ----
cd "$W/repo"
echo "== client env"
just agentx-setup > "$W/agentx-setup.log" 2>&1 || { tail -5 "$W/agentx-setup.log"; echo CLIENT_ENV_FAILED; exit 48; }
CLIENT="$W/repo/skippy/evals/agentx/.venv/bin/agentx"
[ -x "$CLIENT" ] || { echo NO_AGENTX_CLIENT; exit 48; }
find "$W/repo/skippy/evals/agentx/.venv/lib" -name '*.pth' -type f 2>/dev/null | while read -r p; do grep -q "^import" "$p" && rm -f "$p" && echo "removed: $p"; done || true

export HF_HUB_CACHE="$W/hf/hub"
TD="$HF_HUB_CACHE/models--${TOK_REPO//\//--}/snapshots/$TOK_REV"
mkdir -p "$TD"
for f in tokenizer.json tokenizer_config.json special_tokens_map.json config.json generation_config.json; do
  [ -s "$TD/$f" ] || curl -sfL "https://huggingface.co/$TOK_REPO/resolve/$TOK_REV/$f" -o "$TD/$f" || true
done

CORPUS_PATH="$HUB/datasets--${CORPUS_REPO//\//--}/snapshots/$CORPUS_REV/traces.jsonl"
if [ ! -f "$CORPUS_PATH" ]; then
  mkdir -p "$(dirname "$CORPUS_PATH")" 2>/dev/null || { mkdir -p "$W/models"; CORPUS_PATH="$W/models/traces.jsonl"; }
  curl -fL --retry 4 --max-time 2400 "https://huggingface.co/datasets/$CORPUS_REPO/resolve/$CORPUS_REV/traces.jsonl" -o "$CORPUS_PATH" || { echo CORPUS_FAILED; exit 49; }
fi
printf '%s  %s\n' "$CORPUS_SHA" "$CORPUS_PATH" | shasum -a256 -c -

MANIFEST=$(find "$RUNTIME_DIR" -name manifest.json -print -quit)
DIGEST=$(python3 - "$MANIFEST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["build"]["llama_patch_digest"])
PY
)
python3 - <<PYEOF
import json,hashlib
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
w="$W"; repo="$W/repo"; client="$CLIENT"
tools={"backend":"$BACKEND","product_source_commit":"$PRODUCT_COMMIT","native_source_commit":"$NATIVE_COMMIT","native_patch_digest":"$DIGEST","runtime":{"path":"$RUNTIME_DIR","sha256":""},"skippy":{"path":repo+"/target/release/skippy","sha256":sha(repo+"/target/release/skippy")},"llama":{"path":repo+"/.deps/agentx-llama-build/bin/llama-server","sha256":sha(repo+"/.deps/agentx-llama-build/bin/llama-server")},"standard_runtime":{"path":repo+"/.deps/agentx-llama-build","sha256":""},"standard_upstream":"$UPSTREAM_PIN","client":{"path":client,"sha256":sha(client)},"client_environment":{"path":"/".join(client.split("/")[:-2]),"sha256":""},"client_lock":{"path":repo+"/skippy/evals/agentx/uv.lock","sha256":sha(repo+"/skippy/evals/agentx/uv.lock")},"tokenizer":{"path":"$TD","sha256":""},"threads":int("$THREADS"),"diagnostic_generation_lanes":None,"diagnostic_cpu":False,"diagnostic_telemetry":False,"cell_seconds":10200}
req={"schema_version":1,"campaign":"$CAMPAIGN","workspace":repo,"model_key":"$MODEL_KEY","model":"$MODEL","corpus":{"path":"$CORPUS_PATH","sha256":"$CORPUS_SHA"},"tools":tools}
json.dump(req,open(w+'/request.paths.json','w'),indent=1)
print('request.paths.json written')
PYEOF
export JIANYANG_COMMIT="$PRODUCT_COMMIT"
cargo xtool automation agentx pin --input "$W/request.paths.json" --output "$W/request.pinned.json" 2>&1 | tail -3
( while sleep 120; do ls "$W/campaign-out" >> "$JIANYANG_ARTIFACT_DIR/live-progress.log" 2>/dev/null; done ) &
OBS=$!
CMP_RC=0
cargo xtool automation agentx comparison --input "$W/request.pinned.json" --output "$W/campaign-out" 2>&1 | tail -10 || CMP_RC=1
kill $OBS 2>/dev/null || true
tar czf "$JIANYANG_ARTIFACT_DIR/evidence.tgz" -C "$W" campaign-out request.paths.json request.pinned.json 2>/dev/null || tar czf "$JIANYANG_ARTIFACT_DIR/evidence.tgz" -C "$W" campaign-out
shasum -a256 "$JIANYANG_ARTIFACT_DIR/evidence.tgz" | awk '{print $1"  evidence.tgz"}' > "$JIANYANG_ARTIFACT_DIR/evidence.tgz.manifest"
[ "$CMP_RC" = "0" ] && echo CAMPAIGN_OK || { echo "CAMPAIGN_FAILED rc=$CMP_RC"; exit 1; }
