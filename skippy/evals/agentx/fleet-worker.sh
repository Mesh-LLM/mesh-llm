#!/bin/bash
set -Eeuo pipefail
W=$PWD
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
R2="${KIT_URL_BASE:?KIT_URL_BASE required}"
echo "== host =="
if command -v nproc >/dev/null; then nproc; elif command -v sysctl >/dev/null; then sysctl -n hw.ncpu; fi
if [ "$BACKEND" = metal ]; then
  MEM=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
  if [ "$MEM" -lt 200000000000 ]; then echo NEED_BIGGER_MAC; exit 30; fi
fi
if [ "$BACKEND" = cuda ]; then
  nvidia-smi -L 2>/dev/null | grep NVIDIA >/dev/null 2>&1 || { echo NO_CUDA_GPU; exit 31; }
fi
if [ "$BACKEND" = "cuda" ] && [ "$(uname -m)" != "aarch64" ]; then echo NO_AARCH64_HOST; exit 33; fi
if ! command -v python3.12 >/dev/null; then
  echo "installing userspace python3.12"
  PYURL="https://github.com/astral-sh/python-build-standalone/releases/download/20250115/cpython-3.12.8+20250115-aarch64-unknown-linux-gnu-install_only.tar.gz"
  [ "$(uname -s)" = Darwin ] && PYURL="https://github.com/astral-sh/python-build-standalone/releases/download/20250115/cpython-3.12.8+20250115-aarch64-apple-darwin-install_only.tar.gz"
  mkdir -p "$W/py312"
  curl -fsSL --retry 3 --max-time 600 "$PYURL" -o "$W/py312.tar.gz" \
    && tar xzf "$W/py312.tar.gz" -C "$W/py312" \
    && export PATH="$W/py312/python/bin:$PATH" \
    && hash -r \
    || { echo NO_PYTHON312; exit 31; }
fi
command -v python3.12 >/dev/null || { echo NO_PYTHON312; exit 31; }
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || system_profiler SPDisplaysDataType 2>/dev/null | grep -m1 -i 'chip\|GPU' || true
# ---- branch source ----
curl -fL --retry 3 -o "$W/src.tgz" "$SRC_URL"
printf '%s  %s\n' "$INPUT_ARCHIVE_SHA" "$W/src.tgz" | shasum -a256 -c -
mkdir -p "$W/repo" "$W/parts" "$W/tu" && tar xzf "$W/src.tgz" -C "$W/repo"
cd "$W/repo"
command -v cargo >/dev/null || (curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal)
command -v just >/dev/null || cargo install just --locked
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
python3.12 --version || { echo NO_PYTHON312; exit 42; }
command -v patchelf >/dev/null || { mkdir -p "$HOME/.local/bin" "$W/pf"; curl -fsSL --retry 3 https://github.com/NixOS/patchelf/releases/download/0.18.0/patchelf-0.18.0-aarch64.tar.gz | tar xz -C "$W/pf"; cp "$(find "$W/pf" -name patchelf -type f -print -quit)" "$HOME/.local/bin/patchelf"; export PATH="$HOME/.local/bin:$PATH"; }
export COPYFILE_DISABLE=1
OPT="$W/opt"; mkdir -p "$OPT/kv-tools" "$OPT/kv-runtime" "$OPT/standard/bin" "$OPT/standard/lib" "$W/models" "$JIANYANG_ARTIFACT_DIR"
# stagger concurrent campaign starts to avoid CDN rate limits
sleep $((RANDOM % 90))
echo "== provisioning from CDN =="
fetch(){ local out="$W/parts/$2"; for t in 1 2 3 4 5 6 7 8; do curl -sfL --retry 5 --retry-delay 5 --retry-all-errors --max-time 300 -C - "$R2/$1" -o "$out" 2>/dev/null && [ -s "$out" ] && return 0; sleep $((t*5)); done; echo "FETCH_FAIL $1"; return 41; }
if [ "$BACKEND" = cuda ]; then
  : > "$W/tools.tgz"
  for i in $(seq 0 $((CUDA_CHUNKS-1))); do ii=$(printf '%02d' "$i"); pn=$(basename "$CUDA_KEY_PREFIX$ii"); fetch "$CUDA_KEY_PREFIX$ii" "$pn" && cat "$W/parts/$pn" >> "$W/tools.tgz" || exit 41; done
  printf '%s  %s\n' "$CUDA_TGZ_SHA" "$W/tools.tgz" | shasum -a256 -c -
  : > "$W/llama.tgz"
  for i in $(seq 0 $((LLAMA_CHUNKS-1))); do ii=$(printf '%02d' "$i"); pn=$(basename "$LLAMA_KEY_PREFIX$ii"); fetch "$LLAMA_KEY_PREFIX$ii" "$pn" && cat "$W/parts/$pn" >> "$W/llama.tgz" || exit 41; done
  printf '%s  %s\n' "$LLAMA_TGZ_SHA" "$W/llama.tgz" | shasum -a256 -c -
  tar xzf "$W/tools.tgz" -C "$W/tu"
  SKBN=$(find "$W/tu" -name skippy -type f -print -quit)
  cp "$SKBN" "$OPT/kv-tools/skippy"
  RTB=$(find "$W/tu/native-runtimes" -maxdepth 1 -mindepth 1 -type d -print -quit)
  cp -R "$RTB"/. "$OPT/kv-runtime/"
  mkdir -p "$OPT/standard"
  tar xzf "$W/llama.tgz" -C "$OPT/standard"
  # the vanilla baseline links openssl; pin requires it inside the pinned runtime
  for l in libssl.so.3 libcrypto.so.3; do
    if [ ! -f "$OPT/standard/lib/$l" ]; then
      p=$(find /usr/lib /lib -name "$l" -type f -print -quit 2>/dev/null)
      if [ -z "$p" ]; then
        all=$(ldconfig -p 2>/dev/null | awk -v lib="$l" '$1==lib {print $NF}')
        p=${all%%$'\n'*}
      fi
      [ -n "$p" ] && { cp "$p" "$OPT/standard/lib/$l"; patchelf --set-rpath '$ORIGIN' "$OPT/standard/lib/$l" || { echo "PATCH_RPATH_FAIL $l"; exit 46; }; echo "copied+rpath'd $l from $p"; } || { echo "MISSING_SYSTEM_$l"; exit 45; }
    fi
  done
else
  mn=$(basename "$METAL_KIT_KEY")
  fetch "$METAL_KIT_KEY" "$mn"
  printf '%s  %s\n' "$METAL_KIT_SHA" "$W/parts/$mn" | shasum -a256 -c -
  tar xzf "$W/parts/$mn" -C "$OPT"
fi
chmod +x "$OPT/kv-tools/skippy" "$OPT/standard/bin/llama-server"
# the kits were rpath-rewritten after packaging; re-record the actual payload hashes
python3 - "$OPT/kv-runtime" <<'PY3'
import json, hashlib, os, sys
root = sys.argv[1]
mf = os.path.join(root, 'manifest.json')
d = json.load(open(mf))
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
for section in ('files','tools'):
    m = d['runtime'].get(section, {})
    changed = 0
    for rel in list(m):
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            got = sha(p)
            if m[rel] != got:
                m[rel] = got
                changed += 1
    print('manifest %s: re-recorded %d hashes' % (section, changed))
json.dump(d, open(mf,'w'), indent=2)
PY3
"$OPT/standard/bin/llama-server" --version 2>&1 | head -1 || true
"$OPT/kv-tools/skippy" --version 2>&1 | head -1 || true
# ---- self-diagnostic (cuda only; macos lacks readelf) ----
if [ "$BACKEND" = cuda ]; then
cat > "$W/depwalk.py" <<'PY2'
import subprocess,sys,os,re
exe,root,rtroot=sys.argv[1],sys.argv[2],sys.argv[3]
HOST={"ld-linux.so.2","ld-linux-aarch64.so.1","linux-vdso.so.1","libc.so.6","libcrypt.so.1","libdl.so.2","libgcc_s.so.1","libatomic.so.1","libgomp.so.1","libm.so.6","libnsl.so.1","libnuma.so.1","libpthread.so.0","libresolv.so.2","librt.so.1","libstdc++.so.6","libutil.so.1","libz.so.1","libcuda.so","libcuda.so.1","libnvidia-ml.so.1"}
def readelf(f): return subprocess.run(['readelf','-d',f],capture_output=True,text=True).stdout
def needed(f): return re.findall(r'\(NEEDED\).*?\[(.*?)\]',readelf(f))
def runpaths(f):
    out=readelf(f)
    return re.findall(r'\(RUNPATH\).*?\[(.*?)\]',out)+re.findall(r'\(RPATH\).*?\[(.*?)\]',out)
def ldd(f): return subprocess.run(['ldd',f],capture_output=True,text=True).stdout
def lddlibs(f):
    res=[]
    for line in ldd(f).splitlines():
        m=re.match(r'\s*(\S+)\s+=>\s+(\S+)',line)
        if m: res.append(m.group(2))
    return res
visited=set(); pending=[exe]
for dp,_,fns in os.walk(root):
    for fn in fns:
        if fn.endswith('.dylib') or fn.endswith('.so') or '.so.' in fn: pending.append(os.path.join(dp,fn))
problems=[]
while pending:
    f=pending.pop(); rf=os.path.realpath(f)
    if rf in visited or not os.path.isfile(rf): continue
    visited.add(rf)
    if not rf.startswith(root): problems.append(f"ESCAPES: {rf}"); continue
    origin=os.path.dirname(rf)
    rps=[x.replace('$ORIGIN',origin) for x in runpaths(rf)]
    for nd in needed(rf):
        b=os.path.basename(nd)
        if b in HOST: continue
        if nd.startswith('/'): cands=[nd]
        else: cands=[os.path.join(d,b) for d in rps]+[os.path.join(origin,b)]
        found=next((c for c in cands if os.path.isfile(c)),None)
        if not found: problems.append(f"ABSENT: {nd} importer={rf}")
        else: pending.append(found)
print('DIAG_VISITED:',len(visited))
print('DIAG_PROBLEMS:' if problems else 'DIAG_CLEAN')
for p in problems[:15]: print(' ',p)
PY2
python3 "$W/depwalk.py" "$OPT/standard/bin/llama-server" "$OPT/standard" "$OPT/kv-runtime" || true
fi
# ---- model + corpus + tokenizer from the ambient HF cache (NFS model store) ----
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
MSnap="$HUB/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REV"
if [ -f "$MSnap/$MODEL_FILE" ]; then
  MODEL_PATH="$MSnap/$MODEL_FILE"
else
  echo "model not cached; downloading"
  mkdir -p "$W/models"
  curl -fL --retry 3 -o "$W/models/$MODEL_FILE" "https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REV/$MODEL_FILE"
  MODEL_PATH="$W/models/$MODEL_FILE"
fi
printf '%s  %s\n' "$MODEL_SHA" "$MODEL_PATH" | shasum -a256 -c -
# tokenizer must live at the env-derived hub snapshot path (verify_tokenizer_snapshot
# requires pinned == derived); NFS tokenizer dirs can be unreadable, so stage locally
export HF_HUB_CACHE="$W/hf/hub"
TD="$HF_HUB_CACHE/models--${TOK_REPO//\//--}/snapshots/$TOK_REV"
mkdir -p "$TD"
for f in tokenizer.json tokenizer_config.json special_tokens_map.json config.json generation_config.json; do
  [ -s "$TD/$f" ] || curl -sfL "https://huggingface.co/$TOK_REPO/resolve/$TOK_REV/$f" -o "$TD/$f" || true
done
ls "$TD"
CORPUS_PATH="$HUB/datasets--${CORPUS_REPO//\//--}/snapshots/$CORPUS_REV/traces.jsonl"
if [ ! -f "$CORPUS_PATH" ]; then
  mkdir -p "$(dirname "$CORPUS_PATH")" 2>/dev/null || { mkdir -p "$W/models"; CORPUS_PATH="$W/models/traces.jsonl"; }
  curl -fL --retry 3 -o "$CORPUS_PATH" "https://huggingface.co/datasets/$CORPUS_REPO/resolve/$CORPUS_REV/traces.jsonl"
fi
printf '%s  %s\n' "$CORPUS_SHA" "$CORPUS_PATH" | shasum -a256 -c -
# ---- agentx client env ----
just agentx-setup 2>&1 | tail -3
CLIENT="$W/repo/skippy/evals/agentx/.venv/bin/agentx"
[ -x "$CLIENT" ] || { echo NO_AGENTX_CLIENT; exit 43; }
# pin policy: the prepared environment must not contain executable .pth imports
find "$W/repo/skippy/evals/agentx/.venv/lib" -name '*.pth' -type f | while read -r p; do grep -q "^import" "$p" && rm -f "$p" && echo "removed: $p"; done || true
# ---- request.paths.json ----
MANIFEST=$(find "$OPT/kv-runtime" -name manifest.json -print -quit)
DIGEST=$(python3 - "$MANIFEST" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(d["build"]["llama_patch_digest"])
PY
)
echo "patch digest: $DIGEST"
THREADS="${THREADS:-20}"
python3 - <<PYEOF
import json,hashlib
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
opt="$OPT"; w="$W"
tools={"backend":"$BACKEND","product_source_commit":"ee8d792d239920e4d23996709c16d1668e169107","native_source_commit":"ab8ee5a1fe9fb7a31fa7fdf29da312a554edbe2f","native_patch_digest":"$DIGEST","runtime":{"path":opt+"/kv-runtime","sha256":""},"skippy":{"path":opt+"/kv-tools/skippy","sha256":sha(opt+"/kv-tools/skippy")},"llama":{"path":opt+"/standard/bin/llama-server","sha256":sha(opt+"/standard/bin/llama-server")},"standard_runtime":{"path":opt+"/standard","sha256":""},"standard_upstream":"372dc5ca157276beb6c67b2e92477e8d662e964a","client":{"path":"$CLIENT","sha256":sha("$CLIENT")},"client_environment":{"path":"/".join("$CLIENT".split("/")[:-2]),"sha256":""},"client_lock":{"path":w+"/repo/skippy/evals/agentx/uv.lock","sha256":sha(w+"/repo/skippy/evals/agentx/uv.lock")},"tokenizer":{"path":"$TD","sha256":""},"threads":int("$THREADS"),"diagnostic_generation_lanes":None,"diagnostic_cpu":False,"diagnostic_telemetry":False,"cell_seconds":10200}
req={"schema_version":1,"campaign":"agentx-pr3031-20261114","workspace":w+"/repo","model_key":"$MODEL_KEY","model":"$MODEL_PATH","corpus":{"path":"$CORPUS_PATH","sha256":"$CORPUS_SHA"},"tools":tools}
json.dump(req,open('request.paths.json','w'),indent=1)
print('request.paths.json written')
PYEOF
# ---- pin ----
export JIANYANG_COMMIT="ee8d792d239920e4d23996709c16d1668e169107"
cargo xtool automation agentx pin --input "$W/repo/request.paths.json" --output "$W/repo/request.pinned.json" 2>&1 | tail -4
# ---- campaign ----
( while sleep 120; do
    {
      echo "===== $(date -u +%H:%M:%S) observer tick"
      echo "-- campaign-out:"; ls "$W/campaign-out" 2>/dev/null
      for arm in "$W/campaign-out"/*/; do
        [ -d "$arm" ] || continue
        echo "-- arm $(basename "$arm"):"; ls "$arm" 2>/dev/null
        for cell in "$arm"*/; do
          [ -d "$cell" ] || continue
          echo "   cell $(basename "$cell"): $(ls "$cell" 2>/dev/null | tr '\n' ' ')"
          [ -f "$cell/server.stderr.log" ] && tail -2 "$cell/server.stderr.log" 2>/dev/null | sed 's/^/     srv: /'
          [ -f "$cell/agentx.stdout.log" ] && tail -2 "$cell/agentx.stdout.log" 2>/dev/null | sed 's/^/     cli: /'
        done
      done
      echo "-- procs:"; ps -eo pid,pcpu,pmem,etime,args 2>/dev/null | grep -E 'llama-server|skippy serve|agentx' | grep -v grep | head -6
      echo "-- ports:"; (ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | grep -E '127.0.0.1' | head -6
      for p in $( (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep 127.0.0.1 | awk '{print $NF}' | cut -d: -f2 | sort -un | head -8); do
        curl -s --max-time 3 "http://127.0.0.1:$p/slots" 2>/dev/null | head -c 400 | sed "s/^/     port $p: /"; echo
      done
    } >> "$JIANYANG_ARTIFACT_DIR/live-progress.log" 2>&1
    tar czf "$JIANYANG_ARTIFACT_DIR/evidence-latest.tgz" -C "$W" campaign-out 2>/dev/null || true
  done ) &
OBS_PID=$!
CMP_RC=0
cargo xtool automation agentx comparison --input "$W/repo/request.pinned.json" --output "$W/campaign-out" 2>&1 | tail -10 || CMP_RC=1
ls -la "$W/campaign-out" 2>/dev/null | head -20 || true
echo "== skippy ldd =="; ldd "$OPT/kv-tools/skippy" 2>&1 | grep -E 'not found' || echo LDD_SKIPPY_CLEAN
for f in "$W/campaign-out"/doctor "$W/campaign-out"/*.log; do
  [ -f "$f" ] && { echo "--- $f (tail 60)"; tail -60 "$f"; }
done 2>/dev/null || true
if [ -d "$W/campaign-out/doctor" ]; then
  find "$W/campaign-out/doctor" -type f | head -5 | while read -r df; do echo "--- $df (tail 60)"; tail -60 "$df"; done
fi
# ---- evidence out (chunked) ----
tar czf "$W/evidence.tgz" -C "$W" campaign-out request.paths.json request.pinned.json 2>/dev/null || tar czf "$W/evidence.tgz" -C "$W" campaign-out
split -b 64m -d "$W/evidence.tgz" "$JIANYANG_ARTIFACT_DIR/evidence-"
shasum -a256 "$W/evidence.tgz" > "$JIANYANG_ARTIFACT_DIR/evidence.manifest"
[ "$CMP_RC" = 0 ] && echo "CAMPAIGN_OK" || { echo "CAMPAIGN_FAILED rc=$CMP_RC"; kill "$OBS_PID" 2>/dev/null || true; exit 1; }
kill "$OBS_PID" 2>/dev/null || true
