#!/usr/bin/env bash
# Fast fleet controller: starts peers concurrently; readiness waiting is opt-in.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${MESH_FLEET_ENV:-$SCRIPT_DIR/mesh-fleet.env}"
NODE_HELPER="$SCRIPT_DIR/mesh-fleet-node.ps1"
CONNECTION_WRAPPER="${MESH_FLEET_CONNECTION_WRAPPER:-/Users/luciano/Downloads/scripts/connect_machine.sh}"
KEY_DEFAULT="$HOME/.ssh/l2p2p_win"

usage(){ cat <<'EOF'
Usage: mesh-fleet.sh [--env FILE] [--wait-ready] <plan|preflight|start|status|stop|logs> [node]

Fast behavior:
  start launches all enabled join/client nodes concurrently immediately after the
  Mac seed token is available. It checks only process creation by default.
  Add --wait-ready only when you want HTTP management-API readiness polling.
EOF
}
WAIT_READY=0
while [[ $# -gt 0 ]]; do case "$1" in --env) ENV_FILE="$2";shift 2;;--wait-ready)WAIT_READY=1;shift;;-h|--help)usage;exit 0;;*)break;;esac;done
CMD="${1:-}"; TARGET="${2:-}"
[[ -n "$CMD" ]] || { usage; exit 2; }
[[ -f "$ENV_FILE" ]] || { echo "missing fleet config: $ENV_FILE" >&2; exit 2; }
# shellcheck source=/dev/null
source "$ENV_FILE"
# Private Hugging Face credential fallback. The token file is owner-only and is
# deliberately kept outside fleet configuration / version control.
if [[ -z "${HF_TOKEN:-}" ]]; then
  _mesh_fleet_hf_token_file="${MESH_FLEET_HF_TOKEN_FILE:-$HOME/.mesh-llm/huggingface.token}"
  if [[ -r "$_mesh_fleet_hf_token_file" ]]; then
    export HF_TOKEN="$(<"$_mesh_fleet_hf_token_file")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  fi
fi
: "${MESH_NAME:?MESH_NAME required}"; : "${MESH_MODEL:?MESH_MODEL required}"; : "${MESH_VERSION:=0.71.0}"; : "${MESH_WAIT_SECONDS:=15}"; : "${MESH_LAN:=1}"; : "${MESH_SPLIT:=1}"; : "${MESH_ARTIFACT_TRANSFER:=trusted}"
NODE_NAMES=(mac win win2 win3)
node_var(){ local n="$1" f="$2" upper v; upper="$(printf '%s' "$n" | tr '[:lower:]' '[:upper:]')"; v="NODE_${upper}_${f}"; printf '%s' "${!v:-}"; }
enabled(){ [[ "$(node_var "$1" ENABLED)" == "1" ]]; }
json_for(){
 local n="$1" role="$(node_var "$n" ROLE)"; [[ -n "$role" ]]||role=join
 python3 - "$n" "$role" <<'PY'
import json,os,sys
n,role=sys.argv[1:]
def g(k,default=''):
    return os.getenv(f'NODE_{n.upper()}_{k}', os.getenv(f'MESH_{k}',default))
b=lambda x: str(x).lower() in ('1','true','yes','on')
c={
 'node_name':n,'role':role,'mesh_name':os.environ['MESH_NAME'],
 'model':g('MODEL',os.environ['MESH_MODEL']),'expected_version':os.environ.get('MESH_VERSION','0.71.0'),
 'bin':g('BIN'),'console_port':int(g('CONSOLE','3131')),'api_port':int(g('API_PORT','9337')),
 'bind_ip':g('BIND_IP'),'bind_port':g('BIND_PORT'),'lan':b(os.environ.get('MESH_LAN','1')),
 'split':b(os.environ.get('MESH_SPLIT','1')),'ctx_size':g('CTX_SIZE',os.environ.get('MESH_CTX_SIZE','')),
 'max_vram':g('MAX_VRAM'),'device':g('DEVICE'),'llama_flavor':g('LLAMA_FLAVOR'),
 'tensor_split':g('TENSOR_SPLIT'),'artifact_transfer':g('ARTIFACT_TRANSFER',os.environ.get('MESH_ARTIFACT_TRANSFER','')),
 'kv_cache_policy':g('KV_CACHE_POLICY',os.environ.get('MESH_KV_CACHE_POLICY','saver')),
 'kv_offload':g('KV_OFFLOAD',os.environ.get('MESH_KV_OFFLOAD','true')),
 'cache_type_k':g('CACHE_TYPE_K',os.environ.get('MESH_CACHE_TYPE_K','q4_0')),
 'cache_type_v':g('CACHE_TYPE_V',os.environ.get('MESH_CACHE_TYPE_V','q4_0')),
 'gpu_layers':g('GPU_LAYERS',os.environ.get('MESH_GPU_LAYERS','auto')),
 'mmap':g('MMAP',os.environ.get('MESH_MMAP','true')),'web_ui':b(g('WEB_UI','0')),'extra_args':g('EXTRA_ARGS'),
 'wait_seconds':int(os.environ.get('MESH_WAIT_SECONDS','15'))}
print(json.dumps(c,separators=(',',':')))
PY
}
remote_run(){
 local n="$1" ps_action="$2" payload="$3" token="${4:-}" waitflag="${5:-}"
 local b64 helper_b64 remote_helper cmd mode
 b64=$(printf '%s' "$payload" | base64 | tr -d '\n')
 helper_b64=$(base64 < "$NODE_HELPER" | tr -d '\n')
 remote_helper='$env:LOCALAPPDATA\\MeshLLM\\mesh-fleet-node.ps1'
 # Every remote invocation refreshes the small helper locally. This avoids a
 # separate slow deployment phase and keeps Windows helper version in sync.
 local install="\$d=Join-Path \$env:LOCALAPPDATA 'MeshLLM'; New-Item -ItemType Directory -Force -Path \$d|Out-Null; [IO.File]::WriteAllBytes((Join-Path \$d 'mesh-fleet-node.ps1'),[Convert]::FromBase64String('$helper_b64'))"
 cmd="powershell -NoProfile -ExecutionPolicy Bypass -Command \"$install; & (Join-Path \$env:LOCALAPPDATA 'MeshLLM\\mesh-fleet-node.ps1') -Action $ps_action -ConfigBase64 $b64 $waitflag\""
 mode="$(node_var "$n" SSH_MODE)"
 if [[ "$n" == mac ]]; then
   echo "Mac non-start action is not implemented by the fleet PowerShell helper; use scripts/mesh-private-join.sh status or its seed log." >&2
   return 0
 elif [[ "$mode" == wrapper:* ]]; then
   local profile="${mode#wrapper:}"
   if [[ -n "$token" ]]; then printf '%s' "$token" | bash "$CONNECTION_WRAPPER" "$profile" -o ConnectTimeout=15 "$cmd"; else bash "$CONNECTION_WRAPPER" "$profile" -o ConnectTimeout=15 "$cmd"; fi
 else
   local host="$(node_var "$n" SSH_HOST)" user="$(node_var "$n" SSH_USER)" key="$(node_var "$n" SSH_KEY)"; key="${key:-$KEY_DEFAULT}"
   [[ -n "$host" && -n "$user" ]] || { echo "missing direct SSH host/user for $n" >&2; return 2; }
   if [[ -n "$token" ]]; then printf '%s' "$token" | ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o PreferredAuthentications=publickey -i "$key" "$user@$host" "$cmd"; else ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o PreferredAuthentications=publickey -i "$key" "$user@$host" "$cmd"; fi
 fi
}
# Mac seed uses its proven local private helper so fleet token comes from API/token workflow.
seed_start(){
 local macbin;macbin="$(node_var mac BIN)";[[ -x "$macbin" ]] || { echo "Mac binary unavailable: $macbin" >&2; return 2; }
 local args=(seed --name "$MESH_NAME" --model "$(node_var mac MODEL)")
 [[ -n "$(node_var mac CONSOLE)" ]]&&args+=(--console "$(node_var mac CONSOLE)")
 [[ -n "$(node_var mac API_PORT)" ]]&&args+=(--port "$(node_var mac API_PORT)")
 [[ -n "$(node_var mac BIND_PORT)" ]]&&args+=(--bind-port "$(node_var mac BIND_PORT)")
 [[ "$MESH_SPLIT" == 1 ]]&&args+=(--split)
 [[ "$MESH_LAN" == 1 ]]&&args+=(--lan)
 [[ -n "${MESH_CTX_SIZE:-}" ]]&&args+=(--ctx-size "$MESH_CTX_SIZE")
 [[ -n "$(node_var mac MAX_VRAM)" ]]&&args+=(--max-vram "$(node_var mac MAX_VRAM)")
 [[ -n "$(node_var mac DEVICE)" ]]&&args+=(--device "$(node_var mac DEVICE)")
 [[ -n "$(node_var mac LLAMA_FLAVOR)" ]]&&args+=(--llama-flavor "$(node_var mac LLAMA_FLAVOR)")
 [[ -n "$(node_var mac TENSOR_SPLIT)" ]]&&args+=(--tensor-split "$(node_var mac TENSOR_SPLIT)")
 [[ -n "$MESH_ARTIFACT_TRANSFER" ]]&&args+=(--artifact-transfer "$MESH_ARTIFACT_TRANSFER")
 "$SCRIPT_DIR/mesh-private-join.sh" "${args[@]}" --bin "$macbin"
}
selected(){ local n="$1"; [[ -z "$TARGET" || "$TARGET" == all || "$TARGET" == "$n" ]]; }
case "$CMD" in
 plan)
  for n in "${NODE_NAMES[@]}";do enabled "$n"||continue;selected "$n"||continue; echo "$n: role=$(node_var "$n" ROLE) bin=$(node_var "$n" BIN) console=$(node_var "$n" CONSOLE) api=$(node_var "$n" API_PORT) web_ui=$(node_var "$n" WEB_UI) vram=$(node_var "$n" MAX_VRAM) device=$(node_var "$n" DEVICE)";done;;
 preflight|status|stop|logs)
  for n in "${NODE_NAMES[@]}";do enabled "$n"||continue;selected "$n"||continue; echo "[$n]"; remote_run "$n" "$CMD" "$(json_for "$n")" "" "" || true;done;;
 start)
  enabled mac || { echo 'Mac seed must be enabled.' >&2; exit 2; }

  echo 'Starting Mac seed and obtaining invite token...'; token="$(seed_start)"; [[ -n "$token" ]] || { echo 'No token returned by seed' >&2; exit 1; }
  echo 'Seed ready. Launching enabled remote nodes concurrently (no readiness wait).'
  pids=(); names=()
  for n in win win2 win3;do enabled "$n"||continue;selected "$n"||continue; (remote_run "$n" start "$(json_for "$n")" "$token" "$([[ $WAIT_READY -eq 1 ]] && printf -- '-WaitReady')") >"/tmp/mesh-fleet-$n.out" 2>&1 & pids+=("$!");names+=("$n");done
  for i in "${!pids[@]}";do if wait "${pids[$i]}";then echo "[${names[$i]}] launched: $(tr '\n' ' ' </tmp/mesh-fleet-${names[$i]}.out)";else echo "[${names[$i]}] launch failed: $(tail -20 /tmp/mesh-fleet-${names[$i]}.out)" >&2;fi;done;;
 *) usage;exit 2;;
esac
