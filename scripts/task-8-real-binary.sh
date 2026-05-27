#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN_NAME="${TASK8_PLAN_NAME:-embedded-release-attestation}"
EVIDENCE_DIR="${TASK8_EVIDENCE_DIR:-$REPO_ROOT/.sisyphus/evidence}"
NOTEPAD_DIR="${TASK8_NOTEPAD_DIR:-$REPO_ROOT/.sisyphus/notepads/$PLAN_NAME}"
WORK_ROOT="${TASK8_WORK_ROOT:-$(mktemp -d /tmp/mesh-task8-real-binary.XXXXXX)}"
RUNTIME_ROOT="$WORK_ROOT/runtime"
if [[ -x "$REPO_ROOT/target/release/mesh-llm" ]]; then
  BIN_SRC="$REPO_ROOT/target/release/mesh-llm"
else
  BIN_SRC="$REPO_ROOT/target/debug/mesh-llm"
fi

SIGNED_EVIDENCE="$EVIDENCE_DIR/task-8-real-binary-signed-accepted.txt"
UNSIGNED_EVIDENCE="$EVIDENCE_DIR/task-8-real-binary-unsigned-rejected.txt"
TAMPERED_EVIDENCE="$EVIDENCE_DIR/task-8-real-binary-tampered-rejected.txt"
OWNER_KEY="$WORK_ROOT/owner.json"
RELEASE_KEY_PRIVATE="$WORK_ROOT/release-key.json"
RELEASE_KEY_PUBLIC="$WORK_ROOT/release-key.pub.json"

mkdir -p "$EVIDENCE_DIR" "$NOTEPAD_DIR" "$WORK_ROOT" "$RUNTIME_ROOT"

cleanup_all() {
  if [[ -d "$WORK_ROOT" ]]; then
    while IFS= read -r pid_file; do
      local pid
      pid="$(cat "$pid_file" 2>/dev/null || true)"
      if [[ "$pid" =~ ^[0-9]+$ ]]; then
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
      fi
    done < <(find "$WORK_ROOT" -mindepth 2 -maxdepth 2 -name pid -type f 2>/dev/null)
  fi
}

trap cleanup_all EXIT

json_get() {
  local file="$1"
  local expr="$2"
  python3 - "$file" "$expr" <<'PY'
import json, sys
path, expr = sys.argv[1:3]
with open(path, 'r', encoding='utf-8') as fh:
    value = json.load(fh)
for part in expr.split('.'):
    if part:
        value = value[part]
if isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
PY
}

sanitize_stream() {
  python3 -c 'import re, sys; work_root = sys.argv[1]; text = sys.stdin.read().replace(work_root, "<sanitized-temp-root>"); text = re.sub(r"--join\s+[A-Za-z0-9+/=_:-]+", "--join <signed-bootstrap-token>", text); text = re.sub(r"(\"token\":\"[^\"]+\")", "\"token\":\"<redacted>\"", text); sys.stdout.write(text)' "$WORK_ROOT"
}

wait_for_status() {
  local port="$1"
  local out="$2"
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${port}/api/status" > "$out" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

token_fingerprint() {
  local token="$1"
  if [[ -z "$token" ]]; then
    printf 'none'
    return
  fi
  local digest
  digest="$(printf '%s' "$token" | shasum -a 256 | awk '{print $1}')"
  printf 'sha256:%s' "$digest"
}

prepare_binary() {
  local name="$1"
  local node_dir="$WORK_ROOT/$name"
  mkdir -p "$node_dir"
  cp "$BIN_SRC" "$node_dir/mesh-llm"
  chmod +x "$node_dir/mesh-llm"
}

stamp_binary() {
  local name="$1"
  local node_version="$2"
  cargo run -q -p xtask -- release-attestation stamp \
    --binary "$WORK_ROOT/$name/mesh-llm" \
    --signing-key-file "$RELEASE_KEY_PRIVATE" \
    --node-version "$node_version" \
    --protocol-min 1 \
    --protocol-max 1 \
    >/dev/null
}

inspect_binary() {
  local binary="$1"
  local output="$2"
  shift 2
  cargo run -q -p xtask -- release-attestation inspect --binary "$binary" --json "$@" > "$output"
}

init_release_signer() {
  cargo run -q -p xtask -- release-attestation generate-keypair \
    --private-key-out "$RELEASE_KEY_PRIVATE" \
    --public-key-out "$RELEASE_KEY_PUBLIC" \
    >/dev/null
}

init_owner() {
  rm -f "$OWNER_KEY"
  "$BIN_SRC" auth init --owner-key "$OWNER_KEY" --force --no-passphrase >/dev/null
}

start_node() {
  local name="$1"
  local console="$2"
  local api="$3"
  local bind="$4"
  shift 4
  local node_dir="$WORK_ROOT/$name"
  local log="$node_dir/${name}.log"
  local status="$node_dir/status.json"
  local runtime_dir="$RUNTIME_ROOT/$name"
  mkdir -p "$runtime_dir"
  (
    export MESH_LLM_RUNTIME_ROOT="$runtime_dir"
    "$node_dir/mesh-llm" client --log-format json --headless --owner-required --owner-key "$OWNER_KEY" --console "$console" --port "$api" --bind-port "$bind" "$@" > "$log" 2>&1
  ) &
  local pid=$!
  echo "$pid" > "$node_dir/pid"
  wait_for_status "$console" "$status"
}

stop_node() {
  local name="$1"
  local node_dir="$WORK_ROOT/$name"
  if [[ -f "$node_dir/pid" ]]; then
    local pid
    pid="$(cat "$node_dir/pid")"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

status_token() {
  local name="$1"
  json_get "$WORK_ROOT/$name/status.json" token
}

status_summary() {
  local name="$1"
  python3 - "$WORK_ROOT/$name/status.json" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as fh:
    status = json.load(fh)
print(json.dumps({
    'version': status.get('version'),
    'peer_count': len(status.get('peers') or []),
    'release_attestation': status.get('release_attestation'),
    'recent_mesh_rejections': status.get('recent_mesh_rejections'),
}, indent=2))
PY
}

log_tail() {
  local name="$1"
  python3 - "$WORK_ROOT/$name/${name}.log" <<'PY'
import sys
from collections import deque
with open(sys.argv[1], 'r', encoding='utf-8', errors='replace') as fh:
    for line in deque(fh, maxlen=12):
        print(line, end='')
PY
}

write_evidence() {
  local output="$1"
  local scenario="$2"
  local signer_key_id="$3"
  local token="$4"
  local host_name="$5"
  local joiner_name="$6"
  local raw_output
  raw_output="$WORK_ROOT/$(basename "$output").raw"
  {
    printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'work_root=<sanitized-temp-root>\n'
    printf 'scenario=%s\n' "$scenario"
    printf 'signer_key_id=%s\n' "$signer_key_id"
    printf 'join_token_fingerprint=%s\n' "$(token_fingerprint "$token")"
    printf 'host_command=%s\n' "$(printf '%s/mesh-llm client --log-format json --headless --owner-required --owner-key %s --console %s --port %s --bind-port %s --require-release-attestation --release-signer-key %s --min-node-version 0.66.0 --max-node-version 0.66.9 --min-protocol-version 1 --max-protocol-version 1' "$WORK_ROOT/$host_name" "$OWNER_KEY" "${7}" "${8}" "${9}" "$signer_key_id")"
    printf 'joiner_command=%s\n\n' "$(printf '%s/mesh-llm client --log-format json --headless --owner-required --owner-key %s --console %s --port %s --bind-port %s --join %s' "$WORK_ROOT/$joiner_name" "$OWNER_KEY" "${10}" "${11}" "${12}" "$token")"
    printf '[host_inspect]\n'
    cat "$WORK_ROOT/$host_name/${host_name}-inspect.json"
    printf '\n\n[joiner_inspect]\n'
    cat "$WORK_ROOT/$joiner_name/${joiner_name}-inspect.json"
    printf '\n\n[host_status_summary]\n'
    status_summary "$host_name"
    printf '\n\n[joiner_status_summary]\n'
    status_summary "$joiner_name"
    printf '\n\n[host_log_tail]\n'
    log_tail "$host_name"
    printf '\n\n[joiner_log_tail]\n'
    log_tail "$joiner_name"
    printf '\n'
  } > "$raw_output"
  sanitize_stream < "$raw_output" > "$output"
}

run_signed_acceptance() {
  prepare_binary signed-host
  prepare_binary signed-joiner
  stamp_binary signed-host 0.66.2
  stamp_binary signed-joiner 0.66.4
  local signer_key
  signer_key="$(json_get "$RELEASE_KEY_PUBLIC" signer_key_id)"
  inspect_binary "$WORK_ROOT/signed-host/mesh-llm" "$WORK_ROOT/signed-host/signed-host-inspect.json" --public-key-file "$RELEASE_KEY_PUBLIC"
  inspect_binary "$WORK_ROOT/signed-joiner/mesh-llm" "$WORK_ROOT/signed-joiner/signed-joiner-inspect.json" --public-key-file "$RELEASE_KEY_PUBLIC"
  start_node signed-host 3611 9611 8041 --require-release-attestation --release-signer-key "$signer_key" --min-node-version 0.66.0 --max-node-version 0.66.9 --min-protocol-version 1 --max-protocol-version 1
  local token
  token="$(status_token signed-host)"
  start_node signed-joiner 3612 9612 8042 --join "$token"
  sleep 5
  wait_for_status 3611 "$WORK_ROOT/signed-host/status.json"
  wait_for_status 3612 "$WORK_ROOT/signed-joiner/status.json"
  write_evidence "$SIGNED_EVIDENCE" "signed accepted" "$signer_key" "$token" signed-host signed-joiner 3611 9611 8041 3612 9612 8042
  stop_node signed-joiner
  stop_node signed-host
}

run_unsigned_rejection() {
  prepare_binary unsigned-host
  prepare_binary unsigned-joiner
  stamp_binary unsigned-host 0.66.2
  local signer_key
  signer_key="$(json_get "$RELEASE_KEY_PUBLIC" signer_key_id)"
  inspect_binary "$WORK_ROOT/unsigned-host/mesh-llm" "$WORK_ROOT/unsigned-host/unsigned-host-inspect.json" --public-key-file "$RELEASE_KEY_PUBLIC"
  inspect_binary "$WORK_ROOT/unsigned-joiner/mesh-llm" "$WORK_ROOT/unsigned-joiner/unsigned-joiner-inspect.json"
  start_node unsigned-host 3621 9621 8051 --require-release-attestation --release-signer-key "$signer_key" --min-node-version 0.66.0 --max-node-version 0.66.9 --min-protocol-version 1 --max-protocol-version 1
  local token
  token="$(status_token unsigned-host)"
  start_node unsigned-joiner 3622 9622 8052 --join "$token"
  sleep 5
  wait_for_status 3621 "$WORK_ROOT/unsigned-host/status.json"
  wait_for_status 3622 "$WORK_ROOT/unsigned-joiner/status.json"
  write_evidence "$UNSIGNED_EVIDENCE" "unsigned rejected" "$signer_key" "$token" unsigned-host unsigned-joiner 3621 9621 8051 3622 9622 8052
  stop_node unsigned-joiner
  stop_node unsigned-host
}

run_tampered_rejection() {
  prepare_binary tampered-host
  prepare_binary tampered-joiner
  stamp_binary tampered-host 0.66.2
  stamp_binary tampered-joiner 0.66.4
  python3 - "$WORK_ROOT/tampered-joiner/mesh-llm" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
data = bytearray(path.read_bytes())
data[-1] ^= 0x01
path.write_bytes(data)
PY
  local signer_key
  signer_key="$(json_get "$RELEASE_KEY_PUBLIC" signer_key_id)"
  inspect_binary "$WORK_ROOT/tampered-host/mesh-llm" "$WORK_ROOT/tampered-host/tampered-host-inspect.json" --public-key-file "$RELEASE_KEY_PUBLIC"
  inspect_binary "$WORK_ROOT/tampered-joiner/mesh-llm" "$WORK_ROOT/tampered-joiner/tampered-joiner-inspect.json" --public-key-file "$RELEASE_KEY_PUBLIC"
  start_node tampered-host 3631 9631 8061 --require-release-attestation --release-signer-key "$signer_key" --min-node-version 0.66.0 --max-node-version 0.66.9 --min-protocol-version 1 --max-protocol-version 1
  local token
  token="$(status_token tampered-host)"
  start_node tampered-joiner 3632 9632 8062 --join "$token"
  sleep 5
  wait_for_status 3631 "$WORK_ROOT/tampered-host/status.json"
  wait_for_status 3632 "$WORK_ROOT/tampered-joiner/status.json"
  write_evidence "$TAMPERED_EVIDENCE" "tampered rejected" "$signer_key" "$token" tampered-host tampered-joiner 3631 9631 8061 3632 9632 8062
  stop_node tampered-joiner
  stop_node tampered-host
}

append_notepads() {
  cat >> "$NOTEPAD_DIR/learnings.md" <<'EOF'

## Task 8 real-binary script refresh
- `scripts/task-8-real-binary.sh` now models only the embedded attestation flow: generate a temporary release-signing keypair, stamp copied binaries in place, inspect them with `xtask release-attestation inspect --json`, and tamper by mutating the binary bytes directly.
- The maintained checked-in task-8 evidence set is now just the three live embedded-flow scenarios (`signed accepted`, `unsigned rejected`, `tampered rejected`), and the script scrubs bootstrap tokens from recorded commands/logs while keeping inspect output and `/api/status` release-attestation summaries.
EOF
}

run_selected() {
  local target="${1:-all}"
  cleanup_all
  init_release_signer
  init_owner
  case "$target" in
    all)
      run_signed_acceptance
      run_unsigned_rejection
      run_tampered_rejection
      append_notepads
      ;;
    signed)
      run_signed_acceptance
      ;;
    unsigned)
      run_unsigned_rejection
      ;;
    tampered)
      run_tampered_rejection
      ;;
    *)
      printf 'unknown scenario: %s\n' "$target" >&2
      return 1
      ;;
  esac
}

run_selected "$@"
