#!/usr/bin/env bash
set -euo pipefail

# Supported-families certification battery (issue #1434; tiers dropped 2026-08-25).
#
# Every row of the single manifest gets core certification: parity oracle +
# dtype/state lanes; models with MTP/NextN tensors additionally run the
# speculative lane. Hybrid/recurrent rows (sweep_period > 0) also run a
# boundary sweep — one representative split layer for every cut offset modulo
# the family's interleaving period.
#
# Models are NEVER cached through GitHub Actions cache. The family-certify
# runner ships a large pre-warmed, read-only HF cache. When HF_CACHE is set,
# model resolution is forced offline so a cache miss fails without attempting
# to mutate the shared NFS cache. Local runs without HF_CACHE may download into
# their normal user cache.
#
# Usage:
#   scripts/skippy-family-battery.sh [--manifest PATH] [--families CSV]
#     [--preflight-only] [--skip-build] [--dry-run]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${FAMILY_BATTERY_BIN_DIR:-$ROOT/target/debug}"

MANIFEST="$ROOT/ci/llama-canary/family-certified.tsv"
SKIP_BUILD=0
DRY_RUN=0
PREFLIGHT_ONLY=0
FAMILY_FILTER=""
SWEEP_MAX_CUTS="${FAMILY_BATTERY_SWEEP_MAX_CUTS:-3}"
STARTUP_TIMEOUT_MIN_SECS="${FAMILY_BATTERY_STARTUP_TIMEOUT_MIN_SECS:-180}"
STARTUP_TIMEOUT_PER_GIB_SECS="${FAMILY_BATTERY_STARTUP_TIMEOUT_PER_GIB_SECS:-10}"
STARTUP_TIMEOUT_MAX_SECS="${FAMILY_BATTERY_STARTUP_TIMEOUT_MAX_SECS:-900}"
CERT_TIMEOUT_MIN_SECS="${FAMILY_BATTERY_CERT_TIMEOUT_MIN_SECS:-1200}"
CERT_TIMEOUT_STARTUP_MULTIPLIER="${FAMILY_BATTERY_CERT_TIMEOUT_STARTUP_MULTIPLIER:-3}"
CERT_TIMEOUT_MAX_SECS="${FAMILY_BATTERY_CERT_TIMEOUT_MAX_SECS:-3600}"
MIN_FREE_GIB="${FAMILY_BATTERY_MIN_FREE_GIB:-5}"
BATTERY_RUN_ID="${FAMILY_BATTERY_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
ARTIFACT_ROOT="${FAMILY_BATTERY_ARTIFACT_ROOT:-$ROOT/target/family-battery}"
ARTIFACT_DIR="$ARTIFACT_ROOT/$BATTERY_RUN_ID"
MODEL_SCAN_DIR="$ARTIFACT_DIR/model-scans"
PREFLIGHT_DIR="$ARTIFACT_DIR/preflight"
CERT_DIR="$ARTIFACT_DIR/certifications"
RESULTS_JSONL="$ARTIFACT_DIR/results.jsonl"
MTP_CORPUS_TSV="$ARTIFACT_DIR/mtp-corpus.tsv"
RESOLVED_MANIFEST="$ARTIFACT_DIR/resolved-models.tsv"
SUMMARY_TSV="$ARTIFACT_DIR/summary.tsv"
SUMMARY_MD="$ARTIFACT_DIR/summary.md"
SPECULATIVE_CORPUS="$ROOT/crates/skippy-bench/corpora/speculative_coding_prompts.jsonl"

usage() {
  cat >&2 <<'EOF'
usage: scripts/skippy-family-battery.sh [options]

options:
  --manifest PATH           certification manifest;
                            default: ci/llama-canary/family-certified.tsv
  --families CSV            run only these exact family labels
  --preflight-only          resolve, pin, scan, and probe models without certification
  --skip-build              skip the one-time certification binary build;
                            required binaries must already exist
  --dry-run                 print the certification commands only
  -h, --help                show this help
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --manifest) MANIFEST="$2"; shift ;;
    --families) FAMILY_FILTER="$2"; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

if [[ ! -f "$MANIFEST" ]]; then
  echo "error: manifest not found: $MANIFEST" >&2
  exit 1
fi

# The family-certify self-hosted runner exports HF_CACHE pointing at its
# pre-warmed Hugging Face cache. Normalize it into the variables `hf` and
# the parity tooling already honor, so downloads only happen on misses.
if [[ -n "${HF_CACHE:-}" ]]; then
  export HF_HOME="$HF_CACHE"
  export HF_HUB_CACHE="$HF_CACHE/hub"
  export HF_HUB_OFFLINE=1
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 1
  }
}

require_cmd hf
require_cmd jq
require_cmd python3

for value in \
  "$SWEEP_MAX_CUTS" \
  "$STARTUP_TIMEOUT_MIN_SECS" \
  "$STARTUP_TIMEOUT_PER_GIB_SECS" \
  "$STARTUP_TIMEOUT_MAX_SECS" \
  "$CERT_TIMEOUT_MIN_SECS" \
  "$CERT_TIMEOUT_STARTUP_MULTIPLIER" \
  "$CERT_TIMEOUT_MAX_SECS" \
  "$MIN_FREE_GIB"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "family battery numeric settings must be non-negative integers" >&2
    exit 1
  fi
done
if (( STARTUP_TIMEOUT_MIN_SECS == 0 || STARTUP_TIMEOUT_MAX_SECS < STARTUP_TIMEOUT_MIN_SECS )); then
  echo "startup timeout bounds are invalid" >&2
  exit 1
fi
if (( CERT_TIMEOUT_MIN_SECS == 0 || CERT_TIMEOUT_STARTUP_MULTIPLIER == 0 || CERT_TIMEOUT_MAX_SECS < CERT_TIMEOUT_MIN_SECS )); then
  echo "certification timeout bounds are invalid" >&2
  exit 1
fi
if [[ -n "$FAMILY_FILTER" && ! "$FAMILY_FILTER" =~ ^[a-zA-Z0-9._-]+(,[a-zA-Z0-9._-]+)*$ ]]; then
  echo "--families must be a comma-separated list of exact family labels" >&2
  exit 1
fi

mkdir -p "$MODEL_SCAN_DIR" "$PREFLIGHT_DIR" "$CERT_DIR"
: > "$RESULTS_JSONL"
printf 'family\tmodel_id\tsource_revision\tmodel_path\tmtp_layers\n' > "$MTP_CORPUS_TSV"
printf 'family|repo|source_revision|file|selector|sweep_period|layer_end|notes|target_path|draft_repo|draft_revision|draft_file|draft_path|native_mtp|model_size_bytes|mtp_layers|startup_timeout_secs\n' > "$RESOLVED_MANIFEST"

if [[ ! -s "$SPECULATIVE_CORPUS" ]] || ! jq -e -s '
    length > 0 and all(.[]; ((.prompt // .text) | type) == "string")
  ' "$SPECULATIVE_CORPUS" >/dev/null; then
  echo "invalid checked-in speculative corpus: $SPECULATIVE_CORPUS" >&2
  exit 1
fi

FAILURES=()
TOTAL=0
CERT_FAILURE_COUNT=0
PREFLIGHT_FAILURE_COUNT=0
PREFLIGHT_SPEC_FAMILY=""
PREFLIGHT_SPEC_TARGET=""
PREFLIGHT_SPEC_DRAFT=""
PREFLIGHT_SPEC_SIZE=0
PREFLIGHT_FIRST_TARGET=""

family_selected() {
  local family="$1"
  [[ -z "$FAMILY_FILTER" || ",$FAMILY_FILTER," == *",$family,"* ]]
}

snapshot_revision_from_path() {
  python3 - "$1" <<'PY'
import re
import sys
from pathlib import Path

parts = Path(sys.argv[1]).parts
for index, part in enumerate(parts[:-1]):
    if part == "snapshots" and index + 1 < len(parts):
        revision = parts[index + 1]
        if re.fullmatch(r"[0-9a-f]{40,64}", revision):
            print(revision)
            raise SystemExit(0)
raise SystemExit(1)
PY
}

record_preflight_outcome() {
  local name="$1" family="$2" model_id="$3" status="$4" outcome="$5" note="$6"
  jq -n \
    --arg family "$family" \
    --arg model_id "$model_id" \
    --arg status "$status" \
    --arg outcome "$outcome" \
    --arg note "$note" \
    --arg name "$name" \
    '{family:$family,model_id:$model_id,exit_code:(if $status == "pass" then 0 else 1 end),outcomes:[{name:$name,status:$status,outcome:$outcome,exit_code:(if $status == "pass" then 0 else 1 end),note:$note}]}' \
    >> "$RESULTS_JSONL"
}

# resolve_model REPO FILE -> prints a local path. On the family-certify runner,
# HF_HUB_OFFLINE=1 makes a missing pre-warmed artifact a hard, read-only miss.
# Local runs without HF_CACHE retain the normal hf download behavior.
resolve_model() {
  local repo="$1" file="$2" revision="${3:-}"
  local out raw
  local command=(hf download "$repo" "$file")
  if [[ -n "$revision" ]]; then
    command+=(--revision "$revision")
  fi
  if ! raw="$("${command[@]}" 2>/dev/null)"; then
    return 0
  fi
  out="$(
    printf '%s\n' "$raw" \
      | sed -n \
        -e 's/^path=//p' \
        -e 's/^[[:space:]]*path:[[:space:]]*//p' \
      | tail -n 1
  )"
  if [[ -z "$out" ]]; then
    # newer hub-cli versions print the bare path
    out="$(printf '%s\n' "$raw" | tail -n 1)"
  fi
  printf '%s\n' "$out"
}

# Resolve through the cache's mutable ref once, capture the immutable snapshot
# SHA, then resolve the file again by that SHA. All certification commands use
# the returned snapshot path and the exact revision is written to artifacts.
resolve_pinned_model() {
  local repo="$1" file="$2"
  local initial revision pinned
  initial="$(resolve_model "$repo" "$file")"
  if [[ -z "$initial" || ! -f "$initial" ]]; then
    return 1
  fi
  revision="$(snapshot_revision_from_path "$initial")" || return 1
  pinned="$(resolve_model "$repo" "$file" "$revision")"
  if [[ -z "$pinned" || ! -f "$pinned" ]]; then
    return 1
  fi
  printf '%s|%s\n' "$pinned" "$revision"
}

build_certification_binaries() {
  local bins=(skippy-correctness skippy-server skippy-model-package llama-spec-bench)
  local bin

  if (( DRY_RUN == 1 )); then
    if (( SKIP_BUILD == 0 )); then
      echo "env LLAMA_STAGE_BUILD_DIR='<repo>/.deps/llama-build/build-stage-abi-static' cargo build -p skippy-correctness -p skippy-server -p skippy-model-package -p llama-spec-bench"
    fi
    return 0
  fi

  if (( SKIP_BUILD == 0 )); then
    env LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-$ROOT/.deps/llama-build/build-stage-abi-static}" \
      cargo build -p skippy-correctness -p skippy-server -p skippy-model-package -p llama-spec-bench
    return 0
  fi

  for bin in "${bins[@]}"; do
    if [[ ! -x "$BIN_DIR/$bin" ]]; then
      echo "--skip-build requires existing binary: $BIN_DIR/$bin" >&2
      return 1
    fi
  done
}

slugify() {
  printf '%s' "$1" | tr '/[:upper:]' '-[:lower:]' | tr -cs 'a-z0-9._-' '-'
}

startup_timeout_for_bytes() {
  local bytes="$1"
  local gib=$(( (bytes + 1073741823) / 1073741824 ))
  local timeout=$(( STARTUP_TIMEOUT_MIN_SECS + gib * STARTUP_TIMEOUT_PER_GIB_SECS ))
  if (( timeout > STARTUP_TIMEOUT_MAX_SECS )); then
    timeout="$STARTUP_TIMEOUT_MAX_SECS"
  fi
  printf '%s\n' "$timeout"
}

cert_timeout_for_startup() {
  local startup_timeout="$1"
  local timeout=$(( CERT_TIMEOUT_MIN_SECS + startup_timeout * CERT_TIMEOUT_STARTUP_MULTIPLIER ))
  if (( timeout > CERT_TIMEOUT_MAX_SECS )); then
    timeout="$CERT_TIMEOUT_MAX_SECS"
  fi
  printf '%s\n' "$timeout"
}

scan_model() {
  local family="$1" target="$2" model_id="$3" source_revision="$4"
  local family_slug scan_json scan_log
  family_slug="$(slugify "$family")"
  scan_json="$MODEL_SCAN_DIR/$family_slug.json"
  scan_log="$MODEL_SCAN_DIR/$family_slug.log"
  MODEL_HAS_MTP=0
  MODEL_SIZE_BYTES=0
  MODEL_MTP_LAYERS=""

  if (( DRY_RUN == 1 )); then
    echo "$BIN_DIR/skippy-model-package inspect '$target' > '$scan_json'"
    return 0
  fi
  if ! "$BIN_DIR/skippy-model-package" inspect "$target" >"$scan_json" 2>"$scan_log"; then
    jq -n \
      --arg family "$family" \
      --arg model_id "$model_id" \
      --arg target "$target" \
      --arg scan_log "$scan_log" \
      '{family:$family,model_id:$model_id,target_model:$target,exit_code:1,outcomes:[{name:"model-scan",status:"fail",outcome:"harness",log:$scan_log}]}' \
      >> "$RESULTS_JSONL"
    FAILURES+=("$family(mtp-scan)")
    return 1
  fi

  MODEL_SIZE_BYTES="$(jq '[.tensors[].byte_size] | add // 0' "$scan_json")"
  MODEL_MTP_LAYERS="$(jq -r '[.tensors[] | select(.name | contains(".nextn.")) | .layer_index | select(. != null)] | unique | map(tostring) | join(",")' "$scan_json")"
  if [[ -n "$MODEL_MTP_LAYERS" ]]; then
    MODEL_HAS_MTP=1
    printf '%s\t%s\t%s\t%s\t%s\n' "$family" "$model_id" "$source_revision" "$target" "$MODEL_MTP_LAYERS" >> "$MTP_CORPUS_TSV"
  fi
}

preflight_environment() {
  local model_root="${HF_HOME:-$(dirname "$PREFLIGHT_FIRST_TARGET")}"
  python3 - "$ARTIFACT_DIR" "$model_root" "$MIN_FREE_GIB" "$PREFLIGHT_DIR/environment.json" <<'PY'
import json
import shutil
import socket
import sys
from pathlib import Path

artifact_root, model_root, minimum_gib, output = sys.argv[1:]
minimum_bytes = int(minimum_gib) * 1024**3
filesystems = []
for label, path_text in (("artifacts", artifact_root), ("models", model_root)):
    path = Path(path_text)
    while not path.exists() and path != path.parent:
        path = path.parent
    usage = shutil.disk_usage(path)
    filesystems.append(
        {
            "label": label,
            "path": str(path),
            "free_bytes": usage.free,
            "minimum_free_bytes": minimum_bytes,
            "sufficient": usage.free >= minimum_bytes,
        }
    )

busy_ports = []
for port in range(19000, 20032):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.01)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            busy_ports.append(port)
    except OSError:
        busy_ports.append(port)
    finally:
        sock.close()

report = {
    "filesystems": filesystems,
    "port_range": {"start": 19000, "end": 20031, "busy": busy_ports},
}
Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
if any(not item["sufficient"] for item in filesystems) or busy_ports:
    raise SystemExit(1)
PY
}

preflight_speculative_corpus() {
  local report="$PREFLIGHT_DIR/speculative-smoke.json"
  local log="$PREFLIGHT_DIR/speculative-smoke.log"
  local timeout
  timeout="$(cert_timeout_for_startup "$(startup_timeout_for_bytes "$PREFLIGHT_SPEC_SIZE")")"
  local command=(
    env
    "LLAMA_STAGE_BUILD_DIR=${LLAMA_STAGE_BUILD_DIR:-$ROOT/.deps/llama-build/build-stage-abi-static}"
    "$BIN_DIR/llama-spec-bench"
    --target-model-path "$PREFLIGHT_SPEC_TARGET"
    --draft-model-path "$PREFLIGHT_SPEC_DRAFT"
    --prompt-corpus "$SPECULATIVE_CORPUS"
    --prompt-limit 1
    --max-new-tokens 1
    --speculative-window 1
    --ctx-size 128
    --n-gpu-layers 999
    --json-out "$report"
  )
  {
    printf '+ %s\n\n' "$(printf '%q ' "${command[@]}")"
    "$ROOT/scripts/run-command-with-timeout.py" \
      --seconds "$timeout" \
      --label "MTP speculative preflight ($PREFLIGHT_SPEC_FAMILY)" \
      -- "${command[@]}"
  } >"$log" 2>&1
}

run_certify() {
  local family="$1" target="$2" model_id="$3" source_revision="$4" split_layer="$5" layer_end="$6" draft="$7" draft_revision="$8" native_mtp="$9"
  local run_speculative="${10}" startup_timeout="${11}" model_size_bytes="${12}"
  # Two distinct interior cut points so the chain lane (exactly two split
  # indexes) always has valid inputs; distinct from each other and from 0.
  local chain_a=$(( layer_end / 3 ))
  local chain_b=$(( ( layer_end * 2 ) / 3 ))
  if (( chain_b == chain_a || chain_a < 1 )); then
    chain_a=1
    chain_b=2
  fi
  TOTAL=$((TOTAL + 1))
  local cert_run_id cert_run_dir exit_code manifest_path cert_timeout spec_label
  cert_run_id="$(printf '%03d-%s-split-%s' "$TOTAL" "$(slugify "$family")" "$split_layer")"
  cert_run_dir="$CERT_DIR/$cert_run_id"
  cert_timeout="$(cert_timeout_for_startup "$startup_timeout")"
  spec_label="disabled"
  if (( run_speculative == 1 )); then
    spec_label="$(basename "$draft")"
  fi
  echo "==> family-certify: family=$family split=$split_layer mtp=$native_mtp startup_timeout=${startup_timeout}s cert_timeout=${cert_timeout}s draft=$spec_label model=$(basename "$target")"
  local command=(
    "$ROOT/scripts/family-certify.sh"
    --family "$family"
    --target-model "$target"
    --model-id "$model_id"
    --split-layer "$split_layer"
    --layer-end "$layer_end"
    --splits "$chain_a,$chain_b"
    --startup-timeout-secs "$startup_timeout"
    --cert-root "$cert_run_dir"
    --run-id certification
    --require-lanes
    --skip-build
  )
  if (( native_mtp == 1 )); then
    command+=(--require-native-mtp-draft)
  fi
  if (( run_speculative == 1 )); then
    command+=(
      --draft-model "$draft"
      --corpus "$SPECULATIVE_CORPUS"
    )
  else
    command+=(--skip-speculative)
  fi
  if (( DRY_RUN == 1 )); then
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi
  exit_code=0
  "$ROOT/scripts/run-command-with-timeout.py" \
    --seconds "$cert_timeout" \
    --label "family certification $family split $split_layer" \
    -- "${command[@]}" || exit_code=$?
  manifest_path="$(find "$cert_run_dir" -name manifest.json -type f -print -quit)"
  if [[ -n "$manifest_path" ]]; then
    jq -c \
      --arg family "$family" \
      --arg model_id "$model_id" \
      --arg source_revision "$source_revision" \
      --arg draft_revision "$draft_revision" \
      --argjson split_layer "$split_layer" \
      --argjson model_size_bytes "$model_size_bytes" \
      --argjson startup_timeout_secs "$startup_timeout" \
      --argjson certification_timeout_secs "$cert_timeout" \
      --argjson native_mtp "$native_mtp" \
      --argjson exit_code "$exit_code" \
      '{family:$family,model_id:$model_id,source_revision:$source_revision,draft_revision:($draft_revision | if length > 0 then . else null end),split_layer:$split_layer,model_size_bytes:$model_size_bytes,startup_timeout_secs:$startup_timeout_secs,certification_timeout_secs:$certification_timeout_secs,native_mtp:($native_mtp == 1),exit_code:$exit_code,manifest:input_filename,outcomes:.commands}' \
      "$manifest_path" >> "$RESULTS_JSONL"
  else
    jq -n \
      --arg family "$family" \
      --arg model_id "$model_id" \
      --arg source_revision "$source_revision" \
      --arg draft_revision "$draft_revision" \
      --argjson split_layer "$split_layer" \
      --argjson model_size_bytes "$model_size_bytes" \
      --argjson startup_timeout_secs "$startup_timeout" \
      --argjson certification_timeout_secs "$cert_timeout" \
      --argjson native_mtp "$native_mtp" \
      --argjson exit_code "$exit_code" \
      --arg outcome "$(if (( exit_code == 124 )); then printf timeout; else printf harness; fi)" \
      '{family:$family,model_id:$model_id,source_revision:$source_revision,draft_revision:($draft_revision | if length > 0 then . else null end),split_layer:$split_layer,model_size_bytes:$model_size_bytes,startup_timeout_secs:$startup_timeout_secs,certification_timeout_secs:$certification_timeout_secs,native_mtp:($native_mtp == 1),exit_code:$exit_code,outcomes:[{name:"certification-manifest",status:"fail",outcome:$outcome,note:(if $outcome == "timeout" then "family-certify exceeded its wall-clock budget before writing a manifest" else "family-certify produced no manifest" end)}]}' \
      >> "$RESULTS_JSONL"
  fi
  if (( exit_code != 0 )); then
    FAILURES+=("$family@split=$split_layer")
    CERT_FAILURE_COUNT=$((CERT_FAILURE_COUNT + 1))
  fi
}

preflight_manifest() {
  local manifest="$1"
  [[ -f "$manifest" ]] || { echo "missing manifest: $manifest" >&2; exit 1; }

  while IFS='|' read -r family repo file selector sweep_period layer_end notes draft_repo draft_file; do
    [[ -z "$family" || "$family" == \#* ]] && continue
    family_selected "$family" || continue
    if [[ -z "$family" || -z "$repo" || -z "$file" || -z "$sweep_period" || -z "$layer_end" ]]; then
      echo "malformed row in $manifest: $family|$repo|$file" >&2
      exit 1
    fi
    if { [[ -n "$draft_repo" ]] && [[ -z "$draft_file" ]]; } \
      || { [[ -z "$draft_repo" ]] && [[ -n "$draft_file" ]]; }; then
      echo "draft_repo and draft_file must either both be set or both be empty: $family" >&2
      exit 1
    fi

    local target source_revision resolved model_id
    model_id="$repo:$selector"
    if (( DRY_RUN == 0 )); then
      if ! resolved="$(resolve_pinned_model "$repo" "$file")"; then
        echo "failed to resolve and pin $repo/$file from the HF cache or hub" >&2
        FAILURES+=("$family(model-pin)")
        PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
        record_preflight_outcome "model-preflight" "$family" "$model_id" "fail" "harness" "failed to resolve an immutable cached snapshot for $repo/$file"
        continue
      fi
      IFS='|' read -r target source_revision <<< "$resolved"
      if [[ -z "$PREFLIGHT_FIRST_TARGET" ]]; then
        PREFLIGHT_FIRST_TARGET="$target"
      fi
    else
      target="<hf-cache>/$repo/$file"
      source_revision="dry-run"
    fi
    if ! scan_model "$family" "$target" "$model_id" "$source_revision"; then
      PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
      continue
    fi

    # Only models with actual MTP/NextN tensors join the speculative cohort.
    # Non-MTP models keep all core correctness and state lanes, but do not run
    # the unrelated self-draft benchmark.
    local draft="" draft_revision=""
    if (( MODEL_HAS_MTP == 1 )); then
      draft="$target"
      draft_revision="$source_revision"
      if [[ -n "$draft_repo" && -n "$draft_file" ]]; then
        if (( DRY_RUN == 0 )); then
          if ! resolved="$(resolve_pinned_model "$draft_repo" "$draft_file")"; then
            echo "failed to resolve and pin draft $draft_repo/$draft_file" >&2
            FAILURES+=("$family(draft-pin)")
            PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
            record_preflight_outcome "model-preflight" "$family" "$model_id" "fail" "harness" "failed to resolve an immutable draft snapshot for $draft_repo/$draft_file"
            continue
          fi
          IFS='|' read -r draft draft_revision <<< "$resolved"
        else
          draft="<hf-cache>/$draft_repo/$draft_file"
          draft_revision="dry-run"
        fi
      else
        draft_repo="$repo"
        draft_file="$file"
      fi
      if (( DRY_RUN == 0 )) && (( PREFLIGHT_SPEC_SIZE == 0 || MODEL_SIZE_BYTES < PREFLIGHT_SPEC_SIZE )); then
        PREFLIGHT_SPEC_FAMILY="$family"
        PREFLIGHT_SPEC_TARGET="$target"
        PREFLIGHT_SPEC_DRAFT="$draft"
        PREFLIGHT_SPEC_SIZE="$MODEL_SIZE_BYTES"
      fi
    fi

    local startup_timeout
    startup_timeout="$(startup_timeout_for_bytes "$MODEL_SIZE_BYTES")"
    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
      "$family" "$repo" "$source_revision" "$file" "$selector" "$sweep_period" "$layer_end" "$notes" "$target" \
      "$draft_repo" "$draft_revision" "$draft_file" "$draft" "$MODEL_HAS_MTP" "$MODEL_SIZE_BYTES" "$MODEL_MTP_LAYERS" "$startup_timeout" \
      >> "$RESOLVED_MANIFEST"
    if (( DRY_RUN == 0 )); then
      record_preflight_outcome "model-preflight" "$family" "$model_id" "pass" "pass" "resolved immutable snapshot $source_revision; tensor scan complete"
    fi
  done < <(grep -v '^[[:space:]]*#' "$manifest")

  local resolved_count=$(( $(wc -l < "$RESOLVED_MANIFEST") - 1 ))
  if (( resolved_count == 0 )); then
    echo "preflight selected no model rows" >&2
    PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
  fi
  if (( DRY_RUN == 1 )); then
    return 0
  fi
  if (( PREFLIGHT_FAILURE_COUNT > 0 )); then
    return 1
  fi
  if ! preflight_environment; then
    record_preflight_outcome "environment-preflight" "battery" "environment" "fail" "harness" "insufficient disk headroom or occupied certification ports; see preflight/environment.json"
    PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
    return 1
  fi
  record_preflight_outcome "environment-preflight" "battery" "environment" "pass" "pass" "disk headroom and certification port range validated"

  if [[ -n "$PREFLIGHT_SPEC_TARGET" ]]; then
    if ! preflight_speculative_corpus; then
      record_preflight_outcome "speculative-preflight" "$PREFLIGHT_SPEC_FAMILY" "mtp-corpus" "fail" "harness" "one-prompt llama-spec-bench preflight failed; see preflight/speculative-smoke.log"
      PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
      return 1
    fi
    record_preflight_outcome "speculative-preflight" "$PREFLIGHT_SPEC_FAMILY" "mtp-corpus" "pass" "pass" "checked-in corpus consumed by one-token MTP speculative smoke"
  elif [[ -z "$FAMILY_FILTER" ]]; then
    record_preflight_outcome "speculative-preflight" "battery" "mtp-corpus" "fail" "harness" "full manifest contained no model with MTP/NextN tensors"
    PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
    return 1
  fi
}

run_resolved_manifest() {
  local resolved_manifest="$1"
  while IFS='|' read -r family repo source_revision file selector sweep_period layer_end _notes target draft_repo draft_revision draft_file draft native_mtp model_size_bytes _mtp_layers startup_timeout; do
    [[ "$family" == "family" ]] && continue
    local model_id="$repo:$selector"

    # Fixed mid-range split for the base parity + dtype lanes.
    local base_split=$(( layer_end / 2 ))
    run_certify "$family" "$target" "$model_id" "$source_revision" "$base_split" "$layer_end" "$draft" "$draft_revision" "$native_mtp" "$native_mtp" "$startup_timeout" "$model_size_bytes"

    if [[ "$sweep_period" != "0" ]]; then
      # Boundary sweep: every cut offset mod the interleaving period, one
      # representative cut each (then every period up to SWEEP_MAX_CUTS cuts),
      # so planner-cut dependence (the B1 bug class) cannot hide.
      local offset cut cuts
      for (( offset = 1; offset <= sweep_period; offset++ )); do
        cuts=0
        for (( cut = offset; cut < layer_end && cuts < SWEEP_MAX_CUTS; cut += sweep_period )); do
          (( cut == base_split )) && continue
          run_certify "$family" "$target" "$model_id" "$source_revision" "$cut" "$layer_end" "$draft" "$draft_revision" "$native_mtp" "0" "$startup_timeout" "$model_size_bytes"
          cuts=$((cuts + 1))
        done
      done
    fi
  done < "$resolved_manifest"
}

build_certification_binaries
if ! preflight_manifest "$MANIFEST"; then
  echo "family battery preflight failed; no certification lane was started" >&2
elif (( PREFLIGHT_ONLY == 0 )); then
  run_resolved_manifest "$RESOLVED_MANIFEST"
fi

echo
if (( DRY_RUN == 0 )); then
  jq -sr '
    ["family","split_layer","lane","status","outcome","exit_code"],
    (.[] as $row | $row.outcomes[] | [$row.family,($row.split_layer // ""),.name,.status,.outcome,.exit_code])
    | @tsv
  ' "$RESULTS_JSONL" > "$SUMMARY_TSV"
  {
    echo "# Supported-families battery"
    echo
    echo "- Run ID: \`$BATTERY_RUN_ID\`"
    echo "- Certifications: $TOTAL"
    echo "- MTP models: $(( $(wc -l < "$MTP_CORPUS_TSV") - 1 ))"
    echo "- Preflight failures: $PREFLIGHT_FAILURE_COUNT"
    echo "- Startup timeout policy: min ${STARTUP_TIMEOUT_MIN_SECS}s + ${STARTUP_TIMEOUT_PER_GIB_SECS}s/GiB, capped at ${STARTUP_TIMEOUT_MAX_SECS}s"
    echo "- Certification wall-clock policy: min ${CERT_TIMEOUT_MIN_SECS}s + ${CERT_TIMEOUT_STARTUP_MULTIPLIER}x startup timeout, capped at ${CERT_TIMEOUT_MAX_SECS}s"
    echo "- Minimum free space: ${MIN_FREE_GIB} GiB"
    echo
    echo "## Typed outcomes"
    echo
    echo "| Outcome | Count |"
    echo "| --- | ---: |"
    jq -sr '[.[].outcomes[]] | group_by(.outcome)[] | "| \(.[0].outcome) | \(length) |"' "$RESULTS_JSONL"
    echo
    echo "- Results: \`$RESULTS_JSONL\`"
    echo "- Lane summary: \`$SUMMARY_TSV\`"
    echo "- MTP corpus: \`$MTP_CORPUS_TSV\`"
    echo "- Immutable resolved model manifest: \`$RESOLVED_MANIFEST\`"
    echo "- Environment preflight: \`$PREFLIGHT_DIR/environment.json\`"
    echo "- Certifications and logs: \`$CERT_DIR\`"
  } > "$SUMMARY_MD"
fi

if (( PREFLIGHT_ONLY == 1 )); then
  echo "family battery preflight complete: $PREFLIGHT_FAILURE_COUNT failures"
else
  echo "family battery complete: $((TOTAL - CERT_FAILURE_COUNT))/$TOTAL certifications passed"
fi
echo "artifacts: $ARTIFACT_DIR"
if (( ${#FAILURES[@]} > 0 )); then
  printf 'failed: %s\n' "${FAILURES[@]}"
fi
if (( ${#FAILURES[@]} > 0 || PREFLIGHT_FAILURE_COUNT > 0 )); then
  exit 1
fi
