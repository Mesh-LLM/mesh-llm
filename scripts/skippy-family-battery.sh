#!/usr/bin/env bash
set -euo pipefail

# Supported-families certification battery (issue #1434; tiers dropped 2026-08-25).
#
# Every row of the single manifest gets full certification: parity oracle +
# dtype lanes; hybrid/recurrent rows (sweep_period > 0) additionally run a
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
#   scripts/skippy-family-battery.sh [--manifest PATH] [--skip-build] [--dry-run]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MANIFEST="$ROOT/ci/llama-canary/family-certified.tsv"
SKIP_BUILD=0
DRY_RUN=0
SWEEP_MAX_CUTS="${FAMILY_BATTERY_SWEEP_MAX_CUTS:-3}"
STARTUP_TIMEOUT_MIN_SECS="${FAMILY_BATTERY_STARTUP_TIMEOUT_MIN_SECS:-180}"
STARTUP_TIMEOUT_PER_GIB_SECS="${FAMILY_BATTERY_STARTUP_TIMEOUT_PER_GIB_SECS:-10}"
STARTUP_TIMEOUT_MAX_SECS="${FAMILY_BATTERY_STARTUP_TIMEOUT_MAX_SECS:-900}"
BATTERY_RUN_ID="${FAMILY_BATTERY_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
ARTIFACT_ROOT="${FAMILY_BATTERY_ARTIFACT_ROOT:-$ROOT/target/family-battery}"
ARTIFACT_DIR="$ARTIFACT_ROOT/$BATTERY_RUN_ID"
MODEL_SCAN_DIR="$ARTIFACT_DIR/model-scans"
CERT_DIR="$ARTIFACT_DIR/certifications"
RESULTS_JSONL="$ARTIFACT_DIR/results.jsonl"
MTP_CORPUS_TSV="$ARTIFACT_DIR/mtp-corpus.tsv"
SUMMARY_TSV="$ARTIFACT_DIR/summary.tsv"
SUMMARY_MD="$ARTIFACT_DIR/summary.md"
SPECULATIVE_CORPUS="$ROOT/crates/skippy-bench/corpora/speculative_coding_prompts.jsonl"

usage() {
  cat >&2 <<'EOF'
usage: scripts/skippy-family-battery.sh [options]

options:
  --manifest PATH           certification manifest;
                            default: ci/llama-canary/family-certified.tsv
  --skip-build              skip the one-time certification binary build;
                            required binaries must already exist
  --dry-run                 print the certification commands only
  -h, --help                show this help
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --manifest) MANIFEST="$2"; shift ;;
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

for value in "$STARTUP_TIMEOUT_MIN_SECS" "$STARTUP_TIMEOUT_PER_GIB_SECS" "$STARTUP_TIMEOUT_MAX_SECS"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "startup timeout settings must be non-negative integers" >&2
    exit 1
  fi
done
if (( STARTUP_TIMEOUT_MIN_SECS == 0 || STARTUP_TIMEOUT_MAX_SECS < STARTUP_TIMEOUT_MIN_SECS )); then
  echo "startup timeout bounds are invalid" >&2
  exit 1
fi

mkdir -p "$MODEL_SCAN_DIR" "$CERT_DIR"
: > "$RESULTS_JSONL"
printf 'family\tmodel_id\tmodel_path\tmtp_layers\n' > "$MTP_CORPUS_TSV"

if [[ ! -s "$SPECULATIVE_CORPUS" ]] || ! jq -e -s '
    length > 0 and all(.[]; ((.prompt // .text) | type) == "string")
  ' "$SPECULATIVE_CORPUS" >/dev/null; then
  echo "invalid checked-in speculative corpus: $SPECULATIVE_CORPUS" >&2
  exit 1
fi

FAILURES=()
TOTAL=0
CERT_FAILURE_COUNT=0

# resolve_model REPO FILE -> prints a local path. On the family-certify runner,
# HF_HUB_OFFLINE=1 makes a missing pre-warmed artifact a hard, read-only miss.
# Local runs without HF_CACHE retain the normal hf download behavior.
resolve_model() {
  local repo="$1" file="$2"
  local out raw
  if ! raw="$(hf download "$repo" "$file" 2>/dev/null)"; then
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
    if [[ ! -x "$ROOT/target/debug/$bin" ]]; then
      echo "--skip-build requires existing binary: $ROOT/target/debug/$bin" >&2
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

scan_model() {
  local family="$1" target="$2" model_id="$3"
  local family_slug scan_json scan_log
  family_slug="$(slugify "$family")"
  scan_json="$MODEL_SCAN_DIR/$family_slug.json"
  scan_log="$MODEL_SCAN_DIR/$family_slug.log"
  MODEL_HAS_MTP=0
  MODEL_SIZE_BYTES=0
  MODEL_MTP_LAYERS=""

  if (( DRY_RUN == 1 )); then
    echo "target/debug/skippy-model-package inspect '$target' > '$scan_json'"
    return 0
  fi
  if ! "$ROOT/target/debug/skippy-model-package" inspect "$target" >"$scan_json" 2>"$scan_log"; then
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
    printf '%s\t%s\t%s\t%s\n' "$family" "$model_id" "$target" "$MODEL_MTP_LAYERS" >> "$MTP_CORPUS_TSV"
  fi
}

run_certify() {
  local family="$1" target="$2" model_id="$3" split_layer="$4" layer_end="$5" draft="$6" native_mtp="$7" startup_timeout="$8" model_size_bytes="$9"
  # Two distinct interior cut points so the chain lane (exactly two split
  # indexes) always has valid inputs; distinct from each other and from 0.
  local chain_a=$(( layer_end / 3 ))
  local chain_b=$(( ( layer_end * 2 ) / 3 ))
  if (( chain_b == chain_a || chain_a < 1 )); then
    chain_a=1
    chain_b=2
  fi
  TOTAL=$((TOTAL + 1))
  local cert_run_id cert_run_dir exit_code manifest_path
  cert_run_id="$(printf '%03d-%s-split-%s' "$TOTAL" "$(slugify "$family")" "$split_layer")"
  cert_run_dir="$CERT_DIR/$cert_run_id"
  echo "==> family-certify: family=$family split=$split_layer mtp=$native_mtp startup_timeout=${startup_timeout}s draft=$(basename "$draft") model=$(basename "$target")"
  local command=(
    "$ROOT/scripts/family-certify.sh"
    --family "$family"
    --target-model "$target"
    --model-id "$model_id"
    --split-layer "$split_layer"
    --layer-end "$layer_end"
    --splits "$chain_a,$chain_b"
    --draft-model "$draft"
    --corpus "$SPECULATIVE_CORPUS"
    --startup-timeout-secs "$startup_timeout"
    --cert-root "$cert_run_dir"
    --run-id certification
    --require-lanes
    --skip-build
  )
  if (( native_mtp == 1 )); then
    command+=(--require-native-mtp-draft)
  fi
  if (( DRY_RUN == 1 )); then
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi
  exit_code=0
  "${command[@]}" || exit_code=$?
  manifest_path="$(find "$cert_run_dir" -name manifest.json -type f -print -quit)"
  if [[ -n "$manifest_path" ]]; then
    jq -c \
      --arg family "$family" \
      --arg model_id "$model_id" \
      --argjson split_layer "$split_layer" \
      --argjson model_size_bytes "$model_size_bytes" \
      --argjson startup_timeout_secs "$startup_timeout" \
      --argjson native_mtp "$native_mtp" \
      --argjson exit_code "$exit_code" \
      '{family:$family,model_id:$model_id,split_layer:$split_layer,model_size_bytes:$model_size_bytes,startup_timeout_secs:$startup_timeout_secs,native_mtp:($native_mtp == 1),exit_code:$exit_code,manifest:input_filename,outcomes:.commands}' \
      "$manifest_path" >> "$RESULTS_JSONL"
  else
    jq -n \
      --arg family "$family" \
      --arg model_id "$model_id" \
      --argjson split_layer "$split_layer" \
      --argjson model_size_bytes "$model_size_bytes" \
      --argjson startup_timeout_secs "$startup_timeout" \
      --argjson native_mtp "$native_mtp" \
      --argjson exit_code "$exit_code" \
      '{family:$family,model_id:$model_id,split_layer:$split_layer,model_size_bytes:$model_size_bytes,startup_timeout_secs:$startup_timeout_secs,native_mtp:($native_mtp == 1),exit_code:$exit_code,outcomes:[{name:"certification-manifest",status:"fail",outcome:"harness",note:"family-certify produced no manifest"}]}' \
      >> "$RESULTS_JSONL"
  fi
  if (( exit_code != 0 )); then
    FAILURES+=("$family@split=$split_layer")
    CERT_FAILURE_COUNT=$((CERT_FAILURE_COUNT + 1))
  fi
}

run_manifest() {
  local manifest="$1"
  [[ -f "$manifest" ]] || { echo "missing manifest: $manifest" >&2; exit 1; }

  while IFS='|' read -r family repo file selector sweep_period layer_end _notes draft_repo draft_file; do
    [[ -z "$family" || "$family" == \#* ]] && continue
    if [[ -z "$family" || -z "$repo" || -z "$file" || -z "$sweep_period" || -z "$layer_end" ]]; then
      echo "malformed row in $manifest: $family|$repo|$file" >&2
      exit 1
    fi

    local target
    if (( DRY_RUN == 0 )); then
      target="$(resolve_model "$repo" "$file")"
      if [[ -z "$target" || ! -f "$target" ]]; then
        echo "failed to resolve $repo/$file from the HF cache or hub" >&2
        FAILURES+=("$family(model-missing)")
        TOTAL=$((TOTAL + 1))
        continue
      fi
    else
      target="<hf-cache>/$repo/$file"
    fi
    local model_id="$repo:$selector"

    # Draft model for the speculative lane: per-row override via optional
    # draft_repo|draft_file manifest columns; defaults to self-draft (the
    # target GGUF is its own draft) so the lane always runs with declared
    # inputs instead of silently skipping.
    local draft="$target"
    if [[ -n "$draft_repo" && -n "$draft_file" ]]; then
      if (( DRY_RUN == 0 )); then
        draft="$(resolve_model "$draft_repo" "$draft_file")"
        if [[ -z "$draft" || ! -f "$draft" ]]; then
          echo "failed to resolve draft $draft_repo/$draft_file from the HF cache or hub" >&2
          FAILURES+=("$family(draft-missing)")
          TOTAL=$((TOTAL + 1))
          continue
        fi
      else
        draft="<hf-cache>/$draft_repo/$draft_file"
      fi
    fi

    scan_model "$family" "$target" "$model_id" || continue
    local startup_timeout
    startup_timeout="$(startup_timeout_for_bytes "$MODEL_SIZE_BYTES")"

    # Fixed mid-range split for the base parity + dtype lanes.
    local base_split=$(( layer_end / 2 ))
    run_certify "$family" "$target" "$model_id" "$base_split" "$layer_end" "$draft" "$MODEL_HAS_MTP" "$startup_timeout" "$MODEL_SIZE_BYTES"

    if [[ "$sweep_period" != "0" ]]; then
      # Boundary sweep: every cut offset mod the interleaving period, one
      # representative cut each (then every period up to SWEEP_MAX_CUTS cuts),
      # so planner-cut dependence (the B1 bug class) cannot hide.
      local offset cut cuts
      for (( offset = 1; offset <= sweep_period; offset++ )); do
        cuts=0
        for (( cut = offset; cut < layer_end && cuts < SWEEP_MAX_CUTS; cut += sweep_period )); do
          (( cut == base_split )) && continue
          run_certify "$family" "$target" "$model_id" "$cut" "$layer_end" "$draft" "$MODEL_HAS_MTP" "$startup_timeout" "$MODEL_SIZE_BYTES"
          cuts=$((cuts + 1))
        done
      done
    fi
  done < <(grep -v '^[[:space:]]*#' "$manifest")
}

build_certification_binaries
run_manifest "$MANIFEST"

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
    echo "- Startup timeout policy: min ${STARTUP_TIMEOUT_MIN_SECS}s + ${STARTUP_TIMEOUT_PER_GIB_SECS}s/GiB, capped at ${STARTUP_TIMEOUT_MAX_SECS}s"
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
    echo "- Certifications and logs: \`$CERT_DIR\`"
  } > "$SUMMARY_MD"
fi

echo "family battery complete: $((TOTAL - CERT_FAILURE_COUNT))/$TOTAL certifications passed"
echo "artifacts: $ARTIFACT_DIR"
if (( ${#FAILURES[@]} > 0 )); then
  printf 'failed: %s\n' "${FAILURES[@]}"
  exit 1
fi
