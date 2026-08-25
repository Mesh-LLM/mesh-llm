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

FAILURES=()
TOTAL=0

# resolve_model REPO FILE -> prints a local path. On the family-certify runner,
# HF_HUB_OFFLINE=1 makes a missing pre-warmed artifact a hard, read-only miss.
# Local runs without HF_CACHE retain the normal hf download behavior.
resolve_model() {
  local repo="$1" file="$2"
  local out raw
  if ! raw="$(hf download "$repo" "$file" 2>/dev/null)"; then
    return 0
  fi
  out="$(printf '%s\n' "$raw" | sed -n 's/^path=//p' | tail -n 1)"
  if [[ -z "$out" ]]; then
    # newer hub-cli versions print the bare path
    out="$(printf '%s\n' "$raw" | tail -n 1)"
  fi
  printf '%s\n' "$out"
}

build_certification_binaries() {
  local bins=(skippy-correctness skippy-server llama-spec-bench)
  local bin

  if (( DRY_RUN == 1 )); then
    if (( SKIP_BUILD == 0 )); then
      echo "env LLAMA_STAGE_BUILD_DIR='<repo>/.deps/llama-build/build-stage-abi-static' cargo build -p skippy-correctness -p skippy-server -p llama-spec-bench"
    fi
    return 0
  fi

  if (( SKIP_BUILD == 0 )); then
    env LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-$ROOT/.deps/llama-build/build-stage-abi-static}" \
      cargo build -p skippy-correctness -p skippy-server -p llama-spec-bench
    return 0
  fi

  for bin in "${bins[@]}"; do
    if [[ ! -x "$ROOT/target/debug/$bin" ]]; then
      echo "--skip-build requires existing binary: $ROOT/target/debug/$bin" >&2
      return 1
    fi
  done
}

run_certify() {
  local family="$1" target="$2" model_id="$3" split_layer="$4" layer_end="$5" draft="$6"
  # Two distinct interior cut points so the chain lane (exactly two split
  # indexes) always has valid inputs; distinct from each other and from 0.
  local chain_a=$(( layer_end / 3 ))
  local chain_b=$(( ( layer_end * 2 ) / 3 ))
  if (( chain_b == chain_a || chain_a < 1 )); then
    chain_a=1
    chain_b=2
  fi
  TOTAL=$((TOTAL + 1))
  echo "==> family-certify: family=$family split=$split_layer draft=$(basename "$draft") model=$(basename "$target")"
  if (( DRY_RUN == 1 )); then
    echo "scripts/family-certify.sh --family '$family' --target-model '$target' --model-id '$model_id' --split-layer '$split_layer' --layer-end '$layer_end' --splits '$chain_a,$chain_b' --draft-model '$draft' --require-lanes --skip-build"
    return 0
  fi
  if ! "$ROOT/scripts/family-certify.sh" \
      --family "$family" \
      --target-model "$target" \
      --model-id "$model_id" \
      --split-layer "$split_layer" \
      --layer-end "$layer_end" \
      --splits "$chain_a,$chain_b" \
      --draft-model "$draft" \
      --require-lanes \
      --skip-build; then
    FAILURES+=("$family@split=$split_layer")
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

    # Fixed mid-range split for the base parity + dtype lanes.
    local base_split=$(( layer_end / 2 ))
    run_certify "$family" "$target" "$model_id" "$base_split" "$layer_end" "$draft"

    if [[ "$sweep_period" != "0" ]]; then
      # Boundary sweep: every cut offset mod the interleaving period, one
      # representative cut each (then every period up to SWEEP_MAX_CUTS cuts),
      # so planner-cut dependence (the B1 bug class) cannot hide.
      local offset cut cuts
      for (( offset = 1; offset <= sweep_period; offset++ )); do
        cuts=0
        for (( cut = offset; cut < layer_end && cuts < SWEEP_MAX_CUTS; cut += sweep_period )); do
          (( cut == base_split )) && continue
          run_certify "$family" "$target" "$model_id" "$cut" "$layer_end" "$draft"
          cuts=$((cuts + 1))
        done
      done
    fi
  done < <(grep -v '^[[:space:]]*#' "$manifest")
}

build_certification_binaries
run_manifest "$MANIFEST"

echo
echo "family battery complete: $((TOTAL - ${#FAILURES[@]}))/$TOTAL lanes passed"
if (( ${#FAILURES[@]} > 0 )); then
  printf 'failed: %s\n' "${FAILURES[@]}"
  exit 1
fi
