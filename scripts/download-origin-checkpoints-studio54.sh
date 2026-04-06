#!/bin/zsh

set -u

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export a Hugging Face token before running this script." >&2
  exit 1
fi

export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

TARGET_ROOT="${TARGET_ROOT:-$HOME/.cache/mesh-llm-origin-batch}"
LOG_ROOT="${LOG_ROOT:-$TARGET_ROOT/_logs}"

mkdir -p "$TARGET_ROOT" "$LOG_ROOT"

typeset -a HF_CANDIDATES
HF_CANDIDATES=(
  "${HF_BIN:-}"
  "hf"
  "$HOME/Library/Python/3.9/bin/hf"
  "$HOME/Library/Python/3.10/bin/hf"
  "$HOME/Library/Python/3.11/bin/hf"
  "$HOME/Library/Python/3.12/bin/hf"
  "/opt/homebrew/bin/hf"
  "/usr/local/bin/hf"
)

HF_BIN_RESOLVED=""
for candidate in "${HF_CANDIDATES[@]}"; do
  if [[ -z "$candidate" ]]; then
    continue
  fi

  if [[ "$candidate" == */* ]]; then
    if [[ -x "$candidate" ]]; then
      HF_BIN_RESOLVED="$candidate"
      break
    fi
  elif command -v "$candidate" >/dev/null 2>&1; then
    HF_BIN_RESOLVED="$(command -v "$candidate")"
    break
  fi
done

if [[ -z "$HF_BIN_RESOLVED" ]]; then
  echo "Could not find 'hf'. Set HF_BIN=/absolute/path/to/hf if needed." >&2
  exit 1
fi

HF_BIN="$HF_BIN_RESOLVED"

typeset -a SPECS
SPECS=(
  "deepseek|deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "olmo|allenai/OLMo-1B-hf"
  "mamba2|state-spaces/mamba2-2.7b"
  "phi3|microsoft/Phi-3-mini-4k-instruct"
  "phi4-mini|microsoft/Phi-4-mini-instruct"
  "minicpm|openbmb/MiniCPM3-4B"
  "mamba|state-spaces/mamba-2.8b-hf"
  "starcoder2|bigcode/starcoder2-3b"
  "olmo2|allenai/OLMo-2-1124-7B-Instruct"
  "cohere2|CohereLabs/c4ai-command-r7b-12-2024"
  "mistral|mistralai/Mistral-7B-Instruct-v0.3"
)

typeset -a HEAVY_SPECS
HEAVY_SPECS=(
  "cohere-command-r|CohereLabs/c4ai-command-r-v01"
  "jamba|ai21labs/AI21-Jamba-1.5-Mini"
  "mixtral|mistralai/Mixtral-8x7B-Instruct-v0.1"
  "kimi-linear|moonshotai/Kimi-Linear-48B-A3B-Instruct"
)

function usage() {
  cat <<'EOF'
Usage:
  zsh scripts/download-origin-checkpoints-studio54.sh
  zsh scripts/download-origin-checkpoints-studio54.sh mistral phi3 deepseek
  zsh scripts/download-origin-checkpoints-studio54.sh --include-heavy
  TARGET_ROOT=~/mesh-origin zsh scripts/download-origin-checkpoints-studio54.sh

Notes:
  - Downloads run serially with hf download into per-model directories.
  - The default set excludes families that are too large or risky for the
    full same-origin test workflow on studio54's 128 GB M1 Ultra.
  - Use --include-heavy to add: mixtral, cohere-command-r, jamba, kimi-linear.
  - If one repo fails or is gated, the script continues and reports it at the end.
  - Some repos in this list are very large and can consume substantial disk.
EOF
}

typeset -a FILTERS
FILTERS=()
INCLUDE_HEAVY=0

while (( $# > 0 )); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --include-heavy)
      INCLUDE_HEAVY=1
      ;;
    *)
      FILTERS+=("$1")
      ;;
  esac
  shift
done

function should_download() {
  local slug="$1"

  if (( ${#FILTERS[@]} == 0 )); then
    return 0
  fi

  local filter
  for filter in "${FILTERS[@]}"; do
    if [[ "$slug" == "$filter" ]]; then
      return 0
    fi
  done

  return 1
}

function print_progress_snapshot() {
  local target_dir="$1"

  if [[ ! -d "$target_dir" ]]; then
    return
  fi

  echo "  progress: $(du -sh "$target_dir" 2>/dev/null | awk '{print $1}') downloaded so far"

  local files
  files=$(find "$target_dir" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.gguf' \) -print 2>/dev/null | sort)
  if [[ -n "$files" ]]; then
    echo "$files" | xargs ls -lh 2>/dev/null | tail -n 3 | sed 's/^/    /'
  fi
}

function run_download() {
  local repo="$1"
  local target_dir="$2"
  local log_file="$3"

  "$HF_BIN" download "$repo" \
    --token "$HF_TOKEN" \
    --local-dir "$target_dir" \
    > >(tee "$log_file") \
    2> >(tee -a "$log_file" >&2) &

  local download_pid=$!

  while kill -0 "$download_pid" >/dev/null 2>&1; do
    sleep 20
    if kill -0 "$download_pid" >/dev/null 2>&1; then
      print_progress_snapshot "$target_dir"
    fi
  done

  wait "$download_pid"
}

typeset -a SUCCEEDED
typeset -a FAILED
typeset -a SKIPPED
typeset -i index=1

echo "Using hf binary: $HF_BIN"
echo "Target root: $TARGET_ROOT"
echo "Log root: $LOG_ROOT"
if (( INCLUDE_HEAVY == 1 )); then
  echo "Heavy families: included"
else
  echo "Heavy families: excluded by default"
fi
echo

typeset -a ALL_SPECS
ALL_SPECS=("${SPECS[@]}")
if (( INCLUDE_HEAVY == 1 )); then
  ALL_SPECS+=("${HEAVY_SPECS[@]}")
fi

for spec in "${ALL_SPECS[@]}"; do
  slug="${spec%%|*}"
  repo="${spec#*|}"

  if ! should_download "$slug"; then
    SKIPPED+=("$slug")
    continue
  fi

  target_dir="$TARGET_ROOT/$slug"
  log_file="$LOG_ROOT/$slug.log"

  mkdir -p "$target_dir"

  echo "[$index/${#ALL_SPECS[@]}] Downloading $slug from $repo"
  echo "  target: $target_dir"
  echo "  log:    $log_file"

  if run_download "$repo" "$target_dir" "$log_file"; then
    SUCCEEDED+=("$slug")
    echo "  status: ok"
  else
    FAILED+=("$slug")
    echo "  status: failed"
  fi

  echo
  index+=1
done

echo "Summary"
echo "  succeeded: ${#SUCCEEDED[@]}"
if (( ${#SUCCEEDED[@]} > 0 )); then
  printf '    %s\n' "${SUCCEEDED[@]}"
fi

echo "  failed: ${#FAILED[@]}"
if (( ${#FAILED[@]} > 0 )); then
  printf '    %s\n' "${FAILED[@]}"
fi

echo "  skipped: ${#SKIPPED[@]}"
if (( ${#SKIPPED[@]} > 0 )); then
  printf '    %s\n' "${SKIPPED[@]}"
fi

if (( ${#FAILED[@]} > 0 )); then
  exit 1
fi
