#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

skippy_quantize_bin="${SKIPPY_QUANTIZE_BIN:-$repo_root/target/release/skippy-quantize}"
skippy_model_package_bin="${SKIPPY_MODEL_PACKAGE_BIN:-$repo_root/target/release/skippy-model-package}"

work_base="${WORK_BASE:-/Users/lab/glm52-work/q2-routed-down}"
source_root="${SOURCE_ROOT:-/Users/lab/glm52-work/bf16-gguf}"
source_prefix="${SOURCE_PREFIX:-BF16}"
target_root="${TARGET_ROOT:-$work_base/quant-scratch}"
target_prefix="${TARGET_PREFIX:-Q2_K-RoutedDown-MTP-Q8}"
output_basename="${OUTPUT_BASENAME:-GLM-5.2-Q2_K-RoutedDown-MTP-Q8}"
manifest="${MANIFEST:-$work_base/quant-manifest.json}"
package_dir="${PACKAGE_DIR:-$work_base/package/GLM-5.2-Q2_K-RoutedDown-MTP-Q8-layers}"
package_model_id="${PACKAGE_MODEL_ID:-meshllm/GLM-5.2-Q2_K-RoutedDown-MTP-Q8-GGUF:Q2_K-RoutedDown-MTP-Q8}"
package_source_repo="${PACKAGE_SOURCE_REPO:-meshllm/GLM-5.2-Q2_K-RoutedDown-MTP-Q8-GGUF}"
package_source_revision="${PACKAGE_SOURCE_REVISION:-local}"
recipe="${TENSOR_TYPE_FILE:-$repo_root/recipes/quantization/glm-5.2-q2-k-routed-down-mtp-q8.tensor-types.txt}"
work_dir="${WORK_DIR:-$work_base/native-work}"
spool_dir="${SPOOL_DIR:-$work_base/spool}"
record_dir="${RECORD_DIR:-$work_base/records}"
status_file="${JSON_EVENT_FILE:-$work_base/status.json}"
shard_scratch_dir="${SHARD_SCRATCH_DIR:-$work_dir/shard-scratch}"
stages="${STAGES:-2}"
nthreads="${NTHREADS:-}"
dry_run="${DRY_RUN:-0}"

if [[ ! -x "$skippy_quantize_bin" ]]; then
  echo "missing executable: $skippy_quantize_bin" >&2
  echo "build it with: just skippy-quantize-standalone-release-build" >&2
  exit 1
fi

if [[ "$dry_run" != "1" && ! -x "$skippy_model_package_bin" ]]; then
  echo "missing executable: $skippy_model_package_bin" >&2
  echo "build it with: cargo build --release --locked -p skippy-model-package" >&2
  exit 1
fi

if [[ ! -f "$recipe" ]]; then
  echo "missing tensor-type recipe: $recipe" >&2
  exit 1
fi

if [[ ! -d "$source_root/$source_prefix" ]]; then
  echo "missing BF16 source prefix: $source_root/$source_prefix" >&2
  echo "point SOURCE_ROOT/SOURCE_PREFIX at the local BF16 GGUF shards before running" >&2
  exit 1
fi

common_args=(
  --source "$source_root"
  --source-prefix "$source_prefix"
  --target "$target_root"
  --target-prefix "$target_prefix"
  --output-basename "$output_basename"
  --quant Q2_K
  --tensor-type-file "$recipe"
  --window-size 1
  --manifest "$manifest"
  --backend llama-api
  --work-dir "$work_dir"
  --spool-dir "$spool_dir"
  --record-dir "$record_dir"
  --json-event-file "$status_file"
  --json-event-interval-seconds 120
  --json-event-window 8
  --watchdog-seconds 120
)

if [[ -n "$nthreads" ]]; then
  common_args+=(--nthreads "$nthreads")
fi

if [[ "$dry_run" == "1" ]]; then
  exec "$skippy_quantize_bin" quant-job "${common_args[@]}" --preflight-only
fi

export SKIPPY_MODEL_PACKAGE_SHARD_SCRATCH_DIR="$shard_scratch_dir"

exec "$skippy_quantize_bin" quantize-layer-package \
  "${common_args[@]}" \
  --package-dir "$package_dir" \
  --package-model-id "$package_model_id" \
  --package-source-repo "$package_source_repo" \
  --package-source-revision "$package_source_revision" \
  --skippy-model-package-bin "$skippy_model_package_bin" \
  --stages "$stages" \
  --replace-package
