#!/usr/bin/env zsh
set -euo pipefail

HARNESS={harness}
PREPARED_PYTHON={prepared_python}
SDK_GENERATE={sdk_generate}
SDK_EXPERT={sdk_expert}
SDK_EVALUATE={sdk_evaluate}
PREPARED_AGENT={prepared_agent}
RAW_DIR={raw_dir}
INSTANCES={instances}
EXPERT_INSTANCES={expert_instances}
SWEAGENT_OUTPUT={sweagent_output}
PATCHES={patches}
EVAL_DIR={eval_dir}
MODEL={model}
BASE_URL={base_url}
API_KEY="${{SKIPPY_BENCH_API_KEY:?SKIPPY_BENCH_API_KEY is required}}"
DOCKERHUB_USERNAME={dockerhub_username}
DEPLOYMENT_TYPE={deployment_type}
NUM_WORKERS={num_workers}
EVAL_WORKERS={eval_workers}
LOCAL_EVAL_FLAG={local_eval_flag}
DOCKER_PLATFORM={docker_platform}
PARSE_FUNCTION={parse_function}
HF_HOME_DIR={hf_home}
HF_DATASETS_CACHE_DIR={hf_datasets_cache}
UV_CACHE_DIR_LOCAL={uv_cache_dir}
XDG_CACHE_HOME_DIR={xdg_cache_home}

mkdir -p \
  "$RAW_DIR" \
  "$SWEAGENT_OUTPUT" \
  "$EVAL_DIR" \
  "$HF_HOME_DIR" \
  "$HF_DATASETS_CACHE_DIR" \
  "$UV_CACHE_DIR_LOCAL" \
  "$XDG_CACHE_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR_LOCAL"
export XDG_CACHE_HOME="$XDG_CACHE_HOME_DIR"
deployment_timeout_args=()
if [[ "$DEPLOYMENT_TYPE" == "modal" ]]; then
  deployment_timeout_args=(
    --instances.deployment.startup_timeout 1800
    --instances.deployment.runtime_timeout 3600
  )
fi
deployment_platform_args=()
if [[ "$DEPLOYMENT_TYPE" == "docker" && -n "$DOCKER_PLATFORM" ]]; then
  deployment_platform_args=(
    --instances.deployment.platform "$DOCKER_PLATFORM"
  )
fi
deployment_type_args=(
  --instances.deployment.type "$DEPLOYMENT_TYPE"
)
expert_instance_args=(
  --instances.type file
  --instances.path "$INSTANCES"
)
parse_function_args=()
if [[ -n "$PARSE_FUNCTION" ]]; then
  parse_function_args=(
    --agent.tools.parse_function.type "$PARSE_FUNCTION"
  )
fi
cd "$HARNESS"

# Fixed SDK generator needs its pinned sibling modules under isolated Python.
"$PREPARED_PYTHON" -I -B "$SDK_GENERATE" "$HARNESS" \
    --dockerhub_username "$DOCKERHUB_USERNAME" \
    --output_path "$INSTANCES"

if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
  (
    cd "$PREPARED_AGENT"
    "$PREPARED_PYTHON" -I -B "$SDK_EXPERT" "$INSTANCES" "$EXPERT_INSTANCES" "$DOCKER_PLATFORM"
  )
  expert_instance_args=(
    --instances.type expert_file
    --instances.path "$EXPERT_INSTANCES"
  )
  deployment_type_args=()
  deployment_platform_args=()
fi

(
  cd "$PREPARED_AGENT"
  OPENAI_BASE_URL="$BASE_URL" \
  OPENAI_API_KEY="$API_KEY" \
  "$PREPARED_PYTHON" -I -B -m sweagent.run.run run-batch \
    --config config/tool_use.yaml \
    --output_dir "$SWEAGENT_OUTPUT" \
    --num_workers "$NUM_WORKERS" \
    --random_delay_multiplier 1 \
    "${{expert_instance_args[@]}}" \
    --instances.shuffle=False \
    "${{deployment_type_args[@]}}" \
    "${{deployment_timeout_args[@]}}" \
    "${{deployment_platform_args[@]}}" \
    "${{parse_function_args[@]}}" \
    --agent.model.name "$MODEL" \
    --agent.model.api_base "$BASE_URL" \
    --agent.model.max_input_tokens 0 \
    --agent.model.per_instance_cost_limit 0 \
    --agent.model.total_cost_limit 0
)

"$PREPARED_PYTHON" -I -B "$HARNESS/helper_code/gather_patches.py" \
    --directory "$SWEAGENT_OUTPUT" \
    --prefix skippybench \
    --output "$PATCHES"

"$PREPARED_PYTHON" -I -B "$SDK_EVALUATE" "$HARNESS" \
    --raw_sample_path helper_code/sweap_eval_full_v2.jsonl \
    --patch_path "$PATCHES" \
    --output_dir "$EVAL_DIR" \
    --scripts_dir run_scripts \
    --num_workers "$EVAL_WORKERS" \
    --dockerhub_username "$DOCKERHUB_USERNAME" \
    $LOCAL_EVAL_FLAG
