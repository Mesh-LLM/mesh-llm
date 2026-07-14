use super::super::{
    registry::definition,
    run::{mcp_atlas_output_path, mcp_atlas_score_dir},
    *,
};

pub(in crate::evals) fn mcp_atlas_command(
    args: &EvalRunArgs,
    root: &Path,
    run_dir: &Path,
) -> Result<CommandSpec> {
    let harness = harness_dir(root, definition(EvalId::McpAtlas));
    let script = run_dir.join("raw/mcp-atlas-run.sh");
    write_mcp_atlas_run_script(&script, args, root, &harness, run_dir)?;
    Ok(CommandSpec::new("zsh").args([script.display().to_string()]))
}

fn write_mcp_atlas_run_script(
    path: &Path,
    args: &EvalRunArgs,
    cache_root: &Path,
    harness: &Path,
    run_dir: &Path,
) -> Result<()> {
    let raw_dir = run_dir.join("raw");
    let completion_dir = harness.join("services/mcp_eval");
    let default_output_name = format!("skippybench-{}-completion-results.csv", unix_millis()?);
    let model_label = model_label(&args.model);
    let score_dir = mcp_atlas_score_dir(run_dir);
    let completion_concurrency = env_concurrency_matching_endpoint(
        "MCP_ATLAS_COMPLETION_CONCURRENCY",
        args.endpoint_concurrency,
    )?;
    let script = format!(
        r#"#!/usr/bin/env zsh
set -euo pipefail

HARNESS={harness}
COMPLETION_DIR={completion_dir}
RAW_DIR={raw_dir}
BASE_URL={base_url}
API_KEY={api_key}
MODEL={model}
EVAL_MODEL=${{EVAL_LLM_MODEL:-$MODEL}}
OUTPUT={output}
OUTPUT_NAME=${{MCP_ATLAS_COMPLETION_OUTPUT_NAME:-{output_name}}}
MODEL_LABEL={model_label}
SCORE_DIR={score_dir}
COMPLETION_CONCURRENCY={completion_concurrency}
SCORE_CONCURRENCY=${{MCP_ATLAS_SCORE_CONCURRENCY:-5}}
HF_HOME_DIR={hf_home}
HF_DATASETS_CACHE_DIR={hf_datasets_cache}
UV_CACHE_DIR_LOCAL={uv_cache_dir}
XDG_CACHE_HOME_DIR={xdg_cache_home}
agent_started=0
completion_started=0
completion_pid=""

mkdir -p \
  "$HF_HOME_DIR" \
  "$HF_DATASETS_CACHE_DIR" \
  "$UV_CACHE_DIR_LOCAL" \
  "$XDG_CACHE_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR_LOCAL"
export XDG_CACHE_HOME="$XDG_CACHE_HOME_DIR"

port_ready() {{
  python3 - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}}

wait_url() {{
  local name="$1"
  local url="$2"
  local log="$3"
  for _ in {{1..90}}; do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    tail -20 "$log" 2>/dev/null || true
    sleep 2
  done
  echo "timed out waiting for $name at $url" >&2
  return 1
}}

cleanup() {{
  if [[ "$completion_started" == "1" && -n "$completion_pid" ]]; then
    kill "$completion_pid" >/dev/null 2>&1 || true
    wait "$completion_pid" >/dev/null 2>&1 || true
  fi
  if [[ "$agent_started" == "1" ]]; then
    docker rm -f skippy-bench-mcp-atlas-agent-env >/dev/null 2>&1 || true
  fi
}}
trap cleanup EXIT

mkdir -p "$RAW_DIR" "$SCORE_DIR"
cd "$HARNESS"
cp -n env.template .env >/dev/null 2>&1 || true
if ! docker image inspect agent-environment:latest >/dev/null 2>&1; then
  docker tag ghcr.io/scaleapi/mcp-atlas:1.2.5 agent-environment:latest
fi

if ! port_ready 1984; then
  docker rm -f skippy-bench-mcp-atlas-agent-env >/dev/null 2>&1 || true
  docker run --rm \
    --name skippy-bench-mcp-atlas-agent-env \
    -p 1984:1984 \
    --env-file .env \
    agent-environment:latest \
    > "$RAW_DIR/mcp-agent-env.log" 2>&1 &
  agent_started=1
fi
wait_url "MCP-Atlas agent environment" \
  "http://localhost:1984/enabled-servers" \
  "$RAW_DIR/mcp-agent-env.log"

if ! port_ready 3000; then
  (
    cd "$COMPLETION_DIR"
    LLM_BASE_URL="$BASE_URL" \
      LLM_API_KEY="$API_KEY" \
      OPENAI_BASE_URL="$BASE_URL" \
      OPENAI_API_KEY="$API_KEY" \
      uv run python -m mcp_completion.main
  ) > "$RAW_DIR/mcp-completion.log" 2>&1 &
  completion_pid="$!"
  completion_started=1
fi
wait_url "MCP-Atlas completion service" \
  "http://localhost:3000/docs" \
  "$RAW_DIR/mcp-completion.log"

cd "$COMPLETION_DIR"
LLM_BASE_URL="$BASE_URL" \
  LLM_API_KEY="$API_KEY" \
  OPENAI_BASE_URL="$BASE_URL" \
  OPENAI_API_KEY="$API_KEY" \
uv run python mcp_completion_script.py \
    --model "$MODEL" \
    --input_huggingface ScaleAI/MCP-Atlas \
    --output "$OUTPUT_NAME" \
    --no-filter \
    --concurrency "$COMPLETION_CONCURRENCY"
cp "completion_results/$OUTPUT_NAME" "$OUTPUT"
EVAL_LLM_BASE_URL="${{EVAL_LLM_BASE_URL:-$BASE_URL}}" \
  EVAL_LLM_API_KEY="${{EVAL_LLM_API_KEY:-$API_KEY}}" \
uv run python mcp_evals_scores.py \
    --input-file "completion_results/$OUTPUT_NAME" \
    --model-label "$MODEL_LABEL" \
    --evaluator-model "$EVAL_MODEL" \
    --output-dir "$SCORE_DIR" \
    --concurrency "$SCORE_CONCURRENCY"
"#,
        harness = shell_quote(&harness.display().to_string()),
        completion_dir = shell_quote(&completion_dir.display().to_string()),
        raw_dir = shell_quote(&raw_dir.display().to_string()),
        base_url = shell_quote(&args.base_url),
        api_key = shell_quote(&args.api_key),
        model = shell_quote(&litellm_model_name(&args.model)),
        output = shell_quote(&mcp_atlas_output_path(run_dir).display().to_string()),
        output_name = shell_quote(&default_output_name),
        model_label = shell_quote(&model_label),
        score_dir = shell_quote(&score_dir.display().to_string()),
        completion_concurrency = shell_quote(&completion_concurrency),
        hf_home = shell_quote(&cache_root.join("hf").display().to_string()),
        hf_datasets_cache = shell_quote(&cache_root.join("hf-datasets").display().to_string()),
        uv_cache_dir = shell_quote(&cache_root.join("uv").display().to_string()),
        xdg_cache_home = shell_quote(&cache_root.join("xdg").display().to_string()),
    );
    fs::write(path, script).with_context(|| format!("write {}", path.display()))
}
