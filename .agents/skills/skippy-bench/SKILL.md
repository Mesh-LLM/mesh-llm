---
name: skippy-bench
description: Use this skill when running benchmark orchestration, local single-stage or split benchmarks, benchmark report flow, or performance-oriented skippy runtime checks.
metadata:
  short-description: Benchmark skippy stage runtime
---

# skippy-bench

Use this skill for performance, orchestration, and report-oriented checks.
Use `skippy-correctness` when the question is pass/fail exactness.
All reportable benchmark runs need metrics-server. `run`, `focused-runtime`,
and `local-single` start a collector by default; endpoint-driving commands such
as `chat-corpus` and `eval run` require `--metrics-http` to point at an
already-running metrics-server and should use `--metrics-run-id` matching the
target endpoint's Skippy run id.

Benchmark-managed Skippy server runs must use a release `skippy-server` build.
Run `just release-build` before `run`, `focused-runtime`, `local-single`, or
local split binary benchmarks, and use `target/release/skippy-server` (the
SkippyBench default). Do not use `target/debug/skippy-server` for performance or
full-corpus validation; SkippyBench rejects that path because debug builds can
create false timeout and throughput failures.

## Current Repo Shape

Standalone `skippy-bench` may not be present in this mesh checkout yet. Confirm
available packages before using old source-repo commands:

```bash
cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name' | sort
```

Useful current checks:

```bash
cargo test -p skippy-server --lib
cargo test -p mesh-llm-host-runtime --lib inference::skippy
```

When benchmark harnesses are imported, keep reporting separate from request-path
serving. Stage runtimes emit telemetry; benchmark/report tooling owns reports.

## External Agent Evals

Use `skippy-bench eval` for external agent/coding benchmark harnesses. The
local SkippyBench corpora are for runtime behavior, cache behavior, transport
stress, and perf regression; they are not the source of agent benchmark claims.

Core pack:

```bash
skippy-bench eval list
skippy-bench eval info terminal-bench
skippy-bench eval sync --pack core
skippy-bench eval doctor
skippy-bench eval run speed-bench \
  --base-url http://127.0.0.1:9337/v1 \
  --model org/repo:Q4_K_M \
  --endpoint-concurrency 1 \
  --metrics-http http://127.0.0.1:18080 \
  --metrics-run-id run-local-qwen
```

`--timeout-secs` is passed to the native harness as its request/task timeout
where supported. It is not a full-run dataset limit. Use
`--harness-timeout-secs` only when you need a hard wall-clock cap for an
operator/debug run; omit it for canonical full-dataset validation.
`--endpoint-concurrency` must match the target endpoint's
`serve-openai --generation-concurrency` value. SkippyBench keeps native harness
request concurrency equal to that value; adapter-specific request concurrency
overrides such as `SWE_BENCH_PRO_NUM_WORKERS` and
`MCP_ATLAS_COMPLETION_CONCURRENCY` must match it or `eval run` fails before
starting the upstream harness. Do not run multiple LLM workers against a
single-lane Skippy endpoint when validating full corpora.

Core eval ids:

- `speed-bench` — llama.cpp SPEED-Bench client for OpenAI-compatible serving
  latency/throughput. Run the upstream qualitative benchmark across all
  categories with no Skippy-owned sample limit.
- `terminal-bench` — Terminal-Bench CLI via `terminal-bench-core==0.1.1`.
- `swe-bench-pro` — Scale SWE-Bench Pro OS repo; uses the upstream data and
  SWE-agent patch generation/evaluation flow rather than a Skippy-owned mini
  benchmark.
- `mcp-atlas` — Scale MCP-Atlas native harness. `eval run` starts the
  MCP agent environment and completion service when their localhost ports are
  not already live, then runs the upstream completion script with `--no-filter`
  so all Hugging Face dataset rows are attempted, plus the upstream scoring
  path, without Skippy-specific task limits or `tool_choice` overrides.

Use-case routing:

| Need | Eval | Why |
|---|---|---|
| OpenAI-compatible serving latency, tok/s, and full SPEED-Bench traffic | `speed-bench` | Native SPEED-Bench client over the upstream dataset selection. |
| Terminal agent behavior, shell/task execution, Docker sandbox readiness | `terminal-bench` | Exercises an agent loop that has to operate in a real terminal task environment. |
| Coding-agent patch generation and issue-resolution style prompts | `swe-bench-pro` | Uses upstream SWE-agent instance generation, patch gathering, and `swe_bench_pro_eval.py`. |
| MCP tool-use benchmark flow | `mcp-atlas` | Uses upstream MCP-Atlas completion and scoring scripts with the full Hugging Face dataset. |
| Cache, runtime, transport, split, or mesh performance regression | Built-in SkippyBench `run`, `focused-runtime`, `local-single`, or `chat-corpus` | These are Skippy/runtime benchmarks, not external agent-quality claims. |

Optional future packs are intentionally not wired yet:

- `repo-generation`: NL2RepoBench.
- `tool-expanded`: Toolathlon / Tool-Decathlon.

Keep `sync`/`install` opt-in. Do not make normal `just build` or `cargo build`
download external harnesses, datasets, or Docker images.

Terminal-Bench should be installed with `uv tool install --python 3.12
terminal-bench`; Python 3.14 currently breaks the `tb` Typer CLI. Treat Docker
as ready only when `skippy-bench eval doctor` reports that the daemon can start
a container; `docker info` alone is insufficient. `skippy-bench eval run`
performs the same prerequisite checks before launching a native harness. Do not
add Skippy-owned task filters, dataset limits, compatibility shims,
response-format substitutions, or `tool_choice` overrides to external evals
unless the user explicitly asks for a noncanonical experiment.

For MCP-Atlas scoring, the wrapper defaults `EVAL_LLM_MODEL`,
`EVAL_LLM_BASE_URL`, and `EVAL_LLM_API_KEY` to the same local endpoint/model
used for completion, while preserving caller-provided `EVAL_LLM_*` overrides
for judge-model runs. When validating with a very small local Skippy model, run
completion against the normal compatibility endpoint and point `EVAL_LLM_*` at
a separate strict structured-output scorer endpoint, for example a second
`skippy-server serve-openai --openai-guardrails enforce` process; do not patch
or post-process the MCP scorer. For resumed operator runs, set
`MCP_ATLAS_COMPLETION_OUTPUT_NAME` to an existing upstream
`completion_results/*.csv` basename so the native completion script can reuse
its own processed-row skip behavior, and use `MCP_ATLAS_SCORE_CONCURRENCY` for
the scorer's native `--concurrency` setting.

For SWE-Bench Pro, the wrapper defaults to the official Docker image namespace
with local Docker deployment and local Docker evaluation so the core pack can
run without Modal credentials. It still runs upstream
`helper_code/generate_sweagent_instances.py` for the full dataset, then supplies
SWE-agent with a native `expert_file` instance file for local Docker platform,
entrypoint settings and SWE-agent's standalone Python/SWE-Rex Docker runtime.
Local Docker runs install SWE-agent into a dedicated venv and default
`SWE_BENCH_PRO_SWEREX_SPEC` to `swe-rex[modal]==1.4.0`, which keeps the native
SWE-ReX Docker runtime but includes the upstream
`python:3.11.9-slim-bookworm` builder fix. Modal remains an explicit
environment override and uses the Scale SWE-ReX patch flow. Some official
SWE-Pro base images point pip at an unavailable localhost package mirror; local
Docker runs default `SWE_BENCH_PRO_SWEREX_PIP_INDEX_URL` to
`https://pypi.org/simple` for the derived-image SWE-ReX install. Use
`SWE_BENCH_PRO_PARSE_FUNCTION=thought_action` for local OpenAI-compatible models
that do not emit OpenAI tool calls; this is the upstream SWE-agent local-model
path, not a Skippy dataset or harness rewrite.

For TTFT/FTTT, use metrics-server correlation rather than harness-only timing.
`skippy-bench eval run` and `skippy-bench chat-corpus` create/finalize a
metrics-server run and fail if the metrics report cannot be exported. The
target endpoint must be emitting OTLP for the same run id. Debug telemetry is
required for per-token spans such as `stage.openai_decode_token`; without it,
the JSON report will still include a telemetry block explaining why TTFT/FTTT
was unavailable.
