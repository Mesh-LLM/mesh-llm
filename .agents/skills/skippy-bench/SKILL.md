---
name: skippy-bench
description: Use this skill when running benchmark orchestration, local single-stage or split benchmarks, benchmark report flow, or performance-oriented skippy runtime checks.
metadata:
  short-description: Benchmark skippy stage runtime
---

# skippy-bench

New repository benchmark automation follows `../manage-ci/SKILL.md`: do not
add Python tooling. Use typed `tools/xtask` commands behind thin Just recipes;
`cargo xtool repo-consistency ci-crate-lists` works from the repository root
as an alias example, not a benchmark command. Existing upstream Python
benchmark environments below remain explicit external tooling, not a model
for new repository automation.

Use this skill for performance, orchestration, and report-oriented checks.
Use `skippy-correctness` when the question is pass/fail exactness.
All reportable benchmark runs need metrics-server. `run`, `focused-runtime`,
and `local-single` start a collector by default; endpoint-driving commands such
as `chat-corpus` and `eval run` require `--metrics-http` to point at an
already-running metrics-server and should use `--metrics-run-id` matching the
target endpoint's Skippy run id.

Benchmark-managed Skippy server runs must use a release `skippy-serving` build.
Run `just release-build` before `run`, `focused-runtime`, `local-single`, or
local split binary benchmarks, and use `target/release/skippy` (the
SkippyBench default). Do not use `target/debug/skippy` for performance or
full-corpus validation; SkippyBench rejects that path because debug builds can
create false timeout and throughput failures.

## Current repository ownership

`skippy/crates/skippy-bench` is the maintained benchmark launcher and report
owner. Its `src/evals` modules own external source admission, explicit SDK
preparation, and run supervision. Serving remains owned by `skippy-serving`;
stage runtimes emit telemetry and benchmark tooling owns reports.

Build and invoke the existing product through Just:

```bash
just with-lld cargo build --locked --release -p skippy-bench --features skippy-runtime/dynamic-native-runtime
just --command "$PWD/target/release/skippy-bench" eval list
```

Existing focused checks also use the repository toolchain wrapper:

```bash
just with-lld cargo test --locked -p skippy-serving --lib
just with-lld cargo test --locked -p mesh-llm-host-runtime --lib inference::skippy
```

Select the native ABI/runtime appropriate to the behavior under test. These
commands do not establish model, platform, or benchmark qualification by
themselves.

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

Keep source sync and SDK preparation opt-in. Normal builds must not download
external harnesses, datasets, or Docker images. MCP/SWE source refs are fixed
immutable pins; other evals retain their documented source requirements. Every
run records the resolved `harness_commit`.

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
`skippy-serving serve-openai --guardrails enforce` process; do not patch
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
The prepared SDK retains `swe-rex[modal]==1.4.0`. Docker deployment is the
default; Modal is an explicit prepared profile. Use
`SWE_BENCH_PRO_PARSE_FUNCTION=thought_action` for local OpenAI-compatible
models without OpenAI tool calls, preserving the upstream local-model path.

## Explicit MCP and SWE SDK preparation

MCP-Atlas and SWE-Bench Pro use pinned source checkouts and explicitly prepared
SDK environments. `sync` acquires source and MCP's pinned agent image; it
does not prepare these SDKs. Preparation is opt-in on Unix and takes existing
absolute `uv` and Python executable paths. It never downloads an interpreter. Use one explicit cache root for sync,
prepare, doctor, and run:

```bash
just with-lld cargo build --locked --release -p skippy-bench --features skippy-runtime/dynamic-native-runtime
GIT_MASTER=1 just --command "$PWD/target/release/skippy-bench" eval sync mcp-atlas --cache-root "$CACHE_ROOT"
GIT_MASTER=1 just --command "$PWD/target/release/skippy-bench" eval prepare-mcp \
  --cache-root "$CACHE_ROOT" --uv "$UV_EXECUTABLE" --python "$CPYTHON_312_EXECUTABLE"
GIT_MASTER=1 just --command "$PWD/target/release/skippy-bench" eval sync swe-bench-pro --cache-root "$CACHE_ROOT"
GIT_MASTER=1 just --command "$PWD/target/release/skippy-bench" eval prepare-swe \
  --cache-root "$CACHE_ROOT" --uv "$UV_EXECUTABLE" --python "$CPYTHON_31113_EXECUTABLE" \
  --deployment docker --index-url https://pypi.org/simple
```

`CACHE_ROOT` and tool paths must be absolute. MCP requires existing CPython
3.12; SWE requires exactly CPython 3.11.13. `--dry-run` inspects preparation
without installing or publishing a usable receipt. A preparation destination
must be fresh; retain failed evidence or choose another cache root rather than
reusing a partial environment. Run dependency setup only through these native
owners. They use locked projects, bounded child capture and execution, exact
source admission, and environment seals. The prepared Python runs with `-I -B`.
Normal `eval run` admits the receipt and its source/environment before and after
execution; it does not resolve, install, or patch SDK dependencies.

MCP is pinned to `b290e672645791fea0bcb23e2c0f4fec50715cca`. SWE parent is pinned
to `66f92766bba642462d4bbe5479e83f91f9211862`, with SWE-agent gitlink
`402a7b8fdac8193f3f255bb53859ba274234f596`. SWE's source-owned lock retains
`swe-rex[modal]==1.4.0`. Preparation receipts record `benchmark_qualified=false`.
Source admission, SDK import checks, and native tests do not establish dataset,
endpoint, model, platform, Docker/Modal deployment, or benchmark results.

For Modal, explicitly prepare with `--deployment modal`, then set
`SWE_BENCH_PRO_DEPLOYMENT_TYPE=modal` for the run. The runtime deployment and
`SWE_BENCH_PRO_SWEREX_PIP_INDEX_URL` must match the prepared profile. The default
is Docker and `https://pypi.org/simple`. The index must be credential-free
HTTP(S) without a query or fragment. Native preparation applies only the finite
Docker index or Modal bootstrap/retry patch profile before sealing the SDK.
Conflicting `SWE_BENCH_PRO_PYTHON` or `SWE_BENCH_PRO_SWEREX_SPEC` selectors are
refused; they cannot replace the locked interpreter or package graph.

```bash
just --command "$PWD/target/release/skippy-bench" eval run swe-bench-pro \
  --cache-root "$CACHE_ROOT" --base-url http://127.0.0.1:9337/v1 \
  --model org/repo:Q4_K_M --endpoint-concurrency 1 \
  --metrics-http http://127.0.0.1:18080 --metrics-run-id run-local-qwen
```

The same run form supports `mcp-atlas`. A real run still needs its endpoint,
metrics-server, Docker services, credentials where applicable, and full upstream
dataset. Preparation starts no benchmark services and runs no model requests.

For TTFT/FTTT, use metrics-server correlation rather than harness-only timing.
`skippy-bench eval run` and `skippy-bench chat-corpus` create/finalize a
metrics-server run. `eval run` keeps harness success independent from a
finalization/export failure and records telemetry as unavailable; `chat-corpus`
still fails when its metrics report cannot be exported. The target endpoint
must be emitting OTLP for the same run id. Debug telemetry is required for
per-token spans such as `stage.openai_decode_token`; without it, the JSON report
will still include a telemetry block explaining why TTFT/FTTT was unavailable.
