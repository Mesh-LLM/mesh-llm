# skippy-bench

Benchmark launcher and local smoke harness.

This crate is for orchestration and performance measurement. Exactness checks
should move toward `skippy-correctness`; existing local split commands
remain useful while the production tool is being promoted.

## Architecture Role

`skippy-bench` launches and measures the same binary stage chain used by mesh.
It can materialize or rsync stage artifacts, start remote stage servers, point
them at `metrics-server`, drive prompt prefill/decode against the first stage,
drive OpenAI corpus requests through the shared frontend, and collect
`driver-result.json` plus `report.json`.

```mermaid
flowchart LR
    B["skippy-bench<br/>driver + launcher"] --> O["skippy-inference-api<br/>optional corpus path"]
    B --> S0["stage-0"]
    O --> S0
    S0 -->|activation frames| S1["stage-1"]
    S1 -->|activation frames| S2["stage-2"]
    S2 -->|activation frames| S3["final stage"]
    S3 -->|predicted token / ACK| S2
    S2 -->|predicted token / ACK| S1
    S1 -->|predicted token / ACK| S0
    S0 -->|predicted token / ACK| B

    S0 -.-> M["metrics-server"]
    S1 -.-> M
    S2 -.-> M
    S3 -.-> M
    B -.-> M
    M --> R["report.json"]
    B --> D["driver-result.json"]
```

Benchmarks should be read through the staged data path: prompt/control bytes are
small, predicted-token replies are small, and boundary activation frames
dominate transfer volume. Prefill experiments usually focus on layer balance,
chunk size, credit settings, and optional async
prefill-forward overlap. Decode is measured too, but current optimization work
should not assume decode is the bottleneck until the report says so.

## Commands

```bash
skippy-bench run --stage-model skippy-model-package/ --model-id org/repo:Q4_K_M
skippy-bench run --stage-model skippy-model-package/ --cache-type-k q8_0 --cache-type-v q8_0
skippy-bench local-single --model-path model.gguf --model-id org/repo:Q4_K_M
skippy-bench local-split-binary --model-path model.gguf --model-id org/repo:Q4_K_M
skippy-bench local-split-compare --model-path model.gguf --model-id org/repo:Q4_K_M
skippy-bench local-split-chain-binary --model-path model.gguf --model-id org/repo:Q4_K_M
skippy-bench chat-corpus --base-url http://127.0.0.1:9337/v1 --model org/repo:Q4_K_M --metrics-http http://127.0.0.1:18080 --metrics-run-id run-local-qwen --prompt-corpus target/bench-corpora/smoke/corpus.jsonl --max-tokens 64 --stream
skippy-bench token-lengths --model-path model.gguf --prompt-corpus target/bench-corpora/long/corpus.jsonl --ctx-size 8192 --generation-limit 512 --output-tsv target/bench-corpora/long/prompt-lengths.tsv
skippy-bench focused-runtime --schema-smoke --hosts host-a,host-b --splits 1 --layer-end 2
skippy-bench eval list
skippy-bench eval sync --pack core
skippy-bench eval run speed-bench --base-url http://127.0.0.1:9337/v1 --model org/repo:Q4_K_M --metrics-http http://127.0.0.1:18080 --metrics-run-id run-local-qwen
```

SPEED uses a fixed llama.cpp source revision and an explicit optional prepared
SDK. Set `MESH_PYTHON_RESEARCH_SOURCE` to the restored, pinned research checkout
before preparation or execution. With existing absolute uv and CPython 3.12
paths, prepare the locked dependencies and pinned qualitative dataset once:

```bash
GIT_MASTER=1 just --command "$PWD/target/release/skippy-bench" eval sync speed-bench --cache-root "$PWD/.cache/evals"
GIT_MASTER=1 just --command "$PWD/target/release/skippy-bench" eval prepare-speed --cache-root "$PWD/.cache/evals" --uv /absolute/path/to/uv --python /absolute/path/to/python3.12
```

Preparation is bounded and records `benchmark_qualified=false`. Runtime admits
the source, regular dataset bytes and sealed environment, then uses that Python
directly with offline dataset caches. It performs no uv resolution, installation
or Hugging Face dataset acquisition. Actual benchmark requests still go to the
selected endpoint and require the existing metrics-server cadence.

`local-single` starts a public inference endpoint and drives `/v1/completions`.
Split workers communicate only over the binary stage protocol.

Benchmark-managed Skippy server runs require a release `skippy-serving` binary.
Run `just release-build` before `run`, `focused-runtime`, `local-single`, or
local split binary benchmarks. These commands default to
`target/release/skippy` and reject `target/debug/skippy` because
debug builds distort throughput and timeout behavior.

The old standalone `kv-stage-integration` and `kv-hit-regression` commands are
intentionally absent. Mesh does not carry the legacy standalone cache sidecar
path; exact cache work should be reintroduced through the embedded runtime and
mesh-owned lifecycle.

Every reportable benchmark path must use metrics-server. `run`, `focused-runtime`,
and `local-single` launch their own collector by default through
`--metrics-server-bin`, `--metrics-http-addr`, and `--metrics-otlp-grpc-addr`.
Endpoint-driving benchmarks (`chat-corpus` and `eval run`) require an existing
metrics-server at `--metrics-http` and fail before running traffic if the run
cannot be created. For correlated server-side TTFT/FTTT, launch the target
Skippy/OpenAI endpoint so it exports OTLP to that collector with the same
`--metrics-run-id`.

```bash
target/debug/metrics-server serve \
  --db /tmp/skippy-bench-metrics.duckdb \
  --http-addr 127.0.0.1:18080 \
  --otlp-grpc-addr 127.0.0.1:14317
```

Benchmark reports carry `model_identity` beside the public `model_id`. The
public id is a coordinate such as `org/repo:Q4_K_M`; when the model path comes
from the Hugging Face cache, that resolved identity is used for stage configs
and reports, including repo, revision, source file, canonical ref, distribution
id, and selector. Arbitrary local paths are treated as artifact locations, not
as identity, so pass `--model-id` for those runs.

`run` and `local-single` accept `--cache-type-k` and `--cache-type-v`, defaulting
to `f16`. These are written into generated stage configs so benchmark reports
can compare baseline K/V cache storage against runtime-supported package candidates
such as `q8_0`. The experimental TCQ/TurboQuant lane is intentionally not
compiled into mesh-llm.

## External Agent Evals

`skippy-bench eval` manages external benchmark harnesses and points them at an
already-running OpenAI-compatible Skippy or mesh endpoint. External evals are
for agent/coding benchmark claims; the local corpora below remain runtime,
cache, routing, and transport stress traffic.

The current core pack is:

| Eval id | External harness | Default run |
|---|---|---|
| `speed-bench` | llama.cpp `tools/server/bench/speed-bench` | Native SPEED-Bench qualitative run across all categories, no sample limit, `--osl 1024` |
| `terminal-bench` | Pinned Harbor (`ff69e554`) | Harbor `terminal-bench@2.0` dataset with Terminus2 |
| `swe-gym` | Pinned Harbor (`ff69e554`) | SWE-Gym Lite via Harbor `swegym-lite`; use `--task-id` for one task |
| `swe-bench-pro` | Scale SWE-Bench Pro OS repo | Upstream SWE-agent patch generation, patch gathering, and `swe_bench_pro_eval.py` |
| `mcp-atlas` | Scale MCP-Atlas repo | Native MCP-Atlas completion script with upstream `--no-filter`, plus scoring through auto-started MCP services |

```bash
skippy-bench eval list
skippy-bench eval info terminal-bench
skippy-bench eval sync --pack core
skippy-bench eval doctor
skippy-bench eval run terminal-bench \
  --base-url http://127.0.0.1:9337/v1 \
  --model org/repo:Q4_K_M \
  --metrics-http http://127.0.0.1:18080 \
  --metrics-run-id run-local-qwen
```

SWE-Gym Lite one-task smoke:

```bash
skippy-bench eval sync swe-gym
skippy-bench eval run swe-gym \
  --task-id getmoto__moto-5752 \
  --dataset lite \
  --base-url http://127.0.0.1:9337/v1 \
  --session-id swegym-smoke \
  --model org/repo:Q4_K_M \
  --metrics-http http://127.0.0.1:18080 \
  --metrics-run-id swegym-smoke
```

Start `metrics-server` before the run for request correlation. Full SWE-Gym
runs use Harbor's official `-d swegym-lite` dataset path. Single-task
preparation uses `uv run --with swebench adapters/swegym/run_adapter.py`.
If the task container cannot reach Mesh, provide `--harbor-endpoint-url` with
a container-reachable URL.

`--timeout-secs` is forwarded to native harnesses as their request/task timeout
where supported. It is not a SkippyBench dataset limit and does not cap full
canonical runs. Use `--harness-timeout-secs` only when an operator wants a hard
wall-clock cap around the native harness process for debugging or CI guardrails.
`--endpoint-concurrency` declares the target OpenAI endpoint's generation
concurrency and defaults to `1`. SkippyBench keeps each external harness's LLM
request concurrency equal to that value. If an adapter-specific request
concurrency override such as `SWE_BENCH_PRO_NUM_WORKERS` or
`MCP_ATLAS_COMPLETION_CONCURRENCY` is set to a different value, `eval run`
fails before launching the native harness.

MCP-Atlas and SWE-Bench Pro `sync` acquire pinned source without installing
their Python SDK environments. MCP sync also pulls its pinned agent image;
other evals retain their existing tool/dependency setup. Use the explicit SDK
preparation below for MCP/SWE. Each run records the
resolved source commit as `harness_commit` in `run.json`.
Before launching native harness traffic, `eval run` enforces the same required
tool checks as `eval doctor`, including Docker container-start readiness for
Docker-backed evals.
Harbor is synced once at pinned commit `ff69e554` and reused by both
Terminal-Bench and SWE-Gym; runs do not clone it per invocation. The legacy
direct `tb` runner is unsupported. `eval doctor` checks that Docker's daemon is
reachable and can start a tiny container, not just that the
`docker` CLI exists or that `docker info` returns.
MCP-Atlas starts its Docker agent environment and Python completion service
when ports `1984` and `3000` are not already reachable, waits for readiness,
and cleans up only the services that the run started. The adapter runs the
upstream completion script with `--no-filter` so all Hugging Face dataset rows
are attempted, and without Skippy-specific task limits or `tool_choice`
overrides. By default, the MCP-Atlas scorer uses the same local
OpenAI-compatible endpoint/model as the completion run; set `EVAL_LLM_MODEL`,
`EVAL_LLM_BASE_URL`, and `EVAL_LLM_API_KEY` to use a separate judge model. For
small local Skippy validation models, keep the completion endpoint in normal
compatibility mode and point the scorer override at a strict structured-output
endpoint, for example a second `skippy-serving serve-openai
--guardrails enforce` process. The adapter still uses the native scorer
and does not rewrite score data. For operator resumes, set
`MCP_ATLAS_COMPLETION_OUTPUT_NAME` to an existing upstream
`completion_results/*.csv` basename so the native completion script can reuse
its own resume behavior, and set `MCP_ATLAS_SCORE_CONCURRENCY` to the upstream
scorer's `--concurrency` value.
SWE-Bench Pro defaults to the official Docker image namespace (`jefzda`) with
local Docker deployment and local Docker evaluation so the core pack can run
without Modal credentials. The adapter first runs upstream
`helper_code/generate_sweagent_instances.py` for the full dataset, then feeds
SWE-agent a native `expert_file` instance file so local Docker can set the
official image platform, clear image entrypoints, and use SWE-agent's
standalone Python/SWE-Rex Docker runtime. The prepared SDK retains SWE-ReX 1.4.0's native Docker runtime. Override
`SWE_BENCH_PRO_DOCKERHUB_USERNAME`, `SWE_BENCH_PRO_DOCKER_PLATFORM`,
`SWE_BENCH_PRO_NUM_WORKERS`, `SWE_BENCH_PRO_EVAL_WORKERS`, or
`SWE_BENCH_PRO_PARSE_FUNCTION` as needed. Request workers must still match
`--endpoint-concurrency`. Set `SWE_BENCH_PRO_PARSE_FUNCTION=thought_action`
for local OpenAI-compatible models that do not emit OpenAI tool calls.
`SWE_BENCH_PRO_USE_LOCAL_DOCKER=0` preserves the upstream remote evaluation
choice. Deployment and index selection must match explicit preparation.

### Prepared MCP and SWE SDKs

MCP-Atlas and SWE-Bench Pro use pinned source checkouts and explicitly prepared
SDK environments. `sync` acquires source and MCP's pinned agent image; it
does not prepare these SDKs. Preparation is opt-in on Unix and takes existing
absolute `uv` and Python executable paths. It never downloads an interpreter.
Before preparing or running these evals, set `MESH_PYTHON_RESEARCH_SOURCE` to
the restored pinned research checkout. Use one explicit cache root for sync,
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

Every `eval run` writes `run.json` under the run directory with command status,
the resolved harness commit, raw artifact paths, wall-clock duration, and
normalized metrics where the harness exposes them. `speed-bench` records request counts, latency,
prompt/completion/total token counts, prompt and completion tok/s, and draft
acceptance rate when the server returns llama.cpp-compatible `timings`. Because
the upstream SPEED-Bench script does not expose an authorization argument,
SkippyBench launches it through a small adapter that adds the bearer token from
`--api-key` without modifying the upstream harness.
SWE-Bench Pro records OpenAI usage tokens and client-side tok/s when the
upstream flow produces them.
Terminal-Bench and SWE-Gym record Harbor trial counts, pass rates, and raw
Harbor job artifacts. The MCP-Atlas adapter records wall time, raw completion
CSV artifacts, the native scoring output directory, and CSV task row count.

`eval run` requires metrics-server for every external benchmark. `--metrics-http` defaults to
`http://127.0.0.1:18080`; the command creates a metrics-server run before the
harness starts and fails if that run cannot be created. Pass
`--metrics-run-id` to correlate the eval with the target Skippy/OpenAI endpoint
run id. SkippyBench finalizes and fetches
`/v1/runs/<run-id>/report.json`, stores it as `raw/metrics-report.json`, and
adds a `telemetry` block to `run.json`. When the target emits debug telemetry,
SkippyBench derives TTFT/FTTT from the first request span to the first
`stage.openai_decode_token` span, plus request and generation latency
aggregates. A finalization or report-fetch failure marks telemetry unavailable
without changing the native harness result in `report.success`. If the target
endpoint is not emitting the requested run id, or if debug token spans are
disabled, the telemetry block records that status rather than filling
misleading values.

Optional packs intentionally not wired yet:

| Future pack | Candidate |
|---|---|
| `repo-generation` | NL2RepoBench |
| `tool-expanded` | Toolathlon / Tool-Decathlon |

## Benchmark Corpora

Benchmark corpora are generated from Hugging Face datasets instead of checked
into the repository. The checked-in source manifest lives at
`corpora/bench_corpus_sources.json`; generated corpora and downloaded parquet
artifacts live under `target/`.

```bash
just bench-corpus smoke
just bench-corpus long
just bench-corpus coding-loop
just bench-corpus long-context
```

`just bench-corpus` builds the optional Rust `trajectory-reader` with the
`corpus-input` feature. The native Hugging Face client binds each configured
repository to its immutable commit before fetching selected Parquet artifacts.
CommitPackFT and APPS use their pinned repository JSONL files directly; dataset
loader scripts are never executed. Sampling uses the documented
`sha256-source-row-v1` ordering, not the former DuckDB ordering. The resulting
manifest records configuration, artifact and corpus SHA256 digests, quotas,
source revisions, and conversion provenance when applicable.

Generation refuses an existing tier directory before acquisition and publishes
a complete staged corpus/manifest directory in one rename. To regenerate,
choose a fresh output root, for example:

```sh
just bench-corpus smoke --out-root target/bench-corpora-rerun
```

The command may download dataset artifacts and must run within an authorized
data scope. It does not install Python or invoke a dataset script. Each HTTP
request has a 120-second timeout; this does not establish a hard whole-command
or native Xet worker deadline. Unsupported repository layouts require an
explicit `--artifact-manifest` binding the original dataset commit, config,
split, conversion provenance, and each local artifact's size and SHA256.
There is no latest-revision or unbound converted-Parquet fallback. Native
fixture tests qualify the selection policy; they do not qualify public dataset
acquisition or a measured benchmark run.

Generated layout:

```text
target/bench-corpora/smoke/corpus.jsonl
target/bench-corpora/smoke/manifest.json
target/bench-corpora/long/corpus.jsonl
target/bench-corpora/long/manifest.json
target/bench-corpora/long-context/corpus.jsonl
target/bench-corpora/long-context/manifest.json
target/hf-datasets/<dataset>/<resolved-revision>/...
```

Each corpus row uses a shared schema so all benchmark tools can consume it:

```json
{
  "id": "commitpackft-python:train:00000",
  "tier": "smoke",
  "family": "coding_edit",
  "source": "bigcode/commitpackft",
  "source_config": "python",
  "source_revision": "fc56fe33c030c6daa414c2b112c932b8eed085e6",
  "split": "train",
  "session_group": "commitpackft:repo-or-file",
  "prompt": "...",
  "expected_output": null,
  "metadata": {
    "routing_hint": "ngram",
    "adapter": "commitpack_edit"
  }
}
```

`smoke` is a small HF-sourced plumbing check. `long` uses the same sources and
schema with larger quotas for broad performance and routing comparisons. The
`coding-loop` tier is a warm-session speculative decoding corpus built from
native `SWE-bench/SWE-smith-trajectories` agent trajectories on Hugging Face. It
preserves adjacent turns from the same software-engineering session so n-gram
pooling can be measured on repeated coding edits instead of isolated prompts.
The `long-context` tier keeps a much larger prompt character budget and expands
sampled HF text into long stress packets. It is for 32k context capacity and
transport stress only; do not substitute it for the 8k customer-readiness
baseline or quality/speculation decisions.
The built-in manifest intentionally excludes generic chat, math, summarization,
SQL, and standalone function-calling sources such as OASST, Dolly, GSM8K, XSum,
Spider, and xLAM. Agent/coding claims should use the external eval harnesses
above rather than local prompt sampling.
The manifest records source datasets, resolved revisions, downloaded parquet
files, quotas, generated row counts, seed, generator path, and generator git
commit.

After generating a corpus, use `token-lengths` with the actual target GGUF to
produce the M1 token audit artifacts. The command applies the model chat
template before tokenization, matching the chat-completions product path:

```bash
skippy-bench token-lengths \
  --model-path /path/to/qwen3.6.gguf \
  --prompt-corpus target/bench-corpora/long/corpus.jsonl \
  --ctx-size 8192 \
  --generation-limit 512 \
  --enable-thinking false \
  --output-tsv target/bench-corpora/long/prompt-lengths.tsv \
  --summary-json target/bench-corpora/long/prompt-lengths-summary.json
```

For the 32k stress lane, run the same audit against
`target/bench-corpora/long-context/corpus.jsonl` with
`--ctx-size 32768`. The summary must show zero `exceeds_context` rows before
the corpus is promoted for that lane.

Speculative target/draft checks can use the generated corpus directly:

```bash
just bench-corpus smoke
target/debug/llama-spec-bench \
  --target-model-path /path/to/target.gguf \
  --draft-model-path /path/to/draft.gguf \
  --prompt-corpus target/bench-corpora/smoke/corpus.jsonl
```

`chat-corpus` drives `/v1/chat/completions` through an existing
chat-completions frontend such as `skippy-serving serve-openai`. Use it for
customer-facing benchmark numbers after the stage topology is already running:

```bash
skippy-bench chat-corpus \
  --base-url http://127.0.0.1:9337/v1 \
  --model org/repo:Q4_K_M \
  --metrics-http http://127.0.0.1:18080 \
  --metrics-run-id run-local-qwen \
  --prompt-corpus target/bench-corpora/long/corpus.jsonl \
  --max-tokens 512 \
  --concurrency-depth 1 \
  --stream \
  --include-usage true \
  --enable-thinking false \
  --output /Volumes/External/skippy-runtime-bench/qwen36-lab/run/chat-corpus.json
```

The runner preserves chat-style `messages` rows when present and otherwise
wraps `prompt` rows as one user message. If a row contains `session_group` or
`session_id`, that value is sent as the OpenAI `user` field so warm-session
benchmarks can exercise per-session KV or n-gram history. It records
per-request elapsed time, streaming TTFT when `--stream` is enabled, usage
tokens when the frontend returns them, API error codes, and aggregate
latency/token-rate summaries.
The command creates/finalizes a metrics-server run, writes the raw
metrics-server report beside `--output` by default, adds a telemetry summary to
the chat-corpus JSON report, and fails if the metrics-server report cannot be
created. It also sends stable `x-request-id` headers so matching server spans
can be grouped cleanly when the target endpoint exports the same run id.
Use `--concurrency-depth` for depth sweeps; the effective frontend generation
limit, such as `serve-openai --generation-concurrency`, must still be recorded
beside the result.

`run` is the promoted benchmark launcher. Pass the lab host list explicitly,
for example `--hosts 192.168.0.2,192.168.0.4,black.local`.
The host list must contain one unique host per planned stage; duplicate host
assignments are rejected so every staged run uses separate machines.
Local working files default to `/Volumes/External/skippy-runtime-bench`.
With `--execute-remote`, stage 0 is launched as a local child of the coordinator
while later stages are launched over SSH. This keeps the first-stage process on
the same routing and GPU path as the OpenAI frontend and avoids SSHing back into
the launcher host.

Distributed lab runs must also keep stage layer counts evenly balanced. The
launcher rejects splits where the largest and smallest stage differ by more than
one layer. For Qwen3.6's 40-layer package on three hosts, use
`--splits 14,27`; uneven splits are only for local investigation and should
not be reported as lab benchmark results.

Performance runs default to `--n-gpu-layers -1`, and lab commands should pass
that flag explicitly so each stage asks llama.cpp to offload all available
layers for its slice. CPU-only runs should be named and treated as diagnostic
baselines, not production performance numbers.

By default it creates a metrics-server run, writes a deployment plan and stage
configs, finalizes the run, and fetches `report.json` without starting remote
processes. Add `--execute-remote` to rsync configs/binaries and start
`skippy-serving serve-binary` over SSH. Add `--rsync-model-artifacts` to
copy model artifacts for each stage. For `layer-package`, the coordinator
materializes each stage GGUF locally under `--work-dir/model-cache`, reuses it
when the cached file is newer than the selected package parts, then shells out
to `rsync -az` to place the concrete stage GGUF under each host's stable
`model-cache` path. Remote configs load those files as `artifact-slice`, so
workers do not need temporary space for both the package parts and the composed
GGUF. Use `--remote-root-map host=/path` for
hosts with alternate scratch volumes, for example
`--remote-root-map build.local=/Users/jdumay/models/skippy-runtime-bench`.
When that remote root is the same filesystem visible on the coordinator, add
`--remote-shared-root-map host=/local/path` so the launcher can place the stage
GGUF locally and skip rsync for that file. Use `--endpoint-host-map host=addr`
to force binary stage endpoints onto the intended lab fabric, such as the
private `192.168.0.x` network, instead of mDNS-selected addresses.
For remote runs, pass a remote-reachable collector URL with
`--metrics-otlp-grpc-url`, for example `http://studio54.local:14317`.

Remote runs poll each stage until the PID is alive and the stage log shows the
binary listener; stage 0 is checked locally and remote stages are checked over
SSH. The launcher then connects to the first stage with the binary protocol so
readiness proves the downstream chain can handshake transitively. The measured
prompt driver sends prefill/decode frames to the first stage and writes
`driver-result.json` next to the deployment plan. Use `--prompt` when a
local full model or local layer package is available for llama-backed
tokenization, or `--prompt-token-ids` to provide explicit token IDs. Use
`--prompt-corpus corpus.jsonl` to run a JSONL corpus in one deployment; rows may
contain `prompt`, `turns`, or chat-style `messages`. `--prompt-limit` can
scope a corpus run while preserving the same launch path. Corpus driver output
includes aggregate elapsed, wire elapsed, prefill, TTFT, and decode P50/P95/P99
values in `driver-result.json`. Stage telemetry defaults to
`--stage-telemetry-level summary`, which emits one aggregate request summary per
stage connection. Use `--stage-telemetry-level debug` only when debugging
protocol timing; debug mode emits per-message timing spans for stage compute,
downstream forwarding/wait, upstream reply, and activation byte counts. Use
`--stage-telemetry-level off` for collector-isolation checks.
Use `--prefill-chunk-size` to split prompt prefill into multiple binary
prefill frames without changing decode behavior. Add
`--prefill-chunk-threshold` to keep shorter prompts as a single prefill frame
while still chunking longer prompts. Use `--stage-max-inflight` and
`--stage-reply-credit-limit` to sweep prefill ACK deferral/credit behavior on
the binary stage servers. Debug timing spans include the configured prefill
credit limit, pending deferred replies before/after the message, and credit
wait counts.
Use `--stage-async-prefill-forward` to pass `--async-prefill-forward` to each
binary stage server. This moves eligible non-final prefill activation writes to
a bounded background writer and should be treated as an opt-in transport
experiment until the current topology has been benchmarked.
Use `--prefill-chunk-schedule MIN:SIZE[,MIN:SIZE...]` for experimental
prompt-length schedules. The base `--prefill-chunk-size` applies unless the
prefill token count is at least `MIN`, in which case the largest matching
minimum selects the override chunk size. For example,
`--prefill-chunk-size 256 --prefill-chunk-schedule 513:512` uses 256-token
chunks up to 512 prefill tokens and 512-token chunks above that.
Use `--stage-telemetry-queue-capacity` to size each stage server's bounded
non-blocking telemetry queue for large debug corpus runs. Stage telemetry is
batched and retried from an in-memory replay buffer, but it remains best-effort:
if the queue or retry buffer is exhausted, stage execution continues and the
report surfaces the loss counters.

`focused-runtime` is a thin preset/wrapper around `run` for comparing the staged
runtime's cold-start, first-token, steady-decode, and KV-warm-reuse scenarios
with a compact JSON schema. Real performance runs require `--execute-remote` so
the prompt driver produces timing fields; the wrapper reuses the same deployment
plan, launcher, `report.json`, and `driver-result.json` produced by `run`, then
writes `focused-runtime-report.json` next to them unless `--focused-output` is
set. `--focused-output` also still prints the JSON to stdout.

The scenario presets only change safe driver inputs before delegating to `run`:
`cold-startup` and `first-token` default to one prompt, `steady-decode` defaults
to one prompt with a larger decode budget when `--max-new-tokens` is otherwise
left at the CLI default, and `kv-warm-reuse` defaults to two identical prompts so
the second request can exercise warm-prefix reuse where the model family and
runtime path support it. The report records startup readiness separately from
full run wall time, then mirrors the existing prompt-driver P50/P95 latency,
token-count, and throughput fields under compact top-level `topology`, `model`,
`latency_ms`, `throughput_tokens_per_second`, and `token_counts` objects.

For CI or command-shape validation without a GGUF or remote hosts, use the
schema smoke mode:

```bash
target/debug/skippy-bench focused-runtime \
  --schema-smoke \
  --scenario first-token \
  --hosts host-a,host-b \
  --splits 1 \
  --layer-end 2
```

The smoke output contains the same top-level fields as a real focused runtime
report: scenario, topology/model identity, stage hosts, prompt/decode token
counts, P50/P95 elapsed and TTFT values, decode latency, token throughput, and
paths to the underlying artifacts.

Logs are collected into the local run directory. Remote wrapper processes write
`stage.exit` files when they crash or are terminated, and
`remote-status.json` records the observed exit code. Remote PIDs are terminated
at the end unless `--keep-remote` is set. With `--keep-remote`, the launcher
keeps the local SSH wrapper processes alive after the benchmark command returns
so remote stage servers remain foreground children of their SSH sessions instead
of becoming orphaned background processes. This matters on macOS LAN labs where
orphaned processes can lose private-LAN routing privileges.
