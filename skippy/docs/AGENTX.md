# AgentX on Jianyang

AgentX replaces the Jianyang `agentic-replay` job. The new `agentx` entry has
no legacy alias or old command-line compatibility. Its source is
[the repository template](../evals/agentx/templates.yaml); the orchestration is Rust
in `tools/xtask/src/automation/agentx`, with the upstream Python harness in a
[locked external environment](../evals/agentx/pyproject.toml).

One job owns one model. It runs unpatched llama.cpp, then standalone Skippy,
sequentially on the same DGX Spark. Separate model jobs can occupy separate
Sparks. Both engines and AgentX execute inside the same ARM64 CUDA Docker image,
built and launched on the JY-leased Spark. No benchmark cells run on the host.
Each engine gets the identical verified GGUF shards; speculative decode
is disabled, and both chat templates receive `enable_thinking=false`.
The baseline source is the exact commit in
`skippy/llama_cpp/upstream.txt`, built independently with CUDA.

## Run

```sh
jy repos run mesh-llm agentx --ref pr:NUMBER --input model=qwen3.5-4b --input profile=smoke
jy repos run mesh-llm agentx --ref pr:NUMBER --input model=qwen3.5-9b --input profile=smoke
jy repos run mesh-llm agentx --ref pr:NUMBER --input model=qwen3.5-4b-q4-k-m --input profile=smoke
jy repos run mesh-llm agentx --ref pr:NUMBER --input model=qwen3.5-4b-q4-k-m --input profile=benchmark
```

Register the immutable custom result schema once before dispatch:

```sh
jy schemas register agentx 1 skippy/evals/agentx/result.schema.json
```

After merging, move the existing repository registration's `templates_repo` to
null and `templates_path` to `skippy/evals/agentx/templates.yaml`, then run
`jy repos sync mesh-llm`. Sync replaces the installed set: unrelated existing
jobs are included unchanged, and the old `agentic-replay` entry is absent.
Before merging, render this validated template with Jianyang's existing
`renderRepoJob` and the immutable PR SHA, then use `jy submit` on the rendered
specification. Template sync reads GitHub's default branch, so changing the
registration's default launch ref does not make a PR template available.
Do not redirect the production registration to a PR branch.

The current Jianyang lab schema supports an eligible node list, not a GPU-type
predicate. The template lists both registered Sparks (`spark-james`,
`spark-db08`), allowing either to claim a job. Add newly enrolled Sparks to that
list. Runtime admission also requires Linux aarch64 and an NVIDIA GB10.
The template omits `target.os`: Jianyang currently recognises a Linux host only
when its display name starts with `Linux`, while the Spark reports `Ubuntu`.
The explicit node list and runtime hardware checks remain mandatory.

The Spark needs Docker with NVIDIA Container Toolkit and the JY account in the
`docker` group. The template uses `sg docker` to acquire that membership even
when an existing user service still has its old supplementary groups. JY's lab
executor ignores `runtime.image`, so the template explicitly builds and launches
Docker through the repository's typed lifecycle controller.

`just agentx-container-build` builds the image tagged by `JIANYANG_COMMIT`.
The CUDA base is digest-pinned; Rust, Just, uv and the upstream harness are pinned.
`just agentx MODEL PROFILE OUTPUT` requires the JY job environment and runs that
image by immutable ID after checking its ARM64 architecture and source label.
The image contains both compiled engines, native runtimes, and the harness.
Build time is outside the measured cells.

The host's `HF_HOME`, campaign output and JY artifact directory are mounted.
The registered Spark template sets `HF_HOME=/mnt/lab/huggingface`, the existing
NFS-backed Hugging Face cache, and binds it at the identical path inside Docker.
Adjust that host setting when enrolling a Spark with a different cache location.
The container uses the job account's UID/GID and `--gpus all`; neither
the Docker socket nor host networking is passed into it. Both HTTP servers and
the harness communicate over container-local loopback. The controller attaches
stdout/stderr to JY, removes its named container on exit/cancellation, and records
`container-image.json` and `container-lifecycle.json`. A timeout inside the image
also enforces the remaining JY deadline if the host controller disappears.

The output directory must be new. `HF_HOME` must already exist on the host.
Weights, tokenizers and the dataset use the standard HF blobs/snapshots cache;
existing pinned files are reused and missing files download into that same host
cache. There is no separate weight copy or `--local-dir` download. Snapshot paths
are preserved so split GGUF shards resolve their siblings correctly.
Downloads precede measurement and
verify every weight shard against [models.json](../evals/agentx/models.json).
No projector is needed for these text-only traces. Tokenizers have independent
immutable Hugging Face revisions.

The explicit `qwen3.5-4b-q4-k-m` variant uses the existing NAS-cached
`Qwen3.5-4B-Q4_K_M.gguf` at the same pinned revision. Its result names include
the quantization, so these measurements remain distinct from the BF16 model.
For a cache-only campaign, confirm that exact file is present before submitting
the smoke, and submit the benchmark only after the smoke passes.

## Profiles and capacity

| Setting | Smoke | Benchmark |
|---|---|---|
| Parent traces | First 4, complete | All 393, complete |
| Offered agent sessions | 1 | 1, 2, 4, 8 |
| Measured time per engine / concurrency | 60 seconds | Upstream 3,600 seconds |
| Warmup requests per lane | 1 | Upstream 10 |
| Trajectory starting ratio | 0–0 | Upstream 0.25–0.75 |
| Random seed | 42 | 42 |
| Resident generation slots | 1 | 1 |
| Context pool / KV precision | 262,144 / F16 | 262,144 / F16 |
| Logical batch / microbatch tokens | 2,048 / 512 | 2,048 / 512 |

Offered agent concurrency is distinct from resident generation capacity.
Subagents may also issue concurrent requests. Both servers queue into one
resident slot, keeping the 16-bit dense and MoE models within Spark's memory
budget. This is a fixed-capacity comparison, not a maximum-throughput claim.
There is no context shortening, automatic weight quantization, or fallback on
allocation failure. Other resident capacities require a separate campaign.
Skippy's in-memory prefix retention target is explicitly 8 GiB, with disk
caching off; exact snapshots can exceed that target. llama-server retains its
single slot's prompt cache. These cache policies are recorded in launch arguments.

Large CUDA snapshot reads into pageable host memory use a reusable 64 MiB
pinned staging buffer per physical GPU. Capture remains synchronous so later
inference cannot change the exported state. Large staging-to-host copies use
up to eight CPU workers, joining before reusing the staging buffer.
Already pinned destinations and
small reads retain their direct path; unavailable pinned memory or
`GGML_CUDA_NO_PINNED` selects the original path. The native
`test-cuda-host-readback` and `test-cuda-host-readback-unpinned` CTest cases
check byte equality, chunk boundaries, unaligned buffers and concurrent reads.

Fragmented host snapshots of at least 64 MiB reconstruct with up to eight CPU
workers, bounded by available parallelism and block count. Contiguous snapshots
remain borrowed directly. Reconstruction preserves deduplication and cache
accounting, and falls back to a serial copy if worker creation fails.

Sixty seconds is only the smoke measurement window. Building, downloading,
tokenizer synthesis, server readiness, warmup, outstanding-request drain, and
cleanup add time. A single cell has a bounded 80-minute lifecycle for smoke;
the job has a 24-hour hard deadline shared with container build and preparation. These are
failure ceilings, not runtime estimates.

## Corpus and validity

The workload is SemiAnalysis's published 256k Weka corpus:
`semianalysisai/cc-traces-weka-062126-256k`, revision
`8fecd2fc56694469f758f0afbbb6335ad3043740`, SHA-256
`e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`.
The runner verifies the original JSONL, then writes each selected row as a
complete Weka JSON trace without rewriting its requests. `cohort.json` records
the exact selected IDs. Parent/child relationships remain upstream-owned.

AgentX 1.0.7's public loader does not expose a dataset revision option. We use
its stock local Weka loader and `--unsafe-override` with our independently
verified corpus. **Both profiles retain `submission_valid=false`.** These are
reproducible local comparisons, not InferenceX leaderboard submissions. Smoke
also intentionally violates the minimum 900-second profiling duration. No
Jianyang chart points or performance-history promotion are emitted. A future
upstream revision-pinning facility can remove the corpus override; it must
start a new comparison identity.

Jianyang stores the measurements as `custom.schema=agentx`, version `1`, with
separate named cells such as `agentx/qwen3.5-4b/llama/smoke/c1` and
`agentx/qwen3.5-4b/skippy/smoke/c1`. `points` stays empty so paired engines and
invalid submission profiles cannot accidentally enter a shared throughput series.

## Evidence and acceptance

Each cell retains server logs, upstream exports and raw records, launch
provenance, lifecycle status, and a summary containing TTFT, ITL, request
latency and output tokens/second. The full upstream export remains authoritative
for metric units and validity reasons. Engine identities are never collapsed
into one concurrency series. Build and input preparation are outside measurement.

A passing cell requires a zero-exit harness, completed measured requests,
positive output token usage, streaming TTFT/ITL observations, zero request or
context-overflow failures, and clean process-tree shutdown. Failures retain
partial artifacts and do not produce a success summary. `agentx.tar.gz` packages
the evidence for Jianyang's flat artifact uploader. Warmup and smoke results
must not be presented as steady-state performance.

The older GitHub trusted-main trajectory qualification workflow is a separate
consumer of the replay machinery; this change replaces the Jianyang job only.
