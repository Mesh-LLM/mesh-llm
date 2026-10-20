# AgentX on Jianyang

AgentX replaces the Jianyang `agentic-replay` job. The new `agentx-{engine}-{profile}` entries have
no legacy alias or old command-line compatibility. Its source is
[the repository template](../evals/agentx/templates.yaml); the orchestration is Rust
in `tools/xtask/src/automation/agentx`, with the upstream Python harness in a
[locked external environment](../evals/agentx/pyproject.toml).

For the small repeated comparison report, use the standalone `comparison`
profile: four complete deterministic traces, 69 requests and 18,713 output
tokens per c=1/2/4 cell, with ABBAAB engine order on one host. It has no
dispatch cutoff or accelerated warmup and checks every source turn/output
before accepting throughput. See the [fixed-work cohort and result contract](../evals/agentx/README.md#fixed-work-comparison).
This profile can use standalone JY artifact jobs without changing installed
repository templates; keep its workload version separate from smoke/full.

One job owns one model and one engine. Separate llama.cpp and Skippy templates
record standard JY serving points without mixing engines into one concurrency
series. Run each comparison sequentially on the same DGX Spark; separate model
pairs can occupy separate Sparks. Smoke and full benchmark templates are separate. Both engines and AgentX execute inside the same ARM64 CUDA Docker image,
built and launched on the JY-leased Spark. No benchmark cells run on the host.
Each engine gets the identical verified GGUF shards; speculative decode
is disabled, and both chat templates receive `enable_thinking=false`.
The baseline source is the exact commit in
`skippy/llama_cpp/upstream.txt`, built independently with CUDA.

## Run

```sh
jy repos run mesh-llm agentx-llama-smoke --ref pr:2463 --input 'model=hf:unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523#Qwen3.5-4B-Q4_K_M.gguf'
jy repos run mesh-llm agentx-skippy-smoke --ref pr:2463 --input 'model=hf:unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523#Qwen3.5-4B-Q4_K_M.gguf'
# After smoke acceptance, use agentx-llama-benchmark and agentx-skippy-benchmark.
```

Results appear under [JY Benchmarks](https://pilot.jianyang.cloud/benchmarks),
with an entry for each engine and profile. Select the model within the entry.
PR runs appear as provisional results in the existing run table; default-branch
runs populate its trend charts. Throughput, TTFT, inter-token latency and request
latency use JY's standard result fields. Detailed exports and memory samples
remain attached as `agentx.tar.gz`. No custom schema registration is needed.

Before the Mesh PR merges, install these four template blocks in the current
registered template source (`general-inference/jianyang`,
`pilot/templates/mesh-llm.yaml`) through a configuration-only PR, then
`jy repos sync mesh-llm`. Launch with `jy repos run`, which records the actual
repository/template association. Ad hoc `jy submit` jobs do not populate these
benchmark pages. Do not change the production default ref to a PR branch.
After merging, move `templates_repo` to null and `templates_path` to
`skippy/evals/agentx/templates.yaml`, then sync. That cutover replaces the old
`agentic-replay` entry while preserving unrelated templates.

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
`just agentx ENGINE MODEL PROFILE OUTPUT` requires the JY job environment and runs that
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
only existing pinned files are admitted; missing files fail offline. There is no
separate weight copy or `--local-dir` download. Snapshot paths
are preserved so split GGUF shards resolve their siblings correctly.
Cache admission precedes measurement and
verifies every weight shard against [models.json](../evals/agentx/models.json).
No projector is needed for these text-only traces. Tokenizers have independent
immutable Hugging Face revisions.

## Metal qualification

On an Apple Silicon Mac, submit a native JY job at the immutable PR revision and
set `HF_HOME` to that Mac's existing cache. Inside its materialized checkout:

```sh
just agentx-build metal
just automation-bootstrap
target/debug/xtask automation agentx run --engine skippy --model qwen3.5-4b-q4-k-m --profile smoke --output "$JIANYANG_ARTIFACT_DIR/campaign"
```

Use `--engine llama` in the corresponding baseline job. The same runner owns
trace selection, c1/2/4 sweeps, child cleanup, exports, and offline model admission
on both platforms. The Metal build uses the pinned upstream llama.cpp baseline
and the source-built Skippy Metal runtime. Run comparisons in ABBAAB order on
the same leased Mac. The existing Spark templates retain their GPU Docker path;
native Metal qualification uses standalone jobs and artifact results.

`hardware.json` identifies the admitted backend, and the raw hardware report is
retained alongside it. macOS samples process RSS and physical footprint through
`proc_pid_rusage`; summaries expose `sampled_peak_process_rss_bytes` and
`sampled_peak_process_physical_footprint_bytes`. The latter is the process's
memory-accounting footprint, not GPU residency. macOS does not provide an RSS
high-water mark through this API, so that field remains unavailable. Linux keeps
its `/proc` RSS/high-water and cgroup observations. Sampling can miss short peaks
on either platform; none of these process fields substitutes for GPU residency.

## Q4 model matrix

All PR comparisons use Q4_K_M weights. Model keys and result names include the
quantization; previous F16/BF16 measurements are historical evidence, not members
of this matrix. Each file is a single GGUF at the immutable revision and SHA-256
in [models.json](../evals/agentx/models.json). Sizes below are decimal GB of
weights, not peak runtime memory.

| Category | Model key | GGUF file | GB |
|---|---|---|---:|
| Dense attention | `qwen3-4b-q4-k-m` | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2.50 |
| Hybrid dense | `qwen3.5-4b-q4-k-m` | `Qwen3.5-4B-Q4_K_M.gguf` | 2.74 |
| Attention MoE | `qwen3-coder-30b-a3b-q4-k-m` | `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` | 18.56 |
| Hybrid MoE | `qwen3.5-35b-a3b-q4-k-m` | `Qwen3.5-35B-A3B-Q4_K_M.gguf` | 22.02 |

Hybrid means Gated DeltaNet layers interleaved with full attention; dense versus
MoE describes the feed-forward layers. The pinned upstream configs identify
[Qwen3 4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json)
as dense, [Qwen3 Coder](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/b2cff646eb4bb1d68355c01b18ae02e7cf42d120/config.json)
as attention MoE, and the Qwen3.5
[4B](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)
and [35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json)
as hybrid dense and hybrid MoE respectively. These cover four architecture
categories; comparisons are between engines on the same model, not a controlled
architecture comparison across differently sized/trained models.

For a cache-only campaign, confirm the exact pinned file is present before
submitting the smoke, and submit the benchmark only after the smoke passes.

## Profiles and capacity

| Setting | Smoke | Benchmark |
|---|---|---|
| Parent traces | First max(4, concurrency) eligible traces from 128 candidates (36 eligible) | All 83 eligible traces from 393 candidates |
| Offered agent sessions | 1, 2, 4 | 1, 2, 4 |
| Measured time per engine / concurrency | 60 seconds | Upstream 3,600 seconds |
| Warmup requests per lane | 1 | Upstream 10 |
| Trajectory starting ratio | 0–0 | Upstream 0.25–0.75 |
| Random seed | 42 | 42 |
| Resident generation slots | Skippy auto; llama matches offered load | Skippy auto; llama matches offered load |
| Context per slot / KV precision | Engine default / F16 | Engine default / F16 |
| Logical batch / microbatch tokens | 2,048 / 512 | 2,048 / 512 |

Both profiles sweep offered request concurrency independently of serving capacity.
Neither engine receives a context override. Skippy plans context and parallelism;
llama keeps explicit slots matching offered concurrency to preserve the baseline
KV policy while automatically sizing context. These are
default-serving comparisons, not equal-capacity kernel comparisons.
Native context allocation observations are retained in each summary, with raw
server logs preserved; missing observations remain unavailable. Do not infer
execution slots from offered concurrency or native sequence reservations.
Skippy retains its automatic 131,072-token per-request cap even when the model
supports a longer native context. Both engines filter whole traces to the same
recorded input-plus-output budget before measurement. Exclusions are retained
in the corpus manifest. Unexpected runtime context-only rejections are reported as
`unsupported_context`, with overflow counts and diagnostic paths retained.
These cells contribute no serving point or paired throughput comparison;
their memory is labelled as a partial attempt. Mixed failures and inconsistent
accounting remain failures. Traces are never truncated to fit a plan.
Unsupported and failed cells retain evidence, and subsequent levels
continue only after confirmed cleanup. Any failed or incomplete cell fails the
job after the sweep; valid earlier points and explicit failure counts are still
copied from the container into the JY result. A green job therefore requires
every requested cell to complete. Harness versions 4 and later separate these
measurements from the previous fixed-context sweep.
Skippy's in-memory prefix retention target is explicitly 8 GiB, with disk
caching off; exact snapshots can exceed that target. llama-server retains its
per-slot prompt caches. These cache policies are recorded in launch arguments.

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

Both engines have a 900-second readiness deadline. Skippy receives that same
startup timeout explicitly, so its CLI's 60-second default cannot terminate a
large NAS-backed model load while the harness is still waiting for readiness.

Sixty seconds is only the smoke measurement window. Building, cache verification,
tokenizer synthesis, server readiness, warmup, outstanding-request drain, and
cleanup add time. A single cell has a bounded 80-minute lifecycle for smoke;
the job has a 24-hour hard deadline shared with container build and preparation. These are
failure ceilings, not runtime estimates.

Smoke version 8 uses upstream `--burst-phase-starts` so each lane's first
profiling request is eligible immediately. The previous c4 smoke's recorded
first-request spacing was 81.4 seconds, exceeding its 60-second sending window
and admitting only three profiling requests. The burst changes phase-start
spacing only; subsequent inter-turn delays and trace contents remain intact.
Full benchmark version 7 retains upstream phase-start spacing. Compare versions
separately and verify observed HTTP overlap before claiming concurrency coverage.

## Corpus and validity

The workload is SemiAnalysis's published 256k Weka corpus:
`semianalysisai/cc-traces-weka-062126-256k`, revision
`8fecd2fc56694469f758f0afbbb6335ad3043740`, SHA-256
`e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`.
The runner verifies the original JSONL, then writes each selected row as a
complete Weka JSON trace without rewriting its requests. The root `cohort.json`
records the candidate IDs and context eligibility; each cell's `cohort.json`
records the filtered selection and its digest. Parent/child relationships remain
upstream-owned.

AgentX 1.0.12.post1's public loader does not expose a dataset revision option. We use
its stock local Weka loader and `--unsafe-override` with our independently
verified corpus. **Both profiles retain `submission_valid=false`.** These are
reproducible local comparisons, not InferenceX leaderboard submissions. Smoke
also intentionally violates the minimum 900-second profiling duration. No
claim is made that a smoke is steady-state performance. Local-corpus measurements
are still valid inputs to our own JY benchmark views: upstream leaderboard
eligibility does not suppress them. A future upstream revision-pinning facility
must start a new harness identity.

Jianyang receives standard `points` for the selected engine, with integer token
totals and latency distributions in milliseconds. Smoke and full profiles have
different template keys and harness fingerprints. Engine templates are separate
because JY's existing chart series distinguish concurrency, not engine identity.
Raw cell names remain `agentx/MODEL/ENGINE/PROFILE/cN`.

AgentX smoke version 8 and full benchmark version 7 apply the same upstream
`--max-context-length 131072` filter to both engines before selecting traces.
AgentX excludes a whole trace if any parent or subagent turn's recorded input
plus requested output exceeds the limit (zero recorded output reserves one token).
No retained requests, outputs, or subagent topology are truncated. The root
`cohort.json` records every source trace's peak, digest, and eligibility; each
cell records the selected cohort digest and excluded count. Preflight exclusions
are workload scope, not failed requests. This is a common
recorded-token budget, not a server-context override or a rendered chat-template
token guarantee. Unexpected wire overflows still invalidate a cell. Compare
only matching cohort digests and completed raw trace/turn sets. Older workload
versions remain separate. Skippy keeps its automatic planner and 128K cap.

## Evidence and acceptance

Each cell retains server logs, upstream exports and raw records, launch
provenance, lifecycle status, and a summary containing TTFT, ITL, request
latency and output tokens/second. The full upstream export remains authoritative
for metric units and validity reasons. Engine identities are never collapsed
into one concurrency series. Build and input preparation are outside measurement.

Each cell also retains `memory.jsonl` with one-second samples and
`memory-summary.json` with sampled process RSS and container-memory peaks,
the process RSS high-water mark, and minimum host available memory. Sampling
covers the server lifecycle, including loading and warmup. Container memory
includes the harness and file cache; missing counters remain null. These OS
measurements are not GPU residency. Hardware evidence includes NVIDIA memory
counters where supported; GB10 currently reports them as unavailable because
Spark uses unified memory. No native GPU allocation or KV allocation totals
are inferred from RSS.

A passing cell requires a zero-exit harness, completed measured requests,
positive output token usage, streaming TTFT/ITL observations, zero request or
context-overflow failures, and clean process-tree shutdown. Failures retain
partial artifacts and do not produce a success summary. `agentx.tar.gz` packages
the evidence for Jianyang's flat artifact uploader. Warmup and smoke results
must not be presented as steady-state performance.

The older GitHub trusted-main trajectory qualification workflow is a separate
consumer of the replay machinery; this change replaces the Jianyang job only.

## Repeated sweep campaigns

For each model, submit six independent engine sweeps in ABBAAB order: llama,
Skippy, Skippy, llama, llama, Skippy. Pin all six to the same Spark and source.
Four models produce 24 jobs and 72 attempted cells. Ordering applies to whole
sweeps, not individual concurrency levels. Smoke is one sweep per engine.

Each cell writes results and phase accounting before the next starts. Cancelled
requests exclude that cell from serving points; all-failed sweeps retain outcome
counters without invented zero-throughput measurements. JY ingests results at
job completion; intermediate files and logs are evidence, not live chart updates.
Request overlap is not proof of concurrent GPU decode. Full record and HTTP
timing exports are retained for overlap analysis. GPU residency remains
unavailable where the driver does not supply it; host RSS is separate.

Smoke 8 and full 7 pin the [MeshLLM AgentX 1.0.12.post1 fork](https://github.com/Mesh-LLM/agentx-harness/releases/tag/agentx-v1.0.12.post1)
at source commit `8a37952a686d9fe49a2c7376b5b965e8f6480cc9`, with the wheel URL
and SHA-256 locked in the environment. It contains upstream 1.0.12, including
[PR #50](https://github.com/SemiAnalysisAI/agentx-harness/pull/50), plus cleanup
for delayed children and joins declared by responses arriving after the sending
cutoff. Existing HTTP requests drain normally; immediate children still use
the existing issuer admission rules. Both engines consume the same artifact,
without shortening outputs or clamping recorded inter-turn delays. Older
harness versions remain separate.
