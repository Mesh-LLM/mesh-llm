# Jianyang templates

The template file is migrated from `general-inference/jianyang` so code and
job definitions can be reviewed together. Existing unrelated entries remain
unchanged. `agentx` replaces the old `agentic-replay` entry.

Validate without installing:

```sh
jy repos validate mesh-llm skippy/evals/agentx/templates.yaml --json
```

After merging, update the repository registration to read its own
`skippy/evals/agentx/templates.yaml` (`templates_repo: null`), then `jy repos sync mesh-llm`.
Do not sync a partial inventory: sync replaces all installed entries.

For a pre-merge benchmark while template registration remains unmerged, submit
standalone jobs with `jy submit` at an immutable Mesh PR SHA. Their comparison
reports remain in the JY job artifacts; the current jobs have neither
repository benchmark lines nor a `jy result` record. After the definitions are
registered and synced, launch with `jy repos run` to publish on the existing
Benchmarks pages. See [AgentX operations](../../docs/AGENTX.md).

## Attention kernel regression checks

Run on the corresponding GPU worker through Jianyang. The native harness
compares attention outputs against the CPU reference:

```sh
just skippy-attention-tests cuda -b CUDA0 -p kv_bounds_pattern
just skippy-attention-tests metal -b MTL0 -p kv_bounds_pattern
```

Omit `-p kv_bounds_pattern` for the full attention matrix. Report unsupported
cases separately; a backend filter that executes zero cases is not a pass.
The targeted cases cover concurrent masks, shared-prefix holes, alignment,
empty rows with attention sinks, finite biases, and F16/Q8 KV storage.

After building, locate the harness with `just skippy-attention-tests-dir <backend>`.
Its directory includes the patched native revision, backend and configured GPU
architecture so changed native revisions cannot reuse stale test objects.
Run kernel performance cases with
`"$(just --quiet skippy-attention-tests-dir <backend>)/bin/test-backend-ops" perf
-o FLASH_ATTN_EXT -b <device> -p kv_bounds_pattern`. Compare baseline and
candidate in ABBAAB order, including fully visible masks and short contexts.
Kernel checks supplement real-model serving tests with fresh requests and
resident-prefix restores; they do not replace AgentX TTFT, decode, completion,
failure/cancellation, and resident-memory measurements.

## Performance acceptance

Skippy must match or beat llama on both CUDA and Metal before this performance
work is considered complete. Qualify each backend independently with the same
pinned Q4 weights, AgentX traces, sampling settings,
and offered concurrency. The fixed-work comparison matches llama's per-slot
context to the common 128K trace limit while Skippy plans its own capacity and
parallelism. Smoke and full profiles retain llama's automatic context. Record
realized capacities rather than assuming equal allocation. Include both
single-session and concurrent serving;
a single-session win does not qualify concurrent serving.

Run llama (A) and Skippy (B) in ABBAAB order on the same otherwise idle worker.
Report every arm and repeated-run variation for TTFT, decode throughput,
completed requests, errors, cancellations, and memory. Treat an inconclusive
comparison as unqualified, and investigate regressions per model and
concurrency rather than hiding them in an aggregate. Compare completed work
and inspect grace-period cancellations before declaring a winner.

## Fixed-work comparison

Use `--profile comparison` for the small repeated comparison report. Its workload
version is `agentx-1.0.12.post2-local-comparison-3`, separate from smoke 9 and full 8.
For each model, run whole c=1/2/4 sweeps in **llama, Skippy, Skippy, llama, llama,
Skippy** order on the same host. The four Q4_K_M model categories are dense
(Qwen3-4B-Instruct-2507), hybrid dense (Qwen3.5-4B), attention MoE
(Qwen3-Coder-30B-A3B-Instruct), and hybrid MoE (Qwen3.5-35B-A3B).

The fixed cohort uses the pinned corpus in `inputs.rs`. Selection is independent
of engine results: among whole traces within the common 128K budget and with
the last recorded request start at most 600 seconds, take the two smallest
output totals in the <=64K peak-input band, and one each in the 64–96K and
96–128K bands. Ties use trace ID. Original rows, output lengths, child topology
and recorded think times remain intact.

For this profile only, llama starts with `--ctx-size = 131072 * --parallel`,
giving each non-unified slot the same 128K ceiling as the selected workload.
An explicit llama unified-KV control also sets `--kv-unified-per-slot 131072`:
the shared pool keeps the same total size while the advertised slot limit
matches the cohort.
Skippy keeps its automatic planner and 128K cap. This avoids charging llama
for an unused 262K native slot on Qwen3 models; the server's observed slot
capacity and each engine's memory footprint remain reportable evidence.
The runner rejects a llama comparison cell unless its server log confirms
the expected slot count and 131,072-token context per slot; rejected cells
publish no throughput point.
AgentX retains a per-member process receipt in `lifecycle.json`; a completed
trace still fails qualification if its server requires forced shutdown.

| Trace ID | Source requests | Requested output tokens | Peak input tokens |
|---|---:|---:|---:|
| `12cfecf98c158ab54fb65980602421f00d69` | 20 | 6,409 | 110,464 |
| `1493faffdc8942e99daa22277f44871a40b8` | 7 | 4,097 | 46,528 |
| `9ec047754f4ddd94a59fcd0d85213d5f3b1b` | 21 | 4,258 | 89,792 |
| `bbdcb12440a7ab3496b9fac8b5f9824b1672` | 21 | 3,949 | 54,784 |
| **Total** | **69** | **18,713** | |

Each cell starts a fresh server and sends each root trace once, starting at turn
zero. An explicit local config selects agentic replay with four root sessions
and no duration, request cutoff or accelerated warmup. This avoids the named
AgentX scenario restoring its duration and warmup defaults. Initial lanes start
together; later think times remain intact. Children finish even after the last
root is admitted. This is a cold-start, small-cohort comparison, not a leaderboard
submission or a general steady-state capacity claim.

One explicit record processor participates in the harness startup barrier.
Auto-scaling can otherwise register it after the one-shot dataset notification,
leaving completed requests without processed metrics even with mmap caching
disabled. This setting changes metric-service startup, not serving concurrency.

`expected-work.json` records every source trace/outer/inner turn and its requested
output. A cell only publishes throughput after matching every actual source turn
and server-reported output count exactly. Missing, repeated, cancelled, failed,
overflowed or shortened turns fail the cell. `completed-work.json` records the
matched map and SHA-256; compare this hash and the selected cohort hash across
all arms. Safety deadlines fail unfinished work without accepting partial
throughput. Keep raw exports and failed cells in the comparison report.

Report throughput, decode, TTFT, completed work and sampled process RSS/physical
footprint separately for each category/model/backend/concurrency, with all three
arms per engine and their variation. On CUDA, report the sampled peak
`nvidia-smi` process-accounted device memory where available; it is not a
physical-residency counter on unified-memory Spark. Missing values remain null.
Standalone JY artifact jobs can run this profile without merging or syncing
template configuration.

For a cache-path diagnostic, set `AGENTX_SKIPPY_DEBUG_TELEMETRY=1` on the
AgentX runner. It enables Skippy's debug level and stderr telemetry only in its
server child. The cell's `cache-decisions.json` projects at most 512 cache
record/lookup decisions with fixed outcome labels, timestamps and numeric
positions. It excludes raw prompts, cache keys and arbitrary telemetry fields;
`omitted_events` reports if the bound was reached. The ordinary
`server.stderr.log` remains sanitized and size-limited, so it is not a complete
cache-decision trace.
With `SKIPPY_EXACT_STATE_DIAGNOSTICS=1`, `exact_state_events` counts every
recognized resident-checkpoint drop, demotion, publication and eviction.
`exact_state_samples` retains
at most 512 of those events with sequence numbers and allowlisted numeric
positions, byte sizes and limits; `omitted_exact_state_samples` counts any
additional events. This diagnostic is separate from the 512 record/lookup
decision limit.
Leave it unset for paired performance results because debug event emission can
perturb latency.
Use `--only-concurrency 2` with `automation agentx run` to isolate a failing
cell; the default still runs the full 1/2/4 sweep. A single-cell diagnostic is
not a complete comparison report.

For a c=4 Skippy serving sensitivity run, set
`AGENTX_SKIPPY_MICROBATCH_TOKENS=1024` or
`AGENTX_SKIPPY_GENERATION_CONCURRENCY=2` on the native `automation agentx run`
process and pass `--only-concurrency 4`. The microbatch override must be a
positive integer no larger than the 2,048-token batch, and the generation
override must be positive; unset variables keep the normal serving plan.
Local cache-aware prefill chunks follow the native microbatch size rather than
the embedded-stage `--prefill-chunk-size` setting. The exact
server arguments are recorded in the cell's `launch.json`. Compare each variant
with a default Skippy c=4 cell from the same source and machine, with identical
traces and model weights. These diagnostic variants do not replace the default
llama-versus-Skippy ABBAAB qualification.

To isolate hybrid-model KV layout, set `AGENTX_SKIPPY_KV_UNIFIED=false` on the
native AgentX runner. The runner passes `--kv-cache-unified=false` only to the
Skippy server and records the explicit layout in `launch.json`; unset keeps the
normal automatic layout. This is a diagnostic control only. Qualified
production comparisons require observed unified KV; inspect the native
runtime receipt rather than inferring the layout from an unset override.
With `AGENTX_SKIPPY_DEBUG_TELEMETRY=1`, the prefill summary reports
`skippy.scheduler.prefill_admission_wait_ms` and
`skippy.scheduler.prefill_admission_yields`. These count waits and FIFO turns
between native chunks; the non-debug paired run measures throughput without
debug instrumentation.

## Contention diagnostic, smoke, and full profiles

The `contention` profile is a separate **diagnostic** sweep at c=4/8/16. It
selects 16 context-compatible source traces with the lowest total requested
output, breaking ties by trace ID. Each must begin with two ordinary turns no
more than 60 seconds apart. The diagnostic retains those two recorded turns,
caps each input at 8K tokens and each requested output at 128 tokens, and runs
exactly 16 sessions per cell without dataset wrapping. The cohort manifest records
both source and transformed trace digests and output totals. Every cell starts
its initial lanes together and preserves the first follow-up's recorded delay.
An explicit AgentX config runs the 16 sessions once without the scenario's
ten-request-per-lane warmup or a duration cutoff. Weka prompt reconstruction
uses one worker so it finishes before the serving measurement without competing
with the server. Both engines receive the same 128K per-request ceiling; llama
reserves 128K per explicit slot while Skippy retains its automatic four-lane plan
unless an explicitly labelled
`AGENTX_SKIPPY_GENERATION_CONCURRENCY` variant is run. Compare actual completed
requests, output tokens, active HTTP overlap, TTFT, inter-token latency,
errors and memory at each level. These scaled two-turn sessions diagnose
saturation and queueing; they do not qualify a full-trace llama-versus-Skippy
win. Record the serving lane count and any override with each result. Skippy's debug telemetry
is enabled for this profile; `scheduler-profile.json` retains only aggregate
batch size, token count, queue wait, and runtime-lock wait/hold measurements.
The debug run is for bottleneck diagnosis, not a production-speed claim.

Smoke version 8 uses a 60-second dispatch window and drains outstanding requests.
Upstream `--burst-phase-starts` makes every lane's first profiling request
eligible immediately; subsequent recorded inter-turn delays remain intact.
Without it, the pinned c4 cohort spreads initial requests over 81.4 seconds,
leaving its fourth lane outside the smoke window. Full benchmark version 7
keeps upstream phase-start spacing; do not combine these workload versions.
A turn
near that boundary can be included in one repetition and excluded from another,
so the same trace seed does not guarantee identical completed work. Check trace
IDs and turn indices before comparing short-run throughput. Offered sessions
also need not equal observed HTTP overlap: replay delays can leave lanes idle.
The runner reads AgentX 1.0.12.post1's `metadata.benchmark_phase` and HTTP intervals to
report `observed_concurrency.peak_in_flight_requests`; this is not a count of
simultaneous GPU decode streams.

Kernel timings establish the effect of a kernel change, not an end-to-end win
against llama. Correctness and resident-prefix restore checks remain required.
Label host RSS separately from GPU residency; missing GPU counters must remain
unavailable rather than being replaced with host RSS.

The runner records measured phase completions, deadline cancellations, and
errors in each cell's `phase_outcome` and in the normal JY result notes. AgentX
1.0.7's aggregate `was_cancelled=false` does not exclude requests cancelled
during drain; the pinned harness's phase-complete log supplies those counts.
Missing or inconsistent accounting fails result collection.

Skippy retains its automatic 128K per-request context cap. Longer traces are
excluded from both engines before measurement and listed as `unsupported_context`
in the corpus manifest. Unexpected runtime overflows are still classified as
`unsupported_context` only when explicit upstream overflow counts and phase
accounting explain all rejected requests. Such cells
retain their logs, overflow counts and partial-attempt memory evidence, but emit
no serving points or paired throughput comparison. The sweep remains incomplete
and returns a nonzero exit status. Mixed request errors, cancellations, crashes
and missing evidence remain failures; a context rejection must not hide them.

Dataset mmap caching is disabled for both engines to avoid a 1.0.7 startup
race that can leave record processors waiting for dataset configuration after
a fast cache hit. This rebuilds trace inputs before measurement and preserves
the host HF model cache. It is a workaround pending an upstream lifecycle fix.

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

The harness is pinned to the [MeshLLM AgentX 1.0.12.post2 fork](https://github.com/Mesh-LLM/agentx-harness/releases/tag/agentx-v1.0.12.post2),
based on upstream 1.0.12 plus drain fixes and stable cache-bust namespaces at commit
`6633517b9d3d83441dc110613c80e70458f3d75d`. The wheel uses the upstream release
builder, embeds its source commit and is SHA-256 verified by `uv.lock`.
Upstream [PR #50](https://github.com/SemiAnalysisAI/agentx-harness/pull/50)
retires timers already pending at cutoff. The fork also retires delayed children
and join timers created by responses that arrive after cleanup. In-flight
requests still finish normally; immediate DAG children retain existing issuer
admission checks, including root-session quota semantics.

The comparison namespace includes the model key and SHA-256 of the materialized
cohort manifest. It remains the same across engines, concurrencies and ABBAAB
repetitions for unchanged work, while distinct trajectory trees and recycled
plays still get distinct markers. Each server starts with a fresh cache. This
removes run-ID variation from the marker; equal source work alone still does
not prove that the engines rendered identical token IDs.

Smoke 9 and full 8 use the same fork for both engines. Recorded timing and
requested outputs remain intact. Never-issued future requests are reported as
truncated children, not failed HTTP requests. Compare completed raw request/turn
sets as well as the selected cohort digest. Earlier harness versions stay separate.
