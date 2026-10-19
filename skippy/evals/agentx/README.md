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

For a pre-merge benchmark, update the four AgentX blocks in the currently
registered Jianyang template source through a configuration-only PR, sync, and
launch with `jy repos run` at the immutable Mesh PR SHA. A locally rendered
`jy submit` specification lacks the repository association required by the
benchmark pages. See [AgentX operations](../../docs/AGENTX.md).

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
and offered concurrency. Default-serving comparisons let Skippy plan context and parallelism; llama
plans context with explicit slots matching offered concurrency; record the realized capacities instead of assuming they
match. Include both single-session and concurrent serving;
a single-session win does not qualify concurrent serving.

Run llama (A) and Skippy (B) in ABBAAB order on the same otherwise idle worker.
Report every arm and repeated-run variation for TTFT, decode throughput,
completed requests, errors, cancellations, and memory. Treat an inconclusive
comparison as unqualified, and investigate regressions per model and
concurrency rather than hiding them in an aggregate. Compare completed work
and inspect grace-period cancellations before declaring a winner.

Smoke version 7 uses a 60-second dispatch window and drains outstanding requests.
Upstream `--burst-phase-starts` makes every lane's first profiling request
eligible immediately; subsequent recorded inter-turn delays remain intact.
Without it, the pinned c4 cohort spreads initial requests over 81.4 seconds,
leaving its fourth lane outside the smoke window. Full benchmark version 6
keeps upstream phase-start spacing; do not combine these workload versions.
A turn
near that boundary can be included in one repetition and excluded from another,
so the same trace seed does not guarantee identical completed work. Check trace
IDs and turn indices before comparing short-run throughput. Offered sessions
also need not equal observed HTTP overlap: replay delays can leave lanes idle.
The runner reads AgentX 1.0.12's `metadata.benchmark_phase` and HTTP intervals to
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

AgentX smoke version 7 and full benchmark version 6 apply the same upstream
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

The harness is pinned to AgentX 1.0.12. Its upstream
[cutoff cleanup fix](https://github.com/SemiAnalysisAI/agentx-harness/pull/50)
releases delayed subagents and parked continuations that never dispatch before
sending stops. Version 1.0.7 could leave their branch accounting pending after
all HTTP requests finished, making an infinite drain hang. Smoke 7 and full 6
use the corrected harness for both engines; they preserve recorded timing and
requested outputs. This change does not cap individual trace delays or classify
never-issued future requests as failed HTTP requests. Compare completed raw
request/turn sets as well as the selected cohort digest.
