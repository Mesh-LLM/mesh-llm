# Complete-session agentic replay

The workload is real ThoughtWorks sessions, replayed from the first recorded
assistant turn through the last. New generations are measured but never replace
recorded assistant/tool history, and tools are not executed. Checkpoint and
final-only modes are rejected. Dataset tool schemas are reconstructed from the
recorded names; they are not the original harness's complete schemas.

The Metal-only matrix pins Granite 3.1 2B dense, Granite 3.1 3B-A800M MoE,
and Qwen3.5 4B hybrid recurrent at Q4_K_M. Native contexts are at least 128K.
CUDA is deferred. Model changes require new qualification, not just new hashes.

## Workload and selection

`sessions_per_concurrency` is a **total**, not a per-framework count. The initial
16 sessions support two session waves at concurrency 8; levels are 1, 2, 4, 8,
with two passes reversing level order. This is 64 distinct measured sessions,
384 session executions across three models and two passes, and 24 cells.
Warmup sessions are disjoint and excluded from measured summaries.

Selection is `balanced-md5-v2`: the pinned dataset is deduplicated by session ID,
ordered by framework and MD5(session ID), then allocated by quotient/remainder
in declared framework order. Each cohort has 6/5/5 sessions. Cohorts are disjoint;
all models use the same deterministic cohorts. The recorded ISL window is
32,768 inclusive to 131,072 exclusive, with at least five assistant turns.
These dataset counts are selection hints, not runtime token counts. Every
measured session must actually reach 32,768 formatted prompt tokens. Publish
all per-turn lengths, session IDs and expected turn IDs; never resample a failed
cohort to hide a context or memory failure. The window targets long real traces;
it does not assert that the dataset supplies 128K-long sessions.

## Context qualification and measurement

Before each ref's measured passes, launch a disposable server and obtain the
actual model digest and effective stage context from `/api/runtime`. Inspect
that GGUF's architecture/context metadata. Require at least 131,072 tokens in
both the GGUF and runtime, without injecting context-size or RoPE overrides.

The text-only tokenizer endpoint cannot apply mesh's chat/tool formatting.
Therefore qualification sends each recorded prompt through the actual chat
path with a **one-token output budget**. It records the serving tokenizer's
prompt count, checks it plus the real recorded-answer output budget against
the effective window, and fails if any session is incomplete, oversized or too
short. This is potentially expensive prefill work, recorded separately in
`context-preflight/`; it is excluded from throughput measurements. Discard the
entire qualification server/cache before starting measured passes. Measured
prompt counts must match qualification, preventing silent formatting changes.

Every measured session advances serially; sessions run concurrently and a
finished session frees a worker for the next. Prompt-cache namespaces are
stable per recorded session, isolating within-session reuse from cross-session
reuse. Each pass uses a fresh server; cache remains available across turns.
Temperature 0, seed 42, and the 2,048-token maximum budget are matrix-validated.
The per-turn output budget is estimated from recorded output length, as before.

Raw turn records are flushed as they finish, including failures. Completeness
requires every expected ID exactly once and in order within its session.
Per-session summaries expose token-weighted cache rate, lengths and per-turn
TTFT/cache values. Each measured session must show later-turn prompt reuse.
Cache rate need not increase monotonically.

## Recurrent acceptance

Enable the runtime's structured stderr telemetry. Correlate one lookup decision
per sequential turn using the stable session cache namespace. Require actual
`exact_hit` restores with `kv-recurrent` or `recurrent-only` payloads on a later
recorded-history request in **every** measured recurrent session, restoring at
least the long-context threshold (32,768 tokens). A trivial one-token hit cannot
qualify the long-prefix workload. Extra/missing
lookup events fail correlation. Capture counters and ordinary cached-prompt
counts do not prove recurrent restores. A generated-continuation snapshot only
helps when its tokens match a later recorded request's prefix.

Raw native logs and attributed restore events remain in the artifacts. The
admission-policy/runtime fix is a separate dependency; this harness neither
implements it nor treats its absence as a successful prefix-only baseline.

## Workflow, history and rollout

The workflow is trusted-main/manual-only until live qualification is reviewed.
Three serial model steps each have a 360-minute ceiling and preserve sibling
model evidence after a failure. Context qualification is included in that
ceiling. The 1,800-minute job leaves setup, repair and upload headroom. The
repair verifier uses the same matrix-to-command implementation; its existing
360-minute agent-plus-verification ceiling also needs calibration. Do not
restore the schedule until these budgets are demonstrated to fit, including
concurrency 8 and uncontended micstudio execution.

History schema 3 requires the raw request/manifest identity and completed
context/turn/recurrent gates. It compares only matching model artifact,
workload, hardware and session-cohort identities. Old checkpoint rows cannot
become full-session baselines. Missing cells/turns/metrics, OOM, qualification
failures and cancellation cannot trigger performance repair. All available
artifacts survive failure/cancellation. Warmup is never included in history.

Build the optional reader with `just with-lld cargo build --locked -p trajectory-reader --features parquet-input --bin trajectory-reader`. Example, after resolving and verifying the pinned model and dataset:

```sh
cargo xtool automation replay-matrix run-family \
  --matrix ci/agentic-replay-nightly/matrix.json \
  --reader "$PWD/target/debug/trajectory-reader" --timeout 21600 \
  --run-family granite-3.1-2b --ref main=HEAD --model-file /path/to/verified.gguf \
  --dataset-file /path/to/sessions.parquet --output /path/to/evidence/dense
```

No live duration or successful long-context qualification is claimed by this
configuration. Session count may be revised after calibration; selected traces
remain complete and long. Runtime/canary work on micstudio must be coordinated
before starting a validation run.

Matrix validation, export, immutable pin projection and downloaded-byte checks
belong to `cargo xtool automation replay-matrix`. `run-family` admits the explicit
local GGUF against the selected immutable pin, selects whole trajectories through
the separately built native Parquet reader, and supervises the Rust `execute-run` owner.
The verified family-to-local-path mapping is retained for repair; repair never
rediscovers or downloads a model. `--reader` selects the explicit native reader executable; it does not select a model-serving process. Progress uses typed stderr records;
request JSONL remains the evidence source. History, card rendering and hardware
capture have typed Rust owners. The manual comparative/L3 caller cutover remains
migration work, separate from native reader qualification.

Benchmark corpus generation also uses this native tool: `just bench-corpus smoke` builds its optional `corpus-input` feature, reads pinned repository Parquet or the declared CommitPackFT/APPS JSONL layouts, and publishes a complete corpus/manifest directory. Use a fresh `--out-root` to regenerate an existing tier. The obsolete DuckDB project and lock are removed; no Python interpreter is used for trajectory or corpus selection.
