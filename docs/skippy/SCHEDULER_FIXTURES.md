# Scheduler workload fixtures

The scheduler fixture catalog preserves the two workload shapes that define the
waiting-prefix policy boundary:

- `warm-affinity` is an already-grouped two-family trace. Materialized cache
  affinity supplies the same order before waiting-prefix DFS, so the replay
  must stay neutral.
- `agentic-eviction-pressure` is an interleaved eight-family trace with only
  two resident prefix-cache entries. Its 131,072-token aggregate context is
  the smallest power-of-two capacity covering the pinned row totals, request
  multiplicity, and output allowance. Waiting-prefix DFS must reduce family
  switches and the periodic model-backed run must reduce recomputation and
  tail latency.

The source of truth is
[`evals/skippy-scheduler-fixtures.json`](../../evals/skippy-scheduler-fixtures.json).
It pins the Hugging Face commit, selected source, eight session IDs, selection
rules, runtime shape, generated prompt-manifest hash, exact GGUF repository
revision and content hash (including its embedded tokenizer), and acceptance
bounds. It contains no trajectory text or dataset files.

## Fast PR gate

Validate the catalog and run the actual Rust scheduler against both compact
traces:

```bash
python3 evals/skippy-scheduler-fixtures.py validate
just with-lld cargo test -p skippy-scheduler
```

The Rust replay reads the checked-in catalog directly. With
`group_waiting_prefixes=false`, the warm trace retains one family switch and
the eviction-pressure trace retains fifteen. With grouping enabled, the warm
trace remains at one switch while the eviction-pressure trace collapses to
seven. This gate is deterministic and performs no model inference or network
access.

The normal Python script-test lane also validates the catalog, derives the
minimum context from the checked-in rows, exercises the real prompt generator
with canned rows, verifies the strict HF command shape and row-provenance
rejection, and covers profile application in the A/B runner.

## Pinned corpus cache

Prepare the model-backed eviction-pressure prompts with:

```bash
python3 evals/skippy-scheduler-fixtures.py prepare \
  --profile agentic-eviction-pressure \
  --output /tmp/skippy-agentic-eviction-pressure.json
```

`prepare` performs both operations required by the fixture contract:

1. `hf download thoughtworks/agentic-coding-trajectories ... --repo-type dataset --revision cef72d1f4d0caabf85937adf8337a14b7522c782`
2. `hf cache verify ... --fail-on-missing-files` at the same revision

It then selects the pinned rows from `sessions.parquet`, rebuilds the prompt
manifest, and rejects either row drift or a SHA-256 other than
`f1ddbe3d5974f3f4bd06f5d70fa45d0e10305bbafa4eb7399a0f972458d1beef`.
Use `--cache-dir` when a benchmark host owns a dedicated shared cache.

Never copy `sessions.parquet` or the generated prompt manifest into the
repository. The corpus is a derivative multi-source dataset; this fixture uses
only the `swe-smith-claude-3-7-sonnet` rows, whose upstream is
`SWE-bench/SWE-smith-trajectories` (MIT). The checked-in catalog retains row
provenance without redistributing the text.

## Periodic hardware replay

Use exact OLD and NEW release binaries built against the same native ABI, then
run the A/B harness with the named profile:

```bash
python3 evals/skippy-waiting-prefix-ab.py \
  --fixture-profile agentic-eviction-pressure \
  --prompt-manifest /tmp/skippy-agentic-eviction-pressure.json \
  --case-file /path/to/one-model-case.json \
  --old-bin /path/to/old/skippy-server \
  --new-bin /path/to/new/skippy-server \
  --old-commit <old-commit> \
  --new-commit <new-commit> \
  --native-build /path/to/matched/native-build \
  --output-dir /path/to/artifacts
```

The named profile owns rounds, lanes, admission concurrency, cache entries,
output length, and arrival stagger; ad hoc workload flags do not override it.
The case file must also match the profile's pinned model ID and GGUF SHA-256.
HF profiles require their exact generated prompt manifest, while synthetic
profiles reject external manifests. The result records the profile name and
catalog SHA alongside binary, model, and prompt-manifest hashes.

The eviction-pressure certificate requires every request to succeed and, at
minimum, a 50,000-token suffix-prefill baseline and eight family switches so a
drifted non-pressure workload cannot pass. It then requires 10% improvements
in suffix prefill, family switches, p95 TTFT, and makespan plus 10% higher
output throughput. The warm certificate requires
identical suffix prefill and switch counts, with user-facing timing/throughput
movement inside ±5%. A zero baseline is neutral only when both binaries remain
at zero; any nonzero candidate value fails closed. Alternate binary order
across all four rounds and retain raw requests, telemetry, configs, logs,
`comparison.json`, and `report.md`.
