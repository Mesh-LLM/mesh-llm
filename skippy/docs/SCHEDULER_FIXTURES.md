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
[`skippy/skippy/evals/skippy-scheduler-fixtures.json`](../skippy/evals/skippy-scheduler-fixtures.json).
It pins the Hugging Face commit, selected source, eight session IDs, selection
rules, runtime shape, generated prompt-manifest hash, exact GGUF repository
revision and content hash (including its embedded tokenizer), and acceptance
bounds. It contains no trajectory text or dataset files.

## Fast PR gate

Validate the catalog and run the actual Rust scheduler against both compact
traces:

```bash
just with-lld cargo xtool automation agentic-prompt-manifest validate-fixtures \
  skippy/evals/skippy-scheduler-fixtures.json
just with-lld cargo test -p skippy-scheduler
```

The Rust replay reads the checked-in catalog directly. With
`group_waiting_prefixes=false`, the warm trace retains one family switch and
the eviction-pressure trace retains fifteen. With grouping enabled, the warm
trace remains at one switch while the eviction-pressure trace collapses to
seven. This gate is deterministic and performs no model inference or network
access.

The native prompt-manifest tests validate the catalog, derive the minimum
context from the checked-in rows,
check strict HF command arguments and reject row or manifest hash drift before
publication. Run these portable contracts locally with:

```bash
just with-lld cargo test --locked -p xtask --bin xtask agentic_prompt_manifest \
  -- --test-threads=1
```

The A/B runner also checks profile application. Synthetic fixtures do not prove
the generated manifest hash against the pinned corpus; that requires the exact
verified dataset revision.

## Optional Parquet reader

The default automation bootstrap compiles portable selection contracts only.
Build the optional reader separately when materializing Parquet inputs:

```bash
just with-lld cargo build --locked -p trajectory-reader --features parquet-input \
  --bin trajectory-reader
```

The reader includes the Parquet compression codecs. The main automation tool
finds it next to its executable, or uses an explicit absolute
`MESH_LLM_TRAJECTORY_READER_BIN` path. The tool supervises the reader with a
120-second deadline and validates its versioned response file before publication.
Selected trajectory data uses a private temporary file, independently of the
bounded diagnostic capture. The normal `just ci-automation-contracts` gate also
builds the reader, tests actual compressed Parquet inputs, and executes the
frontend and reader together to check exact manifest bytes and refusal cleanup.
Catalog validation and profile display require neither the reader nor HF.

## Pinned corpus cache

Prepare the model-backed eviction-pressure prompts with:

```bash
just with-lld cargo xtool automation agentic-prompt-manifest prepare-fixture \
  --catalog skippy/evals/skippy-scheduler-fixtures.json \
  --profile agentic-eviction-pressure \
  --hf-bin "$(command -v hf)" \
  --output /tmp/skippy-agentic-eviction-pressure.json
```

`prepare-fixture` requires an installed Hugging Face CLI at the explicit
absolute `--hf-bin` path. It performs both operations required by the fixture
contract:

1. `hf download thoughtworks/agentic-coding-trajectories ... --repo-type dataset --revision cef72d1f4d0caabf85937adf8337a14b7522c782`
2. `hf cache verify ... --fail-on-missing-files` at the same revision

It then selects the pinned rows from `sessions.parquet`, rebuilds the prompt
manifest, and rejects either row drift or a SHA-256 other than
`f1ddbe3d5974f3f4bd06f5d70fa45d0e10305bbafa4eb7399a0f972458d1beef`.
Use `--cache-dir` when a benchmark host owns a dedicated shared cache. The download
and verification share a 600-second timeout, configurable with `--timeout`.
Catalog validation and local materialization do not start HF or install Python
packages. To regenerate from an already verified local Parquet file:

```bash
just with-lld cargo xtool automation agentic-prompt-manifest materialize-fixture \
  --catalog skippy/evals/skippy-scheduler-fixtures.json \
  --profile agentic-eviction-pressure \
  --dataset-file /path/to/verified/sessions.parquet \
  --output /tmp/skippy-agentic-eviction-pressure.json
```

Never copy `sessions.parquet` or the generated prompt manifest into the
repository. The corpus is a derivative multi-source dataset; this fixture uses
only the `swe-smith-claude-3-7-sonnet` rows, whose upstream is
`SWE-bench/SWE-smith-trajectories` (MIT). The checked-in catalog retains row
provenance without redistributing the text.

The native fixture commands own catalog validation, profile resolution, pinned HF
fetch/verification, and prompt-manifest publication. Native fixture tests do not qualify
hardware replay, corpus acquisition, or its acceptance metrics.

