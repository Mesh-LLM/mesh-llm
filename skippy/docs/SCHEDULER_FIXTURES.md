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

## Periodic hardware replay

Use exact OLD and NEW release binaries built against the same native ABI, then
run the A/B harness with the named profile:

```bash
python3 skippy/evals/skippy-waiting-prefix-ab.py \
  --fixture-profile agentic-eviction-pressure \
  --acceptance-contract skippy/evals/skippy-capacity-acceptance.json \
  --prompt-manifest /tmp/skippy-agentic-eviction-pressure.json \
  --case-file /path/to/one-model-case.json \
  --old-bin /path/to/old/skippy-serving \
  --new-bin /path/to/new/skippy-serving \
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

The same replay is also the capacity-policy certificate. Pass the checked-in
`skippy-capacity-acceptance.json` contract when comparing the capacity layer;
the measured agentic requests remain identical, but the gate changes from
proving the DFS gain a second time to requiring actual eviction, zero
fail-closed rejections, and no regression over 2% in recomputation, p95 TTFT,
makespan, or throughput. The contract first
seeds eight deterministic synthetic resident prefixes and raises the entry cap
to sixteen, so the measured agentic requests encounter evictable cold state
without reducing the validated per-lane context budget. The runner combines
pre-admission and post-record resident eviction telemetry into per-round token
and entry totals, reports fail-closed capacity rejections, and retains the
planner's deterministic work estimate. Because the current per-token estimate
is uniform within a stage, the certificate describes the effective victim
policy as cold-first LRU rather than attributing results to cost density. This
lets a stacked capacity change be compared against the preceding scheduler
binary without changing the pinned requests or silently treating legacy
proactive eviction as zero.

The waiting-prefix eviction-pressure certificate requires every request to succeed and, at
minimum, a 50,000-token suffix-prefill baseline and eight family switches so a
drifted non-pressure workload cannot pass. It then requires 10% improvements
in suffix prefill, family switches, p95 TTFT, and makespan plus 10% higher
output throughput. The warm certificate requires
identical suffix prefill and switch counts, with user-facing timing/throughput
movement inside ±5%. A zero baseline is neutral only when both binaries remain
at zero; any nonzero candidate value fails closed. Alternate binary order
across all four rounds and retain raw requests, telemetry, configs, logs,
`comparison.json`, and `report.md`.

## Offline A/B acceptance

Check measured A/B aggregates with the native acceptance command:

```bash
just with-lld cargo xtool automation waiting-prefix evaluate \
  --comparison comparison.json \
  --catalog skippy/evals/skippy-scheduler-fixtures.json --profile warm-affinity \
  --output acceptance.json --report report.md
```

For the capacity bounds, replace the catalog/profile pair with
`--contract skippy/evals/skippy-capacity-acceptance.json`. The command checks complete
request success, available measurements, baseline pressure and the selected
regression or gain bounds. A failed measurement comparison writes its check
evidence and report, then exits unsuccessfully. Invalid inputs fail before
publication. The model-backed workload runner still supplies the measured
aggregates.

The native `waiting-prefix summarize --input FILE --output FILE` command
combines request outcomes, generation events, capacity decisions and proactive
eviction decisions into one round. `waiting-prefix aggregate --input FILE
--output FILE` accepts a document with a `cells` list and computes per-binary
round medians. Missing numeric telemetry remains null. Repeated request or
round identities are rejected. These commands support offline evidence analysis;
the model-backed A/B workload runner remains in Python during its cutover.


The native workload plan resolves the entire selected profile before execution:

```bash
just with-lld cargo xtool automation waiting-prefix plan \
  --catalog evals/skippy-scheduler-fixtures.json --profile warm-affinity \
  --model-id "$MODEL_ID" --model-sha256 "$MODEL_SHA256" --output plan.json
```

Use the model identity pinned in the selected catalog profile. For an HF profile,
pass `--prompt-manifest FILE`; its exact bytes must match the catalog SHA-256
and cover every family and request. Synthetic profiles omit that option.
`--contract FILE` applies only the documented cache-entry override and records
cache seeding separately from the measured request count. The plan records the
catalog, contract and manifest hashes. Resolving a plan does not verify a model
file or run inference.

The native `waiting-prefix execute-requests --input FILE --output FILE` command
runs a staggered request phase against an already-running local server. Its
schema-1 input requires `round`, `version` as `old` or `new`, `base_url` as
`http://127.0.0.1:<port>/v1`, `model`, `output_tokens`,
`request_timeout_secs`, `stagger_ms`, and a nonempty `prompts` list of
`family` and `prompt` strings. Results retain request identities in order,
streaming usage, timing and content hashes. HTTP failures, timeouts and
interruption retain failed request evidence and exit unsuccessfully; invalid
input preserves any previous output. The command requires complete streaming
usage and the terminal marker. Server startup, cache seeding, telemetry capture
and the complete old/new comparison remain owned by the Python workload runner
until its replacement passes validation.


For synthetic profiles, `waiting-prefix synthetic-prompts --families N
--requests-per-family N --prefix-blocks N --output FILE` generates the
`stable-prefix-v1` repository contexts in interleaved family order. Using one
request per family generates the same initial tasks for cache seeding. Counts
must be positive and fit the request phase's 10,000-request limit; a conservative
256 MiB allocation budget rejects oversized repeated contexts before generation.

`waiting-prefix stage-config --input FILE --output FILE` emits a single-stage
`runtime-slice` config with `lookup-record` cache mode and one shared-prefix
record. Input contains `model_id`, an absolute `model_path`,
`source_model_sha256`, `layer_end`, `ctx_size`, `lane_count`,
`n_gpu_layers`, `payload` and `cache_entries`. Payload is `resident-kv`,
`kv-recurrent` or the `full-state` correctness diagnostic. The command checks
positive layer/context/lane/cache sizes and the actual model file SHA-256 before
publishing; a changed model preserves any previous config. This prepares
configuration only. It does not inspect GGUF dimensions or start a runtime.


The native `waiting-prefix telemetry-log snapshot --log FILE --output FILE`
command records the seed boundary for generation, capacity and record events.
`telemetry-log collect --log FILE --cursor FILE --expected-generations N --output FILE`
retains only events after that boundary and requires exactly N generation
summaries. The cursor binds the canonical log path, its byte prefix SHA-256,
and recomputed event counts. Appends are accepted; rewrites, truncation,
cross-log cursors, altered counts, and malformed recognized attributes are
rejected before publication. Parsing is bounded to 1 MiB per line and 128 MiB
per snapshot. This collects existing debug telemetry and emits no runtime metrics.


The native `waiting-prefix cell-worker --input FILE --output FILE` waits for the exact local model, optionally seeds its cache, waits for seed telemetry, and measures requests using a verified telemetry-log boundary. It retains completed request evidence when measurement or telemetry fails. The parent must own the server process and stderr log. This worker does not yet replace the complete Python A/B round runner or establish model-backed acceptance.


The native `waiting-prefix server-cell --input FILE --output-directory DIR` owns a pinned Skippy server and its measurement worker under one retained process session. It checks actual binary/model hashes, GGUF context and full layer dimensions, stage/request identity and admission capacity before creating a fresh output directory. It reserves its own loopback endpoint, retains stderr telemetry separately from stdout, stops its owned server after the worker finishes, and writes `lifecycle.json` after cleanup. Local executable fixtures cover process ownership; native-runtime bundle identity and full old/new round qualification still require the complete runner before caller cutover.


The native `waiting-prefix run --input FILE --output-directory DIR` resolves pinned catalog/contract workload and prompts, checks binary/model and provided native artifact bytes, alternates every old/new round, retains attempted cells and failed evidence, and requires the exact complete cell census before aggregation and hardware acceptance. It supervises each retained server-cell under one outer cancellation scope, bounds total process/evidence budgets, and preserves an existing output directory. The supplied commit labels and provided runtime artifact pins describe their respective inputs; they do not alone prove a loaded runtime or model performance. Native executable fixtures and live model/corpus qualification remain distinct. The Python production caller remains pending qualified cutover.
