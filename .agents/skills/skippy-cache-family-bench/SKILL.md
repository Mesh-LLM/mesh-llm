---
name: skippy-cache-family-bench
description: Use when benchmarking Skippy exact-prefix cache across model families and comparing Skippy against llama-server, producing README benchmark tables, or diagnosing cache behavior.
metadata:
  short-description: Benchmark Skippy cache by family
---

# Skippy cache family benchmarks

Repository orchestration follows `../manage-ci/SKILL.md`: typed native owners
behind Just recipes. Cache runs compare production `ResidentKv` and
`KvRecurrent` payloads. `FullState` is not a production cache mode.

Use an explicit operator request with the existing cache-family-plan selection
and locally prepared tools. Native preparation observes tool/build/model bytes,
constructs the complete selected catalog profiles, and records missing-model
rows without launching them. Source commits are caller-declared build provenance,
not proof that a binary was built from that source.

For the CPU workload tools, prepare through the existing recipe:

```bash
just skippy-workload-oracles-build /tmp/cache-tools
```

Its outputs are `/tmp/cache-tools/native/bin/llama-server`,
`/tmp/cache-tools/cargo/debug/skippy`, `skippy-correctness`, and
`skippy-package-builder`, plus `producer.json`. An operator request uses those
paths, the native directory and source commits. For another backend use already
prepared, matched host tools and native build; do not substitute a downloaded
runtime for changed native ABI. MiniMax needs all three sibling shards; DeepSeek3
uses package-only admission with no monolithic baseline. Preparation requires
`artifact_tool` for either artifact kind.

The operator JSON contains `schema_version: 1`, `plan` (the existing
cache-family-plan input), `correctness`, `stage_server`, optional `native_server`
and `artifact_tool`, `native_build`, `old_source_commit`, `new_source_commit`,
`native_source_commit`, `environment`, `toolkit_directories`, and
`execution_seconds`, `cell_seconds`, `request_timeout_ms`, `preparation_seconds`.
`borrow_resident_hits` and `cache_decoded_result_hits` default to false.
Use `cases: []` for all fourteen catalog presets; `cases` and `prefix_sweep`
retain the native planner's selection/order rules. Put the checked use-case
corpus path and byte SHA in `plan.corpus`. The wrapper binds its selected
`SKIPPY_CACHE_USECASE_CORPUS` (default checked-in corpus) through the native
bounded reader and uses that same source for combined reporting. `prepare-full` removes corpus selection;
`prepare-use-cases` selects its complete `all` roster. The native planner validates
all declared bounds and source revisions. Native preparation creates the output
requests and observations; operators do not hand-write each model profile.

Old/new comparison is optional: both `plan.old_server` and `plan.new_server`
are null to omit it, or both bind their actual path and SHA. `stage_server` remains
the correctness tool's server even when paired serving is omitted. The explicit
nonsecret `environment` is shared across correctness and serving. Backend/device
settings have finite allowed names; external toolkit directories use absolute
paths and byte-tree SHA pins. Conflicting aliases or changed pins refuse.

Run both matrices and their combined report:

```bash
SKIPPY_CACHE_OPERATOR_INPUT=/absolute/cache-operator.json \
  SKIPPY_CACHE_SKIP_BUILD=1 \
  skippy/evals/skippy-cache-family-bench.sh /tmp/cache-family-run
```

The wrapper builds with `just skippy-workload-oracles-build` unless build skipping
is explicit. Set `SKIPPY_CACHE_BUILD_OUTPUT` to the tool directory referenced by
the operator request. Defaults preserve prefix128, one runtime lane,
llama parallel1, three serial repeats and three cache-hit repeats; the existing
`PREFIX_TOKENS`, `RUNTIME_LANE_COUNT`, `LLAMA_PARALLEL`, `LLAMA_REPEATS`, and
`CACHE_HIT_REPEATS` environment overrides remain effective. Tool paths and build
provenance are authoritative fields in the operator request.

Each native concurrent/old/new curve keeps one retained host across the complete
ordered concurrency ladder. The serial baseline remains separate and excludes
its first request as warmup. Cell admission verifies pins before and after the
run; later custody/cleanup failure retains observations but refuses accepted
parity. Cancellation/deadline/interrupt-finish failure downgrades final receipts.

Manual native commands are:

```bash
just automation-run automation cache-family-run prepare-full \
  --input /absolute/cache-operator.json --output /tmp/cache-full-input
just automation-run automation cache-family-run \
  --input /tmp/cache-full-input/cache-family-input.json --output /tmp/cache-full
just automation-run automation cache-family-run prepare-use-cases \
  --input /absolute/cache-operator.json --output /tmp/cache-usecase-input
just automation-run automation cache-family-run \
  --input /tmp/cache-usecase-input/cache-family-input.json --output /tmp/cache-usecase
just automation-run automation cache-family-report \
  --input /tmp/cache-full/production-cache-bench.json \
  --input /tmp/cache-usecase/production-cache-bench.json \
  --output /tmp/cache-readme.md
```

Keep raw JSON/cell/process receipts with the report. Supplied byte pins and
inert owning fixtures establish orchestration contracts; actual current native
ABI/model/backend measurements remain separately qualified. Do not publish
performance claims from synthetic fixtures. Exact suffix-prefill depth is an
explicit correctness capability, separate from prefix/context sizing.

Rows stay grouped by Qwen3Next, Falcon-H1, Llama, Qwen3 dense, DeepSeek2,
GLM-4.7 Flash, GLM4, Gemma4 A4B, Gemma4 E4B, Gemma3, Gemma2, OLMo, and
MiniMax M2.7. Compare full-GGUF rows to the monolithic baseline; retain package-only
rows as unavailable baseline. Keep one generated token and matched prefixes for
serial comparisons. Failed correctness rows do not become promoted README
performance evidence.

Preparation keeps a split model's logical snapshot first-shard entrypoint while hashing canonical backing bytes, so HF symlink sibling discovery remains intact. Its fresh output must be disjoint from cache, native build and toolkit source trees. Final input publication rechecks deadline/cancellation after the owned fresh write and revokes that file on late refusal, retaining preparation observations.
