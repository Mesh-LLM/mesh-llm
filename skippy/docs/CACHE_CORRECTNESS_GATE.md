# Cache correctness gate

Prepare local observations with the existing operator owner:

```sh
cargo xtool automation cache-family-run prepare-full --input /absolute/operator.json --output /absolute/fresh-preparation
cargo xtool automation cache-family-correctness batch --input /absolute/gate.json --output /absolute/fresh-gate
```

The operator request selects local catalog models, correctness/stage tools, a native build directory, source revisions, and explicit environment/toolkit settings. Preparation hashes actual bytes; supplied source revisions do not establish a build or loaded runtime attestation. Present batch models require their observed correctness profile. Missing models produce unqualified missing-model rows, never synthetic passes.

A gate request is:

```json
{"schema_version":1,"prepared_input":"/absolute/fresh-preparation/cache-family-input.json","cases":[],"prefix_tokens":null,"n_gpu_layers":null}
```

An empty case selection uses the intersection of the historical default roster and the current source-owned catalog: qwen3_dense, llama, glm4, gemma3, falcon_h1, olmo, qwen3next. Prepare those selected profiles before running the batch. Explicit cases are closed current catalog keys. The default topologies are one-stage, split-middle, and split-final; explicit topologies also admit split-stage0. Package-stage1 belongs to the existing package correctness frontdoor and is not one of this historical batch's topologies.

Optional fields are topologies, prefix_tokens, cache_hit_repeats (3), runtime_lane_count (4), n_gpu_layers, and execution_seconds (900, bounded to 4–86400). Without a prefix override, each catalog prefix is capped at 32. Resident hit borrowing is enabled; decoded-result hit reuse is disabled. The historical Python suffix option defaulted to 3 but never reached its child; this adapter preserves the native one-token suffix and reports actual native fields where present. Missing native sequence/storage measurements remain null.

The batch retains cache-correctness-gate.json raw observations and existing admission/process/trial evidence, cache-correctness-table.json rows, and cache-correctness-table.md. Rows preserve family/model/payload/topology, pass/missing/refusal/failure, remapping and sequence IDs, prompt/suffix counts, payload/recurrent/KV/storage bytes, repeats/matches, and pass versus disabled-or-recompute promotion. Suffix and Hits Markdown columns retain the native match booleans; counts remain in JSON. Final table/Markdown/summary writes are checked before and after each owned publication; late refusal rewrites noncomplete diagnostics through retained owned file descriptors. Failed or terminally refused batches exit nonzero and retain observations; all-missing batches may complete orchestration without qualifying any model. batch-summary.json records this scope and prepared settings.

The gate measures correctness only. It does not replace the full cache matrix or establish serving throughput, package baseline, family promotion, hardware cadence, or real-model qualification. Output directories must be fresh. Prepared paths and byte pins remain subject to the existing producer's before/after custody admission. One inherited absolute batch deadline and cancellation owner supervise every child, with each child's existing phase cap and cleanup reserve.
