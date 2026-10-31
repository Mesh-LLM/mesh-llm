# Four-model KV validation foundation

Explicit manual validation for the four AgentX **Q4_K_M** models in
[models.json](models.json). The catalog is byte-for-byte the manifest from
`6c0464e5531a9ff8ceb89c068ad94cd53c71e334:skippy/evals/agentx/models.json`, SHA-256
`6a2b724cf32208186d80ac614e21a168bfc08ca72e55afdabc3dc39b338c9206`.
It pins repositories, GGUF hashes, model revisions and tokenizer revisions.
Workers never download or substitute another model.

This foundation runs:

* Real-model FA/unified-KV prefill (20–64 tokens) and 16 greedy decode steps.
  CUDA and CPU require all 64 pairs of F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0;
  Metal requires its eight supported same-type pairs and explicitly reports its
  56 preexisting unsupported mixed pairs. Each model loads **once** for this
  matrix, with a fresh context per pair. Every executed attention graph must
  consume indices and lengths, retain requested K/V types, and execute on the
  selected CUDA0, MTL0 or CPU backend. No GPU-to-CPU or dense fallback qualifies.
* One fresh Skippy product server at c1/F16, followed by deterministic
  cold/repeat/growing/repeat HTTP requests. It requires cache hits and exact
  generated-content parity, not just matching output counts.
* The unchanged strict chat-parity and growing-checkpoint Rust regressions for
  the catalog's two recurrent/hybrid workloads. These fixtures retain their
  recurrent message-boundary assertions; dense rows use the generic HTTP gate.
* For those hybrid rows, cold eviction of a 64-token native checkpoint after
  releasing its source lane, followed by model/runtime/cache reload against the
  same L3 directory. Require an actual L3 hit, identical attention and recurrent
  payload bytes, and four matching fixed-input continuation predictions. This
  runs in one test process and does not qualify fresh-process server restart.
  A second model test refuses L3 writes with a one-byte disk budget, requires
  native attention cells to be released, and verifies exact cold-recompute
  predictions before checking that no durable entry was published.

Long c4 state comparisons, L3 restart, pressure, default-budget qualification and
standard-vanilla performance comparisons remain separate work in
[todo.md](../../../todo.md). A green foundation is not throughput parity or full
KV certification. There is no automatic GPU CI trigger or new serving option.

## Prepare once, run many

Build the matching product/runtime and the dynamic serving test binary through
the existing Just/native workflow. Build `skippy-correctness` with the dynamic
runtime dependency feature. The producer image must contain those exact tools,
the prepared native source, the workspace and this automation executable.
Prepare once **inside that immutable image**, preserving paths across workers.
No worker rebuilds native code, Rust or the model matrix executable.

Copy [producer.example.json](producer.example.json), replace its absolute paths
and source commit, and observe byte pins. Empty SHA fields are allowed only as
inputs to the explicit `pin` operation; `prepare` and `run` refuse missing pins.

```sh
just automation-run automation kv-validation pin \
  --input /absolute/producer.paths.json --output /absolute/producer.pinned.json
just skippy-kv-prepare /absolute/producer.pinned.json /absolute/kv-prepared
```

`prepare` verifies the exact runtime leaf/backend/tree hash and tool bytes,
checks the clean native source HEAD before and after compiling the external
test harness, and emits `/absolute/kv-prepared/prepared.json`. It uses shared
libraries and matching private graph headers, without modifying the native
patch queue. Source declarations are provenance; artifact hashes and the trusted
producer image bind the executable bytes actually consumed.

Copy [cell.example.json](cell.example.json), supply the prepared receipt SHA
(`shasum -a 256 /absolute/kv-prepared/prepared.json`) and the actual model path.
All artifact paths must be absolute and outputs fresh.

```sh
just skippy-kv-validate /absolute/cell.json /absolute/new-evidence
```

Choose `cpu`, `metal` or `cuda` in the producer; run a separate prepared producer
for each backend. `run` validates the exact model SHA and GGUF dimensions, checks
the exact runtime leaf with the product doctor, supervises all children with
deadlines/cancellation, and preserves failed evidence. The fixture adapter's
runtime-parent directory contains only the doctor-verified leaf. Disk is off,
checkpoint RAM is explicitly 8 GiB, and server settings/arguments are retained.

Output includes `result.json`, sanitized child logs, settings/identities, typed
cache content/counter evidence in `cache-evidence.json`, and a sibling
`new-evidence.tar.gz` and its `new-evidence.archive.json` hash. Runtime symlinks
are excluded from the archive; their identity remains in JSON. Required child
failure, numerical/content mismatch, missing evidence, deadline or cancellation
fails the command. Preserve both the failed receipt and evidence archive.
The cache receipt is a closed typed projection observed before log sanitization;
it preserves only the declared content, counters and status, never arbitrary
child payloads, protocol headers or environment data.

## Explicit JY jobs

Copy [fleet.example.json](fleet.example.json), bind the **prepared image's
immutable SHA**, automation path and prepared-receipt path/SHA, then select
eligible Sparks. The planner rejects both excluded node IDs (`spark-000280`,
`spark-000499`) and hostnames (`spark-db08`, `spark-9f06`). Each worker also checks
its actual hostname. Jobs are independent per model and reuse the prepared
image; JY's node lease queues jobs rather than stacking GPU measurements.

```sh
just skippy-kv-jy-plan /absolute/fleet.json /absolute/new-jobs
jy submit /absolute/new-jobs/qwen3-4b-q4-k-m.json
jy submit /absolute/new-jobs/qwen3.5-4b-q4-k-m.json
jy submit /absolute/new-jobs/qwen3-coder-30b-a3b-q4-k-m.json
jy submit /absolute/new-jobs/qwen3.5-35b-a3b-q4-k-m.json
```

These specs use the accepted lab/custom JY schema: pinned `model_package`,
stream-delivered `AGENTX_MODEL_PATH`, immutable workspace revision, JY-owned
artifact/result paths and a thin Docker adapter invoking the typed worker.
No patches, credentials or arbitrary policy scripts are embedded. Flat result
and compressed evidence files survive JY's flat-file artifact transport.
The current JY adapter supports Linux CPU/CUDA only. Run Metal natively with the
same `prepare`/`run` commands; a native macOS JY executor requires a separately
verified executor contract and is not silently emulated in Docker.
