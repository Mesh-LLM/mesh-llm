# Fixed complete AgentX comparison

Benchmark-only extraction from PR #2463 source `9684f77ffd95d049eefd375366f31df66efbb56b`.
It runs the locked **external** AgentX1.0.12.post2 client; no Python implementation
or old serving/native fix is transplanted. The four exact Q4_K_M models are
owned by [the KV catalog](../kv-validation/models.json), SHA256
`6a2b724cf32208186d80ac614e21a168bfc08ca72e55afdabc3dc39b338c9206`.

Each model/backend campaign runs six full c1/c2/c4 sweeps in llama, Skippy,
Skippy, llama, llama, Skippy order. Each cell serves the same four intact traces:
69 requests and18,713 requested outputs, original children, think times and
turn/output identities. Missing/repeated/shortened/cancelled/error work or a
changed cohort/numerator fails. Token-count hashes prove work, not token parity.
Run the existing native and `skippy-kv-validate` correctness gates first and
retain their identities alongside the campaign. No throughput parity is implied
by successful work accounting.

## Prepared inputs

Build the product/runtime and independent **unpatched** standard llama-server
once per backend through their owning Just/producer workflow. Baseline source
must equal `skippy/llama_cpp/upstream.txt`. Reuse immutable prepared tools; no
campaign builds or downloads. If preparing the external client, `just
agentx-setup` installs the locked upstream environment; workers consume it
offline. Site packages/interpreter file bytes are pinned, as are the executable,
runtime, tokenizer snapshot and corpus. External directory links/editable
environments are refused; internal aliases such as `lib64 -> lib` are pinned
without traversal, and regular-file interpreter links are hashed by content.

Place the tokenizer snapshot in its pinned Hugging Face cache layout and set
`HF_HUB_CACHE` (or `HF_HOME`) to that cache before pinning and running. The
offline client receives the catalog's repository ID and exact revision; pinning
requires that its resolved snapshot is the same canonical directory as
`tools.tokenizer.path`. The child receives that cache path explicitly, so it
cannot silently consume a different snapshot from the operator's home cache.

Create `/abs/request.paths.json` with:

```json
{
  "schema_version": 1,
  "campaign": "agentx-final-YYYYMMDD",
  "workspace": "/opt/mesh-llm",
  "model_key": "qwen3.5-4b-q4-k-m",
  "model": "/models/Qwen3.5-4B-Q4_K_M.gguf",
  "corpus": {"path": "/models/traces.jsonl", "sha256": ""},
  "tools": "REPLACE_WITH_OBJECT_FROM_tools.example.json"
}
```

Replace `tools` with the object in [tools.example.json](tools.example.json),
actual absolute producer paths and declared product/native source IDs. Those
IDs are provenance; pins observe the actual bytes consumed. `pin` checks the
runtime leaf/backend/ABI/native identity and the exact locked client/catalog/
corpus. It cannot prove a binary was built from an arbitrarily declared commit.

```sh
just automation-run automation agentx pin --input /abs/request.paths.json --output /abs/request.pinned.json
just agentx-comparison /abs/request.pinned.json /abs/new-campaign
```

The corpus is `semianalysisai/cc-traces-weka-062126-256k` revision
`8fecd2fc56694469f758f0afbbb6335ad3043740`, `traces.jsonl` SHA256
`e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef`.
Weights and independent tokenizer revisions must already be cached. A fresh
output is required. Artifacts include per-cell launch/effective config, native
capacity observations, raw client records/HTTP traces, lifecycle/memory/work
receipts, scalar summaries, final result and a sibling archive+SHA receipt.
Failure retains partial evidence and returns nonzero. Clean process-tree
shutdown is required; no partial campaign becomes a successful point.
Doctor, effective-config and MLflow JSON are parsed from bounded structured
capture, independently of redacted diagnostic logs. Public evidence projections
remain available even when ordinary logs suppress a line containing `token`.

## Capacity and backend policy

Skippy preserves its product-auto generation-lane plan independently of offered
c1/c2/c4. It retains the128K per-request cap and records resolved lanes/context/
configured pool separately from observed native allocation. It explicitly uses
checkpoint RAM8GiB, disk off, F16 K/V, FA, batch2048/microbatch512, no speculation,
and matched thread counts. Frontend prefill is explicitly fixed at64 and
validated in the effective config; it is not inferred from microbatch size. An explicit
`diagnostic_generation_lanes` is separately labelled and cannot quietly change
the default policy.

Standard vanilla explicitly uses separate KV slots, parallel=C,
context=C×131072, F16 FA, batch2048/ubatch512 and cache RAM8192MiB. Each baseline
must report the expected slots/context. Unified-llama is not substituted.

Use `backend` cuda/metal with matching prepared bundles/hosts. CPU requires
`diagnostic_cpu:true`; its report is explicitly ineligible for GPU performance
publication. Independent model campaigns can occupy equivalent idle machines;
keep each paired campaign on one host. The actual hostname guard refuses
spark-db08/spark-9f06 (also spark-000280/spark-000499). The admitted external JY
wrapper must also exclude those node IDs and enforce its lease; this command
does not invent a macOS Docker executor or change registered templates.

## Existing MLflow experiment

After auditing the complete campaign, make `/abs/export.json`:

```json
{
  "report": {"path": "/abs/new-campaign/result.json", "sha256": "ACTUAL_REPORT_SHA256"},
  "archive": {"path": "/abs/new-campaign.tar.gz", "sha256": "ACTUAL_ARCHIVE_SHA256"}
}
```

```sh
just agentx-mlflow-export /abs/export.json /abs/new-plan.json
just automation-run automation agentx mlflow-publish --input /abs/new-plan.json --output /abs/new-publication
```

Export does not write MLflow. Explicit publication uses existing operator
`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`
through a private temporary curl config, never job payloads or source. It first
GETs existing experiment6 and requires exact name `agentx-skippy-vs-llama` and
active state. There is no experiment-create call. A new idempotent campaign
root contains model/backend groups and their six arm children through
`mlflow.parentRunId`. The common campaign root remains RUNNING until a separate
eight-group audit; each publication finishes only its model/backend group. Historical source-pinned
runs and ablation parents remain separate. Identity conflicts fail. Per-arm
c1/c2/c4 scalar values, source/native/queue/runtime/tool/corpus/cohort/work
identities and archive hashes are recorded; `cache_read_tokens_avg` is an average,
not a total. Then
metrics are read back, artifacts uploaded/downloaded with matching SHA, then
publication receipt is written. A publishing failure is nonzero and retains
API evidence for retry. Publication is not a numerical-correctness certificate.

The ordinary CI/GPU suite and registered JY templates are unchanged. This
benchmark owner does not mark the long-prefix/cancellation/L3 TODO complete.
