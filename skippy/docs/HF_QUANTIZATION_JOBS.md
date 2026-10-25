# HF quantization Jobs

Use `model-package-generic-jobs` for offline planning and preparation, explicitly
confirmed submission, and correlated collection. Its quant worker is the pinned
image runner's `automation hf-certify quant-job-worker`. Quant-only uses workflow
`quantization` with the original 259200-second allowance; combined publication
and packaging uses `quantization-and-package` with 345600 seconds.

The current source-built `skippy-quantize` llama-api/skippy-abi backend still
rejects `--max-memory` and partial split windows. The native-rust backend converts
but does not quantize. This workflow requires a supplied compatible quantizer
with exact executable, source and native-runtime pins. Independently qualify its
actual recipe, single-window execution and 32G hard memory policy before a real
Job. A manifest, image digest, successful fixture or receipt is not model/RSS,
native ABI, image, pricing or hosted qualification. The coordinator always records
`tool_profile_qualified:false` and `workflow_qualified:false`. Separate
qualification evidence does not change those receipt flags.

## Prepare the closed request

Build the facade through `just generic-conversion-jobs-release-build` and use
[the existing helper preparation and request transport](HF_GENERIC_CONVERSION_JOBS.md).
The Linux image must already contain the pinned xtask runner, compatible supplied
quantizer/runtime, repository helper, receipt publisher and, for combined work,
package writer. This adapter does not construct an image, bootstrap a generic
converter or restore unsupported native quantizer functionality.

The outer delivery object has `schema_version:1`, `namespace`, `worker_input`,
`mounts`, `cpu_plan` and optional `mounted_request`. Every mount contains only
`repo`, immutable `revision` and absolute `mount_path`, and is read-only.
Supply a complete ordered immutable source roster rather than trimming pins.

`worker_input` contains exactly:

- `schema_version:1`, `workflow`, `timeout_secs` and `runner:{path,sha256}`.
- `authority` with `schema_version:1`, digest-pinned `image`, 40-hex
  `mesh_commit` and `git_tree`, `cpu_plan_receipt_sha256`,
  `declared_estimate_usd`, `max_cost_usd`, `tool_kind` and `profile_version`.
- `operator` and `receipt_export` as described below.

Use the offline facade `plan` operation with `workflow:"quantization"` or
`workflow:"quantization-and-package"`, an explicit approved hardware snapshot,
complete source size, requested timeout and maximum cost. The combined workflow
alone permits a 96h declaration. Copy `cpu_plan_receipt_sha256`, `declared_estimate_usd` and `max_cost_usd`
from `bootstrap-resources.json` into `authority`. The digest is SHA-256 of the
compact typed `CpuJobPlan` serialization, and the estimate is the plan's computed
maximum cost; `max_cost_usd` is the supplied request's cost ceiling. Do not hash
the pretty CPU plan file. `cpu_plan.timeout_seconds`, `worker_input.timeout_secs` and
`operator.timeout_seconds` must match. Costs and image availability remain
supplied declarations, not current-price or cloud observations.

The closed operator contains `schema_version:1`, `workflow`, `timeout_seconds`,
`window_template`, `resumes`, `loader:{path,sha256}` and `package`. Quant-only
requires `package:null`. Combined work requires a package object with
`writer:{path,sha256}`, `writer_source:{path,sha256}`,
`generation_defaults:{path,sha256}`, a distinct `target_repo` and
`max_artifact_bytes`. Review generation defaults through the existing
`hf-layer-package-jobs` skill before publication.

The window template contains exactly `schema_version:1`,
`tool_kind:"supplied-window-quantizer"`, `profile_version`, and `{path,sha256}`
pins for `tool`, `tool_source`, `runtime`, `manifest`, `recipe`, `helper` and
`helper_source`. It also contains `source_repo`, immutable `source_revision`,
`source_root`, `source_prefix`, complete ordered `source_parts`, `target_repo`,
`target_root`, `target_prefix`, `basename`, `quant`, `expected_splits`,
`ordinal:1`, `work_root`, `credential_file:null`, `publication_confirmed:true`,
`timeout_seconds` and `resume:null`. Each source part is `{path,sha256}` and is
named `<basename>-<ordinal:05>-of-<expected_splits:05>.gguf` under the source
prefix. The manifest pins schema 1 `QUANTIZE_GGUF`, the same source/target roots,
prefixes, basename, quant, recipe path and split count, with `window_size:1`.

Source and resume artifacts must be within matching immutable model mounts.
Image/helper pins must be outside mutable `/work`; the runner is image-owned,
outside `/models`. Work and target directories are fresh and separate from
source, helper and evidence paths. The per-window allowance is 5 to 86400
seconds. It is capped by the inherited whole deadline; windows do not renew it.

Each resume row is `{ordinal,resume:{commit,record:{path,sha256},shard:{path,sha256}}}`.
The record and shard must be verified at that same immutable commit, with the
record's source/recipe/tool/runtime context and ordinal matching the request.
Filename or size presence alone cannot skip quantization.

`receipt_export` uses the existing `{helper,helper_source}` pins, `repo`,
immutable `parent_commit`, unique `path_in_repo` ending `/native-job.json`,
`credential_environment:true`, `credential_file:null` and
`export_budget_secs` from 10 to 3600. Reserve export time within the whole budget.
The facade reads an explicit private credential file only for authorized submit
or collect; the Job uses its private `MESH_HF_PUBLICATION_TOKEN` secret and an
owned temporary credential file. Do not put a token in request JSON or logs.

## Prepare, submit and collect

Preparation makes no remote call:

```sh
target/release/model-package-generic-jobs prepare \
  --input /absolute/delivery.json --output-directory /absolute/fresh-prepared
```

Only explicitly authorized cloud spending and publication permit submission:

```sh
target/release/model-package-generic-jobs submit --confirm-submission \
  --input /absolute/delivery.json --credential-file /absolute/private-token \
  --output-directory /absolute/fresh-submitted --timeout-seconds 300
```

Archive the exact delivery request and acknowledgment together. Collect against
that unchanged request; for combined work use `--timeout-seconds 345600`:

```sh
target/release/model-package-generic-jobs collect \
  --input /absolute/delivery.json \
  --submitted-file /absolute/fresh-submitted/submitted.json \
  --credential-file /absolute/private-token \
  --output-directory /absolute/fresh-collected --timeout-seconds 259200
```

The inline worker limit is 64 KiB. Larger complete rosters use the existing
`export-request` transport, up to 8 MiB, with exact bytes mounted at their actual
immutable request-repository commit. Request publication is a separately
authorized operation. Use the same full request and acknowledgment for collection.

The worker executes one split at a time using `run-quant --backend llama-api`,
`--max-memory 32G --memory-policy hard --max-windows 1`, paired
`--first-split/--last-split`, `--native-runtime-library <runtime.path>` and
`--keep-staged-source`. Each output is uploaded
and verified before local output unlink; its context record enables immutable
resume. Only verified owned staged files are cleaned. Source mounts are preserved.
After all windows, the coordinator publishes plan/recipe/lineage records and
verifies the full quant roster at one immutable commit. The supplied tool then runs `verify-job --manifest
<immutable-verification-manifest.json> --llama-load --llama-cli <loader.path>
--check-tensors --json`. That phase uses the pinned loader executable; it does
not pass `--native-runtime-library`. Verification must succeed before optional
package writing, package verification and immutable package publication.

The existing receipt exporter binds actual native receipt bytes to the request,
transport and immutable evidence commit. Collection requires completed Jobs
monitoring and the correlated locator, complete quant verification and, for
combined work, package completion. `SUBMITTED` is acknowledgment only. Inspect
`result.json` for `quantization_admitted` and `package_admitted`; neither flag
certifies the supplied tool, image, model, measured memory or actual cost.

Failed windows, native verification, export, signals and deadline boundaries
retain observations and refuse delivery completion. Local monitoring interruption
sends no automatic remote cancel. Inspect authorized Jobs state before retrying
an uncertain submission. Existing `mesh-llm models package --status`, `--logs`
and `--cancel` surfaces remain available; cancellation acknowledgment alone is
not remote terminal proof.

For a separately authorized local single-window publication, the same owner is
`cargo xtool automation hf-certify quantization-window --input /absolute/window.json
--output-directory /absolute/fresh-evidence`. It requires the closed window
request above with an explicit private `credential_file`, a selected ordinal and
optional verified resume. Its observations do not establish whole-job completion.
There is no separate standalone whole-coordinator CLI verb.
