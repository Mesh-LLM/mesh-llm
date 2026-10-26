# Local native MTP composition

Use the native local adapter for already-converted, pinned Nemotron-H MoE MTP GGUF inputs:

```sh
just automation-run automation hf-mtp-compose \
  --input /absolute/compose-input.json \
  --output-directory /absolute/existing-parent/fresh-compose-output
```

The output directory must be fresh, under an existing parent. The adapter creates it; do not pre-create the leaf. The input file contains a closed schema:

```json
{
  "schema_version": 1,
  "binary": {"path": "/absolute/skippy-quantize", "sha256": "REPLACE_WITH_BINARY_SHA256"},
  "target_parts": [
    {"path": "/absolute/Target-00001-of-00002.gguf", "sha256": "REPLACE_WITH_FIRST_SHA256"},
    {"path": "/absolute/Target-00002-of-00002.gguf", "sha256": "REPLACE_WITH_LAST_SHA256"}
  ],
  "mtp_gguf": {"path": "/absolute/mtp.gguf", "sha256": "REPLACE_WITH_MTP_SHA256"},
  "target_basename": "Target",
  "composite_basename": "Target-MTP",
  "expected_parts": 2,
  "mtp_block": 88,
  "supplied_mesh_revision": "REPLACE_WITH_40_HEX_SOURCE_REVISION",
  "native_profile": "standalone-static-skippy-quantize-cpu",
  "timeout_secs": 1800,
  "composite_repo": "organization/model-repository"
}
```

Replace every illustrative pin before use. SHA256 pins are lowercase 64-digit hex. Supply the complete ordered target shard roster, with exact `BASENAME-00001-of-NNNNN.gguf` names; two through 1,024 shards are admitted. Paths must be absolute and unique. The supplied revision is provenance declared by the operator, not independent proof that the binary was built from it. Use a matching supplied standalone static CPU native tool; this adapter does not build one, acquire a runtime, or qualify tokenizer conversion.

The adapter runs native `compose-mtp`, then `validate-mtp-attach` without a projector. It checks correlated reports, every input byte pin, first/last output pins, and unchanged middle parts before and after attachment. It keeps untouched middle shards at their original local paths; the publication plan names their destination filenames. Preserve those source files alongside output evidence.

A successful run writes `report.json` with status `PASS` and an immutable `publication-plan.json` with status `PLAN_ONLY_NOT_PUBLISHED`. Require the plan's matching successful receipt/request hash and re-observe every planned local byte pin before using the plan. `PASS` means the declared local composition/attach contract succeeded. It grants no remote publication authority, backend/build certification or numerical inference claim. The command never uploads, consumes HF credentials, submits a Job, or turns local paths into remote volumes.

Failed or interrupted runs retain partial process/log/identity evidence and a nonpassing report; they must not yield an eligible publication plan. Do not promote scratch output from a failed run. The nominal execution timeout has bounded owned cleanup phases in addition; stalled ordinary filesystem operations or hostile concurrent path replacement are not claimed to be sandboxed by this adapter.

For immutable checkpoint/tokenizer staging, observed source checkout and native
build, conversion/attachment, and confirmed ordered publication, use the
separate default workflow:

```sh
just hf-skippy-mtp-compose-default \
  /absolute/default-mtp-input.json \
  /absolute/existing-parent/fresh-default-mtp-evidence
```

Its closed input is owned by
`tools/xtask/src/automation/hf_certify/mtp_default/contract.rs::Input`; it binds
bootstrap/source identities, immutable checkpoint and tokenizer rosters, an
explicit source-bound tokenizer profile, ordered target parts and helper
identities. Use `dry_run: true` to stage and inspect the plan; publication
requires `confirm_publication: true` and explicit credential custody. A full
publish verifies the complete ordered GGUF/sidecar roster at one parent-bound
immutable commit. The inherited deadline and cancellation cover the workflow.

See [default MTP Jobs](HF_DEFAULT_MTP_COMPOSITION_JOBS.md) and
[common Jobs delivery](HF_GENERIC_CONVERSION_JOBS.md) for native
prepare/confirmed-submit/collect. Local fixture proof does not establish real
family tokenizer/conversion, native build, image delivery or hosted publication
acceptance. Keep the conditional legacy caller/ledger deletion held until its
complete intent map, current owning positives and final normal are admitted.
