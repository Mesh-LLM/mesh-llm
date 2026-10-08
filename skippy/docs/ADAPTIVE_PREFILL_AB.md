# Adaptive-prefill A/B orchestration

The native caller is `cargo xtool automation waiting-prefix adaptive-run`.
It owns alternating old/new rounds, two-stage configurations, one excluded
calibration request per cell, serial measured OpenAI requests, exact typed prefill
telemetry, paired intervals and output parity. It writes `comparison.json` and
`report.md` under a fresh output directory. A failed cell retains earlier rows;
terminal cancellation, deadline expiry or interrupt-finalization failure retains
observations with `comparison_admitted: false` and returns nonzero.

Prepare a closed input before execution:

```bash
cargo xtool automation waiting-prefix adaptive-prepare \
  --input adaptive-input.json --output adaptive-prepared.json
cargo xtool automation waiting-prefix adaptive-run \
  --input adaptive-prepared.json --output-directory adaptive-results
```

Both outputs must be fresh and their parent directories must exist. Preparation
only validates the declared matrix and worker fields and binds the typed manifest
hash; it launches no serving process and does not verify artifact bytes. Execution
supervises byte verification before launching each arm. Run actual native/model
trials only under the separately authorized runtime qualification scope.

`adaptive-input.json` contains `schema_version: 1`, `old`, `new`, `worker`,
`rounds`, `timeout_secs`, `cell_timeout_secs`, and `startup_timeout_secs`.
Each arm contains these existing identity fields:

| Fields | Contract |
|---|---|
| `schema_version`, `round`, `version` | `1`, initial positive round, and `old` or `new`; matrix assigns actual round |
| `binary`, `binary_sha256`, `commit` | Absolute standalone static skippy-server path, lowercase SHA256, and declared 40/64-digit revision |
| `native_build`, `native_build_sha256`, `native_profile` | Absolute native build directory, existing native-identity complete-tree SHA256, and `standalone-static-skippy-server` |
| `model`, `model_sha256`, `model_id` | Same absolute GGUF, lowercase SHA256 and nonempty model identifier in both arms |
| `ctx_size`, `split_layer`, `layer_end`, `n_gpu_layers`, `adaptive_target_ms` | Equal bounded configuration in both arms; target scheduling applies only to new arm |
| `stage_ports`, `openai_port` | Two distinct stage ports and a third distinct OpenAI port; rounds run serially |

Artifact SHA pins are byte observations. `commit` is a caller declaration;
matching a binary SHA and native build tree does not independently attest that the
binary was built from that revision. Obtain native tree pins through the existing
identity owner; an archive SHA or concatenated ad-hoc file hash is a different
contract. Preparation does not manufacture these pins.

The worker contains `schema_version: 1`, `output_tokens`,
`request_timeout_secs`, `timeout_secs`, `manifest` and `provenance` (object).
Its `manifest` is `{ "metadata": { ... }, "prompts": [ ... ] }`; each prompt
requires nonempty `family` and `prompt`, with flattened additional provenance
such as `source_id`, `bucket` or `source_index`. Preparation sets the existing
round/version/model/local endpoint/readiness fields from the old arm. Each actual
cell projects its own arm's fields before execution. An absent
`prompt_manifest_sha256` is generated from the existing typed serializer;
a supplied stale hash is refused. This digest is **not** the raw JSON file hash.
Preparation preserves metadata and flattened prompt provenance.

A data-only assembly example using already prepared arm JSON and prompt JSON:

```bash
jq -n --slurpfile old old-arm.json --slurpfile new new-arm.json \
  --slurpfile manifest prompts.json '
  {schema_version:1, old:$old[0], new:$new[0], rounds:4,
   timeout_secs:32000, cell_timeout_secs:3900, startup_timeout_secs:900,
   worker:{schema_version:1, output_tokens:8, request_timeout_secs:900,
           timeout_secs:3600, manifest:$manifest[0], provenance:{}}}' \
  > adaptive-input.json
```

`rounds` is 1..128, overall timeout 13..86400 seconds, cell timeout
10..86400 seconds, and startup timeout is positive and smaller than the cell
budget. Worker bounds come from the serial request owner: 1..4096 output tokens,
1..1000 prompts, prompt bytes <=256 KiB and total prompt bytes <=8 MiB, and worker timeout <=86400 seconds.
Worker runtime must fit the retained pair's remaining execution budget after
identity/readiness and reserved cleanup; an otherwise well-typed plan can refuse
when that budget cannot fit. The whole matrix deadline remains absolute.

For synthetic prompts use the existing `synthetic-prompts` command with explicit
sizes. For research trajectories use the separately approved isolated trajectory
reader to emit the deterministic manifest; no new Python orchestration is needed.
Preparation accepts either manifest through the same typed serializer. The native
synthetic text need not reproduce the old Python string byte-for-byte, but is
stable and compared identically between arms.

Successful terminal admission requires all configured paired cells and exact
request parity. Reports preserve unavailable metrics as `n/a`; deterministic
10,000-sample paired bootstrap uses the native deterministic owner rather than
emulating Python RNG bytes. Missing telemetry, incomplete pairs, provenance
mismatch and output mismatch refuse admission. Retained metrics on failure are
observations, not accepted performance or actual-model qualification.

The original default 384-block synthetic long-context workload is admitted by preparation; per-prompt/aggregate memory limits remain explicit, and do not substitute for actual native token/context qualification.
