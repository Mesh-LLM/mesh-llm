---
name: hf-gguf-quant-jobs
description: Use when creating, monitoring, validating, or documenting low-memory Hugging Face Jobs or local runs that quantize split BF16/FP16 GGUF model repos into custom quant GGUF repos with skippy-quantize.
---

# HF GGUF Quant Jobs

New repository-side job planning and orchestration follows
`../manage-ci/SKILL.md`: no new Python tooling; use typed `tools/xtask`
commands behind thin Just recipes. From the repository root,
`cargo xtool repo-consistency ci-crate-lists` is a working alias example, not
a job planner. Native quant Jobs use the existing
`model-package-generic-jobs` facade and
`automation hf-certify quant-job-worker`. See
[the operator contract](../../../docs/skippy/HF_QUANTIZATION_JOBS.md).

Use this skill to turn an existing split BF16/FP16 GGUF model repo into a
quantized GGUF model repo without requiring the host to hold the full model in
memory or on local disk at once. The operational tool is `skippy-quantize`; do
not use `llama-quantize`, `llama-quantise`, or wrapper scripts that shell out to
those binaries.

The intended pattern is: mount or point at the source BF16/FP16 GGUF repo,
quantize resumable split windows with `skippy-quantize`, publish completed
output shards to the target model repo, verify immutable remote bytes before
deleting staged files, and
resume from the first missing target shard after cancellation or failure.

## Current capability and Jobs flow

The current source-built `skippy-quantize` llama-api/skippy-abi quant backend
rejects `--max-memory` and partial split windows. It requires the complete
split range. The native-rust backend supports conversion, not quantization.
Manifest creation, status and next-window planning do not establish that the
requested quantization window can execute. Do not remove the memory bound or
quantize the whole model as a substitute for the intended low-residency flow.

The native Jobs coordinator supervises a supplied `skippy-quantize` whose
executable, source revision and native runtime are pinned and whose actual
preflight and finite window run
prove the required recipe, memory and split behavior. An unspecified image or
external job helper is not that proof. No new quantizer feature or native ABI
change is implied by this skill.

## Preconditions

- Use a split BF16/FP16 GGUF repo as the source when possible. Do not re-read
  SafeTensors for requants if a BF16 GGUF artifact already exists.
- Verify the source repo is complete before spending on quantization. Count all
  expected split shards and refuse to run if any are missing.
- Use a tensor-type file for any custom recipe. Treat MTP tensors, output
  tensors, precision-sensitive tensors, and latency-sensitive layer ranges as
  explicit recipe inputs.
- Run jobs under the intended HF org. Native delivery reads an explicit
  private credential file at the facade and supplies
  `MESH_HF_PUBLICATION_TOKEN` as a Job secret. Separately authorized HF CLI
  operations may use `HF_TOKEN`; never print either token.
- Prefer mounted Hub repos over full `hf download` when the job only needs to
  stream or stage one shard/window at a time.
- Use `just skippy-quantize-standalone-release-build` for a current-source
  standalone build, with the limitations above. A supplied quantizer needs its
  own exact executable/source/runtime pins and observed capability evidence.

## Workflow

1. Identify the source BF16/FP16 GGUF repo, target quant repo, output prefix,
   output basename, source prefix, quant type, tensor-type file, memory budget,
   and split window size.
2. Validate the complete immutable source roster and tensor recipe. Native
   `validate-splits` checks local shard presence; `status` and `next-window`
   describe an existing manifest. Separately prove that the selected supplied
   quantizer accepts and executes the required window and memory policy.
3. Prepare the closed operator request with source repo/revision, target repo,
   quant type, shard count, output prefix, tensor policy and resume settings.
   The coordinator publishes `quantization-roster.json`,
   `quantization-manifest.json` and `tensor-policy.recipe`. An experiment may
   retain an optional `quant-plan.json` operator record separately.
4. Keep window size 1 as the intended first-run profile, with a 32G memory
   budget and a three-day Job allowance. Use the native delivery owner only
   with an independently qualified supplied quantizer. Larger windows require
   finite hardware evidence.
5. For each admitted window, stage the required input, execute the pinned
   quantizer, publish finished shards with an observed immutable commit and
   exact-byte verification, then delete local staged input and output files.
6. Monitor for progress markers. A healthy job repeatedly emits staged source
   copies, `quant_window`, publish completion, cleanup, and increasing split
   progress.
7. Validate the target repo after completion by counting GGUF shards, checking
   the first and last shard names, and confirming `quantization-roster.json`,
   `quantization-manifest.json` and `tensor-policy.recipe` at the verified
   immutable commit.
8. Record the artifact in the experiment card and create an iteration card for
   the run, including job id, command, environment, repo SHA, shard count, and
   follow-up decisions.

## Preparation and launch

Local source and recipe checks remain available with an appropriately built
native binary:

```bash
target/release/skippy-quantize backends --json
target/release/skippy-quantize validate-splits \
  --root /mnt/source-gguf --prefix <source-prefix> --json
target/release/skippy-quantize validate-tensor-types /mnt/recipe/tensor-types.txt --json
```

These checks do not submit a Job or prove a bounded quant window. The native
launch owner consumes a closed request containing immutable source and
recipe pins, target repo/prefix/basename, supported supplied tool identity,
window/memory profile, namespace, image and CPU cost plan, three-day allowance,
explicit publication authorization and durable receipt export. Source mounts
must be read-only. A target directory or model mount is not proof of Hub
publication; use verified upload receipts before cleanup or remote resume.
Use the prepare/submit/collect grammar in
[HF quantization Jobs](../../../docs/skippy/HF_QUANTIZATION_JOBS.md).
Preparation makes no remote request; submission requires explicit authorization
and `--confirm-submission`. These receipts always leave
`tool_profile_qualified:false` and `workflow_qualified:false`. Separate real
model, memory and hosted qualification evidence does not change those flags.

## Monitoring

Check status and logs:

```bash
hf jobs inspect <job-id> --namespace meshllm
hf jobs logs <job-id> --namespace meshllm --tail 120
```

For agents, prefer polling `/tmp/skippy-quantize-status.json` over ingesting
full logs when the admitted tool emits that file. It is a compact phase/window
snapshot, not immutable publication or terminal-success evidence.

Expected progress markers from a supported supplied tool:

- `Preflight QuantizeGguf with backend skippy-abi`
- `Source artifact is complete`
- `quant_window`
- `Published /mnt/target-quant/...`
- `Cleaned staged source`
- `split artifact ... 100.00%`

Concerning markers:

- repeated watchdog lines with no shard, tensor, upload, or cache-drop progress;
- cgroup memory pinned near the hardware limit;
- the same split window restarting repeatedly without new uploaded target files;
- fallback quant warnings for tensors that the recipe expected to preserve.

If a job stalls, cancel it before changing code or hardware and confirm the
remote terminal state. Acknowledged cancellation alone is not terminal proof.
Resume only from an immutable, exact-byte-verified target roster correlated
with the source and recipe; filename presence or equal file size is insufficient.

## Validation

After completion, verify the target repo with an authenticated Hub API or CLI
check. Record at least:

- target repo and commit SHA;
- privacy setting;
- total file count;
- GGUF shard count;
- first and last shard names;
- `quantization-roster.json` and `quantization-manifest.json` presence;
- `tensor-policy.recipe` presence.

For local smoke tests, use a small split GGUF source first and verify:

- `skippy-quantize verify-job --manifest <manifest> --llama-load` succeeds;
- `skippy-quantize validate-splits --root <target> --prefix <prefix>` succeeds;
- max RSS stays bounded compared with full-model size;
- `skippy-quantize status --manifest <manifest> --json` reports completion.

## Documentation Contract

For Jianyang-style experiments, update both records:

- the main experiment card with the promoted artifact;
- a phase iteration card with the job id, exact command, environment,
  verification output, decision, and follow-ups.

Keep post-experiment upstream notes separate from the run decision. The job can
be successful while the converter or quantizer patches still need extraction
into clean upstream PRs.
