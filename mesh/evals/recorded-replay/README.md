# Skippy vs llama agentic comparison

This manual benchmark compares a verified Mesh release build with upstream llama.cpp
at the pinned source in `job.json`, using Qwen3.5-4B Q4_K_M. It replays two consecutive
50-turn windows from the pinned Thoughtworks agentic coding dataset, at concurrency 1.
Recorded histories remain intact; generated answers are not fed into later requests.
Prompts are not rewritten. Natural `stop` responses pass regardless of prose,
XML, command formatting or usefulness. This measures engine correctness and
performance, not model quality or task completion.
This is bounded replay, not full-session recurrent-state certification or SkippyBench.

See the [native differential findings](NATIVE_DIFFERENTIAL.md) for the focused
Qwen3.5-4B comparison of prefill batching, sampling, and prefix restoration.

Both measured trajectories and the separate four-turn warmup use the same numeric
sampling profile with temperature 0, seed 42, explicit thinking off and unrestricted
reasoning budget. Temperature 0 selects greedy decoding; it does not guarantee
identical output across engines with different numerical execution. The temperature-0
harness version is distinct from earlier temperature-1 measurements.
Requests omit output-token limits. Current Mesh defaults to the remaining context
budget; older builds may retain a fixed server cap. Any `length` finish fails the
strict run, including during warmup. The runner
retains the offending response's usage, finish, output-shape counts and integrity
problems, stops issuing requests, and shuts down its owned server. It never skips a
bad warmup to obtain a passing score. Tool checks validate JSON structure, not semantic
completion or the original tool schemas (which are permissive reconstructions).

## Local invocation

Use two independent experiments with the same job and bindings:

- `--experiment cache`: the same Mesh binary with prefix caching explicitly off
  and on. Each case requires all 50 request identities, zero reported cache tokens
  when disabled, and at least one measured cache hit when enabled. It compares
  output hashes, token counts and finish reasons. A difference fails certification
  and retains both responses in `cache-comparison.json` for a matched-batching
  native/logit investigation; a difference alone does not prove KV corruption.
- `--experiment performance` (default): cache-enabled Mesh versus cache-enabled
  llama on the same recorded windows. Output equality across engines is not an
  acceptance gate. Actual output lengths and all measured durations are retained.

Run them separately so a correctness investigation cannot become a throughput
score. These are two 50-turn windows per experiment, not one 100-turn session.
Structural integrity gates apply to both experiments and warmups: length finishes,
malformed API tool arguments, missing stream terminals and runtime errors fail.

Prepare an absolute-path bindings file, separate from the portable job definition:

```json
{
  "source_manifest": "/data/generated-manifest.json",
  "source_manifest_sha256": "<SHA-256 of that exact manifest>",
  "mesh_build": "/builds/build-after.json",
  "llama_server": "/builds/llama.cpp/build-cuda/bin/llama-server",
  "llama_source": "/builds/llama.cpp",
  "model": "/models/Qwen3.5-4B-Q4_K_M.gguf",
  "hf_home": "/data/huggingface"
}
```

`mesh_build` is the existing replay `build-arm` receipt, containing commit, executable
and runtime paths and checksums. Those bytes are reverified before serving. Build
products through the normal Just recipes first. The source manifest is the existing
trajectory-reader output, containing the two selected sessions and a disjoint warmup
cohort; its digest and dataset provenance are checked before output is created.
The model digest and upstream Git HEAD are also checked. Source HEAD alone cannot
attest an externally built binary: retain its build log; the replay records executable
version and identity. No paths or credentials from a particular lab run are committed.

```bash
just with-lld cargo build --release --locked -p xtask --bin xtask
target/release/xtask automation replay-matrix recorded-comparison \
  --job mesh/evals/recorded-replay/job.json --input /data/bindings.json \
  --experiment performance \
  --output /data/comparison
```

Add `--plan` to verify bindings and generate both run inputs without serving. The output
path must be new. `early/` and `grown/` contain the normal replay run, per-request,
warmup, server, identity, gate and report artifacts. Failure exits nonzero with partial
evidence retained; a stopped comparison is not a passing benchmark.

## Jianyang

`jianyang.yaml` contains separate performance and cache-correctness templates.
Only performance declares a benchmark harness and publishes a score.
Their build compiles only the replay automation;
input products must already exist as verified receipts. It is lab-only, pinned to the
Spark, and has a zero-dollar budget. Set the `bindings` input to the node-local file.
The selected repository commit identifies the harness; each measured engine's distinct
identity is retained in its receipt. Artifacts go beneath `JIANYANG_ARTIFACT_DIR`.

Mesh's live Jianyang catalog currently comes from the Jianyang repository. Import this
entries into that catalog; do not sync this file over the existing templates.
Committing it does not itself register or replace the live catalog. The checked-in job
is the benchmark policy; bindings are machine-specific inputs, not a replacement job.

## Metrics

- `ttft_seconds`: request start to first generated content/reasoning/tool event.
- `output_stream_seconds`: first to last generated event; token throughput over this
  interval is a **streaming proxy**, not native/kernel decode speed. Single-event spans
  have no defined rate.
- `terminal_seconds`: last generated event to DONE; reported separately, never discarded.
- `elapsed_seconds` and workload throughput retain all request time.
- Legacy `generation_seconds` and `decode_tokens_per_second` remain compatible with
  historical evidence: first generated event to DONE. Reports label them post-first-output
  throughput rather than decode speed. Do not compare that field to native decode TPS.
- Output character counts and tool-call counts distinguish reasoning-only output from
  tool/finish mismatches. This pinned public dataset explicitly enables response
  retention, including content, reasoning and API tool arguments. General replay
  inputs retain hashes only unless `retain_output: true` is selected.
- `unavailable_metrics` explains missing observations and native prefill/decode
  rates. The common OpenAI stream does not expose native timing. TTFT includes
  other work and must not be converted into a claimed native prefill rate.
