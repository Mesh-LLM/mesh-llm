# Skippy vs llama agentic comparison

This manual benchmark compares a verified Mesh release build with upstream llama.cpp
at the pinned source in `job.json`, using Qwen3.5-4B Q4_K_M. It replays two consecutive
50-turn windows from the pinned Thoughtworks agentic coding dataset, at concurrency 1.
Recorded histories are retained; generated answers are not fed into later requests.
Wire requests add the versioned text-action instructions described below.
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
bad warmup to obtain a passing score.

## Text-action contract

The pinned job explicitly selects `text_action_protocol: shell-fence-v1` for both
measured sessions and warmup. These recorded agents use fenced shell commands and
user observations, rather than API function calls. The original warmup's generic
system message did not explain how to submit a command or when to wait for its
result. An introduction ending in `stop` could therefore pass structural checks
without giving an agent runner any action.

The adaptation appends instructions to the system message (or inserts one when
absent): emit one nonempty shell-command fence, then end the response and wait for
an observation. It preserves all other recorded message content. It is an explicit
benchmark adaptation, not a claim to reproduce the original agent's complete
configuration. Input manifests retain the original messages; each request record
identifies the protocol version. Runs with and without this adaptation are different
workloads and must not share a benchmark identity.

The response checker rejects missing, empty, comment-only, non-shell, or unclosed
command blocks and any non-whitespace after the closing fence, including fabricated
results or further actions. Commands may use plain, `bash`, `sh`, or `shell` fences;
longer backtick fences can contain shorter fences in heredocs. Submission commands
use the same format. This checks action structure, not shell syntax, command safety,
execution success, or task completion. Commands are never executed by replay.
One-token context-qualification probes skip the action-format check; warmups do not.

The protocol is opt-in for general replay inputs. Combining it with API tool
definitions fails admission rather than silently mixing two action interfaces.
Unadapted API-tool replay retains its JSON-structure checks and permissive inferred
schemas; those do not certify original tool-schema conformance.

## Local invocation

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
  --output /data/comparison
```

Add `--plan` to verify bindings and generate both run inputs without serving. The output
path must be new. `early/` and `grown/` contain the normal replay run, per-request,
warmup, server, identity, gate and report artifacts. Failure exits nonzero with partial
evidence retained; a stopped comparison is not a passing benchmark.

## Jianyang

`jianyang.yaml` is the reusable template. Its build compiles only the replay automation;
input products must already exist as verified receipts. It is lab-only, pinned to the
Spark, and has a zero-dollar budget. Set the `bindings` input to the node-local file.
The selected repository commit identifies the harness; each measured engine's distinct
identity is retained in its receipt. Artifacts go beneath `JIANYANG_ARTIFACT_DIR`.

Mesh's live Jianyang catalog currently comes from the Jianyang repository. Import this
entry into that catalog; do not sync this single-entry file over the existing templates.
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
  tool/finish mismatches without retaining response text in metric records.
