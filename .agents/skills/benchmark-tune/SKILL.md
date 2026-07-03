---
name: benchmark-tune
description: Use this skill when running, debugging, interpreting, or documenting mesh-llm benchmark tune model-serving throughput trials, including choosing ctx/batch/ubatch/mmap/mlock/speculative-decoding sweeps, running benchmark tune on local or SSH hosts, collecting JSON evidence, and applying tolerance-aware recommendations. Trigger for requests mentioning benchmark tune, tuning tok/s, ctx_size tradeoffs, mmap or mlock tuning, speculative decoding, MTP, ngram, draft models, or replacing old gpu tune usage.
---

# Benchmark Tune

Use `mesh-llm benchmark tune` for model-serving throughput tuning. Do not use
`mesh-llm gpu tune` or `mesh-llm gpus tune`; the GPU namespace is for hardware
inventory and raw fingerprinting (`mesh-llm gpus`, `mesh-llm gpus detect`, and
hidden `gpus run-benchmark`).

## Preflight

Verify the command surface from the current checkout before long runs:

```bash
target/release/mesh-llm benchmark --help
target/release/mesh-llm benchmark tune --help
target/release/mesh-llm gpus --help
```

For performance work, use a release build on the target host:

```bash
just release-build
```

On NVIDIA remote hosts, verify that the release binary is actually using CUDA
before recording performance results. For Jetson/Orin-style aarch64 CUDA hosts,
prefer the repo's CUDA backend build path for the host, for example
`scripts/build-linux.sh --backend cuda --cuda-arch 87`, with the host CUDA
toolkit paths exported as needed. A generic release build that reports CPU
devices is not valid performance evidence for GPU tune work.

If the run is on a remote node over SSH and will take time, use the
`remote-observable-process` skill. Prefer a TTY/login shell and `tee` logs over
detached first attempts.

## Targets

Benchmark tune accepts already-downloaded local/configured model targets only.
It will not fetch remote-only refs. If no explicit target is passed, it uses
configured local models from `~/.mesh-llm/config.toml`.

Use one of:

```bash
mesh-llm benchmark tune --model /models/model.gguf
mesh-llm benchmark tune --models /models/a.gguf,/models/b.gguf
mesh-llm benchmark tune
```

## Candidate Sweep

Start with a bounded sweep, then expand around promising values:

```bash
mesh-llm benchmark tune \
  --model /models/model.gguf \
  --ctx-sizes 8192,32768,131072,262144 \
  --batch-sizes 512,1024,2048 \
  --ubatch-sizes 256,512,1024 \
  --mmap-values auto,true,false \
  --mlock-values false,true \
  --speculative-types auto \
  --throughput-tolerance-pct 10 \
  --max-tokens 128 \
  --debug-telemetry \
  --json
```

Rules:

- `ubatch` must be less than or equal to `batch`; invalid pairs are skipped.
- `mmap` and `mlock` are separate controls. Sweep them independently when
  diagnosing load/runtime behavior.
- If `--mmap-values` is omitted, tune tries `auto`, `true`, and `false`.
- If `--mlock-values` is omitted, tune tries `false` and only tries `true` when
  the current mlock probe says the evaluated budget can be locked.
- If `--speculative-types` is omitted, tune uses `auto`: it tries
  `mtp` first when the model target looks like an MTP model, tries
  discovered local draft-model candidates when available, tries ngram
  candidates as a model-free fallback, then includes a disabled baseline.
- Use `--no-speculative-tune` when you need to reproduce the older
  fit-only/disabled-speculation behavior or isolate non-speculative regressions.
- Use `--speculative-types mtp,draft,ngram,disabled` to force an
  explicit speculative sweep. `draft` requires either `--spec-draft-models`, a
  configured `draft_model_path`, or a local sibling GGUF whose filename looks
  like a draft/EAGLE model for the target.
- MTP and draft sweeps use `--spec-draft-max-tokens` and
  `--spec-draft-min-tokens`. Ngram sweeps use `--spec-ngram-min` and
  `--spec-ngram-max`.
- Use longer `--max-tokens` when decode throughput is noisy; use shorter values
  only for smoke checks.
- Keep `--throughput-tolerance-pct` near the default `10` unless the user asks
  for stricter raw throughput optimization.
- Add `--debug-telemetry` when you need proof that speculative decoding is
  actually active. It runs trial children with Skippy debug telemetry mirrored
  into `target/gpu-tune/.../serve.log`.

## Evidence

Capture machine-readable output and trial logs:

```bash
mkdir -p target/benchmark-tune
mesh-llm benchmark tune ... --json \
  | tee target/benchmark-tune/$(hostname)-$(date +%Y%m%d-%H%M%S).json
```

For remote hosts, include host, branch, commit, binary path, command, and output
path in the final report. Benchmark tune keeps per-trial logs under
`target/gpu-tune/`; inspect those logs when a trial fails or startup readiness
is slow.

Useful JSON fields:

- `benchmarks[].best`: tolerance-aware recommendation.
- `benchmarks[].raw_best`: highest observed decode tok/s.
- `benchmarks[].pareto_frontier`: tradeoff set for decode tok/s vs `ctx_size`.
- `benchmarks[].trials[].decode_tok_s`: measured decode throughput.
- `benchmarks[].trials[].candidate.speculative`: speculative mode and settings
  used for that isolated trial.
- `benchmarks[].trials[].timings`: lifecycle timing stats: `setup_ms`,
  `readiness_ms`, `request_ms`, `shutdown_ms`, `total_ms`, and
  `readiness_attempts`.
- `benchmarks[].trials[].error` and `log_path`: first stop for failures.

## Interpretation

Report both raw best and recommended settings. The recommendation is
tolerance-aware: candidates within `--throughput-tolerance-pct` of raw best are
treated as throughput-equivalent, then larger `ctx_size` is preferred.

Call out tradeoffs explicitly:

- If raw best and recommended differ, explain the tok/s delta and context gain.
- If `mmap` or `mlock` changes the winner, report those controls separately.
- If speculative decoding changes the winner, report both tok/s and the active
  speculative candidate. For MTP, inspect trial logs/telemetry for
  `llama_stage.native_mtp.enabled`, drafted/accepted/rejected counts, and
  accept rate before concluding it is helping. Use `--debug-telemetry` if those
  attributes are not present in the trial log.
- If all trials fail, summarize the shared failure reason and link the trial log
  paths rather than claiming no viable configuration exists.
- If results are close, avoid overfitting decimals; prefer the setting with the
  better context or operational posture.
