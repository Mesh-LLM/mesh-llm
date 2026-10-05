# Native runtime event benchmarks

The maintained commands are `just automation-run automation event-benchmark-run` and `just automation-run automation event-benchmark-compare`. Their help lists the complete supported flags. Use a composed release binary with its adjacent `native-runtimes/` package directory and an approved local GGUF. The runner does not download missing models or runtimes.

For two modes on one binary:

```sh
just automation-run automation event-benchmark-run \
  --binary /absolute/path/mesh-bundle/mesh-llm \
  --model /absolute/path/model.gguf \
  --output-dir /absolute/path/evidence/run-1 \
  --pairs-primary 20 --pairs-scenario 10 --seed 42 \
  --mode production --mode event-disabled --scenario streaming
```

Use a fresh output directory. Each pair runs both sides back to back with the same prompt and seed and deterministic randomized side order. A trial owns a fresh serving process, readiness, one excluded best-effort warmup, one measured streaming request, and cleanup. The server is local-model-only and disables speculative decoding. Metrics use the real advertised model ID; elapsed throughput and decode-only throughput remain separate. Incomplete metrics and unavailable health remain explicit.

For current versus baseline binaries, supply `--baseline-binary /absolute/path/baseline-bundle/mesh-llm` and exactly one `--mode production`; the side IDs are `current` and `baseline`. Each binary requires its own adjacent native package. With no baseline binary, supply exactly two distinct modes from `production`, `event-disabled`, and `off`. The `off` mode measures total event-system cost; `event-disabled` preserves the class-bypass comparison.

`--attempt` defaults to 1. Use `--attempt 2` only for the predefined full-set retry after an adverse first attempt, with a fresh seed and fresh output directory. The comparator permits one retry; do not selectively rerun rows or use a third attempt. Defaults are 64 measured tokens, 120 seconds host/API readiness, 120 seconds per request, 15 seconds per shutdown phase, and an 86,400-second overall budget. The overall budget includes preflight and per-cell identity validation. Admission must leave time for both readiness waits, warmup and measurement, and both members' graceful stop, forced tree wait, and separate EOF drain.

Success prints JSON `manifest_paths`, with one `manifest-<side>.json` per side. Retain both manifests and the entire directory: per-trial provenance, stdout/stderr, identity worker receipts, native package verifier diagnostics, and private runtime logs bind the result. Failure can still publish partial manifests and then return nonzero; such evidence does not become a successful comparison.

Compare the production/reference manifests and the separately recorded production baseline required by the comparison policy:

```sh
just automation-run automation event-benchmark-compare \
  --production /absolute/path/evidence/run-1/manifest-production.json \
  --event-disabled /absolute/path/evidence/run-1/manifest-event-disabled.json \
  --baseline /absolute/path/evidence/baseline/manifest-production.json \
  --output /absolute/path/evidence/report.json \
  --bootstrap-samples 10000 --seed 42 \
  --min-primary-pairs 20 --min-scenario-pairs 10 \
  --max-degradation-percent 3 --max-mdd-percent 10 --report-holm
```

The example thresholds must be replaced by the experiment's approved policy. A written report is not itself a certification pass: inspect `certification_status` and `blocking_reasons`, including paired completeness, order/prompt/model/environment identity, statistical power, latency, exact health counters, and retry exhaustion. The native resampler is the explicitly versioned SHA-256 counter sampler; it preserves paired bootstrap intent and reproducibility without promising Python RNG-identical quantiles.

Historical measurements in [Runtime event architecture repairs](RUNTIME_EVENT_ARCHITECTURE_REPAIRS.md) were produced by the Python scripts at the recorded branch, binary hash, model, seed, and host. Their original commands and numbers remain historical provenance. Native fixture tests and this usage guide do not requalify those measurements or establish new model/GPU performance, Windows ownership, or hosted certification.
