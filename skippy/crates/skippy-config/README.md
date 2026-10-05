# skippy-config

Standalone Skippy settings, shared by the CLI and serving. `validate_config` checks a `StageConfig` (and optionally its `StageTopology`) against the same rules standalone serving enforces, including layer-package and artifact-slice load-mode requirements. `load_json` reads typed configuration documents and `example_config` emits the canonical single-stage example.

Model downloads use the shared Hugging Face cache policy in [`skippy-model-hf`](../skippy-model-hf/README.md). Native-runtime caches and bundle discovery use [`skippy-runtime-install`](../skippy-runtime-install/README.md). There are no Skippy-only cache overrides.

The crate sits below serving and the lifecycle API: it depends only on protocol primitives and path policy, never on `skippy-api` or `skippy-serving`, and reads no configuration beyond the documented environment variables.

[`local_serving`](src/local_serving.rs) owns serving constants for both products:
context and batch fallbacks, concurrency, output budget, prefill controls,
speculative proposal windows, and downstream connection timeouts. Model-dependent
KV policy and draft discovery live in [`skippy-api`](../skippy-api/README.md);
sampling, guardrails, and compaction stay in their Skippy runtime/API owners.
Mesh translates explicit configuration into these policies rather than defining
an independent set of serving defaults.
