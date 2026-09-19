# skippy-config

Standalone Skippy settings and path policy, shared by the CLI and serving. `validate_config` checks a `StageConfig` (and optionally its `StageTopology`) against the same rules standalone serving enforces, including layer-package and artifact-slice load-mode requirements. `load_json` reads typed configuration documents and `example_config` emits the canonical single-stage example.

`paths` owns cache path policy with one resolution order: an explicit flag wins, then the `SKIPPY_MODEL_CACHE_DIR` / `SKIPPY_NATIVE_RUNTIME_CACHE_DIR` / `SKIPPY_NATIVE_RUNTIME_BUNDLE_DIR` environment overrides, then the platform cache directory. Error messages name the flag that resolves the ambiguity.

The crate sits below serving and the lifecycle API: it depends only on protocol primitives and path policy, never on `skippy-api` or `skippy-server`, and reads no configuration beyond the documented environment variables.
