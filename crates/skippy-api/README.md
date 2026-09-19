# skippy-api

Shared Skippy model preparation. `SingleStageOptions` and a resolved `StageSourceIdentity` produce the same single-stage configuration for standalone and embedded callers. Model-family cache policy and checkpoint quantization/importance-matrix preparation live here. Product hooks, diagnostics and model discovery remain caller concerns.

`package::identity_from_package_v2` verifies a local package and returns its model identity. It checks the package generation, content-derived package ID, native ABI, confined artifact paths, sizes and digests. Callers pass an optional `SidecarDigestCache` opened at an explicit directory; `None` disables the advisory digest cache. The API does not read environment variables or choose a home/cache directory. Existing v2 cache records keep their format and keys.

`source` prepares local GGUF/sharded GGUF, safetensors checkpoints and immutable Hugging Face source identities. Strict local GGUF verification always hashes source bytes and preserves the strong-fingerprint checks; the source registry is a locator, never proof of content. Hugging Face snapshot roots and the metadata client factory are caller-supplied. Existing identity hash domains remain unchanged. Managed multipart views now live under `.skippy/multipart-gguf` within the HF repository cache; these are hard links to verified blobs, and existing `.mesh-llm` views are not deleted.

`source::planning` constructs the source-complete graph-planning manifest from verified GGUF identity. Mesh retains its profile-scoped source policy, remote package acquisition and serving hooks. Product-specific network acquisition remains a caller concern.

`stage_admission` owns native graph discovery, independent realization and fail-closed admission. The native input/descriptor reader remains together in its `native` module. `split_certification` owns the bundled architecture roster and checks the exact upstream pin, patch-queue digest and ABI. Standalone and Mesh use the same release certification; experimental admission remains an explicit boolean chosen by the caller. Recipe hashing retains its v1 domain so moving ownership does not change certification identity.

`stage_load` builds admitted stage configs from neutral options, verified resident tensor names and activation frontiers. Mesh maps its control requests and peer endpoints into these types; no coordinator identity, lease or iroh type crosses this boundary. `materialization` resolves local package-v2 closures, verifies required artifacts and preserves metadata-first ordering. Network acquisition and progress reporting remain caller concerns.

The opt-in `direct_graph_admission` test accepts `SKIPPY_TEST_GGUF_PATH` and exercises real native two-stage planning, certification, tensor binding and activation frontiers. Run the complete API test suite with `--include-ignored` only when a prepared static native runtime and the pinned model fixture are available. This planning gate is separate from end-to-end split serving.

`serving::ModelLoadRequest` owns native model loading, graph-bound activation
widths, prediction-return listeners, tokenizer-bound hook construction and OpenAI
backend composition. `serving::OpenAiOptions` is shared resolved configuration;
Mesh translates its product configuration and supplies observers/plugin hooks.
Guardrail/compaction wrapping uses the serving library implementation, including
an optional caller-owned telemetry sink. `ModelOpenEvents` preserves the native
load-with-events choice independently of whether an observer is supplied.

`native_runtime` selects and loads an explicitly supplied runtime bundle/cache.
It performs no product discovery and does not download implicitly. The dependency
direction is API → serving; the serving library does not depend on this API.

`serving::LocalOpenAiOptions` is the standalone HTTP entrypoint. Its conversion
preserves explicit queue/adaptive-concurrency/admission settings and loads through
`ModelLoadRequest`. `LoadedModelBackend::serve_http_with_shutdown` retains native
resources while HTTP requests drain and propagates listener/serving failures.
The serving library selects local execution for a stage-zero configuration with
no downstream peer; split configurations retain embedded stage-zero execution.
