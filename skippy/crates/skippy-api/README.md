# skippy-api

Shared Skippy model preparation. `SingleStageOptions` and a resolved `StageSourceIdentity` produce the same single-stage configuration for standalone and embedded callers. Model-family cache policy and checkpoint quantization/importance-matrix preparation live here. Product hooks, diagnostics and model discovery remain caller concerns.

`package::identity_from_package_v2` verifies a local package and returns its model identity. It checks the package generation, content-derived package ID, confined artifact paths, sizes and digests. The producer native ABI is provenance; runtime loading checks host/library ABI compatibility separately. Callers pass an optional `SidecarDigestCache` opened at an explicit directory; `None` disables the advisory digest cache. The API does not read environment variables or choose a home/cache directory. Existing v2 cache records keep their format and keys.

`package::metadata` verifies metadata-only package identity without fetching layer artifacts. It preserves source provenance separately from the generated metadata digest and pins resolved Hugging Face snapshot references. Callers supply the local directory and optional digest cache; remote acquisition is available through `package::acquisition`.

`source` prepares local GGUF/sharded GGUF, safetensors checkpoints and immutable Hugging Face source identities. Strict local GGUF verification always hashes source bytes and preserves the strong-fingerprint checks; the source registry is a locator, never proof of content. Hugging Face snapshot roots and the metadata client factory are caller-supplied. Existing identity hash domains remain unchanged. Managed multipart views now live under `.skippy/multipart-gguf` within the HF repository cache; these are hard links to verified blobs, and existing `.mesh-llm` views are not deleted.

`source::planning` constructs the source-complete graph-planning manifest from verified GGUF identity. Mesh retains its profile-scoped source policy and serving hooks; callers supply transport and cache policy to package acquisition.

`stage_admission` owns native graph discovery, independent realization and fail-closed admission. The native input/descriptor reader remains together in its `native` module. `split_certification` owns the bundled architecture roster and checks the exact upstream pin, patch-queue digest and ABI. Standalone and Mesh use the same release certification; experimental admission remains an explicit boolean chosen by the caller. Recipe hashing retains its v1 domain so moving ownership does not change certification identity.

`stage_load` builds admitted stage configs from neutral options, verified resident tensor names and activation frontiers. Mesh maps its control requests and peer endpoints into these types; no coordinator identity, lease or iroh type crosses this boundary. `materialization` resolves local package-v2 closures, verifies required artifacts and preserves metadata-first ordering. `package::acquisition::remote::PackageAcquisition` owns network acquisition; callers supply progress observers.

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

`package::inspection` owns local package inspection results and per-layer tensor/artifact accounting, including metadata-only v2 inspection and legacy offline inspection. The caller supplies a resolved directory, original reference and optional digest cache.

`serving::readiness` exposes Skippy binary-stage bind-address selection, wire readiness probes and size-scaled load deadlines. Probe cancellation joins its worker; Mesh retains coordinator claims, stage registration and shutdown orchestration.

`package::acquisition` owns package reference parsing, local artifact selection/verification and cached-snapshot selection. Callers provide an optional integrity-cache path; metadata-only verification always hashes metadata without that cache. `remote::PackageAcquisition` owns remote transfers, the process-wide download lock and exact stage artifact acquisition. Callers supply explicit hub/integrity cache paths and a lazy HF client factory. Complete cache hits never construct a client. The typed batch/file observer contract preserves progress ordering and resource cleanup; its default is silent. Blocking SDK calls run outside an active Tokio runtime through `skippy-model-hf::blocking`.

`materialized_cache` owns pin-aware stage-artifact pruning, source-index previews and source-based removal. Every operation takes an explicit cache root. Active pins preserve their artifact and index; removal errors propagate before an index is discarded. Mesh supplies its existing cache directory.

`package::certification` owns two-stage package materialization checks and OpenAI model/chat/Responses smoke gates. The caller supplies the resolved package reference, acquisition policy and optional digest cache. Mesh retains catalog-name lookup. Missing runtime endpoints remain incomplete; package-only runs explicitly mark runtime gates not required.

Shared serving defaults are defined by `SingleStageOptions` and `OpenAiOptions`,
using [`skippy-config`](../skippy-config/README.md). [`kv_cache`](src/kv_cache.rs)
owns publisher-declared live-KV dtype selection and model compatibility fallback;
[`speculative`](src/speculative.rs) owns installed sibling draft discovery and
conservative automatic pairing. Standalone Skippy and Mesh consume these same
policies. Explicit caller settings take precedence.
