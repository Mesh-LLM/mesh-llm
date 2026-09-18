# skippy-api

Shared Skippy model preparation. `SingleStageOptions` and a resolved `StageSourceIdentity` produce the same single-stage configuration for standalone and embedded callers. Model-family cache policy and checkpoint quantization/importance-matrix preparation live here. Product hooks, diagnostics and model discovery remain caller concerns.

`package::identity_from_package_v2` verifies a local package and returns its model identity. It checks the package generation, content-derived package ID, native ABI, confined artifact paths, sizes and digests. Callers pass an optional `SidecarDigestCache` opened at an explicit directory; `None` disables the advisory digest cache. The API does not read environment variables or choose a home/cache directory. Existing v2 cache records keep their format and keys.

`source` prepares local GGUF/sharded GGUF, safetensors checkpoints and immutable Hugging Face source identities. Strict local GGUF verification always hashes source bytes and preserves the strong-fingerprint checks; the source registry is a locator, never proof of content. Hugging Face snapshot roots and the metadata client factory are caller-supplied. Existing identity hash domains remain unchanged. Managed multipart views now live under `.skippy/multipart-gguf` within the HF repository cache; these are hard links to verified blobs, and existing `.mesh-llm` views are not deleted.

`source::planning` constructs the source-complete graph-planning manifest from verified GGUF identity. Mesh retains its profile-scoped source policy, remote package acquisition and serving hooks. Model acquisition, lifecycle orchestration and split graph admission remain outstanding.
