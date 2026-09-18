# skippy-api

Shared Skippy model preparation. `SingleStageOptions` and a resolved `StageSourceIdentity` produce the same single-stage configuration for standalone and embedded callers. Model-family cache policy and checkpoint quantization/importance-matrix preparation live here. Product hooks, diagnostics and model discovery remain caller concerns.

`package::identity_from_package_v2` verifies a local package and returns its model identity. It checks the package generation, content-derived package ID, native ABI, confined artifact paths, sizes and digests. Callers pass an optional `SidecarDigestCache` opened at an explicit directory; `None` disables the advisory digest cache. The API does not read environment variables or choose a home/cache directory. Existing v2 cache records keep their format and keys.

This boundary does not yet own direct-GGUF/HF identity orchestration, model acquisition, lifecycle orchestration or split graph admission.
