//! Host-side glue for the moved `mesh_llm_membership::weights_digest` module:
//! resolves the Skippy-owned cache directory (the single dependency that kept
//! this module from moving wholesale) and forwards to the membership
//! implementation. `file_fingerprint` is pure and re-exported directly.

use std::path::Path;

/// (size, mtime) fingerprint of `path` -- re-exported from the moved module.
pub(crate) use mesh_llm_membership::file_fingerprint;

/// SHA-256 of a served GGUF's bytes, cached in memory and persisted under
/// `mesh_llm_cache_dir()/weights-digest/`. The cache directory resolves to
/// `skippy-model-hf` on the host side, so it is injected rather than resolved
/// inside the Skippy-free `mesh-llm-membership` crate.
pub(crate) fn weights_digest_for_file(path: &Path) -> Option<String> {
    let cache_dir = crate::models::mesh_llm_cache_dir().join("weights-digest");
    mesh_llm_membership::weights_digest_for_file_in(path, &cache_dir)
}
