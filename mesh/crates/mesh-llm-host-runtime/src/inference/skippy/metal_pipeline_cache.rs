use std::fs;

/// Environment variable read by the patched ggml Metal backend to locate the
/// on-disk `MTLBinaryArchive` cache of compiled compute pipeline states.
const GGML_METAL_PIPELINE_CACHE_DIR: &str = "GGML_METAL_PIPELINE_CACHE_DIR";

/// Point the native Metal backend at a process-wide on-disk pipeline cache,
/// `~/.cache/mesh-llm/metal/shared`.
///
/// The patched ggml Metal backend reads `GGML_METAL_PIPELINE_CACHE_DIR` when
/// the native library initializes, which happens at runtime-library load —
/// before any model id is known and before the first model open initializes
/// the Metal backend. The archive file itself is fingerprint-keyed (device,
/// ggml version, kernel sources), so a process-wide scope is safe. An
/// explicit value set by the user is always left untouched.
///
/// # Safety
///
/// The caller must ensure no other thread can read or write the process
/// environment. Call this exactly once from synchronous process bootstrap,
/// before the Tokio runtime is constructed and before any native runtime
/// library is loaded.
pub(crate) unsafe fn configure_metal_pipeline_cache() {
    if std::env::var_os(GGML_METAL_PIPELINE_CACHE_DIR).is_some() {
        return;
    }

    let dir = crate::models::mesh_llm_cache_dir()
        .join("metal")
        .join("shared");
    if let Err(err) = fs::create_dir_all(&dir) {
        tracing::warn!(
            target: "mesh_llm::inference::skippy::metal_pipeline_cache",
            "cannot create Metal pipeline cache dir {}: {err}",
            dir.display()
        );
        return;
    }

    // SAFETY: UNSAFE CONTRACT — callers must invoke this from single-threaded
    // synchronous bootstrap before the Tokio runtime is constructed. At that
    // point no concurrent runtime work can access the process environment,
    // and the native runtime libraries have not yet been loaded (they read the
    // variable at Metal backend initialization). The
    // shipped binary enforces this by calling from synchronous `main()`
    // bootstrap.
    unsafe { std::env::set_var(GGML_METAL_PIPELINE_CACHE_DIR, &dir) };
}
