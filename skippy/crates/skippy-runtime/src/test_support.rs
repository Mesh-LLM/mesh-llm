//! Native runtime for tests that call into it.
//!
//! Statically linked builds always carry the runtime, so this is inert there.
//! Dynamic builds (Windows, where the runtime is built with MinGW and cannot be
//! linked into an MSVC test binary) load the runtime that
//! `MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR` names, using the same verified load
//! plan and manifest-declared CPU backend modules as product startup.

#[cfg(feature = "dynamic-native-runtime")]
mod libraries;

/// Make sure a native runtime is loaded before a test calls into it.
///
/// Without a runtime, a dynamic build fails here with a message naming the
/// variable to set, rather than later in the FFI loader, so a missing runtime
/// still fails the test instead of skipping it.
pub fn require_native_runtime() {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        static LOADED: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
        if let Err(message) = LOADED.get_or_init(load_bundle_from_env) {
            panic!("{message}");
        }
    }
}

#[cfg(feature = "dynamic-native-runtime")]
fn load_bundle_from_env() -> Result<(), String> {
    const BUNDLE_DIR_ENV: &str = "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR";
    if crate::native_runtime_loaded() {
        return Ok(());
    }
    let bundle_dir = std::env::var_os(BUNDLE_DIR_ENV).ok_or_else(|| {
        format!("this test calls into the native runtime; set {BUNDLE_DIR_ENV} to a runtime bundle")
    })?;
    let bundle_dir = std::path::Path::new(&bundle_dir);
    let plan = libraries::discover_load_plan(bundle_dir).map_err(|error| {
        format!(
            "resolve a verified native runtime from {}: {error:#}",
            bundle_dir.display()
        )
    })?;
    // SAFETY: the load plan comes from runtime-install's verified manifest
    // selection and the loader checks the Skippy ABI before binding symbols.
    unsafe {
        crate::load_native_runtime_libraries_with_cpu_backend_modules(
            &plan.libraries,
            &plan.cpu_backend_modules,
        )
    }
    .map_err(|error| {
        format!(
            "load the native runtime from {}: {error}",
            bundle_dir.display()
        )
    })
}
