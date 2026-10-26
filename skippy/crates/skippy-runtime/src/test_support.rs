//! Native runtime for tests that call into it.
//!
//! Statically linked builds always carry the runtime, so this is inert there.
//! Dynamic builds (Windows, where the runtime is built with MinGW and cannot be
//! linked into an MSVC test binary) load the runtime that
//! `MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR` names, the variable the host and the CI
//! smokes already use, with its libraries in manifest order.

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
    let libraries = libraries::discover_libraries(bundle_dir);
    if libraries.is_empty() {
        return Err(format!(
            "no native runtime libraries found under {}",
            bundle_dir.display()
        ));
    }
    // SAFETY: the libraries come from one runtime bundle's own manifest, and the
    // loader checks the Skippy ABI the library exports before binding symbols.
    unsafe { crate::load_native_runtime_libraries(&libraries) }.map_err(|error| {
        format!(
            "load the native runtime from {}: {error}",
            bundle_dir.display()
        )
    })
}
