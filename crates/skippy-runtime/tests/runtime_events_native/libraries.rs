//! Native runtime library discovery for the gated test.

use std::path::{Path, PathBuf};

/// Resolves the real installed-runtime layout: `MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR`
/// names the PARENT of one or more `<runtime-id>/{manifest.json,lib/*}`
/// subdirectories (see `dist/native-runtimes/README.md` and
/// `mesh-llm-runtime-install`'s own discovery convention), not a flat
/// directory of libraries. Prefers each candidate's own `manifest.json`
/// `runtime.libraries` ORDER — dependencies before the primary
/// `libllama.dylib` — over a lexicographic guess, since symbol-search order
/// in `skippy-ffi::dynamic::Symbols::load_paths` walks the list in reverse
/// and a naive alphabetical sort places `libllama*` before `libmtmd*`,
/// inverting the manifest's own dependency-then-primary contract.
pub fn discover_libraries(bundle_dir: &Path) -> Vec<PathBuf> {
    if let Some(libraries) = libraries_from_flat_dir(bundle_dir) {
        return libraries;
    }
    let Ok(entries) = std::fs::read_dir(bundle_dir) else {
        return Vec::new();
    };
    let mut subdirs: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    subdirs.sort();
    subdirs
        .iter()
        .find_map(|subdir| libraries_from_manifest(subdir))
        .unwrap_or_default()
}

fn libraries_from_flat_dir(dir: &Path) -> Option<Vec<PathBuf>> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut libraries: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| matches!(extension, "dylib" | "so" | "dll"))
        })
        .collect();
    if libraries.is_empty() {
        return None;
    }
    libraries.sort();
    Some(libraries)
}

fn libraries_from_manifest(runtime_dir: &Path) -> Option<Vec<PathBuf>> {
    let manifest_path = runtime_dir.join("manifest.json");
    let manifest_text = std::fs::read_to_string(&manifest_path).ok()?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).ok()?;
    let entries = manifest
        .get("runtime")?
        .get("libraries")?
        .as_array()?
        .iter()
        .filter_map(|value| value.as_str());
    let libraries: Vec<PathBuf> = entries
        .map(|relative| runtime_dir.join(relative))
        .filter(|path| path.is_file())
        .collect();
    (!libraries.is_empty()).then_some(libraries)
}
