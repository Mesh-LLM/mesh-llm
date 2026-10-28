use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub use skippy_model_hf::store::installed::{InstalledModelArtifact, scan_installed_artifacts_in};
pub use skippy_model_hf::store::local::{
    HuggingFaceModelIdentity, direct_hf_cache_root_gguf_paths, find_model_path,
    huggingface_hub_cache, huggingface_hub_cache_dir, huggingface_identity_for_path,
    huggingface_repo_folder_name, layered_package_layer_count_for_path,
    layered_package_total_bytes_for_path, model_ref_for_path, replace_model_ref_path,
    scan_hf_cache_fast, scan_hf_cache_info, scan_installed_models, scan_installed_models_in,
    scan_local_models,
};

pub use mesh_llm_skippy_adapter::mesh_cache_dir as mesh_llm_cache_dir;

pub fn gguf_metadata_cache_path(path: &Path) -> Option<PathBuf> {
    skippy_model_hf::store::local::gguf_metadata_cache_path_in(path, &mesh_llm_cache_dir())
}

#[cfg(test)]
pub use skippy_model_hf::store::local::huggingface_snapshot_path;

pub fn find_mmproj_path(model_name: &str, model_path: &Path) -> Option<PathBuf> {
    if let Some(path) = crate::models::remote_catalog::find_loaded_model_exact(model_name)
        .and_then(|m| m.mmproj)
        .map(|asset| crate::models::catalog::models_dir().join(asset.file))
        .filter(|p| p.exists())
    {
        return Some(path);
    }
    skippy_model_hf::store::local::find_mmproj_path(model_name, model_path)
}

fn served_model_refs() -> &'static Mutex<HashSet<String>> {
    static SERVED_MODEL_REFS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    SERVED_MODEL_REFS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Record the ref this node serves `path` under.
///
/// The served ref is authoritative: it is what `/v1/models` must list and what
/// the request resolver routes on, whether it is a runtime-minted
/// `local-gguf/sha256-…` id, an explicit `--model` alias, or a configured
/// alias. Registering the path lets by-name lookups (GGUF metadata in
/// particular) reach the file; registering the ref lets the public-id
/// derivation know it must not rename the model from that path.
pub fn remember_served_model_ref(model_ref: &str, path: &Path) {
    // Replace, never prefer: the served alias must describe the file that is
    // loaded under it now, even if an earlier registration used another path.
    replace_model_ref_path(model_ref, path);
    if let Ok(mut refs) = served_model_refs().lock() {
        refs.insert(model_ref.to_string());
    }
}

/// Whether `model_ref` was registered by [`remember_served_model_ref`].
pub fn is_served_model_ref(model_ref: &str) -> bool {
    served_model_refs()
        .lock()
        .map(|refs| refs.contains(model_ref))
        .unwrap_or(false)
}
