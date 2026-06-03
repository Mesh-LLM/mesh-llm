use std::path::{Path, PathBuf};

pub use model_hf::store::local::{
    HuggingFaceModelIdentity, direct_hf_cache_root_gguf_paths, find_model_path,
    gguf_metadata_cache_path, huggingface_hub_cache, huggingface_hub_cache_dir,
    huggingface_identity_for_path, huggingface_repo_folder_name,
    layered_package_layer_count_for_path, layered_package_total_bytes_for_path, mesh_llm_cache_dir,
    model_ref_for_path, scan_hf_cache_fast, scan_hf_cache_info, scan_installed_models,
    scan_local_models,
};

#[cfg(test)]
pub use model_hf::store::local::huggingface_snapshot_path;

pub fn find_mmproj_path(model_name: &str, model_path: &Path) -> Option<PathBuf> {
    if let Some(path) = crate::models::remote_catalog::find_loaded_model_exact(model_name)
        .and_then(|m| m.mmproj)
        .map(|asset| crate::models::catalog::models_dir().join(asset.file))
        .filter(|p| p.exists())
    {
        return Some(path);
    }
    model_hf::store::local::find_mmproj_path(model_name, model_path)
}
