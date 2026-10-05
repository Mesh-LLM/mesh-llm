pub mod delete;
pub mod installed;
pub mod local;
pub mod updates;
pub mod usage;

pub use delete::{
    DeleteModelCatalog, DeleteResult, NoDeleteCatalog, collect_delete_paths,
    collect_delete_paths_in, delete_model_by_identifier, delete_model_by_identifier_with_catalog,
    delete_model_by_identifier_with_catalog_in, resolve_model_identifier,
    resolve_model_identifier_with_catalog,
};
pub use local::{
    HuggingFaceModelIdentity, direct_hf_cache_root_gguf_paths, find_mmproj_path, find_model_path,
    gguf_metadata_cache_path, gguf_metadata_cache_path_in, huggingface_hub_cache,
    huggingface_hub_cache_dir, huggingface_identity_for_path, huggingface_repo_folder_name,
    huggingface_snapshot_path, layered_package_layer_count_for_path,
    layered_package_total_bytes_for_path, model_metadata_cache_dir, model_metadata_cache_dir_in,
    model_ref_for_path, scan_hf_cache_fast, scan_hf_cache_info, scan_installed_models,
    scan_installed_models_in, scan_local_models,
};
pub use usage::{
    ModelCleanupPlan, ModelCleanupResult, ModelUsageRecord, execute_model_cleanup,
    execute_model_cleanup_in, load_model_usage_record_for_path,
    load_model_usage_record_for_path_in, model_usage_cache_dir, model_usage_cache_dir_in,
    plan_model_cleanup, plan_model_cleanup_in, track_managed_model_usage,
    track_managed_model_usage_in, track_model_usage, track_model_usage_in,
};
