use skippy_model_hf::store::delete::CuratedDeleteCatalog;

pub use skippy_model_hf::store::delete::DeleteResult;

#[cfg(test)]
pub use skippy_model_hf::store::delete::resolve_huggingface_file_from_sibling_entries;

pub async fn resolve_model_identifier(identifier: &str) -> anyhow::Result<Vec<std::path::PathBuf>> {
    skippy_model_hf::store::delete::resolve_model_identifier_with_catalog(
        identifier,
        &CuratedDeleteCatalog,
    )
    .await
}

pub async fn delete_model_by_identifier(identifier: &str) -> anyhow::Result<DeleteResult> {
    skippy_model_hf::store::delete::delete_model_by_identifier_with_catalog_in(
        identifier,
        &CuratedDeleteCatalog,
        &crate::models::local::mesh_llm_cache_dir(),
    )
    .await
}
