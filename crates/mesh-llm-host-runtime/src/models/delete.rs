use model_hf::store::delete::DeleteModelCatalog;

pub use model_hf::store::delete::DeleteResult;

#[cfg(test)]
pub use model_hf::store::delete::resolve_huggingface_file_from_sibling_entries;

struct HostDeleteCatalog;

impl DeleteModelCatalog for HostDeleteCatalog {
    fn local_stem_for_identifier(&self, identifier: &str) -> Option<String> {
        crate::models::remote_catalog::find_model_exact(identifier)
            .map(|model| model.file.trim_end_matches(".gguf").to_string())
    }
}

pub async fn resolve_model_identifier(identifier: &str) -> anyhow::Result<Vec<std::path::PathBuf>> {
    model_hf::store::delete::resolve_model_identifier_with_catalog(identifier, &HostDeleteCatalog)
        .await
}

pub async fn delete_model_by_identifier(identifier: &str) -> anyhow::Result<DeleteResult> {
    model_hf::store::delete::delete_model_by_identifier_with_catalog(identifier, &HostDeleteCatalog)
        .await
}
