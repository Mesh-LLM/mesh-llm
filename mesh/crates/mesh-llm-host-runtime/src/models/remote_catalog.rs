//! Shared meshllm/catalog lifecycle and resolution policy.

pub use skippy_model_hf::remote_catalog::*;

#[cfg(test)]
pub use skippy_model_resolver::{
    CatalogPackage, CatalogSidecarAsset as CatalogSidecarAssetRef,
    CatalogSidecarRef as CatalogSidecar, CatalogSource, CatalogVariant,
    CuratedMeta as CatalogCurated,
};

#[cfg(test)]
pub fn matching_primary_for_url(url: &str) -> Option<RemoteCatalogModel> {
    let (repo, revision, file) = skippy_model_resolver::parse_hf_resolve_url(url)?;
    matching_primary_for_huggingface(&repo, revision.as_deref(), &file)
}
