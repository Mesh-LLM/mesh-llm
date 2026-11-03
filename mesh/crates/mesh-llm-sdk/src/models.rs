//! Typed discovery of all advertised workloads in the embedded node's mesh.
//!
//! These queries run in process and include non-LLM models such as OCR.

pub use mesh_llm_host_runtime::models::discovery::ModelWorkloadClass;
pub use mesh_llm_host_runtime::models::discovery::{
    MeshModel, MeshModelAvailability, MeshModelKind, MeshModelPage, MeshModelProvider,
    MeshModelQuery,
};

/// A borrowed view of the embedded node's primary mesh model inventory.
pub struct ModelCatalog<'a> {
    pub(crate) handle: &'a mesh_llm_embedded_runtime::EmbeddedMeshNodeHandle,
}

impl ModelCatalog<'_> {
    /// List a bounded page of mesh advertisements, filtering with AND semantics.
    ///
    /// # Errors
    /// Fails for invalid or expired queries and when the runtime has stopped.
    pub async fn list(&self, query: MeshModelQuery) -> anyhow::Result<MeshModelPage> {
        self.handle.mesh_models(query).await
    }

    /// Look up an exact public serving ID, including its quantization/profile.
    ///
    /// # Errors
    /// Fails when the ID is invalid, absent, or the runtime has stopped.
    pub async fn get(&self, id: &str) -> anyhow::Result<MeshModel> {
        self.handle.mesh_model(id).await
    }
}
