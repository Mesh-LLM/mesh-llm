use mesh_llm_sdk::discover_public_meshes as sdk_discover_public_meshes;

use crate::errors::{FfiError, map_mesh_api_error};
use crate::request_types::{PublicMesh, PublicMeshQuery};
use crate::runtime_blocking::block_on;

#[uniffi::export]
pub fn discover_public_meshes(query: PublicMeshQuery) -> Result<Vec<PublicMesh>, FfiError> {
    block_on(sdk_discover_public_meshes(query.into()))
        .map(|meshes| meshes.into_iter().map(PublicMesh::from).collect())
        .map_err(map_mesh_api_error)
}
