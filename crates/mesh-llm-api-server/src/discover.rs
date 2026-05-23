use crate::{InviteToken, MeshApiError, MeshNode, MeshNodeBuilder, OwnerKeypair};
pub use mesh_llm_api_client::{create_auto_client, discover_public_meshes, AutoConnectResult};
use mesh_llm_api_client::{PublicMesh, PublicMeshQuery};

pub struct AutoNodeResult {
    pub node: MeshNode,
    pub selected_mesh: PublicMesh,
}

pub async fn create_auto_node(
    owner_keypair: OwnerKeypair,
    query: PublicMeshQuery,
) -> Result<AutoNodeResult, MeshApiError> {
    let mesh = select_public_mesh(query).await?;
    let node = MeshNodeBuilder::from_public_mesh(owner_keypair, &mesh)?.build()?;
    Ok(AutoNodeResult {
        node,
        selected_mesh: mesh,
    })
}

impl MeshNodeBuilder {
    pub fn from_public_mesh(
        owner_keypair: OwnerKeypair,
        mesh: &PublicMesh,
    ) -> Result<Self, MeshApiError> {
        let token = mesh
            .invite_token
            .parse::<InviteToken>()
            .map_err(|message| MeshApiError::InvalidInviteToken { message })?;
        Ok(MeshNode::builder().identity(owner_keypair).join(token))
    }
}

async fn select_public_mesh(query: PublicMeshQuery) -> Result<PublicMesh, MeshApiError> {
    mesh_llm_api_client::select_public_mesh(query).await
}

#[cfg(test)]
mod tests {
    use super::PublicMesh;

    #[test]
    fn public_mesh_can_build_node_builder() {
        let mesh = PublicMesh {
            invite_token: "mesh-test:abc123".to_string(),
            serving: vec!["Qwen".to_string()],
            wanted: vec![],
            on_disk: vec![],
            total_vram_bytes: 32_000_000_000,
            node_count: 2,
            client_count: 1,
            max_clients: 0,
            name: Some("mesh-llm".to_string()),
            region: Some("AU".to_string()),
            mesh_id: Some("mesh-1".to_string()),
            publisher_npub: "npub1test".to_string(),
            published_at: 1,
            expires_at: Some(2),
        };
        let owner_keypair = crate::OwnerKeypair::generate();
        let builder = crate::MeshNodeBuilder::from_public_mesh(owner_keypair, &mesh);
        assert!(builder.is_ok());
    }
}
