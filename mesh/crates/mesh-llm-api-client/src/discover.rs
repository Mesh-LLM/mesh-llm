use crate::{ClientBuilder, InviteToken, MeshApiError, MeshClient, OwnerKeypair};

#[derive(Clone, Debug, Default)]
pub struct PublicMeshQuery {
    pub model: Option<String>,
    pub min_vram_gb: Option<f64>,
    pub region: Option<String>,
    pub target_name: Option<String>,
    pub relays: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicMesh {
    pub invite_token: String,
    pub serving: Vec<String>,
    pub wanted: Vec<String>,
    pub on_disk: Vec<String>,
    pub total_vram_bytes: u64,
    pub node_count: usize,
    pub client_count: usize,
    pub max_clients: usize,
    pub name: Option<String>,
    pub region: Option<String>,
    pub mesh_id: Option<String>,
    pub publisher_npub: String,
    pub published_at: u64,
    pub expires_at: Option<u64>,
}

pub struct AutoConnectResult {
    pub client: MeshClient,
    pub selected_mesh: PublicMesh,
}

impl PublicMesh {
    pub fn invite_token(&self) -> &str {
        &self.invite_token
    }

    pub fn client_builder(
        &self,
        owner_keypair: OwnerKeypair,
    ) -> Result<ClientBuilder, MeshApiError> {
        ClientBuilder::from_public_mesh(owner_keypair, self)
    }
}

impl From<mesh_client::network::nostr::DiscoveredMesh> for PublicMesh {
    fn from(value: mesh_client::network::nostr::DiscoveredMesh) -> Self {
        Self {
            invite_token: value.listing.invite_token,
            serving: value.listing.serving,
            wanted: value.listing.wanted,
            on_disk: value.listing.on_disk,
            total_vram_bytes: value.listing.total_vram_bytes,
            node_count: value.listing.node_count,
            client_count: value.listing.client_count,
            max_clients: value.listing.max_clients,
            name: value.listing.name,
            region: value.listing.region,
            mesh_id: value.listing.mesh_id,
            publisher_npub: value.publisher_npub,
            published_at: value.published_at,
            expires_at: value.expires_at,
        }
    }
}

pub async fn discover_public_meshes(
    query: PublicMeshQuery,
) -> Result<Vec<PublicMesh>, MeshApiError> {
    Ok(discover_matching_listings(&query)
        .await?
        .into_iter()
        .map(Into::into)
        .collect())
}

async fn discover_matching_listings(
    query: &PublicMeshQuery,
) -> Result<Vec<mesh_client::network::nostr::DiscoveredMesh>, MeshApiError> {
    let relays = resolve_relays(&query.relays);
    let filter = mesh_client::network::nostr::MeshFilter {
        model: query.model.clone(),
        min_vram_gb: query.min_vram_gb,
        region: query.region.clone(),
    };

    let discovered = mesh_client::network::nostr::discover(&relays, &filter, None)
        .await
        .map_err(|error| MeshApiError::Discovery {
            message: error.to_string(),
        })?;

    Ok(discovered
        .into_iter()
        .filter(|mesh| matches_target_name(mesh, query.target_name.as_deref()))
        .collect())
}

pub async fn create_auto_client(
    owner_keypair: OwnerKeypair,
    query: PublicMeshQuery,
) -> Result<AutoConnectResult, MeshApiError> {
    let mesh = select_public_mesh(query).await?;
    let client = mesh.client_builder(owner_keypair)?.build()?;
    Ok(AutoConnectResult {
        client,
        selected_mesh: mesh,
    })
}

impl ClientBuilder {
    pub fn from_public_mesh(
        owner_keypair: OwnerKeypair,
        mesh: &PublicMesh,
    ) -> Result<Self, MeshApiError> {
        let token = mesh
            .invite_token
            .parse::<InviteToken>()
            .map_err(|message| MeshApiError::InvalidInviteToken { message })?;
        Ok(Self::new(owner_keypair, token))
    }
}

pub async fn select_public_mesh(query: PublicMeshQuery) -> Result<PublicMesh, MeshApiError> {
    let discovered = discover_matching_listings(&query).await?;

    match mesh_client::network::nostr::smart_auto(
        &discovered,
        0.0,
        query.target_name.as_deref(),
        None,
    ) {
        mesh_client::network::nostr::AutoDecision::Join { candidates } => {
            prefer_serving_publishers(candidates.into_iter().map(|(_, mesh)| mesh).collect())
                .into_iter()
                .next()
                .map(PublicMesh::from)
                .ok_or(MeshApiError::NoPublicMeshFound)
        }
        mesh_client::network::nostr::AutoDecision::StartNew { .. } => {
            Err(MeshApiError::NoPublicMeshFound)
        }
    }
}

/// A thin client tunnels its HTTP requests to the listing's invite endpoint,
/// so the listing's publisher has to serve a model itself: a client or an
/// idle worker answers 503. Order candidates by what the publisher declares
/// (serves, then not declared by an older node, then does not serve),
/// keeping the scoring order within each group. Nothing is dropped, so an
/// older mesh stays reachable (#2765).
fn prefer_serving_publishers(
    mut candidates: Vec<mesh_client::network::nostr::DiscoveredMesh>,
) -> Vec<mesh_client::network::nostr::DiscoveredMesh> {
    candidates.sort_by_key(|mesh| match mesh.listing.publisher_serves {
        Some(true) => 0,
        None => 1,
        Some(false) => 2,
    });
    candidates
}

fn resolve_relays(relays: &[String]) -> Vec<String> {
    if relays.is_empty() {
        mesh_client::network::nostr::DEFAULT_RELAYS
            .iter()
            .map(|relay| (*relay).to_string())
            .collect()
    } else {
        relays.to_vec()
    }
}

fn matches_target_name(
    mesh: &mesh_client::network::nostr::DiscoveredMesh,
    target_name: Option<&str>,
) -> bool {
    let Some(target_name) = target_name else {
        return true;
    };

    mesh.listing
        .name
        .as_deref()
        .map(|name| name.eq_ignore_ascii_case(target_name))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::{PublicMesh, matches_target_name, prefer_serving_publishers};

    fn sample_mesh(name: Option<&str>) -> mesh_client::network::nostr::DiscoveredMesh {
        listing_from(name, None, "npub1test")
    }

    fn listing_from(
        name: Option<&str>,
        publisher_serves: Option<bool>,
        publisher: &str,
    ) -> mesh_client::network::nostr::DiscoveredMesh {
        mesh_client::network::nostr::DiscoveredMesh {
            listing: mesh_client::network::nostr::MeshListing {
                invite_token: "mesh-test:abc123".to_string(),
                serving: vec!["Qwen".to_string()],
                wanted: vec![],
                on_disk: vec![],
                total_vram_bytes: 32_000_000_000,
                node_count: 2,
                client_count: 1,
                max_clients: 0,
                name: name.map(str::to_string),
                region: Some("AU".to_string()),
                mesh_id: Some("mesh-1".to_string()),
                publisher_serves,
            },
            publisher_npub: publisher.to_string(),
            published_at: 1,
            expires_at: Some(2),
        }
    }

    #[test]
    fn a_serving_publisher_is_chosen_before_a_client_or_idle_one() {
        // Scoring order: a client-published listing first, as on the public
        // mesh in #2228, then an older node's, then a serving host's.
        let ordered = prefer_serving_publishers(vec![
            listing_from(None, Some(false), "client"),
            listing_from(None, None, "older"),
            listing_from(None, Some(true), "host-a"),
            listing_from(None, Some(true), "host-b"),
        ]);
        let publishers: Vec<&str> = ordered
            .iter()
            .map(|mesh| mesh.publisher_npub.as_str())
            .collect();
        // Serving hosts first in their scoring order, then the undeclared
        // listing, and the client-published one last but still reachable.
        assert_eq!(publishers, ["host-a", "host-b", "older", "client"]);
    }

    #[test]
    fn target_name_filter_is_case_insensitive() {
        let mesh = sample_mesh(Some("Mesh-LLM"));
        assert!(matches_target_name(&mesh, Some("mesh-llm")));
        assert!(!matches_target_name(&mesh, Some("other")));
    }

    #[test]
    fn public_mesh_can_build_client_builder() {
        let mesh = PublicMesh::from(sample_mesh(Some("mesh-llm")));
        let owner_keypair = crate::OwnerKeypair::generate();
        let builder = mesh.client_builder(owner_keypair);
        assert!(builder.is_ok());
    }
}
