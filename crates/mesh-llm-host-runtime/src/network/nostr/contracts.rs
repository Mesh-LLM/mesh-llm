//! Shared Nostr listing contracts.

use serde::{Deserialize, Serialize};

/// NIP-89 "Application Handler" kind — used for service advertisements.
pub const MESH_SERVICE_KIND: u16 = 31990;

/// Default public relays.
pub const DEFAULT_RELAYS: &[&str] = &[
    "wss://relay.damus.io",
    "wss://nos.lol",
    "wss://relay.nostr.band",
    "wss://nostr.land",
    "wss://nostr.wine",
];

/// What we publish about a mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshListing {
    /// Base64 join token.
    ///
    /// Legacy meshes publish an endpoint-only invite token. Requirement-aware
    /// meshes publish an origin-signed bootstrap token that carries only the
    /// endpoint bootstrap material plus canonical genesis-policy metadata.
    pub invite_token: String,
    /// Models currently loaded and serving inference
    pub serving: Vec<String>,
    /// Models the mesh wants but nobody is serving yet (need more GPUs)
    #[serde(default)]
    pub wanted: Vec<String>,
    /// Models on disk across the mesh (could be loaded if a GPU becomes free)
    #[serde(default)]
    pub on_disk: Vec<String>,
    /// Total VRAM across all GPU nodes (bytes)
    pub total_vram_bytes: u64,
    /// Number of GPU nodes in the mesh
    pub node_count: usize,
    /// Number of connected clients (API-only nodes)
    #[serde(default)]
    pub client_count: usize,
    /// Maximum clients this mesh accepts (0 = unlimited)
    #[serde(default)]
    pub max_clients: usize,
    /// Optional human-readable name for the mesh
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional geographic region
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Stable mesh identity — all nodes in the same mesh share this ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_id: Option<String>,
}

/// Discovered mesh from Nostr.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DiscoveredMesh {
    pub listing: MeshListing,
    pub publisher_npub: String,
    pub published_at: u64,
    pub expires_at: Option<u64>,
}

impl std::fmt::Display for DiscoveredMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vram_gb = self.listing.total_vram_bytes as f64 / 1e9;
        let models = if self.listing.serving.is_empty() {
            "(no models loaded)".to_string()
        } else {
            self.listing.serving.join(", ")
        };
        write!(
            f,
            "{}  {} node(s), {:.0}GB capacity  serving: {}",
            self.listing.name.as_deref().unwrap_or("(unnamed)"),
            self.listing.node_count,
            vram_gb,
            models,
        )?;
        if let Some(ref region) = self.listing.region {
            write!(f, "  region: {}", region)?;
        }
        if !self.listing.wanted.is_empty() {
            write!(f, "  wanted: {}", self.listing.wanted.join(", "))?;
        }
        Ok(())
    }
}
