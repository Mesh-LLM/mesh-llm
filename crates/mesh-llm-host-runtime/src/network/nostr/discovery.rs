//! Nostr mesh discovery.

#[cfg(test)]
use super::contracts::DEFAULT_RELAYS;
use super::contracts::{DiscoveredMesh, MESH_SERVICE_KIND, MeshListing};
#[cfg(test)]
use super::publish::Publisher;
use anyhow::Result;
use nostr_sdk::prelude::*;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Discovery — find meshes on Nostr
// ---------------------------------------------------------------------------

/// Criteria for filtering discovered meshes.
#[derive(Debug, Clone, Default)]
pub struct MeshFilter {
    /// Match meshes by name (case-insensitive exact match)
    pub name: Option<String>,
    /// Match meshes serving (or wanting) this model name (substring match)
    pub model: Option<String>,
    /// Minimum total VRAM in GB
    pub min_vram_gb: Option<f64>,
    /// Geographic region
    pub region: Option<String>,
}

impl MeshFilter {
    pub fn matches(&self, mesh: &DiscoveredMesh) -> bool {
        if let Some(ref name) = self.name {
            match &mesh.listing.name {
                Some(n) if n.eq_ignore_ascii_case(name) => {}
                _ => return false,
            }
        }
        if let Some(ref model) = self.model {
            let model_lower = model.to_lowercase();
            let has_model = mesh
                .listing
                .serving
                .iter()
                .any(|m| m.to_lowercase().contains(&model_lower))
                || mesh
                    .listing
                    .wanted
                    .iter()
                    .any(|m| m.to_lowercase().contains(&model_lower))
                || mesh
                    .listing
                    .on_disk
                    .iter()
                    .any(|m| m.to_lowercase().contains(&model_lower));
            if !has_model {
                return false;
            }
        }
        if let Some(min_gb) = self.min_vram_gb {
            let vram_gb = mesh.listing.total_vram_bytes as f64 / 1e9;
            if vram_gb < min_gb {
                return false;
            }
        }
        if let Some(ref region) = self.region {
            match &mesh.listing.region {
                Some(r) if r.eq_ignore_ascii_case(region) => {}
                _ => return false,
            }
        }
        true
    }
}

/// A reusable read-only Nostr client for discovery.
/// Create once, pass to repeated `discover()` calls to avoid opening
/// new websocket connections and generating throwaway keys every time.
pub struct DiscoveryClient {
    client: Client,
}

impl DiscoveryClient {
    pub async fn new(relays: &[String]) -> Result<Self> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let keys = Keys::generate();
        let client = Client::new(keys);
        let mut added = 0;
        for relay in relays {
            match client.add_relay(relay).await {
                Ok(_) => added += 1,
                Err(e) => tracing::warn!("Nostr relay {relay}: {e}"),
            }
        }
        if added == 0 {
            anyhow::bail!(
                "Could not connect to any Nostr relay (tried {})",
                relays.len()
            );
        }
        client.connect().await;
        Ok(Self { client })
    }
}

/// Discover meshes from Nostr relays.
///
/// If `cached_client` is provided, reuses its connections.  Otherwise
/// creates (and drops) a one-shot client — fine for the initial
/// `--auto` join but wasteful in tight loops.
pub async fn discover(
    relays: &[String],
    filter: &MeshFilter,
    cached_client: Option<&DiscoveryClient>,
) -> Result<Vec<DiscoveredMesh>> {
    // Build a temporary client only when no cached one is supplied.
    let _tmp;
    let client: &Client = if let Some(cc) = cached_client {
        &cc.client
    } else {
        _tmp = build_discovery_client(relays).await?;
        &_tmp
    };

    let nostr_filter = Filter::new()
        .kind(Kind::Custom(MESH_SERVICE_KIND))
        .custom_tag(
            SingleLetterTag::lowercase(Alphabet::K),
            "mesh-llm".to_string(),
        )
        .limit(100);

    let events = match client
        .fetch_events(nostr_filter, Duration::from_secs(5))
        .await
    {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("Nostr fetch failed: {e}");
            return Ok(Vec::new()); // No results rather than hard error
        }
    };

    let now = Timestamp::now().as_secs();

    // Dedupe by publisher (keep latest per pubkey, using replaceable event semantics)
    let latest = latest_events_by_pubkey(&events);

    let mut meshes = Vec::new();
    for event in latest.values() {
        let Some(discovered) = parse_discovered_mesh(event, now) else {
            continue;
        };

        if filter.matches(&discovered) {
            meshes.push(discovered);
        }
    }

    // Sort by node count (bigger meshes first), then VRAM
    meshes.sort_by(|a, b| {
        b.listing
            .node_count
            .cmp(&a.listing.node_count)
            .then(b.listing.total_vram_bytes.cmp(&a.listing.total_vram_bytes))
    });

    Ok(meshes)
}

async fn build_discovery_client(relays: &[String]) -> Result<Client> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let keys = Keys::generate();
    let client = Client::new(keys);
    let mut added = 0;
    for relay in relays {
        match client.add_relay(relay).await {
            Ok(_) => added += 1,
            Err(e) => tracing::warn!("Nostr relay {relay}: {e}"),
        }
    }
    if added == 0 {
        anyhow::bail!(
            "Could not connect to any Nostr relay (tried {})",
            relays.len()
        );
    }
    client.connect().await;
    Ok(client)
}

fn latest_events_by_pubkey<'a>(events: &'a Events) -> std::collections::HashMap<String, &'a Event> {
    let mut latest: std::collections::HashMap<String, &'a Event> = std::collections::HashMap::new();
    for event in events.iter() {
        let pubkey = event.pubkey.to_hex();
        match latest.get(&pubkey) {
            Some(existing) if event.created_at.as_secs() <= existing.created_at.as_secs() => {}
            _ => {
                latest.insert(pubkey, event);
            }
        }
    }
    latest
}

fn parse_discovered_mesh(event: &Event, now: u64) -> Option<DiscoveredMesh> {
    let expires_at = event
        .tags
        .iter()
        .find(|t| t.as_slice().first().map(|s| s.as_str()) == Some("expiration"))
        .and_then(|t| t.as_slice().get(1))
        .and_then(|s| s.parse::<u64>().ok());
    if expires_at.is_some_and(|exp| exp < now) {
        return None;
    }

    let listing: MeshListing = match serde_json::from_str(&event.content) {
        Ok(listing) => listing,
        Err(err) => {
            tracing::warn!(
                "Skipping Nostr listing from {}: bad JSON: {err}",
                event.pubkey.to_bech32().unwrap_or_default()
            );
            return None;
        }
    };
    if let Err(err) = crate::mesh::Node::decode_invite_token(&listing.invite_token) {
        tracing::warn!(
            "Skipping Nostr listing from {}: {err}",
            event.pubkey.to_bech32().unwrap_or_default()
        );
        return None;
    }

    Some(DiscoveredMesh {
        listing,
        publisher_npub: event.pubkey.to_bech32().unwrap_or_default(),
        published_at: event.created_at.as_secs(),
        expires_at,
    })
}

#[cfg(test)]
mod filter_tests {
    use super::*;

    fn make_mesh_for_filter(
        serving: &[&str],
        wanted: &[&str],
        on_disk: &[&str],
        vram: u64,
        region: Option<&str>,
    ) -> DiscoveredMesh {
        make_mesh_for_filter_named(serving, wanted, on_disk, vram, region, None)
    }

    fn make_mesh_for_filter_named(
        serving: &[&str],
        wanted: &[&str],
        on_disk: &[&str],
        vram: u64,
        region: Option<&str>,
        name: Option<&str>,
    ) -> DiscoveredMesh {
        DiscoveredMesh {
            listing: MeshListing {
                invite_token: "tok".into(),
                serving: serving.iter().map(|s| s.to_string()).collect(),
                wanted: wanted.iter().map(|s| s.to_string()).collect(),
                on_disk: on_disk.iter().map(|s| s.to_string()).collect(),
                total_vram_bytes: vram,
                node_count: 1,
                client_count: 0,
                max_clients: 0,
                name: name.map(|s| s.to_string()),
                region: region.map(|s| s.to_string()),
                mesh_id: None,
            },
            publisher_npub: "npub-test".into(),
            published_at: 1000,
            expires_at: Some(2000),
        }
    }

    #[test]
    fn filter_default_matches_all() {
        let m = make_mesh_for_filter(&["Qwen3-8B-Q4_K_M"], &[], &[], 8_000_000_000, None);
        assert!(MeshFilter::default().matches(&m));
    }

    #[test]
    fn filter_model_serving() {
        let m = make_mesh_for_filter(&["Qwen3-8B-Q4_K_M"], &[], &[], 8_000_000_000, None);
        let f = MeshFilter {
            model: Some("qwen3-8b".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_model_wanted() {
        let m = make_mesh_for_filter(&[], &["Qwen3-32B-Q4_K_M"], &[], 8_000_000_000, None);
        let f = MeshFilter {
            model: Some("32b".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_model_on_disk() {
        let m = make_mesh_for_filter(&[], &[], &["MiniMax-M2.5-Q4_K_M"], 8_000_000_000, None);
        let f = MeshFilter {
            model: Some("minimax".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_model_no_match() {
        let m = make_mesh_for_filter(&["Qwen3-8B-Q4_K_M"], &[], &[], 8_000_000_000, None);
        let f = MeshFilter {
            model: Some("llama".into()),
            ..Default::default()
        };
        assert!(!f.matches(&m));
    }

    #[test]
    fn filter_min_vram() {
        let m = make_mesh_for_filter(&[], &[], &[], 8_000_000_000, None);
        let pass = MeshFilter {
            min_vram_gb: Some(5.0),
            ..Default::default()
        };
        let fail = MeshFilter {
            min_vram_gb: Some(16.0),
            ..Default::default()
        };
        assert!(pass.matches(&m));
        assert!(!fail.matches(&m));
    }

    #[test]
    fn filter_region() {
        let m = make_mesh_for_filter(&[], &[], &[], 8_000_000_000, Some("us-east"));
        let pass = MeshFilter {
            region: Some("us-east".into()),
            ..Default::default()
        };
        let fail = MeshFilter {
            region: Some("eu-west".into()),
            ..Default::default()
        };
        assert!(pass.matches(&m));
        assert!(!fail.matches(&m));
    }

    #[test]
    fn filter_region_case_insensitive() {
        let m = make_mesh_for_filter(&[], &[], &[], 8_000_000_000, Some("US-East"));
        let f = MeshFilter {
            region: Some("us-east".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_combined() {
        let m = make_mesh_for_filter(
            &["Qwen3-8B-Q4_K_M"],
            &[],
            &[],
            16_000_000_000,
            Some("us-east"),
        );
        let pass = MeshFilter {
            model: Some("qwen3".into()),
            min_vram_gb: Some(10.0),
            region: Some("us-east".into()),
            ..Default::default()
        };
        let fail_model = MeshFilter {
            model: Some("llama".into()),
            min_vram_gb: Some(10.0),
            region: Some("us-east".into()),
            ..Default::default()
        };
        assert!(pass.matches(&m));
        assert!(!fail_model.matches(&m));
    }

    #[test]
    fn filter_name_exact() {
        let m = make_mesh_for_filter_named(&[], &[], &[], 8_000_000_000, None, Some("poker-night"));
        let f = MeshFilter {
            name: Some("poker-night".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_name_case_insensitive() {
        let m = make_mesh_for_filter_named(&[], &[], &[], 8_000_000_000, None, Some("Poker-Night"));
        let f = MeshFilter {
            name: Some("poker-night".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_name_no_match() {
        let m = make_mesh_for_filter_named(&[], &[], &[], 8_000_000_000, None, Some("other-mesh"));
        let f = MeshFilter {
            name: Some("poker-night".into()),
            ..Default::default()
        };
        assert!(!f.matches(&m));
    }

    #[test]
    fn filter_name_mesh_unnamed() {
        // Mesh has no name — filter by name should not match.
        let m = make_mesh_for_filter_named(&[], &[], &[], 8_000_000_000, None, None);
        let f = MeshFilter {
            name: Some("poker-night".into()),
            ..Default::default()
        };
        assert!(!f.matches(&m));
    }

    #[test]
    fn filter_name_none_matches_all() {
        // No name filter — matches meshes with and without names.
        let named =
            make_mesh_for_filter_named(&[], &[], &[], 8_000_000_000, None, Some("poker-night"));
        let unnamed = make_mesh_for_filter_named(&[], &[], &[], 8_000_000_000, None, None);
        let f = MeshFilter::default();
        assert!(f.matches(&named));
        assert!(f.matches(&unnamed));
    }
}

// ---------------------------------------------------------------------------
// Integration test — publish/discover against real Nostr relays
// ---------------------------------------------------------------------------
#[cfg(test)]
mod integration_tests {
    use super::*;
    use base64::Engine;

    fn fake_invite_token() -> String {
        // Build a syntactically valid invite token by encoding a minimal
        // EndpointAddr JSON. We use Node::invite_token indirectly by just
        // crafting the JSON that decode_invite_token expects.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        // EndpointAddr serialises as {"id":"<base32>","addrs":[]}
        // We need a valid 32-byte public key. Use a deterministic one.
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&n.to_le_bytes());
        let key = iroh::SecretKey::from_bytes(&seed);
        let addr = iroh::EndpointAddr {
            id: iroh::EndpointId::from(key.public()),
            addrs: Default::default(),
        };
        let json = serde_json::to_vec(&addr).expect("serialize");
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json)
    }

    /// End-to-end: two publishers advertise the same mesh, a reusable
    /// DiscoveryClient finds both listings, and fields round-trip correctly.
    /// Covers publish, discover, multi-publisher, and client reuse in one test.
    #[tokio::test]
    #[ignore = "requires live public Nostr relays"]
    async fn publish_discover_round_trip() {
        let relays: Vec<String> = DEFAULT_RELAYS.iter().map(|s| s.to_string()).collect();
        let mesh_name = format!("mesh-llm-test-{}", rand::random::<u32>());
        let mesh_id = format!("test-id-{}", rand::random::<u32>());

        // Publisher A
        let keys_a = Keys::generate();
        let pub_a = Publisher::new(keys_a.clone(), &relays)
            .await
            .expect("pub_a");
        let token_a = fake_invite_token();
        let token_b = fake_invite_token();
        let listing_a = MeshListing {
            invite_token: token_a.clone(),
            serving: vec!["Qwen3-8B-Q4_K_M".into()],
            wanted: vec![],
            on_disk: vec![],
            total_vram_bytes: 16_000_000_000,
            node_count: 2,
            client_count: 0,
            max_clients: 0,
            name: Some(mesh_name.clone()),
            region: Some("test-region".into()),
            mesh_id: Some(mesh_id.clone()),
        };
        pub_a.publish(&listing_a, 120).await.expect("publish A");

        // Publisher B — same mesh, different invite token
        let keys_b = Keys::generate();
        let pub_b = Publisher::new(keys_b.clone(), &relays)
            .await
            .expect("pub_b");
        let mut listing_b = listing_a.clone();
        listing_b.invite_token = token_b.clone();
        pub_b.publish(&listing_b, 120).await.expect("publish B");

        tokio::time::sleep(Duration::from_secs(3)).await;

        // Discover with reusable client (tests DiscoveryClient + discover)
        let dc = DiscoveryClient::new(&relays).await.expect("dc");
        let meshes = discover(&relays, &MeshFilter::default(), Some(&dc))
            .await
            .expect("discover");

        let found: Vec<_> = meshes
            .iter()
            .filter(|m| m.listing.mesh_id.as_deref() == Some(mesh_id.as_str()))
            .collect();
        assert!(
            found.len() >= 2,
            "should find both publishers for mesh_id={mesh_id}, found {}",
            found.len()
        );

        // Verify fields round-tripped
        let m = &found[0];
        assert_eq!(m.listing.name.as_deref(), Some(mesh_name.as_str()));
        assert_eq!(m.listing.serving, vec!["Qwen3-8B-Q4_K_M"]);
        assert_eq!(m.listing.node_count, 2);
        assert_eq!(m.listing.total_vram_bytes, 16_000_000_000);

        // Both invite tokens present
        let tokens: Vec<_> = found
            .iter()
            .map(|m| m.listing.invite_token.as_str())
            .collect();
        assert!(
            tokens.contains(&token_a.as_str()),
            "missing token_a in {tokens:?}"
        );
        assert!(
            tokens.contains(&token_b.as_str()),
            "missing token_b in {tokens:?}"
        );

        // Second discover with same client still works
        let r2 = discover(&relays, &MeshFilter::default(), Some(&dc))
            .await
            .expect("second discover");
        let found2: Vec<_> = r2
            .iter()
            .filter(|m| m.listing.mesh_id.as_deref() == Some(mesh_id.as_str()))
            .collect();
        assert!(found2.len() >= 2, "reused client should still find both");

        // Cleanup
        pub_a.unpublish().await.ok();
        pub_b.unpublish().await.ok();
    }
}
