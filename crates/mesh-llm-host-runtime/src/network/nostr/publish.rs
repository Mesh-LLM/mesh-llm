//! Nostr publishing loops and listing construction.

use super::contracts::{DiscoveredMesh, MESH_SERVICE_KIND, MeshListing};
use super::discovery::{DiscoveryClient, MeshFilter, discover};
use super::keys::load_or_create_keys;
use anyhow::Result;
use nostr_sdk::prelude::*;
use std::time::Duration;

pub struct PublishLoopConfig {
    pub relays: Vec<String>,
    pub name: Option<String>,
    pub region: Option<String>,
    pub max_clients: Option<usize>,
    pub interval_secs: u64,
    pub status_tx: Option<tokio::sync::watch::Sender<Option<PublishStateUpdate>>>,
}

// ---------------------------------------------------------------------------
// Publisher — background task that keeps the listing fresh
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublishStateUpdate {
    Public,
    PublishFailed,
}

fn report_publish_state(
    status_tx: &Option<tokio::sync::watch::Sender<Option<PublishStateUpdate>>>,
    last_reported: &mut Option<PublishStateUpdate>,
    next: PublishStateUpdate,
) {
    if *last_reported == Some(next) {
        return;
    }
    if let Some(tx) = status_tx {
        let _ = tx.send(Some(next));
    }
    *last_reported = Some(next);
}

pub struct Publisher {
    client: Client,
    keys: Keys,
}

impl Publisher {
    pub async fn new(keys: Keys, relays: &[String]) -> Result<Self> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = Client::new(keys.clone());
        for relay in relays {
            client.add_relay(relay).await?;
        }
        client.connect().await;
        Ok(Self { client, keys })
    }

    pub fn npub(&self) -> String {
        self.keys.public_key().to_bech32().unwrap_or_default()
    }

    /// Publish (or replace) the mesh listing. Uses a replaceable event
    /// (kind 31990 + d-tag) so each publisher has exactly one listing.
    pub async fn publish(&self, listing: &MeshListing, ttl_secs: u64) -> Result<()> {
        let expiration = Timestamp::now().as_secs() + ttl_secs;
        let content = serde_json::to_string(listing)?;

        let tags = vec![
            Tag::custom(TagKind::Custom("d".into()), vec!["mesh-llm".to_string()]),
            Tag::custom(TagKind::Custom("k".into()), vec!["mesh-llm".to_string()]),
            Tag::custom(
                TagKind::Custom("expiration".into()),
                vec![expiration.to_string()],
            ),
        ];

        let builder = EventBuilder::new(Kind::Custom(MESH_SERVICE_KIND), content).tags(tags);
        self.client.send_event_builder(builder).await?;
        Ok(())
    }

    /// Delete our listing (e.g. on shutdown).
    pub async fn unpublish(&self) -> Result<()> {
        // Fetch our own events
        let filter = Filter::new()
            .kind(Kind::Custom(MESH_SERVICE_KIND))
            .author(self.keys.public_key())
            .limit(10);
        let events = self
            .client
            .fetch_events(filter, Duration::from_secs(5))
            .await?;
        for event in events.iter() {
            let request = EventDeletionRequest::new().id(event.id);
            let _ = self
                .client
                .send_event_builder(EventBuilder::delete(request))
                .await;
        }
        Ok(())
    }
}

/// Background publish loop. Republishes every `interval` seconds using
/// fresh data from the mesh node.
///
/// If `max_clients` is set, delists when that many clients are connected
/// and re-publishes when clients drop below the cap.
pub async fn publish_loop(node: crate::mesh::Node, keys: Keys, config: PublishLoopConfig) {
    let PublishLoopConfig {
        relays,
        name,
        region,
        max_clients,
        interval_secs,
        status_tx,
    } = config;
    let mut last_reported = None;
    let Some(publisher) =
        create_publish_loop_publisher(&keys, &relays, &status_tx, &mut last_reported).await
    else {
        return;
    };

    let npub = publisher.npub();
    log_publish_client_cap(max_clients);

    // Wait for local serving to be ready before first publish (up to 60s).
    wait_for_local_serving_ready(&node).await;
    eprintln!(
        "📡 Publishing mesh to Nostr (npub: {}...{})",
        &npub[..12],
        &npub[npub.len() - 8..]
    );

    let mut delisted = false;

    // Reusable client for solo-convergence discovery checks.
    let disco = DiscoveryClient::new(&relays).await.ok();

    loop {
        let peers = node.peers().await;
        let client_count = peer_client_count(&peers);
        if update_delisted_state(
            &publisher,
            max_clients,
            client_count,
            &mut delisted,
            interval_secs,
        )
        .await
        {
            continue;
        }

        if wait_while_delisted(delisted, interval_secs).await {
            continue;
        }

        if maybe_rejoin_larger_mesh(
            &node,
            &publisher,
            &relays,
            name.as_deref(),
            interval_secs,
            disco.as_ref(),
            &peers,
        )
        .await
        {
            continue;
        }

        let invite_token = node.invite_token().await;
        let listing = build_publish_listing(
            &node,
            &peers,
            invite_token,
            client_count,
            max_clients,
            name.clone(),
            region.clone(),
        )
        .await;

        publish_current_listing(
            &publisher,
            &listing,
            interval_secs,
            client_count,
            &status_tx,
            &mut last_reported,
        )
        .await;

        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
    }
}

// ---------------------------------------------------------------------------
// Publish watchdog — take over publishing if the original publisher dies
// ---------------------------------------------------------------------------

/// Watch for our mesh's Nostr listing to disappear, then start publishing.
/// Multiple nodes may start publishing simultaneously — that's fine, each
/// publishes with their own key and invite token, giving discoverers
/// multiple entry points to the same mesh.
///
/// Only runs on active (non-client) nodes that joined via `--auto`.
pub async fn publish_watchdog(
    node: crate::mesh::Node,
    relays: Vec<String>,
    mesh_name: Option<String>,
    region: Option<String>,
    check_interval_secs: u64,
    status_tx: Option<tokio::sync::watch::Sender<Option<PublishStateUpdate>>>,
) {
    watchdog_initial_delay().await;

    // Reusable client for repeated discovery checks.
    let disco = DiscoveryClient::new(&relays).await.ok();
    let filter = MeshFilter::default();

    loop {
        match discover(&relays, &filter, disco.as_ref()).await {
            Ok(meshes) => {
                if should_take_over_publish(&node, &meshes).await {
                    if !confirm_missing_listing_after_backoff(
                        &relays,
                        &filter,
                        disco.as_ref(),
                        &node,
                    )
                    .await
                    {
                        tokio::time::sleep(Duration::from_secs(check_interval_secs)).await;
                        continue;
                    }

                    eprintln!("📡 Taking over Nostr publishing for the mesh");
                    let Some(keys) = load_watchdog_publish_keys(check_interval_secs).await else {
                        continue;
                    };
                    publish_loop(
                        node,
                        keys,
                        watchdog_takeover_publish_config(
                            relays,
                            mesh_name,
                            region,
                            check_interval_secs,
                            status_tx,
                        ),
                    )
                    .await;
                    return;
                }
            }
            Err(e) => {
                tracing::debug!("Publish watchdog: Nostr check failed: {e}");
            }
        }

        // Check frequently so we catch gaps fast
        let next_check = (rand::random::<u64>() % 15) + 20; // 20-35s
        tokio::time::sleep(Duration::from_secs(next_check)).await;
    }
}

async fn wait_for_local_serving_ready(node: &crate::mesh::Node) {
    for _ in 0..120 {
        if node.is_llama_ready().await {
            return;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

fn peer_client_count(peers: &[crate::mesh::PeerInfo]) -> usize {
    peers
        .iter()
        .filter(|peer| matches!(peer.role, crate::mesh::NodeRole::Client))
        .count()
}

fn non_client_peer_count(peers: &[crate::mesh::PeerInfo]) -> usize {
    peers
        .iter()
        .filter(|peer| !matches!(peer.role, crate::mesh::NodeRole::Client))
        .count()
}

async fn maybe_rejoin_larger_mesh(
    node: &crate::mesh::Node,
    publisher: &Publisher,
    relays: &[String],
    mesh_name: Option<&str>,
    interval_secs: u64,
    disco: Option<&DiscoveryClient>,
    peers: &[crate::mesh::PeerInfo],
) -> bool {
    let Some(my_node_count) = solo_mesh_node_count(peers) else {
        return false;
    };
    let Ok(listings) = discover(relays, &MeshFilter::default(), disco).await else {
        return false;
    };
    let my_npub = publisher.npub();
    let my_mesh_id = node.mesh_id().await;
    let target = pick_larger_mesh_target(
        &listings,
        &my_npub,
        my_mesh_id.as_deref(),
        mesh_name,
        my_node_count,
    );
    let Some(target) = target else {
        return false;
    };
    rejoin_larger_mesh_target(node, publisher, target, my_node_count, interval_secs).await
}

async fn create_publish_loop_publisher(
    keys: &Keys,
    relays: &[String],
    status_tx: &Option<tokio::sync::watch::Sender<Option<PublishStateUpdate>>>,
    last_reported: &mut Option<PublishStateUpdate>,
) -> Option<Publisher> {
    match Publisher::new(keys.clone(), relays).await {
        Ok(publisher) => Some(publisher),
        Err(err) => {
            report_publish_state(status_tx, last_reported, PublishStateUpdate::PublishFailed);
            tracing::error!("Failed to create Nostr publisher: {err}");
            None
        }
    }
}

fn log_publish_client_cap(max_clients: Option<usize>) {
    if let Some(cap) = max_clients {
        eprintln!("   Will delist when {} clients connected", cap);
    }
}

async fn wait_while_delisted(delisted: bool, interval_secs: u64) -> bool {
    if !delisted {
        return false;
    }
    tokio::time::sleep(Duration::from_secs(interval_secs)).await;
    true
}

async fn publish_current_listing(
    publisher: &Publisher,
    listing: &MeshListing,
    interval_secs: u64,
    client_count: usize,
    status_tx: &Option<tokio::sync::watch::Sender<Option<PublishStateUpdate>>>,
    last_reported: &mut Option<PublishStateUpdate>,
) {
    let ttl = interval_secs * 2;
    match publisher.publish(listing, ttl).await {
        Ok(()) => {
            report_publish_state(status_tx, last_reported, PublishStateUpdate::Public);
            tracing::debug!(
                "Published mesh listing ({} models, {} nodes, {} clients)",
                listing.serving.len(),
                listing.node_count,
                client_count
            );
        }
        Err(err) => {
            report_publish_state(status_tx, last_reported, PublishStateUpdate::PublishFailed);
            tracing::warn!("Failed to publish to Nostr: {err}");
        }
    }
}

async fn watchdog_initial_delay() {
    let jitter = (rand::random::<u64>() % 20) + 10;
    tokio::time::sleep(Duration::from_secs(jitter)).await;
}

async fn should_take_over_publish(node: &crate::mesh::Node, meshes: &[DiscoveredMesh]) -> bool {
    let our_peers = node.peers().await;
    let served = node.models_being_served().await;
    let our_mesh_id = node.mesh_id().await;
    !mesh_listing_present(meshes, our_mesh_id.as_deref(), &served)
        && (!our_peers.is_empty() || !served.is_empty())
}

async fn confirm_missing_listing_after_backoff(
    relays: &[String],
    filter: &MeshFilter,
    disco: Option<&DiscoveryClient>,
    node: &crate::mesh::Node,
) -> bool {
    let backoff = (rand::random::<u64>() % 7) + 3;
    eprintln!("📡 Mesh listing missing from Nostr — waiting {backoff}s before taking over...");
    tokio::time::sleep(Duration::from_secs(backoff)).await;

    let Ok(recheck) = discover(relays, filter, disco).await else {
        return true;
    };
    let served = node.models_being_served().await;
    let our_mesh_id = node.mesh_id().await;
    let still_missing = !mesh_listing_present(&recheck, our_mesh_id.as_deref(), &served);
    if !still_missing {
        eprintln!("📡 Someone else took over publishing — standing down");
    }
    still_missing
}

async fn load_watchdog_publish_keys(check_interval_secs: u64) -> Option<Keys> {
    match load_or_create_keys() {
        Ok(keys) => Some(keys),
        Err(err) => {
            tracing::warn!("Failed to load Nostr keys for publish takeover: {err}");
            tokio::time::sleep(Duration::from_secs(check_interval_secs)).await;
            None
        }
    }
}

fn watchdog_takeover_publish_config(
    relays: Vec<String>,
    name: Option<String>,
    region: Option<String>,
    interval_secs: u64,
    status_tx: Option<tokio::sync::watch::Sender<Option<PublishStateUpdate>>>,
) -> PublishLoopConfig {
    PublishLoopConfig {
        relays,
        name,
        region,
        max_clients: None,
        interval_secs,
        status_tx,
    }
}

fn solo_mesh_node_count(peers: &[crate::mesh::PeerInfo]) -> Option<usize> {
    let gpu_peers = non_client_peer_count(peers);
    (gpu_peers == 0).then_some(gpu_peers + 1)
}

async fn rejoin_larger_mesh_target(
    node: &crate::mesh::Node,
    publisher: &Publisher,
    target: &DiscoveredMesh,
    my_node_count: usize,
    interval_secs: u64,
) -> bool {
    eprintln!(
        "📡 Found larger mesh '{}' ({} nodes vs our {}) — rejoining",
        target.listing.name.as_deref().unwrap_or("unnamed"),
        target.listing.node_count,
        my_node_count
    );
    unpublish_before_rejoin(publisher).await;
    if join_larger_mesh(node, target).await.is_err() {
        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
        return true;
    }
    eprintln!("📡 Merged into mesh — resuming publish as member");
    tokio::time::sleep(Duration::from_secs(30)).await;
    true
}

async fn unpublish_before_rejoin(publisher: &Publisher) {
    if let Err(e) = publisher.unpublish().await {
        tracing::warn!("Failed to unpublish solo listing: {e}");
    }
}

async fn join_larger_mesh(node: &crate::mesh::Node, target: &DiscoveredMesh) -> Result<()> {
    node.join(&target.listing.invite_token).await.map_err(|e| {
        tracing::warn!("Merge/rejoin failed: {e}");
        e
    })
}

fn pick_larger_mesh_target<'a>(
    listings: &'a [DiscoveredMesh],
    my_npub: &str,
    my_mesh_id: Option<&str>,
    mesh_name: Option<&str>,
    my_node_count: usize,
) -> Option<&'a DiscoveredMesh> {
    let split_target = my_mesh_id.and_then(|mesh_id| {
        listings.iter().find(|mesh| {
            mesh.listing.mesh_id.as_deref() == Some(mesh_id)
                && mesh.publisher_npub != my_npub
                && mesh.listing.node_count > my_node_count
        })
    });
    split_target.or_else(|| {
        (mesh_name.is_none()).then(|| {
            listings.iter().find(|mesh| {
                mesh.publisher_npub != my_npub
                    && mesh.listing.name.is_none()
                    && mesh.listing.node_count > my_node_count
            })
        })?
    })
}

async fn build_publish_listing(
    node: &crate::mesh::Node,
    peers: &[crate::mesh::PeerInfo],
    invite_token: String,
    client_count: usize,
    max_clients: Option<usize>,
    name: Option<String>,
    region: Option<String>,
) -> MeshListing {
    let serving = collect_actually_serving_models(node, peers).await;
    let served_set: std::collections::HashSet<&str> = serving.iter().map(String::as_str).collect();
    let wanted = collect_wanted_models(node, &served_set).await;
    let on_disk = collect_available_models(node, peers, &served_set).await;
    let total_vram_bytes = peers
        .iter()
        .filter(|peer| !matches!(peer.role, crate::mesh::NodeRole::Client))
        .map(|peer| peer.vram_bytes)
        .sum::<u64>()
        + node.vram_bytes();
    let node_count = non_client_peer_count(peers) + 1;
    MeshListing {
        invite_token,
        serving,
        wanted,
        on_disk,
        total_vram_bytes,
        node_count,
        client_count,
        max_clients: max_clients.unwrap_or(0),
        name,
        region,
        mesh_id: node.mesh_id().await,
    }
}

async fn collect_actually_serving_models(
    node: &crate::mesh::Node,
    peers: &[crate::mesh::PeerInfo],
) -> Vec<String> {
    let mut serving = Vec::new();
    if matches!(node.role().await, crate::mesh::NodeRole::Host { .. }) {
        extend_unique(&mut serving, node.hosted_models().await);
    }
    for peer in peers {
        if matches!(peer.role, crate::mesh::NodeRole::Host { .. }) {
            extend_unique(&mut serving, peer.routable_models());
        }
    }
    serving
}

async fn collect_wanted_models(
    node: &crate::mesh::Node,
    served_set: &std::collections::HashSet<&str>,
) -> Vec<String> {
    let mut wanted = Vec::new();
    for model in node.active_demand().await.keys() {
        if !served_set.contains(model.as_str()) && !wanted.contains(model) {
            wanted.push(model.clone());
        }
    }
    wanted
}

async fn collect_available_models(
    node: &crate::mesh::Node,
    peers: &[crate::mesh::PeerInfo],
    served_set: &std::collections::HashSet<&str>,
) -> Vec<String> {
    let mut available = Vec::new();
    extend_unique_filtered(&mut available, node.available_models().await, served_set);
    for peer in peers {
        extend_unique_filtered(&mut available, peer.available_models.clone(), served_set);
    }
    available
}

fn extend_unique(into: &mut Vec<String>, values: Vec<String>) {
    for value in values {
        if !into.contains(&value) {
            into.push(value);
        }
    }
}

fn extend_unique_filtered(
    into: &mut Vec<String>,
    values: Vec<String>,
    served_set: &std::collections::HashSet<&str>,
) {
    for value in values {
        if !served_set.contains(value.as_str()) && !into.contains(&value) {
            into.push(value);
        }
    }
}

fn mesh_listing_present(
    meshes: &[DiscoveredMesh],
    mesh_id: Option<&str>,
    served: &[String],
) -> bool {
    if let Some(mesh_id) = mesh_id {
        return meshes
            .iter()
            .any(|mesh| mesh.listing.mesh_id.as_deref() == Some(mesh_id));
    }
    !served.is_empty()
        && meshes.iter().any(|mesh| {
            served
                .iter()
                .any(|model| mesh.listing.serving.contains(model))
        })
}

async fn update_delisted_state(
    publisher: &Publisher,
    max_clients: Option<usize>,
    client_count: usize,
    delisted: &mut bool,
    interval_secs: u64,
) -> bool {
    let Some(cap) = max_clients else {
        return false;
    };
    if client_count >= cap && !*delisted {
        if let Err(e) = publisher.unpublish().await {
            tracing::warn!("Failed to unpublish from Nostr: {e}");
        }
        eprintln!(
            "📡 Delisted from Nostr ({} clients, cap is {})",
            client_count, cap
        );
        *delisted = true;
        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
        return true;
    }
    if client_count < cap && *delisted {
        eprintln!(
            "📡 Re-publishing to Nostr ({} clients, cap is {})",
            client_count, cap
        );
        *delisted = false;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watchdog_takeover_publish_config_uses_configured_interval() {
        let config = watchdog_takeover_publish_config(
            vec!["wss://relay.example".to_string()],
            Some("mesh".to_string()),
            Some("iad".to_string()),
            17,
            None,
        );

        assert_eq!(config.interval_secs, 17);
        assert_eq!(config.name.as_deref(), Some("mesh"));
        assert_eq!(config.region.as_deref(), Some("iad"));
        assert_eq!(config.relays, vec!["wss://relay.example".to_string()]);
        assert_eq!(config.max_clients, None);
    }
}
