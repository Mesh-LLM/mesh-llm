//! Nostr publishing loops and listing construction.

use super::contracts::{DiscoveredMesh, MESH_SERVICE_KIND, MeshListing};
use super::discovery::{DiscoveryClient, MeshFilter, RelayHealth, discover, discover_with_health};
#[cfg(test)]
use super::discovery::{RelayFetchResult, merge_relay_fetch_results};
use super::keys::load_or_create_keys;
use anyhow::Result;
use mesh_llm_events::{OutputEvent, emit_event};
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
        let client = Client::new();
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
            Tag::custom("d", vec!["mesh-llm".to_string()]),
            Tag::custom("k", vec!["mesh-llm".to_string()]),
            Tag::custom("expiration", vec![expiration.to_string()]),
        ];

        let event = EventBuilder::new(Kind::Custom(MESH_SERVICE_KIND), content)
            .tags(tags)
            .finalize(&self.keys)?;
        self.client.send_event(&event).await?;
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
            .fetch_events(filter)
            .timeout(Duration::from_secs(5))
            .await?;
        for event in events.iter() {
            let request = EventDeletionRequest::new().id(event.id);
            if let Ok(deletion) = request.finalize(&self.keys) {
                let _ = self.client.send_event(&deletion).await;
            }
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
    let _ = emit_event(OutputEvent::NostrPublishing {
        message: format!(
            "📡 Publishing mesh to Nostr (npub: {}...{})",
            &npub[..12],
            &npub[npub.len() - 8..]
        ),
    });

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
    loop {
        if !watchdog_takeover_confirmed(&node, &relays, disco.as_ref(), check_interval_secs).await {
            continue;
        }
        let _ = emit_event(OutputEvent::NostrPublishing {
            message: "📡 Taking over Nostr publishing for the mesh".to_string(),
        });
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

async fn watchdog_takeover_confirmed(
    node: &crate::mesh::Node,
    relays: &[String],
    disco: Option<&DiscoveryClient>,
    check_interval_secs: u64,
) -> bool {
    let filter = MeshFilter::default();
    let (meshes, health) = match discover_with_health(relays, &filter, disco).await {
        Ok(round) => round,
        Err(error) => {
            tracing::debug!("Publish watchdog: Nostr check failed: {error}");
            watchdog_next_check().await;
            return false;
        }
    };
    let mesh_id = node.mesh_id().await;
    if !should_take_over_publish(node, &meshes, health, mesh_id.as_deref()).await {
        watchdog_next_check().await;
        return false;
    }
    if !confirm_missing_listing_after_backoff(relays, &filter, disco, node, mesh_id.as_deref())
        .await
    {
        tokio::time::sleep(Duration::from_secs(check_interval_secs)).await;
        return false;
    }
    true
}

async fn watchdog_next_check() {
    // Check frequently so we catch gaps fast.
    let next_check = (rand::random::<u64>() % 15) + 20; // 20-35s
    tokio::time::sleep(Duration::from_secs(next_check)).await;
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
        let _ = emit_event(OutputEvent::NostrPublishing {
            message: format!("   Will delist when {} clients connected", cap),
        });
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

/// What a discovery round says about our own mesh's listing.
///
/// `Unknown` covers silent, interrupted and limit-truncated queries: the
/// watchdog must not publish a mesh on the strength of unproven absence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ListingPresence {
    /// A current listing for our mesh was seen.
    Present,
    /// At least one relay reached EOSE below the event limit and none
    /// had our listing. Partial or truncated results cannot prove absence.
    Missing,
    /// No relay completed an untruncated query; absence is not evidence.
    Unknown,
}

fn listing_presence(
    meshes: &[DiscoveredMesh],
    health: RelayHealth,
    mesh_id: Option<&str>,
    served: &[String],
) -> ListingPresence {
    if mesh_listing_present(meshes, mesh_id, served) {
        ListingPresence::Present
    } else if health.any_query_completed() {
        ListingPresence::Missing
    } else {
        ListingPresence::Unknown
    }
}

async fn should_take_over_publish(
    node: &crate::mesh::Node,
    meshes: &[DiscoveredMesh],
    health: RelayHealth,
    mesh_id: Option<&str>,
) -> bool {
    let our_peers = node.peers().await;
    let served = node.models_being_served().await;
    listing_presence(meshes, health, mesh_id, &served) == ListingPresence::Missing
        && (!our_peers.is_empty() || !served.is_empty())
}

async fn confirm_missing_listing_after_backoff(
    relays: &[String],
    filter: &MeshFilter,
    disco: Option<&DiscoveryClient>,
    node: &crate::mesh::Node,
    checked_mesh_id: Option<&str>,
) -> bool {
    let backoff = (rand::random::<u64>() % 7) + 3;
    let _ = emit_event(OutputEvent::NostrPublishing {
        message: format!(
            "📡 Mesh listing missing from Nostr — waiting {backoff}s before taking over..."
        ),
    });
    tokio::time::sleep(Duration::from_secs(backoff)).await;

    let recheck = match discover_with_health(relays, filter, disco).await {
        Ok(round) => round,
        // The recheck could not even build a client — the same
        // cannot-tell state as zero relays answering: skip this
        // takeover round rather than publish on an outage.
        Err(_) => return false,
    };
    let served = node.models_being_served().await;
    let our_mesh_id = node.mesh_id().await;
    match listing_presence_after_backoff(
        &recheck.0,
        recheck.1,
        checked_mesh_id,
        our_mesh_id.as_deref(),
        &served,
    ) {
        ListingPresence::Missing => true,
        ListingPresence::Present => {
            let _ = emit_event(OutputEvent::NostrPublishing {
                message: "📡 Someone else took over publishing — standing down".to_string(),
            });
            false
        }
        // An unfinished or limit-truncated query cannot establish absence.
        ListingPresence::Unknown => {
            tracing::debug!(
                "Publish watchdog: absence remains unconfirmed after backoff — not taking over"
            );
            false
        }
    }
}

fn listing_presence_after_backoff(
    meshes: &[DiscoveredMesh],
    health: RelayHealth,
    checked_mesh_id: Option<&str>,
    current_mesh_id: Option<&str>,
    served: &[String],
) -> ListingPresence {
    if current_mesh_id != checked_mesh_id {
        tracing::debug!("Publish watchdog: mesh ID changed during backoff — retrying next round");
        return ListingPresence::Unknown;
    }
    listing_presence(meshes, health, current_mesh_id, served)
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
    let _ = emit_event(OutputEvent::NostrPublishing {
        message: format!(
            "📡 Found larger mesh '{}' ({} nodes vs our {}) — rejoining",
            target.listing.name.as_deref().unwrap_or("unnamed"),
            target.listing.node_count,
            my_node_count
        ),
    });
    unpublish_before_rejoin(publisher).await;
    if join_larger_mesh(node, target).await.is_err() {
        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
        return true;
    }
    let _ = emit_event(OutputEvent::NostrPublishing {
        message: "📡 Merged into mesh — resuming publish as member".to_string(),
    });
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
    // A thin client tunnels HTTP to the invite endpoint, which answers only
    // when this node itself serves a model; a client or an idle worker
    // replies 503 (#2765).
    let publisher_serves = matches!(node.role().await, crate::mesh::NodeRole::Host { .. })
        && !node.hosted_models().await.is_empty();
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
        publisher_serves: Some(publisher_serves),
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
            extend_unique(&mut serving, crate::mesh::routable_models(peer));
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
        let _ = emit_event(OutputEvent::NostrPublishing {
            message: format!(
                "📡 Delisted from Nostr ({} clients, cap is {})",
                client_count, cap
            ),
        });
        *delisted = true;
        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
        return true;
    }
    if client_count < cap && *delisted {
        let _ = emit_event(OutputEvent::NostrPublishing {
            message: format!(
                "📡 Re-publishing to Nostr ({} clients, cap is {})",
                client_count, cap
            ),
        });
        *delisted = false;
    }
    false
}

#[cfg(test)]
mod listing_presence_tests {
    use super::*;
    use crate::network::nostr::MeshListing;

    fn mesh_with_id(mesh_id: Option<&str>, serving: &[&str]) -> DiscoveredMesh {
        DiscoveredMesh {
            listing: MeshListing {
                invite_token: "t".to_string(),
                serving: serving.iter().map(|s| s.to_string()).collect(),
                wanted: vec![],
                on_disk: vec![],
                total_vram_bytes: 0,
                node_count: 1,
                client_count: 0,
                max_clients: 0,
                name: None,
                region: None,
                mesh_id: mesh_id.map(str::to_string),
                publisher_serves: None,
            },
            publisher_npub: "npub-x".to_string(),
            published_at: 0,
            expires_at: None,
        }
    }

    fn health(answered: usize, completed: usize) -> RelayHealth {
        RelayHealth {
            relays_queried: 5,
            relays_answered: answered,
            queries_completed: completed,
        }
    }

    /// The incomplete-query hole: a relay delivered an event for
    /// another mesh and disconnected before EOSE, and nothing else
    /// completed. Absence inference must not rest on partial speech.
    #[test]
    fn an_unrelated_event_without_a_completed_query_is_unknown_not_missing() {
        let meshes = vec![mesh_with_id(Some("other"), &[])];
        // Spoke (delivered an event) but never finished the query.
        let partial = listing_presence(&meshes, health(1, 0), Some("abc"), &[]);
        assert_eq!(partial, ListingPresence::Unknown);
        // The same round with a completed query elsewhere is a verdict.
        let completed = listing_presence(&meshes, health(2, 1), Some("abc"), &[]);
        assert_eq!(completed, ListingPresence::Missing);
    }

    #[test]
    fn backoff_recheck_requires_the_same_mesh_identity() {
        for (checked, current) in [
            (Some("old"), Some("new")),
            (None, Some("new")),
            (Some("old"), None),
        ] {
            assert_eq!(
                listing_presence_after_backoff(&[], health(1, 1), checked, current, &[]),
                ListingPresence::Unknown,
            );
        }
        for id in [None, Some("same")] {
            assert_eq!(
                listing_presence_after_backoff(&[], health(1, 1), id, id, &[]),
                ListingPresence::Missing,
            );
        }
    }

    #[test]
    fn matching_mesh_id_means_present() {
        let meshes = vec![mesh_with_id(Some("abc"), &[])];
        assert_eq!(
            listing_presence(&meshes, health(5, 5), Some("abc"), &[]),
            ListingPresence::Present
        );
    }

    #[test]
    fn absence_with_answers_means_missing() {
        let meshes = vec![mesh_with_id(Some("other"), &[])];
        assert_eq!(
            listing_presence(&meshes, health(5, 5), Some("abc"), &[]),
            ListingPresence::Missing
        );
        // Empty results with a live relay are still a verdict.
        assert_eq!(
            listing_presence(&[], health(1, 1), Some("abc"), &[]),
            ListingPresence::Missing
        );
    }

    #[test]
    fn absence_with_no_answers_is_unknown_not_missing() {
        let meshes = vec![mesh_with_id(Some("other"), &[])];
        assert_eq!(
            listing_presence(&meshes, health(0, 0), Some("abc"), &[]),
            ListingPresence::Unknown
        );
    }

    #[test]
    fn unknown_presence_never_satisfies_the_takeover_gate_even_with_peers() {
        // `should_take_over_publish` requires Missing AND (peers OR
        // served). An outage round is Unknown no matter how healthy the
        // mesh is, so the conjunction must fail on the presence side
        // alone — this is the property that keeps the watchdog from
        // taking over during relay outages.
        let meshes = vec![mesh_with_id(Some("other"), &[])];
        let outage = listing_presence(&meshes, health(0, 0), Some("abc"), &[]);
        assert_ne!(outage, ListingPresence::Missing);
        assert_eq!(outage, ListingPresence::Unknown);

        // And the errored-fetch path feeds the same Unknown: a round
        // where every fetch errored (requests never issued) has zero
        // answered relays, not a quorum of speaking ones.
        let all_errored = (0..5).map(|_| RelayFetchResult::Silent).collect::<Vec<_>>();
        let (_, health) = merge_relay_fetch_results(5, all_errored);
        assert!(!health.any_relay_answered());
    }

    #[test]
    fn legacy_served_model_fallback_still_detects_presence() {
        let meshes = vec![mesh_with_id(None, &["Qwen3-8B-Q4_K_M"])];
        let served = vec!["Qwen3-8B-Q4_K_M".to_string()];
        assert_eq!(
            listing_presence(&meshes, health(3, 3), None, &served),
            ListingPresence::Present
        );
    }
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

    #[tokio::test]
    async fn a_listing_says_whether_its_publisher_serves_a_model_itself() {
        async fn publisher_serves(role: crate::mesh::NodeRole, hosted: &[&str]) -> Option<bool> {
            let node = crate::mesh::Node::new_for_tests(role).await.unwrap();
            node.set_hosted_models(hosted.iter().map(|model| model.to_string()).collect())
                .await;
            build_publish_listing(&node, &[], "invite".into(), 0, None, None, None)
                .await
                .publisher_serves
        }
        let host = crate::mesh::NodeRole::Host { http_port: 9337 };
        assert_eq!(
            publisher_serves(host.clone(), &["Qwen3-0.6B"]).await,
            Some(true)
        );
        // A host that has nothing loaded, a worker and a client all answer a
        // tunnelled request with 503, so they say so.
        assert_eq!(publisher_serves(host, &[]).await, Some(false));
        assert_eq!(
            publisher_serves(crate::mesh::NodeRole::Worker, &[]).await,
            Some(false)
        );
        assert_eq!(
            publisher_serves(crate::mesh::NodeRole::Client, &[]).await,
            Some(false)
        );
    }
}
