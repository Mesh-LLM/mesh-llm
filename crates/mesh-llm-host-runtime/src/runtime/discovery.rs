use crate::mesh;
use crate::network::{discovery as mesh_discovery, nostr};
use crate::runtime::RuntimeOptions;
use mesh_llm_events::{OutputEvent, emit_event};
use std::cmp::Reverse;

/// Health probe: try QUIC connect to the mesh's bootstrap node.
/// Returns Ok if reachable within 10s, Err if not.
/// Re-discover meshes via Nostr when all peers are lost.
/// Only runs for --auto nodes that originally discovered via Nostr.
/// Checks every 30s; if 0 peers for 90s straight, re-discovers and joins.
pub(super) async fn nostr_rediscovery(
    node: mesh::Node,
    nostr_relays: Vec<String>,
    _relay_urls: Vec<String>,
    mesh_name: Option<String>,
) {
    const CHECK_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);
    const GRACE_PERIOD: std::time::Duration = std::time::Duration::from_secs(90);

    tokio::time::sleep(std::time::Duration::from_secs(30)).await;

    let mut alone_since: Option<std::time::Instant> = None;

    loop {
        tokio::time::sleep(CHECK_INTERVAL).await;
        run_rediscovery_tick(
            &node,
            &nostr_relays,
            mesh_name.as_deref(),
            GRACE_PERIOD,
            &mut alone_since,
        )
        .await;
    }
}

/// Re-discover LAN meshes via mDNS when all peers are lost.
///
/// This is only useful when the operator supplied an invite token. The mDNS
/// advertisement intentionally carries a token fingerprint rather than the raw
/// token, so rediscovery remains LAN-local and token-gated.
pub(super) async fn lan_rediscovery(
    node: mesh::Node,
    supplied_join_tokens: Vec<String>,
    mesh_name: Option<String>,
    region: Option<String>,
) {
    if supplied_join_tokens.is_empty() {
        return;
    }

    const CHECK_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);
    const GRACE_PERIOD: std::time::Duration = std::time::Duration::from_secs(90);

    tokio::time::sleep(std::time::Duration::from_secs(30)).await;

    let mut alone_since: Option<std::time::Instant> = None;

    loop {
        tokio::time::sleep(CHECK_INTERVAL).await;
        run_lan_rediscovery_tick(
            &node,
            &supplied_join_tokens,
            mesh_name.as_deref(),
            region.as_deref(),
            GRACE_PERIOD,
            &mut alone_since,
        )
        .await;
    }
}

async fn run_lan_rediscovery_tick(
    node: &mesh::Node,
    supplied_join_tokens: &[String],
    mesh_name: Option<&str>,
    region: Option<&str>,
    grace_period: std::time::Duration,
    alone_since: &mut Option<std::time::Instant>,
) {
    if reset_rediscovery_timer_if_peers_recovered(node, alone_since, "mDNS LAN rediscovery").await {
        return;
    }

    if rediscovery_grace_period_active(alone_since, grace_period, "mDNS LAN rediscovery") {
        return;
    }

    let _ = emit_event(OutputEvent::DiscoveryStarting {
        source: "mDNS LAN re-discovery".to_string(),
    });

    let Some(candidates) =
        discover_lan_rediscovery_candidates(supplied_join_tokens, mesh_name, region, alone_since)
            .await
    else {
        return;
    };

    if candidates.is_empty() {
        report_no_lan_rediscovery_meshes(mesh_name, alone_since);
        return;
    }

    let ranked = rank_lan_rediscovery_candidates(&candidates);
    let our_mesh_id = node.mesh_id().await;
    if try_rejoin_rediscovery_candidates(node, &ranked, our_mesh_id.as_deref()).await {
        *alone_since = None;
    } else {
        report_rediscovery_retry(alone_since);
    }
}

async fn run_rediscovery_tick(
    node: &mesh::Node,
    nostr_relays: &[String],
    mesh_name: Option<&str>,
    grace_period: std::time::Duration,
    alone_since: &mut Option<std::time::Instant>,
) {
    if reset_rediscovery_timer_if_peers_recovered(node, alone_since, "Nostr rediscovery").await {
        return;
    }

    if rediscovery_grace_period_active(alone_since, grace_period, "Nostr rediscovery") {
        return;
    }

    let _ = emit_event(OutputEvent::DiscoveryStarting {
        source: "Nostr re-discovery".to_string(),
    });

    let Some(meshes) = discover_rediscovery_meshes(nostr_relays, alone_since).await else {
        return;
    };

    let filtered = filter_rediscovery_meshes(&meshes, mesh_name);
    if filtered.is_empty() {
        report_no_rediscovery_meshes(mesh_name, alone_since);
        return;
    }

    let candidates = rank_rediscovery_candidates(&filtered);
    let our_mesh_id = node.mesh_id().await;
    if try_rejoin_rediscovery_candidates(node, &candidates, our_mesh_id.as_deref()).await {
        *alone_since = None;
    } else {
        report_rediscovery_retry(alone_since);
    }
}

async fn reset_rediscovery_timer_if_peers_recovered(
    node: &mesh::Node,
    alone_since: &mut Option<std::time::Instant>,
    label: &str,
) -> bool {
    if node.peers().await.is_empty() {
        return false;
    }
    if alone_since.is_some() {
        tracing::debug!("{label}: peers recovered, resetting timer");
        *alone_since = None;
    }
    true
}

fn rediscovery_grace_period_active(
    alone_since: &mut Option<std::time::Instant>,
    grace_period: std::time::Duration,
    label: &str,
) -> bool {
    let now = std::time::Instant::now();
    let start = *alone_since.get_or_insert(now);
    let elapsed = now.duration_since(start);
    if elapsed >= grace_period {
        return false;
    }
    tracing::debug!(
        "{label}: 0 peers for {}s (grace: {}s)",
        elapsed.as_secs(),
        grace_period.as_secs()
    );
    true
}

async fn discover_rediscovery_meshes(
    nostr_relays: &[String],
    alone_since: &mut Option<std::time::Instant>,
) -> Option<Vec<nostr::DiscoveredMesh>> {
    let filter = nostr::MeshFilter::default();
    match nostr::discover(nostr_relays, &filter, None).await {
        Ok(meshes) => Some(meshes),
        Err(err) => {
            let _ = emit_event(OutputEvent::DiscoveryFailed {
                message: "Nostr re-discovery failed".to_string(),
                detail: Some(err.to_string()),
            });
            *alone_since = Some(std::time::Instant::now());
            None
        }
    }
}

async fn discover_lan_rediscovery_candidates(
    supplied_join_tokens: &[String],
    mesh_name: Option<&str>,
    region: Option<&str>,
    alone_since: &mut Option<std::time::Instant>,
) -> Option<Vec<(String, nostr::DiscoveredMesh)>> {
    let filter = nostr::MeshFilter {
        name: mesh_name.map(str::to_string),
        region: region.map(str::to_string),
        ..Default::default()
    };
    let mut candidates = Vec::new();
    for token in supplied_join_tokens
        .iter()
        .map(String::as_str)
        .filter(|token| !token.trim().is_empty())
    {
        match mesh_discovery::discover_lan_join_candidates(
            &filter,
            Some(token),
            std::time::Duration::from_secs(5),
        )
        .await
        {
            Ok(mut discovered) => candidates.append(&mut discovered),
            Err(err) => {
                let _ = emit_event(OutputEvent::DiscoveryFailed {
                    message: "mDNS LAN re-discovery failed".to_string(),
                    detail: Some(err.to_string()),
                });
                *alone_since = Some(std::time::Instant::now());
                return None;
            }
        }
    }
    dedupe_lan_rediscovery_candidates(&mut candidates);
    Some(candidates)
}

fn filter_rediscovery_meshes<'a>(
    meshes: &'a [nostr::DiscoveredMesh],
    mesh_name: Option<&str>,
) -> Vec<&'a nostr::DiscoveredMesh> {
    match mesh_name {
        Some(name) => meshes
            .iter()
            .filter(|mesh| rediscovery_mesh_name_matches(mesh, name))
            .collect(),
        None => meshes.iter().collect(),
    }
}

fn rediscovery_mesh_name_matches(mesh: &nostr::DiscoveredMesh, name: &str) -> bool {
    mesh.listing
        .name
        .as_ref()
        .map(|candidate| candidate.eq_ignore_ascii_case(name))
        .unwrap_or(false)
}

fn report_no_rediscovery_meshes(
    mesh_name: Option<&str>,
    alone_since: &mut Option<std::time::Instant>,
) {
    let name_hint = mesh_name.unwrap_or("any");
    let _ = emit_event(OutputEvent::DiscoveryFailed {
        message: format!("No meshes found on Nostr matching \"{name_hint}\" — will retry"),
        detail: None,
    });
    *alone_since = Some(std::time::Instant::now());
}

fn report_no_lan_rediscovery_meshes(
    mesh_name: Option<&str>,
    alone_since: &mut Option<std::time::Instant>,
) {
    let name_hint = mesh_name.unwrap_or("any");
    let _ = emit_event(OutputEvent::DiscoveryFailed {
        message: format!(
            "No joinable LAN meshes found via mDNS matching \"{name_hint}\" — will retry"
        ),
        detail: Some(
            "mDNS rediscovery only considers advertisements matching a supplied --join token"
                .to_string(),
        ),
    });
    *alone_since = Some(std::time::Instant::now());
}

fn rank_rediscovery_candidates<'a>(
    meshes: &[&'a nostr::DiscoveredMesh],
) -> Vec<(&'a nostr::DiscoveredMesh, i64)> {
    let now_ts = current_unix_secs();
    let last_mesh_id = mesh::load_last_mesh_id();
    let mut candidates: Vec<_> = meshes
        .iter()
        .map(|mesh| {
            (
                *mesh,
                nostr::score_mesh(mesh, now_ts, last_mesh_id.as_deref()),
            )
        })
        .collect();
    candidates.sort_by_key(|candidate| Reverse(candidate.1));
    candidates
}

fn rank_lan_rediscovery_candidates(
    candidates: &[(String, nostr::DiscoveredMesh)],
) -> Vec<(&nostr::DiscoveredMesh, i64)> {
    let meshes = candidates.iter().map(|(_, mesh)| mesh).collect::<Vec<_>>();
    rank_rediscovery_candidates(&meshes)
}

fn dedupe_lan_rediscovery_candidates(candidates: &mut Vec<(String, nostr::DiscoveredMesh)>) {
    let mut seen = std::collections::HashSet::new();
    candidates.retain(|(token, mesh)| {
        let key = (
            token.clone(),
            mesh.publisher_npub.clone(),
            mesh.listing.mesh_id.clone(),
        );
        seen.insert(key)
    });
}

async fn try_rejoin_rediscovery_candidates(
    node: &mesh::Node,
    candidates: &[(&nostr::DiscoveredMesh, i64)],
    our_mesh_id: Option<&str>,
) -> bool {
    for (mesh, _score) in candidates {
        if rediscovery_candidate_is_current_mesh(mesh, our_mesh_id) {
            continue;
        }
        if try_rejoin_rediscovery_mesh(node, mesh).await {
            return true;
        }
    }
    false
}

fn rediscovery_candidate_is_current_mesh(
    mesh: &nostr::DiscoveredMesh,
    our_mesh_id: Option<&str>,
) -> bool {
    match (our_mesh_id, mesh.listing.mesh_id.as_deref()) {
        (Some(ours), Some(theirs)) => ours == theirs,
        _ => false,
    }
}

async fn try_rejoin_rediscovery_mesh(node: &mesh::Node, mesh: &nostr::DiscoveredMesh) -> bool {
    let mesh_label = mesh
        .listing
        .name
        .as_deref()
        .unwrap_or("unnamed")
        .to_string();
    let _ = emit_event(OutputEvent::MeshFound {
        mesh: mesh_label.clone(),
        peers: mesh.listing.node_count,
        region: None,
    });
    match node.join(&mesh.listing.invite_token).await {
        Ok(()) => {
            let _ = emit_event(OutputEvent::DiscoveryJoined { mesh: mesh_label });
            true
        }
        Err(err) => {
            let _ = emit_event(OutputEvent::DiscoveryFailed {
                message: format!(
                    "Failed to re-join mesh {}",
                    mesh.listing.name.as_deref().unwrap_or("unnamed")
                ),
                detail: Some(err.to_string()),
            });
            false
        }
    }
}

fn report_rediscovery_retry(alone_since: &mut Option<std::time::Instant>) {
    let _ = emit_event(OutputEvent::DiscoveryFailed {
        message: "Could not re-join any mesh — will retry".to_string(),
        detail: None,
    });
    *alone_since = Some(std::time::Instant::now());
}

fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Helper for StartNew path — configure CLI to start a new mesh.
pub(super) fn start_new_mesh(
    options: &mut RuntimeOptions,
    models: &[String],
    my_vram_gb: f64,
    has_startup_models: bool,
) {
    let primary = models.first().cloned().unwrap_or_default();
    if !has_startup_models && options.model.is_empty() {
        options.model.push(primary.clone().into());
    }
    let detail = if has_startup_models {
        "using configured startup models".to_string()
    } else {
        format!("serving: {primary}")
    };
    let discovery = if options.publish {
        "publishing for discovery"
    } else {
        "mesh is private — add --publish to advertise it for discovery"
    };
    let _ = emit_event(OutputEvent::Info {
        message: format!(
            "Starting a new mesh — {detail} — capacity: {:.0}GB — {discovery}",
            my_vram_gb
        ),
        context: None,
    });
}

pub fn nostr_relays(cli_relays: &[String]) -> Vec<String> {
    if cli_relays.is_empty() {
        nostr::DEFAULT_RELAYS
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        cli_relays.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rediscovery_mesh(
        publisher: &str,
        mesh_id: Option<&str>,
        nodes: usize,
    ) -> nostr::DiscoveredMesh {
        nostr::DiscoveredMesh {
            listing: nostr::MeshListing {
                invite_token: "join-token".to_string(),
                serving: vec!["Qwen3-8B-Q4_K_M".to_string()],
                wanted: Vec::new(),
                on_disk: Vec::new(),
                total_vram_bytes: (nodes as u64) * 16_000_000_000,
                node_count: nodes,
                client_count: 0,
                max_clients: 4,
                name: Some("lab".to_string()),
                region: Some("LAN".to_string()),
                mesh_id: mesh_id.map(str::to_string),
            },
            publisher_npub: publisher.to_string(),
            published_at: current_unix_secs(),
            expires_at: None,
        }
    }

    #[test]
    fn lan_rediscovery_dedupes_same_token_publisher_and_mesh() {
        let duplicate = (
            "join-token".to_string(),
            rediscovery_mesh("mdns:mesh-a", Some("mesh-a"), 2),
        );
        let mut candidates = vec![
            duplicate.clone(),
            duplicate,
            (
                "join-token".to_string(),
                rediscovery_mesh("mdns:mesh-b", Some("mesh-b"), 2),
            ),
        ];

        dedupe_lan_rediscovery_candidates(&mut candidates);

        assert_eq!(candidates.len(), 2);
        assert!(
            candidates
                .iter()
                .any(|(_, mesh)| mesh.publisher_npub == "mdns:mesh-a")
        );
        assert!(
            candidates
                .iter()
                .any(|(_, mesh)| mesh.publisher_npub == "mdns:mesh-b")
        );
    }

    #[test]
    fn lan_rediscovery_ranks_joinable_candidates_by_existing_mesh_score() {
        let candidates = vec![
            (
                "join-token".to_string(),
                rediscovery_mesh("mdns:small", Some("mesh-small"), 1),
            ),
            (
                "join-token".to_string(),
                rediscovery_mesh("mdns:large", Some("mesh-large"), 4),
            ),
        ];

        let ranked = rank_lan_rediscovery_candidates(&candidates);

        assert_eq!(ranked[0].0.publisher_npub, "mdns:large");
    }
}
