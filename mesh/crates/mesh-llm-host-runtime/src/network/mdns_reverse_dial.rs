//! mDNS reverse-dial: bounded LAN dial-back for relay-less direct paths.
//!
//! In relay-less (mDNS) mode a direct connection is established by the joiner
//! dialing the host. On a multi-homed host (many interfaces, e.g. VPN/utun) the
//! joiner's own QUIC initiator path can fail to traverse even though the OS
//! network path and addresses are correct, leaving the connection stuck.
//!
//! The opposite direction works reliably: a single-homed peer dialing the
//! multi-homed peer establishes a clean LAN direct path. This loop exploits
//! that: every node in mDNS mode publishes its own reachable `EndpointAddr` in
//! its mDNS advert (additive `ep_addr` TXT key), and every node periodically
//! browses the LAN and dials back any advertised peer it is not already
//! connected to. Whichever direction succeeds first wins; `connect_to_peer`
//! is idempotent and skips peers that are already connected.

use std::collections::HashSet;
use std::time::Duration;

use crate::mesh;
use crate::network::discovery as mesh_discovery;
use crate::network::nostr;

/// How often to browse the LAN and attempt reverse-dials.
const REVERSE_DIAL_INTERVAL: Duration = Duration::from_secs(10);
/// How long each browse is allowed to collect advertisements.
const BROWSE_TIMEOUT: Duration = Duration::from_secs(3);

/// Runs the mDNS reverse-dial loop until the node shuts down.
///
/// Bounded: one browse per tick, at most one dial attempt per discovered peer
/// per tick, and only for peers not already connected. Safe to run on both the
/// host and the joiner — the idempotent connect makes double-dialing harmless.
pub(crate) async fn run_loop(node: mesh::Node, mesh_name: Option<String>, region: Option<String>) {
    let self_id = node.id();
    tracing::debug!(
        "mDNS reverse-dial loop started (self={})",
        self_id.fmt_short()
    );
    loop {
        tokio::time::sleep(REVERSE_DIAL_INTERVAL).await;
        reverse_dial_tick(&node, self_id, mesh_name.as_deref(), region.as_deref()).await;
    }
}

async fn reverse_dial_tick(
    node: &mesh::Node,
    self_id: iroh::EndpointId,
    mesh_name: Option<&str>,
    region: Option<&str>,
) {
    let discovered = browse_lan(node, mesh_name, region).await;
    let connected: HashSet<iroh::EndpointId> = node.connected_peer_ids().await;

    for mesh_advert in &discovered {
        if let Some(addr) = dial_target(mesh_advert, self_id, &connected) {
            dial_back(node, addr).await;
        }
    }
}

/// Browse the LAN for mesh advertisements, pinned to the node's bound LAN
/// interface. Returns an empty list on error.
async fn browse_lan(
    node: &mesh::Node,
    mesh_name: Option<&str>,
    region: Option<&str>,
) -> Vec<mesh_discovery::LanDiscoveredMesh> {
    let filter = nostr::MeshFilter {
        name: mesh_name.map(str::to_string),
        region: region.map(str::to_string),
        ..Default::default()
    };
    let lan_ip = node
        .advertised_endpoint_addr()
        .ip_addrs()
        .map(|addr| addr.ip())
        .find(|ip| ip.is_ipv4());

    match mesh_discovery::discover_lan_on_interface(&filter, None, BROWSE_TIMEOUT, lan_ip).await {
        Ok(meshes) => {
            tracing::debug!(
                "mDNS reverse-dial browse (lan_ip={lan_ip:?}) found {} advert(s)",
                meshes.len()
            );
            meshes
        }
        Err(err) => {
            tracing::debug!("mDNS reverse-dial browse failed: {err}");
            Vec::new()
        }
    }
}

/// Returns the peer's advertised dial-back address if it is a new peer worth
/// dialing (not ourselves, not already connected, and carrying an `ep_addr`).
fn dial_target(
    mesh_advert: &mesh_discovery::LanDiscoveredMesh,
    self_id: iroh::EndpointId,
    connected: &HashSet<iroh::EndpointId>,
) -> Option<iroh::EndpointAddr> {
    let addr = mesh_advert.endpoint_addr()?;
    if addr.id == self_id || connected.contains(&addr.id) {
        return None;
    }
    Some(addr.clone())
}

async fn dial_back(node: &mesh::Node, addr: iroh::EndpointAddr) {
    tracing::info!(
        "mDNS reverse-dial: dialing peer {} on advertised LAN address",
        addr.id.fmt_short()
    );
    if let Err(err) = node.dial_peer_addr(addr.clone()).await {
        tracing::debug!(
            "mDNS reverse-dial to {} failed (will retry): {err}",
            addr.id.fmt_short()
        );
    }
}
