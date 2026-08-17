//! Raw-multicast LAN beacon for relay-less direct-path bootstrapping.
//!
//! On multi-homed hosts (many utun/VPN interfaces) both the mDNS service
//! daemon and iroh's relay-less initial handshake can fail to traverse the LAN,
//! because they rely on per-packet source selection that the macOS kernel
//! routes onto the wrong interface. A plain UDP socket *bound to the LAN IP*
//! with `IP_MULTICAST_IF` pinned to that interface reaches LAN peers reliably.
//!
//! This beacon uses exactly that reliable mechanism, independent of mDNS:
//! every node in mDNS mode periodically multicasts its own reachable
//! `EndpointAddr` (plus mesh id) on a dedicated group/port, and listens for
//! peers' beacons. On hearing a peer it is not connected to, it dials that
//! peer's advertised address — the single-homed → multi-homed direction that
//! works. `connect_to_peer` is idempotent, so whichever side connects first
//! wins and duplicates are harmless.
//!
//! The beacon carries no trust-bearing material: only an endpoint id, LAN
//! addresses, and a mesh-id fingerprint. Admission is still enforced by the
//! mesh handshake. A node only dials peers advertising the same mesh id (or an
//! unknown mesh id, to allow first contact before mesh ids converge).

use std::net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use socket2::{Domain, Protocol, Socket, Type};
use tokio::net::UdpSocket;

use crate::mesh;

/// Dedicated multicast group + port for the LAN direct-path beacon.
///
/// `224.0.0.251` is the IANA-assigned mDNS link-local multicast group. Reusing
/// that group on the distinct mesh-llm beacon port keeps packets on the local
/// segment without interacting with mDNS responders on 5353.
const BEACON_GROUP: Ipv4Addr = Ipv4Addr::new(224, 0, 0, 251);
const BEACON_PORT: u16 = 47654;
/// How often to emit our beacon.
const BEACON_INTERVAL: Duration = Duration::from_secs(5);
/// Beacon wire-format version, for forward compatibility.
const BEACON_VERSION: u8 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BeaconMessage {
    v: u8,
    /// Publisher endpoint id (canonical string).
    id: String,
    /// Publisher mesh id, if known.
    mesh_id: Option<String>,
    /// Base64url-JSON of the publisher's `EndpointAddr` (LAN-filtered).
    addr: String,
}

/// Spawn the LAN beacon (sender + listener) for a node in mDNS mode.
///
/// Returns the spawned task's [`JoinHandle`](tokio::task::JoinHandle) so the
/// runtime can abort the beacon (and release its UDP socket) during shutdown.
pub(crate) fn spawn(node: mesh::Node) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(err) = run(node).await {
            tracing::debug!("LAN beacon stopped: {err:#}");
        }
    })
}

async fn run(node: mesh::Node) -> Result<()> {
    let recv_sock = bind_beacon_listener().context("bind LAN beacon listener")?;
    let mut buf = vec![0u8; 4096];
    let mut send_tick = tokio::time::interval(BEACON_INTERVAL);

    tracing::debug!(
        "LAN beacon active on {}:{} (self={})",
        BEACON_GROUP,
        BEACON_PORT,
        node.id().fmt_short()
    );

    loop {
        tokio::select! {
            _ = send_tick.tick() => on_send_tick(&node).await,
            res = recv_sock.recv_from(&mut buf) => on_recv(&node, res, &buf).await,
        }
    }
}

async fn on_send_tick(node: &mesh::Node) {
    if let Err(err) = emit_beacon(node).await {
        tracing::trace!("LAN beacon emit failed: {err:#}");
    }
}

async fn on_recv(node: &mesh::Node, res: std::io::Result<(usize, SocketAddr)>, buf: &[u8]) {
    match res {
        Ok((n, _from)) => handle_beacon(node, &buf[..n]).await,
        Err(err) => tracing::trace!("LAN beacon recv error: {err}"),
    }
}

/// Bind a multicast listener socket joined on all relevant interfaces.
fn bind_beacon_listener() -> Result<UdpSocket> {
    let sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
    sock.set_reuse_address(true)?;
    #[cfg(unix)]
    sock.set_reuse_port(true)?;
    sock.bind(&SocketAddr::from((Ipv4Addr::UNSPECIFIED, BEACON_PORT)).into())?;
    // Join the group on the unspecified interface; the kernel joins on the
    // default interface, which is sufficient for receiving on the LAN.
    sock.join_multicast_v4(&BEACON_GROUP, &Ipv4Addr::UNSPECIFIED)?;
    sock.set_nonblocking(true)?;
    let std_sock: std::net::UdpSocket = sock.into();
    Ok(UdpSocket::from_std(std_sock)?)
}

/// Emit our beacon: multicast (best effort) plus a direct unicast to every
/// known peer's LAN address.
///
/// On multi-homed macOS hosts an in-process multicast send can fail with
/// EHOSTUNREACH even though the route table is correct, while a plain unicast to
/// a known LAN address routes fine. So the unicast path is the reliable carrier:
/// a joiner already knows the host's address (from the invite token / gossip),
/// so it can unicast its own `EndpointAddr` straight to the host, which then
/// dials back on the working direction.
async fn emit_beacon(node: &mesh::Node) -> Result<()> {
    let addr = node.advertised_endpoint_addr();
    let has_v4 = addr
        .ip_addrs()
        .any(|a| matches!(a.ip(), IpAddr::V4(v4) if !v4.is_loopback() && !v4.is_unspecified()));
    if !has_v4 {
        return Ok(());
    }

    let msg = BeaconMessage {
        v: BEACON_VERSION,
        id: node.id().to_string(),
        mesh_id: node.mesh_id().await,
        addr: encode_endpoint_addr(&addr).context("encode endpoint addr")?,
    };
    let payload = serde_json::to_vec(&msg)?;
    let mcast = SocketAddrV4::new(BEACON_GROUP, BEACON_PORT);
    // Unicast to known peers and to join targets (invite-token addresses we may
    // not have connected to yet — the key case for a multi-homed joiner that
    // cannot complete its own outbound QUIC handshake).
    let mut peers = node.known_peer_lan_ipv4().await;
    peers.extend(node.join_target_lan_ipv4().await);
    peers.sort();
    peers.dedup();
    // Beacon to the peer's beacon port, not its QUIC port.
    for p in peers.iter_mut() {
        p.set_port(BEACON_PORT);
    }

    if let Err(err) =
        tokio::task::spawn_blocking(move || emit_blocking(mcast, &peers, &payload)).await
    {
        tracing::warn!(%err, "LAN beacon emit task failed");
    }
    Ok(())
}

/// Send the beacon synchronously: best-effort multicast plus unicast to each
/// known peer LAN address, all on plain unbound sockets (no interface pins,
/// which trigger in-process EHOSTUNREACH on multi-homed macOS hosts).
fn emit_blocking(mcast: SocketAddrV4, peers: &[SocketAddrV4], payload: &[u8]) {
    if let Err(err) = send_multicast(mcast, payload) {
        tracing::trace!("LAN beacon multicast failed: {err}");
    }
    // Each send opens a short-lived socket: beacon traffic is tiny, and avoiding
    // shared multicast socket state keeps interface pins out of unicast sends on
    // multi-homed hosts.
    for peer in peers {
        let res = send_unicast(*peer, payload);
        tracing::trace!("LAN beacon unicast to {peer}: {res:?}");
    }
}

fn send_multicast(dst: SocketAddrV4, payload: &[u8]) -> Result<()> {
    let sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
    sock.set_multicast_ttl_v4(1)?;
    sock.send_to(payload, &SocketAddr::V4(dst).into())?;
    Ok(())
}

fn send_unicast(dst: SocketAddrV4, payload: &[u8]) -> Result<()> {
    let sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
    sock.send_to(payload, &SocketAddr::V4(dst).into())?;
    Ok(())
}

/// Handle a received beacon: dial the peer back if appropriate.
async fn handle_beacon(node: &mesh::Node, payload: &[u8]) {
    let Some((peer_id, addr)) = parse_beacon(node, payload).await else {
        return;
    };
    if node.connected_peer_ids().await.contains(&peer_id) {
        return;
    }
    tracing::info!(
        "LAN beacon: dialing peer {} on advertised LAN address",
        peer_id.fmt_short()
    );
    if let Err(err) = node.dial_peer_addr(addr).await {
        tracing::debug!(
            "LAN beacon dial to {} failed (will retry): {err}",
            peer_id.fmt_short()
        );
    }
}

/// Validate and decode a beacon into a dialable peer, applying mesh-id and
/// self filtering. Returns `None` if the beacon should be ignored.
async fn parse_beacon(
    node: &mesh::Node,
    payload: &[u8],
) -> Option<(iroh::EndpointId, iroh::EndpointAddr)> {
    let msg: BeaconMessage = serde_json::from_slice(payload).ok()?;
    if msg.v != BEACON_VERSION {
        return None;
    }
    let addr = decode_endpoint_addr(&msg.addr)?;
    if addr.id == node.id() {
        return None;
    }
    // Only dial peers in our mesh. Allow unknown/absent mesh ids so the first
    // contact can happen before mesh ids are exchanged.
    if let (Some(ours), Some(theirs)) = (node.mesh_id().await, msg.mesh_id.as_ref())
        && &ours != theirs
    {
        return None;
    }
    Some((addr.id, addr))
}

fn encode_endpoint_addr(addr: &iroh::EndpointAddr) -> Option<String> {
    use base64::Engine;
    let json = serde_json::to_vec(addr).ok()?;
    Some(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json))
}

fn decode_endpoint_addr(value: &str) -> Option<iroh::EndpointAddr> {
    use base64::Engine;
    let raw = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(value)
        .ok()?;
    serde_json::from_slice(&raw).ok()
}
