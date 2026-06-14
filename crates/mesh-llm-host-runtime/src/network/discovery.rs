use anyhow::{Context, Result};
use mdns_sd::{DaemonStatus, ResolvedService, ServiceDaemon, ServiceEvent, ServiceInfo};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::Duration;

pub(crate) use crate::discovery::{DiscoveryScope, MeshDiscoveryMode};
use crate::network::nostr;

pub const LAN_SERVICE_TYPE: &str = "_mesh-llm._tcp.local.";
pub(crate) const LAN_DETAILS_PATH: &str = "/api/discovery/lan-details";
const TXT_SCHEMA_VERSION: u8 = 1;
const TXT_LIST_SEPARATOR: char = '|';
const TXT_VALUE_LIMIT: usize = 220;
const LAN_DETAILS_CHALLENGE_WINDOW_SECS: u64 = 300;
const LAN_INVITE_TOKEN_FINGERPRINT_DOMAIN: &[u8] = b"mesh-llm-lan-invite-token-v1\0";
const LAN_DETAILS_CHALLENGE_DOMAIN: &[u8] = b"mesh-llm-lan-details-challenge-v1\0";
const LAN_DETAILS_TOKEN_PROOF_DOMAIN: &[u8] = b"mesh-llm-lan-details-proof-v1\0";
const DAEMON_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LanJoinMaterial {
    RequiresSuppliedToken,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct LanMeshAdvertisement {
    pub(crate) mesh_id: Option<String>,
    pub(crate) mesh_name: Option<String>,
    pub(crate) region: Option<String>,
    pub(crate) serving_summary: Vec<String>,
    pub(crate) wanted_summary: Vec<String>,
    pub(crate) on_disk_summary: Vec<String>,
    pub(crate) total_vram_bytes: u64,
    pub(crate) node_count: usize,
    pub(crate) client_count: usize,
    pub(crate) max_clients: usize,
    pub token_fingerprint: Option<String>,
    pub(crate) details_path: Option<String>,
    pub(crate) proof_challenge: Option<String>,
    pub(crate) app_version: Option<String>,
    pub join_material: LanJoinMaterial,
    /// Base64url-encoded JSON of the publisher's own [`iroh::EndpointAddr`],
    /// filtered to its bound LAN interface. Additive (TXT key `ep_addr`):
    /// older nodes ignore it. Lets a peer dial the publisher back directly,
    /// which is the working direction when a multi-homed node cannot initiate
    /// a relay-less direct connection itself.
    pub(crate) endpoint_addr_b64: Option<String>,
}

impl LanMeshAdvertisement {
    pub(crate) fn from_listing(
        listing: &nostr::MeshListing,
        supplied_invite_token: Option<&str>,
        app_version: Option<&str>,
        details_reachable: bool,
    ) -> Self {
        // LAN discovery intentionally publishes only a fingerprint of the join
        // token so mDNS remains an untrusted pointer surface rather than a
        // transport for trust-bearing bootstrap material.
        let token_fingerprint = supplied_invite_token
            .filter(|token| !token.trim().is_empty())
            .map(lan_token_fingerprint)
            .or_else(|| {
                (!listing.invite_token.trim().is_empty())
                    .then(|| lan_token_fingerprint(&listing.invite_token))
            });
        let proof_challenge = if details_reachable {
            token_fingerprint
                .as_deref()
                .map(|fingerprint| lan_details_challenge(fingerprint, current_unix_secs()))
        } else {
            None
        };
        let details_path = proof_challenge
            .as_ref()
            .map(|_| LAN_DETAILS_PATH.to_string());

        Self {
            mesh_id: listing.mesh_id.clone(),
            mesh_name: listing.name.clone(),
            region: listing.region.clone(),
            serving_summary: bounded_list(&listing.serving),
            wanted_summary: bounded_list(&listing.wanted),
            on_disk_summary: bounded_list(&listing.on_disk),
            total_vram_bytes: listing.total_vram_bytes,
            node_count: listing.node_count,
            client_count: listing.client_count,
            max_clients: listing.max_clients,
            token_fingerprint,
            details_path,
            proof_challenge,
            app_version: app_version.map(str::to_owned),
            join_material: LanJoinMaterial::RequiresSuppliedToken,
            endpoint_addr_b64: None,
        }
    }

    /// Attach the publisher's own reachable [`EndpointAddr`] so peers can dial
    /// it back directly (mDNS reverse-dial). Encoded as base64url JSON under the
    /// additive `ep_addr` TXT key.
    pub(crate) fn with_endpoint_addr(mut self, addr: &iroh::EndpointAddr) -> Self {
        self.endpoint_addr_b64 = encode_endpoint_addr_b64(addr);
        self
    }

    /// Decode the publisher's advertised [`EndpointAddr`], if present and valid.
    pub(crate) fn endpoint_addr(&self) -> Option<iroh::EndpointAddr> {
        self.endpoint_addr_b64
            .as_deref()
            .and_then(decode_endpoint_addr_b64)
    }

    pub(crate) fn matches_supplied_token(&self, supplied_invite_token: Option<&str>) -> bool {
        let Some(expected) = self.token_fingerprint.as_deref() else {
            return false;
        };
        supplied_invite_token
            .filter(|token| !token.trim().is_empty())
            .map(lan_token_fingerprint)
            .as_deref()
            == Some(expected)
    }

    pub(crate) fn to_txt_properties(&self) -> Result<Vec<(String, String)>> {
        let mut txt = vec![
            ("svc".to_string(), "mesh-llm".to_string()),
            ("schema".to_string(), TXT_SCHEMA_VERSION.to_string()),
            ("join".to_string(), "token-fingerprint".to_string()),
            ("nodes".to_string(), self.node_count.to_string()),
            ("clients".to_string(), self.client_count.to_string()),
            ("max_clients".to_string(), self.max_clients.to_string()),
            ("vram".to_string(), self.total_vram_bytes.to_string()),
            ("serving".to_string(), pack_txt_list(&self.serving_summary)),
            ("wanted".to_string(), pack_txt_list(&self.wanted_summary)),
            ("on_disk".to_string(), pack_txt_list(&self.on_disk_summary)),
        ];
        push_optional_txt(&mut txt, "mesh_id", self.mesh_id.as_deref());
        push_optional_txt(&mut txt, "name", self.mesh_name.as_deref());
        push_optional_txt(&mut txt, "region", self.region.as_deref());
        push_optional_txt(&mut txt, "tok_fp", self.token_fingerprint.as_deref());
        push_optional_txt(&mut txt, "details", self.details_path.as_deref());
        push_optional_txt(&mut txt, "proof_challenge", self.proof_challenge.as_deref());
        push_optional_txt(&mut txt, "version", self.app_version.as_deref());
        push_optional_txt(&mut txt, "ep_addr", self.endpoint_addr_b64.as_deref());

        for (key, value) in &txt {
            anyhow::ensure!(
                key.len() + value.len() < u8::MAX as usize,
                "mDNS TXT property '{key}' exceeds DNS-SD length limit"
            );
        }
        Ok(txt)
    }

    #[cfg(test)]
    pub(crate) fn from_txt_properties(properties: &[(String, String)]) -> Result<Self> {
        let props: HashMap<&str, &str> = properties
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();

        parse_txt_properties(&props)
    }

    fn from_resolved_service(service: &ResolvedService) -> Result<Self> {
        let props = [
            "svc",
            "schema",
            "join",
            "nodes",
            "clients",
            "max_clients",
            "vram",
            "serving",
            "wanted",
            "on_disk",
            "mesh_id",
            "name",
            "region",
            "tok_fp",
            "details",
            "proof_challenge",
            "version",
            "ep_addr",
        ]
        .into_iter()
        .filter_map(|key| service.get_property_val_str(key).map(|value| (key, value)))
        .collect::<HashMap<_, _>>();

        parse_txt_properties(&props)
    }

    fn sanitized_listing(&self) -> nostr::MeshListing {
        nostr::MeshListing {
            // LAN discovery never republishes the actual join token.
            invite_token: String::new(),
            serving: self.serving_summary.clone(),
            wanted: self.wanted_summary.clone(),
            on_disk: self.on_disk_summary.clone(),
            total_vram_bytes: self.total_vram_bytes,
            node_count: self.node_count,
            client_count: self.client_count,
            max_clients: self.max_clients,
            name: self.mesh_name.clone(),
            region: self.region.clone(),
            mesh_id: self.mesh_id.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LanDiscoveredMesh {
    pub mode: &'static str,
    pub scope: DiscoveryScope,
    pub source: &'static str,
    pub service_type: &'static str,
    pub instance_name: String,
    pub host: String,
    pub port: u16,
    pub addresses: Vec<String>,
    pub listing: nostr::MeshListing,
    pub(crate) token_fingerprint: Option<String>,
    pub(crate) details_path: Option<String>,
    pub(crate) proof_challenge: Option<String>,
    pub(crate) join_material: LanJoinMaterial,
    pub joinable_with_supplied_token: bool,
    pub published_version: Option<String>,
    pub discovered_at: u64,
    #[serde(skip)]
    join_token: Option<String>,
    /// Publisher's own dial-back [`EndpointAddr`] (from the additive `ep_addr`
    /// TXT key), if advertised. Used by mDNS reverse-dial.
    #[serde(skip)]
    endpoint_addr: Option<iroh::EndpointAddr>,
}

impl LanDiscoveredMesh {
    pub fn join_token(&self) -> Option<&str> {
        self.join_token.as_deref()
    }

    /// The publisher's advertised dial-back address, if present.
    pub fn endpoint_addr(&self) -> Option<&iroh::EndpointAddr> {
        self.endpoint_addr.as_ref()
    }

    pub(crate) fn to_join_candidate(&self) -> Option<(String, nostr::DiscoveredMesh)> {
        let token = self.join_token.clone()?;
        let mut listing = self.listing.clone();
        listing.invite_token = token.clone();
        Some((
            token,
            nostr::DiscoveredMesh {
                listing,
                publisher_npub: format!("mdns:{}", self.instance_name),
                published_at: self.discovered_at,
                expires_at: None,
            },
        ))
    }
}

pub(crate) struct LanPublishConfig {
    pub(crate) name: Option<String>,
    pub(crate) region: Option<String>,
    pub(crate) max_clients: Option<usize>,
    pub(crate) api_port: u16,
    pub(crate) details_reachable: bool,
    pub(crate) interval_secs: u64,
    pub(crate) status_tx: Option<tokio::sync::watch::Sender<Option<nostr::PublishStateUpdate>>>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct LanDetailsProofRequest {
    pub(crate) token_fingerprint: String,
    pub(crate) challenge: String,
    pub(crate) proof: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LanDetailsResponse {
    pub(crate) mode: &'static str,
    pub(crate) scope: DiscoveryScope,
    pub(crate) source: &'static str,
    pub(crate) service_type: &'static str,
    pub(crate) listing: nostr::MeshListing,
    pub(crate) token_fingerprint: String,
    pub(crate) join_material: LanJoinMaterial,
    pub(crate) joinable_with_supplied_token: bool,
    pub(crate) details_path: &'static str,
    pub(crate) proof_challenge: String,
    pub(crate) published_version: Option<String>,
}

impl LanDetailsResponse {
    pub(crate) fn from_local_listing(
        mut listing: nostr::MeshListing,
        token_fingerprint: String,
        proof_challenge: String,
        published_version: Option<&str>,
    ) -> Self {
        listing.invite_token.clear();
        Self {
            mode: MeshDiscoveryMode::Mdns.as_str(),
            scope: MeshDiscoveryMode::Mdns.scope(),
            source: MeshDiscoveryMode::Mdns.source(),
            service_type: LAN_SERVICE_TYPE,
            listing,
            token_fingerprint,
            join_material: LanJoinMaterial::RequiresSuppliedToken,
            joinable_with_supplied_token: true,
            details_path: LAN_DETAILS_PATH,
            proof_challenge,
            published_version: published_version.map(str::to_string),
        }
    }
}

pub(crate) async fn publish_lan_loop(node: crate::mesh::Node, config: LanPublishConfig) {
    // Restrict the mDNS daemon to the bound LAN interface when known. On
    // multi-homed hosts (e.g. many utun/VPN interfaces) advertising on every
    // interface can prevent the advertisement from reaching the LAN peers
    // listen on. Pinning to the LAN address keeps mDNS on the same interface
    // QUIC is bound to.
    let lan_ip = node
        .advertised_endpoint_addr()
        .ip_addrs()
        .map(|addr| addr.ip())
        .find(|ip| ip.is_ipv4());
    let Some(daemon) = create_lan_publish_daemon(&config.status_tx, lan_ip) else {
        return;
    };

    let instance_name = lan_instance_name(&node).await;
    let host_name = format!("{instance_name}.local.");
    eprintln!("Publishing mesh on local LAN via mDNS ({LAN_SERVICE_TYPE})");

    let mut last_reported = None;
    loop {
        publish_lan_advertisement(LanPublishAttempt {
            daemon: &daemon,
            node: &node,
            name: config.name.clone(),
            region: config.region.clone(),
            max_clients: config.max_clients,
            api_port: config.api_port,
            details_reachable: config.details_reachable,
            status_tx: &config.status_tx,
            last_reported: &mut last_reported,
            instance_name: &instance_name,
            host_name: &host_name,
        })
        .await;
        tokio::time::sleep(Duration::from_secs(config.interval_secs)).await;
    }
}

fn create_lan_publish_daemon(
    status_tx: &Option<tokio::sync::watch::Sender<Option<nostr::PublishStateUpdate>>>,
    lan_ip: Option<std::net::IpAddr>,
) -> Option<ServiceDaemon> {
    match ServiceDaemon::new() {
        Ok(daemon) => {
            // When bound to a specific LAN interface, advertise only there so
            // the advertisement reaches LAN peers on multi-homed hosts.
            restrict_daemon_to_interface(&daemon, lan_ip);
            Some(daemon)
        }
        Err(err) => {
            tracing::warn!("Failed to create mDNS daemon: {err}");
            let _ = send_publish_state(status_tx, nostr::PublishStateUpdate::PublishFailed);
            None
        }
    }
}

struct LanPublishAttempt<'a> {
    daemon: &'a ServiceDaemon,
    node: &'a crate::mesh::Node,
    name: Option<String>,
    region: Option<String>,
    max_clients: Option<usize>,
    api_port: u16,
    details_reachable: bool,
    status_tx: &'a Option<tokio::sync::watch::Sender<Option<nostr::PublishStateUpdate>>>,
    last_reported: &'a mut Option<nostr::PublishStateUpdate>,
    instance_name: &'a str,
    host_name: &'a str,
}

async fn publish_lan_advertisement(attempt: LanPublishAttempt<'_>) {
    let LanPublishAttempt {
        daemon,
        node,
        name,
        region,
        max_clients,
        api_port,
        details_reachable,
        status_tx,
        last_reported,
        instance_name,
        host_name,
    } = attempt;
    let listing = build_local_mesh_listing(node, name, region, max_clients).await;
    let advert = LanMeshAdvertisement::from_listing(
        &listing,
        Some(&listing.invite_token),
        Some(crate::VERSION),
        details_reachable,
    )
    .with_endpoint_addr(&node.advertised_endpoint_addr());
    let Some(service_info) = encode_lan_service_info(
        &advert,
        instance_name,
        host_name,
        api_port,
        status_tx,
        last_reported,
    )
    .await
    else {
        return;
    };
    register_lan_service(daemon, service_info, status_tx, last_reported);
}

async fn encode_lan_service_info(
    advert: &LanMeshAdvertisement,
    instance_name: &str,
    host_name: &str,
    api_port: u16,
    status_tx: &Option<tokio::sync::watch::Sender<Option<nostr::PublishStateUpdate>>>,
    last_reported: &mut Option<nostr::PublishStateUpdate>,
) -> Option<ServiceInfo> {
    match service_info_for_advertisement(advert, instance_name, host_name, api_port) {
        Ok(info) => Some(info),
        Err(err) => {
            tracing::warn!("Failed to encode mDNS mesh advertisement: {err}");
            report_publish_state(
                status_tx,
                last_reported,
                nostr::PublishStateUpdate::PublishFailed,
            );
            None
        }
    }
}

fn register_lan_service(
    daemon: &ServiceDaemon,
    service_info: ServiceInfo,
    status_tx: &Option<tokio::sync::watch::Sender<Option<nostr::PublishStateUpdate>>>,
    last_reported: &mut Option<nostr::PublishStateUpdate>,
) {
    match daemon.register(service_info) {
        Ok(()) => report_publish_state(status_tx, last_reported, nostr::PublishStateUpdate::Public),
        Err(err) => {
            tracing::warn!("Failed to register mDNS mesh advertisement: {err}");
            report_publish_state(
                status_tx,
                last_reported,
                nostr::PublishStateUpdate::PublishFailed,
            );
        }
    }
}

/// Restrict an mDNS daemon to only the interface owning `lan_ip`.
///
/// `enable_interface` is additive on top of the default (all interfaces
/// enabled), so to actually pin to one interface we must first disable all,
/// then enable the LAN one. On multi-homed hosts (many utun/VPN interfaces)
/// this keeps mDNS traffic on the same interface QUIC is bound to, so
/// advertisements and queries reach LAN peers instead of being flooded onto
/// interfaces the peers cannot see.
fn restrict_daemon_to_interface(daemon: &ServiceDaemon, lan_ip: Option<std::net::IpAddr>) {
    // On multi-homed hosts (many utun/VPN interfaces) mdns-sd's default of
    // advertising on every interface can mean the advertisement is multicast on
    // an interface LAN peers cannot see, while the real LAN interface is starved
    // or never picked. A raw `IP_MULTICAST_IF`-pinned socket on the LAN address
    // reaches LAN peers reliably, so we pin the mDNS daemon to the LAN interface
    // the same way: disable all interfaces, then re-enable just the LAN address.
    //
    // Selections apply in order with last-match-wins (see mdns-sd's
    // `apply_intf_selections`), so the LAN `enable` after `disable(All)` keeps
    // exactly that interface active.
    let Some(ip) = lan_ip else {
        return;
    };
    if let Err(err) = daemon.disable_interface(mdns_sd::IfKind::All) {
        tracing::debug!("mDNS: could not disable interfaces before pinning to {ip}: {err}");
        return;
    }
    if let Err(err) = daemon.enable_interface(mdns_sd::IfKind::Addr(ip)) {
        tracing::debug!("mDNS: could not pin daemon to {ip}: {err}");
    }
}

pub async fn discover_lan(
    filter: &nostr::MeshFilter,
    supplied_invite_token: Option<&str>,
    timeout: Duration,
) -> Result<Vec<LanDiscoveredMesh>> {
    discover_lan_on_interface(filter, supplied_invite_token, timeout, None).await
}

/// Like [`discover_lan`] but, when `lan_ip` is set, restricts the browse to the
/// matching interface. On multi-homed hosts this keeps mDNS on the same
/// interface QUIC is bound to so LAN advertisements are seen.
pub async fn discover_lan_on_interface(
    filter: &nostr::MeshFilter,
    supplied_invite_token: Option<&str>,
    timeout: Duration,
    lan_ip: Option<std::net::IpAddr>,
) -> Result<Vec<LanDiscoveredMesh>> {
    let daemon = ServiceDaemon::new().context("create mDNS daemon")?;
    restrict_daemon_to_interface(&daemon, lan_ip);
    let receiver = match daemon.browse(LAN_SERVICE_TYPE) {
        Ok(receiver) => receiver,
        Err(err) => {
            shutdown_lan_daemon(daemon).await;
            return Err(anyhow::Error::new(err).context(format!("browse {LAN_SERVICE_TYPE}")));
        }
    };
    let deadline = tokio::time::Instant::now() + timeout;
    let mut by_instance: HashMap<String, LanDiscoveredMesh> = HashMap::new();

    while tokio::time::Instant::now() < deadline {
        let Some(service) = next_resolved_lan_service(&receiver, deadline).await else {
            break;
        };
        record_lan_service(&mut by_instance, &service, filter, supplied_invite_token);
    }

    stop_lan_browse(receiver, daemon).await;
    Ok(sorted_lan_meshes(by_instance))
}

async fn next_resolved_lan_service(
    receiver: &mdns_sd::Receiver<ServiceEvent>,
    deadline: tokio::time::Instant,
) -> Option<ResolvedService> {
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return None;
        }
        let event = match tokio::time::timeout(remaining, receiver.recv_async()).await {
            Ok(Ok(event)) => event,
            Ok(Err(_)) | Err(_) => return None,
        };
        if let ServiceEvent::ServiceResolved(service) = event {
            return Some(*service);
        }
    }
}

async fn stop_lan_browse(receiver: mdns_sd::Receiver<ServiceEvent>, daemon: ServiceDaemon) {
    drop(receiver);
    if let Err(err) = daemon.stop_browse(LAN_SERVICE_TYPE) {
        tracing::debug!("Failed to stop mDNS LAN browse before daemon shutdown: {err}");
    }
    shutdown_lan_daemon(daemon).await;
}

fn sorted_lan_meshes(by_instance: HashMap<String, LanDiscoveredMesh>) -> Vec<LanDiscoveredMesh> {
    let mut meshes = by_instance.into_values().collect::<Vec<_>>();
    meshes.sort_by(compare_lan_meshes);
    meshes
}

fn compare_lan_meshes(left: &LanDiscoveredMesh, right: &LanDiscoveredMesh) -> std::cmp::Ordering {
    right
        .listing
        .node_count
        .cmp(&left.listing.node_count)
        .then(
            right
                .listing
                .total_vram_bytes
                .cmp(&left.listing.total_vram_bytes),
        )
        .then(left.instance_name.cmp(&right.instance_name))
}

async fn shutdown_lan_daemon(daemon: ServiceDaemon) -> bool {
    let shutdown = tokio::task::spawn_blocking(move || {
        match daemon.shutdown() {
            Ok(receiver) => receiver.recv_timeout(DAEMON_SHUTDOWN_TIMEOUT),
            Err(err) => {
                tracing::debug!("Failed to request mDNS daemon shutdown: {err}");
                return false;
            }
        }
        .map(|status| status == DaemonStatus::Shutdown)
        .unwrap_or(false)
    })
    .await
    .unwrap_or(false);

    if !shutdown {
        tracing::debug!("mDNS daemon shutdown did not report completion before timeout");
    }
    shutdown
}

pub(crate) async fn discover_lan_join_candidates(
    filter: &nostr::MeshFilter,
    supplied_invite_token: Option<&str>,
    timeout: Duration,
) -> Result<Vec<(String, nostr::DiscoveredMesh)>> {
    Ok(discover_lan(filter, supplied_invite_token, timeout)
        .await?
        .into_iter()
        .filter_map(|mesh| mesh.to_join_candidate())
        .collect())
}

pub(crate) fn lan_token_fingerprint(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(LAN_INVITE_TOKEN_FINGERPRINT_DOMAIN);
    hasher.update(token.trim().as_bytes());
    let digest = hasher.finalize();
    hex::encode(&digest[..16])
}

pub(crate) fn lan_details_challenge(token_fingerprint: &str, now_secs: u64) -> String {
    lan_details_challenge_for_bucket(
        token_fingerprint,
        now_secs / LAN_DETAILS_CHALLENGE_WINDOW_SECS,
    )
}

pub(crate) fn lan_details_token_proof(invite_token: &str, challenge: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(LAN_DETAILS_TOKEN_PROOF_DOMAIN);
    hasher.update(invite_token.trim().as_bytes());
    hasher.update(b"\0");
    hasher.update(challenge.trim().as_bytes());
    hex::encode(hasher.finalize())
}

pub(crate) fn verify_lan_details_token_proof(
    expected_invite_token: &str,
    token_fingerprint: &str,
    challenge: &str,
    proof: &str,
    now_secs: u64,
) -> bool {
    let token_fingerprint = token_fingerprint.trim();
    if lan_token_fingerprint(expected_invite_token) != token_fingerprint {
        return false;
    }
    let Some(challenge_bucket) = lan_details_challenge_bucket(challenge.trim()) else {
        return false;
    };
    if !lan_details_challenge_bucket_is_recent(challenge_bucket, now_secs) {
        return false;
    }
    if lan_details_challenge_for_bucket(token_fingerprint, challenge_bucket) != challenge.trim() {
        return false;
    }
    lan_details_token_proof(expected_invite_token, challenge).eq_ignore_ascii_case(proof.trim())
}

fn lan_details_challenge_for_bucket(token_fingerprint: &str, bucket: u64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(LAN_DETAILS_CHALLENGE_DOMAIN);
    hasher.update(token_fingerprint.trim().as_bytes());
    hasher.update(b"\0");
    hasher.update(bucket.to_string().as_bytes());
    let digest = hasher.finalize();
    format!("v1:{bucket}:{}", hex::encode(&digest[..16]))
}

fn lan_details_challenge_bucket(challenge: &str) -> Option<u64> {
    let mut parts = challenge.split(':');
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some("v1"), Some(bucket), Some(digest), None)
            if digest.len() == 32 && digest.chars().all(|ch| ch.is_ascii_hexdigit()) =>
        {
            bucket.parse().ok()
        }
        _ => None,
    }
}

fn lan_details_challenge_bucket_is_recent(bucket: u64, now_secs: u64) -> bool {
    let current = now_secs / LAN_DETAILS_CHALLENGE_WINDOW_SECS;
    bucket.abs_diff(current) <= 1
}

pub(crate) fn discovery_source_label(mode: MeshDiscoveryMode, operation: &str) -> String {
    match mode {
        MeshDiscoveryMode::Nostr => format!("Nostr {operation}"),
        MeshDiscoveryMode::Mdns => format!("mDNS LAN {operation}"),
    }
}

pub(crate) async fn build_local_mesh_listing(
    node: &crate::mesh::Node,
    name: Option<String>,
    region: Option<String>,
    max_clients: Option<usize>,
) -> nostr::MeshListing {
    let peers = node.peers().await;
    let client_count = lan_client_count(&peers);
    let actually_serving = lan_served_models(node, &peers).await;
    let served_set = actually_serving
        .iter()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();
    let wanted = lan_wanted_models(node, &served_set).await;
    let available = lan_available_models(node, &peers, &served_set).await;
    let total_vram_bytes = lan_total_vram_bytes(node, &peers);
    let node_count = lan_serving_node_count(&peers);

    nostr::MeshListing {
        invite_token: node.invite_token().await,
        serving: actually_serving,
        wanted,
        on_disk: available,
        total_vram_bytes,
        node_count,
        client_count,
        max_clients: max_clients.unwrap_or(0),
        name,
        region,
        mesh_id: node.mesh_id().await,
    }
}

fn record_lan_service(
    by_instance: &mut HashMap<String, LanDiscoveredMesh>,
    service: &ResolvedService,
    filter: &nostr::MeshFilter,
    supplied_invite_token: Option<&str>,
) {
    if !service.is_valid() {
        return;
    }
    let Some((advert, listing, discovered)) = lan_discovered_listing(service) else {
        return;
    };
    if !filter.matches(&discovered) {
        return;
    }
    let joinable = advert.matches_supplied_token(supplied_invite_token);
    by_instance.insert(
        service.get_fullname().to_string(),
        lan_discovered_mesh(service, listing, advert, supplied_invite_token, joinable),
    );
}

fn lan_discovered_listing(
    service: &ResolvedService,
) -> Option<(
    LanMeshAdvertisement,
    nostr::MeshListing,
    nostr::DiscoveredMesh,
)> {
    let advert = match LanMeshAdvertisement::from_resolved_service(service) {
        Ok(advert) => advert,
        Err(err) => {
            tracing::debug!(
                "Skipping malformed mDNS mesh advertisement {}: {err}",
                service.get_fullname(),
            );
            return None;
        }
    };
    let listing = advert.sanitized_listing();
    let discovered = nostr::DiscoveredMesh {
        listing: listing.clone(),
        publisher_npub: format!("mdns:{}", service.get_fullname()),
        published_at: current_unix_secs(),
        expires_at: None,
    };
    Some((advert, listing, discovered))
}

fn lan_discovered_mesh(
    service: &ResolvedService,
    listing: nostr::MeshListing,
    advert: LanMeshAdvertisement,
    supplied_invite_token: Option<&str>,
    joinable: bool,
) -> LanDiscoveredMesh {
    let endpoint_addr = advert.endpoint_addr();
    LanDiscoveredMesh {
        mode: MeshDiscoveryMode::Mdns.as_str(),
        scope: MeshDiscoveryMode::Mdns.scope(),
        source: MeshDiscoveryMode::Mdns.source(),
        service_type: LAN_SERVICE_TYPE,
        instance_name: service.get_fullname().to_string(),
        host: service.get_hostname().to_string(),
        port: service.get_port(),
        addresses: service
            .get_addresses()
            .iter()
            .map(ToString::to_string)
            .collect(),
        listing,
        token_fingerprint: advert.token_fingerprint,
        details_path: advert.details_path,
        proof_challenge: advert.proof_challenge,
        join_material: advert.join_material,
        joinable_with_supplied_token: joinable,
        published_version: advert.app_version,
        discovered_at: current_unix_secs(),
        join_token: joinable.then(|| supplied_invite_token.unwrap_or_default().to_string()),
        endpoint_addr,
    }
}

fn lan_client_count(peers: &[crate::mesh::PeerInfo]) -> usize {
    peers
        .iter()
        .filter(|peer| matches!(peer.role, crate::mesh::NodeRole::Client))
        .count()
}

async fn lan_served_models(
    node: &crate::mesh::Node,
    peers: &[crate::mesh::PeerInfo],
) -> Vec<String> {
    let mut actually_serving = Vec::new();
    if matches!(node.role().await, crate::mesh::NodeRole::Host { .. }) {
        for model in node.hosted_models().await {
            push_unique(&mut actually_serving, model);
        }
    }
    for peer in peers {
        if matches!(peer.role, crate::mesh::NodeRole::Host { .. }) {
            for model in peer.routable_models() {
                push_unique(&mut actually_serving, model);
            }
        }
    }
    actually_serving
}

async fn lan_wanted_models(
    node: &crate::mesh::Node,
    served_set: &std::collections::HashSet<&str>,
) -> Vec<String> {
    let mut wanted = Vec::new();
    for model in node.active_demand().await.keys() {
        if !served_set.contains(model.as_str()) {
            push_unique(&mut wanted, model.clone());
        }
    }
    wanted
}

async fn lan_available_models(
    node: &crate::mesh::Node,
    peers: &[crate::mesh::PeerInfo],
    served_set: &std::collections::HashSet<&str>,
) -> Vec<String> {
    let mut available = Vec::new();
    for model in node.available_models().await {
        if !served_set.contains(model.as_str()) {
            push_unique(&mut available, model);
        }
    }
    for peer in peers {
        for model in &peer.available_models {
            if !served_set.contains(model.as_str()) {
                push_unique(&mut available, model.clone());
            }
        }
    }
    available
}

fn lan_total_vram_bytes(node: &crate::mesh::Node, peers: &[crate::mesh::PeerInfo]) -> u64 {
    peers
        .iter()
        .filter(|peer| !matches!(peer.role, crate::mesh::NodeRole::Client))
        .map(|peer| peer.vram_bytes)
        .sum::<u64>()
        + node.vram_bytes()
}

fn lan_serving_node_count(peers: &[crate::mesh::PeerInfo]) -> usize {
    peers
        .iter()
        .filter(|peer| !matches!(peer.role, crate::mesh::NodeRole::Client))
        .count()
        + 1
}

async fn lan_instance_name(node: &crate::mesh::Node) -> String {
    // The mDNS instance name must be unique per node, not per mesh: every node
    // in a mesh advertises its own record (carrying its own `ep_addr`), and two
    // nodes sharing an instance name would clobber each other in mDNS, hiding
    // peers from reverse-dial. Use the node's endpoint id, which is unique.
    let suffix = sanitize_dns_label(&node.id().fmt_short().to_string());
    format!("mesh-llm-{suffix}")
}

fn service_info_for_advertisement(
    advert: &LanMeshAdvertisement,
    instance_name: &str,
    host_name: &str,
    port: u16,
) -> Result<ServiceInfo> {
    let txt = advert.to_txt_properties()?;
    ServiceInfo::new(
        LAN_SERVICE_TYPE,
        instance_name,
        host_name,
        "",
        port,
        txt.as_slice(),
    )
    .map(ServiceInfo::enable_addr_auto)
    .context("create mDNS service info")
}

fn parse_txt_properties(props: &HashMap<&str, &str>) -> Result<LanMeshAdvertisement> {
    anyhow::ensure!(
        props.get("svc") == Some(&"mesh-llm"),
        "not a mesh-llm advertisement"
    );
    let schema = props
        .get("schema")
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(0);
    anyhow::ensure!(
        schema == TXT_SCHEMA_VERSION,
        "unsupported mDNS mesh schema version {schema}"
    );
    anyhow::ensure!(
        props.get("join") == Some(&"token-fingerprint"),
        "unsupported mDNS join material"
    );

    Ok(LanMeshAdvertisement {
        mesh_id: optional_txt(props, "mesh_id"),
        mesh_name: optional_txt(props, "name"),
        region: optional_txt(props, "region"),
        serving_summary: unpack_txt_list(props.get("serving").copied().unwrap_or_default()),
        wanted_summary: unpack_txt_list(props.get("wanted").copied().unwrap_or_default()),
        on_disk_summary: unpack_txt_list(props.get("on_disk").copied().unwrap_or_default()),
        total_vram_bytes: parse_txt_number(props, "vram")?,
        node_count: parse_txt_number(props, "nodes")?,
        client_count: parse_txt_number(props, "clients").unwrap_or(0),
        max_clients: parse_txt_number(props, "max_clients").unwrap_or(0),
        token_fingerprint: optional_txt(props, "tok_fp"),
        details_path: optional_txt(props, "details"),
        proof_challenge: optional_txt(props, "proof_challenge"),
        app_version: optional_txt(props, "version"),
        join_material: LanJoinMaterial::RequiresSuppliedToken,
        endpoint_addr_b64: optional_txt(props, "ep_addr"),
    })
}

/// Encode an [`iroh::EndpointAddr`] as base64url JSON for an mDNS TXT value.
/// Returns `None` if it would exceed the DNS-SD TXT value limit.
fn encode_endpoint_addr_b64(addr: &iroh::EndpointAddr) -> Option<String> {
    use base64::Engine;
    let json = serde_json::to_vec(addr).ok()?;
    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json);
    // Key "ep_addr" (7) + value must stay under the DNS-SD 255 limit; keep margin.
    (encoded.len() < TXT_VALUE_LIMIT).then_some(encoded)
}

/// Decode a base64url-JSON [`iroh::EndpointAddr`] from an mDNS TXT value.
fn decode_endpoint_addr_b64(value: &str) -> Option<iroh::EndpointAddr> {
    use base64::Engine;
    let raw = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(value)
        .ok()?;
    serde_json::from_slice(&raw).ok()
}

fn parse_txt_number<T>(props: &HashMap<&str, &str>, key: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = props
        .get(key)
        .with_context(|| format!("missing mDNS TXT property '{key}'"))?;
    value
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid mDNS TXT property '{key}': {err}"))
}

fn optional_txt(props: &HashMap<&str, &str>, key: &str) -> Option<String> {
    props
        .get(key)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn push_optional_txt(txt: &mut Vec<(String, String)>, key: &str, value: Option<&str>) {
    if let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) {
        txt.push((key.to_string(), truncate_txt_value(value)));
    }
}

fn bounded_list(values: &[String]) -> Vec<String> {
    values
        .iter()
        .filter_map(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| truncate_txt_value(trimmed))
        })
        .take(8)
        .collect()
}

fn pack_txt_list(values: &[String]) -> String {
    truncate_txt_value(
        &values
            .iter()
            .map(|value| value.replace(TXT_LIST_SEPARATOR, " "))
            .collect::<Vec<_>>()
            .join(&TXT_LIST_SEPARATOR.to_string()),
    )
}

fn unpack_txt_list(value: &str) -> Vec<String> {
    if value.trim().is_empty() {
        return Vec::new();
    }
    value
        .split(TXT_LIST_SEPARATOR)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn truncate_txt_value(value: &str) -> String {
    value.chars().take(TXT_VALUE_LIMIT).collect()
}

fn sanitize_dns_label(value: &str) -> String {
    let mut label = value
        .chars()
        .filter_map(|ch| {
            if ch.is_ascii_alphanumeric() {
                Some(ch.to_ascii_lowercase())
            } else if ch == '-' || ch == '_' {
                Some('-')
            } else {
                None
            }
        })
        .collect::<String>();
    label.truncate(48);
    let label = label.trim_matches('-');
    if label.is_empty() {
        "node".to_string()
    } else {
        label.to_string()
    }
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn report_publish_state(
    status_tx: &Option<tokio::sync::watch::Sender<Option<nostr::PublishStateUpdate>>>,
    last_reported: &mut Option<nostr::PublishStateUpdate>,
    next: nostr::PublishStateUpdate,
) {
    if *last_reported == Some(next) {
        return;
    }
    let _ = send_publish_state(status_tx, next);
    *last_reported = Some(next);
}

fn send_publish_state(
    status_tx: &Option<tokio::sync::watch::Sender<Option<nostr::PublishStateUpdate>>>,
    next: nostr::PublishStateUpdate,
) -> Result<(), tokio::sync::watch::error::SendError<Option<nostr::PublishStateUpdate>>> {
    if let Some(tx) = status_tx {
        tx.send(Some(next))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::nostr::MeshListing;

    fn sample_listing(invite_token: &str) -> MeshListing {
        MeshListing {
            invite_token: invite_token.to_string(),
            serving: vec!["Qwen3-8B-Q4_K_M".to_string()],
            wanted: vec!["Qwen3-32B-Q4_K_M".to_string()],
            on_disk: vec!["Qwen3-14B-Q4_K_M".to_string()],
            total_vram_bytes: 64_000_000_000,
            node_count: 2,
            client_count: 1,
            max_clients: 4,
            name: Some("lab-cluster".to_string()),
            region: Some("LAN".to_string()),
            mesh_id: Some("mesh-lab-01".to_string()),
        }
    }

    #[test]
    fn discovery_modes_have_stable_cli_names_and_metadata() {
        assert_eq!(MeshDiscoveryMode::default(), MeshDiscoveryMode::Nostr);
        assert_eq!(MeshDiscoveryMode::Nostr.as_str(), "nostr");
        assert_eq!(MeshDiscoveryMode::Nostr.source(), "nostr-relay");
        assert_eq!(MeshDiscoveryMode::Nostr.scope(), DiscoveryScope::Public);
        assert_eq!(MeshDiscoveryMode::Mdns.as_str(), "mdns");
        assert_eq!(MeshDiscoveryMode::Mdns.source(), "mdns-sd");
        assert_eq!(MeshDiscoveryMode::Mdns.scope(), DiscoveryScope::Lan);
    }

    #[test]
    fn lan_token_fingerprint_is_stable_and_does_not_expose_token() {
        let token = "very-secret-reusable-invite-token";
        let first = lan_token_fingerprint(token);
        let second = lan_token_fingerprint(token);

        assert_eq!(first, second);
        assert!(!first.contains(token));
        assert_ne!(first, lan_token_fingerprint("different-token"));
    }

    #[test]
    fn lan_advertisement_txt_round_trips_without_raw_invite_token() {
        let invite_token = "invite-token-that-must-not-leak";
        let listing = sample_listing(invite_token);
        let advert = LanMeshAdvertisement::from_listing(
            &listing,
            Some(invite_token),
            Some(crate::VERSION),
            true,
        );

        let txt = advert.to_txt_properties().expect("txt should encode");
        let serialized = txt
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(";");

        assert!(!serialized.contains(invite_token));
        assert!(serialized.contains("tok_fp="));

        let decoded = LanMeshAdvertisement::from_txt_properties(&txt).expect("txt should decode");
        assert_eq!(decoded.mesh_id.as_deref(), Some("mesh-lab-01"));
        assert_eq!(decoded.mesh_name.as_deref(), Some("lab-cluster"));
        assert_eq!(decoded.serving_summary, vec!["Qwen3-8B-Q4_K_M"]);
        assert_eq!(
            decoded.token_fingerprint.as_deref(),
            Some(lan_token_fingerprint(invite_token).as_str())
        );
        assert_eq!(
            decoded.join_material,
            LanJoinMaterial::RequiresSuppliedToken
        );
    }

    #[test]
    fn lan_advertisement_endpoint_addr_txt_round_trips() {
        use iroh::{EndpointAddr, SecretKey};
        let secret = SecretKey::from_bytes(&[7u8; 32]);
        let mut addr = EndpointAddr::from(secret.public());
        addr = addr.with_ip_addr("192.168.1.50:9555".parse().unwrap());

        let listing = sample_listing("tok");
        let advert =
            LanMeshAdvertisement::from_listing(&listing, Some("tok"), Some(crate::VERSION), false)
                .with_endpoint_addr(&addr);

        let txt = advert.to_txt_properties().expect("txt should encode");
        assert!(txt.iter().any(|(k, _)| k == "ep_addr"));

        let decoded = LanMeshAdvertisement::from_txt_properties(&txt).expect("txt should decode");
        let decoded_addr = decoded.endpoint_addr().expect("ep_addr should decode");
        assert_eq!(decoded_addr.id, addr.id);
        assert!(
            decoded_addr
                .ip_addrs()
                .any(|a| a.to_string() == "192.168.1.50:9555")
        );
    }

    #[test]
    fn lan_advertisement_without_endpoint_addr_decodes_none() {
        let listing = sample_listing("tok");
        let advert =
            LanMeshAdvertisement::from_listing(&listing, Some("tok"), Some(crate::VERSION), false);
        let txt = advert.to_txt_properties().expect("txt should encode");
        assert!(!txt.iter().any(|(k, _)| k == "ep_addr"));
        let decoded = LanMeshAdvertisement::from_txt_properties(&txt).expect("txt should decode");
        assert!(decoded.endpoint_addr().is_none());
    }

    #[test]
    fn lan_advertisement_exposes_token_gated_details_without_raw_invite_token() {
        let invite_token = "invite-token-for-details-proof";
        let listing = sample_listing(invite_token);
        let advert = LanMeshAdvertisement::from_listing(
            &listing,
            Some(invite_token),
            Some(crate::VERSION),
            true,
        );

        let txt = advert.to_txt_properties().expect("txt should encode");
        let serialized = txt
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(";");

        assert!(!serialized.contains(invite_token));
        assert!(serialized.contains("details=/api/discovery/lan-details"));
        assert!(serialized.contains("proof_challenge="));

        let decoded = LanMeshAdvertisement::from_txt_properties(&txt).expect("txt should decode");
        assert_eq!(decoded.details_path.as_deref(), Some(LAN_DETAILS_PATH));
        assert!(decoded.proof_challenge.is_some());
    }

    #[test]
    fn lan_advertisement_omits_details_when_management_api_is_loopback_only() {
        let invite_token = "invite-token-for-loopback-only-console";
        let listing = sample_listing(invite_token);
        let advert = LanMeshAdvertisement::from_listing(
            &listing,
            Some(invite_token),
            Some(crate::VERSION),
            false,
        );

        let txt = advert.to_txt_properties().expect("txt should encode");
        let serialized = txt
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(";");

        assert!(serialized.contains("tok_fp="));
        assert!(!serialized.contains("details="));
        assert!(!serialized.contains("proof_challenge="));
        assert_eq!(
            advert.token_fingerprint.as_deref(),
            Some(lan_token_fingerprint(invite_token).as_str())
        );
        assert!(advert.details_path.is_none());
        assert!(advert.proof_challenge.is_none());
    }

    #[test]
    fn lan_details_proof_accepts_matching_token_and_recent_challenge() {
        let invite_token = "invite-token-for-proof";
        let token_fingerprint = lan_token_fingerprint(invite_token);
        let challenge = lan_details_challenge(&token_fingerprint, current_unix_secs());
        let proof = lan_details_token_proof(invite_token, &challenge);

        assert!(verify_lan_details_token_proof(
            invite_token,
            &token_fingerprint,
            &challenge,
            &proof,
            current_unix_secs(),
        ));
    }

    #[test]
    fn lan_details_proof_rejects_public_fingerprint_without_token_secret() {
        let invite_token = "invite-token-for-proof";
        let token_fingerprint = lan_token_fingerprint(invite_token);
        let challenge = lan_details_challenge(&token_fingerprint, current_unix_secs());
        let attacker_proof = lan_details_token_proof("wrong-token", &challenge);

        assert!(!verify_lan_details_token_proof(
            invite_token,
            &token_fingerprint,
            &challenge,
            &attacker_proof,
            current_unix_secs(),
        ));
    }

    #[test]
    fn lan_details_response_sanitizes_invite_token() {
        let invite_token = "invite-token-that-response-must-not-return";
        let token_fingerprint = lan_token_fingerprint(invite_token);
        let challenge = lan_details_challenge(&token_fingerprint, current_unix_secs());
        let response = LanDetailsResponse::from_local_listing(
            sample_listing(invite_token),
            token_fingerprint.clone(),
            challenge.clone(),
            Some(crate::VERSION),
        );

        assert!(response.listing.invite_token.is_empty());
        assert_eq!(response.token_fingerprint, token_fingerprint);
        assert_eq!(response.details_path, LAN_DETAILS_PATH);
        assert_eq!(response.proof_challenge, challenge);
    }

    #[test]
    fn lan_advertisement_requires_matching_supplied_join_token() {
        let invite_token = "invite-token-for-lab-mesh";
        let advert = LanMeshAdvertisement::from_listing(
            &sample_listing(invite_token),
            Some(invite_token),
            Some(crate::VERSION),
            true,
        );

        assert!(advert.matches_supplied_token(Some(invite_token)));
        assert!(!advert.matches_supplied_token(None));
        assert!(!advert.matches_supplied_token(Some("wrong-token")));
    }

    #[tokio::test]
    async fn shutdown_lan_daemon_reports_completion_when_available() {
        let Ok(daemon) = ServiceDaemon::new() else {
            eprintln!("mDNS daemon unavailable in this test environment");
            return;
        };

        assert!(shutdown_lan_daemon(daemon).await);
    }
}
