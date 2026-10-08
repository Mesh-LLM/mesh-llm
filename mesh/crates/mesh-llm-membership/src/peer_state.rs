//! Peer-membership state: the gossip announcement model, the neutral
//! `PeerInfo` state, latency/display observations, and the dependency-neutral
//! peer-admission helpers.
//!
//! `PeerInfo`'s state and its non-routing accessors live here. The
//! serving-routing projections (`routable_models`, `routes_model`,
//! `advertised_context_length`, …) stay in `mesh-llm-host-runtime` as free
//! functions over `&PeerInfo`: they resolve raw model ids through
//! `public_model_id_from_identity`/`canonical_demand_model_ref`, which depend
//! on `skippy_model_ref` and the host `models` catalog. Live membership maps and pending handshakes live in this crate
//! (`state` and `connection_reservation`). Host ownership/control listener state
//! and `impl Node` admission side effects remain in the host (see the README).

use std::collections::HashMap;

use iroh::{EndpointAddr, EndpointId, PublicKey};
use mesh_llm_identity::{OwnershipStatus, OwnershipSummary, SignedNodeOwnership, TrustPolicy};
use mesh_llm_protocol::proto::node as proto_node;
use mesh_llm_protocol::protocol::{
    ControlFrameError, STREAM_GOSSIP, STREAM_ROUTE_REQUEST, STREAM_TUNNEL_HTTP,
};
use mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement;
use mesh_llm_types::mesh::{
    AdvertisedMemory, ModelDemand, ModelRuntimeDescriptor, ModelSourceKind, ServedModelDescriptor,
    ServedModelIdentity,
};

use crate::advertised_throughput::ModelThroughputHint;
use crate::release_attestation::{
    ReleaseAttestationSummary, ReleaseBuildAttestation, ReleaseSignerTrustStore,
    verify_release_attestation,
};
use crate::requirements::{DirectNodeAdmissionProof, SignedMeshGenesisPolicy};
use crate::selected_path::SelectedPathObservation;
use crate::types::NodeRole;

/// Whether a peer has recent direct evidence of reachability. Announcement
/// content alone is not liveness: a bridge may retain a departed peer's last
/// advertisement. A live connection or a fresh measured RTT is sufficient.
pub fn peer_has_observed_liveness(peer: &PeerInfo, has_connection: bool) -> bool {
    has_connection
        || peer.display_rtt.as_ref().is_some_and(|observation| {
            observation.observed_at.elapsed() < std::time::Duration::from_secs(PEER_STALE_SECS)
        })
}

/// Whether the owner-trust policy admits a peer with the given ownership
/// summary. Pure admission predicate shared with host gossip/admission paths.
pub fn policy_accepts_peer(policy: TrustPolicy, owner_summary: &OwnershipSummary) -> bool {
    match policy {
        TrustPolicy::Off | TrustPolicy::PreferOwned => true,
        TrustPolicy::RequireOwned | TrustPolicy::Allowlist => {
            owner_summary.status == OwnershipStatus::Verified
        }
    }
}

/// Score how strongly a served-model identity identifies a model
/// (source kind plus canonical-ref/revision bonuses). Pure membership helper
/// used by host peer-descriptor scoring.
pub fn model_identity_score(identity: &ServedModelIdentity) -> u8 {
    let kind_score = match identity.source_kind {
        ModelSourceKind::HuggingFace => 4,
        ModelSourceKind::Catalog => 3,
        ModelSourceKind::DirectUrl => 2,
        ModelSourceKind::LocalGguf => 1,
        ModelSourceKind::Unknown => 0,
    };
    let canonical_bonus = if identity.canonical_ref.is_some() {
        2
    } else {
        0
    };
    let revision_bonus = if identity.revision.is_some() { 1 } else { 0 };
    kind_score + canonical_bonus + revision_bonus
}

/// Gossip payload — extends EndpointAddr with role metadata.
/// Internal mesh gossip model. Legacy JSON v0 is adapted at the boundary.
#[derive(Debug, Clone)]
pub struct PeerAnnouncement {
    pub addr: EndpointAddr,
    pub role: NodeRole,
    pub first_joined_mesh_ts: Option<u64>,
    pub models: Vec<String>,
    pub vram_bytes: u64,
    pub model_source: Option<String>,
    pub serving_models: Vec<String>,
    pub hosted_models: Option<Vec<String>>,
    /// All GGUF filenames on disk in managed or legacy local storage (for mesh catalog)
    pub available_models: Vec<String>,
    pub requested_models: Vec<String>,
    /// Advisory canonical refs this node wants the mesh to consider.
    pub explicit_model_interests: Vec<String>,
    pub version: Option<String>,
    pub model_demand: HashMap<String, ModelDemand>,
    pub mesh_id: Option<String>,
    pub mesh_policy_hash: Option<String>,
    pub gpu_name: Option<String>,
    pub hostname: Option<String>,
    pub is_soc: Option<bool>,
    pub gpu_vram: Option<String>,
    pub gpu_reserved_bytes: Option<String>,
    /// Itemized view of `vram_bytes`; absent from peers that predate it or
    /// that do not enumerate their hardware.
    pub memory: Option<AdvertisedMemory>,
    pub gpu_mem_bandwidth_gbps: Option<String>,
    pub gpu_compute_tflops_fp32: Option<String>,
    pub gpu_compute_tflops_fp16: Option<String>,
    pub available_model_metadata: Vec<proto_node::CompactModelMetadata>,
    pub experts_summary: Option<proto_node::ExpertsSummary>,
    pub available_model_sizes: HashMap<String, u64>,
    pub served_model_descriptors: Vec<ServedModelDescriptor>,
    pub served_model_runtime: Vec<ModelRuntimeDescriptor>,
    pub owner_attestation: Option<SignedNodeOwnership>,
    pub genesis_policy: Option<SignedMeshGenesisPolicy>,
    pub release_attestation: Option<ReleaseBuildAttestation>,
    pub direct_admission_proof: Option<DirectNodeAdmissionProof>,
    pub artifact_transfer_supported: bool,
    pub stage_protocol_generation_supported: bool,
    pub stage_status_list_supported: bool,
    pub local_gguf_content_id_supported: bool,
    pub decode_batch_policy_supported: bool,
    pub advertised_model_throughput: Vec<ModelThroughputHint>,
    #[cfg(feature = "payments")]
    pub lightning_offers:
        std::collections::BTreeMap<String, mesh_llm_payments_types::pricing::Pricing>,
    pub cache_affinity: Option<CacheAffinityAdvertisement>,
    pub latency_ms: Option<u32>,
    pub latency_source: Option<proto_node::LatencySource>,
    pub latency_age_ms: Option<u64>,
    pub latency_observer_id: Option<EndpointId>,
    pub inference_admission_state: Option<proto_node::InferenceAdmissionState>,
    /// Optional self-reported log head; carried opaquely and never verified here.
    pub claimed_log_head: Option<ClaimedLogHead>,
}

/// A peer's latest self-reported claim about the head of its append-only log
/// — see `ClaimedLogHead` in `node.proto` for the wire shape and the
/// signing-scope note. Carried opaquely: mesh-llm never verifies
/// `claimed_signature` itself, hence the name — a consumer that does verify
/// it may define its own `VerifiedLogHead` type; none exists here. Public to cross the host/membership crate boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimedLogHead {
    pub log_id: String,
    pub size: u64,
    pub root: Vec<u8>,
    pub timestamp_unix_ms: u64,
    pub claimed_signature: Vec<u8>,
    pub signature_algorithm: String,
}

/// A single direct RTT measurement (e.g. from gossip exchange).
#[derive(Debug, Clone)]
pub struct DirectLatencyObservation {
    pub rtt_ms: u32,
    pub observed_at: std::time::Instant,
}

/// Latency propagated via transitive gossip (not measured directly).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PropagatedLatencyObservation {
    pub latency_ms: u32,
    pub age_ms_at_received: u64,
    pub received_at: std::time::Instant,
    pub observer_id: Option<EndpointId>,
}

/// Which source a display latency value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayLatencySource {
    Direct,
    Estimated,
    Unknown,
}

/// Computed display latency for UI/API consumption.
#[derive(Debug, Clone)]
pub struct DisplayLatency {
    pub latency_ms: Option<u32>,
    pub source: DisplayLatencySource,
    pub age_ms: u64,
    pub observer_id: Option<EndpointId>,
}

#[derive(Debug, Clone)]
pub struct MeshCatalogEntry {
    pub model_name: String,
    pub descriptor: Option<ServedModelDescriptor>,
}

/// Peers not directly verified within this window are considered stale
/// and excluded from gossip propagation. After 2x this duration they're removed entirely.
pub const PEER_STALE_SECS: u64 = 180; // 3 minutes

/// How long a dead-peer entry blocks transitive re-learning and outbound
/// reconnection. After this period the entry expires silently and the peer
/// can be re-discovered through normal gossip propagation. If the peer is
/// genuinely gone, no bridge peer will mention it and it stays forgotten.
pub const DEAD_PEER_TTL: std::time::Duration = std::time::Duration::from_secs(300); // 5 minutes
/// How long a confirmed-departed peer id stays barred from transitive
/// re-admission. [`DEAD_PEER_TTL`] expires quickly so reconnection attempts
/// can resume, but gossip bridges can keep carrying the departed id's final
/// announcement long after that (issue #1756): re-admitting it transitively
/// resurrects a ghost `state: serving` entry with no direct connection.
/// Only direct proof of life (a gossip exchange or connection with the id
/// itself) clears this record early; otherwise it expires silently.
pub const DEPARTED_PEER_TRANSITIVE_BLOCK_TTL: std::time::Duration =
    std::time::Duration::from_secs(3600); // 1 hour

pub const PEER_DOWN_REPORTER_COOLDOWN_SECS: u64 = 600; // 10 minutes

/// Returns `true` if the given stream type is permitted before a peer has
/// been admitted through gossip, under the node's trust policy.
///
/// With a non-enforcing trust policy (`Off` or `PreferOwned`), three streams
/// bypass the quarantine gate:
/// - `STREAM_GOSSIP (0x01)`: the admission handshake itself.
/// - `STREAM_ROUTE_REQUEST (0x05)`: passive/client request-only path — caller
///   is NEVER promoted to `state.peers`.
/// - `STREAM_TUNNEL_HTTP (0x04)`: passive SDK inference path for callers that
///   have an invite token but should not need a local `/v1` HTTP listener.
///
/// When a trust policy enforces ownership (`RequireOwned` or `Allowlist`), only
/// `STREAM_GOSSIP` bypasses the gate. Otherwise a leaked invite token is a
/// bearer credential for inference: a caller rejected by the trust gate (e.g.
/// `UntrustedOwner` under `Allowlist`) could still route requests via the
/// passive paths without ever being admitted. If a node enforces who may join,
/// the same enforcement must cover who may consume. `PreferOwned` remains
/// advisory and therefore preserves the passive-client behavior of `Off`.
///
/// Every other stream — including raw tunnel (0x02) — always requires the
/// remote to have completed gossip first.
pub fn stream_allowed_before_admission(stream_type: u8, trust_policy: TrustPolicy) -> bool {
    if stream_type == STREAM_GOSSIP {
        return true;
    }
    if matches!(
        trust_policy,
        TrustPolicy::RequireOwned | TrustPolicy::Allowlist
    ) {
        return false;
    }
    stream_type == STREAM_ROUTE_REQUEST || stream_type == STREAM_TUNNEL_HTTP
}

pub fn ingest_tunnel_map(
    remote: EndpointId,
    frame: &proto_node::TunnelMap,
    remote_tunnel_maps: &mut HashMap<EndpointId, HashMap<EndpointId, u16>>,
) -> anyhow::Result<()> {
    if frame.owner_peer_id.as_slice() != remote.as_bytes() {
        anyhow::bail!(
            "TunnelMap owner_peer_id mismatch: frame claims owner {}, but connected peer is {}",
            hex::encode(&frame.owner_peer_id),
            remote.fmt_short()
        );
    }

    let mut tunnel_map: HashMap<EndpointId, u16> = HashMap::new();
    for entry in &frame.entries {
        if entry.target_peer_id.len() != 32 {
            anyhow::bail!(
                "TunnelMap entry has invalid target_peer_id length: {} (expected 32)",
                entry.target_peer_id.len()
            );
        }
        if entry.tunnel_port > u16::MAX as u32 {
            anyhow::bail!(
                "TunnelMap entry has out-of-range tunnel_port: {} (max {})",
                entry.tunnel_port,
                u16::MAX
            );
        }
        let arr: [u8; 32] = entry.target_peer_id.as_slice().try_into().unwrap();
        let eid = EndpointId::from(
            PublicKey::from_bytes(&arr)
                .map_err(|e| anyhow::anyhow!("Invalid target_peer_id bytes: {e}"))?,
        );
        tunnel_map.insert(eid, entry.tunnel_port as u16);
    }

    remote_tunnel_maps.insert(remote, tunnel_map);
    Ok(())
}

/// Validates the sender-identity rule for a validated `PeerLeaving` frame.
/// Returns `Ok(leaving_id)` if `frame.peer_id == remote` (sender is announcing its own departure).
/// Returns `Err(ForgedSender)` if `frame.peer_id != remote` — no peer should be removed.
pub fn resolve_peer_leaving(
    remote: EndpointId,
    frame: &proto_node::PeerLeaving,
) -> Result<EndpointId, ControlFrameError> {
    if frame.peer_id.as_slice() != remote.as_bytes() {
        return Err(ControlFrameError::ForgedSender);
    }
    let arr: [u8; 32] =
        frame
            .peer_id
            .as_slice()
            .try_into()
            .map_err(|_| ControlFrameError::InvalidEndpointId {
                got: frame.peer_id.len(),
            })?;
    let pk = PublicKey::from_bytes(&arr).map_err(|_| ControlFrameError::InvalidEndpointId {
        got: frame.peer_id.len(),
    })?;
    Ok(EndpointId::from(pk))
}

/// Neutral peer state: identity, advertised capability, admission flag, and
/// the last-seen/mention bookkeeping the mesh keeps per peer. Serving-routing
/// projections that resolve raw model ids to public model ids stay in the host
/// as free functions, because they need `skippy_model_ref` and the host model
/// catalog.
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: EndpointId,
    pub addr: EndpointAddr,
    pub mesh_id: Option<String>,
    pub mesh_policy_hash: Option<String>,
    pub genesis_policy: Option<SignedMeshGenesisPolicy>,
    pub role: NodeRole,
    pub first_joined_mesh_ts: Option<u64>,
    pub models: Vec<String>,
    pub vram_bytes: u64,
    pub rtt_ms: Option<u32>,
    pub model_source: Option<String>,
    pub admitted: bool,
    /// All models assigned to this peer, even if not yet healthy.
    pub serving_models: Vec<String>,
    /// Models this node is actively routing inference for.
    pub hosted_models: Vec<String>,
    /// True when this peer explicitly advertised `hosted_models`.
    pub hosted_models_known: bool,
    /// All GGUFs on disk
    pub available_models: Vec<String>,
    /// Models this node has requested the mesh to serve
    pub requested_models: Vec<String>,
    /// Advisory canonical refs this peer wants the mesh to consider.
    pub explicit_model_interests: Vec<String>,
    /// Last time we directly communicated with this peer (gossip, heartbeat, tunnel).
    /// Only updated by direct bi-directional gossip exchanges, heartbeat probes,
    /// and inbound connections — never by transitive mentions.
    /// Used by PeerDown silencing to require independent proof-of-life.
    pub last_seen: std::time::Instant,
    /// Last time a bridge peer mentioned this peer in gossip.
    /// Updated on every transitive gossip update. Used together with `last_seen`
    /// for pruning and `collect_announcements`: a peer is included/kept as long
    /// as either timestamp is fresh.
    pub last_mentioned: std::time::Instant,
    /// mesh-llm version (e.g. "0.23.0")
    pub version: Option<String>,
    /// GPU name/model (e.g. "NVIDIA A100", "Apple M4 Max")
    pub gpu_name: Option<String>,
    /// Hostname of the node
    pub hostname: Option<String>,
    pub is_soc: Option<bool>,
    pub gpu_vram: Option<String>,
    pub gpu_reserved_bytes: Option<String>,
    /// Itemized view of `vram_bytes` when the peer advertised one.
    pub memory: Option<AdvertisedMemory>,
    pub gpu_mem_bandwidth_gbps: Option<String>,
    pub gpu_compute_tflops_fp32: Option<String>,
    pub gpu_compute_tflops_fp16: Option<String>,
    pub available_model_metadata: Vec<proto_node::CompactModelMetadata>,
    pub experts_summary: Option<proto_node::ExpertsSummary>,
    pub available_model_sizes: HashMap<String, u64>,
    pub served_model_descriptors: Vec<ServedModelDescriptor>,
    pub served_model_runtime: Vec<ModelRuntimeDescriptor>,
    pub owner_attestation: Option<SignedNodeOwnership>,
    pub release_attestation_summary: ReleaseAttestationSummary,
    pub artifact_transfer_supported: bool,
    pub stage_protocol_generation_supported: bool,
    pub stage_status_list_supported: bool,
    pub local_gguf_content_id_supported: bool,
    pub decode_batch_policy_supported: bool,
    pub advertised_model_throughput: Vec<ModelThroughputHint>,
    #[cfg(feature = "payments")]
    pub lightning_offers:
        std::collections::BTreeMap<String, mesh_llm_payments_types::pricing::Pricing>,
    pub cache_affinity: Option<mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement>,
    /// Most recent direct RTT sample for display purposes (refreshed periodically).
    pub display_rtt: Option<DirectLatencyObservation>,
    /// Last selected path observed on the mesh control connection to this peer.
    pub selected_path: Option<SelectedPathObservation>,
    /// Latency propagated via transitive gossip.
    pub propagated_latency: Option<PropagatedLatencyObservation>,
    pub owner_summary: OwnershipSummary,
    pub inference_admission_state: Option<proto_node::InferenceAdmissionState>,
}

impl PeerInfo {
    pub fn from_announcement(
        id: EndpointId,
        addr: EndpointAddr,
        ann: &PeerAnnouncement,
        owner_summary: OwnershipSummary,
    ) -> Self {
        Self {
            id,
            addr,
            mesh_id: ann.mesh_id.clone(),
            mesh_policy_hash: ann.mesh_policy_hash.clone(),
            genesis_policy: ann.genesis_policy.clone(),
            role: ann.role.clone(),
            first_joined_mesh_ts: ann.first_joined_mesh_ts,
            models: ann.models.clone(),
            vram_bytes: ann.vram_bytes,
            rtt_ms: None,
            model_source: ann.model_source.clone(),
            admitted: false,
            serving_models: ann.serving_models.clone(),
            hosted_models: ann.hosted_models.clone().unwrap_or_default(),
            hosted_models_known: ann.hosted_models.is_some(),
            available_models: ann.available_models.clone(),
            requested_models: ann.requested_models.clone(),
            explicit_model_interests: ann.explicit_model_interests.clone(),
            last_seen: std::time::Instant::now(),
            last_mentioned: std::time::Instant::now(),
            version: ann.version.clone(),
            gpu_name: ann.gpu_name.clone(),
            hostname: ann.hostname.clone(),
            is_soc: ann.is_soc,
            gpu_vram: ann.gpu_vram.clone(),
            gpu_reserved_bytes: ann.gpu_reserved_bytes.clone(),
            memory: ann.memory,
            gpu_mem_bandwidth_gbps: ann.gpu_mem_bandwidth_gbps.clone(),
            gpu_compute_tflops_fp32: ann.gpu_compute_tflops_fp32.clone(),
            gpu_compute_tflops_fp16: ann.gpu_compute_tflops_fp16.clone(),
            available_model_metadata: ann.available_model_metadata.clone(),
            experts_summary: ann.experts_summary.clone(),
            available_model_sizes: ann.available_model_sizes.clone(),
            served_model_descriptors: ann.served_model_descriptors.clone(),
            served_model_runtime: ann.served_model_runtime.clone(),
            owner_attestation: ann.owner_attestation.clone(),
            release_attestation_summary: verify_release_attestation(
                ann.release_attestation.as_ref(),
                &ReleaseSignerTrustStore::default(),
            ),
            artifact_transfer_supported: ann.artifact_transfer_supported,
            stage_protocol_generation_supported: ann.stage_protocol_generation_supported,
            stage_status_list_supported: ann.stage_status_list_supported,
            local_gguf_content_id_supported: ann.local_gguf_content_id_supported,
            decode_batch_policy_supported: ann.decode_batch_policy_supported,
            advertised_model_throughput: ann.advertised_model_throughput.clone(),
            #[cfg(feature = "payments")]
            lightning_offers: ann.lightning_offers.clone(),
            cache_affinity: ann.cache_affinity.clone(),
            display_rtt: None,
            selected_path: None,
            propagated_latency: None,
            owner_summary,
            inference_admission_state: ann.inference_admission_state,
        }
    }

    pub fn is_admitted(&self) -> bool {
        self.admitted
    }

    /// Return the most recent direct RTT sample for display, falling back to best-seen RTT.
    pub fn current_direct_rtt_ms(&self) -> Option<u32> {
        self.display_rtt.as_ref().map(|d| d.rtt_ms).or(self.rtt_ms)
    }

    pub fn split_stage_path_fallback(&self) -> Option<SelectedPathObservation> {
        let observation = self.selected_path?;
        if observation.path_type != "direct" {
            return Some(observation);
        }
        Some(SelectedPathObservation {
            rtt_ms: self.rtt_ms.or(observation.rtt_ms),
            ..observation
        })
    }

    /// Compute display latency from direct sample or propagated data.
    pub fn display_latency(&self) -> DisplayLatency {
        if let Some(ref direct) = self.display_rtt {
            return DisplayLatency {
                latency_ms: Some(direct.rtt_ms),
                source: DisplayLatencySource::Direct,
                age_ms: direct.observed_at.elapsed().as_millis() as u64,
                observer_id: None,
            };
        }
        if let Some(ref propagated) = self.propagated_latency {
            return DisplayLatency {
                latency_ms: Some(propagated.latency_ms),
                source: DisplayLatencySource::Estimated,
                age_ms: propagated.age_ms_at_received
                    + propagated.received_at.elapsed().as_millis() as u64,
                observer_id: propagated.observer_id,
            };
        }
        DisplayLatency {
            latency_ms: self.rtt_ms,
            source: DisplayLatencySource::Unknown,
            age_ms: 0,
            observer_id: None,
        }
    }

    pub fn is_assigned_model(&self, model: &str) -> bool {
        self.serving_models.iter().any(|m| m == model)
    }

    pub fn accepts_http_inference(&self) -> bool {
        matches!(self.role, NodeRole::Host { .. })
    }
}
