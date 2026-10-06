//! Announcement admission filters, peer-state merges and rebroadcast projection.
//!
//! Local model/plugin snapshots and gossip transport stay with the host.

use crate::peer_state::{
    DisplayLatencySource, PeerAnnouncement, PeerInfo, PropagatedLatencyObservation,
};
use crate::state::MembershipState;
use crate::types::NodeRole;
use iroh::{EndpointAddr, EndpointId};
use mesh_llm_identity::OwnershipSummary;
use std::collections::HashMap;

/// Minimum peer version we accept into the local mesh table and re-broadcast.
///
/// Peers below this floor are rejected at ingest in both `add_peer`
/// (direct gossip exchange) and `update_transitive_peer` (gossip relayed
/// by a bridge peer). They do not appear in `/api/status`, do not appear
/// in the UI, and are not included in outbound gossip. A peer that updates
/// and re-announces with a version at or above the floor is accepted on
/// the next exchange.
///
/// v0.60.0 is the cut where the on-wire `hardware` block landed; peers
/// older than that predate several gossip fields the current mesh relies
/// on. Peers that don't advertise a version at all (some legacy nodes
/// leave the field unset) are conservatively accepted, on the theory that
/// a missing version is more likely to be a legitimate old node than a
/// targeted bypass.
const MIN_REBROADCAST_VERSION_MAJOR: u64 = 0;
const MIN_REBROADCAST_VERSION_MINOR: u64 = 60;

/// Returns `true` if `version` is recent enough to include in outbound
/// gossip. `None` (no advertised version) returns `true` for back-compat.
/// Build metadata after `+` is stripped before parsing.
pub fn version_allowed_for_rebroadcast(version: Option<&str>) -> bool {
    let Some(raw) = version else {
        return true;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return true;
    }
    // Strip build metadata ("0.65.1+skippy.20260504.kv.2" → "0.65.1") and
    // pre-release tag ("0.63.0-rc5" → "0.63.0") so the comparison is
    // purely on the major.minor numeric pair.
    let core = trimmed
        .split('+')
        .next()
        .unwrap_or(trimmed)
        .split('-')
        .next()
        .unwrap_or(trimmed);
    let mut parts = core.split('.');
    let Some(major) = parts.next().and_then(|s| s.parse::<u64>().ok()) else {
        return true; // Unparseable — don't penalise; conservative default.
    };
    let Some(minor) = parts.next().and_then(|s| s.parse::<u64>().ok()) else {
        return true;
    };
    if major != MIN_REBROADCAST_VERSION_MAJOR {
        // Any major > floor (e.g. v1.x.y) is allowed; any major < floor is
        // refused. With MIN_REBROADCAST_VERSION_MAJOR == 0, the "less than"
        // case cannot occur, but we keep the comparison structure for the
        // day the floor bumps to a non-zero major.
        return major > MIN_REBROADCAST_VERSION_MAJOR;
    }
    minor >= MIN_REBROADCAST_VERSION_MINOR
}

/// Returns `true` if the announcement describes a peer the mesh has no
/// observable use for via transitive gossip: a `Client`-role peer that
/// advertises **no identity** (no hostname), has **never been directly
/// measured** by any peer in the mesh, and has **no model interests**
/// (no requested/serving/hosted models).
///
/// Three independent signals must all be absent before we treat a peer
/// as a gossip-only ghost:
///
/// 1. `hostname` — populated synchronously by `system::hardware::survey()`
///    at node construction. Every real client on every supported platform
///    has one from its first gossip frame.
///
/// 2. `latency_source == Direct` — set when *any* peer in the mesh has
///    measured this peer's RTT via direct contact, then propagated
///    through gossip. A peer with a direct measurement is real — someone
///    reached it on the network. The v0.57 swarm uniformly has
///    `latency_source = Unknown`; no peer has ever directly contacted
///    one.
///
/// 3. model interests (`requested`/`serving`/`hosted`) — any of these
///    being populated makes the peer useful to the mesh (demand signal
///    or routable capacity).
///
/// A peer that fails all three is invisible to routing, untraceable on
/// the network, and contributes no demand signal. Real idle clients
/// survive: they have a hostname. Real reachable clients survive: they
/// have a direct measurement. Real demand-signaling clients survive:
/// they have a requested model.
///
/// Direct ingest in `add_peer` ignores this check — a client we actually
/// connect to is admitted regardless of what they advertise.
pub fn peer_is_idle_transitive_client(ann: &PeerAnnouncement) -> bool {
    let directly_measured = matches!(
        ann.latency_source,
        Some(mesh_llm_protocol::proto::node::LatencySource::Direct)
    );
    matches!(ann.role, NodeRole::Client)
        && ann.hostname.is_none()
        && !directly_measured
        && ann.requested_models.is_empty()
        && ann.serving_models.is_empty()
        && ann
            .hosted_models
            .as_ref()
            .map(|h| h.is_empty())
            .unwrap_or(true)
}

pub struct RebroadcastAnnouncements {
    pub announcements: Vec<PeerAnnouncement>,
    pub filtered_old_version: usize,
}

#[cfg(feature = "payments")]
fn lightning_offers_changed(old: &PeerInfo, new: &PeerInfo) -> bool {
    old.lightning_offers != new.lightning_offers
}

#[cfg(not(feature = "payments"))]
fn lightning_offers_changed(_old: &PeerInfo, _new: &PeerInfo) -> bool {
    false
}

pub fn peer_meaningfully_changed(old: &PeerInfo, new: &PeerInfo) -> bool {
    old.addr != new.addr
        || old.mesh_id != new.mesh_id
        || old.mesh_policy_hash != new.mesh_policy_hash
        || old.genesis_policy != new.genesis_policy
        || old.role != new.role
        || old.first_joined_mesh_ts != new.first_joined_mesh_ts
        || old.models != new.models
        || old.vram_bytes != new.vram_bytes
        || old.rtt_ms != new.rtt_ms
        || old.model_source != new.model_source
        || old.serving_models != new.serving_models
        || old.hosted_models_known != new.hosted_models_known
        || old.hosted_models != new.hosted_models
        || old.available_models != new.available_models
        || old.requested_models != new.requested_models
        || old.explicit_model_interests != new.explicit_model_interests
        || old.served_model_descriptors != new.served_model_descriptors
        || old.served_model_runtime != new.served_model_runtime
        || old.artifact_transfer_supported != new.artifact_transfer_supported
        || old.stage_protocol_generation_supported != new.stage_protocol_generation_supported
        || old.stage_status_list_supported != new.stage_status_list_supported
        || old.local_gguf_content_id_supported != new.local_gguf_content_id_supported
        || old.decode_batch_policy_supported != new.decode_batch_policy_supported
        || lightning_offers_changed(old, new)
        || crate::advertised_state_changed(&old.cache_affinity, &new.cache_affinity)
        || old.version != new.version
        || old.owner_summary != new.owner_summary
        || old.gpu_reserved_bytes != new.gpu_reserved_bytes
        || old.memory != new.memory
        || old.propagated_latency != new.propagated_latency
        || old.inference_admission_state != new.inference_admission_state
}

pub fn merge_first_joined_mesh_ts(existing: &mut Option<u64>, incoming: Option<u64>) {
    match (*existing, incoming) {
        (None, Some(v)) => *existing = Some(v),
        (Some(_), None) => {}
        (Some(a), Some(b)) => *existing = Some(a.min(b)),
        (None, None) => {}
    }
}

pub fn apply_transitive_ann(
    existing: &mut PeerInfo,
    addr: &EndpointAddr,
    ann: &PeerAnnouncement,
    bridge_id: EndpointId,
) -> bool {
    let ann_hosted_models = ann.hosted_models.clone().unwrap_or_default();
    existing.mesh_id = ann.mesh_id.clone();
    existing.mesh_policy_hash = ann.mesh_policy_hash.clone();
    existing.genesis_policy = ann.genesis_policy.clone();
    let serving_changed = existing.serving_models != ann.serving_models
        || existing.hosted_models != ann_hosted_models
        || existing.hosted_models_known != ann.hosted_models.is_some();
    existing.serving_models = ann.serving_models.clone();
    existing.hosted_models = ann_hosted_models;
    existing.hosted_models_known = ann.hosted_models.is_some();
    existing.role = ann.role.clone();
    merge_first_joined_mesh_ts(&mut existing.first_joined_mesh_ts, ann.first_joined_mesh_ts);
    let capacity_changed = existing.vram_bytes != ann.vram_bytes;
    existing.vram_bytes = ann.vram_bytes;
    // Only advance addr if the transitive announcement is at least as path-rich,
    // so a direct peer's richer address is not overwritten by a weaker transitive one.
    if !addr.addrs.is_empty() && addr.addrs.len() >= existing.addr.addrs.len() {
        existing.addr = addr.clone();
    }
    if ann.version.is_some() {
        existing.version = ann.version.clone();
    }
    if ann.gpu_name.is_some() {
        existing.gpu_name = ann.gpu_name.clone();
    }
    if ann.hostname.is_some() {
        existing.hostname = ann.hostname.clone();
    }
    if ann.is_soc.is_some() {
        existing.is_soc = ann.is_soc;
    }
    if ann.gpu_vram.is_some() {
        existing.gpu_vram = ann.gpu_vram.clone();
    }
    if ann.gpu_reserved_bytes.is_some() {
        existing.gpu_reserved_bytes = ann.gpu_reserved_bytes.clone();
    }
    match ann.memory {
        Some(memory) => existing.memory = Some(memory),
        // A relay that predates the block strips it. The cached block only
        // explains the capacity it arrived with: keep it while that capacity
        // is unchanged, drop it once the capacity moved, so a stale breakdown
        // is never paired with the new budget and rebroadcast as such.
        None if capacity_changed => existing.memory = None,
        None => {}
    }
    if ann.gpu_mem_bandwidth_gbps.is_some() {
        existing.gpu_mem_bandwidth_gbps = ann.gpu_mem_bandwidth_gbps.clone();
    }
    if ann.gpu_compute_tflops_fp32.is_some() {
        existing.gpu_compute_tflops_fp32 = ann.gpu_compute_tflops_fp32.clone();
    }
    if ann.gpu_compute_tflops_fp16.is_some() {
        existing.gpu_compute_tflops_fp16 = ann.gpu_compute_tflops_fp16.clone();
    }
    existing.models = ann.models.clone();
    existing.available_models.clear();
    existing.requested_models = ann.requested_models.clone();
    existing.explicit_model_interests = ann.explicit_model_interests.clone();
    existing.owner_attestation = ann.owner_attestation.clone();
    if ann.model_source.is_some() {
        existing.model_source = ann.model_source.clone();
    }
    existing.served_model_descriptors = ann.served_model_descriptors.clone();
    existing.served_model_runtime = ann.served_model_runtime.clone();
    existing.artifact_transfer_supported = ann.artifact_transfer_supported;
    existing.stage_status_list_supported = ann.stage_status_list_supported;
    // `stage_protocol_generation_supported` and
    // `local_gguf_content_id_supported` require capability provenance from the
    // peer itself. A transitive announcer is not authoritative in either
    // direction, so it may neither promote nor clear them. Direct announcements
    // update both fields authoritatively.
    existing.advertised_model_throughput = ann.advertised_model_throughput.clone();
    #[cfg(feature = "payments")]
    {
        existing.lightning_offers = ann.lightning_offers.clone();
    }
    crate::merge_advertisement(
        &mut existing.cache_affinity,
        ann.cache_affinity.as_ref(),
        false,
    );
    if ann.inference_admission_state.is_some() {
        existing.inference_admission_state = ann.inference_admission_state;
    }
    if ann.experts_summary.is_some() {
        existing.experts_summary = ann.experts_summary.clone();
    }
    // Propagate latency from the announcement (transitive gossip).
    if let Some(latency_ms) = ann.latency_ms {
        let source = ann
            .latency_source
            .unwrap_or(mesh_llm_protocol::proto::node::LatencySource::Unspecified);
        let is_propagatable_source = matches!(
            source,
            mesh_llm_protocol::proto::node::LatencySource::Direct
                | mesh_llm_protocol::proto::node::LatencySource::Estimated
        );
        if latency_ms > 0 && is_propagatable_source {
            let observer_id = ann
                .latency_observer_id
                .as_ref()
                .and_then(|id_bytes| EndpointId::from_bytes(id_bytes).ok());
            existing.propagated_latency = Some(PropagatedLatencyObservation {
                latency_ms,
                age_ms_at_received: ann.latency_age_ms.unwrap_or(0),
                received_at: std::time::Instant::now(),
                observer_id: observer_id.or(Some(bridge_id)),
            });
        }
    }
    serving_changed
}

pub fn announcement_from_peer(peer: &PeerInfo) -> PeerAnnouncement {
    let latency = peer.display_latency();
    PeerAnnouncement {
        addr: peer.addr.clone(),
        role: peer.role.clone(),
        first_joined_mesh_ts: peer.first_joined_mesh_ts,
        models: peer.models.clone(),
        vram_bytes: peer.vram_bytes,
        model_source: peer.model_source.clone(),
        serving_models: peer.serving_models.clone(),
        hosted_models: peer.hosted_models_known.then(|| peer.hosted_models.clone()),
        available_models: peer.available_models.clone(),
        requested_models: peer.requested_models.clone(),
        explicit_model_interests: peer.explicit_model_interests.clone(),
        version: peer.version.clone(),
        model_demand: HashMap::new(),
        mesh_id: peer.mesh_id.clone(),
        mesh_policy_hash: peer.mesh_policy_hash.clone(),
        gpu_name: peer.gpu_name.clone(),
        hostname: peer.hostname.clone(),
        is_soc: peer.is_soc,
        gpu_vram: peer.gpu_vram.clone(),
        gpu_reserved_bytes: peer.gpu_reserved_bytes.clone(),
        memory: peer.memory,
        gpu_mem_bandwidth_gbps: peer.gpu_mem_bandwidth_gbps.clone(),
        gpu_compute_tflops_fp32: peer.gpu_compute_tflops_fp32.clone(),
        gpu_compute_tflops_fp16: peer.gpu_compute_tflops_fp16.clone(),
        available_model_metadata: peer.available_model_metadata.clone(),
        experts_summary: peer.experts_summary.clone(),
        available_model_sizes: peer.available_model_sizes.clone(),
        served_model_descriptors: peer.served_model_descriptors.clone(),
        served_model_runtime: peer.served_model_runtime.clone(),
        owner_attestation: peer.owner_attestation.clone(),
        genesis_policy: peer.genesis_policy.clone(),
        release_attestation: None,
        direct_admission_proof: None,
        artifact_transfer_supported: peer.artifact_transfer_supported,
        stage_protocol_generation_supported: peer.stage_protocol_generation_supported,
        stage_status_list_supported: peer.stage_status_list_supported,
        local_gguf_content_id_supported: peer.local_gguf_content_id_supported,
        decode_batch_policy_supported: peer.decode_batch_policy_supported,
        advertised_model_throughput: peer.advertised_model_throughput.clone(),
        #[cfg(feature = "payments")]
        lightning_offers: peer.lightning_offers.clone(),
        cache_affinity: peer.cache_affinity.clone(),
        latency_ms: latency.latency_ms,
        latency_source: Some(match latency.source {
            DisplayLatencySource::Direct => mesh_llm_protocol::proto::node::LatencySource::Direct,
            DisplayLatencySource::Estimated => {
                mesh_llm_protocol::proto::node::LatencySource::Estimated
            }
            DisplayLatencySource::Unknown => mesh_llm_protocol::proto::node::LatencySource::Unknown,
        }),
        latency_age_ms: Some(latency.age_ms),
        latency_observer_id: latency.observer_id,
        inference_admission_state: peer.inference_admission_state,
        // Claimed log heads are not retained in admitted peer state.
        claimed_log_head: None,
    }
}

pub fn peer_hardware_changed(old_peer: &PeerInfo, updated_peer: &PeerInfo) -> bool {
    old_peer.gpu_name != updated_peer.gpu_name
        || old_peer.hostname != updated_peer.hostname
        || old_peer.is_soc != updated_peer.is_soc
        || old_peer.gpu_vram != updated_peer.gpu_vram
        || old_peer.gpu_reserved_bytes != updated_peer.gpu_reserved_bytes
        || old_peer.gpu_mem_bandwidth_gbps != updated_peer.gpu_mem_bandwidth_gbps
        || old_peer.gpu_compute_tflops_fp32 != updated_peer.gpu_compute_tflops_fp32
        || old_peer.gpu_compute_tflops_fp16 != updated_peer.gpu_compute_tflops_fp16
}

pub fn update_existing_direct_peer(
    existing: &mut PeerInfo,
    addr: EndpointAddr,
    ann: &PeerAnnouncement,
    owner_summary: OwnershipSummary,
    now: std::time::Instant,
) -> (PeerInfo, bool, bool, bool) {
    let old_peer = existing.clone();
    let role_changed = existing.role != ann.role;
    let ann_hosted_models = ann.hosted_models.clone().unwrap_or_default();
    let serving_changed = existing.serving_models != ann.serving_models
        || existing.hosted_models != ann_hosted_models
        || existing.hosted_models_known != ann.hosted_models.is_some();
    existing.admitted = true;
    existing.mesh_id = ann.mesh_id.clone();
    existing.mesh_policy_hash = ann.mesh_policy_hash.clone();
    existing.genesis_policy = ann.genesis_policy.clone();
    if role_changed {
        tracing::info!(
            target: "mesh_llm_host_runtime::mesh::gossip",
            "Peer {} role updated: {:?} → {:?}",
            existing.id.fmt_short(),
            existing.role,
            ann.role
        );
        existing.role = ann.role.clone();
    }
    if !addr.addrs.is_empty() {
        existing.addr = addr;
    }
    existing.models = ann.models.clone();
    merge_first_joined_mesh_ts(&mut existing.first_joined_mesh_ts, ann.first_joined_mesh_ts);
    existing.vram_bytes = ann.vram_bytes;
    if ann.model_source.is_some() {
        existing.model_source = ann.model_source.clone();
    }
    existing.serving_models = ann.serving_models.clone();
    existing.hosted_models = ann_hosted_models;
    existing.hosted_models_known = ann.hosted_models.is_some();
    existing.available_models.clear();
    existing
        .available_models
        .extend(ann.available_models.clone());
    existing.requested_models = ann.requested_models.clone();
    existing.explicit_model_interests = ann.explicit_model_interests.clone();
    existing.last_seen = now;
    existing.owner_attestation = ann.owner_attestation.clone();
    existing.owner_summary = owner_summary;
    existing.served_model_descriptors = ann.served_model_descriptors.clone();
    existing.served_model_runtime = ann.served_model_runtime.clone();
    existing.artifact_transfer_supported = ann.artifact_transfer_supported;
    existing.stage_protocol_generation_supported = ann.stage_protocol_generation_supported;
    existing.stage_status_list_supported = ann.stage_status_list_supported;
    existing.local_gguf_content_id_supported = ann.local_gguf_content_id_supported;
    existing.decode_batch_policy_supported = ann.decode_batch_policy_supported;
    existing.advertised_model_throughput = ann.advertised_model_throughput.clone();
    #[cfg(feature = "payments")]
    {
        existing.lightning_offers = ann.lightning_offers.clone();
    }
    crate::merge_advertisement(
        &mut existing.cache_affinity,
        ann.cache_affinity.as_ref(),
        true,
    );
    existing.inference_admission_state = ann.inference_admission_state;
    if ann.version.is_some() {
        existing.version = ann.version.clone();
    }
    existing.gpu_name = ann.gpu_name.clone();
    existing.hostname = ann.hostname.clone();
    existing.is_soc = ann.is_soc;
    existing.gpu_vram = ann.gpu_vram.clone();
    existing.gpu_reserved_bytes = ann.gpu_reserved_bytes.clone();
    existing.memory = ann.memory;
    existing.gpu_mem_bandwidth_gbps = ann.gpu_mem_bandwidth_gbps.clone();
    existing.gpu_compute_tflops_fp32 = ann.gpu_compute_tflops_fp32.clone();
    existing.gpu_compute_tflops_fp16 = ann.gpu_compute_tflops_fp16.clone();
    if ann.experts_summary.is_some() {
        existing.experts_summary = ann.experts_summary.clone();
    }
    existing.release_attestation_summary = crate::release_attestation::verify_release_attestation(
        ann.release_attestation.as_ref(),
        &crate::release_attestation::ReleaseSignerTrustStore::default(),
    );
    let updated_peer = existing.clone();
    let changed = peer_meaningfully_changed(&old_peer, &updated_peer)
        || peer_hardware_changed(&old_peer, &updated_peer);
    (updated_peer, changed, role_changed, serving_changed)
}

impl MembershipState {
    pub fn collect_rebroadcast_announcements(
        &self,
        stale_cutoff: std::time::Instant,
    ) -> RebroadcastAnnouncements {
        let mut filtered_old_version = 0;
        let announcements = {
            self.peers
                .values()
                .filter(|peer| {
                    peer.last_seen >= stale_cutoff || peer.last_mentioned >= stale_cutoff
                })
                .filter(|peer| {
                    let allowed = version_allowed_for_rebroadcast(peer.version.as_deref());
                    if !allowed {
                        filtered_old_version += 1;
                    }
                    allowed
                })
                .map(announcement_from_peer)
                .collect()
        };
        RebroadcastAnnouncements {
            announcements,
            filtered_old_version,
        }
    }
}

#[cfg(test)]
mod tests;
