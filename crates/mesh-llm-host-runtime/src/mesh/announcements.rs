//! Building and merging peer announcements.
//!
//! A node turns its own state into a `PeerAnnouncement`, relays the ones it
//! holds, and folds the ones it receives back into `PeerInfo`. That is one
//! responsibility, distinct from the gossip transport in `gossip.rs`, which
//! owns connections, streams and the peer lifecycle. Nothing here touches the
//! wire: the protocol types and their encoding live in `crate::protocol`.

use super::{
    DisplayLatencySource, ModelDemand, ModelRuntimeDescriptor, Node, NodeRole, PeerAnnouncement,
    PeerInfo, ServedModelDescriptor, SignedNodeOwnership, infer_remote_served_descriptors,
};
use crate::mesh::cache_affinity_gossip;
use crate::mesh::peer_state::PropagatedLatencyObservation;
use crate::mesh::requirements::current_time_unix_ms;
use crate::models::append_external_inference_models;
use iroh::{EndpointAddr, EndpointId};
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
pub(super) fn version_allowed_for_rebroadcast(version: Option<&str>) -> bool {
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
pub(super) fn peer_is_idle_transitive_client(ann: &PeerAnnouncement) -> bool {
    let directly_measured = matches!(
        ann.latency_source,
        Some(crate::proto::node::LatencySource::Direct)
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

pub(crate) struct LocalAnnouncementData {
    role: NodeRole,
    first_joined_mesh_ts: Option<u64>,
    models: Vec<String>,
    model_source: Option<String>,
    serving_models: Vec<String>,
    hosted_models: Vec<String>,
    available_models: Vec<String>,
    requested_models: Vec<String>,
    explicit_model_interests: Vec<String>,
    model_demand: HashMap<String, ModelDemand>,
    mesh_id: Option<String>,
    mesh_policy_hash: Option<String>,
    signed_genesis_policy: Option<crate::SignedMeshGenesisPolicy>,
    release_attestation: Option<crate::ReleaseBuildAttestation>,
    direct_admission_proof: Option<crate::DirectNodeAdmissionProof>,
    available_model_metadata: Vec<crate::proto::node::CompactModelMetadata>,
    available_model_sizes: HashMap<String, u64>,
    served_model_descriptors: Vec<ServedModelDescriptor>,
    served_model_runtime: Vec<ModelRuntimeDescriptor>,
    owner_attestation: Option<SignedNodeOwnership>,
    artifact_transfer_supported: bool,
    advertised_model_throughput: Vec<crate::network::metrics::ModelThroughputHint>,
    cache_affinity: Option<mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement>,
    gpu_mem_bandwidth_gbps: Option<String>,
    gpu_compute_tflops_fp32: Option<String>,
    gpu_compute_tflops_fp16: Option<String>,
    inference_admission_state: Option<crate::proto::node::InferenceAdmissionState>,
}

pub(crate) struct RebroadcastAnnouncements {
    pub(super) announcements: Vec<PeerAnnouncement>,
    pub(super) filtered_old_version: usize,
}

pub fn backfill_legacy_descriptors(ann: &mut PeerAnnouncement) {
    if ann.served_model_descriptors.is_empty() {
        let primary_model_name = ann
            .serving_models
            .first()
            .map(String::as_str)
            .unwrap_or_default()
            .to_string();
        ann.served_model_descriptors = infer_remote_served_descriptors(
            &primary_model_name,
            &ann.serving_models,
            ann.model_source.as_deref(),
        );
    }
}

pub(super) fn peer_meaningfully_changed(old: &PeerInfo, new: &PeerInfo) -> bool {
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
        || cache_affinity_gossip::advertised_state_changed(&old.cache_affinity, &new.cache_affinity)
        || old.version != new.version
        || old.owner_summary != new.owner_summary
        || old.gpu_reserved_bytes != new.gpu_reserved_bytes
        || old.memory != new.memory
        || old.propagated_latency != new.propagated_latency
        || old.inference_admission_state != new.inference_admission_state
}

pub(crate) fn merge_first_joined_mesh_ts(existing: &mut Option<u64>, incoming: Option<u64>) {
    match (*existing, incoming) {
        (None, Some(v)) => *existing = Some(v),
        (Some(_), None) => {}
        (Some(a), Some(b)) => *existing = Some(a.min(b)),
        (None, None) => {}
    }
}

pub(super) fn apply_transitive_ann(
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
    existing.stage_protocol_generation_supported = ann.stage_protocol_generation_supported;
    existing.stage_status_list_supported = ann.stage_status_list_supported;
    // Strict local-source admission requires capability provenance from the
    // peer itself. A transitive announcer is not authoritative in either
    // direction, so it may neither promote nor clear this support bit. Direct
    // announcements in `add_peer` update it authoritatively.
    existing.advertised_model_throughput = ann.advertised_model_throughput.clone();
    cache_affinity_gossip::merge_advertisement(
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
            .unwrap_or(crate::proto::node::LatencySource::Unspecified);
        let is_propagatable_source = matches!(
            source,
            crate::proto::node::LatencySource::Direct
                | crate::proto::node::LatencySource::Estimated
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

impl Node {
    pub(crate) async fn collect_rebroadcast_announcements(
        &self,
        stale_cutoff: std::time::Instant,
    ) -> RebroadcastAnnouncements {
        let mut filtered_old_version = 0;
        let announcements = {
            let state = self.state.lock().await;
            state
                .peers
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
                .map(Self::announcement_from_peer)
                .collect()
        };
        RebroadcastAnnouncements {
            announcements,
            filtered_old_version,
        }
    }

    #[expect(
        clippy::cognitive_complexity,
        reason = "local gossip snapshots intentionally gather many independent advertised fields in one atomic view"
    )]
    pub(crate) async fn snapshot_local_announcement_data(&self) -> LocalAnnouncementData {
        let owner_summary = self.owner_summary.lock().await.clone();
        let plugin_models = self.plugin_inference_models().await;
        let mut models = self.models.lock().await.clone();
        append_external_inference_models(&mut models, &plugin_models);
        let mut serving_models = self.serving_models.lock().await.clone();
        append_external_inference_models(&mut serving_models, &plugin_models);
        let mut hosted_models = self.hosted_models.lock().await.clone();
        append_external_inference_models(&mut hosted_models, &plugin_models);
        let activity_advertisement = self
            .activity_policy_guard
            .advertisement_decision(self.public_mesh);
        if activity_advertisement.withdraw_model_availability {
            serving_models.clear();
            hosted_models.clear();
        }
        let advertised_model_throughput = self
            .routing_metrics
            .advertisable_model_throughput(&hosted_models);
        let now_unix_ms = current_time_unix_ms();
        let cache_affinity = cache_affinity_gossip::local_advertisement(
            &self.cache_affinity_inventory,
            self.endpoint.id().as_bytes(),
            now_unix_ms,
        );
        let release_attestation = self.release_attestation.lock().await.clone();
        let (mesh_id, mesh_policy_hash, signed_genesis_policy) =
            if let Some(state) = self.requirement_mesh_state.lock().await.clone() {
                (
                    Some(state.mesh_id),
                    Some(state.policy_hash),
                    state.signed_policy,
                )
            } else {
                (
                    self.mesh_id.lock().await.clone(),
                    self.mesh_policy_hash.lock().await.clone(),
                    self.signed_genesis_policy.lock().await.clone(),
                )
            };
        let direct_admission_proof = match (mesh_id.as_deref(), mesh_policy_hash.as_deref()) {
            (Some(mesh_id), Some(policy_hash)) => self.build_self_direct_admission_proof(
                mesh_id,
                policy_hash,
                release_attestation.as_ref(),
            ),
            _ => None,
        };
        LocalAnnouncementData {
            role: self.role.lock().await.clone(),
            first_joined_mesh_ts: *self.first_joined_mesh_ts.lock().await,
            models,
            model_source: self.model_source.lock().await.clone(),
            serving_models,
            hosted_models,
            available_models: self.available_models.lock().await.clone(),
            requested_models: self.requested_models.lock().await.clone(),
            explicit_model_interests: self.explicit_model_interests.lock().await.clone(),
            model_demand: self.get_demand(),
            mesh_id,
            mesh_policy_hash,
            signed_genesis_policy,
            release_attestation,
            direct_admission_proof,
            available_model_metadata: Vec::new(),
            available_model_sizes: HashMap::new(),
            served_model_descriptors: self.served_model_descriptors.lock().await.clone(),
            served_model_runtime: self.model_runtime_descriptors.lock().await.clone(),
            owner_attestation: self.owner_attestation.lock().await.clone(),
            artifact_transfer_supported:
                crate::models::artifact_transfer::artifact_transfer_advertised(&owner_summary),
            advertised_model_throughput,
            cache_affinity: Some(cache_affinity),
            gpu_mem_bandwidth_gbps: Self::format_optional_locked_f32_list(
                &self.gpu_mem_bandwidth_gbps,
            )
            .await,
            gpu_compute_tflops_fp32: Self::format_optional_locked_f32_list(
                &self.gpu_compute_tflops_fp32,
            )
            .await,
            gpu_compute_tflops_fp16: Self::format_optional_locked_f32_list(
                &self.gpu_compute_tflops_fp16,
            )
            .await,
            inference_admission_state: activity_advertisement.admission_state,
        }
    }

    pub(crate) async fn plugin_inference_models(&self) -> Vec<String> {
        let plugin_manager = self.plugin_manager.lock().await.clone();
        let Some(plugin_manager) = plugin_manager else {
            return Vec::new();
        };
        plugin_manager
            .inference_models()
            .await
            .unwrap_or_else(|error| {
                tracing::debug!(%error, "failed to collect plugin inference models for gossip");
                Vec::new()
            })
    }

    /// Builds the gossip announcement re-broadcast on behalf of an already
    /// admitted peer, from that peer's locally tracked state.
    pub(crate) fn announcement_from_peer(peer: &PeerInfo) -> PeerAnnouncement {
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
            advertised_model_throughput: peer.advertised_model_throughput.clone(),
            cache_affinity: peer.cache_affinity.clone(),
            latency_ms: latency.latency_ms,
            latency_source: Some(match latency.source {
                DisplayLatencySource::Direct => crate::proto::node::LatencySource::Direct,
                DisplayLatencySource::Estimated => crate::proto::node::LatencySource::Estimated,
                DisplayLatencySource::Unknown => crate::proto::node::LatencySource::Unknown,
            }),
            latency_age_ms: Some(latency.age_ms),
            latency_observer_id: latency.observer_id,
            inference_admission_state: peer.inference_admission_state,
            // No live-mesh claimed-log-head state is tracked on `PeerInfo` yet — a
            // rebroadcast of a peer we already admitted carries no opinion on
            // its claimed log head.
            claimed_log_head: None,
        }
    }

    pub(crate) async fn format_optional_locked_f32_list(
        values: &tokio::sync::Mutex<Option<Vec<f64>>>,
    ) -> Option<String> {
        values.lock().await.as_ref().map(|values| {
            values
                .iter()
                .map(|f| format!("{:.2}", f))
                .collect::<Vec<_>>()
                .join(",")
        })
    }

    /// Builds this node's own gossip announcement from freshly collected
    /// local data.
    pub(crate) fn build_local_announcement(&self, data: LocalAnnouncementData) -> PeerAnnouncement {
        PeerAnnouncement {
            addr: self.endpoint_addr_for_advertisement(),
            role: data.role,
            first_joined_mesh_ts: data.first_joined_mesh_ts,
            models: data.models,
            vram_bytes: self.vram_bytes,
            model_source: data.model_source,
            serving_models: data.serving_models,
            hosted_models: Some(data.hosted_models),
            available_models: data.available_models,
            requested_models: data.requested_models,
            explicit_model_interests: data.explicit_model_interests,
            version: Some(crate::VERSION.to_string()),
            model_demand: data.model_demand,
            mesh_id: data.mesh_id,
            mesh_policy_hash: data.mesh_policy_hash,
            gpu_name: self.enumerate_host.then(|| self.gpu_name.clone()).flatten(),
            hostname: self.enumerate_host.then(|| self.hostname.clone()).flatten(),
            is_soc: self.is_soc,
            gpu_vram: self.enumerate_host.then(|| self.gpu_vram.clone()).flatten(),
            gpu_reserved_bytes: self
                .enumerate_host
                .then(|| self.gpu_reserved_bytes.clone())
                .flatten(),
            memory: self.enumerate_host.then_some(self.advertised_memory),
            gpu_mem_bandwidth_gbps: data.gpu_mem_bandwidth_gbps,
            gpu_compute_tflops_fp32: data.gpu_compute_tflops_fp32,
            gpu_compute_tflops_fp16: data.gpu_compute_tflops_fp16,
            available_model_metadata: data.available_model_metadata,
            experts_summary: None,
            available_model_sizes: data.available_model_sizes,
            served_model_descriptors: data.served_model_descriptors,
            served_model_runtime: data.served_model_runtime,
            owner_attestation: data.owner_attestation,
            genesis_policy: data.signed_genesis_policy,
            release_attestation: data.release_attestation,
            direct_admission_proof: data.direct_admission_proof,
            artifact_transfer_supported: data.artifact_transfer_supported,
            stage_protocol_generation_supported: true,
            stage_status_list_supported: true,
            local_gguf_content_id_supported: true,
            advertised_model_throughput: data.advertised_model_throughput,
            cache_affinity: data.cache_affinity,
            latency_ms: None,
            latency_source: None,
            latency_age_ms: None,
            latency_observer_id: None,
            inference_admission_state: data.inference_admission_state,
            // No local claimed-log-head source is wired yet — this node never
            // advertises its own until a companion process is plumbed in.
            claimed_log_head: None,
        }
    }
}
