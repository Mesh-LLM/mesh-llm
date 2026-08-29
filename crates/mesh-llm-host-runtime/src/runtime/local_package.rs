use super::split_planning::{
    RuntimeSliceStagePlan, split_participant_exclusion_labels, split_participant_labels,
};
#[cfg(test)]
use super::split_planning::{split_stage_plan_labels, validate_split_capacity};
use crate::inference::{election, skippy};
use crate::mesh::{self, NodeRole};
use crate::models;
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::Path;

pub(super) const SPLIT_DEFAULT_MIN_PARTICIPANTS: usize = 2;

/// Try to extract GGUF architecture metadata from a layer package's shared
/// metadata file.  Layer packages store a `shared/metadata.gguf` that carries
/// the model's KV pairs (context_length, head counts, etc.) without any tensor
/// data.  This gives the context planner the information it needs for accurate
/// KV cache budget calculations on split models.
pub(super) fn scan_layer_package_metadata(
    package: &skippy::SkippyPackageIdentity,
) -> Option<models::gguf::GgufCompactMeta> {
    // Runtime-slice packages point straight at a cached GGUF. Resolve it
    // before attempting the layer-package-only shared metadata lookup.
    if package.source_model_path.is_file() {
        return models::gguf::scan_gguf_compact_meta(&package.source_model_path);
    }

    // The source_model_path in a layer package identity points to the original
    // GGUF.  But for HF layer packages the source model is not downloaded
    // locally.  Instead, look for the shared metadata file in the package dir.
    //
    // The package_ref looks like "hf://meshllm/Qwen3-layers@rev" which resolves
    // to a local cache directory.  Try to find shared/metadata.gguf there.
    if let Ok(local_ref) =
        skippy::resolve_hf_package_to_local(&package.package_ref, 0, 0, false, false)
    {
        let metadata_path = std::path::Path::new(&local_ref).join("shared/metadata.gguf");
        if metadata_path.is_file() {
            return models::gguf::scan_gguf_compact_meta(&metadata_path);
        }
    }
    None
}

pub(super) fn runtime_model_planning_bytes(model_path: &Path) -> Result<u64> {
    let package_ref = model_path.to_string_lossy().to_string();
    if skippy::is_layer_package_ref(&package_ref) {
        return Ok(skippy::identity_from_layer_package(&package_ref)?.source_model_bytes);
    }
    Ok(election::total_model_bytes(model_path))
}
pub(super) async fn split_runtime_compact_meta(
    package: &skippy::SkippyPackageIdentity,
) -> Result<models::gguf::GgufCompactMeta> {
    let package = package.clone();
    tokio::task::spawn_blocking(move || scan_layer_package_metadata(&package))
        .await
        .ok()
        .flatten()
        .context("split topology planning requires GGUF metadata")
}

pub(super) fn split_runtime_kv_bytes_per_token(
    package: &skippy::SkippyPackageIdentity,
    compact_meta: &models::gguf::GgufCompactMeta,
    model_ref: &str,
    cache_type_k_override: Option<&str>,
    cache_type_v_override: Option<&str>,
) -> Result<u64> {
    let kv_cache_quant = split_effective_kv_cache_quant(
        package,
        compact_meta,
        model_ref,
        cache_type_k_override,
        cache_type_v_override,
    );
    kv_cache_quant
        .kv_cache_bytes_per_token(compact_meta)
        .context("split topology planning requires KV cache byte metadata")
}

/// Resolve the K/V cache types that split stages will actually load with.
///
/// Stage loading applies the family default (for example Inkling's Q4_0 K/V)
/// ahead of the size-tiered `KvCachePolicy`. Planning must resolve K/V the same
/// way, or it budgets for a cheaper cache than the stages allocate and
/// over-packs the topology into an out-of-memory load.
pub(super) fn split_effective_kv_cache_quant(
    package: &skippy::SkippyPackageIdentity,
    compact_meta: &models::gguf::GgufCompactMeta,
    model_ref: &str,
    cache_type_k_override: Option<&str>,
    cache_type_v_override: Option<&str>,
) -> models::gguf::GgufKvCacheQuant {
    // Guard the size-tiered default against the model's quantised-KV
    // compatibility (Flash Attention / block alignment) so planning budgets for
    // the same cache the stage load can actually allocate. Explicit overrides
    // below are never guarded — an override that cannot load must fail loudly.
    let size_policy = skippy::KvCachePolicy::for_model_size(package.source_model_bytes)
        .guarded_for_model(Some(compact_meta));
    let family_default =
        skippy::family_policy_for_compact_meta(compact_meta, Some(model_ref)).default_kv_cache_type;

    // Explicit user overrides win, then the family default, then model size.
    let effective_k = cache_type_k_override
        .or(family_default)
        .unwrap_or(size_policy.cache_type_k());
    let effective_v = cache_type_v_override
        .or(family_default)
        .unwrap_or(size_policy.cache_type_v());

    models::gguf::GgufKvCacheQuant::from_llama_args(effective_k, effective_v).unwrap_or_else(|| {
        split_kv_cache_quant(&size_policy, cache_type_k_override, cache_type_v_override)
    })
}
pub(super) async fn resolve_split_runtime_package(
    model_path: &Path,
    model_ref: &str,
) -> Result<skippy::SkippyPackageIdentity> {
    let model_path_str = model_path.to_string_lossy().to_string();
    if skippy::is_layer_package_ref(&model_path_str) {
        Ok(tokio::task::spawn_blocking(move || {
            skippy::identity_from_layer_package(&model_path_str)
        })
        .await
        .context("join identify skippy layer package task")??)
    } else {
        Ok(skippy::synthetic_direct_gguf_package(
            model_ref, model_path,
        )?)
    }
}

pub(super) fn split_kv_cache_quant(
    split_kv_policy: &skippy::KvCachePolicy,
    cache_type_k_override: Option<&str>,
    cache_type_v_override: Option<&str>,
) -> models::gguf::GgufKvCacheQuant {
    let policy_quant = models::gguf::GgufKvCacheQuant::from_llama_args(
        split_kv_policy.cache_type_k(),
        split_kv_policy.cache_type_v(),
    )
    .unwrap_or(models::gguf::GgufKvCacheQuant::Q8_0);

    match (cache_type_k_override, cache_type_v_override) {
        (None, None) => policy_quant,
        (k_override, v_override) => models::gguf::GgufKvCacheQuant::from_llama_args(
            k_override.unwrap_or(split_kv_policy.cache_type_k()),
            v_override.unwrap_or(split_kv_policy.cache_type_v()),
        )
        .unwrap_or(policy_quant),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitParticipantSnapshot {
    pub(super) participants: Vec<SplitParticipant>,
    pub(super) excluded: Vec<SplitParticipantExclusion>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitParticipantExclusion {
    pub(super) node_id: iroh::EndpointId,
    pub(super) reason: SplitParticipantExclusionReason,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SplitParticipantExclusionReason {
    Client,
    MissingVram,
    MissingModelInterest,
    StageProtocolGeneration,
    StageControlUnreachable,
    ArtifactTransferUnavailable,
    StageInventoryEmpty,
    PackageManifestMismatch,
    MissingModelSource,
}

impl SplitParticipantExclusionReason {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Client => "client",
            Self::MissingVram => "missing_vram",
            Self::MissingModelInterest => "missing_model_interest",
            Self::StageProtocolGeneration => "stage_protocol_generation",
            Self::StageControlUnreachable => "stage_control_unreachable",
            Self::ArtifactTransferUnavailable => "artifact_transfer_unavailable",
            Self::StageInventoryEmpty => "stage_inventory_empty",
            Self::PackageManifestMismatch => "package_manifest_mismatch",
            Self::MissingModelSource => "missing_model_source",
        }
    }

    pub(super) const fn recommendation(self) -> &'static str {
        match self {
            Self::Client => "Run this peer in serve mode if it should contribute compute.",
            Self::MissingVram => {
                "Check GPU visibility or lower --max-vram only after confirming backend/device detection."
            }
            Self::MissingModelInterest => {
                "Start the peer with the same --model value or explicit split model interest."
            }
            Self::StageProtocolGeneration => {
                "Upgrade this peer so it advertises current stage protocol support."
            }
            Self::StageControlUnreachable => {
                "Check stage-control connectivity and peer runtime logs before retrying split serving."
            }
            Self::ArtifactTransferUnavailable => {
                "Enable artifact transfer, use an HF-resolvable package, or choose a peer with the package already cached."
            }
            Self::StageInventoryEmpty => {
                "Wait for stage inventory refresh or prepare the requested package on this peer."
            }
            Self::PackageManifestMismatch => {
                "Refresh stale layer packages so this peer advertises the requested package manifest."
            }
            Self::MissingModelSource => {
                "Start the peer with a resolvable package source or wait for stage inventory to prove the package is available."
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitParticipantBlockerSummary {
    reason: &'static str,
    count: usize,
    short_node_ids: Vec<String>,
    recommendation: &'static str,
}

type SplitParticipantSignature = Vec<(String, u64, u64, u64, Option<u32>, bool, u32)>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SplitParticipant {
    pub(super) node_id: iroh::EndpointId,
    pub(super) vram_bytes: u64,
    first_joined_mesh_ts: Option<u64>,
    pub(super) cached_slice_bytes: u64,
    pub(super) missing_artifact_bytes: u64,
    pub(super) rtt_ms: Option<u32>,
    pub(super) artifact_transfer_supported: bool,
    availability_score: u32,
}

impl SplitParticipant {
    pub(super) fn new(
        node_id: iroh::EndpointId,
        vram_bytes: u64,
        first_joined_mesh_ts: Option<u64>,
    ) -> Self {
        Self {
            node_id,
            vram_bytes,
            first_joined_mesh_ts,
            cached_slice_bytes: 0,
            missing_artifact_bytes: 0,
            rtt_ms: None,
            artifact_transfer_supported: false,
            availability_score: 0,
        }
    }

    pub(super) fn local_package(
        node_id: iroh::EndpointId,
        vram_bytes: u64,
        first_joined_mesh_ts: Option<u64>,
        package: &skippy::SkippyPackageIdentity,
    ) -> Self {
        let mut participant = Self::new(node_id, vram_bytes, first_joined_mesh_ts);
        participant.cached_slice_bytes = package.source_model_bytes;
        participant.artifact_transfer_supported = true;
        participant.availability_score = package.layer_count;
        participant
    }

    pub(super) fn with_package_signals(
        mut self,
        signal: SplitParticipantPackageSignal,
        rtt_ms: Option<u32>,
        artifact_transfer_supported: bool,
    ) -> Self {
        self.cached_slice_bytes = signal.cached_slice_bytes;
        self.missing_artifact_bytes = signal.missing_artifact_bytes;
        self.availability_score = signal.availability_score;
        self.rtt_ms = rtt_ms;
        self.artifact_transfer_supported = artifact_transfer_supported;
        self
    }

    #[cfg(test)]
    pub(super) fn to_topology_participant(self) -> skippy::StageTopologyParticipant {
        skippy::StageTopologyParticipant {
            node_id: self.node_id,
            vram_bytes: self.vram_bytes,
            cached_slice_bytes: self.cached_slice_bytes,
            missing_artifact_bytes: self.missing_artifact_bytes,
            rtt_ms: self.rtt_ms,
            artifact_transfer_supported: self.artifact_transfer_supported,
            availability_score: self.availability_score,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SplitParticipantPackageSignal {
    pub(super) cached_slice_bytes: u64,
    pub(super) missing_artifact_bytes: u64,
    pub(super) availability_score: u32,
}

impl SplitParticipantPackageSignal {
    pub(super) fn can_stage_with(
        self,
        package: &skippy::SkippyPackageIdentity,
        artifact_transfer_supported: bool,
    ) -> bool {
        self.missing_artifact_bytes == 0
            || artifact_transfer_supported
            || package_ref_has_independent_prepare_source(&package.package_ref)
    }
}

pub(super) fn package_ref_has_independent_prepare_source(package_ref: &str) -> bool {
    // HF layer packages can be resolved by the selected worker during prepare;
    // peer artifact transfer is only an optional cache warm path.
    skippy_runtime::package::is_hf_package_ref(package_ref)
}

pub(super) fn ensure_split_participant_timeout_has_quorum(
    model_ref: &str,
    best: &[SplitParticipant],
    best_excluded: &[SplitParticipantExclusion],
) -> Result<()> {
    if best.len() >= SPLIT_DEFAULT_MIN_PARTICIPANTS {
        return Ok(());
    }
    anyhow::bail!(
        "split runtime needs at least two participating nodes for {model_ref}; found {} eligible [{}]; excluded [{}]; blockers [{}]; next_step: {}",
        best.len(),
        split_participant_labels(best).join(", "),
        split_participant_exclusion_labels(best_excluded).join(", "),
        split_participant_blocker_labels(best_excluded).join("; "),
        split_participant_next_step(best_excluded)
    )
}

pub(super) fn split_participant_blocker_labels(
    excluded: &[SplitParticipantExclusion],
) -> Vec<String> {
    split_participant_blockers(excluded)
        .into_iter()
        .map(|blocker| {
            format!(
                "{}={} nodes=[{}]",
                blocker.reason,
                blocker.count,
                blocker.short_node_ids.join(", ")
            )
        })
        .collect()
}

pub(super) fn split_participant_next_step(excluded: &[SplitParticipantExclusion]) -> &'static str {
    split_participant_blockers(excluded)
        .first()
        .map(|blocker| blocker.recommendation)
        .unwrap_or("Start at least one more worker/host with the same --model value and --split.")
}

pub(super) fn split_participant_blockers(
    excluded: &[SplitParticipantExclusion],
) -> Vec<SplitParticipantBlockerSummary> {
    let mut blockers = split_participant_exclusion_reason_order()
        .into_iter()
        .filter_map(|reason| split_participant_blocker(excluded, reason))
        .collect::<Vec<_>>();
    blockers.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| blocker_reason_rank(left.reason).cmp(&blocker_reason_rank(right.reason)))
    });
    blockers
}

fn split_participant_blocker(
    excluded: &[SplitParticipantExclusion],
    reason: SplitParticipantExclusionReason,
) -> Option<SplitParticipantBlockerSummary> {
    let matching = excluded
        .iter()
        .filter(|item| item.reason == reason)
        .collect::<Vec<_>>();
    if matching.is_empty() {
        return None;
    }
    Some(SplitParticipantBlockerSummary {
        reason: reason.as_str(),
        count: matching.len(),
        short_node_ids: matching
            .into_iter()
            .map(|item| item.node_id.fmt_short().to_string())
            .collect(),
        recommendation: reason.recommendation(),
    })
}

pub(super) const fn split_participant_exclusion_reason_order()
-> [SplitParticipantExclusionReason; 9] {
    [
        SplitParticipantExclusionReason::StageControlUnreachable,
        SplitParticipantExclusionReason::PackageManifestMismatch,
        SplitParticipantExclusionReason::ArtifactTransferUnavailable,
        SplitParticipantExclusionReason::StageInventoryEmpty,
        SplitParticipantExclusionReason::MissingModelSource,
        SplitParticipantExclusionReason::StageProtocolGeneration,
        SplitParticipantExclusionReason::MissingVram,
        SplitParticipantExclusionReason::MissingModelInterest,
        SplitParticipantExclusionReason::Client,
    ]
}

pub(super) fn blocker_reason_rank(reason: &str) -> usize {
    split_participant_exclusion_reason_order()
        .iter()
        .position(|candidate| candidate.as_str() == reason)
        .unwrap_or(usize::MAX)
}

pub(super) async fn collect_split_participant_membership(
    node: &mesh::Node,
    model_name: &str,
    model_ref: &str,
) -> SplitParticipantSnapshot {
    let mut participants = vec![SplitParticipant::new(
        node.id(),
        node.vram_bytes(),
        Some(node.first_joined_mesh_ts().await.unwrap_or(0)),
    )];
    let mut excluded = Vec::new();
    for peer in node.peers().await {
        if let Some(reason) = split_peer_preflight_exclusion_reason(&peer, model_name, model_ref) {
            excluded.push(SplitParticipantExclusion {
                node_id: peer.id,
                reason,
            });
            continue;
        }
        participants.push(SplitParticipant::new(
            peer.id,
            peer.vram_bytes,
            peer.first_joined_mesh_ts,
        ));
    }
    sort_split_participants(&mut participants);
    excluded.sort_by_key(|exclusion| exclusion.node_id.to_string());
    excluded.dedup_by_key(|exclusion| exclusion.node_id);
    SplitParticipantSnapshot {
        participants,
        excluded,
    }
}

pub(super) async fn collect_split_participants(
    node: &mesh::Node,
    model_name: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    local_vram_override: Option<u64>,
) -> SplitParticipantSnapshot {
    let mut participants = vec![SplitParticipant::local_package(
        node.id(),
        local_vram_override.unwrap_or_else(|| node.vram_bytes()),
        Some(node.first_joined_mesh_ts().await.unwrap_or(0)),
        package,
    )];
    let mut excluded = Vec::new();
    for peer in node.peers().await {
        if let Some(reason) = split_peer_preflight_exclusion_reason(&peer, model_name, model_ref) {
            excluded.push(SplitParticipantExclusion {
                node_id: peer.id,
                reason,
            });
            continue;
        }

        let artifact_transfer_allowed = node.artifact_transfer_allowed_for_peer(&peer).await;
        match split_peer_package_signal(
            node,
            peer.id,
            model_ref,
            package,
            artifact_transfer_allowed,
        )
        .await
        {
            Ok(package_signal) => {
                participants.push(
                    SplitParticipant::new(peer.id, peer.vram_bytes, peer.first_joined_mesh_ts)
                        .with_package_signals(
                            package_signal,
                            peer.rtt_ms,
                            artifact_transfer_allowed,
                        ),
                );
            }
            Err(reason) => {
                excluded.push(SplitParticipantExclusion {
                    node_id: peer.id,
                    reason,
                });
            }
        }
    }
    sort_split_participants(&mut participants);
    excluded.sort_by_key(|exclusion| exclusion.node_id.to_string());
    excluded.dedup_by_key(|exclusion| exclusion.node_id);
    SplitParticipantSnapshot {
        participants,
        excluded,
    }
}

fn sort_split_participants(participants: &mut Vec<SplitParticipant>) {
    participants.sort_by_key(|participant| participant.node_id.to_string());
    participants.dedup_by_key(|participant| participant.node_id);
}

pub(super) fn split_peer_preflight_exclusion_reason(
    peer: &mesh::PeerInfo,
    model_name: &str,
    model_ref: &str,
) -> Option<SplitParticipantExclusionReason> {
    if let Some(reason) = split_peer_stage_host_exclusion_reason(peer) {
        return Some(reason);
    }
    if !split_peer_wants_model(peer, model_name, model_ref) {
        return Some(SplitParticipantExclusionReason::MissingModelInterest);
    }
    if !peer.stage_protocol_generation_supported {
        return Some(SplitParticipantExclusionReason::StageProtocolGeneration);
    }
    None
}

pub(super) fn split_peer_stage_host_exclusion_reason(
    peer: &mesh::PeerInfo,
) -> Option<SplitParticipantExclusionReason> {
    if !split_peer_can_run_stage_runtime(peer) {
        return Some(SplitParticipantExclusionReason::Client);
    }
    if peer.vram_bytes == 0 {
        return Some(SplitParticipantExclusionReason::MissingVram);
    }
    None
}

pub(super) fn split_peer_can_run_stage_runtime(peer: &mesh::PeerInfo) -> bool {
    matches!(peer.role, NodeRole::Worker | NodeRole::Host { .. })
}

pub(super) fn split_peer_wants_model(
    peer: &mesh::PeerInfo,
    model_name: &str,
    model_ref: &str,
) -> bool {
    peer.requested_models
        .iter()
        .any(|model| model == model_name)
        || peer.routes_model(model_ref)
        || peer.serving_models.iter().any(|model| model == model_name)
        || peer
            .available_models
            .iter()
            .any(|model| model == model_name)
        || peer
            .explicit_model_interests
            .iter()
            .any(|model| model == model_ref)
}

pub(super) async fn split_peer_package_signal(
    node: &mesh::Node,
    peer_id: iroh::EndpointId,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    artifact_transfer_supported: bool,
) -> std::result::Result<SplitParticipantPackageSignal, SplitParticipantExclusionReason> {
    let request = skippy::StageInventoryRequest {
        model_id: model_ref.to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
    };
    let result = node
        .send_stage_control(peer_id, skippy::StageControlRequest::Inventory(request))
        .await;
    let Ok(response) = result else {
        return Err(SplitParticipantExclusionReason::StageControlUnreachable);
    };
    let skippy::StageControlResponse::Inventory(inventory) = response else {
        return Err(SplitParticipantExclusionReason::StageControlUnreachable);
    };
    split_inventory_package_signal_result(&inventory, package, artifact_transfer_supported)
}

pub(super) fn split_inventory_package_signal_result(
    inventory: &skippy::StageLayerInventory,
    package: &skippy::SkippyPackageIdentity,
    artifact_transfer_supported: bool,
) -> std::result::Result<SplitParticipantPackageSignal, SplitParticipantExclusionReason> {
    if split_inventory_manifest_mismatch(inventory, package) {
        return Err(SplitParticipantExclusionReason::PackageManifestMismatch);
    }
    if split_inventory_has_no_stage_surface(inventory) {
        return Err(SplitParticipantExclusionReason::StageInventoryEmpty);
    }
    let signal = split_inventory_package_signal(inventory, package);
    if signal.can_stage_with(package, artifact_transfer_supported) {
        return Ok(signal);
    }
    if signal.missing_artifact_bytes > 0 && !artifact_transfer_supported {
        return Err(SplitParticipantExclusionReason::ArtifactTransferUnavailable);
    }
    Err(SplitParticipantExclusionReason::MissingModelSource)
}

pub(super) fn split_inventory_manifest_mismatch(
    inventory: &skippy::StageLayerInventory,
    package: &skippy::SkippyPackageIdentity,
) -> bool {
    inventory.package_ref != package.package_ref
        || inventory.manifest_sha256 != package.manifest_sha256
}

pub(super) fn split_inventory_has_no_stage_surface(
    inventory: &skippy::StageLayerInventory,
) -> bool {
    inventory.layer_count == 0
        && inventory.ready_ranges.is_empty()
        && inventory.available_ranges.is_empty()
        && inventory.missing_ranges.is_empty()
        && inventory.preparing_ranges.is_empty()
        && inventory.source_model_path.is_none()
        && inventory.source_model_bytes.is_none()
        && matches!(
            inventory.source_model_kind,
            skippy::SourceModelKind::Unknown
        )
}

pub(super) fn split_inventory_package_signal(
    inventory: &skippy::StageLayerInventory,
    package: &skippy::SkippyPackageIdentity,
) -> SplitParticipantPackageSignal {
    let cached_slice_bytes = split_inventory_range_bytes(
        inventory
            .available_ranges
            .iter()
            .chain(inventory.ready_ranges.iter()),
        package,
    );
    let explicit_missing_bytes =
        split_inventory_range_bytes(inventory.missing_ranges.iter(), package);
    let missing_artifact_bytes = if explicit_missing_bytes > 0 {
        explicit_missing_bytes
    } else if cached_slice_bytes >= package.source_model_bytes {
        0
    } else if inventory.layer_count == 0 && cached_slice_bytes == 0 {
        package.source_model_bytes
    } else {
        package
            .source_model_bytes
            .saturating_sub(cached_slice_bytes)
    };
    SplitParticipantPackageSignal {
        cached_slice_bytes,
        missing_artifact_bytes,
        availability_score: split_inventory_covered_layers(
            inventory
                .available_ranges
                .iter()
                .chain(inventory.ready_ranges.iter()),
            package.layer_count,
        ),
    }
}

fn split_inventory_range_bytes<'a>(
    ranges: impl Iterator<Item = &'a skippy::LayerRange>,
    package: &skippy::SkippyPackageIdentity,
) -> u64 {
    if package.layer_count == 0 || package.source_model_bytes == 0 {
        return 0;
    }
    let covered_layers = u128::from(split_inventory_covered_layers(ranges, package.layer_count));
    let layer_count = u128::from(package.layer_count);
    let bytes = u128::from(package.source_model_bytes).saturating_mul(covered_layers) / layer_count;
    bytes.min(u128::from(package.source_model_bytes)) as u64
}

fn split_inventory_covered_layers<'a>(
    ranges: impl Iterator<Item = &'a skippy::LayerRange>,
    layer_count: u32,
) -> u32 {
    let mut ranges = ranges
        .filter_map(|range| {
            let start = range.layer_start.min(layer_count);
            let end = range.layer_end.min(layer_count);
            (start < end).then_some((start, end))
        })
        .collect::<Vec<_>>();
    ranges.sort_unstable();
    let mut covered = 0u32;
    let mut current: Option<(u32, u32)> = None;
    for (start, end) in ranges {
        match current {
            Some((current_start, current_end)) if start <= current_end => {
                current = Some((current_start, current_end.max(end)));
            }
            Some((current_start, current_end)) => {
                covered = covered.saturating_add(current_end.saturating_sub(current_start));
                current = Some((start, end));
            }
            None => current = Some((start, end)),
        }
    }
    if let Some((start, end)) = current {
        covered = covered.saturating_add(end.saturating_sub(start));
    }
    covered
}

pub(super) fn split_participant_signature(
    participants: &[SplitParticipant],
) -> SplitParticipantSignature {
    participants
        .iter()
        .map(|participant| {
            (
                participant.node_id.to_string(),
                participant.vram_bytes,
                participant.cached_slice_bytes,
                participant.missing_artifact_bytes,
                participant.rtt_ms,
                participant.artifact_transfer_supported,
                participant.availability_score,
            )
        })
        .collect()
}

pub(super) fn split_participant_set_hash(participants: &[SplitParticipant]) -> String {
    let mut hasher = Sha256::new();
    for participant in split_participant_signature(participants) {
        hasher.update(participant.0.as_bytes());
        hasher.update(participant.1.to_le_bytes());
        hasher.update(participant.2.to_le_bytes());
        hasher.update(participant.3.to_le_bytes());
        hasher.update(participant.4.unwrap_or_default().to_le_bytes());
        hasher.update([u8::from(participant.5)]);
        hasher.update(participant.6.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub(super) fn split_topology_hash(stages: &[RuntimeSliceStagePlan]) -> String {
    let mut hasher = Sha256::new();
    for stage in stages {
        hasher.update(stage.stage_id.as_bytes());
        hasher.update(stage.stage_index.to_le_bytes());
        hasher.update(stage.node_id.to_string().as_bytes());
        hasher.update(stage.layer_start.to_le_bytes());
        hasher.update(stage.layer_end.to_le_bytes());
        hasher.update(stage.parameter_bytes.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub(super) fn split_node_labels(nodes: &[iroh::EndpointId]) -> Vec<String> {
    nodes
        .iter()
        .map(|node| node.fmt_short().to_string())
        .collect()
}

#[cfg(test)]
pub(super) fn plan_runtime_slice_topology(
    topology_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
) -> Result<Vec<RuntimeSliceStagePlan>> {
    plan_runtime_slice_topology_with_exclusions(topology_id, model_ref, package, participants, &[])
}

#[cfg(test)]
pub(super) fn plan_runtime_slice_topology_with_exclusions(
    topology_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    excluded: &[SplitParticipantExclusion],
) -> Result<Vec<RuntimeSliceStagePlan>> {
    tracing::info!(
        topology_id,
        model_ref,
        participants = ?split_participant_labels(participants),
        layer_count = package.layer_count,
        "planning split runtime topology"
    );
    let topology_participants = collect_topology_participants(participants);
    let plan = skippy::plan_package_identity_topology(
        topology_id,
        model_ref,
        package,
        &topology_participants,
    )?;
    log_topology_plan_diagnostics(topology_id, model_ref, &plan.diagnostics);
    let mut stages = plan
        .stages
        .into_iter()
        .map(|stage| RuntimeSliceStagePlan {
            stage_id: stage.stage_id,
            stage_index: stage.stage_index,
            node_id: stage.node_id,
            layer_start: stage.layer_start,
            layer_end: stage.layer_end,
            parameter_bytes: stage.parameter_bytes,
        })
        .collect::<Vec<_>>();
    stages.sort_by_key(|stage| stage.stage_index);
    validate_split_capacity(model_ref, package, participants, &stages, excluded)?;
    tracing::info!(
        topology_id,
        model_ref,
        stages = ?split_stage_plan_labels(&stages),
        "planned split runtime topology"
    );
    Ok(stages)
}

#[cfg(test)]
pub(super) fn collect_topology_participants(
    participants: &[SplitParticipant],
) -> Vec<skippy::StageTopologyParticipant> {
    participants
        .iter()
        .copied()
        .map(SplitParticipant::to_topology_participant)
        .collect()
}

#[cfg(test)]
pub(super) fn log_topology_plan_diagnostics(
    topology_id: &str,
    model_ref: &str,
    diagnostics: &[String],
) {
    if !diagnostics.is_empty() {
        tracing::debug!(
            topology_id,
            model_ref,
            diagnostics = ?diagnostics,
            "package-aware split topology planner emitted diagnostics"
        );
    }
}
