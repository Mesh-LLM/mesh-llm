use super::MeshApi;
use super::status::{ModelTargetCapacityAdvicePayload, ModelTargetCapacityAdviceState};
use crate::mesh::{NodeRole, PeerInfo};
use serde::Serialize;

const MIN_SPLIT_PARTICIPANTS: usize = 2;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SplitReadinessInput {
    pub(crate) model_ref: String,
    pub(crate) local: SplitReadinessNodeInput,
    pub(crate) peers: Vec<SplitReadinessNodeInput>,
    pub(crate) capacity_advice: Option<ModelTargetCapacityAdvicePayload>,
    pub(crate) active_topology_count: usize,
    pub(crate) active_stage_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SplitReadinessNodeInput {
    pub(crate) node_id: String,
    pub(crate) short_node_id: String,
    pub(crate) source: SplitReadinessNodeSource,
    pub(crate) role: SplitReadinessNodeRole,
    pub(crate) vram_bytes: u64,
    pub(crate) requested_models: Vec<String>,
    pub(crate) explicit_model_interests: Vec<String>,
    pub(crate) serving_models: Vec<String>,
    pub(crate) hosted_models: Vec<String>,
    pub(crate) available_models: Vec<String>,
    pub(crate) model_source: Option<String>,
    pub(crate) stage_protocol_generation_supported: bool,
    pub(crate) artifact_transfer_supported: bool,
    pub(crate) rtt_ms: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SplitReadinessNodeSource {
    Local,
    Peer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SplitReadinessNodeRole {
    Worker,
    Host,
    Client,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SplitReadinessVerdict {
    Ready,
    WaitingForPeers,
    InsufficientCapacity,
    UnknownModelSize,
    NoModel,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct SplitReadinessReport {
    pub(crate) model_ref: String,
    pub(crate) verdict: SplitReadinessVerdict,
    pub(crate) participant_count: usize,
    pub(crate) exclusion_count: usize,
    pub(crate) active_topology_count: usize,
    pub(crate) active_stage_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) capacity_advice: Option<ModelTargetCapacityAdvicePayload>,
    pub(crate) participants: Vec<SplitReadinessParticipant>,
    pub(crate) exclusions: Vec<SplitReadinessExclusion>,
    pub(crate) recommendations: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct SplitReadinessParticipant {
    pub(crate) node_id: String,
    pub(crate) short_node_id: String,
    pub(crate) source: SplitReadinessNodeSource,
    pub(crate) role: SplitReadinessNodeRole,
    pub(crate) vram_bytes: u64,
    pub(crate) artifact_transfer_supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) rtt_ms: Option<u32>,
    pub(crate) model_source_state: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct SplitReadinessExclusion {
    pub(crate) node_id: String,
    pub(crate) short_node_id: String,
    pub(crate) source: SplitReadinessNodeSource,
    pub(crate) role: SplitReadinessNodeRole,
    pub(crate) reason: &'static str,
    pub(crate) recommendation: &'static str,
    pub(crate) vram_bytes: u64,
}

impl MeshApi {
    pub(crate) async fn split_readiness_report(&self, model_ref: &str) -> SplitReadinessReport {
        let node = self.inner.lock().await.node.clone();
        let model_target_lookup = self.model_target_lookup().await;
        let capacity_advice = model_target_lookup
            .by_model_ref
            .get(model_ref)
            .or_else(|| model_target_lookup.by_model_name.get(model_ref))
            .map(|target| target.capacity_advice.clone());

        let role = node.role().await;
        let local = SplitReadinessNodeInput {
            node_id: node.id().to_string(),
            short_node_id: node.id().fmt_short().to_string(),
            source: SplitReadinessNodeSource::Local,
            role: split_node_role(&role),
            vram_bytes: node.vram_bytes(),
            requested_models: node.requested_models().await,
            explicit_model_interests: node.explicit_model_interests().await,
            serving_models: node.serving_models().await,
            hosted_models: node.hosted_models().await,
            available_models: node.available_models().await,
            model_source: None,
            stage_protocol_generation_supported: true,
            artifact_transfer_supported: true,
            rtt_ms: None,
        };
        let peers = node
            .peers()
            .await
            .into_iter()
            .map(peer_readiness_input)
            .collect();
        let active_topology_count = node.stage_topologies().await.len();
        let active_stage_count = node
            .stage_runtime_statuses()
            .await
            .into_iter()
            .filter(|status| status.model_id == model_ref)
            .count();

        build_split_readiness_report(SplitReadinessInput {
            model_ref: model_ref.to_string(),
            local,
            peers,
            capacity_advice,
            active_topology_count,
            active_stage_count,
        })
    }
}

pub(crate) fn build_split_readiness_report(input: SplitReadinessInput) -> SplitReadinessReport {
    let mut participants = Vec::new();
    let mut exclusions = Vec::new();
    for node in std::iter::once(input.local).chain(input.peers) {
        match split_node_exclusion_reason(&input.model_ref, &node) {
            Some(reason) => exclusions.push(split_exclusion(node, reason)),
            None => participants.push(split_participant(&input.model_ref, node)),
        }
    }
    let capacity_advice =
        participant_capacity_advice(input.capacity_advice, participants.as_slice());
    let verdict = split_readiness_verdict(
        &input.model_ref,
        participants.len(),
        capacity_advice.as_ref(),
    );
    let recommendations = split_readiness_recommendations(&input.model_ref, verdict, &exclusions);
    SplitReadinessReport {
        model_ref: input.model_ref,
        verdict,
        participant_count: participants.len(),
        exclusion_count: exclusions.len(),
        active_topology_count: input.active_topology_count,
        active_stage_count: input.active_stage_count,
        capacity_advice,
        participants,
        exclusions,
        recommendations,
    }
}

fn peer_readiness_input(peer: PeerInfo) -> SplitReadinessNodeInput {
    let rtt_ms = peer.current_direct_rtt_ms();
    SplitReadinessNodeInput {
        node_id: peer.id.to_string(),
        short_node_id: peer.id.fmt_short().to_string(),
        source: SplitReadinessNodeSource::Peer,
        role: split_node_role(&peer.role),
        vram_bytes: peer.vram_bytes,
        requested_models: peer.requested_models,
        explicit_model_interests: peer.explicit_model_interests,
        serving_models: peer.serving_models,
        hosted_models: peer.hosted_models,
        available_models: peer.available_models,
        model_source: peer.model_source,
        stage_protocol_generation_supported: peer.stage_protocol_generation_supported,
        artifact_transfer_supported: peer.artifact_transfer_supported,
        rtt_ms,
    }
}

fn split_node_role(role: &NodeRole) -> SplitReadinessNodeRole {
    match role {
        NodeRole::Worker => SplitReadinessNodeRole::Worker,
        NodeRole::Host { .. } => SplitReadinessNodeRole::Host,
        NodeRole::Client => SplitReadinessNodeRole::Client,
    }
}

fn split_node_exclusion_reason(
    model_ref: &str,
    node: &SplitReadinessNodeInput,
) -> Option<SplitReadinessExclusionReason> {
    if node.role == SplitReadinessNodeRole::Client {
        return Some(SplitReadinessExclusionReason::Client);
    }
    if node.vram_bytes == 0 {
        return Some(SplitReadinessExclusionReason::MissingVram);
    }
    if !node_wants_model(model_ref, node) {
        return Some(SplitReadinessExclusionReason::MissingModelInterest);
    }
    if !node.stage_protocol_generation_supported {
        return Some(SplitReadinessExclusionReason::StageProtocolGeneration);
    }
    if node.source == SplitReadinessNodeSource::Peer && !node_has_stage_source(model_ref, node) {
        return Some(SplitReadinessExclusionReason::MissingModelSource);
    }
    None
}

fn split_participant(model_ref: &str, node: SplitReadinessNodeInput) -> SplitReadinessParticipant {
    let model_source_state = model_source_state(model_ref, &node);
    SplitReadinessParticipant {
        node_id: node.node_id,
        short_node_id: node.short_node_id,
        source: node.source,
        role: node.role,
        vram_bytes: node.vram_bytes,
        artifact_transfer_supported: node.artifact_transfer_supported,
        rtt_ms: node.rtt_ms,
        model_source_state,
    }
}

fn split_exclusion(
    node: SplitReadinessNodeInput,
    reason: SplitReadinessExclusionReason,
) -> SplitReadinessExclusion {
    SplitReadinessExclusion {
        node_id: node.node_id,
        short_node_id: node.short_node_id,
        source: node.source,
        role: node.role,
        reason: reason.as_str(),
        recommendation: reason.recommendation(),
        vram_bytes: node.vram_bytes,
    }
}

fn split_readiness_verdict(
    model_ref: &str,
    participant_count: usize,
    capacity_advice: Option<&ModelTargetCapacityAdvicePayload>,
) -> SplitReadinessVerdict {
    if model_ref.trim().is_empty() {
        return SplitReadinessVerdict::NoModel;
    }
    if participant_count < MIN_SPLIT_PARTICIPANTS {
        return SplitReadinessVerdict::WaitingForPeers;
    }
    let Some(capacity_advice) = capacity_advice else {
        return SplitReadinessVerdict::UnknownModelSize;
    };
    match capacity_advice.state {
        ModelTargetCapacityAdviceState::InsufficientCapacity
        | ModelTargetCapacityAdviceState::UnknownCapacity
        | ModelTargetCapacityAdviceState::NoEligibleHosts => {
            SplitReadinessVerdict::InsufficientCapacity
        }
        ModelTargetCapacityAdviceState::UnknownModelSize => SplitReadinessVerdict::UnknownModelSize,
        _ => SplitReadinessVerdict::Ready,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SplitReadinessCapacitySummary {
    best_single_node_capacity_bytes: Option<u64>,
    aggregate_capacity_bytes: u64,
    eligible_node_count: usize,
}

fn participant_capacity_advice(
    capacity_advice: Option<ModelTargetCapacityAdvicePayload>,
    participants: &[SplitReadinessParticipant],
) -> Option<ModelTargetCapacityAdvicePayload> {
    let mut advice = capacity_advice?;
    let summary = participant_capacity_summary(participants);
    advice.best_single_node_capacity_bytes = summary.best_single_node_capacity_bytes;
    advice.aggregate_capacity_bytes = summary.aggregate_capacity_bytes;
    advice.eligible_node_count = summary.eligible_node_count;
    advice.excluded_client_node_count = 0;
    advice.missing_capacity_node_count = 0;

    if advice.state == ModelTargetCapacityAdviceState::AlreadyServing {
        return Some(advice);
    }

    let Some(required_bytes) = advice.required_bytes else {
        advice.state = ModelTargetCapacityAdviceState::UnknownModelSize;
        advice.reason = "model_size_unknown";
        advice.shortfall_bytes = None;
        return Some(advice);
    };

    advice.state = participant_capacity_state(required_bytes, advice.split_capable, summary);
    advice.reason = participant_capacity_reason(advice.state);
    advice.shortfall_bytes =
        participant_capacity_shortfall(required_bytes, advice.split_capable, summary);
    Some(advice)
}

fn participant_capacity_summary(
    participants: &[SplitReadinessParticipant],
) -> SplitReadinessCapacitySummary {
    let mut aggregate_capacity_bytes = 0_u64;
    let mut best_single_node_capacity_bytes: Option<u64> = None;
    for participant in participants {
        aggregate_capacity_bytes = aggregate_capacity_bytes.saturating_add(participant.vram_bytes);
        best_single_node_capacity_bytes = Some(
            best_single_node_capacity_bytes
                .unwrap_or_default()
                .max(participant.vram_bytes),
        );
    }
    SplitReadinessCapacitySummary {
        best_single_node_capacity_bytes,
        aggregate_capacity_bytes,
        eligible_node_count: participants.len(),
    }
}

fn participant_capacity_state(
    required_bytes: u64,
    split_capable: bool,
    summary: SplitReadinessCapacitySummary,
) -> ModelTargetCapacityAdviceState {
    if summary.eligible_node_count == 0 {
        return ModelTargetCapacityAdviceState::NoEligibleHosts;
    }
    if summary
        .best_single_node_capacity_bytes
        .is_some_and(|capacity| capacity >= required_bytes)
    {
        return ModelTargetCapacityAdviceState::SingleNodeFit;
    }
    if split_capable
        && summary.eligible_node_count >= MIN_SPLIT_PARTICIPANTS
        && summary.aggregate_capacity_bytes >= required_bytes
    {
        return ModelTargetCapacityAdviceState::SplitCandidate;
    }
    ModelTargetCapacityAdviceState::InsufficientCapacity
}

const fn participant_capacity_reason(state: ModelTargetCapacityAdviceState) -> &'static str {
    match state {
        ModelTargetCapacityAdviceState::AlreadyServing => "already_serving",
        ModelTargetCapacityAdviceState::SingleNodeFit => "single_node_capacity_available",
        ModelTargetCapacityAdviceState::SplitCandidate => "aggregate_split_capacity_available",
        ModelTargetCapacityAdviceState::InsufficientCapacity => {
            "participant_split_capacity_insufficient"
        }
        ModelTargetCapacityAdviceState::UnknownModelSize => "model_size_unknown",
        ModelTargetCapacityAdviceState::UnknownCapacity => "capacity_unknown",
        ModelTargetCapacityAdviceState::NoEligibleHosts => "no_worker_or_host_capacity",
    }
}

fn participant_capacity_shortfall(
    required_bytes: u64,
    split_capable: bool,
    summary: SplitReadinessCapacitySummary,
) -> Option<u64> {
    let comparable_capacity = if split_capable && summary.eligible_node_count >= 2 {
        summary.aggregate_capacity_bytes
    } else {
        summary.best_single_node_capacity_bytes.unwrap_or_default()
    };
    let shortfall = required_bytes.saturating_sub(comparable_capacity);
    (shortfall > 0).then_some(shortfall)
}

fn split_readiness_recommendations(
    model_ref: &str,
    verdict: SplitReadinessVerdict,
    exclusions: &[SplitReadinessExclusion],
) -> Vec<String> {
    let mut recommendations = Vec::new();
    if verdict == SplitReadinessVerdict::WaitingForPeers {
        recommendations.push(format!(
            "Start at least one more worker/host with --model {model_ref} --split and join it to this mesh."
        ));
        recommendations.push(
            "When testing multiple nodes on one machine, use distinct --port, --console, and --bind-port values for every process."
                .to_string(),
        );
    }
    if exclusions
        .iter()
        .any(|item| item.reason == SplitReadinessExclusionReason::StageProtocolGeneration.as_str())
    {
        recommendations.push(
            "Upgrade excluded peers so they advertise current stage protocol support.".to_string(),
        );
    }
    if exclusions
        .iter()
        .any(|item| item.reason == SplitReadinessExclusionReason::MissingVram.as_str())
    {
        recommendations.push(
            "Run mesh-llm gpus on excluded peers and check backend/device visibility before attempting split serving."
                .to_string(),
        );
    }
    if exclusions
        .iter()
        .any(|item| item.reason == SplitReadinessExclusionReason::MissingModelSource.as_str())
    {
        recommendations.push(
            "Start excluded peers with a resolvable package source or wait until stage inventory can prove the package is available before retrying split serving."
                .to_string(),
        );
    }
    recommendations
}

fn node_wants_model(model_ref: &str, node: &SplitReadinessNodeInput) -> bool {
    [
        node.requested_models.as_slice(),
        node.explicit_model_interests.as_slice(),
        node.serving_models.as_slice(),
        node.hosted_models.as_slice(),
        node.available_models.as_slice(),
    ]
    .into_iter()
    .flatten()
    .any(|candidate| model_matches(candidate, model_ref))
        || node
            .model_source
            .as_deref()
            .is_some_and(|candidate| model_matches(candidate, model_ref))
}

fn model_source_state(model_ref: &str, node: &SplitReadinessNodeInput) -> &'static str {
    if node
        .model_source
        .as_deref()
        .is_some_and(|source| !source.trim().is_empty())
    {
        return "declared";
    }
    if node
        .serving_models
        .iter()
        .chain(node.hosted_models.iter())
        .any(|candidate| model_matches(candidate, model_ref))
    {
        return "serving";
    }
    if node
        .available_models
        .iter()
        .any(|candidate| model_matches(candidate, model_ref))
    {
        return "available";
    }
    if node.artifact_transfer_supported {
        return "transfer_supported";
    }
    "unknown"
}

fn node_has_stage_source(model_ref: &str, node: &SplitReadinessNodeInput) -> bool {
    matches!(
        model_source_state(model_ref, node),
        "declared" | "serving" | "available"
    )
}

fn model_matches(candidate: &str, model_ref: &str) -> bool {
    let candidate = candidate.trim();
    let model_ref = model_ref.trim();
    if candidate.eq_ignore_ascii_case(model_ref) {
        return true;
    }
    let candidate_base = candidate.rsplit('/').next().unwrap_or(candidate);
    let model_base = model_ref.rsplit('/').next().unwrap_or(model_ref);
    candidate_base.eq_ignore_ascii_case(model_base)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SplitReadinessExclusionReason {
    Client,
    MissingVram,
    MissingModelInterest,
    StageProtocolGeneration,
    MissingModelSource,
}

impl SplitReadinessExclusionReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Client => "client",
            Self::MissingVram => "missing_vram",
            Self::MissingModelInterest => "missing_model_interest",
            Self::StageProtocolGeneration => "stage_protocol_generation",
            Self::MissingModelSource => "missing_model_source",
        }
    }

    const fn recommendation(self) -> &'static str {
        match self {
            Self::Client => "Run this peer in serve mode if it should contribute compute.",
            Self::MissingVram => {
                "Check GPU visibility or pass a lower --max-vram only after confirming the backend is detected."
            }
            Self::MissingModelInterest => {
                "Start the peer with the same --model value or add explicit model interest."
            }
            Self::StageProtocolGeneration => {
                "Upgrade this peer; its stage protocol generation is too old for split serving."
            }
            Self::MissingModelSource => {
                "Ensure this peer can resolve or inventory the layer package before split serving."
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::status::{ModelTargetCapacityAdvicePayload, ModelTargetCapacityAdviceState};

    fn advice(state: ModelTargetCapacityAdviceState) -> ModelTargetCapacityAdvicePayload {
        ModelTargetCapacityAdvicePayload {
            state,
            reason: "test",
            required_bytes: Some(10_000_000_000),
            best_single_node_capacity_bytes: Some(6_000_000_000),
            aggregate_capacity_bytes: 12_000_000_000,
            shortfall_bytes: None,
            eligible_node_count: 2,
            missing_capacity_node_count: 0,
            excluded_client_node_count: 0,
            split_capable: true,
        }
    }

    fn node(
        id: &str,
        role: SplitReadinessNodeRole,
        requested_models: &[&str],
    ) -> SplitReadinessNodeInput {
        SplitReadinessNodeInput {
            node_id: id.to_string(),
            short_node_id: id.chars().take(8).collect(),
            source: SplitReadinessNodeSource::Peer,
            role,
            vram_bytes: 8_000_000_000,
            requested_models: requested_models
                .iter()
                .map(|value| value.to_string())
                .collect(),
            explicit_model_interests: Vec::new(),
            serving_models: Vec::new(),
            hosted_models: Vec::new(),
            available_models: Vec::new(),
            model_source: None,
            stage_protocol_generation_supported: true,
            artifact_transfer_supported: true,
            rtt_ms: Some(4),
        }
    }

    fn local_node(requested_models: &[&str]) -> SplitReadinessNodeInput {
        let mut local = node(
            "local00000000000000000000000000000000",
            SplitReadinessNodeRole::Host,
            requested_models,
        );
        local.source = SplitReadinessNodeSource::Local;
        local.rtt_ms = None;
        local
    }

    #[test]
    fn split_readiness_waits_when_only_local_node_wants_model() {
        let report = build_split_readiness_report(SplitReadinessInput {
            model_ref: "meshllm/Qwen3-8B-Q4_K_M-layers".to_string(),
            local: local_node(&["meshllm/Qwen3-8B-Q4_K_M-layers"]),
            peers: vec![node(
                "peer000000000000000000000000000000000",
                SplitReadinessNodeRole::Worker,
                &[],
            )],
            capacity_advice: Some(advice(ModelTargetCapacityAdviceState::SplitCandidate)),
            active_topology_count: 0,
            active_stage_count: 0,
        });

        assert_eq!(report.verdict, SplitReadinessVerdict::WaitingForPeers);
        assert_eq!(report.participants.len(), 1);
        assert_eq!(report.exclusions[0].reason, "missing_model_interest");
        assert!(
            report
                .recommendations
                .iter()
                .any(|item| item.contains("--model meshllm/Qwen3-8B-Q4_K_M-layers"))
        );
    }

    #[test]
    fn split_readiness_is_ready_with_two_interested_stage_hosts() {
        let mut peer = node(
            "peer000000000000000000000000000000000",
            SplitReadinessNodeRole::Worker,
            &["meshllm/Qwen3-8B-Q4_K_M-layers"],
        );
        peer.available_models = vec!["meshllm/Qwen3-8B-Q4_K_M-layers".to_string()];
        let report = build_split_readiness_report(SplitReadinessInput {
            model_ref: "meshllm/Qwen3-8B-Q4_K_M-layers".to_string(),
            local: local_node(&["meshllm/Qwen3-8B-Q4_K_M-layers"]),
            peers: vec![peer],
            capacity_advice: Some(advice(ModelTargetCapacityAdviceState::SplitCandidate)),
            active_topology_count: 0,
            active_stage_count: 0,
        });

        assert_eq!(report.verdict, SplitReadinessVerdict::Ready);
        assert_eq!(report.participants.len(), 2);
        assert!(report.exclusions.is_empty());
    }

    #[test]
    fn split_readiness_does_not_borrow_capacity_from_excluded_peer() {
        let mut local = local_node(&["meshllm/Qwen3-8B-Q4_K_M-layers"]);
        local.vram_bytes = 4_000_000_000;
        let mut peer = node(
            "peer000000000000000000000000000000000",
            SplitReadinessNodeRole::Worker,
            &["meshllm/Qwen3-8B-Q4_K_M-layers"],
        );
        peer.available_models = vec!["meshllm/Qwen3-8B-Q4_K_M-layers".to_string()];
        peer.vram_bytes = 4_000_000_000;
        let mut excluded = node(
            "excluded000000000000000000000000000",
            SplitReadinessNodeRole::Worker,
            &[],
        );
        excluded.vram_bytes = 40_000_000_000;
        let mut stale_advice = advice(ModelTargetCapacityAdviceState::SplitCandidate);
        stale_advice.aggregate_capacity_bytes = 48_000_000_000;
        stale_advice.best_single_node_capacity_bytes = Some(40_000_000_000);

        let report = build_split_readiness_report(SplitReadinessInput {
            model_ref: "meshllm/Qwen3-8B-Q4_K_M-layers".to_string(),
            local,
            peers: vec![peer, excluded],
            capacity_advice: Some(stale_advice),
            active_topology_count: 0,
            active_stage_count: 0,
        });

        let capacity = report.capacity_advice.as_ref().expect("capacity advice");
        assert_eq!(report.verdict, SplitReadinessVerdict::InsufficientCapacity);
        assert_eq!(report.participant_count, 2);
        assert_eq!(report.exclusions[0].reason, "missing_model_interest");
        assert_eq!(capacity.aggregate_capacity_bytes, 8_000_000_000);
        assert_eq!(
            capacity.state,
            ModelTargetCapacityAdviceState::InsufficientCapacity
        );
        assert_eq!(capacity.shortfall_bytes, Some(2_000_000_000));
    }

    #[test]
    fn split_readiness_counts_peer_with_available_model_as_participant() {
        let mut peer = node(
            "peer000000000000000000000000000000000",
            SplitReadinessNodeRole::Worker,
            &[],
        );
        peer.available_models = vec!["Qwen3-8B-Q4_K_M-layers".to_string()];

        let report = build_split_readiness_report(SplitReadinessInput {
            model_ref: "meshllm/Qwen3-8B-Q4_K_M-layers".to_string(),
            local: local_node(&["meshllm/Qwen3-8B-Q4_K_M-layers"]),
            peers: vec![peer],
            capacity_advice: Some(advice(ModelTargetCapacityAdviceState::SplitCandidate)),
            active_topology_count: 0,
            active_stage_count: 0,
        });

        assert_eq!(report.verdict, SplitReadinessVerdict::Ready);
        assert_eq!(report.participants.len(), 2);
        assert!(report.exclusions.is_empty());
    }

    #[test]
    fn split_readiness_excludes_interested_peer_with_unknown_model_source() {
        let mut peer = node(
            "peer000000000000000000000000000000000",
            SplitReadinessNodeRole::Worker,
            &["meshllm/Qwen3-8B-Q4_K_M-layers"],
        );
        peer.artifact_transfer_supported = false;

        let report = build_split_readiness_report(SplitReadinessInput {
            model_ref: "meshllm/Qwen3-8B-Q4_K_M-layers".to_string(),
            local: local_node(&["meshllm/Qwen3-8B-Q4_K_M-layers"]),
            peers: vec![peer],
            capacity_advice: Some(advice(ModelTargetCapacityAdviceState::SplitCandidate)),
            active_topology_count: 0,
            active_stage_count: 0,
        });

        assert_eq!(report.verdict, SplitReadinessVerdict::WaitingForPeers);
        assert_eq!(report.participant_count, 1);
        assert_eq!(report.exclusions[0].reason, "missing_model_source");
        assert!(
            report
                .recommendations
                .iter()
                .any(|item| item.contains("resolvable package source"))
        );
    }
}
