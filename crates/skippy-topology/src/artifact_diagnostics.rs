use crate::{DiagnosticSeverity, NodePlacementSignal, PlanDiagnostic, PlanReasonCode, StagePlan};

pub(crate) fn append_artifact_diagnostics(
    diagnostics: &mut Vec<PlanDiagnostic>,
    stages: &[StagePlan],
    placement_signals: &[NodePlacementSignal],
) {
    let cached_bytes = stages
        .iter()
        .map(cached_artifact_bytes_for_stage)
        .sum::<u64>();
    let missing_bytes = stages
        .iter()
        .map(missing_artifact_bytes_for_stage)
        .sum::<u64>();

    if cached_bytes == 0 && missing_bytes == 0 {
        return;
    }

    let peer_transfer_bytes = missing_bytes_by_transfer_support(stages, placement_signals, true);
    let remote_download_bytes = missing_bytes.saturating_sub(peer_transfer_bytes);
    let code = if missing_bytes > 0 {
        PlanReasonCode::ArtifactTransferPenalty
    } else {
        PlanReasonCode::CacheLocalityPreferred
    };

    diagnostics.push(PlanDiagnostic {
        severity: DiagnosticSeverity::Info,
        code,
        message: format!(
            "artifact cold-start plan: cached={} bytes, missing={} bytes, peer-transfer-eligible={} bytes, remote-download-fallback={} bytes",
            cached_bytes, missing_bytes, peer_transfer_bytes, remote_download_bytes
        ),
    });
}

fn missing_bytes_by_transfer_support(
    stages: &[StagePlan],
    placement_signals: &[NodePlacementSignal],
    transfer_supported: bool,
) -> u64 {
    stages
        .iter()
        .filter(|stage| {
            placement_signals
                .iter()
                .find(|signal| signal.node_id == stage.node_id)
                .is_some_and(|signal| signal.artifact_transfer_supported == transfer_supported)
        })
        .map(missing_artifact_bytes_for_stage)
        .sum()
}

fn cached_artifact_bytes_for_stage(stage: &StagePlan) -> u64 {
    stage.cached_slice_bytes.min(stage.parameter_bytes)
}

fn missing_artifact_bytes_for_stage(stage: &StagePlan) -> u64 {
    stage.missing_artifact_bytes.min(stage.parameter_bytes)
}
