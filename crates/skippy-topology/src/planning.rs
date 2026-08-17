use crate::edge_order::StageEdgeSignal;
use crate::validation::{PlanError, validate_request};
use crate::{
    BoundaryDecision, BoundaryPlan, DiagnosticSeverity, ExactStateMobility, FamilyCapabilityRecord,
    LayerSpec, MigrationPolicy, NodePlacementSignal, NodeSpec, PlanDiagnostic, PlanReasonCode,
    PlannerPolicy, SidebandKind, SplitConstraintKind, StagePlan, StageRole, StateAffinity,
    TopologyPlan, TopologyPlanRequest, WireDType, WireValidation, artifact_diagnostics, edge_order,
};

pub fn plan_even_contiguous(request: &TopologyPlanRequest) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;

    let stage_count = request.nodes.len().min(request.layers.len());
    let base = request.layers.len() / stage_count;
    let remainder = request.layers.len() % stage_count;
    let mut next_layer = 0usize;
    let mut ranges = Vec::with_capacity(stage_count);

    for stage_index in 0..stage_count {
        let layer_count = base + usize::from(stage_index < remainder);
        ranges.push((next_layer, next_layer + layer_count));
        next_layer += layer_count;
    }

    plan_ranges(request, &ranges)
}

pub fn plan_weighted_contiguous(request: &TopologyPlanRequest) -> Result<TopologyPlan, PlanError> {
    plan_weighted_contiguous_with_signals(request, &[])
}

fn plan_weighted_contiguous_with_signals(
    request: &TopologyPlanRequest,
    placement_signals: &[NodePlacementSignal],
) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;

    let stage_count = request.nodes.len().min(request.layers.len());
    let nodes = &request.nodes[..stage_count];
    let total_weight: u64 = nodes.iter().map(|node| node.vram_bytes).sum();
    if total_weight == 0 {
        let base = request.layers.len() / stage_count;
        let remainder = request.layers.len() % stage_count;
        let mut next_layer = 0usize;
        let mut ranges = Vec::with_capacity(stage_count);

        for stage_index in 0..stage_count {
            let layer_count = base + usize::from(stage_index < remainder);
            ranges.push((next_layer, next_layer + layer_count));
            next_layer += layer_count;
        }

        return plan_ranges_with_signals(request, &ranges, placement_signals);
    }

    let mut ranges = Vec::with_capacity(stage_count);
    let mut layer_start = 0usize;
    for (stage_index, node) in nodes.iter().enumerate() {
        let remaining_stages = stage_count - stage_index;
        let remaining_layers = request.layers.len() - layer_start;
        let mut span = if remaining_stages == 1 {
            remaining_layers
        } else {
            (((request.layers.len() as u128) * (node.vram_bytes as u128)) / (total_weight as u128))
                .try_into()
                .unwrap_or(usize::MAX)
        };
        span = span.max(1).min(remaining_layers - (remaining_stages - 1));
        let layer_end = layer_start + span;
        ranges.push((layer_start, layer_end));
        layer_start = layer_end;
    }

    plan_ranges_with_signals(request, &ranges, placement_signals)
}

pub fn plan_package_aware_contiguous(
    request: &TopologyPlanRequest,
) -> Result<TopologyPlan, PlanError> {
    plan_package_aware_contiguous_with_signals(request, &[])
}

pub fn plan_package_aware_contiguous_with_signals(
    request: &TopologyPlanRequest,
    placement_signals: &[NodePlacementSignal],
) -> Result<TopologyPlan, PlanError> {
    plan_package_aware_contiguous_with_transport(request, placement_signals, &[])
}

pub fn plan_package_aware_contiguous_with_transport(
    request: &TopologyPlanRequest,
    placement_signals: &[NodePlacementSignal],
    edge_signals: &[StageEdgeSignal],
) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;

    if !request.nodes.iter().any(|node| node.cached_slice_bytes > 0)
        && !placement_signals.iter().any(has_package_aware_signal)
        && edge_signals.is_empty()
    {
        return plan_weighted_contiguous(request);
    }

    let mut nodes = request
        .nodes
        .iter()
        .cloned()
        .enumerate()
        .collect::<Vec<_>>();
    nodes.sort_by(|(left_index, left), (right_index, right)| {
        let left_signal = placement_signal_for(placement_signals, &left.node_id);
        let right_signal = placement_signal_for(placement_signals, &right.node_id);
        node_package_score(right, right_signal)
            .cmp(&node_package_score(left, left_signal))
            .then_with(|| left_index.cmp(right_index))
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    let nodes = nodes
        .into_iter()
        .map(|(_, node)| node)
        .enumerate()
        .collect::<Vec<_>>();
    let nodes = edge_order::order_pipeline_nodes(nodes, placement_signals, edge_signals);

    let sorted_request = TopologyPlanRequest {
        topology_id: request.topology_id.clone(),
        model_id: request.model_id.clone(),
        layers: request.layers.clone(),
        nodes: nodes.into_iter().map(|(_, node)| node).collect(),
        family: request.family.clone(),
        policy: request.policy,
    };
    let mut plan = plan_weighted_contiguous_with_signals(&sorted_request, placement_signals)?;
    edge_order::append_edge_diagnostics(&mut plan, edge_signals);
    Ok(plan)
}

pub fn plan_contiguous_with_splits(
    request: &TopologyPlanRequest,
    splits: &[u32],
) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;

    let layer_start = request
        .layers
        .first()
        .expect("validated non-empty layers")
        .index;
    let layer_end = request
        .layers
        .last()
        .expect("validated non-empty layers")
        .index
        + 1;
    let mut previous = layer_start;
    let mut boundaries = Vec::with_capacity(splits.len() + 2);
    boundaries.push(layer_start);
    for &boundary in splits {
        if boundary <= layer_start || boundary >= layer_end {
            return Err(PlanError::InvalidSplitBoundary {
                boundary,
                layer_start,
                layer_end,
            });
        }
        if boundary <= previous {
            return Err(PlanError::NonAscendingSplitBoundary { previous, boundary });
        }
        boundaries.push(boundary);
        previous = boundary;
    }
    boundaries.push(layer_end);

    let stage_count = boundaries.len() - 1;
    if request.nodes.len() < stage_count {
        return Err(PlanError::NotEnoughNodesForSplits {
            stages: stage_count,
            nodes: request.nodes.len(),
        });
    }

    let first_layer = request.layers[0].index;
    let ranges = boundaries
        .windows(2)
        .map(|window| {
            (
                (window[0] - first_layer) as usize,
                (window[1] - first_layer) as usize,
            )
        })
        .collect::<Vec<_>>();

    plan_ranges(request, &ranges)
}

fn plan_ranges(
    request: &TopologyPlanRequest,
    ranges: &[(usize, usize)],
) -> Result<TopologyPlan, PlanError> {
    plan_ranges_with_signals(request, ranges, &[])
}

fn plan_ranges_with_signals(
    request: &TopologyPlanRequest,
    ranges: &[(usize, usize)],
    placement_signals: &[NodePlacementSignal],
) -> Result<TopologyPlan, PlanError> {
    let mut stages = Vec::with_capacity(ranges.len());

    for (stage_index, &(start, end)) in ranges.iter().enumerate() {
        let layers = &request.layers[start..end];
        let layer_start = layers.first().expect("validated non-empty range").index;
        let layer_end = layers.last().expect("validated non-empty range").index + 1;
        let state_affinity = classify_layers_with_family(layers, request.family.as_ref());
        let migration_policy = migration_policy(state_affinity, request.policy);
        let parameter_bytes = layers.iter().map(|layer| layer.parameter_bytes).sum();
        let node = &request.nodes[stage_index];
        let node_id = node.node_id.clone();
        let placement_signal = placement_signal_for(placement_signals, &node_id);
        let mut reason_codes =
            stage_reason_codes(state_affinity, migration_policy, request.family.as_ref());
        reason_codes.extend(node_reason_codes(node, placement_signal));

        stages.push(StagePlan {
            stage_id: format!("stage-{stage_index}"),
            stage_index: stage_index as u32,
            node_id,
            roles: stage_roles(stage_index, ranges.len()),
            layer_start,
            layer_end,
            layer_count: (end - start) as u32,
            parameter_bytes,
            state_affinity,
            migration_policy,
            reason_codes,
            cached_slice_bytes: placement_signal
                .map(|signal| signal.cached_slice_bytes.max(node.cached_slice_bytes))
                .unwrap_or(node.cached_slice_bytes),
            missing_artifact_bytes: placement_signal
                .map(|signal| signal.missing_artifact_bytes)
                .unwrap_or_default(),
            rtt_ms: placement_signal.and_then(|signal| signal.rtt_ms),
        });
    }

    let boundaries = boundaries_for(&stages, request.family.as_ref());
    let diagnostics = diagnostics_for(
        &stages,
        &boundaries,
        placement_signals,
        request.family.as_ref(),
        request.policy,
    );

    Ok(TopologyPlan {
        topology_id: request.topology_id.clone(),
        model_id: request.model_id.clone(),
        family_id: request
            .family
            .as_ref()
            .map(|family| family.family_id.clone()),
        stages,
        boundaries,
        diagnostics,
    })
}

fn placement_signal_for<'a>(
    placement_signals: &'a [NodePlacementSignal],
    node_id: &str,
) -> Option<&'a NodePlacementSignal> {
    placement_signals
        .iter()
        .find(|signal| signal.node_id == node_id)
}

fn has_package_aware_signal(signal: &NodePlacementSignal) -> bool {
    signal.cached_slice_bytes > 0
        || signal.missing_artifact_bytes > 0
        || signal.rtt_ms.is_some()
        || signal.artifact_transfer_supported
        || signal.availability_score > 0
}

fn node_package_score(node: &NodeSpec, signal: Option<&NodePlacementSignal>) -> i128 {
    let mut score = i128::from(node.vram_bytes);
    let cached_slice_bytes = signal
        .map(|signal| signal.cached_slice_bytes.max(node.cached_slice_bytes))
        .unwrap_or(node.cached_slice_bytes);
    score += i128::from(cached_slice_bytes).saturating_mul(2);
    if let Some(signal) = signal {
        score -= i128::from(signal.missing_artifact_bytes).saturating_mul(4);
        if signal.missing_artifact_bytes > 0 && !signal.artifact_transfer_supported {
            score -= i128::from(signal.missing_artifact_bytes).saturating_mul(4);
        }
        if let Some(rtt_ms) = signal.rtt_ms {
            score -= i128::from(rtt_ms).saturating_mul(16 * 1024 * 1024);
        }
        score += i128::from(signal.availability_score).saturating_mul(1024 * 1024);
    }
    score
}

fn node_reason_codes(node: &NodeSpec, signal: Option<&NodePlacementSignal>) -> Vec<PlanReasonCode> {
    let mut codes = Vec::new();
    let cached_slice_bytes = signal
        .map(|signal| signal.cached_slice_bytes.max(node.cached_slice_bytes))
        .unwrap_or(node.cached_slice_bytes);
    if cached_slice_bytes > 0 {
        codes.push(PlanReasonCode::CacheLocalityPreferred);
    }
    if let Some(signal) = signal {
        if signal.missing_artifact_bytes > 0 {
            codes.push(PlanReasonCode::ArtifactTransferPenalty);
        }
        if signal.rtt_ms.is_some_and(|rtt| rtt > 0) {
            codes.push(PlanReasonCode::NetworkPipelineCost);
        }
        if signal.availability_score > 0 {
            codes.push(PlanReasonCode::PeerAvailabilityPreferred);
        }
    }
    codes
}

pub fn classify_layers(layers: &[LayerSpec]) -> StateAffinity {
    classify_layers_with_family(layers, None)
}

fn classify_layers_with_family(
    layers: &[LayerSpec],
    family: Option<&FamilyCapabilityRecord>,
) -> StateAffinity {
    let has_attention = layers.iter().any(|layer| layer.attention);
    let has_recurrent = layers.iter().any(|layer| {
        layer.recurrent
            || family.is_some_and(|family| {
                family
                    .recurrent_ranges
                    .iter()
                    .any(|range| range.contains_layer(layer.index))
            })
    });

    match (has_attention, has_recurrent) {
        (false, false) => StateAffinity::Stateless,
        (true, false) => StateAffinity::AttentionKv,
        (false, true) => StateAffinity::Recurrent,
        (true, true) => StateAffinity::Mixed,
    }
}

fn migration_policy(affinity: StateAffinity, policy: PlannerPolicy) -> MigrationPolicy {
    match affinity {
        StateAffinity::Stateless => MigrationPolicy::FreelyMovable,
        StateAffinity::AttentionKv => MigrationPolicy::CostedKv,
        StateAffinity::Recurrent | StateAffinity::Mixed => {
            if policy.allow_recurrent_state_transfer {
                MigrationPolicy::RecurrentStateTransferAllowed
            } else {
                MigrationPolicy::StickyRecurrentOwner
            }
        }
    }
}

fn stage_reason_codes(
    affinity: StateAffinity,
    migration_policy: MigrationPolicy,
    family: Option<&FamilyCapabilityRecord>,
) -> Vec<PlanReasonCode> {
    let mut codes = Vec::new();
    match migration_policy {
        MigrationPolicy::FreelyMovable => {}
        MigrationPolicy::CostedKv => codes.push(PlanReasonCode::AttentionKvCosted),
        MigrationPolicy::StickyRecurrentOwner => codes.push(PlanReasonCode::RecurrentOwnerSticky),
        MigrationPolicy::RecurrentStateTransferAllowed => {
            codes.push(PlanReasonCode::RecurrentStateTransferAllowed)
        }
    }
    if matches!(affinity, StateAffinity::Recurrent | StateAffinity::Mixed)
        && !codes.contains(&PlanReasonCode::RecurrentOwnerSticky)
        && !codes.contains(&PlanReasonCode::RecurrentStateTransferAllowed)
    {
        codes.push(PlanReasonCode::RecurrentOwnerSticky);
    }
    if let Some(family) = family {
        match family.exact_state_mobility {
            ExactStateMobility::Accepted => codes.push(PlanReasonCode::ExactStateMobilityAccepted),
            ExactStateMobility::RejectedTooLarge => {
                codes.push(PlanReasonCode::ExactStateMobilityRejected)
            }
            ExactStateMobility::Untested => {}
        }
    }
    codes
}

fn stage_roles(stage_index: usize, stage_count: usize) -> Vec<StageRole> {
    let mut roles = Vec::new();
    if stage_index == 0 {
        roles.push(StageRole::Driver);
        roles.push(StageRole::Embedding);
    }
    if stage_index + 1 == stage_count {
        roles.push(StageRole::Readout);
    } else if stage_index > 0 {
        roles.push(StageRole::Intermediate);
    }
    roles
}

fn boundaries_for(
    stages: &[StagePlan],
    family: Option<&FamilyCapabilityRecord>,
) -> Vec<BoundaryPlan> {
    stages
        .windows(2)
        .map(|window| {
            let producer = &window[0];
            let consumer = &window[1];
            let layer_boundary = producer.layer_end;
            let mut decision = BoundaryDecision::Accepted;
            let mut reason_codes = vec![PlanReasonCode::ActivationOnlyBoundary];
            let mut messages = vec![format!(
                "activation boundary after layer {}; send activation frame from {} to {}",
                layer_boundary, producer.stage_id, consumer.stage_id
            )];

            if matches!(
                producer.migration_policy,
                MigrationPolicy::StickyRecurrentOwner
            ) || matches!(
                consumer.migration_policy,
                MigrationPolicy::StickyRecurrentOwner
            ) {
                reason_codes.push(PlanReasonCode::RecurrentOwnerSticky);
                messages.push(
                    "recurrent state remains with the owning stage; only activation crosses this boundary"
                        .to_string(),
                );
            }

            let (wire_dtype, raw_activation_bytes_per_token, wire_payload_bytes_per_token) =
                if let Some(family) = family {
                    apply_family_boundary_rules(
                        family,
                        layer_boundary,
                        &mut decision,
                        &mut reason_codes,
                        &mut messages,
                    );
                    let payload_multiplier =
                        activation_payload_multiplier_for_boundary(family, layer_boundary);
                    let raw = u64::from(family.activation_width) * 4 * payload_multiplier;
                    let wire = wire_payload_bytes_per_token(
                        family.activation_width,
                        family.default_wire_dtype,
                    ) * payload_multiplier;
                    (family.default_wire_dtype, raw, wire)
                } else {
                    (WireDType::F16, 0, 0)
                };

            BoundaryPlan {
                producer_stage_index: producer.stage_index,
                consumer_stage_index: consumer.stage_index,
                layer_boundary,
                decision,
                wire_dtype,
                raw_activation_bytes_per_token,
                wire_payload_bytes_per_token,
                reason_codes,
                messages,
            }
        })
        .collect()
}

fn apply_family_boundary_rules(
    family: &FamilyCapabilityRecord,
    layer_boundary: u32,
    decision: &mut BoundaryDecision,
    reason_codes: &mut Vec<PlanReasonCode>,
    messages: &mut Vec<String>,
) {
    if family.default_wire_dtype == WireDType::F16 {
        reason_codes.push(PlanReasonCode::DefaultWireDtypeF16);
    }

    match family.q8_wire_validation {
        WireValidation::Validated => reason_codes.push(PlanReasonCode::Q8WireValidated),
        WireValidation::Rejected => reason_codes.push(PlanReasonCode::Q8WireRejected),
        WireValidation::Untested => {}
    }

    for constraint in &family.split_constraints {
        if constraint.forbidden_boundaries.contains(&layer_boundary)
            || (constraint.reject_boundary_inside
                && constraint.range.contains_boundary(layer_boundary))
        {
            *decision = BoundaryDecision::Rejected;
            reason_codes.push(match constraint.kind {
                SplitConstraintKind::SharedKvProducerConsumer => PlanReasonCode::SharedKvRegionCut,
            });
            messages.push(constraint.reason.clone());
        }
    }

    for sideband in &family.sidebands {
        if layer_boundary <= sideband.first_required_layer {
            reason_codes.push(match sideband.kind {
                SidebandKind::TokenIds => PlanReasonCode::TokenSidebandRequired,
                SidebandKind::Rwkv7VFirst | SidebandKind::Gemma3nAltup => {
                    PlanReasonCode::ActivationSidebandRequired
                }
            });
            messages.push(sideband.reason.clone());
        }
    }
}

fn activation_payload_multiplier_for_boundary(
    family: &FamilyCapabilityRecord,
    layer_boundary: u32,
) -> u64 {
    let has_gemma3n_altup_sideband = family.sidebands.iter().any(|sideband| {
        sideband.kind == SidebandKind::Gemma3nAltup
            && layer_boundary <= sideband.first_required_layer
    });
    if has_gemma3n_altup_sideband {
        return 4;
    }

    let has_rwkv7_v_first_sideband = family.sidebands.iter().any(|sideband| {
        sideband.kind == SidebandKind::Rwkv7VFirst
            && layer_boundary <= sideband.first_required_layer
    });
    if has_rwkv7_v_first_sideband { 2 } else { 1 }
}

pub fn wire_payload_bytes_per_token(activation_width: u32, dtype: WireDType) -> u64 {
    match dtype {
        WireDType::F32 => u64::from(activation_width) * 4,
        WireDType::F16 => u64::from(activation_width) * 2,
        WireDType::Q8 => u64::from(activation_width) + 4,
    }
}

fn diagnostics_for(
    stages: &[StagePlan],
    boundaries: &[BoundaryPlan],
    placement_signals: &[NodePlacementSignal],
    family: Option<&FamilyCapabilityRecord>,
    policy: PlannerPolicy,
) -> Vec<PlanDiagnostic> {
    let mut diagnostics = Vec::new();
    artifact_diagnostics::append_artifact_diagnostics(&mut diagnostics, stages, placement_signals);
    for stage in stages {
        if matches!(
            stage.migration_policy,
            MigrationPolicy::StickyRecurrentOwner
        ) {
            diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Info,
                code: PlanReasonCode::RecurrentOwnerSticky,
                message: format!(
                    "{} owns recurrent state for layers {}..{}; route future tokens back to {} and only transfer activations across stage boundaries",
                    stage.stage_id, stage.layer_start, stage.layer_end, stage.node_id
                ),
            });
        }
    }

    if policy.allow_recurrent_state_transfer {
        diagnostics.push(PlanDiagnostic {
            severity: DiagnosticSeverity::Warning,
            code: PlanReasonCode::RecurrentStateTransferAllowed,
            message: "recurrent state transfer is enabled; this should be reserved for explicit recompute-or-transfer flows, not normal routing".to_string(),
        });
    }

    if let Some(family) = family {
        match family.exact_state_mobility {
            ExactStateMobility::Accepted => diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Info,
                code: PlanReasonCode::ExactStateMobilityAccepted,
                message: format!(
                    "{} exact state mobility is within current payload policy",
                    family.family_id
                ),
            }),
            ExactStateMobility::RejectedTooLarge => diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Warning,
                code: PlanReasonCode::ExactStateMobilityRejected,
                message: format!(
                    "{} exact state mobility is rejected for normal routing; route activations and keep live state sticky",
                    family.family_id
                ),
            }),
            ExactStateMobility::Untested => {}
        }
    }

    for boundary in boundaries {
        if boundary.decision == BoundaryDecision::Rejected {
            diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Error,
                code: boundary
                    .reason_codes
                    .iter()
                    .copied()
                    .find(|code| *code == PlanReasonCode::SharedKvRegionCut)
                    .unwrap_or(PlanReasonCode::RecurrentStateTransferRejected),
                message: format!(
                    "boundary at layer {} is rejected: {}",
                    boundary.layer_boundary,
                    boundary.messages.join("; ")
                ),
            });
        }
    }

    diagnostics
}
