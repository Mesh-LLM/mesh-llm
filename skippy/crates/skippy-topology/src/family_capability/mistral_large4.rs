use crate::{
    ExactStateMobility, FamilyCapabilityRecord, LayerRange, SplitConstraint, SplitConstraintKind,
};

pub(super) fn capability(layer_count: u32, activation_width: u32) -> FamilyCapabilityRecord {
    let mut record = super::dense_family_capability(
        "mistral_large4",
        layer_count,
        activation_width,
        ExactStateMobility::Untested,
    );
    record.split_constraints.push(SplitConstraint {
        kind: SplitConstraintKind::SharedKvProducerConsumer,
        range: LayerRange {
            start: 35,
            end: layer_count,
        },
        forbidden_boundaries: Vec::new(),
        reject_boundary_inside: true,
        reason: "Mistral Large 4 layers after 35 reuse layer 35's KV cache; keep their source and consumers in the final slice".to_string(),
    });
    record
}

#[cfg(test)]
mod tests {
    use crate::{
        BoundaryDecision, NodeSpec, PlannerPolicy, TopologyPlanRequest, dense_attention_layers,
        plan_balanced_accepted_contiguous, plan_contiguous_with_splits,
    };

    #[test]
    fn shared_kv_source_stays_in_final_slice() {
        let model_id = "unsloth/Mistral-Large-4-1T-A52B-GGUF:UD-IQ1_S";
        let request = TopologyPlanRequest {
            topology_id: "mistral-large4-manual".to_string(),
            model_id: model_id.to_string(),
            layers: dense_attention_layers(72, 1),
            nodes: (0..3)
                .map(|i| NodeSpec {
                    node_id: format!("stage-{i}"),
                    cached_slice_bytes: 0,
                    vram_bytes: 0,
                })
                .collect(),
            family: super::super::infer_family_capability(model_id, 72, 6144),
            policy: PlannerPolicy::default(),
        };
        let invalid = plan_contiguous_with_splits(&request, &[36]).unwrap();
        assert_eq!(invalid.boundaries[0].decision, BoundaryDecision::Rejected);
        let two = plan_balanced_accepted_contiguous(&request, 2).unwrap();
        let three = plan_balanced_accepted_contiguous(&request, 3).unwrap();
        assert_eq!(two.boundaries[0].layer_boundary, 35);
        assert_eq!(three.boundaries[1].layer_boundary, 35);
        assert!(three.boundaries[0].layer_boundary < 35);
    }
}
