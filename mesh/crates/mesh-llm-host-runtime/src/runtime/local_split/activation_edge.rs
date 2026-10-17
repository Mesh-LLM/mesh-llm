use anyhow::{Context, Result};
use skippy_runtime::ActivationBoundaryDesc;

pub(super) fn required_boundary(
    descriptor: Option<ActivationBoundaryDesc>,
    stage_id: &str,
    edge: &str,
) -> Result<ActivationBoundaryDesc> {
    descriptor.with_context(|| {
        format!("stage {stage_id} did not advertise its graph-observed {edge} activation boundary")
    })
}

pub(super) fn validate_activation_edge(
    producer_id: &str,
    producer: ActivationBoundaryDesc,
    consumer_id: &str,
    consumer: ActivationBoundaryDesc,
) -> Result<()> {
    producer
        .validate_transport_types()
        .with_context(|| format!("invalid graph boundary advertised by producer {producer_id}"))?;
    consumer
        .validate_transport_types()
        .with_context(|| format!("invalid graph boundary advertised by consumer {consumer_id}"))?;
    producer
        .primary_token_width("output")
        .with_context(|| format!("invalid graph boundary advertised by producer {producer_id}"))?;
    consumer
        .primary_token_width("input")
        .with_context(|| format!("invalid graph boundary advertised by consumer {consumer_id}"))?;
    anyhow::ensure!(
        producer == consumer,
        "activation boundary mismatch between producer {producer_id} and consumer {consumer_id}: producer={producer:?} consumer={consumer:?}"
    );
    Ok(())
}

#[cfg(test)]
mod activation_boundary_tests {
    use super::{required_boundary, validate_activation_edge};
    use skippy_runtime::{
        ACTIVATION_BOUNDARY_DESC_VERSION, ACTIVATION_MAX_PARTS, ActivationBoundaryDesc,
        ActivationPartDesc, GGML_TYPE_F32,
    };

    fn boundary(elements_per_token: u64) -> ActivationBoundaryDesc {
        let mut parts = [ActivationPartDesc::default(); ACTIVATION_MAX_PARTS];
        parts[0] = ActivationPartDesc {
            identity: [1; 32],
            ggml_type: GGML_TYPE_F32,
            rank: 2,
            token_axis: 1,
            dimensions: [elements_per_token as i64, -1, 0, 0],
            byte_strides: [4, elements_per_token * 4, 0, 0],
            ..ActivationPartDesc::default()
        };
        ActivationBoundaryDesc {
            version: ACTIVATION_BOUNDARY_DESC_VERSION,
            part_count: 1,
            frontier_identity: [9; 32],
            parts,
        }
    }

    #[test]
    fn matching_graph_boundaries_form_a_valid_stage_edge() {
        validate_activation_edge("stage-0", boundary(1024), "stage-1", boundary(1024))
            .expect("identical graph boundary contracts must match");
    }

    #[test]
    fn three_stage_topology_validates_each_edge_independently() {
        let first_output = boundary(1024);
        let middle_input = boundary(1024);
        let middle_output = boundary(2048);
        let final_input = boundary(2048);

        validate_activation_edge("stage-0", first_output, "stage-1", middle_input)
            .expect("first edge must match");
        validate_activation_edge("stage-1", middle_output, "stage-2", final_input)
            .expect("second edge must match even when its contract differs from the first");

        let error = validate_activation_edge("stage-1", middle_output, "stage-2", boundary(1024))
            .expect_err("a mismatch on the second edge must block the topology");
        assert!(error.to_string().contains("stage-1"));
        assert!(error.to_string().contains("stage-2"));
    }

    #[test]
    fn integer_frontier_can_activate_a_matching_edge() {
        let mut descriptor = boundary(8);
        descriptor.parts[0].ggml_type = skippy_runtime::GGML_TYPE_I32;
        validate_activation_edge("producer", descriptor, "consumer", descriptor).unwrap();
    }

    #[test]
    fn unsupported_types_in_any_part_block_matching_edges() {
        for part_index in 0..3 {
            let mut descriptor = boundary(8);
            descriptor.part_count = 3;
            descriptor.parts[1] = descriptor.parts[0];
            descriptor.parts[1].dimensions[0] = -1;
            descriptor.parts[2] = descriptor.parts[0];
            descriptor.parts[part_index].ggml_type = u32::MAX;
            let error = validate_activation_edge("producer", descriptor, "consumer", descriptor)
                .expect_err("unsupported types must fail even after a dynamic part");
            assert!(format!("{error:#}").contains("transport does not support ggml type"));
        }
    }

    #[test]
    fn supported_types_and_dynamic_auxiliary_parts_can_activate() {
        for ggml_type in [GGML_TYPE_F32, skippy_runtime::GGML_TYPE_I32, 1, 30] {
            let mut descriptor = boundary(8);
            descriptor.part_count = 2;
            descriptor.parts[1] = descriptor.parts[0];
            descriptor.parts[1].ggml_type = ggml_type;
            descriptor.parts[1].dimensions[0] = -1;
            validate_activation_edge("producer", descriptor, "consumer", descriptor).unwrap();
        }
    }

    #[test]
    fn graph_boundary_mismatch_blocks_topology_activation() {
        let error = validate_activation_edge("stage-0", boundary(1024), "stage-1", boundary(896))
            .expect_err("different graph boundary contracts must not connect");
        assert!(error.to_string().contains("activation boundary mismatch"));
    }

    #[test]
    fn every_graph_boundary_field_participates_in_edge_matching() {
        let producer = boundary(1024);
        let mut changed_identity = producer;
        changed_identity.frontier_identity[0] = 8;
        let mut changed_part_identity = producer;
        changed_part_identity.parts[0].identity[0] = 2;
        let mut changed_type = producer;
        changed_type.parts[0].ggml_type = 1;
        let mut changed_rank = producer;
        changed_rank.parts[0].rank = 3;
        let mut changed_axis = producer;
        changed_axis.parts[0].token_axis = 0;
        let mut changed_dimension = producer;
        changed_dimension.parts[0].dimensions[0] = 896;
        let mut changed_stride = producer;
        changed_stride.parts[0].byte_strides[1] += 4;
        let mutations = [
            ActivationBoundaryDesc {
                part_count: 2,
                ..producer
            },
            changed_identity,
            changed_part_identity,
            changed_type,
            changed_rank,
            changed_axis,
            changed_dimension,
            changed_stride,
        ];

        for consumer in mutations {
            assert!(
                validate_activation_edge("stage-0", producer, "stage-1", consumer).is_err(),
                "mutated consumer contract must not activate: {consumer:?}"
            );
        }
    }

    #[test]
    fn identical_but_invalid_graph_boundaries_do_not_activate() {
        let invalid = ActivationBoundaryDesc {
            version: 99,
            ..boundary(1024)
        };
        let error = validate_activation_edge("stage-0", invalid, "stage-1", invalid)
            .expect_err("matching invalid descriptors must fail semantic validation");
        assert!(error.to_string().contains("invalid graph boundary"));
    }

    #[test]
    fn missing_graph_boundary_is_not_reconstructed_from_manifest_width() {
        let error = required_boundary(None, "stage-1", "input")
            .expect_err("generation 11 requires graph-observed boundary descriptors");
        assert!(
            error
                .to_string()
                .contains("did not advertise its graph-observed input")
        );
    }
}
