use skippy_runtime::{
    ACTIVATION_BOUNDARY_DESC_VERSION, ACTIVATION_FRAME_VERSION, ACTIVATION_IDENTITY_BYTES,
    ACTIVATION_MAX_PARTS, ActivationBoundaryDesc, ActivationDesc, ActivationFrame,
    ActivationPartDesc, GGML_TYPE_F32, GGML_TYPE_I32,
};

pub(crate) fn boundary_f32(elements_per_token: u64) -> ActivationBoundaryDesc {
    let mut parts = [ActivationPartDesc::default(); ACTIVATION_MAX_PARTS];
    parts[0] = ActivationPartDesc {
        identity: [1; ACTIVATION_IDENTITY_BYTES],
        ggml_type: GGML_TYPE_F32,
        rank: 2,
        token_axis: 1,
        dimensions: [elements_per_token as i64, -1, 0, 0],
        byte_strides: [4, elements_per_token.saturating_mul(4), 0, 0],
        ..ActivationPartDesc::default()
    };
    ActivationBoundaryDesc {
        version: ACTIVATION_BOUNDARY_DESC_VERSION,
        part_count: 1,
        frontier_identity: [9; ACTIVATION_IDENTITY_BYTES],
        parts,
    }
}

pub(crate) struct PartBytes {
    pub identity: u8,
    pub ggml_type: u32,
    pub flags: u32,
    pub bytes: Vec<u8>,
}

pub(crate) fn frame(token_count: u32, part_specs: Vec<PartBytes>) -> ActivationFrame {
    let mut payload = Vec::new();
    let mut parts = [ActivationPartDesc::default(); ACTIVATION_MAX_PARTS];
    for (index, spec) in part_specs.iter().enumerate() {
        let element_bytes = match spec.ggml_type {
            GGML_TYPE_F32 | GGML_TYPE_I32 => 4,
            other => panic!("unsupported test activation type {other}"),
        };
        let token_count_usize = token_count as usize;
        assert!(token_count_usize > 0);
        assert_eq!(spec.bytes.len() % token_count_usize, 0);
        let row_bytes = spec.bytes.len() / token_count_usize;
        assert_eq!(row_bytes % element_bytes, 0);
        parts[index] = ActivationPartDesc {
            identity: [spec.identity; ACTIVATION_IDENTITY_BYTES],
            ggml_type: spec.ggml_type,
            rank: 2,
            token_axis: 1,
            flags: spec.flags,
            dimensions: [
                (row_bytes / element_bytes) as i64,
                i64::from(token_count),
                0,
                0,
            ],
            byte_strides: [element_bytes as u64, row_bytes as u64, 0, 0],
            payload_offset: payload.len() as u64,
            payload_bytes: spec.bytes.len() as u64,
        };
        payload.extend_from_slice(&spec.bytes);
    }
    ActivationFrame {
        desc: ActivationDesc {
            version: ACTIVATION_FRAME_VERSION,
            producer_stage_index: 1,
            layer_start: 4,
            layer_end: 8,
            token_count,
            sequence_count: 1,
            part_count: part_specs.len() as u32,
            payload_bytes: payload.len() as u64,
            frontier_identity: [9; ACTIVATION_IDENTITY_BYTES],
            parts,
        },
        payload,
    }
}

pub(crate) fn f32_frame(token_count: u32, values: &[f32]) -> ActivationFrame {
    frame(
        token_count,
        vec![PartBytes {
            identity: 1,
            ggml_type: GGML_TYPE_F32,
            flags: 0,
            bytes: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }],
    )
}
