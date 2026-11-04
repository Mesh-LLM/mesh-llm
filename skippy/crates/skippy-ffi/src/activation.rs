//! Ergonomic aliases for the generated activation-frame ABI types.

use crate::bindings;

pub const ACTIVATION_FRAME_VERSION: u32 = bindings::skippy::SKIPPY_ACTIVATION_FRAME_VERSION;
pub const ACTIVATION_BOUNDARY_DESC_VERSION: u32 =
    bindings::skippy::SKIPPY_ACTIVATION_BOUNDARY_DESC_VERSION;
pub const ACTIVATION_IDENTITY_BYTES: usize =
    bindings::skippy::SKIPPY_ACTIVATION_IDENTITY_BYTES as usize;
pub const ACTIVATION_MAX_DIMS: usize = bindings::skippy::SKIPPY_ACTIVATION_MAX_DIMS as usize;
pub const ACTIVATION_MAX_PARTS: usize = bindings::skippy::SKIPPY_ACTIVATION_MAX_PARTS as usize;
pub const ACTIVATION_PART_OPTIONAL: u32 =
    bindings::skippy::skippy_ffi_bridge_activation_part_optional;

pub type TensorInfo = bindings::skippy::skippy_tensor_info;
pub type ActivationPartDesc = bindings::skippy::skippy_activation_part_desc;

#[allow(clippy::derivable_impls)]
impl Default for ActivationPartDesc {
    fn default() -> Self {
        Self {
            identity: [0; ACTIVATION_IDENTITY_BYTES],
            ggml_type: 0,
            rank: 0,
            token_axis: 0,
            flags: 0,
            dimensions: [0; ACTIVATION_MAX_DIMS],
            byte_strides: [0; ACTIVATION_MAX_DIMS],
            payload_offset: 0,
            payload_bytes: 0,
        }
    }
}

pub type ActivationDesc = bindings::skippy::skippy_activation_desc;
pub type ActivationBoundaryDesc = bindings::skippy::skippy_activation_boundary_desc;

#[allow(clippy::derivable_impls)]
impl Default for ActivationBoundaryDesc {
    fn default() -> Self {
        Self {
            version: ACTIVATION_BOUNDARY_DESC_VERSION,
            part_count: 0,
            frontier_identity: [0; ACTIVATION_IDENTITY_BYTES],
            parts: [ActivationPartDesc::default(); ACTIVATION_MAX_PARTS],
        }
    }
}

pub type LogitBias = bindings::skippy::llama_logit_bias;
