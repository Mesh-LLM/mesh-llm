use std::ffi::c_char;

use crate::TensorRole;

pub const ACTIVATION_FRAME_VERSION: u32 = 2;
pub const ACTIVATION_BOUNDARY_DESC_VERSION: u32 = 2;
pub const ACTIVATION_IDENTITY_BYTES: usize = 32;
pub const ACTIVATION_MAX_DIMS: usize = 4;
pub const ACTIVATION_MAX_PARTS: usize = 16;
pub const ACTIVATION_PART_OPTIONAL: u32 = 1 << 0;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorInfo {
    pub name: *const c_char,
    pub layer_index: i32,
    pub role: TensorRole,
    pub ggml_type: u32,
    pub byte_size: u64,
    pub element_count: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ActivationPartDesc {
    pub identity: [u8; ACTIVATION_IDENTITY_BYTES],
    pub ggml_type: u32,
    pub rank: u32,
    pub token_axis: i32,
    pub flags: u32,
    pub dimensions: [i64; ACTIVATION_MAX_DIMS],
    pub byte_strides: [u64; ACTIVATION_MAX_DIMS],
    pub payload_offset: u64,
    pub payload_bytes: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ActivationDesc {
    pub version: u32,
    pub producer_stage_index: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub token_count: u32,
    pub sequence_count: u32,
    pub part_count: u32,
    pub reserved: u32,
    pub payload_bytes: u64,
    pub frontier_identity: [u8; ACTIVATION_IDENTITY_BYTES],
    pub parts: [ActivationPartDesc; ACTIVATION_MAX_PARTS],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ActivationBoundaryDesc {
    pub version: u32,
    pub part_count: u32,
    pub frontier_identity: [u8; ACTIVATION_IDENTITY_BYTES],
    pub parts: [ActivationPartDesc; ACTIVATION_MAX_PARTS],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LogitBias {
    pub token_id: i32,
    pub bias: f32,
}
