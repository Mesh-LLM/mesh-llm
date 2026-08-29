use std::ffi::c_char;

use crate::{ActivationDType, ActivationLayout, TensorRole};

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
#[derive(Debug, Clone, Copy)]
pub struct ActivationDesc {
    pub version: u32,
    pub dtype: ActivationDType,
    pub layout: ActivationLayout,
    pub producer_stage_index: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub token_count: u32,
    pub sequence_count: u32,
    pub payload_bytes: u64,
    pub flags: u64,
}

/// Native boundary tensor introspection for the last emitted activation frame.
///
/// Mirrors `skippy_boundary_tensor_info` from `include/skippy/activation.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BoundaryTensorInfo {
    pub ggml_type: i32,
    pub ne: [i64; 4],
    pub element_size: u32,
    pub emits_frame: u8,
    pub reserved: [u8; 7],
}

impl Default for BoundaryTensorInfo {
    fn default() -> Self {
        Self {
            ggml_type: -1,
            ne: [0; 4],
            element_size: 0,
            emits_frame: 0,
            reserved: [0; 7],
        }
    }
}

pub const ACTIVATION_FLAG_INKLING_MTP_EMBD: u64 = 1 << 2;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LogitBias {
    pub token_id: i32,
    pub bias: f32,
}
