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

pub const ACTIVATION_FLAG_INKLING_MTP_EMBD: u64 = 1 << 2;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LogitBias {
    pub token_id: i32,
    pub bias: f32,
}
