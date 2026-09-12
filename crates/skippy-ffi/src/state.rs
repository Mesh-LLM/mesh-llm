use std::ffi::c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvPageComponentDesc {
    pub version: u32,
    pub role: u32,
    pub token_start: u64,
    pub token_count: u64,
    pub layer_count: u32,
    pub k_type: u32,
    pub v_type: u32,
    pub k_row_bytes: u32,
    pub v_row_bytes: u32,
    pub v_element_bytes: u32,
    pub k_idx_row_bytes: u32,
    pub payload_offset: u64,
    pub payload_bytes: u64,
    pub flags: u64,
}

pub const KV_PAGE_CODEC_SINGLE_V1: u32 = 1;
pub const KV_PAGE_CODEC_ISWA_COMPOSITE_V1: u32 = 2;

pub const KV_PAGE_FLAG_V_TRANSPOSED: u64 = 1 << 0;
pub const KV_PAGE_FLAG_HAS_K_IDX: u64 = 1 << 1;

pub const CACHEGEN_RECORD_V1_ABI_VERSION: u32 = 1;
pub const CACHEGEN_RECORD_F16: u32 = 0;
pub const CACHEGEN_RECORD_EXACT: u32 = 1;
pub const CACHEGEN_RECORD_F16_TRANSPOSED: u32 = 2;
pub const CACHEGEN_RECORD_F32: u32 = 3;
pub const CACHEGEN_RECORD_F32_TRANSPOSED: u32 = 4;
pub const CACHEGEN_RECORD_Q8_0: u32 = 5;
pub const CACHEGEN_RECORD_Q4_0: u32 = 6;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CacheGenRecordV1 {
    pub abi_version: u32,
    pub kind: u32,
    pub element_bytes: u32,
    pub reserved0: u32,
    pub output_offset: u64,
    pub decoded_bytes: u64,
    pub token_count: u64,
    pub token_start: u64,
    pub total_tokens: u64,
    pub payload: *const c_void,
    pub payload_bytes: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct KvPageDesc {
    pub version: u32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub token_start: u64,
    pub token_count: u64,
    pub layer_count: u32,
    pub k_type: u32,
    pub v_type: u32,
    pub k_row_bytes: u32,
    pub v_row_bytes: u32,
    pub v_element_bytes: u32,
    pub k_idx_row_bytes: u32,
    pub payload_bytes: u64,
    pub flags: u64,
    pub codec: u32,
    pub component_count: u32,
    pub components: [KvPageComponentDesc; 2],
}
