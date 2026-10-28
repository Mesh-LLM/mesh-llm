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

/// `skippy_memory_cache_capability` bits.
pub const MEMORY_CACHE_RESIDENT: u32 = 1 << 0;
pub const MEMORY_CACHE_KV_RECURRENT: u32 = 1 << 1;
pub const MEMORY_CACHE_FULL_STATE: u32 = 1 << 2;
/// Every attention cell lives in a host-addressable backend buffer, so a KV
/// page can be described in place with `skippy_export_kv_page_spans`.
pub const MEMORY_CACHE_HOST_VISIBLE_KV: u32 = 1 << 3;
/// Hybrid memory whose attention cells can be tagged into a cache sequence
/// (one unified stream); the recurrent state travels as a snapshot.
pub const MEMORY_CACHE_RESIDENT_KV_RECURRENT: u32 = 1 << 4;

/// One host-addressable slice of a KV page, in payload order. Mirrors
/// `struct skippy_kv_page_span`. The pointer aliases live native cells and is
/// only meaningful under the validity window `skippy_export_kv_page_spans`
/// documents.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KvPageSpan {
    pub data: *const c_void,
    pub bytes: usize,
}

impl Default for KvPageSpan {
    fn default() -> Self {
        Self {
            data: std::ptr::null(),
            bytes: 0,
        }
    }
}

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
