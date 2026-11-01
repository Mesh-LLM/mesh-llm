//! Ergonomic aliases for the generated KV-page and CacheGen ABI types.

use crate::bindings;

pub type KvPageComponentDesc = bindings::skippy::skippy_kv_page_component_desc;

#[allow(clippy::derivable_impls)]
impl Default for KvPageComponentDesc {
    fn default() -> Self {
        Self {
            version: 0,
            role: 0,
            token_start: 0,
            token_count: 0,
            layer_count: 0,
            k_type: 0,
            v_type: 0,
            k_row_bytes: 0,
            v_row_bytes: 0,
            v_element_bytes: 0,
            k_idx_row_bytes: 0,
            payload_offset: 0,
            payload_bytes: 0,
            flags: 0,
        }
    }
}

pub const KV_PAGE_CODEC_SINGLE_V1: u32 = bindings::skippy::skippy_kv_page_codec::SingleV1 as u32;
pub const KV_PAGE_CODEC_ISWA_COMPOSITE_V1: u32 =
    bindings::skippy::skippy_kv_page_codec::IswaCompositeV1 as u32;

pub const KV_PAGE_FLAG_V_TRANSPOSED: u64 =
    bindings::skippy::skippy_kv_page_flag::VTransposed as u64;
pub const KV_PAGE_FLAG_HAS_K_IDX: u64 = bindings::skippy::skippy_kv_page_flag::HasKIdx as u64;

pub const CACHEGEN_RECORD_V1_ABI_VERSION: u32 =
    bindings::skippy::SKIPPY_CACHEGEN_RECORD_V1_ABI_VERSION;
pub const CACHEGEN_RECORD_F16: u32 = bindings::skippy::skippy_cachegen_record_kind::F16 as u32;
pub const CACHEGEN_RECORD_EXACT: u32 = bindings::skippy::skippy_cachegen_record_kind::Exact as u32;
pub const CACHEGEN_RECORD_F16_TRANSPOSED: u32 =
    bindings::skippy::skippy_cachegen_record_kind::F16Transposed as u32;
pub const CACHEGEN_RECORD_F32: u32 = bindings::skippy::skippy_cachegen_record_kind::F32 as u32;
pub const CACHEGEN_RECORD_F32_TRANSPOSED: u32 =
    bindings::skippy::skippy_cachegen_record_kind::F32Transposed as u32;
pub const CACHEGEN_RECORD_Q8_0: u32 = bindings::skippy::skippy_cachegen_record_kind::Q8_0 as u32;
pub const CACHEGEN_RECORD_Q4_0: u32 = bindings::skippy::skippy_cachegen_record_kind::Q4_0 as u32;

pub type CacheGenRecordV1 = bindings::skippy::skippy_cachegen_record_v1;

pub type KvPageDesc = bindings::skippy::skippy_kv_page_desc;

#[allow(clippy::derivable_impls)]
impl Default for KvPageDesc {
    fn default() -> Self {
        Self {
            version: 0,
            layer_start: 0,
            layer_end: 0,
            token_start: 0,
            token_count: 0,
            layer_count: 0,
            k_type: 0,
            v_type: 0,
            k_row_bytes: 0,
            v_row_bytes: 0,
            v_element_bytes: 0,
            k_idx_row_bytes: 0,
            payload_bytes: 0,
            flags: 0,
            codec: 0,
            component_count: 0,
            components: [KvPageComponentDesc::default(); 2],
        }
    }
}
