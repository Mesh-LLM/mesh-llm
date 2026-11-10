//! Ergonomic aliases for the generated KV-page ABI types.

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
