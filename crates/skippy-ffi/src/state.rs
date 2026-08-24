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
    pub payload_offset: u64,
    pub payload_bytes: u64,
    pub flags: u64,
}

pub const KV_PAGE_CODEC_SINGLE_V1: u32 = 1;
pub const KV_PAGE_CODEC_ISWA_COMPOSITE_V1: u32 = 2;

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
    pub payload_bytes: u64,
    pub flags: u64,
    pub codec: u32,
    pub component_count: u32,
    pub components: [KvPageComponentDesc; 2],
}
