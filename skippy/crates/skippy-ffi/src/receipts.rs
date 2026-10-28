use std::ffi::{c_char, c_void};

pub const RECEIPT_TENSOR_RESIDUAL: i32 = 0;
pub const RECEIPT_TENSOR_ATTN_Q: i32 = 1;
pub const RECEIPT_TENSOR_ATTN_K: i32 = 2;
pub const RECEIPT_TENSOR_ATTN_V: i32 = 3;
pub const RECEIPT_TENSOR_ATTN_OUT: i32 = 4;
pub const RECEIPT_TENSOR_ATTN_IN: i32 = 5;
pub const RECEIPT_TENSOR_ATTN_PROJ: i32 = 6;
pub const RECEIPT_TENSOR_FFN_IN: i32 = 7;
pub const RECEIPT_TENSOR_FFN_GATE: i32 = 8;
pub const RECEIPT_TENSOR_FFN_UP: i32 = 9;
pub const RECEIPT_TENSOR_FFN_DOWN_IN: i32 = 10;
pub const RECEIPT_TENSOR_FFN_DOWN: i32 = 11;

pub const RECEIPT_MEMORY_CLEAR: i32 = 0;
pub const RECEIPT_MEMORY_SEQ_RM: i32 = 1;
pub const RECEIPT_MEMORY_SEQ_CP: i32 = 2;
pub const RECEIPT_MEMORY_SEQ_KEEP: i32 = 3;
pub const RECEIPT_MEMORY_SEQ_SHIFT: i32 = 4;
pub const RECEIPT_MEMORY_STATE_READ: i32 = 5;
pub const RECEIPT_MEMORY_COMPACT: i32 = 6;

pub const RECEIPT_REPLAY_DECODE: i32 = 0;
pub const RECEIPT_REPLAY_SEQ_RM: i32 = 1;

/// Mirrors `struct skippy_receipt_tensor`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReceiptTensor {
    pub layer: i32,
    pub kind: i32,
    pub len: i64,
    pub n_cols: i64,
    pub values: *const f32,
}

/// Mirrors `struct skippy_receipt_ubatch`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReceiptUbatch {
    pub step: i64,
    pub n_tokens: i32,
    pub token: *const i32,
    pub seq: *const i32,
    pub pos: *const i32,
    pub output: *const i8,
    pub cell: *const u32,
    pub n_kv: u32,
    pub input: *const c_void,
    pub input_row_bytes: i64,
    pub input_type: i32,
    pub n_tensors: usize,
    pub tensors: *const ReceiptTensor,
    pub error: *const c_char,
}

/// Mirrors `struct skippy_receipt_memory_event`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReceiptMemoryEvent {
    pub op: i32,
    pub seq: i32,
    pub seq_dst: i32,
    pub p0: i32,
    pub p1: i32,
}

/// Mirrors `struct skippy_receipt_replay_step`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReceiptReplayStep {
    pub kind: i32,
    pub n_tokens: i32,
    pub token: *const i32,
    pub seq: *const i32,
    pub pos: *const i32,
    pub output: *const i8,
    pub cell: *const u32,
    pub n_kv: u32,
    pub input: *const c_void,
    pub input_row_bytes: i64,
    pub input_type: i32,
    pub p0: i32,
    pub p1: i32,
    pub capture: i32,
}

/// Mirrors `skippy_receipt_ubatch_fn`.
pub type ReceiptUbatchFn =
    Option<unsafe extern "C" fn(user_data: *mut c_void, ubatch: *const ReceiptUbatch)>;

/// Mirrors `skippy_receipt_memory_fn`.
pub type ReceiptMemoryFn =
    Option<unsafe extern "C" fn(user_data: *mut c_void, event: *const ReceiptMemoryEvent)>;
