use std::ffi::{c_char, c_void};

pub const RECEIPT_TENSOR_RESIDUAL: i32 = 0;
pub const RECEIPT_TENSOR_MIXING_INPUT: i32 = 1;
pub const RECEIPT_TENSOR_MIXING_STORED: i32 = 2;
pub const RECEIPT_TENSOR_MIXING_OUTPUT: i32 = 3;
pub const RECEIPT_TENSOR_MATMUL_INPUT: i32 = 4;
pub const RECEIPT_TENSOR_MATMUL_OUTPUT: i32 = 5;
pub const RECEIPT_TENSOR_EXPERT_IDS: i32 = 6;
pub const RECEIPT_TENSOR_SELECTION: i32 = 7;
pub const RECEIPT_TENSOR_SELECTION_SCORES: i32 = 8;

pub const RECEIPT_TENSOR_ROUTER: i32 = 1;

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
    pub index: i32,
    pub ggml_type: i32,
    pub flags: i32,
    pub len: i64,
    pub row_bytes: i64,
    pub n_cols: i64,
    pub values: *const c_void,
    pub name: *const c_char,
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
    pub n_unchecked_weights: usize,
    pub unchecked_weights: *const *const c_char,
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

/// Mirrors `struct skippy_receipt_glue_value`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReceiptGlueValue {
    pub kind: i32,
    pub index: i32,
    pub ggml_type: i32,
    pub len: i64,
    pub values: *const c_void,
}

/// Mirrors `struct skippy_receipt_glue_request`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReceiptGlueRequest {
    pub layer: i32,
    pub n_cols: i32,
    pub pos: *const i32,
    pub residual: *const f32,
    pub residual_len: i64,
    pub n_values: usize,
    pub values: *const ReceiptGlueValue,
}

/// Mirrors `skippy_receipt_ubatch_fn`.
pub type ReceiptUbatchFn =
    Option<unsafe extern "C" fn(user_data: *mut c_void, ubatch: *const ReceiptUbatch)>;

/// Mirrors `skippy_receipt_memory_fn`.
pub type ReceiptMemoryFn =
    Option<unsafe extern "C" fn(user_data: *mut c_void, event: *const ReceiptMemoryEvent)>;
