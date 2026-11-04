//! Ergonomic aliases for the generated receipt ABI types.

use crate::bindings::skippy as ffi;
use ffi::{
    skippy_receipt_attention_window_kind as AttentionWindow, skippy_receipt_memory_op as MemoryOp,
    skippy_receipt_replay_kind as ReplayKind, skippy_receipt_tensor_flag as TensorFlag,
    skippy_receipt_tensor_kind as TensorKind,
};

pub const RECEIPT_TENSOR_RESIDUAL: i32 = TensorKind::Residual as i32;
pub const RECEIPT_TENSOR_MIXING_INPUT: i32 = TensorKind::MixingInput as i32;
pub const RECEIPT_TENSOR_MIXING_STORED: i32 = TensorKind::MixingStored as i32;
pub const RECEIPT_TENSOR_MIXING_OUTPUT: i32 = TensorKind::MixingOutput as i32;
pub const RECEIPT_TENSOR_MATMUL_INPUT: i32 = TensorKind::MatmulInput as i32;
pub const RECEIPT_TENSOR_MATMUL_OUTPUT: i32 = TensorKind::MatmulOutput as i32;
pub const RECEIPT_TENSOR_EXPERT_IDS: i32 = TensorKind::ExpertIds as i32;
pub const RECEIPT_TENSOR_SELECTION: i32 = TensorKind::Selection as i32;
pub const RECEIPT_TENSOR_SELECTION_SCORES: i32 = TensorKind::SelectionScores as i32;
pub const RECEIPT_TENSOR_LOGITS: i32 = TensorKind::Logits as i32;
pub const RECEIPT_TENSOR_MIXING_STATE: i32 = TensorKind::MixingState as i32;
pub const RECEIPT_TENSOR_MIXING_STATE_OUT: i32 = TensorKind::MixingStateOut as i32;

pub const RECEIPT_TENSOR_ROUTER: i32 = TensorFlag::Router as i32;

pub const RECEIPT_MEMORY_CLEAR: i32 = MemoryOp::Clear as i32;
pub const RECEIPT_MEMORY_SEQ_RM: i32 = MemoryOp::SeqRm as i32;
pub const RECEIPT_MEMORY_SEQ_CP: i32 = MemoryOp::SeqCp as i32;
pub const RECEIPT_MEMORY_SEQ_KEEP: i32 = MemoryOp::SeqKeep as i32;
pub const RECEIPT_MEMORY_SEQ_SHIFT: i32 = MemoryOp::SeqShift as i32;
pub const RECEIPT_MEMORY_STATE_READ: i32 = MemoryOp::StateRead as i32;
pub const RECEIPT_MEMORY_COMPACT: i32 = MemoryOp::Compact as i32;

pub const RECEIPT_REPLAY_DECODE: i32 = ReplayKind::Decode as i32;
pub const RECEIPT_REPLAY_SEQ_RM: i32 = ReplayKind::SeqRm as i32;

pub const RECEIPT_ATTENTION_CAUSAL: i32 = AttentionWindow::Causal as i32;
pub const RECEIPT_ATTENTION_SLIDING: i32 = AttentionWindow::Sliding as i32;
pub const RECEIPT_ATTENTION_CHUNKED: i32 = AttentionWindow::Chunked as i32;

pub type ReceiptTensor = ffi::skippy_receipt_tensor;
pub type ReceiptUbatch = ffi::skippy_receipt_ubatch;
pub type ReceiptMemoryEvent = ffi::skippy_receipt_memory_event;
pub type ReceiptReplayStep = ffi::skippy_receipt_replay_step;
pub type ReceiptAnchor = ffi::skippy_receipt_anchor;
pub type ReceiptAttention = ffi::skippy_receipt_attention;
pub type ReceiptGlueValue = ffi::skippy_receipt_glue_value;
pub type ReceiptGlueRequest = ffi::skippy_receipt_glue_request;
pub type ReceiptUbatchFn = ffi::skippy_receipt_ubatch_fn;
pub type ReceiptMemoryFn = ffi::skippy_receipt_memory_fn;
pub type ReceiptStateFn = ffi::skippy_receipt_state_fn;
