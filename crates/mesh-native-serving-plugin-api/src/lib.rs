//! Stable native plugin boundary for Mesh's local Skippy serving path.
//!
//! The ABI is deliberately smaller than Mesh's Rust crate graph. Plugins are
//! independently compiled dynamic libraries, so only fixed-layout values,
//! opaque handles, borrowed byte/token slices, and host-owned output buffers
//! cross the boundary. Rust collections, trait objects, futures, and allocator
//! ownership never do.
//!
//! Every function in the table must be thread-safe and return promptly. In
//! particular, proposal submission and polling must never wait for proposer
//! work. Mesh invokes proposal functions on an isolated host worker and owns
//! the absolute decode deadline; a plugin that violates this contract can
//! strand only that worker, never the model decode thread.

use std::ffi::{c_char, c_void};

pub const NATIVE_SERVING_PLUGIN_ABI_V2: u32 = 2;
pub const NATIVE_SERVING_PLUGIN_ENTRY_V2: &[u8] = b"mesh_native_serving_plugin_v2\0";
pub const MAX_DECISION_ID_BYTES: usize = 64;
pub const TOKENIZER_INVENTORY_SCHEMA: u32 = 1;
pub const TOKENIZER_CAPABILITY_ABI: u32 = 1;
pub const MAX_TOKENIZER_INVENTORY_ENTRIES: usize = 1_000_000;
pub const MAX_TOKENIZER_INPUT_PIECES: usize = 4_096;

/// Host-owned typed inventory. This Rust value never crosses the ABI directly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerInventory {
    pub schema_version: u32,
    pub model_id: String,
    pub source_model_sha256: String,
    pub tokenizer_id: String,
    pub tokens: Vec<TokenizerInventoryToken>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerInventoryToken {
    pub id: u32,
    pub piece: TokenizerInventoryPiece,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenizerInventoryPiece {
    Bytes {
        bytes: Vec<u8>,
    },
    /// Opaque bytes for a native special-token descriptor. Mesh never parses
    /// or names this value; the plugin owns its interpretation.
    Control {
        descriptor: Vec<u8>,
    },
}

pub type PluginInstance = *mut c_void;
pub type ProposalOperation = u64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ByteSlice {
    pub pointer: *const u8,
    pub length: usize,
}

impl ByteSlice {
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            pointer: bytes.as_ptr(),
            length: bytes.len(),
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerPieceKind(pub u32);

impl TokenizerPieceKind {
    pub const BYTES: Self = Self(0);
    pub const CONTROL: Self = Self(1);
}

/// Borrowed ABI view of one immutable native token. The referenced bytes are
/// valid only for the duration of `activate`; a plugin must copy or transform
/// them before it returns.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TokenizerInventoryEntry {
    pub id: u32,
    pub piece_kind: TokenizerPieceKind,
    pub bytes: ByteSlice,
}

/// Borrowed ABI view of the complete vocabulary. The host owns the entries and
/// their bytes and passes them only while activating the plugin.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TokenizerInventoryView {
    pub struct_size: usize,
    pub schema_version: u32,
    pub entries: *const TokenizerInventoryEntry,
    pub entry_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TokenizerLimits {
    pub max_input_bytes: usize,
    pub max_output_tokens: usize,
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerInputPieceKind(pub u32);

impl TokenizerInputPieceKind {
    pub const BYTES: Self = Self(0);
    pub const CONTROL: Self = Self(1);
}

impl Default for TokenizerInputPieceKind {
    fn default() -> Self {
        Self::BYTES
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TokenizerInputPiece {
    pub kind: TokenizerInputPieceKind,
    pub bytes: ByteSlice,
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerEncodeStatus(pub u32);

impl TokenizerEncodeStatus {
    pub const OK: Self = Self(0);
    pub const INVALID_ARGUMENT: Self = Self(1);
    pub const UNSUPPORTED_INPUT: Self = Self(2);
    pub const OUTPUT_TOO_SMALL: Self = Self(3);
    pub const LIMIT_EXCEEDED: Self = Self(4);
    pub const UNAVAILABLE: Self = Self(5);
    pub const INTERNAL_ERROR: Self = Self(6);
}

/// A model-bound, host-owned tokenizer capability. The table and its
/// inventory view are lent only for `activate`; the `context` passed to
/// `encode` remains valid until plugin shutdown. Plugins must copy any
/// activation data they need and must perform preparation/encoding outside the
/// latency-sensitive proposal callbacks. Mesh never invokes `encode` from the
/// proposal path.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TokenizerCapability {
    pub struct_size: usize,
    pub abi_version: u32,
    pub model_id: ByteSlice,
    pub source_model_sha256: ByteSlice,
    pub tokenizer_id: ByteSlice,
    pub limits: TokenizerLimits,
    pub binding_digest: [u8; 32],
    pub inventory: *const TokenizerInventoryView,
    pub context: *mut c_void,
    pub encode: EncodeTokenizer,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TokenSlice {
    pub pointer: *const i32,
    pub length: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct U64Slice {
    pub pointer: *const u64,
    pub length: usize,
}

impl U64Slice {
    #[must_use]
    pub fn from_values(values: &[u64]) -> Self {
        Self {
            pointer: values.as_ptr(),
            length: values.len(),
        }
    }
}

impl TokenSlice {
    #[must_use]
    pub fn from_tokens(tokens: &[i32]) -> Self {
        Self {
            pointer: tokens.as_ptr(),
            length: tokens.len(),
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PluginStatus(pub u32);

impl PluginStatus {
    pub const OK: Self = Self(0);
    pub const INVALID_ARGUMENT: Self = Self(1);
    pub const INCOMPATIBLE: Self = Self(2);
    pub const UNAVAILABLE: Self = Self(3);
    pub const INTERNAL_ERROR: Self = Self(4);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProposalPollStatus(pub u32);

impl ProposalPollStatus {
    pub const PENDING: Self = Self(0);
    pub const READY: Self = Self(1);
    pub const ABSTAIN: Self = Self(2);
    pub const FAILED: Self = Self(3);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GenerationTermination(pub u32);

impl GenerationTermination {
    pub const CALLBACK_STOP: Self = Self(0);
    pub const MAX_TOKENS: Self = Self(1);
    pub const CANCELLED: Self = Self(2);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProposalDisposition(pub u32);

impl ProposalDisposition {
    pub const FULL_ACCEPT: Self = Self(0);
    pub const FIRST_MISMATCH: Self = Self(1);
    pub const STOPPED: Self = Self(2);
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProposalDiscardReason(pub u32);

impl ProposalDiscardReason {
    pub const DEADLINE_EXCEEDED: Self = Self(0);
    pub const INVALID_TOKEN_COUNT: Self = Self(1);
    pub const INVALID_TOKEN_ID: Self = Self(2);
    pub const POSITION_MISMATCH: Self = Self(3);
    pub const EXECUTION_FAILED: Self = Self(4);
}

pub type MonotonicNowNs = unsafe extern "C" fn(context: *mut c_void) -> u64;

pub type EncodeTokenizer = unsafe extern "C" fn(
    context: *mut c_void,
    input_pieces: *const TokenizerInputPiece,
    input_piece_count: usize,
    output_tokens: *mut i32,
    output_capacity: usize,
    output_length: *mut usize,
) -> TokenizerEncodeStatus;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ActivationContext {
    pub struct_size: usize,
    pub model_id: ByteSlice,
    pub source_model_sha256: ByteSlice,
    pub tokenizer_id: ByteSlice,
    pub tokenizer_capability: *const TokenizerCapability,
    pub config_path: ByteSlice,
    pub state_directory: ByteSlice,
    pub proposal_deadline_ns: u64,
    pub host_clock_context: *mut c_void,
    pub monotonic_now_ns: MonotonicNowNs,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PluginActivation {
    pub instance: PluginInstance,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationStart {
    pub struct_size: usize,
    pub request_id: u64,
    pub session_id: u64,
    pub agent_session_id: ByteSlice,
    pub prompt_token_ids: TokenSlice,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationCommit {
    pub struct_size: usize,
    pub request_id: u64,
    pub session_id: u64,
    pub generated_token_count: u64,
    pub token_ids: TokenSlice,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationAbort {
    pub struct_size: usize,
    pub request_id: u64,
    pub session_id: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GenerationFinish {
    pub struct_size: usize,
    pub request_id: u64,
    pub session_id: u64,
    pub prompt_token_count: u64,
    pub prompt_token_digest: [u8; 32],
    pub prompt_token_ids: TokenSlice,
    pub generated_token_ids: TokenSlice,
    pub final_session_position: u64,
    pub termination: GenerationTermination,
    pub model_generation_elapsed_us: u64,
    pub has_request_to_first_token: bool,
    pub request_to_first_token_us: u64,
    pub request_to_token_emission_us: U64Slice,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ProposalQuery {
    pub struct_size: usize,
    pub request_id: u64,
    pub session_id: u64,
    pub prompt_token_count: u64,
    pub committed_token_count: u64,
    pub decode_step: u64,
    pub max_proposal_tokens: u64,
    pub absolute_deadline_ns: u64,
}

#[repr(C)]
#[derive(Debug)]
pub struct ProposalOutput {
    pub struct_size: usize,
    pub decision_id: *mut u8,
    pub decision_id_capacity: usize,
    pub decision_id_length: usize,
    pub token_ids: *mut i32,
    pub token_capacity: usize,
    pub token_length: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProposalOutcome {
    pub struct_size: usize,
    pub decision_id: ByteSlice,
    pub disposition: ProposalDisposition,
    pub proposal_token_count: u64,
    pub verification_rows: u64,
    pub accepted_proposal_tokens: u64,
    pub committed_tokens: TokenSlice,
    pub verification_row_predictions: TokenSlice,
    pub canonical_prediction_count: u64,
    pub has_correction_or_boundary_token: bool,
    pub correction_or_boundary_token: i32,
    pub base_position: u64,
    pub position_after_verification: u64,
    pub canonical_position: u64,
    pub trimmed_rows: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProposalDiscard {
    pub struct_size: usize,
    pub decision_id: ByteSlice,
    pub reason: ProposalDiscardReason,
}

pub type ActivatePlugin = unsafe extern "C" fn(
    context: *const ActivationContext,
    activation: *mut PluginActivation,
) -> PluginStatus;
pub type ShutdownPlugin = unsafe extern "C" fn(instance: PluginInstance) -> PluginStatus;
pub type BeginGeneration =
    unsafe extern "C" fn(instance: PluginInstance, event: *const GenerationStart) -> PluginStatus;
pub type CommitGeneration =
    unsafe extern "C" fn(instance: PluginInstance, event: *const GenerationCommit) -> PluginStatus;
pub type AbortGeneration =
    unsafe extern "C" fn(instance: PluginInstance, event: *const GenerationAbort) -> PluginStatus;
pub type FinishGeneration =
    unsafe extern "C" fn(instance: PluginInstance, event: *const GenerationFinish) -> PluginStatus;
pub type StartProposal = unsafe extern "C" fn(
    instance: PluginInstance,
    query: *const ProposalQuery,
    operation: *mut ProposalOperation,
) -> PluginStatus;
pub type PollProposal = unsafe extern "C" fn(
    instance: PluginInstance,
    operation: ProposalOperation,
    output: *mut ProposalOutput,
) -> ProposalPollStatus;
pub type CancelProposal =
    unsafe extern "C" fn(instance: PluginInstance, operation: ProposalOperation);
pub type ReportProposal =
    unsafe extern "C" fn(instance: PluginInstance, outcome: *const ProposalOutcome) -> PluginStatus;
pub type DiscardProposal =
    unsafe extern "C" fn(instance: PluginInstance, discard: *const ProposalDiscard) -> PluginStatus;
pub type LastError =
    unsafe extern "C" fn(instance: PluginInstance, output: *mut c_char, capacity: usize) -> usize;

#[repr(C)]
pub struct NativeServingPluginV2 {
    pub abi_version: u32,
    pub struct_size: usize,
    pub plugin_name: ByteSlice,
    pub activate: ActivatePlugin,
    pub shutdown: ShutdownPlugin,
    pub begin_generation: BeginGeneration,
    pub commit_generation: CommitGeneration,
    pub abort_generation: AbortGeneration,
    pub finish_generation: FinishGeneration,
    pub start_proposal: StartProposal,
    pub poll_proposal: PollProposal,
    pub cancel_proposal: CancelProposal,
    pub report_proposal: ReportProposal,
    pub discard_proposal: DiscardProposal,
    pub last_error: LastError,
}

// SAFETY: the ABI requires this table and the bytes referenced by
// `plugin_name` to remain immutable and valid for the loaded library's entire
// lifetime. The host copies the name during load and only calls function
// pointers afterward.
unsafe impl Sync for NativeServingPluginV2 {}

pub type NativeServingPluginEntryV2 = unsafe extern "C" fn() -> *const NativeServingPluginV2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_slices_preserve_exact_addresses_and_lengths() {
        let bytes = b"cacheline";
        let tokens = [-1, 0, 42];
        let byte_slice = ByteSlice::from_bytes(bytes);
        let token_slice = TokenSlice::from_tokens(&tokens);
        assert_eq!(byte_slice.pointer, bytes.as_ptr());
        assert_eq!(byte_slice.length, bytes.len());
        assert_eq!(token_slice.pointer, tokens.as_ptr());
        assert_eq!(token_slice.length, tokens.len());
    }

    #[test]
    fn initial_contract_is_v2() {
        assert_eq!(MAX_DECISION_ID_BYTES, 64);
        assert_eq!(NATIVE_SERVING_PLUGIN_ABI_V2, 2);
        assert_eq!(TOKENIZER_CAPABILITY_ABI, 1);
    }

    #[test]
    fn structured_input_piece_kinds_are_stable_and_opaque() {
        assert_eq!(TokenizerInputPieceKind::BYTES.0, 0);
        assert_eq!(TokenizerInputPieceKind::CONTROL.0, 1);
        let descriptor = [0xff, 0x00];
        let piece = TokenizerInputPiece {
            kind: TokenizerInputPieceKind::CONTROL,
            bytes: ByteSlice::from_bytes(&descriptor),
        };
        assert_eq!(piece.bytes.length, 2);
        assert_eq!(piece.bytes.pointer, descriptor.as_ptr());
    }
}
