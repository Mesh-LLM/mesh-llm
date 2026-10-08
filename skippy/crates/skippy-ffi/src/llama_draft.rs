//! Upstream llama.cpp context API for draft models that run beside a Skippy
//! target context.
//!
//! Some speculative drafters (for example DFlash) are ordinary upstream
//! llama.cpp models whose context borrows tensors from the target through
//! `llama_context_params::ctx_other` and consumes target hidden states captured
//! with `llama_set_embeddings_layer_inp`. Skippy's ABI does not create such
//! contexts, so the owning runtime drives them through these upstream entry
//! points directly.
//!
//! The struct mirrors follow the pinned `include/llama.h` field for field. The
//! patch queue does not change these layouts; [`LlamaContextParams`] and
//! [`LlamaModelParams`] are passed by value, so any upstream layout change
//! must be mirrored here before the pin moves.

use std::ffi::{c_char, c_void};

use crate::Opaque;

pub type LlamaToken = i32;
pub type LlamaPos = i32;
pub type LlamaSeqId = i32;

/// `LLAMA_CONTEXT_TYPE_DEFAULT`.
pub const LLAMA_CONTEXT_TYPE_DEFAULT: i32 = 0;
/// `LLAMA_ROPE_TYPE_MROPE` (`GGML_ROPE_TYPE_MROPE`).
pub const LLAMA_ROPE_TYPE_MROPE: i32 = 8;
/// `LLAMA_SPLIT_MODE_NONE`.
pub const LLAMA_SPLIT_MODE_NONE: i32 = 0;

/// Mirror of upstream `llama_batch`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaBatch {
    pub n_tokens: i32,
    pub token: *mut LlamaToken,
    pub embd: *mut f32,
    pub pos: *mut LlamaPos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut LlamaSeqId,
    pub logits: *mut i8,
}

pub type LlamaProgressCallback =
    Option<unsafe extern "C" fn(progress: f32, user_data: *mut c_void) -> bool>;
pub type GgmlSchedEvalCallback =
    Option<unsafe extern "C" fn(tensor: *mut c_void, ask: bool, user_data: *mut c_void) -> bool>;
pub type GgmlAbortCallback = Option<unsafe extern "C" fn(data: *mut c_void) -> bool>;

/// Mirror of upstream `llama_model_params`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaModelParams {
    pub devices: *mut *mut Opaque,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub load_mode: i32,
    pub lazy_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: LlamaProgressCallback,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

/// Mirror of upstream `llama_context_params`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    pub flash_attn_type: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: GgmlSchedEvalCallback,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: i32,
    pub type_v: i32,
    pub abort_callback: GgmlAbortCallback,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut c_void,
    pub n_samplers: usize,
    pub ctx_other: *mut Opaque,
}

/// Upstream llama.cpp entry points for an external draft context.
///
/// Every pointer argument and return value is an upstream llama.cpp object
/// owned by the loaded native runtime (`llama_model`, `llama_context`,
/// `llama_vocab`, `llama_memory_t`, or `ggml_backend_dev_t`).
#[derive(Clone, Copy)]
pub struct LlamaDraftApi {
    pub model_default_params: unsafe extern "C" fn() -> LlamaModelParams,
    pub context_default_params: unsafe extern "C" fn() -> LlamaContextParams,
    pub model_load_from_file:
        unsafe extern "C" fn(path: *const c_char, params: LlamaModelParams) -> *mut Opaque,
    pub model_free: unsafe extern "C" fn(model: *mut Opaque),
    pub init_from_model:
        unsafe extern "C" fn(model: *mut Opaque, params: LlamaContextParams) -> *mut Opaque,
    pub free: unsafe extern "C" fn(ctx: *mut Opaque),
    pub decode: unsafe extern "C" fn(ctx: *mut Opaque, batch: LlamaBatch) -> i32,
    pub get_model: unsafe extern "C" fn(ctx: *const Opaque) -> *const Opaque,
    pub get_memory: unsafe extern "C" fn(ctx: *const Opaque) -> *mut Opaque,
    pub memory_seq_rm: unsafe extern "C" fn(
        mem: *mut Opaque,
        seq_id: LlamaSeqId,
        p0: LlamaPos,
        p1: LlamaPos,
    ) -> bool,
    pub get_logits_ith: unsafe extern "C" fn(ctx: *mut Opaque, i: i32) -> *mut f32,
    pub n_ctx: unsafe extern "C" fn(ctx: *const Opaque) -> u32,
    pub n_batch: unsafe extern "C" fn(ctx: *const Opaque) -> u32,
    pub n_ubatch: unsafe extern "C" fn(ctx: *const Opaque) -> u32,
    pub model_n_embd: unsafe extern "C" fn(model: *const Opaque) -> i32,
    pub model_n_layer: unsafe extern "C" fn(model: *const Opaque) -> i32,
    pub model_rope_type: unsafe extern "C" fn(model: *const Opaque) -> i32,
    pub model_get_vocab: unsafe extern "C" fn(model: *const Opaque) -> *const Opaque,
    pub vocab_n_tokens: unsafe extern "C" fn(vocab: *const Opaque) -> i32,
    pub vocab_mask: unsafe extern "C" fn(vocab: *const Opaque) -> LlamaToken,
    pub set_causal_attn: unsafe extern "C" fn(ctx: *mut Opaque, causal_attn: bool),
    pub set_embeddings_nextn: unsafe extern "C" fn(ctx: *mut Opaque, value: bool, masked: bool),
    pub get_embeddings_nextn: unsafe extern "C" fn(ctx: *mut Opaque) -> *mut f32,
    pub set_embeddings_layer_inp: unsafe extern "C" fn(ctx: *mut Opaque, lid: u32, value: bool),
    pub get_embeddings_layer_inp: unsafe extern "C" fn(ctx: *mut Opaque, lid: u32) -> *mut f32,
    pub model_target_layer_ids: unsafe extern "C" fn(model: *const Opaque) -> *const i32,
    pub model_target_layer_ids_n: unsafe extern "C" fn(model: *const Opaque) -> u32,
    pub model_dflash_selector_top_k: unsafe extern "C" fn(model: *const Opaque) -> i32,
    pub backend_dev_by_name: unsafe extern "C" fn(name: *const c_char) -> *mut Opaque,
}

/// Returns the draft-context API, or `None` when the loaded runtime does not
/// export every entry point (for example a runtime built before DFlash).
#[cfg(feature = "dynamic-runtime")]
pub fn llama_draft_api() -> Option<&'static LlamaDraftApi> {
    crate::dynamic::llama_draft_api()
}

/// The `src/llama-ext.h` entry points keep C++ linkage and are only bound
/// under the Itanium mangling, so Windows builds have no draft-context API.
#[cfg(all(not(feature = "dynamic-runtime"), windows))]
pub fn llama_draft_api() -> Option<&'static LlamaDraftApi> {
    None
}

/// Returns the statically linked draft-context API.
#[cfg(all(not(feature = "dynamic-runtime"), not(windows)))]
pub fn llama_draft_api() -> Option<&'static LlamaDraftApi> {
    use crate::static_bindings as raw;
    static API: LlamaDraftApi = LlamaDraftApi {
        model_default_params: raw::llama_model_default_params,
        context_default_params: raw::llama_context_default_params,
        model_load_from_file: raw::llama_model_load_from_file,
        model_free: raw::llama_model_free,
        init_from_model: raw::llama_init_from_model,
        free: raw::llama_free,
        decode: raw::llama_decode,
        get_model: raw::llama_get_model,
        get_memory: raw::llama_get_memory,
        memory_seq_rm: raw::llama_memory_seq_rm,
        get_logits_ith: raw::llama_get_logits_ith,
        n_ctx: raw::llama_n_ctx,
        n_batch: raw::llama_n_batch,
        n_ubatch: raw::llama_n_ubatch,
        model_n_embd: raw::llama_model_n_embd,
        model_n_layer: raw::llama_model_n_layer,
        model_rope_type: raw::llama_model_rope_type,
        model_get_vocab: raw::llama_model_get_vocab,
        vocab_n_tokens: raw::llama_vocab_n_tokens,
        vocab_mask: raw::llama_vocab_mask,
        set_causal_attn: raw::llama_set_causal_attn,
        set_embeddings_nextn: raw::llama_set_embeddings_nextn,
        get_embeddings_nextn: raw::llama_get_embeddings_nextn,
        set_embeddings_layer_inp: raw::llama_set_embeddings_layer_inp,
        get_embeddings_layer_inp: raw::llama_get_embeddings_layer_inp,
        model_target_layer_ids: raw::llama_model_target_layer_ids,
        model_target_layer_ids_n: raw::llama_model_target_layer_ids_n,
        model_dflash_selector_top_k: raw::llama_model_dflash_selector_top_k,
        backend_dev_by_name: raw::ggml_backend_dev_by_name,
    };
    Some(&API)
}

#[cfg(all(test, target_pointer_width = "64"))]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::{LlamaBatch, LlamaContextParams, LlamaModelParams};

    // Sizes and offsets measured from the pinned `include/llama.h` with a C
    // compiler on a 64-bit target.
    #[test]
    fn mirrors_match_the_pinned_upstream_layouts() {
        assert_eq!(size_of::<LlamaBatch>(), 56);

        assert_eq!(size_of::<LlamaModelParams>(), 80);
        assert_eq!(offset_of!(LlamaModelParams, tensor_split), 40);
        assert_eq!(offset_of!(LlamaModelParams, kv_overrides), 64);
        assert_eq!(offset_of!(LlamaModelParams, load_mtp), 77);

        assert_eq!(size_of::<LlamaContextParams>(), 160);
        assert_eq!(offset_of!(LlamaContextParams, flash_attn_type), 52);
        assert_eq!(offset_of!(LlamaContextParams, cb_eval), 88);
        assert_eq!(offset_of!(LlamaContextParams, type_k), 104);
        assert_eq!(offset_of!(LlamaContextParams, embeddings), 128);
        assert_eq!(offset_of!(LlamaContextParams, samplers), 136);
        assert_eq!(offset_of!(LlamaContextParams, ctx_other), 152);
    }
}
