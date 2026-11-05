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
//! The struct layouts are generated from the pinned `include/llama.h`; this
//! module keeps only the ergonomic aliases, the draft-context function table,
//! and the symbol resolvers. The `llama-ext.h` entry points keep C++ linkage
//! and are bound by their Itanium-mangled names.

use std::ffi::c_char;

use crate::Opaque;
use crate::bindings;

pub type LlamaToken = bindings::llama::llama_token;
pub type LlamaPos = bindings::llama::llama_pos;
pub type LlamaSeqId = bindings::llama::llama_seq_id;
pub type LlamaBatch = bindings::llama::llama_batch;
pub type LlamaModelParams = bindings::llama::llama_model_params;
pub type LlamaContextParams = bindings::llama::llama_context_params;
pub type LlamaProgressCallback = bindings::llama::llama_progress_callback;
pub type GgmlSchedEvalCallback = bindings::llama::ggml_backend_sched_eval_callback;
pub type GgmlAbortCallback = bindings::llama::ggml_abort_callback;
pub type LlamaContextType = bindings::llama::llama_context_type;
pub type LlamaFlashAttnType = bindings::llama::llama_flash_attn_type;
pub type LlamaRopeType = bindings::llama::llama_rope_type;
pub type LlamaSplitMode = bindings::llama::llama_split_mode;
pub type GgmlType = bindings::llama::ggml_type;

/// `LLAMA_CONTEXT_TYPE_DEFAULT`.
pub const LLAMA_CONTEXT_TYPE_DEFAULT: LlamaContextType = LlamaContextType::Default;
/// `LLAMA_ROPE_TYPE_MROPE` (`GGML_ROPE_TYPE_MROPE`), as the raw `i32` the
/// draft model reports through `model_rope_type`.
pub const LLAMA_ROPE_TYPE_MROPE: i32 = LlamaRopeType::Mrope as i32;
/// `LLAMA_SPLIT_MODE_NONE`.
pub const LLAMA_SPLIT_MODE_NONE: LlamaSplitMode = LlamaSplitMode::None;

/// Decode a cache-type code (0 selects F32, a positive value selects that
/// ggml type ordinal) into the generated `ggml_type`.
///
/// Returns `None` for codes that are not declared `ggml_type` variants: the
/// generated enum is `#[repr(u32)]` with gaps, so an arbitrary code must
/// never be materialized unchecked. Each arm reads its discriminant from the
/// generated enum, so a header-side renumber is followed coherently and a
/// removed variant fails to compile; a newly added variant simply decodes as
/// `None` here until this table learns it, which callers propagate as an
/// error instead of undefined behavior.
pub fn ggml_type_from_cache_code(code: i32) -> Option<GgmlType> {
    use bindings::llama::ggml_type as raw;
    let value = match code {
        value if value == raw::F32 as i32 => raw::F32,
        value if value == raw::F16 as i32 => raw::F16,
        value if value == raw::Q4_0 as i32 => raw::Q4_0,
        value if value == raw::Q4_1 as i32 => raw::Q4_1,
        value if value == raw::Q5_0 as i32 => raw::Q5_0,
        value if value == raw::Q5_1 as i32 => raw::Q5_1,
        value if value == raw::Q8_0 as i32 => raw::Q8_0,
        value if value == raw::Q8_1 as i32 => raw::Q8_1,
        value if value == raw::Q2K as i32 => raw::Q2K,
        value if value == raw::Q3K as i32 => raw::Q3K,
        value if value == raw::Q4K as i32 => raw::Q4K,
        value if value == raw::Q5K as i32 => raw::Q5K,
        value if value == raw::Q6K as i32 => raw::Q6K,
        value if value == raw::Q8K as i32 => raw::Q8K,
        value if value == raw::IQ2XXS as i32 => raw::IQ2XXS,
        value if value == raw::IQ2XS as i32 => raw::IQ2XS,
        value if value == raw::IQ3XXS as i32 => raw::IQ3XXS,
        value if value == raw::IQ1S as i32 => raw::IQ1S,
        value if value == raw::IQ4NL as i32 => raw::IQ4NL,
        value if value == raw::IQ3S as i32 => raw::IQ3S,
        value if value == raw::IQ2S as i32 => raw::IQ2S,
        value if value == raw::IQ4XS as i32 => raw::IQ4XS,
        value if value == raw::I8 as i32 => raw::I8,
        value if value == raw::I16 as i32 => raw::I16,
        value if value == raw::I32 as i32 => raw::I32,
        value if value == raw::I64 as i32 => raw::I64,
        value if value == raw::F64 as i32 => raw::F64,
        value if value == raw::IQ1M as i32 => raw::IQ1M,
        value if value == raw::Bf16 as i32 => raw::Bf16,
        value if value == raw::TQ1_0 as i32 => raw::TQ1_0,
        value if value == raw::TQ2_0 as i32 => raw::TQ2_0,
        value if value == raw::Mxfp4 as i32 => raw::Mxfp4,
        value if value == raw::Nvfp4 as i32 => raw::Nvfp4,
        value if value == raw::Q1_0 as i32 => raw::Q1_0,
        value if value == raw::Q2_0 as i32 => raw::Q2_0,
        _ => return None,
    };
    Some(value)
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

        assert_eq!(size_of::<LlamaContextParams>(), 168);
        assert_eq!(offset_of!(LlamaContextParams, flash_attn_type), 52);
        assert_eq!(offset_of!(LlamaContextParams, cb_eval), 88);
        assert_eq!(offset_of!(LlamaContextParams, type_k), 104);
        assert_eq!(offset_of!(LlamaContextParams, moe_cache_size), 112);
        assert_eq!(offset_of!(LlamaContextParams, embeddings), 136);
        assert_eq!(offset_of!(LlamaContextParams, samplers), 144);
        assert_eq!(offset_of!(LlamaContextParams, ctx_other), 160);
    }
}
