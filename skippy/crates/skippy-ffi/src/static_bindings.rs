//! Static-link function surface.
//!
//! With `dynamic-runtime` disabled the native archives are linked into the
//! binary and the generated `extern "C"` declarations in
//! [`crate::bindings`] bind directly to the linked symbols. This module
//! re-exports them under the same signatures the dynamic facade exposes, so
//! downstream code compiles identically regardless of linkage mode.
//!
//! Most skippy ABI functions already share their types through the crate's
//! aliases and are re-exported unchanged. The upstream llama/mtmd helpers
//! name llama.cpp opaque types (`llama_context`, `llama_model`,
//! `mtmd_input_chunk`, ...) where the dynamic facade has always used
//! [`crate::Opaque`]; those get thin pointer-cast adapters here. The casts
//! are between mutually opaque C pointer types and are ABI no-ops.

use std::ffi::{c_char, c_int, c_void};

use crate::bindings::{llama, mtmd};
use crate::llama_draft::{LlamaBatch, LlamaContextParams, LlamaModelParams};
use crate::{
    LlamaLogCallback, MtmdContext, MtmdContextParams, MtmdDecoderPos, MtmdHelperGenAudio,
    MtmdInputChunkType, MtmdInputChunks, Opaque,
};

pub use crate::bindings::skippy::*;

pub use llama::{
    ggml_log_set, llama_log_set, llama_model_is_diffusion, llama_model_is_hybrid,
    llama_model_is_recurrent, llama_model_meta_val_str, llama_model_quantize,
    llama_model_quantize_default_params, llama_perf_context,
};
pub use mtmd::{
    mtmd_bitmap_free, mtmd_bitmap_get_n_bytes, mtmd_context_params_default, mtmd_decode_use_mrope,
    mtmd_default_marker, mtmd_free, mtmd_gen_audio_get_info, mtmd_helper_bitmap_init_from_buf,
    mtmd_helper_gen_audio_free, mtmd_helper_gen_audio_get_output, mtmd_helper_gen_audio_reset,
    mtmd_helper_gen_audio_set_input, mtmd_helper_gen_audio_step_gen,
    mtmd_helper_gen_audio_step_prompt, mtmd_helper_get_n_pos, mtmd_helper_get_n_tokens,
    mtmd_helper_init_opt_default, mtmd_helper_video_free, mtmd_input_chunks_free,
    mtmd_input_chunks_init, mtmd_input_chunks_size, mtmd_tokenize,
};

/// # Safety
/// `ctx` must be a live llama context borrowed from the loaded runtime.
pub unsafe fn llama_get_embeddings_ith(ctx: *mut Opaque, index: i32) -> *mut f32 {
    unsafe { llama::llama_get_embeddings_ith(ctx.cast(), index) }
}

/// # Safety
/// `ctx` must be a live llama context borrowed from the loaded runtime.
pub unsafe fn llama_set_embeddings(ctx: *mut Opaque, embeddings: bool) {
    unsafe { llama::llama_set_embeddings(ctx.cast(), embeddings) }
}

/// # Safety
/// `model` must be a live llama model borrowed from the loaded runtime.
pub unsafe extern "C" fn llama_model_get_vocab(model: *const Opaque) -> *const Opaque {
    unsafe { llama::llama_model_get_vocab(model.cast()).cast() }
}

/// # Safety
/// `vocab` must be a live llama vocabulary borrowed from the loaded runtime.
pub unsafe extern "C" fn llama_vocab_n_tokens(vocab: *const Opaque) -> i32 {
    unsafe { llama::llama_vocab_n_tokens(vocab.cast()) }
}

/// # Safety
/// `text_model` must be a live llama model borrowed from the loaded runtime.
pub unsafe fn mtmd_init_from_file(
    mmproj_fname: *const c_char,
    text_model: *const Opaque,
    ctx_params: MtmdContextParams,
) -> *mut MtmdContext {
    unsafe { mtmd::mtmd_init_from_file(mmproj_fname, text_model.cast(), ctx_params) }
}

/// # Safety
/// `lctx` must be a live llama context borrowed from the loaded runtime.
pub unsafe fn mtmd_helper_gen_audio_init(
    lctx: *mut Opaque,
    mctx: *mut MtmdContext,
) -> *mut MtmdHelperGenAudio {
    unsafe { mtmd::mtmd_helper_gen_audio_init(lctx.cast(), mctx) }
}

/// # Safety
/// `image` must be a live mtmd image-tokens pointer borrowed from the runtime.
pub unsafe fn mtmd_helper_image_get_decoder_pos(
    image: *const Opaque,
    pos_0: i32,
    out_pos: *mut MtmdDecoderPos,
) {
    unsafe { mtmd::mtmd_helper_image_get_decoder_pos(image.cast(), pos_0, out_pos) }
}

/// # Safety
/// `lctx` must be a live llama context borrowed from the loaded runtime.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mtmd_helper_eval_chunks(
    ctx: *mut MtmdContext,
    lctx: *mut Opaque,
    chunks: *const MtmdInputChunks,
    n_past: i32,
    seq_id: i32,
    n_batch: i32,
    logits_last: bool,
    new_n_past: *mut i32,
) -> c_int {
    unsafe {
        mtmd::mtmd_helper_eval_chunks(
            ctx,
            lctx.cast(),
            chunks,
            n_past,
            seq_id,
            n_batch,
            logits_last,
            new_n_past,
        )
    }
}

/// # Safety
/// `lctx` must be a live llama context borrowed from the loaded runtime.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mtmd_helper_eval_chunk_single(
    ctx: *mut MtmdContext,
    lctx: *mut Opaque,
    chunk: *const Opaque,
    n_past: i32,
    seq_id: i32,
    n_batch: i32,
    logits_last: bool,
    new_n_past: *mut i32,
) -> c_int {
    unsafe {
        mtmd::mtmd_helper_eval_chunk_single(
            ctx,
            lctx.cast(),
            chunk.cast(),
            n_past,
            seq_id,
            n_batch,
            logits_last,
            new_n_past,
        )
    }
}

/// # Safety
/// `lctx` must be a live llama context borrowed from the loaded runtime.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mtmd_helper_eval_chunk_single_with_callback(
    ctx: *mut MtmdContext,
    lctx: *mut Opaque,
    chunk: *const Opaque,
    n_past: i32,
    seq_id: i32,
    n_batch: i32,
    logits_last: bool,
    new_n_past: *mut i32,
    callback: Option<unsafe extern "C" fn(i32, *mut c_void) -> c_int>,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        mtmd::mtmd_helper_eval_chunk_single_with_callback(
            ctx,
            lctx.cast(),
            chunk.cast(),
            n_past,
            seq_id,
            n_batch,
            logits_last,
            new_n_past,
            callback,
            user_data,
        )
    }
}

/// # Safety
/// The callback must satisfy the usual FFI callback contract for the loaded
/// runtime's logging entrypoints.
pub unsafe fn mtmd_helper_log_set(log_callback: LlamaLogCallback, user_data: *mut c_void) {
    // The llama and mtmd translation units each carry their own copy of
    // `ggml_log_callback` (and its `ggml_log_level` parameter enum). Both are
    // ABI-identical: the enum is `repr(u32)` over the same C enumeration, and
    // `Option<extern "C" fn>` is FFI-safe. Adapt the facade's callback type
    // to the mtmd copy for this call.
    let adapted: mtmd::ggml_log_callback = unsafe { std::mem::transmute(log_callback) };
    unsafe { mtmd::mtmd_helper_log_set(adapted, user_data) }
}

/// # Safety
/// `chunks` must be a live mtmd input-chunks handle borrowed from the runtime.
pub unsafe fn mtmd_input_chunks_get(chunks: *const MtmdInputChunks, index: usize) -> *const Opaque {
    unsafe { mtmd::mtmd_input_chunks_get(chunks, index).cast() }
}

/// # Safety
/// `chunk` must be a live mtmd input-chunk pointer borrowed from the runtime.
pub unsafe fn mtmd_input_chunk_get_type(chunk: *const Opaque) -> MtmdInputChunkType {
    unsafe { mtmd::mtmd_input_chunk_get_type(chunk.cast()) }
}

/// # Safety
/// `chunk` must be a live mtmd input-chunk pointer borrowed from the runtime.
pub unsafe fn mtmd_input_chunk_get_n_tokens(chunk: *const Opaque) -> usize {
    unsafe { mtmd::mtmd_input_chunk_get_n_tokens(chunk.cast()) }
}

/// # Safety
/// `chunk` must be a live mtmd input-chunk pointer borrowed from the runtime.
pub unsafe fn mtmd_input_chunk_get_tokens_text(
    chunk: *const Opaque,
    out_count: *mut usize,
) -> *const i32 {
    unsafe { mtmd::mtmd_input_chunk_get_tokens_text(chunk.cast(), out_count) }
}

/// # Safety
/// `chunk` must be a live mtmd input-chunk pointer borrowed from the runtime.
pub unsafe fn mtmd_input_chunk_get_tokens_image(chunk: *const Opaque) -> *const Opaque {
    unsafe { mtmd::mtmd_input_chunk_get_tokens_image(chunk.cast()).cast() }
}

// Upstream llama.cpp entry points for external draft contexts; see
// `crate::llama_draft`. The generated declarations name llama.cpp's opaque
// types; the draft table has always used `Opaque`, so adapt at the boundary
// (ABI-identical pointer casts). The `llama_model_get_vocab` and
// `llama_vocab_n_tokens` adapters above complete the table.
pub use llama::{llama_context_default_params, llama_model_default_params};

/// # Safety
/// `path` must be a valid NUL-terminated path; `params` is consumed by value.
pub unsafe extern "C" fn llama_model_load_from_file(
    path: *const c_char,
    params: LlamaModelParams,
) -> *mut Opaque {
    unsafe { llama::llama_model_load_from_file(path, params).cast() }
}

/// # Safety
/// `model` must be a live upstream llama model owned by this runtime.
pub unsafe extern "C" fn llama_model_free(model: *mut Opaque) {
    unsafe { llama::llama_model_free(model.cast()) }
}

/// # Safety
/// `model` must be a live upstream llama model owned by this runtime.
pub unsafe extern "C" fn llama_init_from_model(
    model: *mut Opaque,
    params: LlamaContextParams,
) -> *mut Opaque {
    unsafe { llama::llama_init_from_model(model.cast(), params).cast() }
}

/// # Safety
/// `ctx` must be a live upstream llama context owned by this runtime.
pub unsafe extern "C" fn llama_free(ctx: *mut Opaque) {
    unsafe { llama::llama_free(ctx.cast()) }
}

/// # Safety
/// `ctx` must be a live upstream llama context; `batch` must satisfy the
/// upstream `llama_decode` contract.
pub unsafe extern "C" fn llama_decode(ctx: *mut Opaque, batch: LlamaBatch) -> i32 {
    unsafe { llama::llama_decode(ctx.cast(), batch) }
}

/// # Safety
/// `ctx` must be a live upstream llama context owned by this runtime.
pub unsafe extern "C" fn llama_get_model(ctx: *const Opaque) -> *const Opaque {
    unsafe { llama::llama_get_model(ctx.cast()).cast() }
}

/// # Safety
/// `ctx` must be a live upstream llama context owned by this runtime.
pub unsafe extern "C" fn llama_get_memory(ctx: *const Opaque) -> *mut Opaque {
    unsafe { llama::llama_get_memory(ctx.cast()).cast() }
}

/// # Safety
/// `mem` must be a live upstream llama memory owned by this runtime.
pub unsafe extern "C" fn llama_memory_seq_rm(mem: *mut Opaque, seq_id: i32, p0: i32, p1: i32) -> bool {
    unsafe { llama::llama_memory_seq_rm(mem.cast(), seq_id, p0, p1) }
}

/// # Safety
/// `ctx` must be a live upstream llama context owned by this runtime.
pub unsafe extern "C" fn llama_get_logits_ith(ctx: *mut Opaque, i: i32) -> *mut f32 {
    unsafe { llama::llama_get_logits_ith(ctx.cast(), i) }
}

/// # Safety
/// `ctx` must be a live upstream llama context owned by this runtime.
pub unsafe extern "C" fn llama_n_ctx(ctx: *const Opaque) -> u32 {
    unsafe { llama::llama_n_ctx(ctx.cast()) }
}

/// # Safety
/// `ctx` must be a live upstream llama context owned by this runtime.
pub unsafe extern "C" fn llama_n_batch(ctx: *const Opaque) -> u32 {
    unsafe { llama::llama_n_batch(ctx.cast()) }
}

/// # Safety
/// `ctx` must be a live upstream llama context owned by this runtime.
pub unsafe extern "C" fn llama_n_ubatch(ctx: *const Opaque) -> u32 {
    unsafe { llama::llama_n_ubatch(ctx.cast()) }
}

/// # Safety
/// `model` must be a live upstream llama model owned by this runtime.
pub unsafe extern "C" fn llama_model_n_embd(model: *const Opaque) -> i32 {
    unsafe { llama::llama_model_n_embd(model.cast()) }
}

/// # Safety
/// `model` must be a live upstream llama model owned by this runtime.
pub unsafe extern "C" fn llama_model_n_layer(model: *const Opaque) -> i32 {
    unsafe { llama::llama_model_n_layer(model.cast()) }
}

/// # Safety
/// `model` must be a live upstream llama model owned by this runtime.
pub unsafe extern "C" fn llama_model_rope_type(model: *const Opaque) -> i32 {
    unsafe { llama::llama_model_rope_type(model.cast()) as i32 }
}

/// # Safety
/// `vocab` must be a live upstream llama vocabulary owned by this runtime.
pub unsafe extern "C" fn llama_vocab_mask(vocab: *const Opaque) -> i32 {
    unsafe { llama::llama_vocab_mask(vocab.cast()) }
}

/// # Safety
/// `ctx` must be a live upstream llama context owned by this runtime.
pub unsafe extern "C" fn llama_set_causal_attn(ctx: *mut Opaque, causal_attn: bool) {
    unsafe { llama::llama_set_causal_attn(ctx.cast(), causal_attn) }
}

/// # Safety
/// `name` must be a valid NUL-terminated backend device name.
pub unsafe extern "C" fn ggml_backend_dev_by_name(name: *const c_char) -> *mut Opaque {
    unsafe { llama::ggml_backend_dev_by_name(name).cast() }
}

// `src/llama-ext.h` is a C++ staging header, so these keep C++ linkage and are
// bound by their Itanium-mangled names. The mangling encodes the parameter
// types, so an upstream signature change fails to link instead of miscalling.
#[cfg(not(windows))]
unsafe extern "C" {
    #[link_name = "_Z26llama_set_embeddings_nextnP13llama_contextbb"]
    pub fn llama_set_embeddings_nextn(ctx: *mut Opaque, value: bool, masked: bool);
    #[link_name = "_Z26llama_get_embeddings_nextnP13llama_context"]
    pub fn llama_get_embeddings_nextn(ctx: *mut Opaque) -> *mut f32;
    #[link_name = "_Z30llama_set_embeddings_layer_inpP13llama_contextjb"]
    pub fn llama_set_embeddings_layer_inp(ctx: *mut Opaque, lid: u32, value: bool);
    #[link_name = "_Z30llama_get_embeddings_layer_inpP13llama_contextj"]
    pub fn llama_get_embeddings_layer_inp(ctx: *mut Opaque, lid: u32) -> *mut f32;
    #[link_name = "_Z28llama_model_target_layer_idsPK11llama_model"]
    pub fn llama_model_target_layer_ids(model: *const Opaque) -> *const i32;
    #[link_name = "_Z30llama_model_target_layer_ids_nPK11llama_model"]
    pub fn llama_model_target_layer_ids_n(model: *const Opaque) -> u32;
    #[link_name = "_Z33llama_model_dflash_selector_top_kPK11llama_model"]
    pub fn llama_model_dflash_selector_top_k(model: *const Opaque) -> i32;
}
