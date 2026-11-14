use std::{
    ffi::{CStr, CString, c_char},
    ptr::NonNull,
};

#[cfg(feature = "dynamic-runtime")]
use crate::dynamic;
#[cfg(not(feature = "dynamic-runtime"))]
use crate::static_bindings;

use crate::Opaque;
#[cfg(not(feature = "dynamic-runtime"))]
use crate::{LlamaPerfContextFn, SkippyDecodeStepSampledMtpFn, SkippyModelAttachMtpDraftModelFn};

#[cfg(not(feature = "dynamic-runtime"))]
/// Mark the statically linked native runtime as already available.
pub fn native_runtime_loaded() -> bool {
    true
}

#[cfg(not(feature = "dynamic-runtime"))]
/// No-op for statically linked builds.
///
/// # Safety
///
/// Static builds resolve the native ABI at process link/load time, so this
/// function does not dereference the supplied path or mutate loader state.
pub unsafe fn load_native_runtime_library(
    _path: impl AsRef<std::path::Path>,
) -> Result<(), NativeRuntimeLoadError> {
    Ok(())
}

#[cfg(not(feature = "dynamic-runtime"))]
/// No-op for statically linked builds.
///
/// # Safety
///
/// Static builds resolve the native ABI at process link/load time, so this
/// function does not dereference the supplied paths or mutate loader state.
pub unsafe fn load_native_runtime_libraries<I, P>(_paths: I) -> Result<(), NativeRuntimeLoadError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    Ok(())
}

#[derive(Debug)]
pub enum NativeRuntimeLoadError {
    Load(String),
    AlreadyLoaded,
}

/// A llama.cpp LoRA adapter loaded for one model. The adapter must be applied
/// only to contexts created from that model and must outlive every context
/// using it.
pub struct LlamaLoraAdapter {
    raw: NonNull<Opaque>,
    free: unsafe extern "C" fn(*mut Opaque),
    set: unsafe extern "C" fn(*mut Opaque, *mut *mut Opaque, usize, *mut f32) -> i32,
}

impl LlamaLoraAdapter {
    /// Loads a GGUF LoRA adapter for `model`.
    ///
    /// # Safety
    ///
    /// `model` must be a live llama.cpp model from the currently loaded runtime.
    pub unsafe fn load(model: *const Opaque, path: &CStr) -> Result<Self, String> {
        if model.is_null() {
            return Err("cannot load a LoRA adapter for a null model".into());
        }
        #[cfg(feature = "dynamic-runtime")]
        let (init, free, set) = dynamic::llama_lora_adapter_api_optional().ok_or_else(|| {
            "loaded llama.cpp runtime does not export the LoRA adapter API".to_string()
        })?;
        #[cfg(not(feature = "dynamic-runtime"))]
        let (init, free, set) = (
            static_bindings::llama_adapter_lora_init,
            static_bindings::llama_adapter_lora_free,
            static_bindings::llama_set_adapters_lora,
        );

        // SAFETY: the caller guarantees the model belongs to this runtime;
        // `path` is NUL-terminated and remains valid for the duration of init.
        let raw = NonNull::new(unsafe { init(model, path.as_ptr()) }).ok_or_else(|| {
            format!(
                "llama.cpp could not load LoRA adapter {}",
                path.to_string_lossy()
            )
        })?;
        Ok(Self { raw, free, set })
    }

    /// Applies this adapter to one llama.cpp context at the requested scale.
    ///
    /// # Errors
    ///
    /// Returns an error when llama.cpp rejects the adapter for this context.
    ///
    /// # Safety
    ///
    /// `context` must be a live llama.cpp context created from the model used
    /// to load this adapter, and must not be concurrently mutated.
    pub unsafe fn apply(&self, context: *mut Opaque, scale: f32) -> Result<(), String> {
        if context.is_null() {
            return Err("cannot apply a LoRA adapter to a null context".into());
        }
        if !scale.is_finite() {
            return Err("LoRA adapter scale must be finite".into());
        }
        let mut adapter = self.raw.as_ptr();
        let mut scale = scale;
        // SAFETY: the caller guarantees context/model identity and exclusive
        // context access; the adapter remains owned by `self`.
        let status = unsafe { (self.set)(context, &mut adapter, 1, &mut scale) };
        if status != 0 {
            return Err(format!(
                "llama.cpp rejected the LoRA adapter (status {status})"
            ));
        }
        Ok(())
    }
}

impl Drop for LlamaLoraAdapter {
    fn drop(&mut self) {
        // SAFETY: `raw` is the unique adapter handle returned by init and is
        // freed once, before its owning model is released by StageModelInner.
        unsafe { (self.free)(self.raw.as_ptr()) };
    }
}

impl std::fmt::Display for NativeRuntimeLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(message) => write!(f, "{message}"),
            Self::AlreadyLoaded => write!(f, "native runtime library is already loaded"),
        }
    }
}

impl std::error::Error for NativeRuntimeLoadError {}

#[cfg(feature = "dynamic-runtime")]
/// Returns the skippy ABI feature bitmask.
/// Requires the native runtime to be loaded first (checked by caller).
pub fn skippy_abi_features() -> u64 {
    try_abi_features().expect("skippy_abi_features not available in loaded runtime")
}

/// Returns the Skippy ABI feature bitmask when the loaded dynamic runtime
/// exports feature probing.
#[cfg(feature = "dynamic-runtime")]
pub fn try_abi_features() -> Option<u64> {
    dynamic::skippy_abi_features_optional().map(|features| unsafe { features() })
}

/// Returns the active Skippy ABI feature bitmask through a safe Rust wrapper.
#[cfg(feature = "dynamic-runtime")]
pub fn abi_features() -> u64 {
    skippy_abi_features()
}

/// Reports whether the loaded model uses recurrent state when the native
/// runtime exposes the corresponding llama.cpp capability probe.
///
/// `None` is the fail-closed result for an older compatible runtime that does
/// not export this optional upstream symbol.
///
/// # Safety
///
/// `model` must be a valid llama.cpp model pointer owned by the loaded runtime.
pub unsafe fn llama_model_is_recurrent(model: *const Opaque) -> Option<bool> {
    #[cfg(feature = "dynamic-runtime")]
    {
        dynamic::llama_model_is_recurrent_fn().map(|probe| unsafe { probe(model) })
    }
    #[cfg(not(feature = "dynamic-runtime"))]
    {
        Some(unsafe { static_bindings::llama_model_is_recurrent(model) })
    }
}

/// Reports whether the loaded model uses hybrid state when the native runtime
/// exposes the corresponding llama.cpp capability probe.
///
/// `None` is the fail-closed result for an older compatible runtime that does
/// not export this optional upstream symbol.
///
/// # Safety
///
/// `model` must be a valid llama.cpp model pointer owned by the loaded runtime.
pub unsafe fn llama_model_is_hybrid(model: *const Opaque) -> Option<bool> {
    #[cfg(feature = "dynamic-runtime")]
    {
        dynamic::llama_model_is_hybrid_fn().map(|probe| unsafe { probe(model) })
    }
    #[cfg(not(feature = "dynamic-runtime"))]
    {
        Some(unsafe { static_bindings::llama_model_is_hybrid(model) })
    }
}

/// Reads a string-valued GGUF metadata entry from the loaded model when the
/// native runtime exposes the corresponding upstream accessor.
///
/// Returns `None` for an older compatible runtime that does not export this
/// optional upstream symbol or for a missing key, matching the upstream
/// `llama_model_meta_val_str` contract of returning -1 and writing `'\0'`.
///
/// # Safety
///
/// `model` must be a valid llama.cpp model pointer owned by the loaded runtime.
pub unsafe fn llama_model_meta_val_str(model: *const Opaque, key: &str) -> Option<String> {
    let key = CString::new(key).ok()?;
    // Upstream truncates at buf_size and always NUL-terminates, so a truncated
    // read reports the required length; retry once with that exact size.
    let mut buf = vec![0u8; 128];
    loop {
        #[cfg(feature = "dynamic-runtime")]
        let written = {
            let probe = dynamic::llama_model_meta_val_str_fn()?;
            unsafe {
                probe(
                    model,
                    key.as_ptr(),
                    buf.as_mut_ptr().cast::<c_char>(),
                    buf.len(),
                )
            }
        };
        #[cfg(not(feature = "dynamic-runtime"))]
        let written = unsafe {
            static_bindings::llama_model_meta_val_str(
                model,
                key.as_ptr(),
                buf.as_mut_ptr().cast::<c_char>(),
                buf.len(),
            )
        };
        if written < 0 {
            return None;
        }
        let written = usize::try_from(written).expect("upstream returns a non-negative length");
        if written < buf.len() {
            break;
        }
        buf.resize(written + 1, 0);
    }
    let end = buf.iter().position(|&byte| byte == 0).unwrap_or(buf.len());
    Some(String::from_utf8_lossy(&buf[..end]).into_owned())
}

/// Reports whether the loaded model is diffusion-based when the native runtime
/// exposes the corresponding llama.cpp capability probe.
///
/// `None` is the fail-closed result for an older compatible runtime that does
/// not export this optional upstream symbol.
///
/// # Safety
///
/// `model` must be a valid llama.cpp model pointer owned by the loaded runtime.
pub unsafe fn llama_model_is_diffusion(model: *const Opaque) -> Option<bool> {
    #[cfg(feature = "dynamic-runtime")]
    {
        dynamic::llama_model_is_diffusion_fn().map(|probe| unsafe { probe(model) })
    }
    #[cfg(not(feature = "dynamic-runtime"))]
    {
        Some(unsafe { static_bindings::llama_model_is_diffusion(model) })
    }
}

/// Returns the statically linked Skippy ABI feature bitmask.
#[cfg(not(feature = "dynamic-runtime"))]
pub fn try_abi_features() -> Option<u64> {
    // SAFETY: the statically linked ABI exposes this nullary query with no
    // caller-owned pointers or lifetime requirements.
    Some(unsafe { static_bindings::skippy_abi_features() })
}

/// Returns the statically linked Skippy ABI feature bitmask.
#[cfg(not(feature = "dynamic-runtime"))]
pub fn abi_features() -> u64 {
    // SAFETY: the statically linked ABI exposes this nullary query with no
    // caller-owned pointers or lifetime requirements.
    unsafe { static_bindings::skippy_abi_features() }
}

#[cfg(not(feature = "dynamic-runtime"))]
pub fn skippy_model_attach_mtp_draft_model_fn() -> Option<SkippyModelAttachMtpDraftModelFn> {
    Some(static_bindings::skippy_model_attach_mtp_draft_model)
}

#[cfg(not(feature = "dynamic-runtime"))]
pub fn skippy_decode_step_sampled_mtp_fn() -> Option<SkippyDecodeStepSampledMtpFn> {
    Some(static_bindings::skippy_decode_step_sampled_mtp)
}

/// Graph reuse counters for a statically linked native runtime.
///
/// There is nothing to probe here: a static runtime either exports the symbol
/// at link time or the build fails, so the counters are always available.
#[cfg(not(feature = "dynamic-runtime"))]
pub fn llama_perf_context_optional() -> Option<LlamaPerfContextFn> {
    Some(static_bindings::llama_perf_context)
}
