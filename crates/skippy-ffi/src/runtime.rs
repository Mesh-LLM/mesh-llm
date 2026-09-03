#[cfg(feature = "dynamic-runtime")]
use crate::dynamic;
#[cfg(not(feature = "dynamic-runtime"))]
use crate::static_bindings;

use crate::Opaque;
#[cfg(not(feature = "dynamic-runtime"))]
use crate::{SkippyDecodeStepSampledMtpFn, SkippyModelAttachMtpDraftModelFn};

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
