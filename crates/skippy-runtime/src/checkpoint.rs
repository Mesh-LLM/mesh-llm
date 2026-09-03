use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow};
use skippy_ffi::{Error, Model as RawModel, ModelTensorSourceV1, Status};
use skippy_model::gguf_writer::DirectCheckpoint;

use crate::RuntimeConfig;
use crate::error::ensure_ok;

pub(crate) fn checkpoint_root(source: &Path) -> &Path {
    if source.is_dir() {
        source
    } else {
        source.parent().unwrap_or(source)
    }
}

pub(crate) fn is_safetensors_checkpoint(source: &Path) -> bool {
    let root = checkpoint_root(source);
    root.join("config.json").is_file()
        && (root.join("model.safetensors").is_file()
            || root.join("model.safetensors.index.json").is_file())
}

/// Quantization policy applied tensor-by-tensor while a SafeTensors checkpoint opens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CheckpointQuantization {
    /// Preserve the checkpoint's canonical F32/F16/BF16 tensor types.
    #[default]
    Preserve,
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4KS,
    Q4KM,
    Q5KM,
    Q6K,
    Q8_0,
}

impl CheckpointQuantization {
    fn llama_ftype(self) -> i32 {
        match self {
            Self::Preserve => -1,
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 7,
            Self::Q4KS => 14,
            Self::Q4KM => 15,
            Self::Q5KM => 17,
            Self::Q6K => 18,
            Self::Bf16 => 32,
        }
    }
}

impl FromStr for CheckpointQuantization {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let normalized = value
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .flat_map(char::to_uppercase)
            .collect::<String>();
        match normalized.as_str() {
            "PRESERVE" | "DIRECT" | "NONE" => Ok(Self::Preserve),
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::Bf16),
            "Q40" => Ok(Self::Q4_0),
            "Q4KS" => Ok(Self::Q4KS),
            "Q4KM" => Ok(Self::Q4KM),
            "Q5KM" => Ok(Self::Q5KM),
            "Q6K" => Ok(Self::Q6K),
            "Q80" => Ok(Self::Q8_0),
            _ => Err(format!(
                "unsupported direct checkpoint quantization {value:?}; expected preserve, F32, F16, BF16, Q4_0, Q4_K_S, Q4_K_M, Q5_K_M, Q6_K, or Q8_0"
            )),
        }
    }
}

struct CallbackState {
    checkpoint: DirectCheckpoint,
    last_error: Option<CString>,
}

impl CallbackState {
    fn set_error(&mut self, message: impl AsRef<str>, out_message: *mut *const c_char) {
        let sanitized = message.as_ref().replace('\0', "\\0");
        self.last_error = CString::new(sanitized).ok();
        if !out_message.is_null() {
            // SAFETY: native code supplies a writable pointer for the duration of the callback.
            unsafe {
                *out_message = self
                    .last_error
                    .as_ref()
                    .map(|message| message.as_ptr())
                    .unwrap_or(ptr::null());
            }
        }
    }
}

unsafe extern "C" fn read_tensor_f32(
    tensor_name: *const c_char,
    destination: *mut f32,
    element_count: usize,
    out_message: *mut *const c_char,
    user_data: *mut c_void,
) -> Status {
    if tensor_name.is_null() || destination.is_null() || user_data.is_null() {
        return Status::InvalidArgument;
    }
    // SAFETY: all pointers are validated above. The native loader invokes this callback
    // synchronously, keeps the name alive, and allocates `element_count` writable F32 values.
    let state = unsafe { &mut *user_data.cast::<CallbackState>() };
    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `tensor_name` is a NUL-terminated llama.cpp tensor name for this callback.
        let name = unsafe { CStr::from_ptr(tensor_name) }
            .to_str()
            .context("native tensor name is not UTF-8")?;
        // SAFETY: the native callback contract guarantees this exact writable extent.
        let destination = unsafe { std::slice::from_raw_parts_mut(destination, element_count) };
        state.checkpoint.read_tensor_f32(name, destination)
    }));
    match result {
        Ok(Ok(())) => Status::Ok,
        Ok(Err(error)) => {
            state.set_error(format!("{error:#}"), out_message);
            Status::ModelError
        }
        Err(_) => {
            state.set_error("panic while decoding checkpoint tensor", out_message);
            Status::RuntimeError
        }
    }
}

pub(crate) fn open_safetensors(
    source: &Path,
    quantization: CheckpointQuantization,
    config: &RuntimeConfig,
) -> Result<*mut RawModel> {
    let source = checkpoint_root(source);
    let checkpoint = DirectCheckpoint::open(source, 8 * 1024 * 1024)
        .with_context(|| format!("open SafeTensors checkpoint {}", source.display()))?;
    let mut callback_state = CallbackState {
        checkpoint,
        last_error: None,
    };
    let callback_source = ModelTensorSourceV1 {
        abi_version: skippy_ffi::MODEL_TENSOR_SOURCE_V1_ABI_VERSION,
        struct_size: u32::try_from(std::mem::size_of::<ModelTensorSourceV1>())
            .context("model tensor source ABI struct size exceeds u32")?,
        read_tensor_f32: Some(read_tensor_f32),
        user_data: (&mut callback_state as *mut CallbackState).cast(),
    };
    let raw_config = config.as_raw()?;
    let mut raw = ptr::null_mut();
    let mut error: *mut Error = ptr::null_mut();
    // SAFETY: metadata and callback state remain alive until the synchronous native open returns;
    // output pointers are valid and the configuration owns its backing C strings.
    let status = unsafe {
        skippy_ffi::skippy_model_open_from_source(
            callback_state.checkpoint.metadata_gguf().as_ptr().cast(),
            callback_state.checkpoint.metadata_gguf().len(),
            &callback_source,
            quantization.llama_ftype(),
            &raw_config.raw,
            &mut raw,
            &mut error,
        )
    };
    ensure_ok(status, error)?;
    if raw.is_null() {
        return Err(anyhow!(
            "skippy_model_open_from_source returned a null handle"
        ));
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_documented_quantization_names() {
        assert_eq!("bf16".parse(), Ok(CheckpointQuantization::Bf16));
        assert_eq!("Q4_K_M".parse(), Ok(CheckpointQuantization::Q4KM));
        assert_eq!("q8-0".parse(), Ok(CheckpointQuantization::Q8_0));
        assert!("IQ2_XXS".parse::<CheckpointQuantization>().is_err());
    }
}
