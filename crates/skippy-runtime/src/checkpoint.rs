use std::ffi::{CStr, CString, c_char, c_void};
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow};
use sha2::{Digest, Sha256};
use skippy_ffi::{Error, Model as RawModel, ModelImatrixEntryV1, ModelTensorSourceV1, Status};
use skippy_model::gguf_writer::DirectCheckpoint;
use skippy_model::imatrix::Imatrix;

use crate::RuntimeConfig;
use crate::error::ensure_ok;

pub(crate) fn checkpoint_root(source: &Path) -> &Path {
    if source.is_dir() {
        source
    } else {
        source.parent().unwrap_or(source)
    }
}

/// Return whether `source` belongs to a complete local SafeTensors checkpoint.
pub fn is_safetensors_checkpoint(source: &Path) -> bool {
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
    Q1_0,
    Q2_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    IQ2XXS,
    IQ2XS,
    IQ2S,
    IQ2M,
    IQ1S,
    IQ1M,
    TQ1_0,
    TQ2_0,
    Q2K,
    Q2KS,
    IQ3XS,
    IQ3XXS,
    IQ3S,
    IQ3M,
    Q3KS,
    Q3KM,
    Q3KL,
    IQ4NL,
    IQ4XS,
    Q4KS,
    Q4KM,
    Q5KS,
    Q5KM,
    Q6K,
    Q8_0,
    Mxfp4Moe,
}

impl CheckpointQuantization {
    pub const VALID_NAMES: &'static [&'static str] = &[
        "preserve",
        "F32",
        "F16",
        "BF16",
        "Q1_0",
        "Q2_0",
        "Q4_0",
        "Q4_1",
        "Q5_0",
        "Q5_1",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ2_M",
        "IQ1_S",
        "IQ1_M",
        "TQ1_0",
        "TQ2_0",
        "Q2_K",
        "Q2_K_S",
        "IQ3_XS",
        "IQ3_XXS",
        "IQ3_S",
        "IQ3_M",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "IQ4_NL",
        "IQ4_XS",
        "Q4_K_S",
        "Q4_K_M",
        "Q5_K_S",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
        "MXFP4_MOE",
    ];

    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Preserve => "preserve",
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::Q1_0 => "Q1_0",
            Self::Q2_0 => "Q2_0",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ2S => "IQ2_S",
            Self::IQ2M => "IQ2_M",
            Self::IQ1S => "IQ1_S",
            Self::IQ1M => "IQ1_M",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::Q2K => "Q2_K",
            Self::Q2KS => "Q2_K_S",
            Self::IQ3XS => "IQ3_XS",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ3S => "IQ3_S",
            Self::IQ3M => "IQ3_M",
            Self::Q3KS => "Q3_K_S",
            Self::Q3KM => "Q3_K_M",
            Self::Q3KL => "Q3_K_L",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ4XS => "IQ4_XS",
            Self::Q4KS => "Q4_K_S",
            Self::Q4KM => "Q4_K_M",
            Self::Q5KS => "Q5_K_S",
            Self::Q5KM => "Q5_K_M",
            Self::Q6K => "Q6_K",
            Self::Q8_0 => "Q8_0",
            Self::Mxfp4Moe => "MXFP4_MOE",
        }
    }

    fn llama_ftype(self) -> i32 {
        match self {
            Self::Preserve => -1,
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q8_0 => 7,
            Self::Q5_0 => 8,
            Self::Q5_1 => 9,
            Self::Q2K => 10,
            Self::Q3KS => 11,
            Self::Q3KM => 12,
            Self::Q3KL => 13,
            Self::Q4KS => 14,
            Self::Q4KM => 15,
            Self::Q5KS => 16,
            Self::Q5KM => 17,
            Self::Q6K => 18,
            Self::IQ2XXS => 19,
            Self::IQ2XS => 20,
            Self::Q2KS => 21,
            Self::IQ3XS => 22,
            Self::IQ3XXS => 23,
            Self::IQ1S => 24,
            Self::IQ4NL => 25,
            Self::IQ3S => 26,
            Self::IQ3M => 27,
            Self::IQ2S => 28,
            Self::IQ2M => 29,
            Self::IQ4XS => 30,
            Self::IQ1M => 31,
            Self::Bf16 => 32,
            Self::TQ1_0 => 36,
            Self::TQ2_0 => 37,
            Self::Mxfp4Moe => 38,
            Self::Q1_0 => 40,
            Self::Q2_0 => 41,
        }
    }

    pub fn requires_imatrix(self) -> bool {
        matches!(
            self,
            Self::IQ1S
                | Self::IQ1M
                | Self::IQ2XXS
                | Self::IQ2XS
                | Self::IQ2S
                | Self::IQ2M
                | Self::IQ3XXS
                | Self::Q2KS
        )
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
            "PRESERVE" | "DIRECT" | "NONE" | "COPY" => Ok(Self::Preserve),
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::Bf16),
            "Q10" => Ok(Self::Q1_0),
            "Q20" => Ok(Self::Q2_0),
            "Q40" => Ok(Self::Q4_0),
            "Q41" => Ok(Self::Q4_1),
            "Q50" => Ok(Self::Q5_0),
            "Q51" => Ok(Self::Q5_1),
            "IQ2XXS" => Ok(Self::IQ2XXS),
            "IQ2XS" => Ok(Self::IQ2XS),
            "IQ2S" => Ok(Self::IQ2S),
            "IQ2M" => Ok(Self::IQ2M),
            "IQ1S" => Ok(Self::IQ1S),
            "IQ1M" => Ok(Self::IQ1M),
            "TQ10" => Ok(Self::TQ1_0),
            "TQ20" => Ok(Self::TQ2_0),
            "Q2K" => Ok(Self::Q2K),
            "Q2KS" => Ok(Self::Q2KS),
            "IQ3XS" => Ok(Self::IQ3XS),
            "IQ3XXS" => Ok(Self::IQ3XXS),
            "IQ3S" => Ok(Self::IQ3S),
            "IQ3M" => Ok(Self::IQ3M),
            "Q3KS" => Ok(Self::Q3KS),
            "Q3K" | "Q3KM" => Ok(Self::Q3KM),
            "Q3KL" => Ok(Self::Q3KL),
            "IQ4NL" => Ok(Self::IQ4NL),
            "IQ4XS" => Ok(Self::IQ4XS),
            "Q4KS" => Ok(Self::Q4KS),
            "Q4K" | "Q4KM" => Ok(Self::Q4KM),
            "Q5KS" => Ok(Self::Q5KS),
            "Q5K" | "Q5KM" => Ok(Self::Q5KM),
            "Q6K" => Ok(Self::Q6K),
            "Q80" => Ok(Self::Q8_0),
            "MXFP4MOE" => Ok(Self::Mxfp4Moe),
            _ => Err(format!(
                "unsupported direct checkpoint quantization {value:?}; valid values: {}",
                Self::VALID_NAMES.join(", ")
            )),
        }
    }
}

struct CallbackState {
    checkpoint: DirectCheckpoint,
    last_error: Option<CString>,
}

struct NativeImatrix {
    imatrix: Imatrix,
    names: Vec<CString>,
    entries: Vec<ModelImatrixEntryV1>,
}

impl NativeImatrix {
    fn load(path: &Path, expected_sha256: Option<&str>) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("read checkpoint importance matrix {}", path.display()))?;
        if let Some(expected_sha256) = expected_sha256 {
            let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
            if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
                return Err(anyhow!(
                    "checkpoint importance matrix SHA-256 mismatch for {}: expected {expected_sha256}, got {actual_sha256}",
                    path.display()
                ));
            }
        }
        let imatrix = Imatrix::from_bytes(path, &bytes, &[], &[])?;
        let names = imatrix
            .entries()
            .iter()
            .map(|entry| CString::new(entry.name.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let entries = names
            .iter()
            .zip(imatrix.entries())
            .map(|(name, entry)| ModelImatrixEntryV1 {
                tensor_name: name.as_ptr(),
                values: entry.values.as_ptr(),
                value_count: entry.values.len(),
            })
            .collect();
        Ok(Self {
            imatrix,
            names,
            entries,
        })
    }

    fn as_slice(&self) -> &[ModelImatrixEntryV1] {
        debug_assert_eq!(self.imatrix.entry_count(), self.names.len());
        &self.entries
    }
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
    let imatrix_path = config.checkpoint_imatrix.as_deref().map(|path| {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            source.join(path)
        }
    });
    let imatrix = imatrix_path
        .as_deref()
        .map(|path| NativeImatrix::load(path, config.checkpoint_imatrix_sha256.as_deref()))
        .transpose()
        .context("load checkpoint importance matrix")?;
    if quantization.requires_imatrix() && imatrix.is_none() {
        return Err(anyhow!(
            "checkpoint quantization {quantization:?} requires hardware.checkpoint_imatrix"
        ));
    }
    let imatrix_entries = imatrix.as_ref().map_or(&[][..], NativeImatrix::as_slice);
    let imatrix_ptr = if imatrix_entries.is_empty() {
        ptr::null()
    } else {
        imatrix_entries.as_ptr()
    };
    let callback_source = ModelTensorSourceV1 {
        abi_version: skippy_ffi::MODEL_TENSOR_SOURCE_V1_ABI_VERSION,
        struct_size: u32::try_from(std::mem::size_of::<ModelTensorSourceV1>())
            .context("model tensor source ABI struct size exceeds u32")?,
        read_tensor_f32: Some(read_tensor_f32),
        user_data: (&mut callback_state as *mut CallbackState).cast(),
        imatrix: imatrix_ptr,
        imatrix_count: imatrix_entries.len(),
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
    use std::io::Write;

    fn write_minimal_legacy_imatrix(file: &mut tempfile::NamedTempFile) {
        let name = "blk.0.attn_q.weight";
        file.write_all(&1_i32.to_le_bytes()).unwrap();
        file.write_all(&(name.len() as i32).to_le_bytes()).unwrap();
        file.write_all(name.as_bytes()).unwrap();
        file.write_all(&1_i32.to_le_bytes()).unwrap();
        file.write_all(&1_i32.to_le_bytes()).unwrap();
        file.write_all(&1.0_f32.to_le_bytes()).unwrap();
    }

    #[test]
    fn canonical_quantization_catalog_round_trips() {
        assert_eq!(CheckpointQuantization::VALID_NAMES.len(), 36);
        for name in CheckpointQuantization::VALID_NAMES {
            let parsed = name
                .parse::<CheckpointQuantization>()
                .unwrap_or_else(|error| panic!("parse {name}: {error}"));
            assert_eq!(parsed.canonical_name(), *name);
        }
    }

    #[test]
    fn parses_documented_quantization_names() {
        assert_eq!("bf16".parse(), Ok(CheckpointQuantization::Bf16));
        assert_eq!("Q1_0".parse(), Ok(CheckpointQuantization::Q1_0));
        assert_eq!("Q2_0".parse(), Ok(CheckpointQuantization::Q2_0));
        assert_eq!("Q4_1".parse(), Ok(CheckpointQuantization::Q4_1));
        assert_eq!("q5-0".parse(), Ok(CheckpointQuantization::Q5_0));
        assert_eq!("Q5_1".parse(), Ok(CheckpointQuantization::Q5_1));
        assert_eq!("Q2_K".parse(), Ok(CheckpointQuantization::Q2K));
        assert_eq!("Q2_K_S".parse(), Ok(CheckpointQuantization::Q2KS));
        assert_eq!("IQ1_M".parse(), Ok(CheckpointQuantization::IQ1M));
        assert_eq!("IQ2_XXS".parse(), Ok(CheckpointQuantization::IQ2XXS));
        assert_eq!("IQ3_XS".parse(), Ok(CheckpointQuantization::IQ3XS));
        assert_eq!("IQ4_NL".parse(), Ok(CheckpointQuantization::IQ4NL));
        assert_eq!("TQ1_0".parse(), Ok(CheckpointQuantization::TQ1_0));
        assert_eq!("Q3_K_S".parse(), Ok(CheckpointQuantization::Q3KS));
        assert_eq!("Q3_K_M".parse(), Ok(CheckpointQuantization::Q3KM));
        assert_eq!("Q3_K".parse(), Ok(CheckpointQuantization::Q3KM));
        assert_eq!("Q3_K_L".parse(), Ok(CheckpointQuantization::Q3KL));
        assert_eq!("Q4_K_M".parse(), Ok(CheckpointQuantization::Q4KM));
        assert_eq!("Q4_K".parse(), Ok(CheckpointQuantization::Q4KM));
        assert_eq!("Q5_K_S".parse(), Ok(CheckpointQuantization::Q5KS));
        assert_eq!("Q5_K".parse(), Ok(CheckpointQuantization::Q5KM));
        assert_eq!("q8-0".parse(), Ok(CheckpointQuantization::Q8_0));
        assert_eq!("MXFP4_MOE".parse(), Ok(CheckpointQuantization::Mxfp4Moe));
        assert_eq!("COPY".parse(), Ok(CheckpointQuantization::Preserve));
        assert!("Q4_2".parse::<CheckpointQuantization>().is_err());
        assert!(CheckpointQuantization::IQ2XXS.requires_imatrix());
        assert!(CheckpointQuantization::IQ3XXS.requires_imatrix());
        assert!(!CheckpointQuantization::IQ3XS.requires_imatrix());
    }

    #[test]
    fn native_imatrix_verifies_the_bytes_it_parses() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write_minimal_legacy_imatrix(&mut file);
        let bytes = fs::read(file.path()).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));

        let loaded = NativeImatrix::load(file.path(), Some(&digest)).unwrap();

        assert_eq!(loaded.imatrix.entry_count(), 1);
    }

    #[test]
    fn native_imatrix_rejects_bytes_that_do_not_match_identity() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write_minimal_legacy_imatrix(&mut file);

        let error = NativeImatrix::load(file.path(), Some(&"0".repeat(64)))
            .err()
            .expect("mismatched digest must be rejected");

        assert!(error.to_string().contains("SHA-256 mismatch"));
    }
}
