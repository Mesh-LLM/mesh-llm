use std::ffi::CStr;
use std::path::Path;
use std::ptr;

use anyhow::{Result, anyhow};
use skippy_ffi::{ModelInfo as RawModelInfo, TensorInfo as RawTensorInfo, TensorRole};

use crate::TensorInfo;
use crate::error::ensure_ok;
use crate::path_cstring::path_to_cstring;

pub struct ModelInfo {
    raw: *mut RawModelInfo,
}

impl ModelInfo {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let path = path_to_cstring(path, "model path")?;
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_model_info_open(path.as_ptr(), &mut raw, &mut error) };
        ensure_ok(status, error)?;
        if raw.is_null() {
            return Err(anyhow!("skippy_model_info_open returned a null handle"));
        }
        Ok(Self { raw })
    }

    pub fn tensor_count(&self) -> Result<usize> {
        let mut count = 0usize;
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_model_info_tensor_count(self.raw, &mut count, &mut error) };
        ensure_ok(status, error)?;
        Ok(count)
    }

    pub fn tensor_at(&self, index: usize) -> Result<TensorInfo> {
        let mut raw = RawTensorInfo {
            name: ptr::null(),
            layer_index: -1,
            role: TensorRole::Unknown,
            ggml_type: 0,
            byte_size: 0,
            element_count: 0,
        };
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_model_info_tensor_at(self.raw, index, &mut raw, &mut error)
        };
        ensure_ok(status, error)?;

        let name = if raw.name.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(raw.name) }
                .to_string_lossy()
                .into_owned()
        };

        Ok(TensorInfo {
            name,
            layer_index: u32::try_from(raw.layer_index).ok(),
            role: raw.role,
            ggml_type: raw.ggml_type,
            byte_size: raw.byte_size,
            element_count: raw.element_count,
        })
    }

    pub fn tensors(&self) -> Result<Vec<TensorInfo>> {
        let count = self.tensor_count()?;
        (0..count).map(|index| self.tensor_at(index)).collect()
    }
}

impl Drop for ModelInfo {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = skippy_ffi::skippy_model_info_free(self.raw, ptr::null_mut());
            }
        }
    }
}

pub fn write_gguf_from_parts(
    input_paths: &[impl AsRef<Path>],
    output_path: impl AsRef<Path>,
) -> Result<()> {
    write_gguf_from_parts_impl(input_paths, output_path, false)
}

/// Materialize GGUF parts into one file, unlinking each input part as soon as
/// its tensors have been absorbed into the output.
///
/// Use this only when the inputs are scratch files owned by the caller. The
/// per-artifact staging peak drops from parts-plus-output to roughly one
/// output file, which keeps sharded splits inside ephemeral-storage budgets
/// such as the HF Jobs 50G container limit. Bench materialization reads
/// published package files and must keep using [`write_gguf_from_parts`].
pub fn write_gguf_from_parts_consuming(
    input_paths: &[impl AsRef<Path>],
    output_path: impl AsRef<Path>,
) -> Result<()> {
    write_gguf_from_parts_impl(input_paths, output_path, true)
}

fn write_gguf_from_parts_impl(
    input_paths: &[impl AsRef<Path>],
    output_path: impl AsRef<Path>,
    consume_inputs: bool,
) -> Result<()> {
    if input_paths.is_empty() {
        return Err(anyhow!("at least one GGUF part path is required"));
    }

    let input_paths = input_paths
        .iter()
        .map(|path| path_to_cstring(path.as_ref(), "input path"))
        .collect::<Result<Vec<_>>>()?;
    let input_ptrs = input_paths
        .iter()
        .map(|path| path.as_ptr())
        .collect::<Vec<_>>();
    let output_path = path_to_cstring(output_path.as_ref(), "output path")?;
    let mut error = ptr::null_mut();
    let status = unsafe {
        if consume_inputs {
            skippy_ffi::skippy_write_gguf_from_parts_consuming(
                input_ptrs.as_ptr(),
                input_ptrs.len(),
                output_path.as_ptr(),
                &mut error,
            )
        } else {
            skippy_ffi::skippy_write_gguf_from_parts(
                input_ptrs.as_ptr(),
                input_ptrs.len(),
                output_path.as_ptr(),
                &mut error,
            )
        }
    };
    ensure_ok(status, error)
}

pub fn write_gguf_metadata_from_parts(
    input_paths: &[impl AsRef<Path>],
    output_path: impl AsRef<Path>,
) -> Result<()> {
    if input_paths.is_empty() {
        return Err(anyhow!(
            "at least one GGUF metadata source path is required"
        ));
    }

    let input_paths = input_paths
        .iter()
        .map(|path| path_to_cstring(path.as_ref(), "input path"))
        .collect::<Result<Vec<_>>>()?;
    let input_ptrs = input_paths
        .iter()
        .map(|path| path.as_ptr())
        .collect::<Vec<_>>();
    let output_path = path_to_cstring(output_path.as_ref(), "output path")?;
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_write_gguf_metadata_from_parts(
            input_ptrs.as_ptr(),
            input_ptrs.len(),
            output_path.as_ptr(),
            &mut error,
        )
    };
    ensure_ok(status, error)
}
