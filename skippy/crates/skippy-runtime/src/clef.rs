//! Native handle for a Clef decision model (`skippy/clef.h`).
//!
//! The native side renders the `systemone` template, tokenizes the prompt
//! pieces with their decision order, and scores every option in one pass.
//! This wrapper owns the handle, its metadata, and the raw-score read.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

use anyhow::{Context, Result, anyhow};

use crate::error::ensure_ok;
use crate::logging::write_native_log_note;
use crate::path_cstring::path_to_cstring;

#[derive(Debug, Clone, PartialEq)]
pub struct ClefModelInfo {
    pub context_tokens: usize,
    pub embedding_size: usize,
    pub layer_count: usize,
    pub vocab_size: usize,
    pub parameter_count: u64,
    pub model_bytes: u64,
    pub decision_type: String,
    /// `<arch>.decision.*` metadata as `(key suffix, value)` in GGUF order.
    pub decision_metadata: Vec<(String, String)>,
}

pub struct ClefModel {
    raw: *mut skippy_ffi::ClefModel,
    info: ClefModelInfo,
}

// The native model serializes reads behind its own lock.
unsafe impl Send for ClefModel {}
unsafe impl Sync for ClefModel {}

impl ClefModel {
    /// Opens a Clef GGUF with a single-batch context of `context_tokens`.
    pub fn open(
        path: impl AsRef<Path>,
        context_tokens: usize,
        threads: usize,
        gpu_layers: Option<usize>,
    ) -> Result<Self> {
        ensure_clef_supported()?;
        let path = path.as_ref();
        write_native_log_note(format!(
            "skippy_clef_model_open begin path={} n_ctx={context_tokens}",
            path.display()
        ));
        let c_path = path_to_cstring(path, "Clef model path")?;
        let n_ctx = i32::try_from(context_tokens.max(1)).unwrap_or(i32::MAX);
        let threads = i32::try_from(threads.max(1)).unwrap_or(i32::MAX);
        let n_gpu_layers =
            gpu_layers.map_or(-1, |layers| i32::try_from(layers).unwrap_or(i32::MAX));
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_clef_model_open(
                c_path.as_ptr(),
                n_ctx,
                threads,
                n_gpu_layers,
                &mut raw,
                &mut error,
            )
        };
        ensure_ok(status, error).with_context(|| format!("open Clef model {}", path.display()))?;
        if raw.is_null() {
            return Err(anyhow!("skippy_clef_model_open returned a null handle"));
        }
        let mut model = Self {
            raw,
            info: ClefModelInfo {
                context_tokens: 0,
                embedding_size: 0,
                layer_count: 0,
                vocab_size: 0,
                parameter_count: 0,
                model_bytes: 0,
                decision_type: String::new(),
                decision_metadata: Vec::new(),
            },
        };
        model.info = model.query_info()?;
        write_native_log_note(format!(
            "skippy_clef_model_open returned layers={} n_ctx={} metadata={}",
            model.info.layer_count,
            model.info.context_tokens,
            model.info.decision_metadata.len()
        ));
        Ok(model)
    }

    pub fn info(&self) -> &ClefModelInfo {
        &self.info
    }

    fn query_info(&self) -> Result<ClefModelInfo> {
        let mut raw_info = skippy_ffi::ClefInfoV1::default();
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_clef_model_info_v1(self.raw, &mut raw_info, &mut error) };
        ensure_ok(status, error).context("query Clef model info")?;
        let decision_type = CStr::from_bytes_until_nul(&raw_info.decision_type)
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut decision_metadata = Vec::new();
        for index in 0..raw_info.decision_metadata_count {
            decision_metadata.push(self.query_metadata(index)?);
        }
        Ok(ClefModelInfo {
            context_tokens: usize::try_from(raw_info.n_ctx).unwrap_or(0),
            embedding_size: usize::try_from(raw_info.n_embd).unwrap_or(0),
            layer_count: usize::try_from(raw_info.n_layer).unwrap_or(0),
            vocab_size: usize::try_from(raw_info.n_vocab).unwrap_or(0),
            parameter_count: u64::try_from(raw_info.parameter_count).unwrap_or(0),
            model_bytes: raw_info.model_bytes,
            decision_type,
            decision_metadata,
        })
    }

    fn query_metadata(&self, index: u32) -> Result<(String, String)> {
        let mut key = vec![0 as c_char; 256];
        let mut value = vec![0 as c_char; 256];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_clef_model_decision_metadata(
                self.raw,
                index,
                key.as_mut_ptr(),
                key.len(),
                value.as_mut_ptr(),
                value.len(),
                &mut error,
            )
        };
        ensure_ok(status, error).context("query Clef decision metadata")?;
        let key = unsafe { CStr::from_ptr(key.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let value = unsafe { CStr::from_ptr(value.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Ok((key, value))
    }

    /// Plans the prompt of a request JSON: `(token ids, decision orders)`.
    pub fn plan(&self, request_json: &str) -> Result<(Vec<i32>, Vec<i32>)> {
        let mut count = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_clef_plan(
                self.raw,
                request_json.as_ptr().cast(),
                request_json.len(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut count,
                &mut error,
            )
        };
        if status != skippy_ffi::Status::BufferTooSmall {
            ensure_ok(status, error).context("plan Clef prompt")?;
            return Ok((Vec::new(), Vec::new()));
        }
        unsafe { skippy_ffi::skippy_error_free(error) };
        let mut tokens = vec![0i32; count];
        let mut orders = vec![0i32; count];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_clef_plan(
                self.raw,
                request_json.as_ptr().cast(),
                request_json.len(),
                tokens.as_mut_ptr(),
                orders.as_mut_ptr(),
                tokens.len(),
                &mut count,
                &mut error,
            )
        };
        ensure_ok(status, error).context("plan Clef prompt")?;
        tokens.truncate(count);
        orders.truncate(count);
        Ok((tokens, orders))
    }

    /// Scores `option_count` options of a request JSON in prompt order.
    pub fn read(&self, request_json: &str, option_count: usize) -> Result<Vec<f32>> {
        let mut scores = vec![0f32; option_count];
        let mut count = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_clef_read(
                self.raw,
                request_json.as_ptr().cast(),
                request_json.len(),
                scores.as_mut_ptr(),
                scores.len(),
                &mut count,
                &mut error,
            )
        };
        ensure_ok(status, error).context("read Clef decision")?;
        if count != option_count {
            return Err(anyhow!(
                "Clef read returned {count} scores for {option_count} options"
            ));
        }
        Ok(scores)
    }
}

impl Drop for ClefModel {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { skippy_ffi::skippy_clef_model_free(self.raw) };
        }
    }
}

fn ensure_clef_supported() -> Result<()> {
    if skippy_ffi::try_abi_features()
        .is_none_or(|features| features & skippy_ffi::FEATURE_CLEF_DECISIONS == 0)
    {
        return Err(anyhow!(
            "native runtime does not support Clef decision reads"
        ));
    }
    Ok(())
}
