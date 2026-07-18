use std::ffi::CString;
use std::ptr;

use anyhow::{Context, Result, anyhow};
use skippy_ffi::{LoadMode, RuntimeConfig as RawRuntimeConfig};

pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const LLAMA_SERVER_DEFAULT_N_BATCH: u32 = 2048;
pub const LLAMA_SERVER_DEFAULT_N_UBATCH: u32 = 512;
/// Smaller default prefill batch for multi-lane skippy serving.
///
/// When `lane_count > 1`, skippy enables llama.cpp unified KV mode: every
/// lane shares one `n_ctx` cell pool. A smaller default batch reduces the
/// amount of KV space each prefill asks the shared pool to reserve at once
/// after other lanes reset or preserve resident prefixes.
pub const SKIPPY_UNIFIED_KV_DEFAULT_N_BATCH: u32 = 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum FlashAttentionType {
    #[default]
    Auto = -1,
    Disabled = 0,
    Enabled = 1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub ctx_size: u32,
    pub lane_count: u32,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub n_threads: Option<u32>,
    pub n_threads_batch: Option<u32>,
    pub n_gpu_layers: i32,
    pub mmap: Option<bool>,
    pub mlock: bool,
    pub selected_backend_device: Option<String>,
    pub cache_type_k: u32,
    pub cache_type_v: u32,
    pub flash_attn_type: FlashAttentionType,
    pub load_mode: LoadMode,
    pub projector_path: Option<String>,
    pub include_embeddings: bool,
    pub include_output: bool,
    pub filter_tensors_on_load: bool,
}

impl RuntimeConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.layer_start >= self.layer_end {
            return Err("layer_start must be less than layer_end");
        }
        if self
            .selected_backend_device
            .as_deref()
            .is_some_and(str::is_empty)
        {
            return Err("selected_backend_device must not be empty");
        }
        if self.projector_path.as_deref().is_some_and(str::is_empty) {
            return Err("projector_path must not be empty");
        }
        if self.n_batch == Some(0) {
            return Err("n_batch must be greater than zero when provided");
        }
        if self.n_ubatch == Some(0) {
            return Err("n_ubatch must be greater than zero when provided");
        }
        if self.n_threads == Some(0) {
            return Err("n_threads must be greater than zero when provided");
        }
        if self.n_threads_batch == Some(0) {
            return Err("n_threads_batch must be greater than zero when provided");
        }
        Ok(())
    }

    pub(crate) fn as_raw(&self) -> Result<RawRuntimeConfigParts> {
        self.validate().map_err(anyhow::Error::msg)?;
        let n_batch = self
            .n_batch
            .unwrap_or_else(|| default_n_batch_for_lane_count(self.lane_count));
        let n_ubatch = self.n_ubatch.unwrap_or(LLAMA_SERVER_DEFAULT_N_UBATCH);
        let selected_backend_device = self
            .selected_backend_device
            .as_ref()
            .map(|device| {
                CString::new(device.as_bytes())
                    .context("selected_backend_device contains an interior NUL byte")
            })
            .transpose()?;
        let selected_backend_device_ptr = selected_backend_device
            .as_ref()
            .map(|device| device.as_ptr())
            .unwrap_or(ptr::null());
        Ok(RawRuntimeConfigParts {
            raw: RawRuntimeConfig {
                stage_index: i32::try_from(self.stage_index).context("stage_index exceeds i32")?,
                layer_start: i32::try_from(self.layer_start).context("layer_start exceeds i32")?,
                layer_end: i32::try_from(self.layer_end).context("layer_end exceeds i32")?,
                ctx_size: i32::try_from(self.ctx_size).context("ctx_size exceeds i32")?,
                lane_count: i32::try_from(self.lane_count).context("lane_count exceeds i32")?,
                n_batch: i32::try_from(n_batch).context("n_batch exceeds i32")?,
                n_ubatch: i32::try_from(n_ubatch).context("n_ubatch exceeds i32")?,
                n_threads: self
                    .n_threads
                    .map(i32::try_from)
                    .transpose()
                    .context("n_threads exceeds i32")?
                    .unwrap_or(0),
                n_threads_batch: self
                    .n_threads_batch
                    .or(self.n_threads)
                    .map(i32::try_from)
                    .transpose()
                    .context("n_threads_batch exceeds i32")?
                    .unwrap_or(0),
                n_gpu_layers: self.n_gpu_layers,
                has_mmap_override: self.mmap.is_some(),
                use_mmap: self.mmap.unwrap_or(false),
                use_mlock: self.mlock,
                cache_type_k: i32::try_from(self.cache_type_k)
                    .context("cache_type_k exceeds i32")?,
                cache_type_v: i32::try_from(self.cache_type_v)
                    .context("cache_type_v exceeds i32")?,
                flash_attn_type: self.flash_attn_type as i32,
                load_mode: self.load_mode,
                disable_repack: false,
                filter_tensors_on_load: self.filter_tensors_on_load,
                include_embeddings: self.include_embeddings,
                include_output: self.include_output,
                selected_backend_device: selected_backend_device_ptr,
            },
            _selected_backend_device: selected_backend_device,
        })
    }

    pub(crate) fn native_log_summary(&self) -> String {
        let n_batch = self
            .n_batch
            .unwrap_or_else(|| default_n_batch_for_lane_count(self.lane_count));
        let n_ubatch = self.n_ubatch.unwrap_or(LLAMA_SERVER_DEFAULT_N_UBATCH);
        format!(
            "stage_index={} layers={}..{} ctx={} lanes={} n_batch={} n_ubatch={} n_gpu_layers={} mmap={} mlock={} backend={} cache_k={} cache_v={} flash_attn={:?} load_mode={:?} include_embeddings={} include_output={} filter_tensors_on_load={}",
            self.stage_index,
            self.layer_start,
            self.layer_end,
            self.ctx_size,
            self.lane_count,
            n_batch,
            n_ubatch,
            self.n_gpu_layers,
            self.mmap
                .map(|value| value.to_string())
                .unwrap_or_else(|| "auto".to_string()),
            self.mlock,
            self.selected_backend_device.as_deref().unwrap_or("auto"),
            self.cache_type_k,
            self.cache_type_v,
            self.flash_attn_type,
            self.load_mode,
            self.include_embeddings,
            self.include_output,
            self.filter_tensors_on_load,
        )
    }
}

pub(crate) fn default_n_batch_for_lane_count(lane_count: u32) -> u32 {
    if lane_count > 1 {
        SKIPPY_UNIFIED_KV_DEFAULT_N_BATCH
    } else {
        LLAMA_SERVER_DEFAULT_N_BATCH
    }
}

pub(crate) struct RawRuntimeConfigParts {
    pub(crate) raw: RawRuntimeConfig,
    _selected_backend_device: Option<CString>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            ctx_size: 512,
            lane_count: 1,
            n_batch: Some(LLAMA_SERVER_DEFAULT_N_BATCH),
            n_ubatch: Some(LLAMA_SERVER_DEFAULT_N_UBATCH),
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            selected_backend_device: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
            flash_attn_type: FlashAttentionType::Auto,
            load_mode: LoadMode::RuntimeSlice,
            projector_path: None,
            include_embeddings: true,
            include_output: true,
            filter_tensors_on_load: false,
        }
    }
}

pub fn parse_cache_type(value: &str) -> Result<u32> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "" | "f16" => Ok(GGML_TYPE_F16),
        "q4" | "q4_0" => Ok(GGML_TYPE_Q4_0),
        "q8" | "q8_0" => Ok(GGML_TYPE_Q8_0),
        _ => Err(anyhow!("unsupported KV cache type {value:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_config_rejects_empty_selected_backend_device() {
        let config = RuntimeConfig {
            selected_backend_device: Some(String::new()),
            ..RuntimeConfig::default()
        };

        assert_eq!(
            config.validate(),
            Err("selected_backend_device must not be empty")
        );
    }

    #[test]
    fn runtime_config_rejects_zero_thread_counts() {
        let thread_config = RuntimeConfig {
            n_threads: Some(0),
            ..RuntimeConfig::default()
        };
        let batch_thread_config = RuntimeConfig {
            n_threads_batch: Some(0),
            ..RuntimeConfig::default()
        };

        assert_eq!(
            thread_config.validate(),
            Err("n_threads must be greater than zero when provided")
        );
        assert_eq!(
            batch_thread_config.validate(),
            Err("n_threads_batch must be greater than zero when provided")
        );
    }

    #[test]
    fn runtime_config_raw_mmap_override_and_mlock_are_distinct() -> anyhow::Result<()> {
        let forced_config = RuntimeConfig {
            mmap: Some(false),
            mlock: true,
            ..RuntimeConfig::default()
        };
        let forced_raw = forced_config.as_raw()?;

        assert!(forced_raw.raw.has_mmap_override);
        assert!(!forced_raw.raw.use_mmap);
        assert!(forced_raw.raw.use_mlock);

        let auto_config = RuntimeConfig {
            mmap: None,
            mlock: false,
            ..RuntimeConfig::default()
        };
        let auto_raw = auto_config.as_raw()?;

        assert!(!auto_raw.raw.has_mmap_override);
        assert!(!auto_raw.raw.use_mmap);
        assert!(!auto_raw.raw.use_mlock);

        Ok(())
    }

    #[test]
    fn parse_cache_type_accepts_legacy_mesh_kv_defaults() -> anyhow::Result<()> {
        assert_eq!(parse_cache_type("f16")?, GGML_TYPE_F16);
        assert_eq!(parse_cache_type("q8_0")?, GGML_TYPE_Q8_0);
        assert_eq!(parse_cache_type("q4_0")?, GGML_TYPE_Q4_0);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_preserves_selected_backend_device() -> anyhow::Result<()> {
        let config = RuntimeConfig {
            selected_backend_device: Some("MTL0".to_string()),
            ..RuntimeConfig::default()
        };

        let raw = config.as_raw()?;
        let device =
            unsafe { std::ffi::CStr::from_ptr(raw.raw.selected_backend_device).to_string_lossy() };

        assert_eq!(device, "MTL0");
        Ok(())
    }

    #[test]
    fn runtime_config_raw_uses_smaller_batch_for_unified_kv_defaults() -> anyhow::Result<()> {
        let config = RuntimeConfig {
            lane_count: 4,
            n_batch: None,
            n_ubatch: None,
            ..RuntimeConfig::default()
        };

        let raw = config.as_raw()?;

        assert_eq!(raw.raw.n_batch, SKIPPY_UNIFIED_KV_DEFAULT_N_BATCH as i32);
        assert_eq!(raw.raw.n_ubatch, LLAMA_SERVER_DEFAULT_N_UBATCH as i32);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_keeps_llama_batch_default_for_single_lane() -> anyhow::Result<()> {
        let config = RuntimeConfig {
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            ..RuntimeConfig::default()
        };

        let raw = config.as_raw()?;

        assert_eq!(raw.raw.n_batch, LLAMA_SERVER_DEFAULT_N_BATCH as i32);
        assert_eq!(raw.raw.n_ubatch, LLAMA_SERVER_DEFAULT_N_UBATCH as i32);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_preserves_explicit_unified_kv_batch() -> anyhow::Result<()> {
        let config = RuntimeConfig {
            lane_count: 4,
            n_batch: Some(2048),
            n_ubatch: Some(256),
            ..RuntimeConfig::default()
        };

        let raw = config.as_raw()?;

        assert_eq!(raw.raw.n_batch, 2048);
        assert_eq!(raw.raw.n_ubatch, 256);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_preserves_thread_counts_and_batch_defaults() -> anyhow::Result<()> {
        let config = RuntimeConfig {
            n_batch: None,
            n_ubatch: None,
            n_threads: Some(12),
            n_threads_batch: Some(6),
            ..RuntimeConfig::default()
        };

        let raw = config.as_raw()?;

        assert_eq!(raw.raw.n_batch, LLAMA_SERVER_DEFAULT_N_BATCH as i32);
        assert_eq!(raw.raw.n_ubatch, LLAMA_SERVER_DEFAULT_N_UBATCH as i32);
        assert_eq!(raw.raw.n_threads, 12);
        assert_eq!(raw.raw.n_threads_batch, 6);
        Ok(())
    }
}
