//! Mesh stage preparation and load diagnostics.
use crate::{
    SkippyDeviceDescriptor, SkippyModelLoadOptions, checkpoint, synthetic_direct_gguf_package,
};
use anyhow::Result;
use skippy_protocol::{StageConfig, StageDevice};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn single_stage_config(options: &SkippyModelLoadOptions) -> Result<StageConfig> {
    let prepared_options = skippy_api::SingleStageOptions {
        ctx_size: options.ctx_size,
        generation_concurrency: options.generation_concurrency,
        selected_device: options.selected_device.clone().map(Into::into),
        model_id: options.model_id.clone(),
        model_path: options.model_path.clone(),
        layer_start: options.layer_start,
        layer_end: options.layer_end,
        projector_path: options.projector_path.clone(),
        projector_use_gpu: options.projector_use_gpu,
        media_marker: options.media_marker.clone(),
        image_min_tokens: options.image_min_tokens,
        image_max_tokens: options.image_max_tokens,
        batch_max_tokens: options.batch_max_tokens,
        glm_dsa_policy: options.glm_dsa_policy,
        generation_signal_window: options.generation_signal_window,
        n_batch: options.n_batch,
        n_ubatch: options.n_ubatch,
        n_gpu_layers: options.n_gpu_layers,
        mmap: options.mmap,
        mlock: options.mlock,
        repack: options.repack,
        op_offload: options.op_offload,
        no_host_buffer: options.no_host_buffer,
        check_tensors: options.check_tensors,
        direct_io: options.direct_io,
        main_gpu: options.main_gpu,
        split_mode: options.split_mode,
        cache_type_k: options.cache_type_k.clone(),
        cache_type_v: options.cache_type_v.clone(),
        flash_attn_type: options.flash_attn_type,
        kv_offload: options.kv_offload,
        kv_unified: options.kv_unified,
        swa_full: options.swa_full,
        cache_idle_slots: options.cache_idle_slots,
        checkpoint_quantization: options.checkpoint_quantization.clone(),
        checkpoint_imatrix: options.checkpoint_imatrix.clone(),
        native_mtp_enabled: options.native_mtp_enabled,
        kv_cache: options.kv_cache.clone(),
    };
    prepared_options.validate()?;
    let package_identity = match options.package_identity.as_ref() {
        Some(identity) => identity.clone(),
        None => synthetic_direct_gguf_package(&options.model_id, &options.model_path)?,
    };
    let config = skippy_api::single_stage_config_with_graph_evidence(
        &prepared_options,
        package_identity,
        format!("mesh-skippy-{}", now_unix_nanos()),
    )?;
    checkpoint::emit_load_notice(
        &options.model_path,
        config
            .checkpoint_quantization
            .as_deref()
            .unwrap_or("preserve")
            .parse()
            .map_err(anyhow::Error::msg)?,
        config.checkpoint_imatrix.is_some(),
    );
    Ok(config)
}

impl From<SkippyDeviceDescriptor> for StageDevice {
    fn from(device: SkippyDeviceDescriptor) -> Self {
        Self {
            backend_device: device.backend_device,
            stable_id: device.stable_id,
            index: device.index,
            vram_bytes: device.vram_bytes,
        }
    }
}

impl From<StageDevice> for SkippyDeviceDescriptor {
    fn from(device: StageDevice) -> Self {
        Self {
            backend_device: device.backend_device,
            stable_id: device.stable_id,
            index: device.index,
            vram_bytes: device.vram_bytes,
        }
    }
}

fn now_unix_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}
