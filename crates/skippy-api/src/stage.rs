use crate::{checkpoint, family_policy::family_policy_for_model_path};
use anyhow::Result;
use skippy_protocol::{FlashAttentionType, LoadMode, StageConfig, StageDevice, StageKvCacheConfig};
use std::path::PathBuf;

/// Resolved source metadata supplied by the caller after verifying its model.
/// This builder preserves identity; it does not acquire or verify model bytes.
#[derive(Clone, Debug)]
pub struct StageSourceIdentity {
    pub package_ref: String,
    pub manifest_sha256: String,
    pub source_model_path: PathBuf,
    pub source_model_sha256: String,
    pub source_model_bytes: u64,
    pub layer_count: u32,
}

#[derive(Clone, Debug)]
pub struct SingleStageOptions {
    pub ctx_size: u32,
    pub generation_concurrency: usize,
    pub selected_device: Option<StageDevice>,
    pub model_id: String,
    pub model_path: PathBuf,
    pub layer_start: u32,
    pub layer_end: Option<u32>,
    pub projector_path: Option<PathBuf>,
    pub projector_use_gpu: Option<bool>,
    pub media_marker: Option<String>,
    pub image_min_tokens: Option<u32>,
    pub image_max_tokens: Option<u32>,
    pub batch_max_tokens: Option<u32>,
    pub glm_dsa_policy: skippy_protocol::GlmDsaPolicy,
    pub generation_signal_window: Option<u32>,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub n_gpu_layers: i32,
    pub mmap: Option<bool>,
    pub mlock: bool,
    pub repack: bool,
    pub op_offload: Option<bool>,
    pub no_host_buffer: bool,
    pub check_tensors: bool,
    pub direct_io: bool,
    pub main_gpu: Option<u32>,
    pub split_mode: skippy_protocol::SplitMode,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub flash_attn_type: FlashAttentionType,
    pub kv_offload: Option<bool>,
    pub kv_unified: Option<bool>,
    pub swa_full: Option<bool>,
    pub cache_idle_slots: Option<u32>,
    pub checkpoint_quantization: Option<String>,
    pub checkpoint_imatrix: Option<String>,
    pub native_mtp_enabled: bool,
    pub kv_cache: Option<StageKvCacheConfig>,
}

impl SingleStageOptions {
    pub fn new(model_id: impl Into<String>, model_path: impl Into<PathBuf>) -> Self {
        Self {
            ctx_size: 4096,
            generation_concurrency: 1,
            selected_device: None,
            model_id: model_id.into(),
            model_path: model_path.into(),
            layer_start: 0,
            layer_end: None,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
            generation_signal_window: None,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            checkpoint_quantization: None,
            checkpoint_imatrix: None,
            native_mtp_enabled: true,
            kv_cache: None,
        }
    }
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.ctx_size > 0,
            "skippy ctx_size must be greater than zero"
        );
        anyhow::ensure!(
            self.generation_concurrency > 0,
            "skippy generation_concurrency must be greater than zero"
        );
        if let Some(device) = self.selected_device.as_ref() {
            anyhow::ensure!(
                !device.backend_device.is_empty(),
                "skippy selected backend device must not be empty"
            );
        }
        Ok(())
    }
}

/// Build a local stage without loading a native runtime or emitting diagnostics.
/// Partial-layer package graph admission is a separate operation.
pub fn single_stage_config(
    options: &SingleStageOptions,
    package_identity: StageSourceIdentity,
    run_id: String,
) -> Result<StageConfig> {
    options.validate()?;
    let layer_start = options.layer_start;
    let layer_end = options.layer_end.unwrap_or(package_identity.layer_count);
    anyhow::ensure!(
        layer_end > 0,
        "skippy stage layer_end must be greater than zero"
    );
    anyhow::ensure!(
        layer_start < layer_end,
        "skippy stage layer range must satisfy layer_start < layer_end"
    );
    let family_policy = family_policy_for_model_path(&options.model_path);
    let checkpoint = checkpoint::prepare(options)?;
    let mut config = StageConfig {
        run_id: run_id.clone(),
        topology_id: format!("topology-{run_id}"),
        model_id: options.model_id.clone(),
        package_ref: Some(package_identity.package_ref),
        manifest_sha256: Some(package_identity.manifest_sha256),
        source_model_path: Some(
            package_identity
                .source_model_path
                .to_string_lossy()
                .to_string(),
        ),
        source_model_sha256: Some(package_identity.source_model_sha256),
        source_model_bytes: Some(package_identity.source_model_bytes),
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(options.model_path.to_string_lossy().to_string()),
        model_part_paths: Vec::new(),
        projector_path: options
            .projector_path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        projector_use_gpu: options.projector_use_gpu,
        media_marker: options.media_marker.clone(),
        image_min_tokens: options.image_min_tokens,
        image_max_tokens: options.image_max_tokens,
        batch_max_tokens: options.batch_max_tokens,
        glm_dsa_policy: options.glm_dsa_policy,
        generation_signal_window: options.generation_signal_window,
        activation_codec: skippy_protocol::StageActivationCodec::default(),
        activation_codec_policy: skippy_protocol::StageActivationCodecPolicy::default(),
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start,
        layer_end,
        ctx_size: options.ctx_size,
        lane_count: options.generation_concurrency as u32,
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
        filter_tensors_on_load: false,
        resident_tensor_names: Vec::new(),
        activation_import_identities: Vec::new(),
        activation_import_bindings: Vec::new(),
        activation_export_identities: Vec::new(),
        activation_export_bindings: Vec::new(),
        checkpoint_quantization: options
            .checkpoint_quantization
            .as_ref()
            .map(|_| checkpoint.quantization.canonical_name().to_string()),
        checkpoint_imatrix: checkpoint.imatrix,
        checkpoint_imatrix_sha256: checkpoint.imatrix_sha256,
        selected_device: options.selected_device.clone(),
        kv_cache: None,
        native_mtp_enabled: options.native_mtp_enabled,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: None,
    };
    config.kv_cache = options
        .kv_cache
        .clone()
        .or_else(|| family_policy.stage_kv_cache_config_for_stage(&config));
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(path: PathBuf) -> StageSourceIdentity {
        StageSourceIdentity {
            package_ref: "package:test".into(),
            manifest_sha256: "ab".repeat(32),
            source_model_path: path,
            source_model_sha256: "cd".repeat(32),
            source_model_bytes: 1234,
            layer_count: 30,
        }
    }

    #[test]
    fn full_stage_retains_tokenizer_identity_and_caller_run_identity() {
        let options = SingleStageOptions::new("model", "/missing/model.gguf");
        let config = single_stage_config(
            &options,
            identity("/source/model.gguf".into()),
            "standalone-run".into(),
        )
        .unwrap();
        assert_eq!(config.run_id, "standalone-run");
        assert_eq!(config.topology_id, "topology-standalone-run");
        assert_eq!(
            config.source_model_path.as_deref(),
            Some("/source/model.gguf")
        );
        assert_eq!(config.source_model_sha256, Some("cd".repeat(32)));
        assert_eq!(config.source_model_bytes, Some(1234));
        assert_eq!(config.manifest_sha256, Some("ab".repeat(32)));
        assert_eq!((config.layer_start, config.layer_end), (0, 30));
        assert!(config.upstream.is_none() && config.downstream.is_none());
        assert!(!config.filter_tensors_on_load);
    }

    #[test]
    fn invalid_options_fail_before_checkpoint_file_access() {
        let mut options = SingleStageOptions::new("model", "/missing/model.gguf");
        options.ctx_size = 0;
        options.checkpoint_imatrix = Some("missing.imatrix".into());
        let error =
            single_stage_config(&options, identity(options.model_path.clone()), "run".into())
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("ctx_size must be greater than zero")
        );
        options.ctx_size = 512;
        options.layer_start = 30;
        let error =
            single_stage_config(&options, identity(options.model_path.clone()), "run".into())
                .unwrap_err();
        assert!(error.to_string().contains("layer_start < layer_end"));
    }
}
