//! Build transport-neutral stage configurations from admitted planning facts.
use anyhow::{Context, Result, anyhow};
use skippy_protocol::{
    FlashAttentionType, LoadMode, PeerConfig, SplitMode, StageConfig, StageDevice,
};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedStagePackage {
    pub local_ref: String,
    pub source_model_path: String,
    pub source_model_sha256: String,
    pub source_model_bytes: Option<u64>,
    pub model_part_paths: Vec<PathBuf>,
    pub projector_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageLoadRuntimeSettings {
    pub repack: bool,
    pub op_offload: Option<bool>,
    pub no_host_buffer: bool,
    pub check_tensors: bool,
    pub direct_io: bool,
    pub main_gpu: Option<u32>,
    pub split_mode: SplitMode,
    pub kv_offload: Option<bool>,
    pub kv_unified: Option<bool>,
    pub swa_full: Option<bool>,
    pub cache_idle_slots: Option<u32>,
    pub activation_codec_policy: skippy_protocol::StageActivationCodecPolicy,
}

#[derive(Clone, Debug)]
pub struct AdmittedStageOptions {
    pub topology_id: String,
    pub run_id: String,
    pub model_id: String,
    pub stage_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub ctx_size: u32,
    pub lane_count: u32,
    pub selected_device: Option<StageDevice>,
    pub package_ref: String,
    pub manifest_sha256: String,
    pub model_path: Option<String>,
    pub source_model_sha256: Option<String>,
    pub source_model_bytes: Option<u64>,
    pub projector_path: Option<String>,
    pub projector_use_gpu: Option<bool>,
    pub media_marker: Option<String>,
    pub image_min_tokens: Option<u32>,
    pub image_max_tokens: Option<u32>,
    pub batch_max_tokens: Option<u32>,
    pub glm_dsa_policy: skippy_protocol::GlmDsaPolicy,
    pub generation_signal_window: Option<u32>,
    pub activation_codec: skippy_protocol::StageActivationCodec,
    pub activation_codec_policy: skippy_protocol::StageActivationCodecPolicy,
    pub stage_index: u32,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub n_gpu_layers: i32,
    pub mmap: Option<bool>,
    pub mlock: bool,
    pub runtime_settings: StageLoadRuntimeSettings,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub flash_attn_type: FlashAttentionType,
    pub load_mode: LoadMode,
    pub native_mtp_enabled: bool,
    pub bind_addr: String,
    pub upstream: Option<PeerConfig>,
    pub downstream: Option<PeerConfig>,
    pub admission: skippy_protocol::StageAdmissionDescriptor,
}

impl AdmittedStageOptions {
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.topology_id.is_empty(), "topology_id is required");
        anyhow::ensure!(!self.run_id.is_empty(), "run_id is required");
        anyhow::ensure!(!self.model_id.is_empty(), "model_id is required");
        anyhow::ensure!(!self.stage_id.is_empty(), "stage_id is required");
        anyhow::ensure!(
            self.layer_start < self.layer_end,
            "invalid stage layer range"
        );
        anyhow::ensure!(self.ctx_size > 0, "ctx_size must be greater than zero");
        anyhow::ensure!(self.lane_count > 0, "lane_count must be greater than zero");
        if let Some(device) = self.selected_device.as_ref() {
            anyhow::ensure!(
                !device.backend_device.is_empty(),
                "selected backend device must not be empty"
            );
        }
        Ok(())
    }
}

pub fn admitted_stage_config(
    load: &AdmittedStageOptions,
    package: Option<&ResolvedStagePackage>,
    resident_tensor_names: Vec<String>,
) -> Result<StageConfig> {
    load.validate()?;
    let frontier_profile = admitted_activation_frontier(&load.admission)?;
    let mut config = StageConfig {
        run_id: load.run_id.clone(),
        topology_id: load.topology_id.clone(),
        model_id: load.model_id.clone(),
        package_ref: Some(load.package_ref.clone()),
        manifest_sha256: Some(load.manifest_sha256.clone()),
        source_model_path: package
            .map(|package| package.source_model_path.clone())
            .or_else(|| load.model_path.clone()),
        source_model_sha256: package
            .map(|package| package.source_model_sha256.clone())
            .or_else(|| load.source_model_sha256.clone()),
        source_model_bytes: package
            .and_then(|package| package.source_model_bytes)
            .or(load.source_model_bytes),
        materialized_path: None,
        materialized_pinned: false,
        model_path: load.model_path.clone(),
        model_part_paths: package
            .map(|package| {
                package
                    .model_part_paths
                    .iter()
                    .map(|path| path.to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default(),
        projector_path: load.projector_path.clone().or_else(|| {
            package
                .and_then(|package| package.projector_path.as_ref())
                .map(|path| path.to_string_lossy().into_owned())
        }),
        projector_use_gpu: load.projector_use_gpu,
        media_marker: load.media_marker.clone(),
        image_min_tokens: load.image_min_tokens,
        image_max_tokens: load.image_max_tokens,
        batch_max_tokens: load.batch_max_tokens,
        glm_dsa_policy: load.glm_dsa_policy,
        generation_signal_window: load.generation_signal_window,
        activation_codec: load.activation_codec,
        activation_codec_policy: load.activation_codec_policy,
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        ctx_size: load.ctx_size,
        lane_count: load.lane_count,
        n_batch: load.n_batch,
        n_ubatch: load.n_ubatch,
        n_gpu_layers: load.n_gpu_layers,
        mmap: load.mmap,
        mlock: load.mlock,
        repack: load.runtime_settings.repack,
        op_offload: load.runtime_settings.op_offload,
        no_host_buffer: load.runtime_settings.no_host_buffer,
        check_tensors: load.runtime_settings.check_tensors,
        direct_io: load.runtime_settings.direct_io,
        main_gpu: load.runtime_settings.main_gpu,
        split_mode: load.runtime_settings.split_mode,
        cache_type_k: empty_to_default(&load.cache_type_k, "f16"),
        cache_type_v: empty_to_default(&load.cache_type_v, "f16"),
        flash_attn_type: load.flash_attn_type,
        kv_offload: load.runtime_settings.kv_offload,
        kv_unified: load.runtime_settings.kv_unified,
        swa_full: load.runtime_settings.swa_full,
        cache_idle_slots: load.runtime_settings.cache_idle_slots,
        filter_tensors_on_load: matches!(
            load.load_mode,
            LoadMode::RuntimeSlice | LoadMode::LayerPackage
        ),
        resident_tensor_names,
        activation_import_identities: frontier_profile.activation_imports.clone(),
        activation_import_bindings: frontier_profile.activation_import_bindings.clone(),
        activation_export_identities: frontier_profile.activation_exports.clone(),
        activation_export_bindings: frontier_profile.activation_export_bindings.clone(),
        checkpoint_quantization: None,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
        selected_device: load.selected_device.clone(),
        kv_cache: None,
        native_mtp_enabled: load.native_mtp_enabled,
        load_mode: load.load_mode.clone(),
        bind_addr: load.bind_addr.clone(),
        upstream: load.upstream.clone(),
        downstream: load.downstream.clone(),
    };
    let family_policy = crate::family_policy::family_policy_for_stage_config(&config);
    config.kv_cache = package.map_or_else(
        || family_policy.stage_kv_cache_config_for_stage(&config),
        |package| {
            family_policy.stage_kv_cache_config_for_package(&config, Path::new(&package.local_ref))
        },
    );
    Ok(config)
}

pub fn admitted_activation_frontier(
    admission: &skippy_protocol::StageAdmissionDescriptor,
) -> Result<&skippy_protocol::StageAdmissionProfile> {
    let frontier = admission
        .profiles
        .first()
        .context("stage admission descriptor has no execution profiles")?;
    anyhow::ensure!(
        admission.profiles.iter().all(|profile| {
            profile.activation_imports == frontier.activation_imports
                && profile.activation_exports == frontier.activation_exports
                && profile.activation_import_bindings == frontier.activation_import_bindings
                && profile.activation_export_bindings == frontier.activation_export_bindings
        }),
        "stage admission execution profiles disagree on activation frontier identities"
    );
    Ok(frontier)
}

pub fn admitted_resident_tensor_names(
    admission: &skippy_protocol::StageAdmissionDescriptor,
    manifest: &skippy_package_format::PackageManifest,
) -> Result<Vec<String>> {
    let package_id = manifest
        .computed_package_id()
        .context("compute planning identity for resident tensor binding")?;
    anyhow::ensure!(
        package_id == admission.package_id,
        "stage admission package identity differs from the local planning manifest"
    );
    let mut names = manifest
        .materialization_tensors(&admission.resident_tensor_ids)
        .map_err(|error| anyhow!(error.to_string()))?
        .into_iter()
        .map(|tensor| tensor.native_name.to_string())
        .collect::<Vec<_>>();
    names.sort();
    anyhow::ensure!(
        !names.is_empty() && names.windows(2).all(|window| window[0] < window[1]),
        "admitted resident tensor names must be non-empty, strictly sorted, and unique"
    );
    Ok(names)
}

fn empty_to_default(value: &str, default: &str) -> String {
    if value.is_empty() {
        default.to_string()
    } else {
        value.to_string()
    }
}
