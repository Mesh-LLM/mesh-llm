use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use skippy_protocol::{FlashAttentionType, LoadMode, StageConfig};
use skippy_runtime::{
    ActivationFrame, DecodeBatchRequest, DecodeFrameBatchOutput, DecodeFrameBatchRequest,
    FlashAttentionType as RuntimeFlashAttentionType, GenerationSignalWindow, IterationBatchOutput,
    IterationBatchPhase, IterationBatchRequest, MediaInput, MediaPrefill, MediaPrefillFrame,
    MtpSource, NativeMtpDraft, RuntimeConfig, RuntimeKvPage, RuntimeKvPageDesc, RuntimeLoadMode,
    SamplingConfig, StageModel, StageSession, TokenSignal, parse_cache_type,
};

use crate::package::select_package_parts;

mod frame_operations;
mod lane_lifecycle;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeLaunchOverrides {
    pub n_threads: Option<usize>,
    pub n_threads_batch: Option<usize>,
    pub mtp_source: MtpSource,
}

pub struct RuntimeState {
    pub model: StageModel,
    layer_start: u32,
    layer_end: u32,
    lane_count: u32,
    /// Size of the context's KV cell pool, in tokens (llama.cpp `n_ctx`). In
    /// unified-KV mode every lane draws decode/prefill cells from this single
    /// shared pool, so it is the real ceiling for scheduler admission — see
    /// [`Self::kv_pool_tokens`].
    ctx_size: u32,
    /// High-water mark of lane indices ever handed out. Combined with
    /// [`Self::free_lane_indices`], the count of live lanes equals
    /// `next_lane_index - free_lane_indices.len()`.
    next_lane_index: usize,
    /// Lane indices that were previously handed out but are now free
    /// to reuse. An index lands here only when the lane's underlying
    /// StageSession has been dropped (which calls skippy_session_free
    /// on the C side, clearing that seq_id's KV cells).
    ///
    /// Without this list, a discarded lane (see
    /// [`Self::drop_session_timed`]) would permanently consume one of
    /// the slots represented by [`Self::next_lane_index`], leading to
    /// "all execution lanes are busy" errors long before the runtime
    /// has actually run out of capacity.
    free_lane_indices: Vec<usize>,
    sessions: BTreeMap<String, RuntimeLaneSession>,
    idle_sessions: Vec<RuntimeLaneSession>,
    session_token_counts: BTreeMap<String, u64>,
    session_resident_prefixes: BTreeMap<String, ResidentLanePrefix>,
}

struct RuntimeLaneSession {
    index: usize,
    session: StageSession,
    resident_prefix: Option<ResidentLanePrefix>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeSessionLaneStats {
    pub index: usize,
    pub active: bool,
    pub session_id: Option<String>,
    pub token_count: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeSessionStats {
    pub lane_count: usize,
    pub active_sessions: usize,
    pub idle_sessions: usize,
    pub idle_resident_prefixes: usize,
    pub tracked_token_counts: usize,
    pub max_session_tokens: u64,
    pub total_session_tokens: u64,
    pub lanes: Vec<RuntimeSessionLaneStats>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeSessionDropStats {
    pub reset_session: bool,
    pub reset_ms: f64,
    pub preserved_resident_prefix: bool,
    /// True when the lane could not be returned to the idle pool because
    /// the underlying StageSession failed to reset cleanly. The lane is
    /// dropped (which invokes the C-side skippy_session_free) and the
    /// pool capacity is restored on the next prewarm/admission cycle.
    pub lane_discarded: bool,
    /// Reset-error detail, when [`Self::lane_discarded`] is true.
    pub lane_discard_reason: Option<String>,
    pub stats_after: RuntimeSessionStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeSessionAlignStats {
    pub before_token_count: u64,
    pub after_token_count: u64,
}

pub struct RuntimeDecodeBatchRequest<'a> {
    pub session_id: &'a str,
    pub token_id: i32,
    pub sampling: Option<&'a SamplingConfig>,
}

pub struct RuntimeDecodeFrameBatchRequest<'a> {
    pub session_id: &'a str,
    pub token_id: i32,
    pub sampling: Option<&'a SamplingConfig>,
    pub input: Option<&'a ActivationFrame>,
}

pub struct RuntimeIterationBatchRequest<'a> {
    pub session_id: &'a str,
    pub token_ids: &'a [i32],
    pub positions: &'a [i32],
    pub sampling: Option<&'a SamplingConfig>,
    pub input: Option<&'a ActivationFrame>,
    pub sample_last: bool,
    pub phase: IterationBatchPhase,
}

#[derive(Debug, Clone)]
struct ResidentLanePrefix {
    page_id: String,
    token_count: u64,
}

impl RuntimeState {
    /// A runtime with no model behind it, for tests that exercise code paths
    /// which never touch the model.
    ///
    /// [`Self::session_stats`] is pure Rust over the lane bookkeeping below, so
    /// status/observability behaviour can be tested without loading a GGUF.
    /// Any call that reaches [`Self::model`] will dereference a null handle, so
    /// this must not be used to drive inference.
    #[cfg(test)]
    pub(crate) fn new_modelless_for_test(lane_count: u32) -> Self {
        Self::new_modelless_with_capacity_for_test(lane_count, 0)
    }

    #[cfg(test)]
    pub(crate) fn new_modelless_with_capacity_for_test(lane_count: u32, ctx_size: u32) -> Self {
        Self {
            model: StageModel::new_dummy(),
            layer_start: 0,
            layer_end: 1,
            lane_count,
            ctx_size,
            next_lane_index: 0,
            free_lane_indices: Vec::new(),
            sessions: BTreeMap::new(),
            idle_sessions: Vec::new(),
            session_token_counts: BTreeMap::new(),
            session_resident_prefixes: BTreeMap::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn track_session_tokens_for_test(&mut self, session_id: &str, token_count: u64) {
        self.session_token_counts
            .insert(session_id.to_string(), token_count);
    }

    pub fn lane_count(&self) -> u32 {
        self.lane_count
    }

    /// Total KV cell pool available to this context, in tokens (`n_ctx`).
    ///
    /// In unified-KV mode all lanes share this single pool, so it is the real
    /// token budget the iteration scheduler must admit against. Returns 0 for
    /// the modelless test runtime, in which case callers should fall back to a
    /// configured default.
    pub fn kv_pool_tokens(&self) -> u32 {
        self.ctx_size
    }
}

impl Drop for RuntimeState {
    fn drop(&mut self) {
        self.sessions.clear();
        self.idle_sessions.clear();
    }
}

pub fn load_runtime(config: &StageConfig) -> Result<Option<Arc<Mutex<RuntimeState>>>> {
    load_runtime_with_overrides(config, &RuntimeLaunchOverrides::default())
}

pub fn load_runtime_with_overrides(
    config: &StageConfig,
    overrides: &RuntimeLaunchOverrides,
) -> Result<Option<Arc<Mutex<RuntimeState>>>> {
    let mut runtime_config = runtime_config_from_stage_config(config, overrides)?;

    let model = match config.load_mode {
        _ if std::env::var("MESH_LLM_BYPASS_SKIPPY_MODEL_LOAD").is_ok() => {
            skippy_runtime::StageModel::new_dummy()
        }
        LoadMode::LayerPackage => {
            let selected =
                select_package_parts(config).context("select layer package parts for stage")?;
            if runtime_config.projector_path.is_none() && should_attach_package_projector(config) {
                runtime_config.projector_path = selected
                    .projector_paths
                    .first()
                    .map(|path| path.to_string_lossy().to_string());
            }
            open_stage_model_from_parts(&selected.absolute_paths, &runtime_config)?
        }
        _ => {
            let Some(model_path) = config.model_path.as_ref().map(std::path::Path::new) else {
                return Ok(None);
            };
            open_stage_model(model_path, &runtime_config)?
        }
    };

    Ok(Some(Arc::new(Mutex::new(RuntimeState {
        model,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        lane_count: config.lane_count,
        ctx_size: config.ctx_size,
        next_lane_index: 0,
        free_lane_indices: Vec::new(),
        sessions: BTreeMap::new(),
        idle_sessions: Vec::new(),
        session_token_counts: BTreeMap::new(),
        session_resident_prefixes: BTreeMap::new(),
    }))))
}

pub fn load_runtime_with_overrides_and_open_events(
    config: &StageConfig,
    overrides: &RuntimeLaunchOverrides,
    model_open_event_reporter: Option<&mut (dyn FnMut(skippy_runtime::RuntimeEvent) + Send)>,
) -> Result<Option<Arc<Mutex<RuntimeState>>>> {
    let mut runtime_config = runtime_config_from_stage_config(config, overrides)?;

    let model = match config.load_mode {
        _ if std::env::var("MESH_LLM_BYPASS_SKIPPY_MODEL_LOAD").is_ok() => {
            skippy_runtime::StageModel::new_dummy()
        }
        LoadMode::LayerPackage => {
            let selected =
                select_package_parts(config).context("select layer package parts for stage")?;
            if runtime_config.projector_path.is_none() && should_attach_package_projector(config) {
                runtime_config.projector_path = selected
                    .projector_paths
                    .first()
                    .map(|path| path.to_string_lossy().to_string());
            }
            open_stage_model_from_parts_with_events(
                &selected.absolute_paths,
                &runtime_config,
                model_open_event_reporter,
            )?
        }
        _ => {
            let Some(model_path) = config.model_path.as_ref().map(std::path::Path::new) else {
                return Ok(None);
            };
            open_stage_model_with_events(model_path, &runtime_config, model_open_event_reporter)?
        }
    };

    Ok(Some(Arc::new(Mutex::new(RuntimeState {
        model,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        lane_count: config.lane_count,
        ctx_size: config.ctx_size,
        next_lane_index: 0,
        free_lane_indices: Vec::new(),
        sessions: BTreeMap::new(),
        idle_sessions: Vec::new(),
        session_token_counts: BTreeMap::new(),
        session_resident_prefixes: BTreeMap::new(),
    }))))
}

fn should_attach_package_projector(config: &StageConfig) -> bool {
    config.stage_index == 0 && config.layer_start == 0
}

fn runtime_config_from_stage_config(
    config: &StageConfig,
    overrides: &RuntimeLaunchOverrides,
) -> Result<RuntimeConfig> {
    let cache_type_k = parse_cache_type(&config.cache_type_k)
        .with_context(|| format!("parse cache_type_k for {}", config.stage_id))?;
    let cache_type_v = parse_cache_type(&config.cache_type_v)
        .with_context(|| format!("parse cache_type_v for {}", config.stage_id))?;
    let n_threads = overrides
        .n_threads
        .map(u32::try_from)
        .transpose()
        .with_context(|| format!("n_threads exceeds u32 for {}", config.stage_id))?;
    let n_threads_batch = overrides
        .n_threads_batch
        .map(u32::try_from)
        .transpose()
        .with_context(|| format!("n_threads_batch exceeds u32 for {}", config.stage_id))?;
    Ok(RuntimeConfig {
        stage_index: config.stage_index,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        ctx_size: config.ctx_size,
        lane_count: config.lane_count,
        n_batch: config.n_batch,
        n_ubatch: config.n_ubatch,
        n_threads,
        n_threads_batch,
        n_gpu_layers: config.n_gpu_layers,
        mmap: config.mmap,
        mlock: config.mlock,
        selected_backend_device: config
            .selected_device
            .as_ref()
            .map(|device| device.backend_device.clone()),
        cache_type_k,
        cache_type_v,
        flash_attn_type: match config.flash_attn_type {
            FlashAttentionType::Auto => RuntimeFlashAttentionType::Auto,
            FlashAttentionType::Disabled => RuntimeFlashAttentionType::Disabled,
            FlashAttentionType::Enabled => RuntimeFlashAttentionType::Enabled,
        },
        load_mode: match config.load_mode {
            LoadMode::RuntimeSlice => RuntimeLoadMode::RuntimeSlice,
            LoadMode::LayerPackage => RuntimeLoadMode::LayerPackage,
            LoadMode::ArtifactSlice => RuntimeLoadMode::ArtifactSlice,
        },
        projector_path: config.projector_path.clone(),
        include_embeddings: config.layer_start == 0
            || (config.load_mode == LoadMode::LayerPackage && config.downstream.is_none()),
        include_output: config.downstream.is_none(),
        mtp_source: overrides.mtp_source,
        filter_tensors_on_load: config.filter_tensors_on_load,
    })
}

fn open_stage_model(path: &std::path::Path, runtime_config: &RuntimeConfig) -> Result<StageModel> {
    StageModel::open(path, runtime_config)
}

fn open_stage_model_with_events(
    path: &std::path::Path,
    runtime_config: &RuntimeConfig,
    model_open_event_reporter: Option<&mut (dyn FnMut(skippy_runtime::RuntimeEvent) + Send)>,
) -> Result<StageModel> {
    match model_open_event_reporter {
        Some(event_reporter) => StageModel::open_with_events(path, runtime_config, event_reporter),
        None => StageModel::open(path, runtime_config),
    }
}

fn open_stage_model_from_parts(
    paths: &[std::path::PathBuf],
    runtime_config: &RuntimeConfig,
) -> Result<StageModel> {
    StageModel::open_from_parts(paths, runtime_config)
}

fn open_stage_model_from_parts_with_events(
    paths: &[std::path::PathBuf],
    runtime_config: &RuntimeConfig,
    model_open_event_reporter: Option<&mut (dyn FnMut(skippy_runtime::RuntimeEvent) + Send)>,
) -> Result<StageModel> {
    match model_open_event_reporter {
        Some(event_reporter) => {
            StageModel::open_from_parts_with_events(paths, runtime_config, event_reporter)
        }
        None => StageModel::open_from_parts(paths, runtime_config),
    }
}

#[cfg(test)]
mod tests {
    use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig, StageConfig, StageDevice};
    use skippy_runtime::{
        ActivationDesc, ActivationFrame, FlashAttentionType as RuntimeFlashAttentionType,
        MtpSource, RuntimeActivationDType, RuntimeActivationLayout, RuntimeConfig, SamplingConfig,
    };

    use super::{
        RuntimeLaunchOverrides, RuntimeState, load_runtime_with_overrides,
        runtime_config_from_stage_config, should_attach_package_projector,
    };

    #[test]
    fn modelless_runtime_reports_zero_kv_pool_so_scheduler_uses_fallback() {
        // The scheduler derives its admission budget from `kv_pool_tokens()` and
        // keeps its configured default when the runtime reports 0. The modelless
        // test runtime carries no context, so it must report 0 (not panic or
        // report a stale non-zero pool).
        let rt = RuntimeState::new_modelless_for_test(4);
        assert_eq!(rt.kv_pool_tokens(), 0);
        assert_eq!(rt.lane_count(), 4);
    }

    #[test]
    fn runtime_config_preserves_selected_backend_device_and_thread_overrides() {
        let config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: Some("/tmp/mmproj.gguf".to_string()),
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 24,
            ctx_size: 512,
            lane_count: 2,
            n_batch: Some(1024),
            n_ubatch: Some(256),
            n_gpu_layers: -1,
            mmap: Some(false),
            mlock: true,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Enabled,
            filter_tensors_on_load: true,
            selected_device: Some(StageDevice {
                backend_device: "Vulkan1".into(),
                stable_id: Some("pci:0000:65:00.0".into()),
                index: Some(1),
                vram_bytes: Some(16_000_000_000),
            }),
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        };

        let overrides = RuntimeLaunchOverrides {
            n_threads: Some(8),
            n_threads_batch: Some(4),
            mtp_source: MtpSource::External,
        };

        let runtime_config = runtime_config_from_stage_config(&config, &overrides).unwrap();

        assert_eq!(
            runtime_config.selected_backend_device.as_deref(),
            Some("Vulkan1")
        );
        assert_eq!(runtime_config.lane_count, 2);
        assert_eq!(runtime_config.n_batch, Some(1024));
        assert_eq!(runtime_config.n_ubatch, Some(256));
        assert_eq!(runtime_config.n_threads, Some(8));
        assert_eq!(runtime_config.n_threads_batch, Some(4));
        assert_eq!(runtime_config.mmap, Some(false));
        assert!(runtime_config.mlock);
        assert_eq!(
            runtime_config.flash_attn_type,
            RuntimeFlashAttentionType::Enabled
        );
        assert_eq!(runtime_config.mtp_source, MtpSource::External);
    }

    #[test]
    fn runtime_config_keeps_package_embeddings_for_final_non_first_stage() {
        let config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: Some("/tmp/package".to_string()),
            manifest_sha256: Some("manifest".to_string()),
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/package".to_string()),
            projector_path: None,
            stage_id: "stage-2".to_string(),
            stage_index: 2,
            layer_start: 20,
            layer_end: 30,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: true,
            selected_device: Some(StageDevice {
                backend_device: "CPU".into(),
                stable_id: None,
                index: None,
                vram_bytes: None,
            }),
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::LayerPackage,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: Some(PeerConfig {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                endpoint: "tcp://127.0.0.1:19001".to_string(),
            }),
            downstream: None,
        };

        let runtime_config =
            runtime_config_from_stage_config(&config, &RuntimeLaunchOverrides::default()).unwrap();

        assert!(runtime_config.include_embeddings);
        assert!(runtime_config.include_output);
        assert_eq!(runtime_config.mtp_source, MtpSource::Disabled);
    }

    fn glm52_mtp_fixture() -> Option<(std::path::PathBuf, StageConfig)> {
        let package_path =
            std::env::var_os("SKIPPY_GLM52_MTP_PACKAGE").map(std::path::PathBuf::from)?;
        if !package_path.join("model-package.json").is_file() {
            eprintln!(
                "skipping: {} does not look like a layer package",
                package_path.display()
            );
            return None;
        }
        let config = StageConfig {
            run_id: "glm52-mtp-smoke".to_string(),
            topology_id: "glm52-mtp-smoke-topology".to_string(),
            model_id: "meshllm/GLM-5.2-Q2_K-MTP-Q8-layers".to_string(),
            package_ref: Some(package_path.to_string_lossy().to_string()),
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some(package_path.to_string_lossy().to_string()),
            projector_path: None,
            stage_id: "stage-final".to_string(),
            stage_index: 1,
            // GLM-DSA stages must begin on a full-indexer layer. Layer 78 is
            // the auxiliary next-token head rather than a base transformer
            // layer, so the smallest valid final-stage fixture is 74..78;
            // native MTP loading retains layer 78's nextn tensors alongside it.
            layer_start: 74,
            layer_end: 78,
            ctx_size: 128,
            lane_count: 1,
            n_batch: Some(1),
            n_ubatch: Some(1),
            n_gpu_layers: 0,
            mmap: Some(true),
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Disabled,
            filter_tensors_on_load: true,
            selected_device: Some(StageDevice {
                backend_device: "CPU".into(),
                stable_id: None,
                index: None,
                vram_bytes: None,
            }),
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::LayerPackage,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: Some(PeerConfig {
                stage_id: "stage-prev".to_string(),
                stage_index: 0,
                endpoint: "tcp://127.0.0.1:19000".to_string(),
            }),
            downstream: None,
        };
        Some((package_path, config))
    }

    fn glm52_mtp_input(token_count: u32) -> ActivationFrame {
        let hidden_bytes = 6144 * token_count as usize * std::mem::size_of::<f32>();
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 0,
                layer_start: 0,
                layer_end: 74,
                token_count,
                sequence_count: 1,
                payload_bytes: hidden_bytes as u64,
                flags: 0,
            },
            payload: vec![0; hidden_bytes],
        }
    }

    #[test]
    fn glm52_final_stage_package_executes_native_mtp_when_fixture_is_set() -> anyhow::Result<()> {
        let Some((_package_path, config)) = glm52_mtp_fixture() else {
            eprintln!("skipping: SKIPPY_GLM52_MTP_PACKAGE is not set");
            return Ok(());
        };

        let runtime = load_runtime_with_overrides(
            &config,
            &RuntimeLaunchOverrides {
                mtp_source: MtpSource::Integrated,
                ..RuntimeLaunchOverrides::default()
            },
        )?
        .expect("GLM final stage should load from the package");
        let mut runtime = runtime.lock().expect("runtime mutex poisoned");
        let input = glm52_mtp_input(1);
        let sampling = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        let (predicted, draft, _output) =
            runtime.decode_frame_sampled_mtp("smoke", 1, Some(&sampling), Some(&input), 0, 1)?;
        let draft = draft.expect("GLM final stage should return a native MTP draft");

        assert!(predicted >= 0);
        assert_eq!(draft.token_ids.len(), 1);
        assert!(draft.token_ids[0] >= 0);
        let verify_inputs = [predicted, draft.token_ids[0]];
        let (verified, _next_draft, _output) = runtime.verify_frame_sampled(
            "smoke",
            &verify_inputs,
            Some(&sampling),
            Some(&glm52_mtp_input(2)),
            0,
            1,
        )?;
        assert!(!verified.is_empty());
        runtime.retire_verify_checkpoint("smoke", 1, 2)?;
        Ok(())
    }

    #[test]
    fn glm52_final_stage_does_not_create_integrated_mtp_when_disabled() -> anyhow::Result<()> {
        let Some((_package_path, config)) = glm52_mtp_fixture() else {
            eprintln!("skipping: SKIPPY_GLM52_MTP_PACKAGE is not set");
            return Ok(());
        };
        let runtime = load_runtime_with_overrides(
            &config,
            &RuntimeLaunchOverrides {
                mtp_source: MtpSource::Disabled,
                ..RuntimeLaunchOverrides::default()
            },
        )?
        .expect("GLM final stage should load from the package");
        let mut runtime = runtime.lock().expect("runtime mutex poisoned");
        let sampling = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        let (predicted, draft, _output) = runtime.decode_frame_sampled_mtp(
            "disabled-mtp",
            1,
            Some(&sampling),
            Some(&glm52_mtp_input(1)),
            0,
            1,
        )?;

        assert!(predicted >= 0);
        assert!(
            draft.is_none(),
            "disabled MTP must not create a draft context"
        );
        Ok(())
    }

    #[test]
    fn glm52_external_sidecar_attaches_when_target_has_integrated_mtp_tensors() -> anyhow::Result<()>
    {
        let Some((_package_path, config)) = glm52_mtp_fixture() else {
            eprintln!("skipping: SKIPPY_GLM52_MTP_PACKAGE is not set");
            return Ok(());
        };
        let Some(sidecar_path) = std::env::var_os("SKIPPY_GLM52_MTP_SIDECAR") else {
            eprintln!("skipping: SKIPPY_GLM52_MTP_SIDECAR is not set");
            return Ok(());
        };
        let sidecar_path = std::path::PathBuf::from(sidecar_path);
        if !sidecar_path.is_file() {
            eprintln!(
                "skipping: {} is not an MTP sidecar GGUF",
                sidecar_path.display()
            );
            return Ok(());
        }

        let runtime = load_runtime_with_overrides(
            &config,
            &RuntimeLaunchOverrides {
                mtp_source: MtpSource::External,
                ..RuntimeLaunchOverrides::default()
            },
        )?
        .expect("GLM final stage should load from the package");
        let mut runtime = runtime.lock().expect("runtime mutex poisoned");
        runtime.model.attach_mtp_draft_model(
            &sidecar_path,
            &RuntimeConfig {
                ctx_size: config.ctx_size,
                lane_count: config.lane_count,
                n_batch: config.n_batch,
                n_ubatch: config.n_ubatch,
                n_gpu_layers: config.n_gpu_layers,
                mmap: config.mmap,
                mlock: config.mlock,
                selected_backend_device: Some("CPU".to_string()),
                mtp_source: MtpSource::External,
                ..RuntimeConfig::default()
            },
        )?;
        let sampling = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        let (predicted, draft, _output) = runtime.decode_frame_sampled_mtp(
            "external-mtp",
            1,
            Some(&sampling),
            Some(&glm52_mtp_input(1)),
            0,
            1,
        )?;
        let draft = draft.expect("external MTP sidecar should attach to the target");

        assert!(predicted >= 0);
        assert_eq!(draft.token_ids.len(), 1);
        assert!(draft.token_ids[0] >= 0);
        Ok(())
    }

    #[test]
    fn runtime_config_preserves_default_runtime_threads_when_omitted() {
        let config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 24,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: false,
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        };

        let runtime_config =
            runtime_config_from_stage_config(&config, &RuntimeLaunchOverrides::default()).unwrap();

        assert_eq!(runtime_config.n_threads, None);
        assert_eq!(runtime_config.n_threads_batch, None);
        assert_eq!(runtime_config.n_batch, None);
        assert_eq!(runtime_config.n_ubatch, None);
    }

    #[test]
    fn runtime_config_rejects_unsupported_cache_type_before_launch() {
        let config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 24,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            cache_type_k: "auto".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: false,
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        };

        let error = runtime_config_from_stage_config(&config, &RuntimeLaunchOverrides::default())
            .expect_err("unsupported cache types should fail during runtime config construction");

        assert!(
            error.to_string().contains("parse cache_type_k for stage-0"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn package_projector_fallback_is_stage_zero_only() {
        let mut config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: Some("/tmp/package".to_string()),
            manifest_sha256: Some("manifest".to_string()),
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/package".to_string()),
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 10,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: true,
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::LayerPackage,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: Some(PeerConfig {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                endpoint: "tcp://127.0.0.1:19001".to_string(),
            }),
        };

        assert!(should_attach_package_projector(&config));

        config.stage_id = "stage-1".to_string();
        config.stage_index = 1;
        config.layer_start = 10;
        config.layer_end = 20;
        config.upstream = Some(PeerConfig {
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            endpoint: "tcp://127.0.0.1:19000".to_string(),
        });
        config.downstream = None;

        assert!(!should_attach_package_projector(&config));
    }
}
