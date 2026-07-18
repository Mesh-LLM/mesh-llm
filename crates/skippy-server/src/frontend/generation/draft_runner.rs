use crate::runtime_state::RuntimeState;
use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use skippy_protocol::StageConfig;
use skippy_runtime::FlashAttentionType as RuntimeFlashAttentionType;
use skippy_runtime::ModelInfo;
use skippy_runtime::RuntimeConfig;
use skippy_runtime::RuntimeLoadMode;
use skippy_runtime::StageModel;
use skippy_runtime::StageSession;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

pub(in crate::frontend) struct DraftRunner {
    pub(in crate::frontend) path: PathBuf,
    pub(in crate::frontend) window: usize,
    pub(in crate::frontend) _model: StageModel,
    pub(in crate::frontend) session: StageSession,
}

impl DraftRunner {
    pub(in crate::frontend) fn open(
        path: &Path,
        config: &StageConfig,
        n_gpu_layers: Option<i32>,
        window: usize,
    ) -> Result<Self> {
        if !path.is_file() {
            bail!("draft model does not exist: {}", path.display());
        }
        let layer_count = model_layer_count(path)?;
        let model = StageModel::open(
            path,
            &RuntimeConfig {
                stage_index: 0,
                layer_start: 0,
                layer_end: layer_count,
                ctx_size: config.ctx_size,
                lane_count: 1,
                n_batch: None,
                n_ubatch: None,
                n_threads: None,
                n_threads_batch: None,
                n_gpu_layers: n_gpu_layers.unwrap_or(config.n_gpu_layers),
                mmap: config.mmap,
                mlock: config.mlock,
                selected_backend_device: config
                    .selected_device
                    .as_ref()
                    .map(|device| device.backend_device.clone()),
                cache_type_k: skippy_runtime::GGML_TYPE_F16,
                cache_type_v: skippy_runtime::GGML_TYPE_F16,
                flash_attn_type: RuntimeFlashAttentionType::Auto,
                load_mode: RuntimeLoadMode::RuntimeSlice,
                projector_path: None,
                include_embeddings: true,
                include_output: true,
                filter_tensors_on_load: false,
            },
        )
        .with_context(|| format!("open draft model {}", path.display()))?;
        let session = model.create_session().context("create draft session")?;
        Ok(Self {
            path: path.to_path_buf(),
            window,
            _model: model,
            session,
        })
    }

    pub(in crate::frontend) fn reset_to_context(&mut self, context_tokens: &[i32]) -> Result<()> {
        self.session.reset().context("reset draft session")?;
        if context_tokens.len() > 1 {
            self.session
                .prefill_chunk(&context_tokens[..context_tokens.len() - 1])
                .context("prefill draft context")?;
        }
        Ok(())
    }

    pub(in crate::frontend) fn propose(
        &mut self,
        mut current: i32,
        max_tokens: usize,
    ) -> Result<Vec<i32>> {
        let mut tokens = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            current = self
                .session
                .decode_step(current)
                .context("draft decode step")?;
            tokens.push(current);
        }
        Ok(tokens)
    }
}

pub(in crate::frontend) fn open_draft_runner(
    path: Option<&Path>,
    config: &StageConfig,
    n_gpu_layers: Option<i32>,
    window: usize,
) -> Result<Option<Arc<Mutex<DraftRunner>>>> {
    let Some(path) = path else {
        return Ok(None);
    };
    Ok(Some(Arc::new(Mutex::new(DraftRunner::open(
        path,
        config,
        n_gpu_layers,
        window,
    )?))))
}

pub(in crate::frontend) fn attach_native_mtp_draft_model(
    path: Option<&Path>,
    runtime: &Arc<Mutex<RuntimeState>>,
    config: &StageConfig,
    n_gpu_layers: Option<i32>,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    if !path.is_file() {
        bail!("MTP draft model does not exist: {}", path.display());
    }
    let layer_count = model_layer_count(path)?;
    let mut runtime = runtime
        .lock()
        .map_err(|_| anyhow!("runtime lock poisoned"))?;
    runtime
        .model
        .attach_mtp_draft_model(
            path,
            &RuntimeConfig {
                stage_index: 0,
                layer_start: 0,
                layer_end: layer_count,
                ctx_size: config.ctx_size,
                lane_count: config.lane_count,
                n_batch: None,
                n_ubatch: None,
                n_threads: None,
                n_threads_batch: None,
                n_gpu_layers: n_gpu_layers.unwrap_or(config.n_gpu_layers),
                mmap: config.mmap,
                mlock: config.mlock,
                selected_backend_device: config
                    .selected_device
                    .as_ref()
                    .map(|device| device.backend_device.clone()),
                cache_type_k: skippy_runtime::GGML_TYPE_F16,
                cache_type_v: skippy_runtime::GGML_TYPE_F16,
                flash_attn_type: RuntimeFlashAttentionType::Auto,
                load_mode: RuntimeLoadMode::RuntimeSlice,
                projector_path: None,
                include_embeddings: true,
                include_output: true,
                filter_tensors_on_load: false,
            },
        )
        .with_context(|| format!("attach MTP draft model {}", path.display()))
}

pub(in crate::frontend) fn model_layer_count(path: &Path) -> Result<u32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model info {}", path.display()))?;
    let layer_count = info
        .tensors()?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|index| index + 1)
        .ok_or_else(|| anyhow!("could not infer layer count for {}", path.display()))?;
    Ok(layer_count)
}
