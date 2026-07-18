//! Partial-layer MLX implementation of the engine-neutral Skippy stage contract.

mod nemotron_h;

use std::{collections::BTreeMap, path::PathBuf, sync::mpsc, thread};

use anyhow::{Context, Result, anyhow, bail, ensure};
use safemlx::module::{Module, ModuleParameters, ModuleParametersExt};
use safemlx::ops::indexing::{NewAxis, TryIndexOp};
use safemlx::{Array, Device, DeviceType, Dtype, Stream, arange};
use safemlx_lm::{
    cache::{ConcatKeyValueCache, KeyValueCache},
    models::{
        common::linear::project_logits_maybe_quantized,
        llama::{self, AttentionInput, TransformerBlock},
    },
    quantization::{AffineQuantization, WeightQuantization},
    weights::{
        StrictLoadConfig, StrictLoadReport, load_safetensors_dir_quantized_strict,
        load_safetensors_dir_strict,
    },
};
use skippy_engine::{
    StageActivation, StageEngine, StageEngineInfo, StageExecutionKind, StageExecutionOutput,
    StageExecutionRequest,
};

use self::nemotron_h::NemotronHMoeStage;
pub use self::nemotron_h::{
    MlxNemotronHStageValidationReport, MlxNemotronHWireValidationReport,
    validate_nemotron_h_binary_wire, validate_nemotron_h_binary_wire_tokens,
    validate_nemotron_h_stage_engine,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MlxComputeDtype {
    F16,
    #[default]
    Bf16,
    F32,
}

impl MlxComputeDtype {
    fn mlx(self) -> Dtype {
        match self {
            Self::F16 => Dtype::Float16,
            Self::Bf16 => Dtype::Bfloat16,
            Self::F32 => Dtype::Float32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlxWeightQuantization {
    Affine { group_size: i32, bits: i32 },
    MxFp4,
}

impl MlxWeightQuantization {
    pub(crate) fn safemlx(self) -> Result<WeightQuantization> {
        match self {
            Self::Affine { group_size, bits } => {
                Ok(AffineQuantization::new(group_size, bits)?.into())
            }
            Self::MxFp4 => Ok(WeightQuantization::MxFp4),
        }
    }

    pub(crate) fn label(self) -> String {
        match self {
            Self::Affine { group_size, bits } => format!("affine-{bits}bit-g{group_size}"),
            Self::MxFp4 => "mxfp4".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MlxStageEngineConfig {
    pub model_dir: PathBuf,
    pub model_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub compute_dtype: MlxComputeDtype,
    pub weight_quantization: Option<MlxWeightQuantization>,
    pub ctx_size: Option<u32>,
}

enum WorkerJob {
    Execute {
        request: StageExecutionRequest,
        reply: mpsc::Sender<Result<StageExecutionOutput, String>>,
    },
    Reset {
        session_id: u64,
        reply: mpsc::Sender<Result<(), String>>,
    },
}

/// Send+Sync handle whose worker thread exclusively owns all MLX objects.
pub struct MlxStageEngine {
    info: StageEngineInfo,
    jobs: mpsc::Sender<WorkerJob>,
}

impl MlxStageEngine {
    pub fn spawn(config: MlxStageEngineConfig) -> Result<Self> {
        let (jobs, job_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        thread::Builder::new()
            .name(format!("mlx-stage-{}", config.stage_index))
            .spawn(move || run_worker(config, job_rx, ready_tx))?;
        match ready_rx.recv() {
            Ok(Ok(info)) => Ok(Self { info, jobs }),
            Ok(Err(error)) => Err(anyhow!("MLX stage load failed: {error}")),
            Err(_) => Err(anyhow!("MLX stage worker exited before readiness")),
        }
    }

    pub fn stage_info(&self) -> &StageEngineInfo {
        &self.info
    }

    fn request<T>(
        &self,
        make_job: impl FnOnce(mpsc::Sender<Result<T, String>>) -> WorkerJob,
    ) -> Result<T> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.jobs
            .send(make_job(reply_tx))
            .map_err(|_| anyhow!("MLX stage worker is not running"))?;
        reply_rx
            .recv()
            .map_err(|_| anyhow!("MLX stage worker dropped its reply"))?
            .map_err(anyhow::Error::msg)
    }
}

impl StageEngine for MlxStageEngine {
    fn info(&self) -> &StageEngineInfo {
        &self.info
    }

    fn execute(&self, request: StageExecutionRequest) -> Result<StageExecutionOutput> {
        self.request(|reply| WorkerJob::Execute { request, reply })
    }

    fn reset_session(&self, session_id: u64) -> Result<()> {
        self.request(|reply| WorkerJob::Reset { session_id, reply })
    }
}

struct LlamaLoadedStage {
    model: llama::Model,
    stream: Stream,
    compute_dtype: Dtype,
    ctx_size: Option<usize>,
    info: StageEngineInfo,
    sessions: BTreeMap<u64, Vec<Option<ConcatKeyValueCache>>>,
}

enum LoadedStage {
    Llama(Box<LlamaLoadedStage>),
    NemotronH(Box<NemotronHMoeStage>),
}

impl LoadedStage {
    fn info(&self) -> &StageEngineInfo {
        match self {
            Self::Llama(stage) => &stage.info,
            Self::NemotronH(stage) => stage.info(),
        }
    }

    fn execute(&mut self, request: StageExecutionRequest) -> Result<StageExecutionOutput> {
        match self {
            Self::Llama(stage) => stage.execute(request),
            Self::NemotronH(stage) => stage.execute(request),
        }
    }

    fn reset_session(&mut self, session_id: u64) {
        match self {
            Self::Llama(stage) => {
                stage.sessions.remove(&session_id);
            }
            Self::NemotronH(stage) => stage.reset_session(session_id),
        }
    }
}

fn run_worker(
    config: MlxStageEngineConfig,
    job_rx: mpsc::Receiver<WorkerJob>,
    ready_tx: mpsc::Sender<Result<StageEngineInfo, String>>,
) {
    let mut stage = match load_stage(config) {
        Ok(stage) => {
            let _ = ready_tx.send(Ok(stage.info().clone()));
            stage
        }
        Err(error) => {
            let _ = ready_tx.send(Err(format!("{error:#}")));
            return;
        }
    };
    while let Ok(job) = job_rx.recv() {
        match job {
            WorkerJob::Execute { request, reply } => {
                let _ = reply.send(stage.execute(request).map_err(|error| format!("{error:#}")));
            }
            WorkerJob::Reset { session_id, reply } => {
                stage.reset_session(session_id);
                let _ = reply.send(Ok(()));
            }
        }
    }
}

fn load_stage(config: MlxStageEngineConfig) -> Result<LoadedStage> {
    if model_type(&config.model_dir)?.as_deref() == Some("nemotron_h") {
        return Ok(LoadedStage::NemotronH(Box::new(NemotronHMoeStage::load(
            config,
        )?)));
    }
    Ok(LoadedStage::Llama(Box::new(load_llama_stage(config)?)))
}

fn model_type(model_dir: &std::path::Path) -> Result<Option<String>> {
    let config_path = model_dir.join("config.json");
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path)
            .with_context(|| format!("read MLX stage config {}", config_path.display()))?,
    )
    .with_context(|| format!("parse MLX stage config {}", config_path.display()))?;
    Ok(config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned))
}

fn load_llama_stage(config: MlxStageEngineConfig) -> Result<LlamaLoadedStage> {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut model_args = llama::get_llama_model_args(&config.model_dir)?;
    let total_layers = u32::try_from(model_args.num_hidden_layers)?;
    let info = StageEngineInfo {
        engine: "mlx".to_string(),
        model_id: config.model_id,
        stage_index: config.stage_index,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        total_layers,
        activation_width: u32::try_from(model_args.hidden_size)?,
    };
    info.validate()?;

    let quantization = config
        .weight_quantization
        .map(MlxWeightQuantization::safemlx)
        .transpose()?;
    let weight_quantization_label = config.weight_quantization.map_or_else(
        || {
            model_args
                .quantization
                .or(model_args.quantization_config)
                .map_or_else(
                    || "none".to_string(),
                    |value| format!("checkpoint-{value:?}"),
                )
        },
        MlxWeightQuantization::label,
    );
    if let Some(quantization) = quantization {
        ensure!(
            model_args.quantization.is_none() && model_args.quantization_config.is_none(),
            "MLX stage load-time quantization requires a dense source checkpoint"
        );
        model_args.quantization = Some(quantization);
    }
    let mut model = llama::Model::new(model_args, &stream)?;
    let load_config = partial_stage_load_config(&info);
    let mut load_report = StrictLoadReport::default();
    match quantization {
        Some(quantization) => load_safetensors_dir_quantized_strict(
            &mut model,
            &config.model_dir,
            &weights_stream,
            &stream,
            quantization,
            &load_config,
            &mut load_report,
        )?,
        None => load_safetensors_dir_strict(
            &mut model,
            &config.model_dir,
            &weights_stream,
            &load_config,
            &mut load_report,
        )?,
    }
    load_report.finish(&model, &load_config)?;
    retain_local_layers(&mut model, info.layer_start, info.layer_end)?;
    copy_stage_weights_to_compute_stream(&mut model, &info, &stream)?;
    stream.synchronize()?;
    eprintln!(
        "MLX partial stage loaded: model={} stage={} layers={}..{} tensors={} weight_quantization={}",
        info.model_id,
        info.stage_index,
        info.layer_start,
        info.layer_end,
        model.parameters().flatten().len(),
        weight_quantization_label,
    );
    Ok(LlamaLoadedStage {
        model,
        stream,
        compute_dtype: config.compute_dtype.mlx(),
        ctx_size: config.ctx_size.map(usize::try_from).transpose()?,
        info,
        sessions: BTreeMap::new(),
    })
}

fn partial_stage_load_config(info: &StageEngineInfo) -> StrictLoadConfig {
    let mut config = StrictLoadConfig::default();
    for layer in 0..info.total_layers {
        if layer < info.layer_start || layer >= info.layer_end {
            config = config.allow_missing_contains(format!("model.layers.{layer}."));
        }
    }
    if !info.is_first() && !info.is_final() {
        config = config.allow_missing_contains("model.embed_tokens.");
    }
    if !info.is_final() {
        config = config
            .allow_missing_contains("model.norm.")
            .allow_missing_contains("lm_head.");
    }
    config
}

fn retain_local_layers(model: &mut llama::Model, start: u32, end: u32) -> Result<()> {
    let start = usize::try_from(start)?;
    let end = usize::try_from(end)?;
    ensure!(
        end <= model.model.layers.len(),
        "stage layer range is out of bounds"
    );
    model.model.layers = model.model.layers.drain(start..end).collect();
    model.model.num_hidden_layers = i32::try_from(model.model.layers.len())?;
    Ok(())
}

fn copy_stage_weights_to_compute_stream(
    model: &mut llama::Model,
    info: &StageEngineInfo,
    stream: &Stream,
) -> Result<()> {
    if info.is_first() || info.is_final() {
        model.model.embed_tokens.copy_to_stream(stream)?;
    }
    for layer in &mut model.model.layers {
        layer.copy_to_stream(stream)?;
    }
    if info.is_final() {
        model.model.norm.copy_to_stream(stream)?;
        if let Some(lm_head) = &mut model.lm_head {
            lm_head.copy_to_stream(stream)?;
        }
    }
    Ok(())
}

impl LlamaLoadedStage {
    fn execute(&mut self, request: StageExecutionRequest) -> Result<StageExecutionOutput> {
        if request.kind == StageExecutionKind::Verify {
            bail!("MLX dense stage verification is not implemented yet");
        }
        if request
            .sampling
            .as_ref()
            .is_some_and(|sampling| sampling.enabled())
        {
            bail!("MLX staged execution currently supports greedy sampling only");
        }
        ensure!(!request.token_ids.is_empty(), "stage request has no tokens");
        let token_count = request.token_ids.len();
        self.ensure_context_capacity(request.session_id, token_count)?;
        if let Some(input) = request.input.as_ref() {
            ensure!(
                input.token_count == token_count,
                "input activation token count does not match token sideband"
            );
        }

        let mut hidden = self.input_hidden(&request)?;
        let caches = self.sessions.entry(request.session_id).or_default();
        let mask = attention_mask(&hidden, caches, &self.stream)?;
        if caches.is_empty() {
            *caches = (0..self.model.model.layers.len())
                .map(|_| Some(ConcatKeyValueCache::default()))
                .collect();
        }
        hidden = forward_blocks(
            &mut self.model.model.layers,
            hidden,
            mask.as_ref(),
            caches,
            &self.stream,
        )?;

        if self.info.is_final() {
            let hidden = self.model.model.norm.forward(&hidden, &self.stream)?;
            let logits = project_logits_maybe_quantized(
                &mut self.model.lm_head,
                &mut self.model.model.embed_tokens,
                &hidden,
                &self.stream,
            )?;
            let predicted = last_argmax(&logits, &self.stream)?;
            return Ok(StageExecutionOutput {
                activation: None,
                predicted_tokens: vec![predicted],
            });
        }

        Ok(StageExecutionOutput {
            activation: Some(array_activation(&hidden, &self.stream)?),
            predicted_tokens: Vec::new(),
        })
    }

    fn ensure_context_capacity(&self, session_id: u64, token_count: usize) -> Result<()> {
        let Some(ctx_size) = self.ctx_size else {
            return Ok(());
        };
        let offset = self
            .sessions
            .get(&session_id)
            .and_then(|caches| caches.first())
            .and_then(Option::as_ref)
            .map(KeyValueCache::offset)
            .map(usize::try_from)
            .transpose()?
            .unwrap_or_default();
        ensure!(
            offset.saturating_add(token_count) <= ctx_size,
            "MLX stage context limit {ctx_size} exceeded by {} tokens",
            offset.saturating_add(token_count)
        );
        Ok(())
    }

    fn input_hidden(&mut self, request: &StageExecutionRequest) -> Result<Array> {
        if self.info.is_first() {
            ensure!(
                request.input.is_none(),
                "first stage cannot accept residual input"
            );
            let tokens = request
                .token_ids
                .iter()
                .copied()
                .map(|token| u32::try_from(token).context("negative token ID"))
                .collect::<Result<Vec<_>>>()?;
            let shape = [1, i32::try_from(tokens.len())?];
            let tokens = Array::from_slice(&tokens, &shape);
            return Ok(self
                .model
                .model
                .embed_tokens
                .forward(&tokens, &self.stream)?);
        }
        let input = request
            .input
            .as_ref()
            .context("non-first stage requires residual input")?;
        ensure!(
            input.width == self.info.activation_width as usize,
            "input activation width mismatch"
        );
        let values = input.values();
        let hidden = Array::from_slice(
            &values,
            &[
                1,
                i32::try_from(input.token_count)?,
                i32::try_from(input.width)?,
            ],
        );
        Ok(hidden.as_dtype(self.compute_dtype, &self.stream)?)
    }
}

fn attention_mask(
    hidden: &Array,
    cache: &[Option<ConcatKeyValueCache>],
    stream: &Stream,
) -> Result<Option<Array>> {
    let sequence = hidden.shape()[1];
    if sequence == 1 {
        return Ok(None);
    }
    let offset = cache
        .first()
        .and_then(Option::as_ref)
        .map_or(0, KeyValueCache::offset);
    let right = arange!(stop = offset + sequence, stream = stream)?;
    let left = arange!(start = offset, stop = offset + sequence, stream = stream)?;
    let left = left.try_index_device((.., NewAxis), stream)?;
    let right = right.try_index_device(NewAxis, stream)?;
    Ok(Some(left.ge(&right, stream)?))
}

fn forward_blocks(
    blocks: &mut [TransformerBlock],
    mut hidden: Array,
    mask: Option<&Array>,
    cache: &mut [Option<ConcatKeyValueCache>],
    stream: &Stream,
) -> Result<Array> {
    ensure!(cache.len() == blocks.len(), "stage cache length mismatch");
    for (block, layer_cache) in blocks.iter_mut().zip(cache.iter_mut()) {
        hidden = block.forward(
            AttentionInput {
                x: &hidden,
                mask,
                cache: layer_cache.as_mut(),
                generated_sliding_window: None,
            },
            stream,
        )?;
    }
    Ok(hidden)
}

fn array_activation(hidden: &Array, stream: &Stream) -> Result<StageActivation> {
    let shape = hidden.shape().to_vec();
    ensure!(
        shape.len() == 3 && shape[0] == 1,
        "unexpected residual shape"
    );
    let values = hidden
        .as_dtype(Dtype::Float32, stream)?
        .evaluated()?
        .as_slice::<f32>()
        .to_vec();
    StageActivation::from_values(shape[1] as usize, shape[2] as usize, &values)
}

fn last_argmax(logits: &Array, stream: &Stream) -> Result<i32> {
    let row = logits
        .try_index_device((0, -1, ..), stream)?
        .as_dtype(Dtype::Float32, stream)?;
    let evaluated = row.evaluated()?;
    let values = evaluated.as_slice::<f32>();
    let (index, _) = values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .context("cannot argmax empty logits")?;
    Ok(i32::try_from(index)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_nemotron_h_stage_family_from_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"model_type":"nemotron_h"}"#,
        )
        .unwrap();
        assert_eq!(
            model_type(dir.path()).unwrap().as_deref(),
            Some("nemotron_h")
        );
    }

    #[test]
    fn validates_requested_weight_quantization_before_model_load() {
        let affine = MlxWeightQuantization::Affine {
            group_size: 64,
            bits: 4,
        }
        .safemlx()
        .unwrap();
        assert_eq!(affine.group_size(), 64);
        assert_eq!(affine.bits(), 4);
        assert!(
            MlxWeightQuantization::Affine {
                group_size: 48,
                bits: 4,
            }
            .safemlx()
            .is_err()
        );
        assert_eq!(
            MlxWeightQuantization::MxFp4.safemlx().unwrap(),
            WeightQuantization::MxFp4
        );
    }
}
