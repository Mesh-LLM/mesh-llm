//! Bounded exact-range to MLX-quantized stage artifact conversion.

use std::{
    borrow::Cow,
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use memmap2::MmapOptions;
use model_hf::safetensors_stage::{SafetensorsStageMaterializer, SafetensorsStageRequest};
use safemlx::{
    Array, Device, DeviceType, Dtype as MlxDtype, Stream,
    memory::{active_memory, cache_memory, peak_memory, reset_peak_memory},
    transforms::eval,
};
use safemlx_lm::quantization::{WeightQuantization, quantize_tensor};
use safetensors::tensor::{Dtype as SafeDtype, SafeTensors, View, serialize_to_file};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::stage::MlxWeightQuantization;
use nemotron_h::NemotronHDerivation;
pub(crate) use nemotron_h::{
    nemotron_h_validation_values, validate_nemotron_h_moe_stage_output,
    validate_nemotron_h_moe_stage_output_for_tokens,
};

mod cache;
mod expert_bank;
mod nemotron_h;

pub use cache::{
    MlxDerivedStageCacheConfig, MlxDerivedStageCacheResult, derive_quantized_stage_cached,
    load_prepared_quantized_stage, mlx_derived_stage_cache_root,
};
pub use nemotron_h::{MlxNemotronHValidationReport, validate_nemotron_h_moe_stage};

pub(super) const DERIVED_STAGE_SCHEMA_VERSION: u32 = 1;
const DERIVED_STAGE_IMPLEMENTATION: &str = "mesh-mlx-range-derived-v1";
const SAFEMLX_REVISION: &str = "c6b47418f3ea0e7b304464a80d8bc8f63f3bbc22";
const PLAN_FILE: &str = "stage-plan.json";
pub(super) const REPORT_FILE: &str = "derived-stage.json";
static DERIVED_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Expected source identity and cooperative cancellation for a derivation.
#[derive(Clone, Debug, Default)]
pub struct MlxDerivationControl {
    expected_checkpoint_sha256: Option<String>,
    cancelled: Option<Arc<AtomicBool>>,
}

impl MlxDerivationControl {
    pub fn new(
        expected_checkpoint_sha256: Option<String>,
        cancelled: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self {
            expected_checkpoint_sha256,
            cancelled,
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled
            .as_ref()
            .is_some_and(|cancelled| cancelled.load(Ordering::Acquire))
    }

    fn ensure_active(&self) -> Result<()> {
        ensure!(!self.is_cancelled(), "MLX stage derivation cancelled");
        Ok(())
    }

    fn verify_checkpoint(&self, actual: &str) -> Result<()> {
        if let Some(expected) = &self.expected_checkpoint_sha256 {
            ensure!(
                actual == expected,
                "MLX checkpoint identity {actual} does not match stage claim {expected}"
            );
        }
        Ok(())
    }
}

/// Configuration for producing one MLX-quantized partial stage.
#[derive(Clone, Debug)]
pub struct MlxDerivedStageConfig {
    pub source: SafetensorsStageRequest,
    pub output_dir: PathBuf,
    pub quantization: MlxWeightQuantization,
    pub control: MlxDerivationControl,
    /// Soft output bundle target. A single packed tensor may exceed this size.
    pub shard_size_bytes: usize,
}

/// One finalized SafeTensors shard in a derived stage artifact.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MlxDerivedStageShard {
    pub file: String,
    pub file_bytes: u64,
    pub sha256: String,
}

/// Evidence and identity for a bounded quantized stage derivation.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MlxDerivedStageReport {
    pub schema_version: u32,
    /// Identity of the source checkpoint, stage plan, quantization, implementation, and sharding.
    pub derivation_recipe_sha256: String,
    /// Corruption-detecting digest of every published artifact file except this report.
    pub output_content_sha256: String,
    pub checkpoint_sha256: String,
    pub plan_sha256: String,
    pub repo: String,
    pub revision: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub quantization: Value,
    pub quantization_label: String,
    pub safemlx_revision: String,
    pub output_dir: PathBuf,
    pub source_tensor_count: usize,
    pub source_tensor_bytes: u64,
    pub source_range_request_count: usize,
    pub source_temporary_file_peak_bytes: u64,
    pub quantized_tensor_count: usize,
    pub copied_tensor_count: usize,
    pub output_tensor_bytes: u64,
    /// Bytes in the published artifact files, excluding this report.
    pub artifact_file_bytes: u64,
    /// Measured source-tensor plus artifact payload high-water mark.
    ///
    /// Filesystem allocation, lock files, and this report are intentionally excluded.
    pub working_disk_peak_bytes: u64,
    pub mlx_active_memory_bytes: usize,
    pub mlx_cache_memory_bytes: usize,
    pub mlx_peak_memory_bytes: usize,
    pub shards: Vec<MlxDerivedStageShard>,
}

struct PendingShard {
    tensors: BTreeMap<String, OwnedTensor>,
    bytes: usize,
}

impl PendingShard {
    fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
            bytes: 0,
        }
    }

    fn insert(&mut self, name: String, tensor: OwnedTensor) -> Result<()> {
        ensure!(
            !self.tensors.contains_key(&name),
            "derived tensor name {name:?} was produced more than once"
        );
        self.bytes = self
            .bytes
            .checked_add(tensor.data.len())
            .context("pending derived shard byte count overflow")?;
        self.tensors.insert(name, tensor);
        Ok(())
    }
}

struct OwnedTensor {
    dtype: SafeDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedTensor {
    fn dtype(&self) -> SafeDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

struct BuildState {
    pending: PendingShard,
    temporary_shards: Vec<PathBuf>,
    locations: BTreeMap<String, usize>,
    quantized_tensor_count: usize,
    copied_tensor_count: usize,
    output_tensor_bytes: u64,
    written_output_file_bytes: u64,
    working_disk_peak_bytes: u64,
}

enum TensorDerivation {
    Llama,
    NemotronH(NemotronHDerivation),
}

impl TensorDerivation {
    fn new(
        config: &Value,
        layer_start: u32,
        layer_end: u32,
        quantization: WeightQuantization,
    ) -> Result<Self> {
        match config.get("model_type").and_then(Value::as_str) {
            Some("llama") => Ok(Self::Llama),
            Some("nemotron_h") => {
                ensure!(
                    matches!(quantization, WeightQuantization::Affine(_)),
                    "bounded Nemotron-H expert banks currently require affine quantization"
                );
                Ok(Self::NemotronH(NemotronHDerivation::new(
                    config,
                    layer_start,
                    layer_end,
                )?))
            }
            model_type => anyhow::bail!(
                "derived MLX stages support model_type=llama or nemotron_h, got {model_type:?}"
            ),
        }
    }

    fn finish(self) -> Result<Vec<(String, OwnedTensor)>> {
        match self {
            Self::Llama => Ok(Vec::new()),
            Self::NemotronH(derivation) => derivation.finish(),
        }
    }
}

impl BuildState {
    fn new(initial_output_file_bytes: u64) -> Self {
        Self {
            pending: PendingShard::new(),
            temporary_shards: Vec::new(),
            locations: BTreeMap::new(),
            quantized_tensor_count: 0,
            copied_tensor_count: 0,
            output_tensor_bytes: 0,
            written_output_file_bytes: initial_output_file_bytes,
            working_disk_peak_bytes: initial_output_file_bytes,
        }
    }

    fn observe_source_file(&mut self, bytes: u64) {
        self.working_disk_peak_bytes = self
            .working_disk_peak_bytes
            .max(self.written_output_file_bytes.saturating_add(bytes));
    }
}

/// Derives an MLX-compatible quantized partial stage without retaining a dense stage artifact.
pub fn derive_quantized_stage(
    materializer: &SafetensorsStageMaterializer,
    config: &MlxDerivedStageConfig,
) -> Result<MlxDerivedStageReport> {
    config.control.ensure_active()?;
    ensure!(
        config.shard_size_bytes > 0,
        "derived shard size must be non-zero"
    );
    remove_abandoned_outputs_for_destination(&config.output_dir)?;
    ensure!(
        !config.output_dir.exists(),
        "derived output already exists: {}",
        config.output_dir.display()
    );
    let visit = materializer.prepare_tensor_visit(config.source.clone())?;
    config.control.ensure_active()?;
    config
        .control
        .verify_checkpoint(visit.checkpoint_sha256())?;
    let plan = visit.plan().clone();
    let source_config = parse_source_config(visit.config())?;
    ensure_unquantized_source_config(&source_config)?;
    let quantization = config.quantization.safemlx()?;
    let mut tensor_derivation = TensorDerivation::new(
        &source_config,
        plan.layer_start,
        plan.layer_end,
        quantization,
    )?;
    let quantization_value = serde_json::to_value(quantization)?;
    let plan_bytes = serde_json::to_vec(&plan)?;
    let plan_sha256 = sha256_bytes(&plan_bytes);
    let derivation_recipe_sha256 = derived_identity(
        &plan,
        &plan_sha256,
        &quantization_value,
        config.shard_size_bytes,
    )?;
    let temporary = TemporaryOutput::create(&config.output_dir)?;
    write_quantized_config(
        temporary.path().join("config.json"),
        &source_config,
        &quantization_value,
    )?;
    write_json(temporary.path().join(PLAN_FILE), &plan)?;

    reset_peak_memory()?;
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let quantization_stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let initial_output_file_bytes = directory_file_bytes(temporary.path())?;
    let mut state = BuildState::new(initial_output_file_bytes);
    let visit_report = visit.visit_tensor_files_cancellable(
        || config.control.is_cancelled(),
        |tensor| {
            config.control.ensure_active()?;
            state.observe_source_file(tensor.file_bytes);
            let arrays = convert_tensor(
                tensor,
                quantization,
                &weights_stream,
                &quantization_stream,
                &mut tensor_derivation,
                &mut state,
            )?;
            config.control.ensure_active()?;
            append_arrays(
                arrays,
                config.shard_size_bytes,
                temporary.path(),
                tensor.file_bytes,
                &mut state,
            )?;
            config.control.ensure_active()
        },
    )?;
    config.control.ensure_active()?;
    let final_arrays = tensor_derivation.finish()?;
    if !final_arrays.is_empty() {
        append_arrays(
            final_arrays,
            config.shard_size_bytes,
            temporary.path(),
            0,
            &mut state,
        )?;
    }
    config.control.ensure_active()?;
    if !state.pending.tensors.is_empty() {
        flush_shard(temporary.path(), &mut state)?;
    }
    config.control.ensure_active()?;
    ensure!(
        !state.temporary_shards.is_empty(),
        "derived stage contains no tensors"
    );
    let finalized = finalize_shards(temporary.path(), &state)?;
    config.control.ensure_active()?;
    let shards = finalized
        .iter()
        .map(|path| {
            config.control.ensure_active()?;
            let shard = derived_shard(path)?;
            config.control.ensure_active()?;
            Ok(shard)
        })
        .collect::<Result<Vec<_>>>()?;
    config.control.ensure_active()?;
    let output_content_sha256 = output_content_sha256(temporary.path())?;
    config.control.ensure_active()?;
    let artifact_file_bytes = artifact_file_bytes(temporary.path())?;
    state.working_disk_peak_bytes = state.working_disk_peak_bytes.max(artifact_file_bytes);
    let report = MlxDerivedStageReport {
        schema_version: DERIVED_STAGE_SCHEMA_VERSION,
        derivation_recipe_sha256,
        output_content_sha256,
        checkpoint_sha256: plan.checkpoint_sha256.clone(),
        plan_sha256,
        repo: plan.repo.clone(),
        revision: plan.revision.clone(),
        layer_start: plan.layer_start,
        layer_end: plan.layer_end,
        quantization: quantization_value,
        quantization_label: config.quantization.label(),
        safemlx_revision: SAFEMLX_REVISION.to_string(),
        output_dir: config.output_dir.clone(),
        source_tensor_count: visit_report.visited_tensor_count,
        source_tensor_bytes: visit_report.visited_tensor_bytes,
        source_range_request_count: visit_report.source_range_request_count,
        source_temporary_file_peak_bytes: visit_report.temporary_file_peak_bytes,
        quantized_tensor_count: state.quantized_tensor_count,
        copied_tensor_count: state.copied_tensor_count,
        output_tensor_bytes: state.output_tensor_bytes,
        artifact_file_bytes,
        working_disk_peak_bytes: state.working_disk_peak_bytes,
        mlx_active_memory_bytes: active_memory()?,
        mlx_cache_memory_bytes: cache_memory()?,
        mlx_peak_memory_bytes: peak_memory()?,
        shards,
    };
    config.control.ensure_active()?;
    write_json(temporary.path().join(REPORT_FILE), &report)?;
    config.control.ensure_active()?;
    temporary.publish(&config.output_dir)?;
    Ok(report)
}

fn convert_tensor(
    tensor: &model_hf::safetensors_stage::SafetensorsStageTensorFile,
    quantization: WeightQuantization,
    weights_stream: &Stream,
    quantization_stream: &Stream,
    derivation: &mut TensorDerivation,
    state: &mut BuildState,
) -> Result<Vec<(String, OwnedTensor)>> {
    let file = File::open(&tensor.path)?;
    // SAFETY: the mapping is read-only and remains alive until all MLX work
    // derived from its sole TensorView is evaluated and synchronized below.
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;
    ensure!(
        tensors.len() == 1,
        "ephemeral SafeTensors file did not contain exactly {}",
        tensor.name
    );
    let dense = Array::try_from(tensors.tensor(&tensor.name)?)?.copy(weights_stream)?;
    if let TensorDerivation::NemotronH(nemotron) = derivation
        && nemotron.consume_expert(&tensor.name, &dense, quantization, quantization_stream)?
    {
        state.quantized_tensor_count += 1;
        weights_stream.synchronize()?;
        quantization_stream.synchronize()?;
        return Ok(Vec::new());
    }
    let output_name = match derivation {
        TensorDerivation::Llama => tensor.name.clone(),
        TensorDerivation::NemotronH(nemotron) => nemotron.rewrite_name(&tensor.name)?,
    };
    let keep_dense = matches!(
        derivation,
        TensorDerivation::NemotronH(_) if NemotronHDerivation::keep_dense(&output_name)
    );
    let arrays =
        if !keep_dense && should_quantize_source_weight(&output_name, &dense, quantization)? {
            state.quantized_tensor_count += 1;
            quantize_tensor(&dense, quantization, quantization_stream)?
                .into_named_arrays(&output_name)?
        } else {
            state.copied_tensor_count += 1;
            vec![(output_name, dense)]
        };
    eval(arrays.iter().map(|(_, array)| array))?;
    weights_stream.synchronize()?;
    quantization_stream.synchronize()?;
    arrays
        .into_iter()
        .map(|(name, array)| Ok((name, owned_tensor(&array)?)))
        .collect()
}

fn should_quantize_source_weight(
    name: &str,
    tensor: &Array,
    quantization: WeightQuantization,
) -> Result<bool> {
    ensure!(
        !name.ends_with(".scales")
            && !name.ends_with(".biases")
            && !name.ends_with("_scales")
            && !name.ends_with("_biases"),
        "source checkpoint already contains packed quantization companion {name}"
    );
    if !name.ends_with(".weight") || tensor.ndim() < 2 {
        return Ok(false);
    }
    ensure!(
        tensor.ndim() == 2,
        "derived stage only supports dense rank-2 matrix weights; {name} has rank {}",
        tensor.ndim()
    );
    ensure!(
        tensor.dtype().is_float(),
        "source weight {name} is already packed or uses unsupported dtype {:?}",
        tensor.dtype()
    );
    ensure!(
        tensor.dim(1) % quantization.group_size() == 0 && tensor.dim(1) % 32 == 0,
        "source weight {name} input dimension {} is incompatible with {}",
        tensor.dim(1),
        quantization_label(quantization)
    );
    Ok(true)
}

fn quantization_label(quantization: WeightQuantization) -> String {
    format!(
        "{:?}-{}bit-g{}",
        quantization.mode(),
        quantization.bits(),
        quantization.group_size()
    )
}

fn owned_tensor(array: &Array) -> Result<OwnedTensor> {
    let evaluated = array.evaluated()?;
    let (dtype, data) = match array.dtype() {
        MlxDtype::Bool => (
            SafeDtype::BOOL,
            evaluated
                .as_slice::<bool>()
                .iter()
                .map(|value| u8::from(*value))
                .collect(),
        ),
        MlxDtype::Uint8 => (SafeDtype::U8, evaluated.as_slice::<u8>().to_vec()),
        MlxDtype::Uint16 => (
            SafeDtype::U16,
            bytemuck::cast_slice(evaluated.as_slice::<u16>()).to_vec(),
        ),
        MlxDtype::Uint32 => (
            SafeDtype::U32,
            bytemuck::cast_slice(evaluated.as_slice::<u32>()).to_vec(),
        ),
        MlxDtype::Uint64 => (
            SafeDtype::U64,
            bytemuck::cast_slice(evaluated.as_slice::<u64>()).to_vec(),
        ),
        MlxDtype::Int8 => (
            SafeDtype::I8,
            bytemuck::cast_slice(evaluated.as_slice::<i8>()).to_vec(),
        ),
        MlxDtype::Int16 => (
            SafeDtype::I16,
            bytemuck::cast_slice(evaluated.as_slice::<i16>()).to_vec(),
        ),
        MlxDtype::Int32 => (
            SafeDtype::I32,
            bytemuck::cast_slice(evaluated.as_slice::<i32>()).to_vec(),
        ),
        MlxDtype::Int64 => (
            SafeDtype::I64,
            bytemuck::cast_slice(evaluated.as_slice::<i64>()).to_vec(),
        ),
        MlxDtype::Float16 => (
            SafeDtype::F16,
            bytemuck::cast_slice(evaluated.as_slice::<f16>()).to_vec(),
        ),
        MlxDtype::Bfloat16 => (
            SafeDtype::BF16,
            bytemuck::cast_slice(evaluated.as_slice::<bf16>()).to_vec(),
        ),
        MlxDtype::Float32 => (
            SafeDtype::F32,
            bytemuck::cast_slice(evaluated.as_slice::<f32>()).to_vec(),
        ),
        MlxDtype::Float64 => (
            SafeDtype::F64,
            bytemuck::cast_slice(evaluated.as_slice::<f64>()).to_vec(),
        ),
        MlxDtype::Complex64 => {
            anyhow::bail!("complex MLX tensors cannot be saved as stage weights")
        }
    };
    let shape = array
        .shape()
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(OwnedTensor { dtype, shape, data })
}

fn append_arrays(
    arrays: Vec<(String, OwnedTensor)>,
    shard_size_bytes: usize,
    output_dir: &Path,
    source_file_bytes: u64,
    state: &mut BuildState,
) -> Result<()> {
    let incoming_bytes = arrays
        .iter()
        .map(|(_, tensor)| tensor.data.len())
        .sum::<usize>();
    if !state.pending.tensors.is_empty()
        && state.pending.bytes.saturating_add(incoming_bytes) > shard_size_bytes
    {
        flush_shard(output_dir, state)?;
        state.observe_source_file(source_file_bytes);
    }
    for (name, tensor) in arrays {
        state.output_tensor_bytes = state
            .output_tensor_bytes
            .checked_add(u64::try_from(tensor.data.len())?)
            .context("derived output tensor byte count overflow")?;
        state.pending.insert(name, tensor)?;
    }
    Ok(())
}

fn flush_shard(output_dir: &Path, state: &mut BuildState) -> Result<()> {
    let index = state.temporary_shards.len();
    let path = output_dir.join(format!(".derived-{index:05}.safetensors"));
    serialize_to_file(
        state
            .pending
            .tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor)),
        None,
        &path,
    )?;
    state.written_output_file_bytes = state
        .written_output_file_bytes
        .checked_add(fs::metadata(&path)?.len())
        .context("derived output file byte count overflow")?;
    for name in state.pending.tensors.keys() {
        state.locations.insert(name.clone(), index);
    }
    state.pending.tensors.clear();
    state.pending.bytes = 0;
    state.temporary_shards.push(path);
    Ok(())
}

fn finalize_shards(output_dir: &Path, state: &BuildState) -> Result<Vec<PathBuf>> {
    if state.temporary_shards.len() == 1 {
        let output = output_dir.join("model.safetensors");
        fs::rename(&state.temporary_shards[0], &output)?;
        return Ok(vec![output]);
    }
    let count = state.temporary_shards.len();
    let mut outputs = Vec::with_capacity(count);
    for (index, temporary) in state.temporary_shards.iter().enumerate() {
        let output = output_dir.join(format!("model-{:05}-of-{count:05}.safetensors", index + 1));
        fs::rename(temporary, &output)?;
        outputs.push(output);
    }
    let weight_map = state
        .locations
        .iter()
        .map(|(name, index)| {
            (
                name.clone(),
                Value::String(
                    outputs[*index]
                        .file_name()
                        .expect("derived shard has a file name")
                        .to_string_lossy()
                        .into_owned(),
                ),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    write_json(
        output_dir.join("model.safetensors.index.json"),
        &json!({
            "metadata": { "total_size": state.output_tensor_bytes },
            "weight_map": weight_map,
        }),
    )?;
    Ok(outputs)
}

fn write_quantized_config(path: PathBuf, source: &Value, quantization: &Value) -> Result<()> {
    let mut config = source.clone();
    let object = config
        .as_object_mut()
        .context("source config.json must contain an object")?;
    object.insert("quantization".to_string(), quantization.clone());
    object.insert("quantization_config".to_string(), quantization.clone());
    write_json(path, &config)
}

fn ensure_unquantized_source_config(config: &Value) -> Result<()> {
    let object = config
        .as_object()
        .context("source config.json must contain an object")?;
    ensure!(
        matches!(
            object.get("model_type").and_then(Value::as_str),
            Some("llama" | "nemotron_h")
        ),
        "derived MLX stages support model_type=llama or nemotron_h"
    );
    for key in ["quantization", "quantization_config", "compression_config"] {
        ensure!(
            object.get(key).is_none_or(Value::is_null),
            "source checkpoint declares {key}; implicit dequantization/requantization is unsupported"
        );
    }
    Ok(())
}

fn parse_source_config(source: &[u8]) -> Result<Value> {
    match serde_json::from_slice(source) {
        Ok(config) => Ok(config),
        Err(strict_error) => {
            let normalized = normalize_nonfinite_json_tokens(source)?;
            serde_json::from_slice(&normalized).with_context(|| {
                format!(
                    "parse source config.json after normalizing non-finite values; strict JSON error: {strict_error}"
                )
            })
        }
    }
}

fn normalize_nonfinite_json_tokens(source: &[u8]) -> Result<Vec<u8>> {
    let source = std::str::from_utf8(source).context("source config.json is not UTF-8")?;
    let bytes = source.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            output.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
            output.push(byte);
            index += 1;
            continue;
        }
        let replacement = ["-Infinity", "+Infinity", "Infinity", "NaN"]
            .into_iter()
            .find(|token| source[index..].starts_with(token));
        if let Some(token) = replacement {
            let end = index + token.len();
            let leading_boundary = index == 0
                || bytes[index - 1].is_ascii_whitespace()
                || matches!(bytes[index - 1], b':' | b',' | b'[');
            let trailing_boundary = bytes.get(end).is_none_or(|next| {
                next.is_ascii_whitespace() || matches!(next, b',' | b']' | b'}')
            });
            if leading_boundary && trailing_boundary {
                output.extend_from_slice(b"null");
                index = end;
                continue;
            }
        }
        output.push(byte);
        index += 1;
    }
    Ok(output)
}

pub(super) fn prepare_derivation_recipe(
    materializer: &SafetensorsStageMaterializer,
    source: SafetensorsStageRequest,
    quantization: MlxWeightQuantization,
    shard_size_bytes: usize,
    control: &MlxDerivationControl,
) -> Result<String> {
    control.ensure_active()?;
    let visit = materializer.prepare_tensor_visit(source)?;
    control.ensure_active()?;
    control.verify_checkpoint(visit.checkpoint_sha256())?;
    let source_config = parse_source_config(visit.config())?;
    ensure_unquantized_source_config(&source_config)?;
    let quantization = quantization.safemlx()?;
    TensorDerivation::new(
        &source_config,
        visit.plan().layer_start,
        visit.plan().layer_end,
        quantization,
    )?;
    let quantization = serde_json::to_value(quantization)?;
    let plan_bytes = serde_json::to_vec(visit.plan())?;
    let plan_sha256 = sha256_bytes(&plan_bytes);
    derived_identity(visit.plan(), &plan_sha256, &quantization, shard_size_bytes)
}

fn derived_identity(
    plan: &model_hf::safetensors_stage::SafetensorsStagePlan,
    plan_sha256: &str,
    quantization: &Value,
    shard_size_bytes: usize,
) -> Result<String> {
    let bytes = serde_json::to_vec(&(
        DERIVED_STAGE_IMPLEMENTATION,
        DERIVED_STAGE_SCHEMA_VERSION,
        SAFEMLX_REVISION,
        &plan.checkpoint_sha256,
        plan_sha256,
        plan.layer_start,
        plan.layer_end,
        &plan.include_prefixes,
        quantization,
        shard_size_bytes,
    ))?;
    Ok(sha256_bytes(&bytes))
}

fn derived_shard(path: &Path) -> Result<MlxDerivedStageShard> {
    Ok(MlxDerivedStageShard {
        file: path
            .file_name()
            .context("derived shard has no file name")?
            .to_string_lossy()
            .into_owned(),
        file_bytes: fs::metadata(path)?.len(),
        sha256: sha256_file(path)?,
    })
}

pub(super) fn output_content_sha256(path: &Path) -> Result<String> {
    let mut files = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_file() && entry.file_name() != REPORT_FILE {
            files.push(entry);
        }
    }
    files.sort_by_key(|entry| entry.file_name());
    let mut hasher = Sha256::new();
    hasher.update(b"mesh-mlx-derived-output-v1");
    for entry in files {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        hasher.update(u64::try_from(name.len())?.to_le_bytes());
        hasher.update(name.as_bytes());
        let mut reader = BufReader::new(File::open(entry.path())?);
        let file_bytes = reader.get_ref().metadata()?.len();
        hasher.update(file_bytes.to_le_bytes());
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn directory_file_bytes(path: &Path) -> Result<u64> {
    fs::read_dir(path)?.try_fold(0_u64, |total, entry| {
        let entry = entry?;
        let bytes = if entry.file_type()?.is_file() {
            entry.metadata()?.len()
        } else {
            0
        };
        total
            .checked_add(bytes)
            .context("derived directory byte count overflow")
    })
}

pub(super) fn artifact_file_bytes(path: &Path) -> Result<u64> {
    fs::read_dir(path)?.try_fold(0_u64, |total, entry| {
        let entry = entry?;
        let bytes = if entry.file_type()?.is_file() && entry.file_name() != REPORT_FILE {
            entry.metadata()?.len()
        } else {
            0
        };
        total
            .checked_add(bytes)
            .context("derived artifact byte count overflow")
    })
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(path, bytes)?;
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

struct TemporaryOutput {
    path: Option<PathBuf>,
    lock_path: PathBuf,
    _lock: File,
}

impl TemporaryOutput {
    fn create(destination: &Path) -> Result<Self> {
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let name = destination
            .file_name()
            .context("derived output path has no file name")?
            .to_string_lossy();
        remove_abandoned_outputs(parent, &name)?;
        for _ in 0..100 {
            let sequence = DERIVED_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let base = format!(".{name}.{}.{}", std::process::id(), sequence);
            let path = parent.join(format!("{base}.partial"));
            let lock_path = parent.join(format!("{base}.lock"));
            let lock = open_locked(&lock_path, false)?.expect("blocking lock is acquired");
            match fs::create_dir(&path) {
                Ok(()) => {
                    return Ok(Self {
                        path: Some(path),
                        lock_path,
                        _lock: lock,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    drop(lock);
                    let _ = fs::remove_file(lock_path);
                }
                Err(error) => {
                    drop(lock);
                    let _ = fs::remove_file(lock_path);
                    return Err(error).context("create derived stage temporary output");
                }
            }
        }
        anyhow::bail!("could not allocate a unique derived stage temporary output")
    }

    fn path(&self) -> &Path {
        self.path.as_deref().expect("temporary output is active")
    }

    fn publish(mut self, destination: &Path) -> Result<()> {
        let path = self.path.as_ref().expect("temporary output is active");
        fs::rename(path, destination).context("publish derived stage output")?;
        self.path = None;
        let _ = fs::remove_file(&self.lock_path);
        Ok(())
    }
}

fn remove_abandoned_outputs_for_destination(destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = destination
        .file_name()
        .context("derived output path has no file name")?
        .to_string_lossy();
    remove_abandoned_outputs(parent, &name)
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            let _ = fs::remove_dir_all(path);
        }
        let _ = fs::remove_file(&self.lock_path);
    }
}

fn remove_abandoned_outputs(parent: &Path, destination_name: &str) -> Result<()> {
    let prefix = format!(".{destination_name}.");
    for entry in fs::read_dir(parent)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(base) = name.strip_suffix(".partial") else {
            continue;
        };
        if !base.starts_with(&prefix) || !entry.file_type()?.is_dir() {
            continue;
        }
        let lock_path = parent.join(format!("{base}.lock"));
        let Some(lock) = open_locked(&lock_path, true)? else {
            continue;
        };
        fs::remove_dir_all(entry.path())?;
        drop(lock);
        fs::remove_file(lock_path)?;
    }
    Ok(())
}

pub(super) fn open_locked(path: &Path, nonblocking: bool) -> Result<Option<File>> {
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;

    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)?;
    let operation = libc::LOCK_EX | if nonblocking { libc::LOCK_NB } else { 0 };
    // SAFETY: `file` owns a valid descriptor for the duration of this call.
    let result = unsafe { libc::flock(file.as_raw_fd(), operation) };
    if result == 0 {
        return Ok(Some(file));
    }
    let error = std::io::Error::last_os_error();
    if nonblocking
        && error
            .raw_os_error()
            .is_some_and(|code| code == libc::EWOULDBLOCK || code == libc::EAGAIN)
    {
        Ok(None)
    } else {
        Err(error).context("lock derived stage temporary output")
    }
}

#[cfg(test)]
mod tests {
    use safemlx_lm::quantization::AffineQuantization;

    use super::*;

    #[test]
    fn derivation_control_rejects_cancellation_and_wrong_checkpoint() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let control = MlxDerivationControl::new(
            Some("expected-checkpoint".to_string()),
            Some(Arc::clone(&cancelled)),
        );

        assert!(control.ensure_active().is_ok());
        assert!(control.verify_checkpoint("expected-checkpoint").is_ok());
        assert!(control.verify_checkpoint("other-checkpoint").is_err());

        cancelled.store(true, Ordering::Release);
        assert!(control.ensure_active().is_err());
    }

    #[test]
    fn quantized_config_preserves_source_and_adds_both_metadata_keys() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.json");
        let quantization = serde_json::to_value(WeightQuantization::Affine(
            AffineQuantization::new(64, 4).unwrap(),
        ))
        .unwrap();

        write_quantized_config(path.clone(), &json!({"model_type": "llama"}), &quantization)
            .unwrap();

        let config: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(config["model_type"], "llama");
        assert_eq!(config["quantization"], quantization);
        assert_eq!(config["quantization_config"], quantization);
    }

    #[test]
    fn rejects_prequantized_source_metadata_and_ineligible_weights() {
        let error = ensure_unquantized_source_config(&json!({
            "model_type": "llama",
            "quantization_config": {"quant_method": "fp8"}
        }))
        .unwrap_err();
        assert!(error.to_string().contains("implicit dequantization"));

        let error =
            ensure_unquantized_source_config(&json!({"model_type": "unknown"})).unwrap_err();
        assert!(error.to_string().contains("llama or nemotron_h"));

        let error = TensorDerivation::new(
            &json!({
                "model_type": "nemotron_h",
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "hybrid_override_pattern": "E",
                "n_routed_experts": 2,
                "moe_intermediate_size": 32
            }),
            0,
            1,
            WeightQuantization::MxFp4,
        )
        .err()
        .expect("Nemotron-H MXFP4 should fail before tensor payloads");
        assert!(error.to_string().contains("require affine"));

        let quantization: WeightQuantization = AffineQuantization::new(64, 4).unwrap().into();
        let packed = Array::from_slice(&vec![0_u32; 128], &[2, 64]);
        assert!(
            should_quantize_source_weight("model.layers.0.q_proj.weight", &packed, quantization)
                .unwrap_err()
                .to_string()
                .contains("already packed")
        );
        let incompatible = Array::from_slice(&vec![0_f32; 126], &[2, 63]);
        assert!(
            should_quantize_source_weight(
                "model.layers.0.q_proj.weight",
                &incompatible,
                quantization
            )
            .unwrap_err()
            .to_string()
            .contains("incompatible")
        );
        let expert_bank = Array::from_slice(&vec![0_f32; 256], &[2, 2, 64]);
        assert!(
            should_quantize_source_weight(
                "model.layers.0.experts.weight",
                &expert_bank,
                quantization
            )
            .unwrap_err()
            .to_string()
            .contains("rank-2 matrix")
        );
    }

    #[test]
    fn normalizes_nonfinite_config_values_but_not_strings() {
        let config = parse_source_config(
            br#"{
                "model_type":"nemotron_h",
                "label":"Infinity and NaN",
                "limits":[Infinity,-Infinity,+Infinity,NaN]
            }"#,
        )
        .unwrap();

        assert_eq!(config["label"], "Infinity and NaN");
        assert_eq!(config["limits"], json!([null, null, null, null]));
        assert!(parse_source_config(br#"{"bad":123Infinity}"#).is_err());
    }

    #[test]
    fn pending_shard_rejects_canonical_name_collisions() {
        let tensor = || OwnedTensor {
            dtype: SafeDtype::F32,
            shape: vec![1],
            data: 1_f32.to_le_bytes().to_vec(),
        };
        let mut pending = PendingShard::new();

        pending
            .insert("model.weight".to_string(), tensor())
            .unwrap();
        let error = pending
            .insert("model.weight".to_string(), tensor())
            .unwrap_err();

        assert!(error.to_string().contains("more than once"));
    }

    #[test]
    fn pure_rust_safetensors_output_preserves_bfloat16_and_packed_u32_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        let source = [bf16::from_f32(1.5), bf16::from_f32(-2.25)];
        let array = Array::from_slice(&source, &[2]);
        let tensor = owned_tensor(&array).unwrap();

        serialize_to_file([("weight", &tensor)], None, &path).unwrap();

        let bytes = fs::read(path).unwrap();
        let saved = SafeTensors::deserialize(&bytes).unwrap();
        let saved = saved.tensor("weight").unwrap();
        assert_eq!(saved.dtype(), SafeDtype::BF16);
        assert_eq!(saved.shape(), &[2]);
        assert_eq!(saved.data(), bytemuck::cast_slice::<bf16, u8>(&source));

        let packed_path = directory.path().join("packed.safetensors");
        let packed_source = [0x0123_4567_u32, 0x89ab_cdef_u32];
        let packed_array = Array::from_slice(&packed_source, &[1, 2]);
        let packed_tensor = owned_tensor(&packed_array).unwrap();
        serialize_to_file([("weight", &packed_tensor)], None, &packed_path).unwrap();

        let packed_bytes = fs::read(packed_path).unwrap();
        let packed_saved = SafeTensors::deserialize(&packed_bytes).unwrap();
        let packed_saved = packed_saved.tensor("weight").unwrap();
        assert_eq!(packed_saved.dtype(), SafeDtype::U32);
        assert_eq!(packed_saved.shape(), &[1, 2]);
        assert_eq!(
            packed_saved.data(),
            bytemuck::cast_slice::<u32, u8>(&packed_source)
        );
    }

    #[test]
    fn removes_abandoned_output_but_preserves_concurrent_output() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("stage");
        let abandoned = directory.path().join(".stage.999.0.partial");
        fs::create_dir(&abandoned).unwrap();
        fs::write(abandoned.join("large.safetensors"), b"stale").unwrap();

        let first = TemporaryOutput::create(&destination).unwrap();
        let first_path = first.path().to_path_buf();
        let second = TemporaryOutput::create(&destination).unwrap();

        assert!(!abandoned.exists());
        assert!(first_path.is_dir());
        drop(second);
        assert!(first_path.is_dir());
        drop(first);
        assert!(fs::read_dir(directory.path()).unwrap().next().is_none());
    }

    #[test]
    fn removes_abandoned_output_even_when_destination_exists() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("stage");
        fs::create_dir(&destination).unwrap();
        let abandoned = directory.path().join(".stage.999.0.partial");
        fs::create_dir(&abandoned).unwrap();
        fs::write(abandoned.join("large.safetensors"), b"stale").unwrap();

        remove_abandoned_outputs_for_destination(&destination).unwrap();

        assert!(destination.is_dir());
        assert!(!abandoned.exists());
    }
}
