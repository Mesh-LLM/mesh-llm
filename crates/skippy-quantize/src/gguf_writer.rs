use std::collections::{BTreeMap, btree_map::Entry};
use std::fs::{self, File};
use std::io::{Seek, Write};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::float_convert::{FloatDType, convert_float_chunk, target_dtype_for_tensor};
pub(crate) use crate::gguf_metadata::GgufKv;
use crate::gguf_metadata::write_kv;
#[cfg(test)]
use crate::gguf_metadata::{
    GGUF_TYPE_ARRAY, GGUF_TYPE_BOOL, GGUF_TYPE_FLOAT32, GGUF_TYPE_INT32, GGUF_TYPE_STRING,
    GGUF_TYPE_UINT16, GGUF_TYPE_UINT32, GGUF_TYPE_UINT64,
};
use crate::hf_checkpoint::{SafetensorFile, SafetensorTensorInfo, open_safetensor_files};
use crate::tensor_map::{
    TensorNameMap, hf_layer_id, inkling_mtp_depth, is_inkling_fused_w13, is_mtp_source_tensor,
    is_shared_mtp_context_tensor,
};
use crate::types::ConvertOutputType;

mod glm_dsa;

use glm_dsa::{
    GlmDsaKvBSplitMode, TensorTransform, enrich_glm_dsa_indexshare_metadata, glm_dsa_kv_b_layer,
    glm_dsa_kv_b_split_mode, stream_transformed_segment,
};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGUF_ALIGNMENT: u64 = 32;
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_BF16: u32 = 30;
#[derive(Debug, Clone)]
pub(crate) struct RawGgufWriteOptions {
    pub(crate) buffer_size: usize,
    pub(crate) metadata: Option<Vec<GgufKv>>,
    pub(crate) tensor_name_map: TensorNameMap,
    pub(crate) split: Option<GgufSplit>,
    pub(crate) output_type: Option<ConvertOutputType>,
    pub(crate) tensor_selection: TensorSelection,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum TensorSelection {
    #[default]
    All,
    ExcludeMtp {
        layer_start: u32,
    },
    MtpOnly {
        layer_start: u32,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GgufSplit {
    pub(crate) split_index: u32,
    pub(crate) split_count: u32,
}

pub(crate) fn write_raw_safetensors_gguf(
    source: &Path,
    output: &Path,
    options: RawGgufWriteOptions,
) -> Result<()> {
    let PreparedGgufWrite {
        files,
        tensors,
        metadata,
    } = prepare_raw_safetensors_gguf(source, &options)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut writer =
        File::create(output).with_context(|| format!("create {}", output.display()))?;
    write_header_and_tensor_table(&mut writer, &metadata, &tensors)?;
    stream_tensor_data(&mut writer, &files, &tensors, options.buffer_size)
}

pub(crate) fn validate_raw_safetensors_gguf(
    source: &Path,
    options: RawGgufWriteOptions,
) -> Result<RawGgufValidation> {
    let PreparedGgufWrite {
        tensors, metadata, ..
    } = prepare_raw_safetensors_gguf(source, &options)?;
    Ok(RawGgufValidation {
        selected_tensor_count: tensors.len(),
        selected_tensor_bytes: tensors.iter().map(|tensor| tensor.byte_len).sum(),
        metadata_count: metadata.len(),
        output_type: options.output_type.map(|kind| kind.as_arg().to_string()),
    })
}

pub(crate) fn recommended_raw_safetensors_gguf_split_count(
    source: &Path,
    mut options: RawGgufWriteOptions,
    max_tensor_bytes: u64,
) -> Result<u32> {
    ensure!(
        max_tensor_bytes > 0,
        "split maximum tensor bytes must be greater than zero"
    );
    options.split = None;
    let PreparedGgufWrite { tensors, .. } = prepare_raw_safetensors_gguf(source, &options)?;
    let largest_tensor_bytes = tensors
        .iter()
        .map(|tensor| tensor.byte_len)
        .max()
        .unwrap_or_default();
    ensure!(
        largest_tensor_bytes <= max_tensor_bytes,
        "largest selected tensor is {largest_tensor_bytes} bytes, exceeding split maximum {max_tensor_bytes} bytes"
    );
    let total_tensor_bytes = tensors
        .iter()
        .try_fold(0_u64, |total, tensor| total.checked_add(tensor.byte_len));
    let total_tensor_bytes = total_tensor_bytes.context("selected tensor byte total overflow")?;
    let minimum_count = total_tensor_bytes
        .div_ceil(max_tensor_bytes)
        .max(1)
        .min(tensors.len() as u64);
    let minimum_count = u32::try_from(minimum_count).context("split count does not fit u32")?;
    let maximum_count = u32::try_from(tensors.len()).context("tensor count does not fit u32")?;

    for split_count in minimum_count..=maximum_count {
        let split = GgufSplit {
            split_index: 1,
            split_count,
        };
        let boundaries = byte_balanced_split_boundaries(&tensors, split)?;
        let every_split_fits = boundaries.windows(2).all(|range| {
            tensors[range[0]..range[1]]
                .iter()
                .map(|tensor| tensor.byte_len)
                .sum::<u64>()
                <= max_tensor_bytes
        });
        if every_split_fits {
            return Ok(split_count);
        }
    }

    anyhow::bail!("could not partition selected tensors within the split maximum")
}

struct PreparedGgufWrite {
    files: Vec<SafetensorFile>,
    tensors: Vec<TensorSource>,
    metadata: Vec<GgufKv>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RawGgufValidation {
    pub(crate) selected_tensor_count: usize,
    pub(crate) selected_tensor_bytes: u64,
    pub(crate) metadata_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) output_type: Option<String>,
}

fn prepare_raw_safetensors_gguf(
    source: &Path,
    options: &RawGgufWriteOptions,
) -> Result<PreparedGgufWrite> {
    ensure!(
        options.buffer_size > 0,
        "buffer_size must be greater than zero"
    );
    let files = open_safetensor_files(source)?;
    ensure!(
        !files.is_empty(),
        "no safetensors files found under {}",
        source.display()
    );
    let metadata_seed = options.metadata.clone();
    let glm_dsa_kv_b_split = glm_dsa_kv_b_split_mode(metadata_seed.as_deref())?;
    let tensors = collect_tensor_sources(
        &files,
        options.tensor_name_map,
        options.output_type,
        options.tensor_selection,
        glm_dsa_kv_b_split,
    )?;
    ensure!(
        !tensors.is_empty(),
        "no tensors found under {}",
        source.display()
    );
    let total_tensor_count = tensors.len();
    let mut metadata = metadata_seed.unwrap_or_else(|| raw_metadata(source, total_tensor_count));
    enrich_glm_dsa_indexshare_metadata(&mut metadata, &tensors)?;
    let mut tensors = select_split_tensors(tensors, options.split)?;
    assign_gguf_offsets(&mut tensors)?;
    let metadata = split_metadata(metadata, options.split, total_tensor_count)?;
    Ok(PreparedGgufWrite {
        files,
        tensors,
        metadata,
    })
}

fn select_split_tensors(
    tensors: Vec<TensorSource>,
    split: Option<GgufSplit>,
) -> Result<Vec<TensorSource>> {
    let Some(split) = split else {
        return Ok(tensors);
    };
    split.validate()?;
    let total_tensors = tensors.len();
    ensure!(
        usize::try_from(split.split_count).is_ok_and(|count| count <= total_tensors),
        "split_count {} cannot exceed tensor count {}",
        split.split_count,
        total_tensors
    );
    let split_index =
        usize::try_from(split.split_index).context("split_index does not fit usize")?;
    let boundaries = byte_balanced_split_boundaries(&tensors, split)?;
    let start = boundaries[split_index - 1];
    let end = boundaries[split_index];
    ensure!(
        start < end,
        "split {} of {} would contain no tensors",
        split.split_index,
        split.split_count
    );
    Ok(tensors
        .into_iter()
        .enumerate()
        .filter_map(|(index, tensor)| (start <= index && index < end).then_some(tensor))
        .collect())
}

fn byte_balanced_split_boundaries(
    tensors: &[TensorSource],
    split: GgufSplit,
) -> Result<Vec<usize>> {
    split.validate()?;
    let split_count =
        usize::try_from(split.split_count).context("split_count does not fit usize")?;
    ensure!(
        split_count <= tensors.len(),
        "split_count {} cannot exceed tensor count {}",
        split.split_count,
        tensors.len()
    );
    let total_bytes = tensors
        .iter()
        .try_fold(0_u128, |acc, tensor| {
            acc.checked_add(tensor.byte_len as u128)
        })
        .context("split tensor byte total overflow")?;
    let mut boundaries = vec![0_usize];
    let mut accumulated = 0_u128;
    for (index, tensor) in tensors.iter().enumerate() {
        accumulated = accumulated
            .checked_add(tensor.byte_len as u128)
            .context("split tensor byte total overflow")?;
        let remaining_tensors = tensors.len() - (index + 1);
        let remaining_splits = split_count - boundaries.len();
        if boundaries.len() < split_count && remaining_tensors >= remaining_splits {
            let target = total_bytes
                .checked_mul(boundaries.len() as u128)
                .context("split target byte overflow")?
                / split_count as u128;
            if accumulated >= target {
                boundaries.push(index + 1);
            }
        }
    }
    while boundaries.len() < split_count {
        let next = boundaries.last().copied().unwrap_or(0) + 1;
        boundaries.push(next);
    }
    boundaries.push(tensors.len());
    Ok(boundaries)
}

fn assign_gguf_offsets(tensors: &mut [TensorSource]) -> Result<()> {
    let mut offset = 0_u64;
    for tensor in tensors {
        offset = align_to(offset, GGUF_ALIGNMENT);
        tensor.gguf_offset = offset;
        offset = offset
            .checked_add(tensor.byte_len)
            .with_context(|| format!("GGUF data offset overflow after {}", tensor.name))?;
    }
    Ok(())
}

fn split_metadata(
    mut metadata: Vec<GgufKv>,
    split: Option<GgufSplit>,
    total_tensor_count: usize,
) -> Result<Vec<GgufKv>> {
    let Some(split) = split else {
        return Ok(metadata);
    };
    split.validate()?;
    metadata.push(GgufKv::u16(
        "split.no",
        u16::try_from(split.split_index - 1).context("split index does not fit uint16")?,
    ));
    metadata.push(GgufKv::u16(
        "split.count",
        u16::try_from(split.split_count).context("split count does not fit uint16")?,
    ));
    metadata.push(GgufKv::i32(
        "split.tensors.count",
        i32::try_from(total_tensor_count).context("tensor count does not fit int32")?,
    ));
    Ok(metadata)
}

impl GgufSplit {
    fn validate(self) -> Result<()> {
        ensure!(
            self.split_count > 0,
            "split_count must be greater than zero"
        );
        ensure!(
            self.split_index > 0,
            "split_index is 1-based and cannot be zero"
        );
        ensure!(
            self.split_index <= self.split_count,
            "split_index {} exceeds split_count {}",
            self.split_index,
            self.split_count
        );
        ensure!(
            u16::try_from(self.split_count).is_ok(),
            "split_count {} exceeds GGUF uint16 split metadata",
            self.split_count
        );
        Ok(())
    }
}

fn collect_tensor_sources(
    files: &[SafetensorFile],
    tensor_name_map: TensorNameMap,
    output_type: Option<ConvertOutputType>,
    tensor_selection: TensorSelection,
    glm_dsa_kv_b_split: GlmDsaKvBSplitMode,
) -> Result<Vec<TensorSource>> {
    let mut tensors = Vec::new();
    let mut expert_groups = BTreeMap::<ExpertGroupKey, ExpertGroup>::new();
    for (file_index, file) in files.iter().enumerate() {
        for tensor in file.tensors().values() {
            if !tensor_selection.includes(tensor.name())? {
                continue;
            }
            if is_inkling_fused_w13(tensor.name()) {
                tensors.extend(inkling_w13_tensor_sources(
                    file_index,
                    tensor,
                    tensor_name_map,
                    output_type,
                )?);
                continue;
            }
            if matches!(
                tensor_name_map,
                TensorNameMap::HfToGguf | TensorNameMap::HfToGgufWithMtp { .. }
            ) && let Some(expert) = ExpertSourceTensor::parse(tensor.name())?
            {
                match expert_groups.entry(expert.group_key()) {
                    Entry::Vacant(entry) => {
                        entry
                            .insert(ExpertGroup::new(expert.group_key(), tensor, output_type)?)
                            .push(file_index, tensor, expert.expert_id)?;
                    }
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().push(file_index, tensor, expert.expert_id)?;
                    }
                }
                continue;
            }
            if let Some(layer) = glm_dsa_kv_b_layer(tensor.name())? {
                match glm_dsa_kv_b_split {
                    GlmDsaKvBSplitMode::Config(split) => {
                        tensors.extend(TensorSource::from_glm_dsa_kv_b_split(
                            file_index,
                            tensor,
                            layer,
                            split,
                            output_type,
                        )?);
                        continue;
                    }
                    GlmDsaKvBSplitMode::MissingMetadata => {
                        anyhow::bail!(
                            "GLM-DSA tensor {} requires attention head/value/rope/kv_lora metadata for kv_b split",
                            tensor.name()
                        );
                    }
                    GlmDsaKvBSplitMode::Disabled => {}
                }
            }
            tensors.push(TensorSource::from_safetensor(
                file_index,
                tensor,
                tensor_name_map,
                output_type,
            )?);
        }
    }
    for group in expert_groups.into_values() {
        tensors.push(group.into_tensor_source()?);
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(tensors)
}

impl TensorSelection {
    fn includes(self, name: &str) -> Result<bool> {
        let is_mtp = match self {
            Self::All => return Ok(true),
            Self::ExcludeMtp { layer_start } | Self::MtpOnly { layer_start } => {
                is_mtp_source_tensor(name)
                    || hf_layer_id(name)?.is_some_and(|layer| layer >= layer_start)
            }
        };
        match self {
            Self::All => Ok(true),
            Self::ExcludeMtp { .. } => Ok(!is_mtp),
            Self::MtpOnly { .. } => Ok(is_mtp || is_shared_mtp_context_tensor(name)),
        }
    }
}

struct TensorSource {
    segments: Vec<TensorSegment>,
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    byte_len: u64,
    gguf_offset: u64,
}

impl TensorSource {
    fn from_safetensor(
        file_index: usize,
        tensor: &SafetensorTensorInfo,
        tensor_name_map: TensorNameMap,
        output_type: Option<ConvertOutputType>,
    ) -> Result<Self> {
        let source_dtype = FloatDType::from_safetensor(tensor.dtype()).with_context(|| {
            format!("unsupported dtype {} for {}", tensor.dtype(), tensor.name())
        })?;
        let name = tensor_name_map.map_tensor_name(tensor.name())?;
        let target_dtype =
            target_dtype_for_mapped_tensor(source_dtype, output_type, tensor.shape(), &name)?;
        let element_count = tensor_element_count(tensor)?;
        let dims = mapped_tensor_dims(tensor.shape(), &name)?;
        Ok(Self {
            segments: vec![TensorSegment {
                file_index,
                source_name: tensor.name().to_string(),
                source_dtype,
                target_dtype,
                element_count,
                source_byte_len: tensor.byte_len(),
                target_byte_len: tensor_byte_len(element_count, target_dtype)?,
                transform: TensorTransform::Identity,
            }],
            name,
            dims,
            ggml_type: ggml_type_for_dtype(target_dtype),
            byte_len: tensor_byte_len(element_count, target_dtype)?,
            gguf_offset: 0,
        })
    }
}

fn target_dtype_for_mapped_tensor(
    source_dtype: FloatDType,
    output_type: Option<ConvertOutputType>,
    shape: &[u64],
    mapped_name: &str,
) -> Result<FloatDType> {
    if mapped_name.ends_with("attn_rel_proj.weight") || mapped_name.contains(".shortconv_") {
        return Ok(FloatDType::F32);
    }
    target_dtype_for_tensor(source_dtype, output_type, shape)
}

fn mapped_tensor_dims(shape: &[u64], mapped_name: &str) -> Result<Vec<u64>> {
    if mapped_name.contains(".shortconv_") {
        ensure!(
            shape.len() == 3 && shape[1] == 1,
            "Inkling shortconv tensor {mapped_name} must have shape [channels, 1, kernel], got {shape:?}"
        );
        return Ok(vec![shape[2], shape[0]]);
    }
    Ok(shape.iter().rev().copied().collect())
}

fn inkling_w13_tensor_sources(
    file_index: usize,
    tensor: &SafetensorTensorInfo,
    tensor_name_map: TensorNameMap,
    output_type: Option<ConvertOutputType>,
) -> Result<Vec<TensorSource>> {
    let layer = if let Some(depth) = inkling_mtp_depth(tensor.name())? {
        let TensorNameMap::HfToGgufWithMtp { layer_start } = tensor_name_map else {
            anyhow::bail!("Inkling MTP conversion requires an MTP-aware tensor name map");
        };
        layer_start
            .checked_add(depth)
            .context("Inkling MTP layer id overflow")?
    } else {
        ensure!(
            matches!(
                tensor_name_map,
                TensorNameMap::HfToGguf | TensorNameMap::HfToGgufWithMtp { .. }
            ),
            "Inkling fused w13 conversion requires an HF tensor name map"
        );
        hf_layer_id(tensor.name())?
            .with_context(|| format!("missing Inkling layer id in {}", tensor.name()))?
    };
    ensure!(
        tensor.shape().len() == 2,
        "Inkling MTP fused w13 tensor {} must be rank 2, got {:?}",
        tensor.name(),
        tensor.shape()
    );
    ensure!(
        tensor.shape()[0].is_multiple_of(2),
        "Inkling MTP fused w13 tensor {} must have an even row count",
        tensor.name()
    );
    let source_dtype = FloatDType::from_safetensor(tensor.dtype())
        .with_context(|| format!("unsupported dtype {} for {}", tensor.dtype(), tensor.name()))?;
    let output_shape = [tensor.shape()[0] / 2, tensor.shape()[1]];
    let target_dtype = target_dtype_for_tensor(source_dtype, output_type, &output_shape)?;
    let element_count = output_shape[0]
        .checked_mul(output_shape[1])
        .context("Inkling MTP w13 output element count overflow")?;
    let target_byte_len = tensor_byte_len(element_count, target_dtype)?;
    let dims = output_shape.iter().rev().copied().collect::<Vec<_>>();
    Ok([("ffn_gate", 0_u64), ("ffn_up", 1_u64)]
        .into_iter()
        .map(|(projection, parity)| TensorSource {
            segments: vec![TensorSegment {
                file_index,
                source_name: tensor.name().to_string(),
                source_dtype,
                target_dtype,
                element_count,
                source_byte_len: tensor.byte_len(),
                target_byte_len,
                transform: TensorTransform::AlternatingRows {
                    parity,
                    row_elements: tensor.shape()[1],
                },
            }],
            name: format!("blk.{layer}.{projection}.weight"),
            dims: dims.clone(),
            ggml_type: ggml_type_for_dtype(target_dtype),
            byte_len: target_byte_len,
            gguf_offset: 0,
        })
        .collect())
}

struct TensorSegment {
    file_index: usize,
    source_name: String,
    source_dtype: FloatDType,
    target_dtype: FloatDType,
    element_count: u64,
    source_byte_len: u64,
    target_byte_len: u64,
    transform: TensorTransform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ExpertGroupKey {
    layer: u32,
    projection: ExpertProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ExpertProjection {
    Down,
    Gate,
    Up,
}

impl ExpertProjection {
    fn gguf_name(self, layer: u32) -> String {
        match self {
            Self::Down => format!("blk.{layer}.ffn_down_exps.weight"),
            Self::Gate => format!("blk.{layer}.ffn_gate_exps.weight"),
            Self::Up => format!("blk.{layer}.ffn_up_exps.weight"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ExpertSourceTensor {
    layer: u32,
    expert_id: u32,
    projection: ExpertProjection,
}

impl ExpertSourceTensor {
    fn parse(name: &str) -> Result<Option<Self>> {
        let Some(rest) = name.strip_prefix("model.layers.") else {
            return Ok(None);
        };
        let Some((layer, suffix)) = rest.split_once('.') else {
            return Ok(None);
        };
        let Some(expert_suffix) = suffix.strip_prefix("mlp.experts.") else {
            return Ok(None);
        };
        let Some((expert_id, projection_suffix)) = expert_suffix.split_once('.') else {
            return Ok(None);
        };
        let layer = layer
            .parse::<u32>()
            .with_context(|| format!("parse expert layer id in {name}"))?;
        let expert_id = expert_id
            .parse::<u32>()
            .with_context(|| format!("parse expert id in {name}"))?;
        let projection = match projection_suffix {
            "down_proj.weight" => ExpertProjection::Down,
            "gate_proj.weight" => ExpertProjection::Gate,
            "up_proj.weight" => ExpertProjection::Up,
            _ => return Ok(None),
        };
        Ok(Some(Self {
            layer,
            expert_id,
            projection,
        }))
    }

    fn group_key(self) -> ExpertGroupKey {
        ExpertGroupKey {
            layer: self.layer,
            projection: self.projection,
        }
    }
}

struct ExpertGroup {
    key: ExpertGroupKey,
    source_dtype: FloatDType,
    target_dtype: FloatDType,
    shape: Vec<u64>,
    source_byte_len_per_expert: u64,
    target_byte_len_per_expert: u64,
    experts: BTreeMap<u32, TensorSegment>,
}

impl ExpertGroup {
    fn new(
        key: ExpertGroupKey,
        tensor: &SafetensorTensorInfo,
        output_type: Option<ConvertOutputType>,
    ) -> Result<Self> {
        let source_dtype = FloatDType::from_safetensor(tensor.dtype()).with_context(|| {
            format!("unsupported dtype {} for {}", tensor.dtype(), tensor.name())
        })?;
        let target_dtype = target_dtype_for_tensor(source_dtype, output_type, tensor.shape())?;
        let element_count = tensor_element_count(tensor)?;
        Ok(Self {
            key,
            source_dtype,
            target_dtype,
            shape: tensor.shape().to_vec(),
            source_byte_len_per_expert: tensor.byte_len(),
            target_byte_len_per_expert: tensor_byte_len(element_count, target_dtype)?,
            experts: BTreeMap::new(),
        })
    }

    fn push(
        &mut self,
        file_index: usize,
        tensor: &SafetensorTensorInfo,
        expert_id: u32,
    ) -> Result<()> {
        ensure!(
            FloatDType::from_safetensor(tensor.dtype()) == Some(self.source_dtype),
            "expert tensor {} dtype {} does not match group dtype {:?}",
            tensor.name(),
            tensor.dtype(),
            self.source_dtype
        );
        ensure!(
            tensor.shape() == self.shape,
            "expert tensor {} shape {:?} does not match group shape {:?}",
            tensor.name(),
            tensor.shape(),
            self.shape
        );
        ensure!(
            tensor.byte_len() == self.source_byte_len_per_expert,
            "expert tensor {} byte length {} does not match group byte length {}",
            tensor.name(),
            tensor.byte_len(),
            self.source_byte_len_per_expert
        );
        let element_count = tensor_element_count(tensor)?;
        let previous = self.experts.insert(
            expert_id,
            TensorSegment {
                file_index,
                source_name: tensor.name().to_string(),
                source_dtype: self.source_dtype,
                target_dtype: self.target_dtype,
                element_count,
                source_byte_len: tensor.byte_len(),
                target_byte_len: tensor_byte_len(element_count, self.target_dtype)?,
                transform: TensorTransform::Identity,
            },
        );
        ensure!(
            previous.is_none(),
            "duplicate expert tensor id {expert_id} for {}",
            self.key.projection.gguf_name(self.key.layer)
        );
        Ok(())
    }

    fn into_tensor_source(self) -> Result<TensorSource> {
        ensure!(
            !self.experts.is_empty(),
            "expert group {} has no tensors",
            self.key.projection.gguf_name(self.key.layer)
        );
        for (expected, actual) in self.experts.keys().copied().enumerate() {
            ensure!(
                expected as u32 == actual,
                "expert group {} is missing expert id {}",
                self.key.projection.gguf_name(self.key.layer),
                expected
            );
        }
        let expert_count = self.experts.len() as u64;
        let mut dims = self.shape.iter().rev().copied().collect::<Vec<_>>();
        dims.push(expert_count);
        let byte_len = self
            .target_byte_len_per_expert
            .checked_mul(expert_count)
            .with_context(|| {
                format!(
                    "expert group {} byte length overflow",
                    self.key.projection.gguf_name(self.key.layer)
                )
            })?;
        Ok(TensorSource {
            segments: self.experts.into_values().collect(),
            name: self.key.projection.gguf_name(self.key.layer),
            dims,
            ggml_type: ggml_type_for_dtype(self.target_dtype),
            byte_len,
            gguf_offset: 0,
        })
    }
}

fn tensor_element_count(tensor: &SafetensorTensorInfo) -> Result<u64> {
    tensor.shape().iter().try_fold(1_u64, |acc, dim| {
        acc.checked_mul(*dim)
            .with_context(|| format!("tensor {} element count overflow", tensor.name()))
    })
}

fn tensor_byte_len(element_count: u64, dtype: FloatDType) -> Result<u64> {
    element_count
        .checked_mul(dtype.byte_size())
        .context("target tensor byte length overflow")
}

fn ggml_type_for_dtype(dtype: FloatDType) -> u32 {
    match dtype {
        FloatDType::F32 => GGML_TYPE_F32,
        FloatDType::F16 => GGML_TYPE_F16,
        FloatDType::Bf16 => GGML_TYPE_BF16,
    }
}

fn raw_metadata(source: &Path, tensor_count: usize) -> Vec<GgufKv> {
    vec![
        GgufKv::string("general.architecture", "raw-safetensors"),
        GgufKv::string(
            "general.name",
            source
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("checkpoint"),
        ),
        GgufKv::bool("skippy.convert.raw_safetensors", true),
        GgufKv::u64("skippy.convert.tensor_count", tensor_count as u64),
    ]
}

fn write_header_and_tensor_table<W: Write>(
    writer: &mut W,
    metadata: &[GgufKv],
    tensors: &[TensorSource],
) -> Result<()> {
    writer.write_all(GGUF_MAGIC)?;
    write_u32(writer, GGUF_VERSION)?;
    write_u64(writer, tensors.len() as u64)?;
    write_u64(writer, metadata.len() as u64)?;
    for kv in metadata {
        write_kv(writer, kv)?;
    }
    for tensor in tensors {
        write_string(writer, &tensor.name)?;
        write_u32(writer, tensor.dims.len() as u32)?;
        for dim in &tensor.dims {
            write_u64(writer, *dim)?;
        }
        write_u32(writer, tensor.ggml_type)?;
        write_u64(writer, tensor.gguf_offset)?;
    }
    Ok(())
}

fn stream_tensor_data(
    writer: &mut File,
    files: &[SafetensorFile],
    tensors: &[TensorSource],
    buffer_size: usize,
) -> Result<()> {
    pad_writer_to_alignment(writer, GGUF_ALIGNMENT)?;
    let data_start = writer.stream_position()?;
    for tensor in tensors {
        let expected_position = data_start + tensor.gguf_offset;
        pad_writer_to_position(writer, expected_position)?;
        let mut copied = 0_u64;
        for segment in &tensor.segments {
            let segment_copied =
                stream_segment(writer, &files[segment.file_index], segment, buffer_size)?;
            ensure!(
                segment_copied == segment.target_byte_len,
                "copied {} bytes for {}, expected {}",
                segment_copied,
                segment.source_name,
                segment.target_byte_len
            );
            copied += segment_copied;
        }
        ensure!(
            copied == tensor.byte_len,
            "copied {} bytes for {}, expected {}",
            copied,
            tensor.name,
            tensor.byte_len
        );
    }
    Ok(())
}

fn stream_segment(
    writer: &mut File,
    file: &SafetensorFile,
    segment: &TensorSegment,
    buffer_size: usize,
) -> Result<u64> {
    if let TensorTransform::AlternatingRows {
        parity,
        row_elements,
    } = segment.transform
    {
        return stream_alternating_rows(writer, file, segment, buffer_size, parity, row_elements);
    }
    if let Some(written) = stream_transformed_segment(writer, file, segment, buffer_size)? {
        return Ok(written);
    }

    if segment.source_dtype == segment.target_dtype {
        let copied = file.stream_tensor(&segment.source_name, writer, buffer_size)?;
        ensure!(
            copied == segment.source_byte_len,
            "read {} bytes for {}, expected {}",
            copied,
            segment.source_name,
            segment.source_byte_len
        );
        return Ok(copied);
    }

    let source_element_size = usize::try_from(segment.source_dtype.byte_size())
        .context("source dtype byte size does not fit usize")?;
    let chunk_size = aligned_chunk_size(buffer_size, source_element_size);
    let mut output_bytes = 0_u64;
    let mut source_bytes = 0_u64;
    file.stream_tensor_chunks(&segment.source_name, chunk_size, |chunk| {
        ensure!(
            chunk.len() % source_element_size == 0,
            "chunk for {} split an element boundary",
            segment.source_name
        );
        source_bytes += chunk.len() as u64;
        output_bytes +=
            convert_float_chunk(chunk, segment.source_dtype, segment.target_dtype, writer)?;
        Ok(())
    })?;
    ensure!(
        source_bytes == segment.source_byte_len,
        "read {} bytes for {}, expected {}",
        source_bytes,
        segment.source_name,
        segment.source_byte_len
    );
    ensure!(
        source_bytes / segment.source_dtype.byte_size() == segment.element_count,
        "read element count mismatch for {}",
        segment.source_name
    );
    Ok(output_bytes)
}

/// Deinterleave alternating rows of a fused SwiGLU tensor (Inkling MTP fused
/// w13): parity 0 keeps even rows (gate), parity 1 keeps odd rows (up).
fn stream_alternating_rows(
    writer: &mut File,
    file: &SafetensorFile,
    segment: &TensorSegment,
    buffer_size: usize,
    parity: u64,
    row_elements: u64,
) -> Result<u64> {
    ensure!(parity < 2, "alternating-row parity must be zero or one");
    ensure!(row_elements > 0, "alternating-row width must be non-zero");
    let row_bytes = row_elements
        .checked_mul(segment.source_dtype.byte_size())
        .context("alternating-row byte length overflow")?;
    let row_bytes = usize::try_from(row_bytes).context("row byte length does not fit usize")?;
    let chunk_size = aligned_chunk_size(buffer_size, row_bytes);
    let mut source_bytes = 0_u64;
    let mut output_bytes = 0_u64;
    let mut row_index = 0_u64;
    file.stream_tensor_chunks(&segment.source_name, chunk_size, |chunk| {
        ensure!(
            chunk.len() % row_bytes == 0,
            "chunk for {} split a fused SwiGLU row",
            segment.source_name
        );
        source_bytes += chunk.len() as u64;
        for row in chunk.chunks_exact(row_bytes) {
            if row_index % 2 == parity {
                output_bytes +=
                    convert_float_chunk(row, segment.source_dtype, segment.target_dtype, writer)?;
            }
            row_index += 1;
        }
        Ok(())
    })?;
    ensure!(
        source_bytes == segment.source_byte_len,
        "read {} bytes for {}, expected {}",
        source_bytes,
        segment.source_name,
        segment.source_byte_len
    );
    ensure!(
        output_bytes == segment.target_byte_len,
        "deinterleaved {} bytes for {}, expected {}",
        output_bytes,
        segment.source_name,
        segment.target_byte_len
    );
    Ok(output_bytes)
}

fn aligned_chunk_size(buffer_size: usize, element_size: usize) -> usize {
    let aligned = buffer_size - (buffer_size % element_size);
    aligned.max(element_size)
}

fn pad_writer_to_alignment(writer: &mut File, alignment: u64) -> Result<()> {
    let position = writer.stream_position()?;
    pad_writer_to_position(writer, align_to(position, alignment))
}

fn pad_writer_to_position(writer: &mut File, position: u64) -> Result<()> {
    let current = writer.stream_position()?;
    ensure!(
        current <= position,
        "writer is past requested output position {position}"
    );
    let mut remaining = position - current;
    let zeros = [0_u8; 4096];
    while remaining > 0 {
        let write_len = zeros.len().min(remaining as usize);
        writer.write_all(&zeros[..write_len])?;
        remaining -= write_len as u64;
    }
    Ok(())
}

fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    write_u64(writer, value.len() as u64)?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

#[cfg(test)]
#[path = "gguf_writer_tests.rs"]
mod tests;
