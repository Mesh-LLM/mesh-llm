//! Canonical GGUF metadata and streaming tensor preparation.

use std::collections::{BTreeMap, btree_map::Entry};
use std::fs::{self, File};
use std::io::{Seek, Write};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::Value;

use crate::ConvertOutputType;
use crate::float_convert::{
    FloatDType, convert_float_chunk, read_float_element, target_dtype_for_tensor,
    write_float_element,
};
pub use crate::gguf_metadata::GgufKv;
use crate::gguf_metadata::write_kv;
#[cfg(test)]
use crate::gguf_metadata::{
    GGUF_TYPE_ARRAY, GGUF_TYPE_BOOL, GGUF_TYPE_FLOAT32, GGUF_TYPE_INT32, GGUF_TYPE_STRING,
    GGUF_TYPE_UINT16, GGUF_TYPE_UINT32, GGUF_TYPE_UINT64,
};
use crate::gguf_template::{metadata_from_hf_config, mtp_layer_start_from_hf_config};
use crate::hf_checkpoint::{SafetensorFile, SafetensorTensorInfo, open_safetensor_files};
use crate::tensor_map::{
    TensorNameMap, hf_layer_id, inkling_mtp_depth, is_inkling_fused_w13, is_mtp_source_tensor,
    is_shared_mtp_context_tensor,
};

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
pub struct RawGgufWriteOptions {
    pub buffer_size: usize,
    pub metadata: Option<Vec<GgufKv>>,
    pub tensor_name_map: TensorNameMap,
    pub split: Option<GgufSplit>,
    pub output_type: Option<ConvertOutputType>,
    pub tensor_selection: TensorSelection,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum TensorSelection {
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
pub struct GgufSplit {
    pub split_index: u32,
    pub split_count: u32,
}

pub fn write_raw_safetensors_gguf(
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

pub fn validate_raw_safetensors_gguf(
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

pub fn recommended_raw_safetensors_gguf_split_count(
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

/// A validated Hugging Face checkpoint prepared for direct runtime loading.
///
/// The object owns immutable mappings for every SafeTensors shard. Its GGUF
/// buffer contains metadata and tensor descriptors only; tensor bytes are read
/// from the source mappings on demand.
pub struct DirectCheckpoint {
    files: Vec<SafetensorFile>,
    tensors: Vec<TensorSource>,
    metadata_gguf: Vec<u8>,
    buffer_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImatrixTensorLayout {
    pub name: String,
    pub value_count: usize,
}

impl DirectCheckpoint {
    /// Open a checkpoint directory or SafeTensors file and prepare canonical
    /// llama.cpp metadata and names.
    pub fn open(source: &Path, buffer_size: usize) -> Result<Self> {
        let source = if source.is_file() {
            ensure!(
                source.extension().and_then(|value| value.to_str()) == Some("safetensors"),
                "checkpoint file must use the .safetensors extension: {}",
                source.display()
            );
            source
                .parent()
                .with_context(|| format!("checkpoint file has no parent: {}", source.display()))?
        } else {
            source
        };
        let files = open_safetensor_files(source)?;
        ensure!(
            !files.is_empty(),
            "no safetensors files found under {}",
            source.display()
        );
        let tensor_count = files.iter().map(|file| file.tensors().len()).sum();
        let mtp_layer_start = mtp_layer_start_from_hf_config(source)?;
        let tensor_name_map = mtp_layer_start
            .map(|layer_start| TensorNameMap::HfToGgufWithMtp { layer_start })
            .unwrap_or(TensorNameMap::HfToGguf);
        let prepared = prepare_raw_safetensors_gguf_with_files(
            source,
            &RawGgufWriteOptions {
                buffer_size,
                metadata: Some(metadata_from_hf_config(source, tensor_count)?),
                tensor_name_map,
                split: None,
                output_type: None,
                tensor_selection: TensorSelection::All,
            },
            files,
        )?;
        let mut metadata_gguf = Vec::new();
        write_header_and_tensor_table(&mut metadata_gguf, &prepared.metadata, &prepared.tensors)?;
        let aligned_len = usize::try_from(align_to(metadata_gguf.len() as u64, GGUF_ALIGNMENT))
            .context("metadata GGUF length does not fit usize")?;
        metadata_gguf.resize(aligned_len, 0);
        Ok(Self {
            files: prepared.files,
            tensors: prepared.tensors,
            metadata_gguf,
            buffer_size,
        })
    }

    /// Metadata-only GGUF consumed by the native model constructor.
    pub fn metadata_gguf(&self) -> &[u8] {
        &self.metadata_gguf
    }

    /// Number of canonical tensors exposed by this checkpoint.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Canonical tensor names and importance-vector widths expected by llama.cpp.
    pub fn imatrix_layout(&self) -> Result<Vec<ImatrixTensorLayout>> {
        self.tensors
            .iter()
            .filter(|tensor| tensor.dims.len() >= 2)
            .map(|tensor| {
                let ne0 = usize::try_from(tensor.dims[0]).with_context(|| {
                    format!("imatrix width does not fit usize for {}", tensor.name)
                })?;
                let ne2 = tensor
                    .dims
                    .get(2)
                    .copied()
                    .map(usize::try_from)
                    .transpose()
                    .with_context(|| {
                        format!("imatrix depth does not fit usize for {}", tensor.name)
                    })?
                    .unwrap_or(1);
                let value_count = ne0.checked_mul(ne2).with_context(|| {
                    format!("imatrix value count overflows usize for {}", tensor.name)
                })?;
                Ok(ImatrixTensorLayout {
                    name: tensor.name.clone(),
                    value_count,
                })
            })
            .collect()
    }

    /// Decode one canonical tensor into the caller-provided F32 destination.
    pub fn read_tensor_f32(&self, name: &str, destination: &mut [f32]) -> Result<()> {
        let tensor = self
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .with_context(|| format!("canonical tensor {name} not found in checkpoint"))?;
        let target_dtype = tensor
            .segments
            .first()
            .context("tensor has no source segments")?
            .target_dtype;
        ensure!(
            tensor
                .segments
                .iter()
                .all(|segment| segment.target_dtype == target_dtype),
            "tensor {name} has mixed target dtypes"
        );
        let expected_elements = tensor.byte_len / target_dtype.byte_size();
        ensure!(
            destination.len() as u64 == expected_elements,
            "tensor {name} destination has {} elements, expected {expected_elements}",
            destination.len()
        );
        let mut bytes = Vec::with_capacity(
            usize::try_from(tensor.byte_len).context("tensor byte length does not fit usize")?,
        );
        for segment in &tensor.segments {
            stream_segment(
                &mut bytes,
                &self.files[segment.file_index],
                segment,
                self.buffer_size,
            )?;
        }
        ensure!(
            bytes.len() as u64 == tensor.byte_len,
            "decoded {} bytes for {name}, expected {}",
            bytes.len(),
            tensor.byte_len
        );
        for (index, value) in destination.iter_mut().enumerate() {
            *value = read_float_element(&bytes, target_dtype, index);
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct RawGgufValidation {
    pub selected_tensor_count: usize,
    pub selected_tensor_bytes: u64,
    pub metadata_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_type: Option<String>,
}

fn prepare_raw_safetensors_gguf(
    source: &Path,
    options: &RawGgufWriteOptions,
) -> Result<PreparedGgufWrite> {
    let files = open_safetensor_files(source)?;
    prepare_raw_safetensors_gguf_with_files(source, options, files)
}

fn prepare_raw_safetensors_gguf_with_files(
    source: &Path,
    options: &RawGgufWriteOptions,
    files: Vec<SafetensorFile>,
) -> Result<PreparedGgufWrite> {
    ensure!(
        options.buffer_size > 0,
        "buffer_size must be greater than zero"
    );
    ensure!(
        !files.is_empty(),
        "no safetensors files found under {}",
        source.display()
    );
    let metadata_seed = options.metadata.clone();
    let glm_dsa_kv_b_split = glm_dsa_kv_b_split_mode(metadata_seed.as_deref())?;
    let hf_layout = HfTensorLayout::from_checkpoint(source, options.tensor_name_map)?;
    let tensors = collect_tensor_sources(
        &files,
        options.tensor_name_map,
        options.output_type,
        options.tensor_selection,
        glm_dsa_kv_b_split,
        hf_layout,
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
    hf_layout: HfTensorLayout,
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
            if matches!(hf_layout, HfTensorLayout::GraniteHybrid { .. })
                && tensor.name().ends_with("shared_mlp.input_linear.weight")
            {
                tensors.extend(granite_shared_mlp_tensor_sources(
                    file_index,
                    tensor,
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
                hf_layout,
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
        hf_layout: HfTensorLayout,
    ) -> Result<Self> {
        let source_dtype = FloatDType::from_safetensor(tensor.dtype()).with_context(|| {
            format!("unsupported dtype {} for {}", tensor.dtype(), tensor.name())
        })?;
        let name = tensor_name_map.map_tensor_name(tensor.name())?;
        let target_dtype =
            target_dtype_for_mapped_tensor(source_dtype, output_type, tensor.shape(), &name)?;
        let element_count = tensor_element_count(tensor)?;
        let dims = mapped_tensor_dims(tensor.shape(), &name, hf_layout)?;
        let transform = tensor_transform(tensor, &name, hf_layout)?;
        Ok(Self {
            segments: vec![TensorSegment {
                file_index,
                source_name: tensor.name().to_string(),
                source_dtype,
                target_dtype,
                element_count,
                source_byte_len: tensor.byte_len(),
                target_byte_len: tensor_byte_len(element_count, target_dtype)?,
                transform,
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
    if mapped_name.ends_with("attn_rel_proj.weight")
        || mapped_name.contains(".shortconv_")
        || mapped_name.ends_with(".ssm_conv1d.weight")
    {
        return Ok(FloatDType::F32);
    }
    target_dtype_for_tensor(source_dtype, output_type, shape)
}

fn mapped_tensor_dims(
    shape: &[u64],
    mapped_name: &str,
    hf_layout: HfTensorLayout,
) -> Result<Vec<u64>> {
    if mapped_name.contains(".shortconv_") {
        ensure!(
            shape.len() == 3 && shape[1] == 1,
            "Inkling shortconv tensor {mapped_name} must have shape [channels, 1, kernel], got {shape:?}"
        );
        return Ok(vec![shape[2], shape[0]]);
    }
    if matches!(hf_layout, HfTensorLayout::GraniteHybrid { .. })
        && mapped_name.ends_with(".ssm_conv1d.weight")
    {
        ensure!(
            shape.len() == 3 && shape[1] == 1,
            "Granite Hybrid SSM convolution {mapped_name} must have shape [channels, 1, kernel], got {shape:?}"
        );
        return Ok(vec![shape[2], shape[0]]);
    }
    if matches!(hf_layout, HfTensorLayout::GraniteHybrid { .. })
        && (mapped_name.ends_with(".ssm_a") || mapped_name.ends_with(".ssm_d"))
    {
        ensure!(
            shape.len() == 1,
            "Granite Hybrid {mapped_name} must be rank 1, got {shape:?}"
        );
        return Ok(vec![1, shape[0]]);
    }
    if let HfTensorLayout::GraniteHybrid {
        ssm_group_count, ..
    } = hf_layout
        && mapped_name.ends_with(".ssm_norm.weight")
    {
        ensure!(
            shape.len() == 1 && shape[0].is_multiple_of(ssm_group_count),
            "Granite Hybrid {mapped_name} shape {shape:?} is not divisible by SSM group count {ssm_group_count}"
        );
        return Ok(vec![shape[0] / ssm_group_count, ssm_group_count]);
    }
    Ok(shape.iter().rev().copied().collect())
}

fn tensor_transform(
    tensor: &SafetensorTensorInfo,
    mapped_name: &str,
    hf_layout: HfTensorLayout,
) -> Result<TensorTransform> {
    if matches!(hf_layout, HfTensorLayout::GraniteHybrid { .. }) && mapped_name.ends_with(".ssm_a")
    {
        return Ok(TensorTransform::NegativeExp);
    }
    let head_count = match (hf_layout, mapped_name) {
        (
            HfTensorLayout::LlamaLike {
                head_count,
                kv_head_count: _,
            }
            | HfTensorLayout::GraniteHybrid {
                head_count,
                kv_head_count: _,
                ssm_group_count: _,
            },
            name,
        ) if name.ends_with(".attn_q.weight") || name.ends_with(".attn_q.bias") => Some(head_count),
        (
            HfTensorLayout::LlamaLike { kv_head_count, .. }
            | HfTensorLayout::GraniteHybrid { kv_head_count, .. },
            name,
        ) if name.ends_with(".attn_k.weight") || name.ends_with(".attn_k.bias") => {
            Some(kv_head_count)
        }
        _ => None,
    };
    let Some(head_count) = head_count else {
        return Ok(TensorTransform::Identity);
    };
    let row_count = *tensor
        .shape()
        .first()
        .with_context(|| format!("RoPE tensor {} has no dimensions", tensor.name()))?;
    let row_elements = tensor.shape()[1..].iter().try_fold(1_u64, |acc, dim| {
        acc.checked_mul(*dim)
            .with_context(|| format!("RoPE row width overflow for {}", tensor.name()))
    })?;
    ensure!(
        head_count > 0 && row_count.is_multiple_of(head_count * 2),
        "RoPE tensor {} row count {row_count} must be divisible by twice the head count {head_count}",
        tensor.name()
    );
    Ok(TensorTransform::RopePermutation {
        head_count,
        row_count,
        row_elements,
    })
}

#[derive(Debug, Clone, Copy, Default)]
enum HfTensorLayout {
    #[default]
    Generic,
    LlamaLike {
        head_count: u64,
        kv_head_count: u64,
    },
    GraniteHybrid {
        head_count: u64,
        kv_head_count: u64,
        ssm_group_count: u64,
    },
}

impl HfTensorLayout {
    fn from_checkpoint(source: &Path, tensor_name_map: TensorNameMap) -> Result<Self> {
        if !matches!(
            tensor_name_map,
            TensorNameMap::HfToGguf | TensorNameMap::HfToGgufWithMtp { .. }
        ) {
            return Ok(Self::Generic);
        }
        let path = source.join("config.json");
        if !path.is_file() {
            return Ok(Self::Generic);
        }
        let config: Value = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        )
        .with_context(|| format!("parse {}", path.display()))?;
        let model_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let value_u64 = |key: &str| -> Result<u64> {
            config
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("config missing integer {key}"))
        };
        let head_count = || value_u64("num_attention_heads");
        let kv_head_count = || {
            config
                .get("num_key_value_heads")
                .and_then(Value::as_u64)
                .map(Ok)
                .unwrap_or_else(head_count)
        };
        match model_type {
            "llama" | "mistral" => Ok(Self::LlamaLike {
                head_count: head_count()?,
                kv_head_count: kv_head_count()?,
            }),
            "granitemoehybrid" => Ok(Self::GraniteHybrid {
                head_count: head_count()?,
                kv_head_count: kv_head_count()?,
                ssm_group_count: value_u64("mamba_n_groups")?,
            }),
            _ => Ok(Self::Generic),
        }
    }
}

fn granite_shared_mlp_tensor_sources(
    file_index: usize,
    tensor: &SafetensorTensorInfo,
    output_type: Option<ConvertOutputType>,
) -> Result<Vec<TensorSource>> {
    ensure!(
        tensor.shape().len() == 2 && tensor.shape()[0].is_multiple_of(2),
        "Granite Hybrid shared MLP input {} must have shape [2 * intermediate, hidden], got {:?}",
        tensor.name(),
        tensor.shape()
    );
    let layer = hf_layer_id(tensor.name())?
        .with_context(|| format!("missing Granite Hybrid layer id in {}", tensor.name()))?;
    let source_dtype = FloatDType::from_safetensor(tensor.dtype())
        .with_context(|| format!("unsupported dtype {} for {}", tensor.dtype(), tensor.name()))?;
    let row_count = tensor.shape()[0] / 2;
    let row_elements = tensor.shape()[1];
    let output_shape = [row_count, row_elements];
    let target_dtype = target_dtype_for_tensor(source_dtype, output_type, &output_shape)?;
    let element_count = row_count
        .checked_mul(row_elements)
        .context("Granite Hybrid shared MLP element count overflow")?;
    let target_byte_len = tensor_byte_len(element_count, target_dtype)?;
    let dims = vec![row_elements, row_count];
    Ok([("ffn_gate", 0_u64), ("ffn_up", row_count)]
        .into_iter()
        .map(|(projection, row_start)| TensorSource {
            segments: vec![TensorSegment {
                file_index,
                source_name: tensor.name().to_string(),
                source_dtype,
                target_dtype,
                element_count,
                source_byte_len: tensor.byte_len(),
                target_byte_len,
                transform: TensorTransform::ContiguousRows {
                    row_start,
                    row_count,
                    row_elements,
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

fn stream_segment<W: Write>(
    writer: &mut W,
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
    if let TensorTransform::ContiguousRows {
        row_start,
        row_count,
        row_elements,
    } = segment.transform
    {
        return stream_contiguous_rows(
            writer,
            file,
            segment,
            buffer_size,
            row_start,
            row_count,
            row_elements,
        );
    }
    if let TensorTransform::RopePermutation {
        head_count,
        row_count,
        row_elements,
    } = segment.transform
    {
        return stream_rope_permutation(
            writer,
            file,
            segment,
            buffer_size,
            head_count,
            row_count,
            row_elements,
        );
    }
    if matches!(segment.transform, TensorTransform::NegativeExp) {
        return stream_negative_exp(writer, file, segment, buffer_size);
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
fn stream_alternating_rows<W: Write>(
    writer: &mut W,
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

fn stream_contiguous_rows<W: Write>(
    writer: &mut W,
    file: &SafetensorFile,
    segment: &TensorSegment,
    buffer_size: usize,
    row_start: u64,
    row_count: u64,
    row_elements: u64,
) -> Result<u64> {
    ensure!(row_count > 0, "contiguous-row count must be non-zero");
    ensure!(row_elements > 0, "contiguous-row width must be non-zero");
    let source = read_segment_source(file, segment, buffer_size)?;
    let source_width = segment.source_dtype.byte_size();
    let start_element = row_start
        .checked_mul(row_elements)
        .context("contiguous-row start overflow")?;
    let element_count = row_count
        .checked_mul(row_elements)
        .context("contiguous-row element count overflow")?;
    let start_byte = start_element
        .checked_mul(source_width)
        .context("contiguous-row byte start overflow")?;
    let byte_len = element_count
        .checked_mul(source_width)
        .context("contiguous-row byte length overflow")?;
    let end_byte = start_byte
        .checked_add(byte_len)
        .context("contiguous-row byte end overflow")?;
    let start = usize::try_from(start_byte).context("contiguous-row start does not fit usize")?;
    let end = usize::try_from(end_byte).context("contiguous-row end does not fit usize")?;
    ensure!(
        end <= source.len(),
        "contiguous-row range {start}..{end} exceeds {} source bytes for {}",
        source.len(),
        segment.source_name
    );
    let written = if segment.source_dtype == segment.target_dtype {
        writer.write_all(&source[start..end])?;
        byte_len
    } else {
        convert_float_chunk(
            &source[start..end],
            segment.source_dtype,
            segment.target_dtype,
            writer,
        )?
    };
    ensure!(
        written == segment.target_byte_len,
        "wrote {written} bytes for contiguous rows of {}, expected {}",
        segment.source_name,
        segment.target_byte_len
    );
    Ok(written)
}

fn stream_rope_permutation<W: Write>(
    writer: &mut W,
    file: &SafetensorFile,
    segment: &TensorSegment,
    buffer_size: usize,
    head_count: u64,
    row_count: u64,
    row_elements: u64,
) -> Result<u64> {
    ensure!(head_count > 0, "RoPE head count must be non-zero");
    ensure!(row_elements > 0, "RoPE row width must be non-zero");
    ensure!(
        row_count.is_multiple_of(head_count * 2),
        "RoPE row count {row_count} must be divisible by twice head count {head_count}"
    );
    let source = read_segment_source(file, segment, buffer_size)?;
    let head_dim = row_count / head_count;
    let source_element_count = row_count
        .checked_mul(row_elements)
        .context("RoPE tensor element count overflow")?;
    ensure!(
        source_element_count == segment.element_count,
        "RoPE transform element count {source_element_count} does not match {} for {}",
        segment.element_count,
        segment.source_name
    );
    let flush_limit = buffer_size.max(segment.target_dtype.byte_size() as usize);
    let mut output = Vec::with_capacity(flush_limit);
    let mut written = 0_u64;
    for head in 0..head_count {
        for target_row_in_head in 0..head_dim {
            let source_row_in_head =
                (target_row_in_head % 2) * (head_dim / 2) + target_row_in_head / 2;
            let source_row = head * head_dim + source_row_in_head;
            for column in 0..row_elements {
                let source_index = source_row
                    .checked_mul(row_elements)
                    .and_then(|value| value.checked_add(column))
                    .context("RoPE source index overflow")?;
                let source_index = usize::try_from(source_index)
                    .context("RoPE source index does not fit usize")?;
                ensure!(
                    source_index < segment.element_count as usize,
                    "RoPE source index {source_index} exceeds {} elements",
                    segment.element_count
                );
                write_float_element(
                    &mut output,
                    segment.target_dtype,
                    read_float_element(&source, segment.source_dtype, source_index),
                );
                written += segment.target_dtype.byte_size();
                if output.len() >= flush_limit {
                    writer.write_all(&output)?;
                    output.clear();
                }
            }
        }
    }
    writer.write_all(&output)?;
    ensure!(
        written == segment.target_byte_len,
        "wrote {written} bytes for RoPE permutation of {}, expected {}",
        segment.source_name,
        segment.target_byte_len
    );
    Ok(written)
}

fn stream_negative_exp<W: Write>(
    writer: &mut W,
    file: &SafetensorFile,
    segment: &TensorSegment,
    buffer_size: usize,
) -> Result<u64> {
    let source = read_segment_source(file, segment, buffer_size)?;
    let flush_limit = buffer_size.max(segment.target_dtype.byte_size() as usize);
    let mut output = Vec::with_capacity(flush_limit);
    let mut written = 0_u64;
    let element_count = usize::try_from(segment.element_count)
        .context("negative-exp element count does not fit usize")?;
    for index in 0..element_count {
        let value = -read_float_element(&source, segment.source_dtype, index).exp();
        write_float_element(&mut output, segment.target_dtype, value);
        written += segment.target_dtype.byte_size();
        if output.len() >= flush_limit {
            writer.write_all(&output)?;
            output.clear();
        }
    }
    writer.write_all(&output)?;
    ensure!(
        written == segment.target_byte_len,
        "wrote {written} bytes for negative-exp transform of {}, expected {}",
        segment.source_name,
        segment.target_byte_len
    );
    Ok(written)
}

fn read_segment_source(
    file: &SafetensorFile,
    segment: &TensorSegment,
    buffer_size: usize,
) -> Result<Vec<u8>> {
    let mut source = Vec::with_capacity(
        usize::try_from(segment.source_byte_len)
            .context("transformed source byte length does not fit usize")?,
    );
    file.stream_tensor_chunks(&segment.source_name, buffer_size, |chunk| {
        source.extend_from_slice(chunk);
        Ok(())
    })?;
    ensure!(
        source.len() as u64 == segment.source_byte_len,
        "read {} bytes for {}, expected {}",
        source.len(),
        segment.source_name,
        segment.source_byte_len
    );
    Ok(source)
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
