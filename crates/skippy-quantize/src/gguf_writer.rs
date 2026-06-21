use std::collections::{BTreeMap, btree_map::Entry};
use std::fs::{self, File};
use std::io::{Seek, Write};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Serialize;

use crate::float_convert::{FloatDType, convert_float_chunk, target_dtype_for_tensor};
use crate::hf_checkpoint::{SafetensorFile, SafetensorTensorInfo, open_safetensor_files};
use crate::tensor_map::{
    TensorNameMap, hf_layer_id, is_mtp_source_tensor, is_shared_mtp_context_tensor,
};
use crate::types::ConvertOutputType;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGUF_ALIGNMENT: u64 = 32;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_UINT64: u32 = 10;
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
    let tensors = collect_tensor_sources(
        &files,
        options.tensor_name_map,
        options.output_type,
        options.tensor_selection,
    )?;
    ensure!(
        !tensors.is_empty(),
        "no tensors found under {}",
        source.display()
    );
    let total_tensor_count = tensors.len();
    let tensors = select_split_tensors(tensors, options.split)?;
    let metadata = options
        .metadata
        .clone()
        .unwrap_or_else(|| raw_metadata(source, total_tensor_count));
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
    let split_count =
        usize::try_from(split.split_count).context("split_count does not fit usize")?;
    let start = (split_index - 1) * total_tensors / split_count;
    let end = split_index * total_tensors / split_count;
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
) -> Result<Vec<TensorSource>> {
    let mut tensors = Vec::new();
    let mut expert_groups = BTreeMap::<ExpertGroupKey, ExpertGroup>::new();
    for (file_index, file) in files.iter().enumerate() {
        for tensor in file.tensors().values() {
            if !tensor_selection.includes(tensor.name())? {
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
    let mut offset = 0_u64;
    for tensor in &mut tensors {
        offset = align_to(offset, GGUF_ALIGNMENT);
        tensor.gguf_offset = offset;
        offset = offset
            .checked_add(tensor.byte_len)
            .with_context(|| format!("GGUF data offset overflow after {}", tensor.name))?;
    }
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
        let target_dtype = target_dtype_for_tensor(source_dtype, output_type, tensor.shape())?;
        let name = tensor_name_map.map_tensor_name(tensor.name())?;
        let element_count = tensor_element_count(tensor)?;
        Ok(Self {
            segments: vec![TensorSegment {
                file_index,
                source_name: tensor.name().to_string(),
                source_dtype,
                target_dtype,
                element_count,
                source_byte_len: tensor.byte_len(),
                target_byte_len: tensor_byte_len(element_count, target_dtype)?,
            }],
            name,
            dims: tensor.shape().iter().rev().copied().collect(),
            ggml_type: ggml_type_for_dtype(target_dtype),
            byte_len: tensor_byte_len(element_count, target_dtype)?,
            gguf_offset: 0,
        })
    }
}

struct TensorSegment {
    file_index: usize,
    source_name: String,
    source_dtype: FloatDType,
    target_dtype: FloatDType,
    element_count: u64,
    source_byte_len: u64,
    target_byte_len: u64,
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

#[derive(Debug, Clone)]
pub(crate) enum GgufKv {
    ArrayF32 { key: String, value: Vec<f32> },
    ArrayI32 { key: String, value: Vec<i32> },
    ArrayString { key: String, value: Vec<String> },
    Bool { key: String, value: bool },
    F32 { key: String, value: f32 },
    I32 { key: String, value: i32 },
    String { key: String, value: String },
    U16 { key: String, value: u16 },
    U32 { key: String, value: u32 },
    U64 { key: String, value: u64 },
}

impl GgufKv {
    pub(crate) fn array_f32(key: &str, value: Vec<f32>) -> Self {
        Self::ArrayF32 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn array_i32(key: &str, value: Vec<i32>) -> Self {
        Self::ArrayI32 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn array_string(key: &str, value: Vec<String>) -> Self {
        Self::ArrayString {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn bool(key: &str, value: bool) -> Self {
        Self::Bool {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn f32(key: &str, value: f32) -> Self {
        Self::F32 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn i32(key: &str, value: i32) -> Self {
        Self::I32 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn string(key: &str, value: &str) -> Self {
        Self::String {
            key: key.to_string(),
            value: value.to_string(),
        }
    }

    pub(crate) fn u16(key: &str, value: u16) -> Self {
        Self::U16 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn u32(key: &str, value: u32) -> Self {
        Self::U32 {
            key: key.to_string(),
            value,
        }
    }

    pub(crate) fn u64(key: &str, value: u64) -> Self {
        Self::U64 {
            key: key.to_string(),
            value,
        }
    }
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

fn write_kv<W: Write>(writer: &mut W, kv: &GgufKv) -> Result<()> {
    match kv {
        GgufKv::ArrayF32 { key, value } => {
            write_array_header(writer, key, GGUF_TYPE_FLOAT32, value.len())?;
            for item in value {
                writer.write_all(&item.to_le_bytes())?;
            }
        }
        GgufKv::ArrayI32 { key, value } => {
            write_array_header(writer, key, GGUF_TYPE_INT32, value.len())?;
            for item in value {
                writer.write_all(&item.to_le_bytes())?;
            }
        }
        GgufKv::ArrayString { key, value } => {
            write_array_header(writer, key, GGUF_TYPE_STRING, value.len())?;
            for item in value {
                write_string(writer, item)?;
            }
        }
        GgufKv::Bool { key, value } => {
            write_string(writer, key)?;
            write_u32(writer, GGUF_TYPE_BOOL)?;
            writer.write_all(&[*value as u8])?;
        }
        GgufKv::F32 { key, value } => {
            write_string(writer, key)?;
            write_u32(writer, GGUF_TYPE_FLOAT32)?;
            writer.write_all(&value.to_le_bytes())?;
        }
        GgufKv::I32 { key, value } => {
            write_string(writer, key)?;
            write_u32(writer, GGUF_TYPE_INT32)?;
            writer.write_all(&value.to_le_bytes())?;
        }
        GgufKv::String { key, value } => {
            write_string(writer, key)?;
            write_u32(writer, GGUF_TYPE_STRING)?;
            write_string(writer, value)?;
        }
        GgufKv::U16 { key, value } => {
            write_string(writer, key)?;
            write_u32(writer, GGUF_TYPE_UINT16)?;
            writer.write_all(&value.to_le_bytes())?;
        }
        GgufKv::U32 { key, value } => {
            write_string(writer, key)?;
            write_u32(writer, GGUF_TYPE_UINT32)?;
            write_u32(writer, *value)?;
        }
        GgufKv::U64 { key, value } => {
            write_string(writer, key)?;
            write_u32(writer, GGUF_TYPE_UINT64)?;
            write_u64(writer, *value)?;
        }
    }
    Ok(())
}

fn write_array_header<W: Write>(
    writer: &mut W,
    key: &str,
    element_type: u32,
    len: usize,
) -> Result<()> {
    ensure!(!key.is_empty(), "GGUF metadata key cannot be empty");
    ensure!(
        len > 0,
        "GGUF array metadata {key:?} cannot be empty because llama.cpp rejects empty arrays"
    );
    write_string(writer, key)?;
    write_u32(writer, GGUF_TYPE_ARRAY)?;
    write_u32(writer, element_type)?;
    write_u64(writer, len as u64)
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
