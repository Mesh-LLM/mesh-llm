use std::collections::BTreeSet;
use std::fs::File;
use std::io::Write;

use anyhow::{Context, Result, ensure};

use crate::float_convert::{
    FloatDType, read_float_element, target_dtype_for_tensor, write_float_element,
};
use crate::hf_checkpoint::{SafetensorFile, SafetensorTensorInfo};
use crate::types::ConvertOutputType;

use super::{
    GgufKv, TensorSegment, TensorSource, ggml_type_for_dtype, tensor_byte_len, tensor_element_count,
};

const INDEXER_TENSORS: &[&str] = &[
    "indexer.k_norm.weight",
    "indexer.k_norm.bias",
    "indexer.proj.weight",
    "indexer.attn_k.weight",
    "indexer.attn_q_b.weight",
];

pub(super) fn enrich_glm_dsa_indexshare_metadata(
    metadata: &mut Vec<GgufKv>,
    tensors: &[TensorSource],
) -> Result<()> {
    if !is_glm_dsa_metadata(metadata) {
        return Ok(());
    }
    if metadata_has_key(metadata, "glm-dsa.attention.indexer.types")
        || metadata_has_key(metadata, "glm-dsa.attention.indexer.top_k_frequency")
    {
        return Ok(());
    }

    let block_count = metadata_u32(metadata, "glm-dsa.block_count")
        .context("GLM-DSA metadata missing glm-dsa.block_count")?;
    let nextn_layers = metadata_u32(metadata, "glm-dsa.nextn_predict_layers").unwrap_or(0);
    ensure!(
        nextn_layers < block_count,
        "GLM-DSA nextn_predict_layers {nextn_layers} must be less than block_count {block_count}"
    );
    let decoder_layers = block_count - nextn_layers;
    let tensor_names = tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    let mut roles = Vec::with_capacity(decoder_layers as usize);
    for layer in 0..decoder_layers {
        let indexer_count = INDEXER_TENSORS
            .iter()
            .filter(|suffix| tensor_names.contains(format!("blk.{layer}.{suffix}").as_str()))
            .count();
        ensure!(
            indexer_count == 0 || indexer_count == INDEXER_TENSORS.len(),
            "GLM-DSA layer {layer} has partial indexer tensor group ({indexer_count}/{})",
            INDEXER_TENSORS.len()
        );
        roles.push(if indexer_count == INDEXER_TENSORS.len() {
            "full".to_string()
        } else {
            "shared".to_string()
        });
    }
    metadata.push(GgufKv::array_string(
        "glm-dsa.attention.indexer.types",
        roles,
    ));
    Ok(())
}

pub(super) fn glm_dsa_kv_b_split_mode(metadata: Option<&[GgufKv]>) -> Result<GlmDsaKvBSplitMode> {
    let Some(metadata) = metadata.filter(|metadata| is_glm_dsa_metadata(metadata)) else {
        return Ok(GlmDsaKvBSplitMode::Disabled);
    };
    Ok(match glm_dsa_kv_b_split_config(metadata)? {
        Some(config) => GlmDsaKvBSplitMode::Config(config),
        None => GlmDsaKvBSplitMode::MissingMetadata,
    })
}

fn glm_dsa_kv_b_split_config(metadata: &[GgufKv]) -> Result<Option<GlmDsaKvBSplitConfig>> {
    if !is_glm_dsa_metadata(metadata) {
        return Ok(None);
    }
    let Some(head_count) = metadata_u32(metadata, "glm-dsa.attention.head_count") else {
        return Ok(None);
    };
    let Some(key_length) = metadata_u32(metadata, "glm-dsa.attention.key_length_mla")
        .or_else(|| metadata_u32(metadata, "glm-dsa.attention.key_length"))
    else {
        return Ok(None);
    };
    let Some(rope_dim) = metadata_u32(metadata, "glm-dsa.rope.dimension_count") else {
        return Ok(None);
    };
    let Some(value_head_dim) = metadata_u32(metadata, "glm-dsa.attention.value_length") else {
        return Ok(None);
    };
    let Some(kv_lora_rank) = metadata_u32(metadata, "glm-dsa.attention.kv_lora_rank") else {
        return Ok(None);
    };
    ensure!(
        key_length > rope_dim,
        "glm-dsa.attention.key_length_mla must be greater than rope.dimension_count for kv_b split"
    );
    Ok(Some(GlmDsaKvBSplitConfig {
        head_count: u64::from(head_count),
        qk_nope_head_dim: u64::from(key_length - rope_dim),
        value_head_dim: u64::from(value_head_dim),
        kv_lora_rank: u64::from(kv_lora_rank),
    }))
}

pub(super) fn glm_dsa_kv_b_layer(name: &str) -> Result<Option<u32>> {
    let Some(rest) = name.strip_prefix("model.layers.") else {
        return Ok(None);
    };
    let Some((layer, suffix)) = rest.split_once('.') else {
        return Ok(None);
    };
    if suffix != "self_attn.kv_b_proj.weight" {
        return Ok(None);
    }
    layer
        .parse::<u32>()
        .map(Some)
        .with_context(|| format!("parse GLM-DSA kv_b layer id in {name}"))
}

impl TensorSource {
    pub(super) fn from_glm_dsa_kv_b_split(
        file_index: usize,
        tensor: &SafetensorTensorInfo,
        layer: u32,
        split: GlmDsaKvBSplitConfig,
        output_type: Option<ConvertOutputType>,
    ) -> Result<Vec<Self>> {
        ensure!(
            tensor.shape().len() == 2,
            "GLM-DSA tensor {} must be rank-2 for kv_b split, got shape {:?}",
            tensor.name(),
            tensor.shape()
        );
        let expected_rows = split
            .head_count
            .checked_mul(split.qk_nope_head_dim + split.value_head_dim)
            .context("GLM-DSA kv_b expected row count overflow")?;
        ensure!(
            tensor.shape()[0] == expected_rows && tensor.shape()[1] == split.kv_lora_rank,
            "GLM-DSA tensor {} shape {:?} does not match expected [{expected_rows}, {}]",
            tensor.name(),
            tensor.shape(),
            split.kv_lora_rank
        );
        let source_dtype = FloatDType::from_safetensor(tensor.dtype()).with_context(|| {
            format!("unsupported dtype {} for {}", tensor.dtype(), tensor.name())
        })?;
        let target_dtype = target_dtype_for_tensor(source_dtype, output_type, tensor.shape())?;
        let source_element_count = tensor_element_count(tensor)?;
        let k_element_count = split
            .head_count
            .checked_mul(split.qk_nope_head_dim)
            .and_then(|value| value.checked_mul(split.kv_lora_rank))
            .context("GLM-DSA attn_k_b element count overflow")?;
        let v_element_count = split
            .head_count
            .checked_mul(split.value_head_dim)
            .and_then(|value| value.checked_mul(split.kv_lora_rank))
            .context("GLM-DSA attn_v_b element count overflow")?;
        Ok(vec![
            Self {
                segments: vec![TensorSegment {
                    file_index,
                    source_name: tensor.name().to_string(),
                    source_dtype,
                    target_dtype,
                    element_count: source_element_count,
                    source_byte_len: tensor.byte_len(),
                    target_byte_len: tensor_byte_len(k_element_count, target_dtype)?,
                    transform: TensorTransform::GlmDsaKvB {
                        split,
                        part: GlmDsaKvBPart::K,
                    },
                }],
                name: format!("blk.{layer}.attn_k_b.weight"),
                dims: vec![split.qk_nope_head_dim, split.kv_lora_rank, split.head_count],
                ggml_type: ggml_type_for_dtype(target_dtype),
                byte_len: tensor_byte_len(k_element_count, target_dtype)?,
                gguf_offset: 0,
            },
            Self {
                segments: vec![TensorSegment {
                    file_index,
                    source_name: tensor.name().to_string(),
                    source_dtype,
                    target_dtype,
                    element_count: source_element_count,
                    source_byte_len: tensor.byte_len(),
                    target_byte_len: tensor_byte_len(v_element_count, target_dtype)?,
                    transform: TensorTransform::GlmDsaKvB {
                        split,
                        part: GlmDsaKvBPart::V,
                    },
                }],
                name: format!("blk.{layer}.attn_v_b.weight"),
                dims: vec![split.kv_lora_rank, split.value_head_dim, split.head_count],
                ggml_type: ggml_type_for_dtype(target_dtype),
                byte_len: tensor_byte_len(v_element_count, target_dtype)?,
                gguf_offset: 0,
            },
        ])
    }
}

pub(super) fn stream_transformed_segment(
    writer: &mut File,
    file: &SafetensorFile,
    segment: &TensorSegment,
    buffer_size: usize,
) -> Result<Option<u64>> {
    let TensorTransform::GlmDsaKvB { split, part } = segment.transform else {
        return Ok(None);
    };
    stream_glm_dsa_kv_b_split(writer, file, segment, buffer_size, split, part).map(Some)
}

fn stream_glm_dsa_kv_b_split(
    writer: &mut File,
    file: &SafetensorFile,
    segment: &TensorSegment,
    buffer_size: usize,
    split: GlmDsaKvBSplitConfig,
    part: GlmDsaKvBPart,
) -> Result<u64> {
    let mut source = Vec::with_capacity(
        usize::try_from(segment.source_byte_len)
            .context("GLM-DSA kv_b source byte length does not fit usize")?,
    );
    let mut source_bytes = 0_u64;
    file.stream_tensor_chunks(&segment.source_name, buffer_size, |chunk| {
        source.extend_from_slice(chunk);
        source_bytes += chunk.len() as u64;
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
        source.len() % segment.source_dtype.byte_size() as usize == 0,
        "GLM-DSA kv_b source split an element boundary"
    );

    let combined_head_dim = split.qk_nope_head_dim + split.value_head_dim;
    let mut output_bytes = 0_u64;
    let flush_limit = buffer_size.max(segment.target_dtype.byte_size() as usize);
    let mut output = Vec::with_capacity(flush_limit);
    for head in 0..split.head_count {
        match part {
            GlmDsaKvBPart::K => {
                for lora in 0..split.kv_lora_rank {
                    for qk in 0..split.qk_nope_head_dim {
                        output_bytes += push_transformed_float(
                            writer,
                            &source,
                            segment.source_dtype,
                            segment.target_dtype,
                            &mut output,
                            flush_limit,
                            (head * combined_head_dim + qk) * split.kv_lora_rank + lora,
                        )?;
                    }
                }
            }
            GlmDsaKvBPart::V => {
                for value in 0..split.value_head_dim {
                    for lora in 0..split.kv_lora_rank {
                        output_bytes += push_transformed_float(
                            writer,
                            &source,
                            segment.source_dtype,
                            segment.target_dtype,
                            &mut output,
                            flush_limit,
                            (head * combined_head_dim + split.qk_nope_head_dim + value)
                                * split.kv_lora_rank
                                + lora,
                        )?;
                    }
                }
            }
        }
    }
    writer.write_all(&output)?;
    ensure!(
        output_bytes == segment.target_byte_len,
        "wrote {} bytes for transformed {}, expected {}",
        output_bytes,
        segment.source_name,
        segment.target_byte_len
    );
    Ok(output_bytes)
}

fn push_transformed_float(
    writer: &mut File,
    source: &[u8],
    source_dtype: FloatDType,
    target_dtype: FloatDType,
    output: &mut Vec<u8>,
    flush_limit: usize,
    source_index: u64,
) -> Result<u64> {
    let source_index =
        usize::try_from(source_index).context("GLM-DSA kv_b source index does not fit usize")?;
    let source_width = source_dtype.byte_size() as usize;
    ensure!(
        source_index
            .checked_add(1)
            .and_then(|count| count.checked_mul(source_width))
            .is_some_and(|end| end <= source.len()),
        "GLM-DSA kv_b source index {source_index} is outside source tensor"
    );
    let value = read_float_element(source, source_dtype, source_index);
    write_float_element(output, target_dtype, value);
    let written = target_dtype.byte_size();
    if output.len() >= flush_limit {
        writer.write_all(output)?;
        output.clear();
    }
    Ok(written)
}

fn metadata_has_key(metadata: &[GgufKv], key: &str) -> bool {
    metadata.iter().any(|kv| kv.key() == key)
}

fn metadata_string<'a>(metadata: &'a [GgufKv], key: &str) -> Option<&'a str> {
    metadata.iter().find_map(|kv| match kv {
        GgufKv::String {
            key: item_key,
            value,
        } if item_key == key => Some(value.as_str()),
        _ => None,
    })
}

fn is_glm_dsa_metadata(metadata: &[GgufKv]) -> bool {
    metadata_string(metadata, "general.architecture") == Some("glm-dsa")
}

fn metadata_u32(metadata: &[GgufKv], key: &str) -> Option<u32> {
    metadata.iter().find_map(|kv| match kv {
        GgufKv::U32 {
            key: item_key,
            value,
        } if item_key == key => Some(*value),
        _ => None,
    })
}

impl GgufKv {
    fn key(&self) -> &str {
        match self {
            Self::ArrayF32 { key, .. }
            | Self::ArrayI32 { key, .. }
            | Self::ArrayString { key, .. }
            | Self::Bool { key, .. }
            | Self::F32 { key, .. }
            | Self::I32 { key, .. }
            | Self::String { key, .. }
            | Self::U16 { key, .. }
            | Self::U32 { key, .. }
            | Self::U64 { key, .. } => key,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum TensorTransform {
    Identity,
    GlmDsaKvB {
        split: GlmDsaKvBSplitConfig,
        part: GlmDsaKvBPart,
    },
}

#[derive(Debug, Clone, Copy)]
pub(super) enum GlmDsaKvBSplitMode {
    Disabled,
    MissingMetadata,
    Config(GlmDsaKvBSplitConfig),
}

#[derive(Debug, Clone, Copy)]
pub(super) struct GlmDsaKvBSplitConfig {
    head_count: u64,
    qk_nope_head_dim: u64,
    value_head_dim: u64,
    kv_lora_rank: u64,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum GlmDsaKvBPart {
    K,
    V,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_indexshare_types_from_mapped_tensor_names() {
        let mut metadata = minimal_metadata(3);
        let tensors = [0_u32, 2]
            .into_iter()
            .flat_map(|layer| {
                INDEXER_TENSORS
                    .iter()
                    .map(move |suffix| mock_tensor_source(&format!("blk.{layer}.{suffix}")))
            })
            .collect::<Vec<_>>();

        enrich_glm_dsa_indexshare_metadata(&mut metadata, &tensors).unwrap();

        assert_eq!(
            string_array(&metadata, "glm-dsa.attention.indexer.types"),
            Some(vec!["full", "shared", "full"])
        );
    }

    #[test]
    fn rejects_partial_indexshare_group() {
        let mut metadata = minimal_metadata(2);
        let tensors = vec![mock_tensor_source("blk.0.indexer.k_norm.weight")];

        let err = enrich_glm_dsa_indexshare_metadata(&mut metadata, &tensors).unwrap_err();

        assert!(
            err.to_string().contains("partial indexer tensor group"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn detects_kv_b_projection_layer() {
        assert_eq!(
            glm_dsa_kv_b_layer("model.layers.12.self_attn.kv_b_proj.weight").unwrap(),
            Some(12)
        );
        assert_eq!(
            glm_dsa_kv_b_layer("model.layers.12.self_attn.q_b_proj.weight").unwrap(),
            None
        );
    }

    fn minimal_metadata(block_count: u32) -> Vec<GgufKv> {
        vec![
            GgufKv::string("general.architecture", "glm-dsa"),
            GgufKv::u32("glm-dsa.block_count", block_count),
        ]
    }

    fn mock_tensor_source(name: &str) -> TensorSource {
        TensorSource {
            segments: Vec::new(),
            name: name.to_string(),
            dims: vec![1],
            ggml_type: 0,
            byte_len: 4,
            gguf_offset: 0,
        }
    }

    fn string_array<'a>(metadata: &'a [GgufKv], key: &str) -> Option<Vec<&'a str>> {
        metadata.iter().find_map(|kv| match kv {
            GgufKv::ArrayString {
                key: item_key,
                value,
            } if item_key == key => Some(value.iter().map(String::as_str).collect()),
            _ => None,
        })
    }
}
