use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_ffi::TensorRole;
use skippy_runtime::TensorInfo;

use crate::write::ModelSource;

#[derive(Debug, Serialize)]
pub(crate) struct PlanOutput {
    pub(crate) schema_version: u32,
    pub(crate) stage_count: usize,
    pub(crate) layer_count: u32,
    pub(crate) stages: Vec<StagePlan>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct StagePlan {
    pub(crate) stage_index: usize,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) includes_embeddings: bool,
    pub(crate) includes_output: bool,
    pub(crate) tensor_count: usize,
    pub(crate) tensor_bytes: u64,
}

pub(crate) fn build_plan(model: &Path, stages: usize) -> Result<PlanOutput> {
    if stages == 0 {
        bail!("--stages must be greater than zero");
    }
    let source = ModelSource::open(model)?;
    build_plan_from_tensors(stages, &source.tensors)
}

pub(crate) fn build_plan_from_tensors(stages: usize, tensors: &[TensorInfo]) -> Result<PlanOutput> {
    let layer_count = layer_count(tensors)?;
    if stages as u32 > layer_count {
        bail!("--stages must not exceed model layer count {layer_count}");
    }
    let ranges = partition_layers(layer_count, stages);
    let mut stage_tensors: BTreeMap<usize, Vec<&TensorInfo>> = BTreeMap::new();
    for (stage_index, (layer_start, layer_end)) in ranges.iter().copied().enumerate() {
        let tensors_for_stage = tensors
            .iter()
            .filter(|tensor| tensor_in_stage(tensor, stage_index, stages, layer_start, layer_end))
            .collect();
        stage_tensors.insert(stage_index, tensors_for_stage);
    }

    Ok(PlanOutput {
        schema_version: 1,
        stage_count: stages,
        layer_count,
        stages: ranges
            .into_iter()
            .enumerate()
            .map(|(stage_index, (layer_start, layer_end))| {
                let tensors = stage_tensors.remove(&stage_index).unwrap_or_default();
                StagePlan {
                    stage_index,
                    layer_start,
                    layer_end,
                    includes_embeddings: stage_index == 0,
                    includes_output: stage_index + 1 == stages,
                    tensor_count: tensors.len(),
                    tensor_bytes: tensors.iter().map(|tensor| tensor.byte_size).sum(),
                }
            })
            .collect(),
    })
}

pub(crate) fn layer_count(tensors: &[TensorInfo]) -> Result<u32> {
    tensors
        .iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|max_layer| max_layer + 1)
        .context("model has no layer tensors")
}

pub(crate) fn stage_plan_from_tensors(
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    includes_embeddings: bool,
    includes_output: bool,
    tensors: &[TensorInfo],
) -> StagePlan {
    let selected = tensors
        .iter()
        .filter(|tensor| {
            tensor_in_explicit_stage(
                tensor,
                layer_start,
                layer_end,
                includes_embeddings,
                includes_output,
            )
        })
        .collect::<Vec<_>>();
    StagePlan {
        stage_index,
        layer_start,
        layer_end,
        includes_embeddings,
        includes_output,
        tensor_count: selected.len(),
        tensor_bytes: selected.iter().map(|tensor| tensor.byte_size).sum(),
    }
}

fn tensor_in_stage(
    tensor: &TensorInfo,
    stage_index: usize,
    stages: usize,
    layer_start: u32,
    layer_end: u32,
) -> bool {
    tensor_in_explicit_stage(
        tensor,
        layer_start,
        layer_end,
        stage_index == 0,
        stage_index + 1 == stages,
    )
}

fn tensor_in_explicit_stage(
    tensor: &TensorInfo,
    layer_start: u32,
    layer_end: u32,
    includes_embeddings: bool,
    includes_output: bool,
) -> bool {
    matches!(
        tensor.layer_index,
        Some(layer) if layer >= layer_start && layer < layer_end
    ) || (includes_embeddings && tensor.role == TensorRole::Embedding)
        || (includes_output && matches!(tensor.role, TensorRole::FinalNorm | TensorRole::Output))
        || matches!(
            tensor.role,
            TensorRole::Metadata | TensorRole::Tokenizer | TensorRole::Unknown
        )
}

pub(crate) fn parse_layer_range(layers: &str) -> Result<(u32, u32)> {
    let Some((start, end)) = layers.split_once("..") else {
        bail!("--layers must use START..END syntax");
    };
    let start = start.parse::<u32>().context("parse layer range start")?;
    let end = end.parse::<u32>().context("parse layer range end")?;
    if start >= end {
        bail!("layer range start must be less than end");
    }
    Ok((start, end))
}

pub(crate) fn partition_layers(layer_count: u32, stages: usize) -> Vec<(u32, u32)> {
    let base = layer_count / stages as u32;
    let extra = layer_count % stages as u32;
    let mut start = 0;
    (0..stages)
        .map(|stage_index| {
            let width = base + u32::from((stage_index as u32) < extra);
            let end = start + width;
            let range = (start, end);
            start = end;
            range
        })
        .collect()
}
