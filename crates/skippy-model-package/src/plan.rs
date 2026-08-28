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
    #[serde(skip_serializing)]
    pub(crate) includes_per_layer_token_embd: bool,
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
            .filter(|tensor| {
                tensor_in_stage(tensor, tensors, stage_index, stages, layer_start, layer_end)
            })
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
                    includes_per_layer_token_embd: tensors
                        .iter()
                        .any(|tensor| is_per_layer_token_embd(&tensor.name)),
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
                tensors,
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
        includes_per_layer_token_embd: selected
            .iter()
            .any(|tensor| is_per_layer_token_embd(&tensor.name)),
        tensor_count: selected.len(),
        tensor_bytes: selected.iter().map(|tensor| tensor.byte_size).sum(),
    }
}

fn tensor_in_stage(
    tensor: &TensorInfo,
    all_tensors: &[TensorInfo],
    stage_index: usize,
    stages: usize,
    layer_start: u32,
    layer_end: u32,
) -> bool {
    tensor_in_explicit_stage(
        tensor,
        all_tensors,
        layer_start,
        layer_end,
        stage_index == 0,
        stage_index + 1 == stages,
    )
}

fn tensor_in_explicit_stage(
    tensor: &TensorInfo,
    all_tensors: &[TensorInfo],
    layer_start: u32,
    layer_end: u32,
    includes_embeddings: bool,
    includes_output: bool,
) -> bool {
    if is_per_layer_token_embd(&tensor.name) {
        return per_layer_embedding_retained(all_tensors, layer_start, layer_end);
    }
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

pub(crate) const PER_LAYER_TOKEN_EMBD: &str = "per_layer_token_embd.weight";

fn is_per_layer_token_embd(name: &str) -> bool {
    name == PER_LAYER_TOKEN_EMBD
}

/// Decide whether a stage must retain the shared per-layer token embedding table.
///
/// The table is classified as an embedding tensor, but it is gathered from by
/// per-layer consumers rather than by the stage that owns the token embeddings.
/// Selecting it on embedding ownership alone drops it from a stage that needs
/// it; selecting it unconditionally ships a very large table to every stage.
///
/// Only qwen4exp is known to consume it from a *sparse* subset of layers
/// (`blk.N.ple_*`; `qwen4exp.ple.layers = [1]` on Qwen3.8-Flash-Next), so only
/// one stage of a split can ever read it. Retain it only with those consumers.
///
/// Retain the table on every stage for every other artifact. This matters:
/// Gemma3n/Gemma4 gather the same table through per-block tensors named
/// `blk.N.inp_gate`, `blk.N.proj`
/// and `blk.N.post_norm` (`llama-arch.cpp:568-570`) — *not* `per_layer_*` — so
/// a name-based consumer scan finds nothing for them. Failing closed here would
/// silently drop their table from every stage.
fn per_layer_embedding_retained(
    all_tensors: &[TensorInfo],
    layer_start: u32,
    layer_end: u32,
) -> bool {
    let mut saw_sparse_consumer = false;
    let mut retained = false;
    for tensor in all_tensors {
        if !is_sparse_per_layer_embedding_consumer(&tensor.name) {
            continue;
        }
        saw_sparse_consumer = true;
        if matches!(tensor.layer_index, Some(layer) if layer >= layer_start && layer < layer_end) {
            retained = true;
            break;
        }
    }
    !saw_sparse_consumer || retained
}

fn is_sparse_per_layer_embedding_consumer(name: &str) -> bool {
    name.split_once('.')
        .and_then(|(_, rest)| rest.split_once('.'))
        .is_some_and(|(_, suffix)| suffix.starts_with("ple_"))
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
