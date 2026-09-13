//! Artifact layout planning for repacked v2 packages.
//!
//! Decides which payload-bearing artifact each source tensor binds to, using
//! the native role classifier. The metadata carrier is planning metadata, not
//! storage: no tensor binds to it. Every logical tensor binds exactly once to
//! an artifact with a real stored extent.
use std::collections::BTreeMap;

use anyhow::Result;
use skippy_ffi::TensorRole;
use skippy_runtime::TensorInfo;

/// Default per-artifact payload ceiling for HF Jobs split packages. The HF
/// Jobs kubelet evicts pods above 50G of container-local ephemeral storage, so
/// payload artifacts are kept well below that even while several parts and
/// their upload buffers coexist on local disk. Oversized model layers are
/// subdivided into byte-balanced part artifacts instead.
pub(crate) const DEFAULT_MAX_ARTIFACT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// A payload artifact the writer must emit, with the source tensor names that
/// bind to it. `layer` is set only for per-layer artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedArtifact {
    pub(crate) id: String,
    /// Relative package path; the verifier derives layer ordinals from
    /// `layers/layer-N.gguf` (and `layers/layer-N-partMM.gguf`) naming.
    pub(crate) path: String,
    pub(crate) kind: PlannedArtifactKind,
    /// Canonical tensor names bound to this artifact. Repeated copies emitted
    /// by the native slice writer stay unreferenced.
    pub(crate) tensor_names: Vec<String>,
}

impl PlannedArtifact {
    /// Whether this artifact is one of several byte-balanced parts of an
    /// oversized layer, written by the Rust part writer rather than the native
    /// whole-layer slice writer.
    pub(crate) fn is_part(&self) -> bool {
        self.id.contains("-part")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlannedArtifactKind {
    /// Metadata/Tokenizer/Unknown-role payload tensors. Emitted only when the
    /// set is nonempty; a plain slice carries them with real payload.
    Common,
    Embeddings,
    /// FinalNorm + Output role tensors.
    Output,
    Layer {
        ordinal: u32,
    },
}

/// Assign every native tensor to exactly one payload artifact, subdividing
/// artifacts whose payload would exceed `max_artifact_bytes`.
///
/// `tensors` must be the complete native inventory of the source. Fails if any
/// tensor cannot be bound, so the writer never produces a package with an
/// unowned tensor. A single tensor larger than the budget stays whole: it is
/// already the smallest indivisible unit, and failing the package on it would
/// publish nothing at all.
pub(crate) fn plan_artifacts_with_budget(
    tensors: &[TensorInfo],
    max_artifact_bytes: u64,
) -> Result<Vec<PlannedArtifact>> {
    ensure_budget(max_artifact_bytes)?;
    let mut common: Vec<String> = Vec::new();
    let mut embeddings: Vec<String> = Vec::new();
    let mut output: Vec<String> = Vec::new();
    let mut layers: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for tensor in tensors {
        match tensor.role {
            TensorRole::Layer => {
                let ordinal = tensor.layer_index.ok_or_else(|| {
                    anyhow::anyhow!("layer tensor {:?} has no layer index", tensor.name)
                })?;
                layers.entry(ordinal).or_default().push(tensor.name.clone());
            }
            TensorRole::Embedding => embeddings.push(tensor.name.clone()),
            TensorRole::FinalNorm | TensorRole::Output => output.push(tensor.name.clone()),
            TensorRole::Metadata | TensorRole::Tokenizer | TensorRole::Unknown => {
                common.push(tensor.name.clone())
            }
        }
    }
    common.sort();
    embeddings.sort();
    output.sort();

    let mut planned = Vec::new();
    if !common.is_empty() {
        planned.push(PlannedArtifact {
            id: "common".to_string(),
            path: "shared/common.gguf".to_string(),
            kind: PlannedArtifactKind::Common,
            tensor_names: common,
        });
    }
    if !embeddings.is_empty() {
        planned.push(PlannedArtifact {
            id: "embeddings".to_string(),
            path: "shared/embeddings.gguf".to_string(),
            kind: PlannedArtifactKind::Embeddings,
            tensor_names: embeddings,
        });
    }
    if !output.is_empty() {
        planned.push(PlannedArtifact {
            id: "output".to_string(),
            path: "shared/output.gguf".to_string(),
            kind: PlannedArtifactKind::Output,
            tensor_names: output,
        });
    }
    for (ordinal, mut names) in layers {
        names.sort();
        let split = split_oversized_group(&names, tensors, max_artifact_bytes, ordinal)?;
        if split.len() == 1 {
            planned.push(PlannedArtifact {
                id: format!("layer-{ordinal:05}"),
                path: format!("layers/layer-{ordinal:05}.gguf"),
                kind: PlannedArtifactKind::Layer { ordinal },
                tensor_names: names,
            });
            continue;
        }
        for (index, part) in split.into_iter().enumerate() {
            planned.push(PlannedArtifact {
                id: format!("layer-{ordinal:05}-part{index:02}"),
                path: format!("layers/layer-{ordinal:05}-part{index:02}.gguf"),
                kind: PlannedArtifactKind::Layer { ordinal },
                tensor_names: part,
            });
        }
    }
    Ok(planned)
}

/// Split a sorted tensor-name group into byte-balanced part groups when its
/// payload exceeds `max_artifact_bytes`. A single tensor larger than the budget
/// stays whole: it is already the smallest indivisible unit, and failing the
/// package on it would publish nothing at all. Parts are nonempty and every
/// input name is assigned exactly once.
fn split_oversized_group(
    names: &[String],
    tensors: &[TensorInfo],
    max_artifact_bytes: u64,
    ordinal: u32,
) -> Result<Vec<Vec<String>>> {
    let bytes_of = |name: &str| -> u64 {
        tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .map(|tensor| tensor.byte_size)
            .unwrap_or(0)
    };
    let total: u64 = names.iter().map(|name| bytes_of(name)).sum();
    if total <= max_artifact_bytes {
        return Ok(vec![names.to_vec()]);
    }
    anyhow::ensure!(
        !names.is_empty(),
        "layer {ordinal} has no tensors to subdivide"
    );
    // A dominating tensor is the smallest indivisible unit: keep it whole
    // instead of emitting parts that still exceed the budget.
    if names.len() == 1 || bytes_of(&names[0]) > max_artifact_bytes {
        return Ok(vec![names.to_vec()]);
    }
    let split_count = usize::try_from(total.div_ceil(max_artifact_bytes))
        .unwrap_or(usize::MAX)
        .clamp(2, names.len());
    // Byte-balanced boundaries mirroring the safetensors GGUF splitter
    // (`byte_balanced_split_boundaries`): close part k once accumulated bytes
    // reach k/split_count of the layer total, keeping enough tensors back to
    // keep every remaining part nonempty.
    let mut boundaries = vec![0_usize];
    let mut accumulated: u128 = 0;
    for (index, name) in names.iter().enumerate() {
        accumulated += u128::from(bytes_of(name));
        let remaining_tensors = names.len() - (index + 1);
        let remaining_splits = split_count - (boundaries.len() - 1);
        if boundaries.len() < split_count && remaining_tensors >= remaining_splits {
            let target = u128::from(total) * boundaries.len() as u128 / split_count as u128;
            if accumulated >= target {
                boundaries.push(index + 1);
            }
        }
    }
    while boundaries.len() < split_count {
        let next = boundaries.last().copied().unwrap_or(0) + 1;
        boundaries.push(next);
    }
    boundaries.push(names.len());
    let mut parts = Vec::with_capacity(split_count);
    for window in boundaries.windows(2) {
        let part = &names[window[0]..window[1]];
        anyhow::ensure!(
            !part.is_empty(),
            "layer {ordinal} byte-balanced split produced an empty part"
        );
        parts.push(part.to_vec());
    }
    let assigned: usize = parts.iter().map(Vec::len).sum();
    anyhow::ensure!(
        assigned == names.len(),
        "layer {ordinal} byte-balanced split dropped tensors"
    );
    Ok(parts)
}

fn ensure_budget(max_artifact_bytes: u64) -> Result<()> {
    anyhow::ensure!(
        max_artifact_bytes > 0,
        "max artifact bytes must be greater than zero"
    );
    Ok(())
}

#[cfg(test)]
mod tests;
