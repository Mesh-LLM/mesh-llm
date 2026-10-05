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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlannedArtifactKind {
    /// Every non-layer tensor. Physical grouping never assigns stage
    /// ownership; exact graph closure does that at admission time.
    Common,
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
    let mut layers: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for tensor in tensors {
        match tensor.role {
            TensorRole::Layer => {
                let ordinal = tensor.layer_index.ok_or_else(|| {
                    anyhow::anyhow!("layer tensor {:?} has no layer index", tensor.name)
                })?;
                layers.entry(ordinal).or_default().push(tensor.name.clone());
            }
            TensorRole::Embedding
            | TensorRole::FinalNorm
            | TensorRole::Output
            | TensorRole::Metadata
            | TensorRole::Tokenizer
            | TensorRole::Unknown => common.push(tensor.name.clone()),
        }
    }
    common.sort();

    let mut planned = Vec::new();
    if !common.is_empty() {
        let split = split_oversized_group(&common, tensors, max_artifact_bytes, "common")?;
        if split.len() == 1 {
            planned.push(PlannedArtifact {
                id: "common".to_string(),
                path: "shared/common.gguf".to_string(),
                kind: PlannedArtifactKind::Common,
                tensor_names: common,
            });
        } else {
            for (index, part) in split.into_iter().enumerate() {
                planned.push(PlannedArtifact {
                    id: format!("common-part{index:02}"),
                    path: format!("shared/common-part{index:02}.gguf"),
                    kind: PlannedArtifactKind::Common,
                    tensor_names: part,
                });
            }
        }
    }
    for (ordinal, mut names) in layers {
        names.sort();
        let group = format!("layer {ordinal}");
        let split = split_oversized_group(&names, tensors, max_artifact_bytes, &group)?;
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

/// Split a sorted tensor-name group into deterministic budget-bounded parts.
/// A single tensor larger than the budget stays whole because it is already
/// the smallest indivisible unit. Parts are nonempty and every input name is
/// assigned exactly once.
fn split_oversized_group(
    names: &[String],
    tensors: &[TensorInfo],
    max_artifact_bytes: u64,
    group: &str,
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
    anyhow::ensure!(!names.is_empty(), "{group} has no tensors to subdivide");
    let mut parts = Vec::new();
    let mut current = Vec::new();
    let mut current_bytes = 0_u64;
    for name in names {
        let tensor_bytes = bytes_of(name);
        if !current.is_empty() && current_bytes.saturating_add(tensor_bytes) > max_artifact_bytes {
            parts.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current.push(name.clone());
        current_bytes = current_bytes.saturating_add(tensor_bytes);
    }
    if !current.is_empty() {
        parts.push(current);
    }
    let assigned: usize = parts.iter().map(Vec::len).sum();
    anyhow::ensure!(
        assigned == names.len(),
        "{group} budget split dropped tensors"
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
