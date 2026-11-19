//! Whether an audit can check the weights a stage multiplies.
//!
//! An audit checks a stage's matrix products with the rounding profile of
//! its backend for each weight type
//! ([`skippy_receipts::rounding_profiles::profiles_for`]), and a pair
//! without a profile makes every audit of the stage end as no audit: f16
//! and bf16 weights on CUDA, for example, or Q4_1 anywhere. A stage that
//! issued receipts for such a model would look audited to a client that
//! asks for receipts, and never be. So it issues none, as on a backend no
//! audit covers ([`super::audited_backend`]).
//!
//! The stage reads the types from the tensor directories of its GGUF
//! files, which costs a header read per file. It checks the matrices of
//! its own layers, as a client does, and the output head on the final
//! stage. A model whose files it cannot read that way (a checkpoint that
//! is not a GGUF) keeps its receipts, and the client's coverage check
//! decides.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use skippy_protocol::StageConfig;
use skippy_receipts::rounding_profiles::{BackendFamily, unprofiled};

/// A tensor of the stage's files, as far as its type matters here.
struct Weight {
    name: String,
    layer: Option<u32>,
    ggml_type: u32,
}

/// Why a stage of `config` on `backend` issues no receipts, when an audit
/// cannot check its weights.
pub(super) fn unaudited(config: &StageConfig, backend: &str) -> Option<String> {
    let family = BackendFamily::of(backend)?;
    let tensors = match stage_tensors(config) {
        Ok(tensors) => tensors,
        Err(error) => {
            tracing::debug!(
                stage = config.stage_index,
                "the stage's weight types are unknown, so the audit decides: {error:#}"
            );
            return None;
        }
    };
    let holds_head = config.downstream.is_none();
    let reason = unprofiled(
        family,
        multiplied(&tensors, config.layer_start..config.layer_end, holds_head),
    )?;
    Some(format!("{reason}, so no audit can check this stage"))
}

/// The tensor directory of the stage's GGUF files.
fn stage_tensors(config: &StageConfig) -> Result<Vec<Weight>> {
    let files: Vec<PathBuf> = if config.model_part_paths.is_empty() {
        let path = config
            .model_path
            .as_deref()
            .context("the stage names no model file")?;
        skippy_runtime::gguf_shard_paths(Path::new(path))?
    } else {
        config.model_part_paths.iter().map(PathBuf::from).collect()
    };
    let mut tensors = Vec::new();
    for file in files {
        let info = skippy_runtime::ModelInfo::open(&file)
            .with_context(|| format!("read the tensors of {}", file.display()))?;
        tensors.extend(info.tensors()?.into_iter().map(|t| Weight {
            name: t.name,
            layer: t.layer_index,
            ggml_type: t.ggml_type,
        }));
    }
    Ok(tensors)
}

/// The weights of `tensors` a stage of `layers` multiplies, as a client
/// checks them: every weight of its layers (a vector among them, such as
/// a norm, is f32 in a GGUF, which has a profile on every backend), and
/// the output head when the stage holds it (`output.weight`, else the
/// token embeddings of a tied head).
fn multiplied(
    tensors: &[Weight],
    layers: std::ops::Range<u32>,
    holds_head: bool,
) -> impl Iterator<Item = (&str, u32)> {
    let head = ["output.weight", "token_embd.weight"]
        .into_iter()
        .find_map(|name| tensors.iter().find(|t| t.name == name))
        .filter(|_| holds_head);
    tensors
        .iter()
        .filter(move |t| {
            t.name.ends_with(".weight") && t.layer.is_some_and(|l| layers.contains(&l))
        })
        .chain(head)
        .map(|t| (t.name.as_str(), t.ggml_type))
}

#[cfg(test)]
mod tests {
    use super::*;

    const F32: u32 = 0;
    const F16: u32 = 1;
    const Q8_0: u32 = 8;

    fn tensor(name: &str, layer: Option<u32>, ggml_type: u32) -> Weight {
        Weight {
            name: name.into(),
            layer,
            ggml_type,
        }
    }

    fn reason(tensors: &[Weight], family: BackendFamily, holds_head: bool) -> Option<String> {
        unprofiled(family, multiplied(tensors, 0..2, holds_head))
    }

    #[test]
    fn f16_weights_on_cuda_stop_receipts_and_on_the_cpu_do_not() {
        let f16 = [
            tensor("token_embd.weight", None, F16),
            tensor("blk.0.attn_norm.weight", Some(0), F32),
            tensor("blk.0.attn_q.weight", Some(0), F16),
            tensor("blk.1.attn_q.weight", Some(1), F16),
        ];
        let why = reason(&f16, BackendFamily::Cuda, false).unwrap();
        assert!(why.contains("f16 weights (blk.0.attn_q.weight)"), "{why}");
        assert_eq!(reason(&f16, BackendFamily::Cpu, true), None);
    }

    #[test]
    fn only_the_stage_layers_and_its_head_count() {
        let tensors = [
            tensor("token_embd.weight", None, Q8_0),
            tensor("blk.1.attn_q.weight", Some(1), Q8_0),
            // a later stage's layer
            tensor("blk.2.attn_q.weight", Some(2), F16),
            tensor("output.weight", None, F16),
        ];
        assert_eq!(reason(&tensors, BackendFamily::Cuda, false), None);
        let why = reason(&tensors, BackendFamily::Cuda, true).unwrap();
        assert!(why.contains("(output.weight)"), "{why}");
        // a tied head is the token embeddings
        let tied = [
            tensor("token_embd.weight", None, F16),
            tensor("blk.0.attn_q.weight", Some(0), Q8_0),
        ];
        assert_eq!(reason(&tied, BackendFamily::Cuda, false), None);
        let why = reason(&tied, BackendFamily::Cuda, true).unwrap();
        assert!(why.contains("(token_embd.weight)"), "{why}");
    }

    #[test]
    fn a_stage_without_readable_weights_keeps_its_receipts() {
        let config = StageConfig {
            model_path: Some("/nonexistent/model.safetensors".into()),
            ..StageConfig::default()
        };
        assert_eq!(unaudited(&config, "CUDA0"), None);
        // a backend no audit covers is said by `audited_backend`
        assert_eq!(unaudited(&config, "Vulkan0"), None);
    }
}
