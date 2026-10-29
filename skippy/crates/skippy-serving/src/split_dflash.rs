//! DFlash speculative decoding across a split target.
//!
//! The draft runs on the final stage, which owns the output head it borrows
//! and already returns native MTP drafts on its replies. Every stage that owns
//! one of the draft's target layers captures that layer's input and forwards
//! it with its activations (see `skippy_runtime::DFlashFeatureRows`). Stage 0
//! drives the decode loop as it does for split native MTP: it sends drafted
//! blocks as verify windows, commits through the first mismatch, and rolls
//! every stage back to the committed position. The final stage drafts the next
//! block right after each decode or verify and returns it on the reply.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use skippy_protocol::StageConfig;
use skippy_runtime::DFlashDraftOptions;

use crate::frontend::{DFlashProposalConfig, SpeculativeDecodeConfig};
use crate::runtime_state::{RuntimeLaunchOverrides, RuntimeState};

/// Draft tokens in a trained 16-token block, the most a plan without a cap
/// can propose.
const UNCAPPED_DRAFT_TOKENS: usize = 15;
/// The wire carries a requested draft depth in four bits.
const MAX_WIRE_DRAFT_TOKENS: usize = 15;

/// One stage's part in a split DFlash plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SplitDFlashStage {
    /// Every target layer the draft reads, across all stages.
    pub target_layers: Vec<u32>,
    /// The draft, on the final stage only.
    pub draft_model_path: Option<PathBuf>,
    pub max_draft_tokens: Option<usize>,
    pub draft_n_gpu_layers: Option<i32>,
}

impl SplitDFlashStage {
    /// The plan for a stage of a split target served with `dflash`, reading
    /// the draft's target layers from its header.
    pub fn from_draft(
        dflash: &DFlashProposalConfig,
        final_stage: bool,
        draft_n_gpu_layers: Option<i32>,
    ) -> Result<Self> {
        let target_layers = draft_target_layers(&dflash.draft_model_path)?;
        Ok(Self {
            target_layers,
            draft_model_path: final_stage.then(|| dflash.draft_model_path.clone()),
            max_draft_tokens: dflash.max_draft_tokens,
            draft_n_gpu_layers,
        })
    }

    /// Recurrent rollback depth every stage needs to verify the longest block
    /// in place: every drafted token plus the anchor.
    pub fn recurrent_rollback_tokens(&self) -> u32 {
        let drafted = self.max_draft_tokens.unwrap_or(UNCAPPED_DRAFT_TOKENS);
        u32::try_from(drafted.saturating_add(1)).unwrap_or(u32::MAX)
    }
}

/// The target layers a DFlash draft reads, from its GGUF header.
pub fn draft_target_layers(path: &Path) -> Result<Vec<u32>> {
    let layers = skippy_model_artifact::gguf::scan_gguf_dflash_target_layers(path)
        .with_context(|| format!("read DFlash target layers from {}", path.display()))?;
    if layers.is_empty() {
        bail!("DFlash draft {} names no target layers", path.display());
    }
    Ok(layers)
}

/// Attaches this stage's part of a split DFlash plan before any serving
/// session exists: the draft on the final stage, target-layer capture on the
/// others. A final stage without a usable draft serves without one; stage 0
/// then receives no drafts and decodes serially.
pub(crate) fn attach_split_dflash(
    runtime: &mut RuntimeState,
    config: &StageConfig,
    overrides: &RuntimeLaunchOverrides,
    plan: &SplitDFlashStage,
) -> Result<()> {
    if config.downstream.is_none() {
        let Some(path) = plan.draft_model_path.as_deref() else {
            return Ok(());
        };
        if !path.is_file() {
            tracing::warn!(
                draft = %path.display(),
                "split DFlash draft is not on this node; serving without speculation"
            );
            return Ok(());
        }
        let target_config =
            crate::runtime_state::runtime_config_from_stage_config(config, overrides)?;
        let options = DFlashDraftOptions {
            max_draft_tokens: plan.max_draft_tokens,
            n_gpu_layers: plan.draft_n_gpu_layers,
            device: None,
        };
        let info = runtime
            .model
            .attach_dflash_draft(path, &options, &target_config)
            .with_context(|| format!("attach split DFlash draft {}", path.display()))?;
        tracing::info!(
            stage = config.stage_id,
            variant = info.variant.as_str(),
            max_draft_tokens = info.max_draft_tokens,
            target_layers = ?info.target_layer_ids,
            "split DFlash draft attached on the final stage"
        );
        return Ok(());
    }
    let layers = runtime.model.attach_dflash_capture(
        &plan.target_layers,
        config.layer_start,
        config.layer_end,
    )?;
    tracing::info!(
        stage = config.stage_id,
        layers = ?layers,
        "split DFlash target-layer capture attached"
    );
    Ok(())
}

/// Stage 0's decode plan for a split DFlash target: the final stage's DFlash
/// drafts arrive where native MTP drafts do, so stage 0 drives them through
/// the native MTP verify windows. Each window verifies the whole block.
/// `None` when the plan has no DFlash draft.
pub(crate) fn split_stage0_plan(plan: &SpeculativeDecodeConfig) -> Option<SpeculativeDecodeConfig> {
    let dflash = plan.dflash.as_ref()?;
    let block = dflash
        .max_draft_tokens
        .unwrap_or(UNCAPPED_DRAFT_TOKENS)
        .clamp(1, MAX_WIRE_DRAFT_TOKENS);
    let mut stage0 = plan.clone();
    stage0.dflash = None;
    stage0.ngram = None;
    stage0.extension = None;
    stage0.native_mtp.enabled = true;
    stage0.native_mtp.max_draft_tokens = block;
    stage0.native_mtp.min_draft_tokens = 1;
    stage0.native_mtp.reject_cooldown_tokens = 0;
    stage0.native_mtp.suppress_cooldown_drafts = false;
    stage0.verify_window.min_tokens = block;
    stage0.verify_window.max_tokens = block;
    Some(stage0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dflash_plan(cap: Option<usize>) -> SpeculativeDecodeConfig {
        SpeculativeDecodeConfig {
            requested_strategy: "dflash".to_string(),
            effective_strategy: "dflash".to_string(),
            dflash: Some(DFlashProposalConfig {
                draft_model_path: PathBuf::from("/models/draft-dflash.gguf"),
                max_draft_tokens: cap,
            }),
            ..SpeculativeDecodeConfig::default()
        }
    }

    #[test]
    fn stage0_verifies_whole_blocks_through_native_mtp_windows() {
        let plan = split_stage0_plan(&dflash_plan(Some(7))).unwrap();
        assert!(plan.native_mtp.enabled);
        assert!(plan.dflash.is_none());
        assert_eq!(plan.native_mtp.max_draft_tokens, 7);
        assert_eq!(plan.native_mtp.min_draft_tokens, 1);
        assert_eq!(
            (plan.verify_window.min_tokens, plan.verify_window.max_tokens),
            (7, 7)
        );
    }

    #[test]
    fn stage0_block_fits_the_wire_depth() {
        let uncapped = split_stage0_plan(&dflash_plan(None)).unwrap();
        assert_eq!(uncapped.native_mtp.max_draft_tokens, 15);
        let wide = split_stage0_plan(&dflash_plan(Some(31))).unwrap();
        assert_eq!(wide.native_mtp.max_draft_tokens, 15);
    }

    #[test]
    fn a_plan_without_dflash_is_left_alone() {
        assert!(split_stage0_plan(&SpeculativeDecodeConfig::default()).is_none());
    }

    #[test]
    fn rollback_depth_covers_the_block_and_the_anchor() {
        let stage = |cap| SplitDFlashStage {
            max_draft_tokens: cap,
            ..SplitDFlashStage::default()
        };
        assert_eq!(stage(Some(7)).recurrent_rollback_tokens(), 8);
        assert_eq!(stage(None).recurrent_rollback_tokens(), 16);
    }
}
