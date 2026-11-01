//! DFlash block-diffusion draft settings in the resolved speculative plan.

use std::path::PathBuf;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use super::SpeculativeDecodeConfig;

/// Effective strategy name for a DFlash or DFlash2 draft.
pub const DFLASH_STRATEGY: &str = "dflash";

/// Draft tokens in a trained 16-token block, the most a plan without a cap
/// can propose.
const UNCAPPED_DRAFT_TOKENS: usize = 15;

/// A DFlash or DFlash2 draft attached to a single-stage target. The draft
/// reads target hidden states, so it never runs on its own session.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DFlashProposalConfig {
    pub draft_model_path: PathBuf,
    /// Upper bound on draft tokens per block. `None` uses the trained block.
    #[serde(default)]
    pub max_draft_tokens: Option<usize>,
}

impl DFlashProposalConfig {
    /// Recurrent rollback depth the target needs to verify the longest
    /// window in place: every drafted token plus the anchor.
    pub fn recurrent_rollback_tokens(&self) -> u32 {
        let drafted = self.max_draft_tokens.unwrap_or(UNCAPPED_DRAFT_TOKENS);
        u32::try_from(drafted.saturating_add(1)).unwrap_or(u32::MAX)
    }
}

pub(super) fn validate_dflash_plan(plan: &SpeculativeDecodeConfig) -> Result<()> {
    let Some(dflash) = &plan.dflash else {
        return Ok(());
    };
    if plan.effective_strategy != DFLASH_STRATEGY {
        bail!("a DFlash draft requires effective strategy {DFLASH_STRATEGY}");
    }
    if plan.native_mtp.enabled || plan.ngram.is_some() || plan.extension.is_some() {
        bail!("DFlash speculation cannot be combined with native MTP or N-gram proposals");
    }
    if dflash.max_draft_tokens == Some(0) {
        bail!("DFlash max_draft_tokens must be positive");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dflash_plan() -> SpeculativeDecodeConfig {
        SpeculativeDecodeConfig {
            requested_strategy: DFLASH_STRATEGY.to_string(),
            effective_strategy: DFLASH_STRATEGY.to_string(),
            dflash: Some(DFlashProposalConfig {
                draft_model_path: PathBuf::from("/models/draft-dflash.gguf"),
                max_draft_tokens: Some(8),
            }),
            ..SpeculativeDecodeConfig::default()
        }
    }

    #[test]
    fn accepts_a_dflash_only_plan() {
        dflash_plan().validate().unwrap();
    }

    #[test]
    fn rejects_dflash_combined_with_native_mtp() {
        let mut plan = dflash_plan();
        plan.native_mtp.enabled = true;
        assert!(
            plan.validate()
                .unwrap_err()
                .to_string()
                .contains("cannot be combined")
        );
    }

    #[test]
    fn rejects_a_mislabeled_or_empty_dflash_plan() {
        let mut plan = dflash_plan();
        plan.effective_strategy = "disabled".to_string();
        assert!(plan.validate().is_err());

        let mut plan = dflash_plan();
        plan.dflash.as_mut().unwrap().max_draft_tokens = Some(0);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn rollback_depth_covers_the_draft_cap_and_the_anchor() {
        let mut plan = dflash_plan();
        let dflash = plan.dflash.as_mut().unwrap();
        for (cap, depth) in [(Some(15), 16), (Some(6), 7), (None, 16)] {
            dflash.max_draft_tokens = cap;
            assert_eq!(dflash.recurrent_rollback_tokens(), depth, "{cap:?}");
        }
    }

    #[test]
    fn older_plans_without_dflash_still_deserialize() {
        let mut value = serde_json::to_value(SpeculativeDecodeConfig::default()).unwrap();
        value.as_object_mut().unwrap().remove("dflash");
        let plan: SpeculativeDecodeConfig = serde_json::from_value(value).unwrap();
        assert_eq!(plan.dflash, None);
    }
}
