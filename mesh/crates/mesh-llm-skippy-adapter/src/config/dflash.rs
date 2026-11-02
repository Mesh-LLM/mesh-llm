//! DFlash draft resolution.
//!
//! A DFlash draft reads target hidden states, so it is not an ordinary draft
//! model: the architecture pairing check would always reject it. It is only
//! chosen explicitly with `speculative.strategy = "dflash"`. Under `auto` a
//! DFlash draft still fails pairing, so speculation stays off until the native
//! verify path is fast enough to select DFlash by default.

use std::path::Path;

use anyhow::{Result, bail};
use skippy_api::speculative::is_dflash_draft;
use skippy_serving::{DFLASH_STRATEGY, DFlashProposalConfig, SpeculativeDecodeConfig};

/// What the resolver already decided before choosing DFlash.
pub(super) struct DFlashSelection<'a> {
    pub(super) strategy: &'a str,
    pub(super) mode: &'a str,
    pub(super) draft_model_path: Option<&'a Path>,
}

/// The DFlash draft to attach, or `None` when DFlash does not apply.
pub(super) fn select_dflash_draft<'a>(selection: &DFlashSelection<'a>) -> Result<Option<&'a Path>> {
    if selection.strategy != DFLASH_STRATEGY {
        return Ok(None);
    }
    if selection.mode == "disabled" {
        bail!(
            "skippy speculative.strategy = \"dflash\" conflicts with speculative.mode = \"disabled\""
        );
    }
    let Some(path) = selection
        .draft_model_path
        .filter(|path| is_dflash_draft(path))
    else {
        bail!(
            "skippy speculative.strategy = \"dflash\" requires a draft_model whose GGUF architecture is dflash"
        );
    };
    Ok(Some(path))
}

/// Turns a base plan (draft placement and gate settings already resolved)
/// into a DFlash plan for `path`.
pub(super) fn dflash_decode_config(
    mut base: SpeculativeDecodeConfig,
    requested_strategy: &str,
    path: &Path,
    draft_max_tokens: u32,
) -> SpeculativeDecodeConfig {
    base.requested_strategy = requested_strategy.to_string();
    base.effective_strategy = DFLASH_STRATEGY.to_string();
    base.native_mtp.enabled = false;
    base.ngram = None;
    base.extension = None;
    base.dflash = Some(DFlashProposalConfig {
        draft_model_path: path.to_path_buf(),
        max_draft_tokens: (draft_max_tokens > 0).then_some(draft_max_tokens as usize),
    });
    base
}

#[cfg(test)]
mod tests {
    use super::super::test_support::temp_model_file_with_architecture;
    use super::*;

    fn selection<'a>(strategy: &'a str, path: Option<&'a Path>) -> DFlashSelection<'a> {
        DFlashSelection {
            strategy,
            mode: "auto",
            draft_model_path: path,
        }
    }

    #[test]
    fn only_the_dflash_strategy_selects_a_dflash_draft() {
        let dflash_file = temp_model_file_with_architecture("dflash");
        let dflash = dflash_file.path().to_path_buf();

        assert_eq!(
            select_dflash_draft(&selection("dflash", Some(&dflash))).unwrap(),
            Some(dflash.as_path())
        );
        for strategy in ["auto", "mtp", "disabled"] {
            assert_eq!(
                select_dflash_draft(&selection(strategy, Some(&dflash))).unwrap(),
                None,
                "{strategy}"
            );
        }
    }

    #[test]
    fn explicit_dflash_requires_a_dflash_draft() {
        let ordinary_file = temp_model_file_with_architecture("qwen3");
        let ordinary = ordinary_file.path().to_path_buf();

        assert!(select_dflash_draft(&selection("dflash", None)).is_err());
        assert!(select_dflash_draft(&selection("dflash", Some(&ordinary))).is_err());
    }

    #[test]
    fn dflash_plan_replaces_other_proposers() {
        let mut base = SpeculativeDecodeConfig::default();
        base.native_mtp.enabled = true;
        let plan = dflash_decode_config(base, "auto", Path::new("/m/draft-dflash.gguf"), 7);

        assert_eq!(plan.requested_strategy, "auto");
        assert_eq!(plan.effective_strategy, DFLASH_STRATEGY);
        assert!(!plan.native_mtp.enabled);
        assert_eq!(plan.dflash.as_ref().unwrap().max_draft_tokens, Some(7));
        plan.validate().unwrap();
    }
}
