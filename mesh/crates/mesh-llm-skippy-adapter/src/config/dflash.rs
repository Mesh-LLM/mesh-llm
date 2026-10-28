//! DFlash draft resolution.
//!
//! A DFlash draft reads target hidden states, so it is not an ordinary draft
//! model: the architecture pairing check would always reject it. It is chosen
//! explicitly with `speculative.strategy = "dflash"` and a draft, or by default
//! under `auto` when the target is one of the known pairings in
//! [`super::dflash_catalog`] and nothing else in the configuration speaks for
//! speculation. A DFlash draft named as an ordinary draft under `auto` still
//! fails pairing.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use skippy_api::speculative::is_dflash_draft;
use skippy_model_artifact::gguf::{GgufCompactMeta, scan_gguf_compact_meta};
use skippy_serving::{DFLASH_STRATEGY, DFlashProposalConfig, SpeculativeDecodeConfig};

use super::dflash_catalog::{DFlashPairing, default_dflash_max_draft_tokens, pairing_for_target};
use super::types::ResolvedSpeculativeConfig;

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

/// What else the resolver decided that keeps a known DFlash pairing from
/// being the default.
pub(super) struct DFlashDefaultSelection<'a> {
    pub(super) auto_defaults_enabled: bool,
    pub(super) native_mtp_enabled: bool,
    pub(super) has_ngram_settings: bool,
    pub(super) package_declares_speculation: bool,
    pub(super) target_meta: Option<&'a GgufCompactMeta>,
    pub(super) model_path: &'a Path,
}

/// The DFlash draft to attach and, when it was chosen by default, its known
/// pairing. An explicit DFlash draft wins; otherwise a known target defaults
/// to its draft unless the configuration, the package, or another proposer
/// already decides.
pub(super) fn choose_dflash_draft(
    explicit: &DFlashSelection<'_>,
    default: &DFlashDefaultSelection<'_>,
) -> Result<Option<(PathBuf, Option<&'static DFlashPairing>)>> {
    if let Some(path) = select_dflash_draft(explicit)? {
        return Ok(Some((path.to_path_buf(), None)));
    }
    if explicit.strategy != "auto"
        || explicit.mode != "auto"
        || explicit.draft_model_path.is_some()
        || !default.auto_defaults_enabled
        || default.native_mtp_enabled
        || default.has_ngram_settings
        || default.package_declares_speculation
    {
        return Ok(None);
    }
    let pairing = match default.target_meta {
        Some(meta) => pairing_for_target(meta),
        None => scan_gguf_compact_meta(default.model_path)
            .as_ref()
            .and_then(pairing_for_target),
    }
    .filter(|pairing| pairing.default_on(cfg!(target_os = "macos")));
    Ok(pairing.map(|pairing| (pairing.draft.cache_path(), Some(pairing))))
}

/// A configured cap wins; otherwise the backend default.
pub(super) fn dflash_max_draft_tokens(configured: u32) -> u32 {
    if configured > 0 {
        configured
    } else {
        default_dflash_max_draft_tokens(cfg!(target_os = "macos"))
    }
}

impl ResolvedSpeculativeConfig {
    /// Points the default DFlash plan at its fetched draft.
    pub fn set_default_dflash_draft(&mut self, path: &Path) {
        if let Some(dflash) = self.decode.dflash.as_mut() {
            dflash.draft_model_path = path.to_path_buf();
        }
    }

    /// Serves without speculation when the default DFlash draft cannot be
    /// fetched or does not fit the target.
    pub fn drop_default_dflash(&mut self) {
        self.default_dflash = None;
        self.decode.dflash = None;
        self.decode.effective_strategy = "disabled".to_string();
    }
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
