//! DFlash draft resolution.
//!
//! A DFlash draft reads target hidden states, so it is not an ordinary draft
//! model: the architecture pairing check would always reject it. It is chosen
//! explicitly with `speculative.strategy = "dflash"` and a draft, or by default
//! under `auto` when the target is one of the known pairings in
//! [`super::dflash_catalog`] and nothing else in the configuration speaks for
//! speculation. Native MTP that is on only because the target carries NextN
//! layers yields to a known pairing, and serves if the draft is dropped. A
//! DFlash draft named as an ordinary draft under `auto` still fails pairing.

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

/// Whether this build's native runtime can host a DFlash draft context.
/// Windows runtimes export no draft-context API (`skippy-ffi`'s
/// `llama_draft_api()` is `None` there), so a known target on Windows keeps
/// native MTP or plain decoding instead of a default draft it cannot attach.
pub(super) const DFLASH_DRAFT_CONTEXT_AVAILABLE: bool = !cfg!(windows);

/// What else the resolver decided that keeps a known DFlash pairing from
/// being the default.
pub(super) struct DFlashDefaultSelection<'a> {
    pub(super) auto_defaults_enabled: bool,
    /// The runtime can host a draft context; see
    /// [`DFLASH_DRAFT_CONTEXT_AVAILABLE`].
    pub(super) draft_context_available: bool,
    /// Native MTP is on and was configured, declared, or cannot yield.
    pub(super) native_mtp_decides: bool,
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
        || !default.draft_context_available
        || default.native_mtp_decides
        || default.has_ngram_settings
        || default.package_declares_speculation
    {
        return Ok(None);
    }
    let metal = cfg!(target_os = "macos");
    let pairing = match default.target_meta {
        Some(meta) => pairing_for_target(meta, metal),
        None => scan_gguf_compact_meta(default.model_path)
            .as_ref()
            .and_then(|meta| pairing_for_target(meta, metal)),
    };
    Ok(pairing.map(|pairing| (pairing.draft.cache_path(), Some(pairing))))
}

/// A configured cap wins; otherwise the backend default, held to what a
/// default draft's block can propose.
pub(super) fn dflash_max_draft_tokens(
    configured: u32,
    default_pairing: Option<&DFlashPairing>,
) -> u32 {
    if configured > 0 {
        return configured;
    }
    let backend = default_dflash_max_draft_tokens(cfg!(target_os = "macos"));
    default_pairing.map_or(backend, |pairing| backend.min(pairing.max_draft_tokens))
}

impl ResolvedSpeculativeConfig {
    /// Points the default DFlash plan at its fetched draft.
    pub fn set_default_dflash_draft(&mut self, path: &Path) {
        if let Some(dflash) = self.decode.dflash.as_mut() {
            dflash.draft_model_path = path.to_path_buf();
        }
    }

    /// Serves without the default DFlash draft when it cannot be fetched,
    /// does not fit, or the target is split: native MTP if the draft replaced
    /// it, else no speculation. An explicitly configured DFlash draft is not a
    /// default and is left in place.
    pub fn drop_default_dflash(&mut self) {
        if self.default_dflash.take().is_none() {
            return;
        }
        if let Some(fallback) = self.native_mtp_fallback.take() {
            *self = *fallback;
            return;
        }
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

    #[test]
    fn a_runtime_without_a_draft_context_keeps_no_default_draft() {
        let meta = super::super::test_support::target_meta("qwen35", 32, 4096, 0);
        let choose = |draft_context_available| {
            choose_dflash_draft(
                &selection("auto", None),
                &DFlashDefaultSelection {
                    auto_defaults_enabled: true,
                    draft_context_available,
                    native_mtp_decides: false,
                    has_ngram_settings: false,
                    package_declares_speculation: false,
                    target_meta: Some(&meta),
                    model_path: Path::new("/m/Qwen3.5-9B-Q4_K_M.gguf"),
                },
            )
            .unwrap()
        };

        assert!(choose(true).is_some_and(|(_, pairing)| pairing.is_some()));
        assert_eq!(choose(false), None);
        assert_eq!(DFLASH_DRAFT_CONTEXT_AVAILABLE, !cfg!(windows));
    }
}
