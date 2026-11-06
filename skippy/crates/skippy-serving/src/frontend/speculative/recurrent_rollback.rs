use super::SpeculativeDecodeConfig;

impl SpeculativeDecodeConfig {
    /// Native recurrent snapshots needed by the enabled verification routes.
    ///
    /// `native_mtp_max_tokens` also covers callers using the legacy MTP options;
    /// `draft_window` describes the embedded serial draft-model route and is
    /// present only when that proposal model is attached. Dynamic local hook
    /// proposals have no pre-open width contract and use checkpoint fallback.
    /// Zero disables snapshots for ordinary decoding. This is a per-window bound,
    /// not retention for every in-flight window: native limits and older pipeline
    /// checkpoints continue to use the existing checkpoint/replay fallback.
    pub fn recurrent_rollback_tokens(
        &self,
        native_mtp_max_tokens: Option<usize>,
        draft_window: Option<usize>,
    ) -> u32 {
        let mut depth = self
            .dflash
            .as_ref()
            .map_or(0, super::DFlashProposalConfig::recurrent_rollback_tokens);
        let mtp_tokens = native_mtp_max_tokens
            .into_iter()
            .chain(
                self.native_mtp
                    .enabled
                    .then_some(self.native_mtp.max_draft_tokens),
            )
            .max();
        if let Some(tokens) = mtp_tokens {
            // Local MTP verifies current + all drafts. Embedded MTP can take
            // the configured verify width, also with an initial anchor.
            depth = depth.max(with_anchor(tokens.max(self.verify_width())));
        }
        if let Some(ngram) = &self.ngram {
            let pipelined = self.verify_window.pipeline_depth > 1
                || self.verify_window.runahead_max_tokens > 0
                || self.verify_window.runahead_auto;
            let tokens = if pipelined || self.extension.is_some() {
                // An epoch's first pipelined/composite window has an anchor;
                // continuation windows can be shorter.
                with_anchor(self.verify_width())
            } else {
                // Classic serial verification replaces the final proposal
                // with current, so its input length is the proposal length.
                bounded_depth(ngram.max_proposal_tokens.max(1))
            };
            depth = depth.max(tokens);
        }
        if let Some(window) = draft_window {
            depth = depth.max(bounded_depth(window.max(1)));
        }
        depth
    }

    fn verify_width(&self) -> usize {
        self.verify_window
            .max_tokens
            .max(self.verify_window.min_tokens)
            .max(1)
    }
}

fn with_anchor(tokens: usize) -> u32 {
    bounded_depth(tokens.saturating_add(1))
}

fn bounded_depth(tokens: usize) -> u32 {
    u32::try_from(tokens).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::speculative::{
        NgramExtensionConfig, NgramProposalConfig, NgramProposerKind,
    };

    fn ngram() -> SpeculativeDecodeConfig {
        SpeculativeDecodeConfig {
            ngram: Some(NgramProposalConfig {
                kind: NgramProposerKind::Cache,
                min_ngram: 2,
                max_ngram: 4,
                max_proposal_tokens: 7,
            }),
            ..SpeculativeDecodeConfig::default()
        }
    }

    #[test]
    fn ordinary_decode_does_not_enable_default_verify_window() {
        let plan = SpeculativeDecodeConfig::default();
        assert_eq!(plan.verify_window.max_tokens, 4);
        assert_eq!(plan.recurrent_rollback_tokens(None, None), 0);
    }

    #[test]
    fn mtp_local_and_embedded_routes_include_one_anchor() {
        let mut plan = SpeculativeDecodeConfig::default();
        plan.native_mtp.enabled = true;
        plan.native_mtp.max_draft_tokens = 7;
        assert_eq!(plan.recurrent_rollback_tokens(None, None), 8);
        plan.verify_window.max_tokens = 12;
        assert_eq!(plan.recurrent_rollback_tokens(None, None), 13);
        plan.native_mtp.enabled = false;
        assert_eq!(plan.recurrent_rollback_tokens(Some(20), None), 21);
    }

    #[test]
    fn classic_ngram_and_draft_have_no_extra_anchor() {
        assert_eq!(ngram().recurrent_rollback_tokens(None, None), 7);
        assert_eq!(ngram().recurrent_rollback_tokens(None, Some(11)), 11);
        assert_eq!(
            SpeculativeDecodeConfig::default().recurrent_rollback_tokens(None, Some(9)),
            9
        );
    }

    #[test]
    fn composite_and_pipeline_epoch_start_include_one_anchor() {
        let mut plan = ngram();
        plan.verify_window.max_tokens = 10;
        plan.native_mtp.enabled = true;
        plan.native_mtp.max_draft_tokens = 2;
        plan.extension = Some(NgramExtensionConfig { max_tokens: 40 });
        plan.validate().unwrap();
        // The buffered composite can exceed one verify window; dispatch
        // selects at most verify_width proposals before adding its anchor.
        assert_eq!(plan.recurrent_rollback_tokens(None, None), 11);
        plan.extension = None;
        plan.native_mtp.enabled = false;
        plan.verify_window.pipeline_depth = 2;
        plan.ngram_fallback_draft = true;
        assert_eq!(plan.recurrent_rollback_tokens(None, Some(7)), 11);
        plan.verify_window.pipeline_depth = 1;
        plan.verify_window.runahead_max_tokens = 20;
        assert_eq!(plan.recurrent_rollback_tokens(None, None), 11);
    }

    #[test]
    fn oversized_windows_remain_explicit_for_native_fallback() {
        let mut plan = ngram();
        plan.verify_window.pipeline_depth = 3;
        plan.verify_window.max_tokens = 100;
        assert_eq!(plan.recurrent_rollback_tokens(None, None), 101);
    }
}
