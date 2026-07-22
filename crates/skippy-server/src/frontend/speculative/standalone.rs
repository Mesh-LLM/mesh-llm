use openai_frontend::{OpenAiError, OpenAiResult};

use super::{
    HistoryNgramProposer, NgramProposerKind, SpeculativeDecodeConfig, propose_ngram_tokens,
};

/// A standalone N-gram draft plus the proposer kind that produced it.
pub(in crate::frontend) struct ConfiguredNgramProposal {
    pub(in crate::frontend) tokens: Vec<i32>,
    pub(in crate::frontend) source: &'static str,
}

/// Maximum draft length the configured N-gram proposer may emit, or 0 when none.
pub(in crate::frontend) fn standalone_ngram_proposal_limit(
    config: &SpeculativeDecodeConfig,
) -> usize {
    config
        .ngram
        .as_ref()
        .map_or(0, |ngram| ngram.max_proposal_tokens)
}

/// Minimum match length the simple fallback proposer scans for. The history
/// proposers use longer configured bounds that the simple proposer cannot.
const SIMPLE_FALLBACK_MIN_NGRAM: usize = 2;

/// Runs the configured standalone N-gram proposer (simple, cache, or suffix)
/// over committed history and returns its draft. When enabled, a history
/// proposer miss falls back to the simple proposer.
pub(in crate::frontend) fn propose_configured_ngram_tokens(
    config: &SpeculativeDecodeConfig,
    history_proposer: &mut Option<HistoryNgramProposer>,
    committed_history: &[i32],
    proposal_limit: usize,
) -> OpenAiResult<ConfiguredNgramProposal> {
    let Some(ngram) = config.ngram.as_ref() else {
        return Ok(ConfiguredNgramProposal {
            tokens: Vec::new(),
            source: "none",
        });
    };
    let proposal_limit = proposal_limit.min(ngram.max_proposal_tokens);
    let tokens = match ngram.kind {
        NgramProposerKind::Simple => {
            propose_ngram_tokens(committed_history, ngram.min_ngram, proposal_limit)?
        }
        NgramProposerKind::Cache | NgramProposerKind::Suffix => history_proposer
            .as_mut()
            .ok_or_else(|| OpenAiError::backend("configured history N-gram proposer is missing"))?
            .propose(committed_history, &[], proposal_limit)?,
    };
    if tokens.is_empty() && ngram.fallback_simple && ngram.kind != NgramProposerKind::Simple {
        let fallback =
            propose_ngram_tokens(committed_history, SIMPLE_FALLBACK_MIN_NGRAM, proposal_limit)?;
        if !fallback.is_empty() {
            return Ok(ConfiguredNgramProposal {
                tokens: fallback,
                source: "simple",
            });
        }
    }
    Ok(ConfiguredNgramProposal {
        tokens,
        source: ngram.kind.as_str(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::speculative::NgramProposalConfig;

    fn config(
        kind: NgramProposerKind,
        min_ngram: usize,
        max_ngram: usize,
    ) -> SpeculativeDecodeConfig {
        SpeculativeDecodeConfig {
            effective_strategy: format!("ngram-{}", kind.as_str()),
            ngram: Some(NgramProposalConfig {
                kind,
                min_ngram,
                max_ngram,
                max_proposal_tokens: 3,
                fallback_simple: false,
            }),
            ..SpeculativeDecodeConfig::default()
        }
    }

    fn config_with_fallback(
        kind: NgramProposerKind,
        min_ngram: usize,
        max_ngram: usize,
    ) -> SpeculativeDecodeConfig {
        let mut config = config(kind, min_ngram, max_ngram);
        config.ngram.as_mut().unwrap().fallback_simple = true;
        config
    }

    fn propose(config: &SpeculativeDecodeConfig, history: &[i32]) -> ConfiguredNgramProposal {
        let mut proposer = HistoryNgramProposer::from_config(config).unwrap();
        propose_configured_ngram_tokens(config, &mut proposer, history, 8).unwrap()
    }

    #[test]
    fn standalone_limits_apply_to_every_ngram_kind() {
        for kind in [
            NgramProposerKind::Simple,
            NgramProposerKind::Cache,
            NgramProposerKind::Suffix,
        ] {
            let config = config(kind, 3, 8);
            assert_eq!(standalone_ngram_proposal_limit(&config), 3);
        }
    }

    #[test]
    fn simple_is_a_standalone_proposer() {
        let proposal = propose(
            &config(NgramProposerKind::Simple, 2, 4),
            &[1, 2, 3, 4, 9, 2, 3, 4],
        );
        assert_eq!(proposal.source, "simple");
        assert_eq!(proposal.tokens, vec![9, 2, 3]);
    }

    #[test]
    fn cache_is_a_standalone_proposer_without_simple_fallback() {
        let proposal = propose(
            &config(NgramProposerKind::Cache, 2, 4),
            &[1, 2, 3, 1, 2, 3, 1, 2],
        );
        assert_eq!(proposal.source, "cache");
        assert_eq!(proposal.tokens, vec![3, 1, 2]);

        let miss = propose(&config(NgramProposerKind::Cache, 2, 4), &[1, 2, 3, 4]);
        assert_eq!(miss.source, "cache");
        assert!(miss.tokens.is_empty());
    }

    #[test]
    fn suffix_is_a_standalone_proposer_without_an_mtp_prefix() {
        let proposal = propose(
            &config(NgramProposerKind::Suffix, 3, 8),
            &[1, 2, 3, 4, 5, 1, 2, 3],
        );
        assert_eq!(proposal.source, "suffix");
        assert_eq!(proposal.tokens, vec![4, 5, 1]);
    }

    #[test]
    fn simple_fallback_fires_on_a_history_proposer_miss() {
        let proposal = propose(
            &config_with_fallback(NgramProposerKind::Suffix, 3, 8),
            &[5, 6, 1, 2, 9, 7, 1, 2],
        );
        assert_eq!(proposal.source, "simple");
        assert_eq!(proposal.tokens, vec![9, 7, 1]);
    }

    #[test]
    fn simple_fallback_does_not_replace_a_history_proposer_hit() {
        let proposal = propose(
            &config_with_fallback(NgramProposerKind::Suffix, 3, 8),
            &[1, 2, 3, 4, 5, 1, 2, 3],
        );
        assert_eq!(proposal.source, "suffix");
        assert_eq!(proposal.tokens, vec![4, 5, 1]);
    }

    #[test]
    fn simple_fallback_stays_off_by_default() {
        let proposal = propose(
            &config(NgramProposerKind::Suffix, 3, 8),
            &[5, 6, 1, 2, 9, 1, 2],
        );
        assert_eq!(proposal.source, "suffix");
        assert!(proposal.tokens.is_empty());
    }

    #[test]
    fn simple_fallback_miss_reports_the_primary_source() {
        let proposal = propose(
            &config_with_fallback(NgramProposerKind::Cache, 2, 4),
            &[1, 2, 3, 4],
        );
        assert_eq!(proposal.source, "cache");
        assert!(proposal.tokens.is_empty());
    }
}
