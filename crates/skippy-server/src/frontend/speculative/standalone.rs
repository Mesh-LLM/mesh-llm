use openai_frontend::{OpenAiError, OpenAiResult};

use super::{HistoryNgramProposer, NgramProposerKind, SpeculativeDecodeConfig};

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

/// Runs the configured standalone N-gram proposer (simple, cache, or suffix)
/// over committed history and returns its draft.
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
        NgramProposerKind::Cache | NgramProposerKind::Suffix => history_proposer
            .as_mut()
            .ok_or_else(|| OpenAiError::backend("configured history N-gram proposer is missing"))?
            .propose(committed_history, &[], proposal_limit)?,
    };
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
            }),
            ..SpeculativeDecodeConfig::default()
        }
    }

    fn propose(config: &SpeculativeDecodeConfig, history: &[i32]) -> ConfiguredNgramProposal {
        let mut proposer = HistoryNgramProposer::from_config(config).unwrap();
        propose_configured_ngram_tokens(config, &mut proposer, history, 8).unwrap()
    }

    #[test]
    fn standalone_limits_apply_to_every_ngram_kind() {
        for kind in [NgramProposerKind::Cache, NgramProposerKind::Suffix] {
            let config = config(kind, 3, 8);
            assert_eq!(standalone_ngram_proposal_limit(&config), 3);
        }
    }

    #[test]
    fn cache_is_a_standalone_proposer() {
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
}
