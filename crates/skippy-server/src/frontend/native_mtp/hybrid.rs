use std::collections::VecDeque;

use openai_frontend::{OpenAiError, OpenAiResult};

use super::NativeMtpDecodeOptions;
use crate::frontend::speculative::HistoryNgramProposer;

const MIN_NGRAM_EXTENSION_TOKENS: usize = 2;

/// Builds one speculative candidate from a native-MTP prefix and an optional
/// N-gram continuation. The N-gram proposal must independently predict the
/// native-MTP prefix before its remaining tokens may extend that prefix.
#[derive(Debug, Clone, Copy)]
pub(in crate::frontend) struct CompositeProposalProvider {
    enabled: bool,
    max_proposal_tokens: usize,
}

impl CompositeProposalProvider {
    pub(in crate::frontend) fn from_options(options: NativeMtpDecodeOptions) -> Self {
        Self {
            enabled: options.ngram_hybrid,
            max_proposal_tokens: options.ngram_max_proposal_tokens,
        }
    }
}

impl CompositeProposalProvider {
    pub(in crate::frontend) fn propose_with_ngram_extension(
        &self,
        native_mtp_tokens: &[i32],
        context_tokens: &[i32],
        max_proposal_tokens: usize,
        max_ngram_tokens: usize,
        cached_ngram_proposer: Option<&mut HistoryNgramProposer>,
    ) -> OpenAiResult<NativeMtpHybridProposal> {
        let native_mtp_tokens =
            &native_mtp_tokens[..native_mtp_tokens.len().min(max_proposal_tokens)];
        if !self.enabled || self.max_proposal_tokens == 0 || max_ngram_tokens == 0 {
            return Ok(NativeMtpHybridProposal::from_native_mtp_tokens(
                native_mtp_tokens.to_vec(),
            ));
        }

        let ngram_limit = max_proposal_tokens
            .saturating_sub(native_mtp_tokens.len())
            .min(self.max_proposal_tokens)
            .min(max_ngram_tokens);
        let (ngram_tokens, ngram_span_available) = if let Some(cache) = cached_ngram_proposer {
            // The cache sees only committed target history. Native MTP is
            // an optional read-only continuation, so this returns the
            // sidecar tail directly rather than trying to re-predict it.
            let mut tail = cache.propose(context_tokens, native_mtp_tokens, ngram_limit)?;
            tail.truncate(ngram_limit);
            let available = !tail.is_empty();
            (tail, available)
        } else {
            (Vec::new(), false)
        };
        let minimum_extension_tokens = MIN_NGRAM_EXTENSION_TOKENS.min(max_ngram_tokens);
        let ngram_tokens = if ngram_tokens.len() >= minimum_extension_tokens {
            ngram_tokens
        } else {
            Vec::new()
        };
        let mut tokens = native_mtp_tokens.to_vec();
        tokens.extend(ngram_tokens);
        Ok(NativeMtpHybridProposal {
            native_mtp_token_count: native_mtp_tokens.len(),
            ngram_token_count: tokens.len().saturating_sub(native_mtp_tokens.len()),
            tokens,
            ngram_span_available,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpHybridProposal {
    tokens: Vec<i32>,
    native_mtp_token_count: usize,
    ngram_token_count: usize,
    ngram_span_available: bool,
}

impl NativeMtpHybridProposal {
    pub(in crate::frontend) fn from_parts(
        tokens: Vec<i32>,
        native_mtp_token_count: usize,
        ngram_span_available: bool,
    ) -> Self {
        let native_mtp_token_count = native_mtp_token_count.min(tokens.len());
        Self {
            ngram_token_count: tokens.len().saturating_sub(native_mtp_token_count),
            native_mtp_token_count,
            tokens,
            ngram_span_available,
        }
    }

    pub(in crate::frontend) fn from_native_mtp_tokens(tokens: Vec<i32>) -> Self {
        Self::from_parts(tokens, usize::MAX, false)
    }

    pub(in crate::frontend) fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    pub(in crate::frontend) fn native_mtp_token_count(&self) -> usize {
        self.native_mtp_token_count
    }

    pub(in crate::frontend) fn ngram_token_count(&self) -> usize {
        self.ngram_token_count
    }

    pub(in crate::frontend) fn is_pure_ngram(&self) -> bool {
        self.native_mtp_token_count == 0 && self.ngram_token_count > 0
    }

    pub(in crate::frontend) fn ngram_span_available(&self) -> bool {
        self.ngram_span_available
    }

    pub(in crate::frontend) fn append_ngram_tokens(&mut self, tokens: &[i32]) {
        if tokens.is_empty() {
            return;
        }
        self.tokens.extend_from_slice(tokens);
        self.ngram_token_count = self.ngram_token_count.saturating_add(tokens.len());
        self.ngram_span_available = true;
    }

    /// A tail mismatch is not evidence that the native MTP prefix was bad.
    /// Keep the native reject cooldown scoped to mismatches inside that prefix.
    pub(in crate::frontend) fn native_mtp_prefix_rejected(
        &self,
        accepted_proposal_tokens: usize,
    ) -> bool {
        accepted_proposal_tokens < self.native_mtp_token_count
    }

    /// A mismatch after the MTP prefix belongs to the optional N-gram sidecar.
    /// It must not penalize native MTP; dependent in-flight work is discarded,
    /// and a later exact request-local match may start a new epoch.
    pub(in crate::frontend) fn ngram_tail_rejected(&self, accepted_proposal_tokens: usize) -> bool {
        self.native_mtp_token_count > 0
            && self.ngram_token_count > 0
            && accepted_proposal_tokens >= self.native_mtp_token_count
            && accepted_proposal_tokens < self.tokens.len()
    }

    /// Positional speculation needs at least two consecutive candidates so
    /// more than one contiguous verification chunk can be in flight.
    pub(in crate::frontend) fn supports_positional_pipeline(&self, pipeline_depth: usize) -> bool {
        pipeline_depth >= 2 && self.tokens.len() >= 2
    }
}

/// Request-local policy for extending native MTP with an exact N-gram match.
///
/// Target verification is authoritative and the fixed-depth pipeline bounds
/// wasted work. An available match therefore uses the configured horizon
/// immediately; there is no probation, promotion, cooldown, or slow start.
#[derive(Debug)]
pub(in crate::frontend) struct NgramSidecarController {
    max_extension_tokens: usize,
}

impl NgramSidecarController {
    pub(in crate::frontend) fn new(max_extension_tokens: usize) -> Self {
        Self {
            max_extension_tokens,
        }
    }

    /// Returns the configured N-gram horizon whenever a match is available.
    /// Target verification remains authoritative, so the Shard-style path does
    /// not require a serial probation phase before filling the pipeline.
    pub(in crate::frontend) fn extension_limit(
        &self,
        _native_mtp_tokens: &[i32],
        available_tokens: usize,
    ) -> usize {
        if self.max_extension_tokens < MIN_NGRAM_EXTENSION_TOKENS
            || available_tokens < MIN_NGRAM_EXTENSION_TOKENS
        {
            return 0;
        }
        available_tokens.min(self.max_extension_tokens)
    }

    /// Keep extending the optimistic suffix while the request-local cache can
    /// produce a continuation. Rejections are bounded by pipeline depth and do
    /// not disable later exact N-gram matches.
    pub(in crate::frontend) fn refill_limit(&self, available_tokens: usize) -> usize {
        if self.max_extension_tokens < MIN_NGRAM_EXTENSION_TOKENS
            || available_tokens < MIN_NGRAM_EXTENSION_TOKENS
        {
            return 0;
        }
        available_tokens.min(self.max_extension_tokens)
    }

    /// Any non-empty exact match may seed the fixed-depth pipeline.
    pub(in crate::frontend) fn permit_pipeline_start(&self) -> bool {
        self.max_extension_tokens >= MIN_NGRAM_EXTENSION_TOKENS
    }

    /// Serial fallback uses the same full verification chunks as the pipeline.
    pub(in crate::frontend) fn verify_width(&self, requested_width: usize) -> usize {
        requested_width
    }

    /// Returns true when target verification rejected inside the N-gram tail.
    /// A rejection invalidates dependent in-flight work but does not disable a
    /// later exact match.
    pub(in crate::frontend) fn observe_tail_outcome(
        &self,
        proposal: &NativeMtpHybridProposal,
        accepted_proposal_tokens: usize,
    ) -> bool {
        if proposal.ngram_token_count() == 0 {
            return false;
        }
        if proposal.native_mtp_token_count() > 0
            && accepted_proposal_tokens < proposal.native_mtp_token_count()
        {
            return false;
        }
        if accepted_proposal_tokens >= proposal.tokens().len() {
            return false;
        }
        proposal.native_mtp_token_count() == 0
            || proposal.ngram_tail_rejected(accepted_proposal_tokens)
    }
}

/// Holds the unverified portion of a composite proposal. A fully accepted
/// verify window may advance one additional target token, so that token is
/// removed from the buffer only when it agrees with the buffered candidate.
#[derive(Debug)]
pub(in crate::frontend) struct BufferedCompositeProposal {
    proposal: NativeMtpHybridProposal,
    remaining_tokens: VecDeque<i32>,
    accepted_tokens: usize,
}

impl BufferedCompositeProposal {
    pub(in crate::frontend) fn new(proposal: NativeMtpHybridProposal) -> Self {
        Self {
            remaining_tokens: proposal.tokens.iter().copied().collect(),
            proposal,
            accepted_tokens: 0,
        }
    }

    pub(in crate::frontend) fn proposal(&self) -> &NativeMtpHybridProposal {
        &self.proposal
    }

    pub(in crate::frontend) fn verify_tokens(&self, width: usize) -> Vec<i32> {
        self.remaining_tokens.iter().copied().take(width).collect()
    }

    pub(in crate::frontend) fn remaining_len(&self) -> usize {
        self.remaining_tokens.len()
    }

    pub(in crate::frontend) fn is_empty(&self) -> bool {
        self.remaining_tokens.is_empty()
    }

    pub(in crate::frontend) fn accepted_tokens(&self) -> usize {
        self.accepted_tokens
    }

    /// Returns how much of the unverified buffer still belongs to native MTP.
    /// Verification may consume one extra buffered token from the target's
    /// already-produced next prediction, so cap the count by the live buffer.
    pub(in crate::frontend) fn remaining_native_mtp_tokens(&self) -> usize {
        self.proposal
            .native_mtp_token_count()
            .saturating_sub(self.accepted_tokens)
            .min(self.remaining_tokens.len())
    }

    pub(in crate::frontend) fn native_mtp_prefix_rejected_after(
        &self,
        accepted_window_tokens: usize,
    ) -> bool {
        self.proposal
            .native_mtp_prefix_rejected(self.accepted_tokens.saturating_add(accepted_window_tokens))
    }

    pub(in crate::frontend) fn accept_window(
        &mut self,
        verified_tokens: &[i32],
        next_target_token: Option<i32>,
    ) -> bool {
        for expected in verified_tokens {
            let consumed = self.remaining_tokens.pop_front();
            debug_assert_eq!(consumed, Some(*expected));
        }
        self.accepted_tokens += verified_tokens.len();
        if let Some(next_target_token) = next_target_token {
            if self.remaining_tokens.front() == Some(&next_target_token) {
                self.remaining_tokens.pop_front();
                self.accepted_tokens += 1;
            } else if !self.remaining_tokens.is_empty() {
                self.remaining_tokens.clear();
                return true;
            }
        }
        false
    }

    pub(in crate::frontend) fn reject_window(&mut self, accepted_tokens: usize) {
        for _ in 0..accepted_tokens {
            self.remaining_tokens
                .pop_front()
                .expect("accepted composite prefix must remain buffered");
        }
        self.accepted_tokens += accepted_tokens;
        self.remaining_tokens.clear();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpVerifyWindowDecision {
    pub(in crate::frontend) accepted_proposal_tokens: usize,
    pub(in crate::frontend) commit_count: usize,
    pub(in crate::frontend) rejected: bool,
}

pub(in crate::frontend) fn classify_native_mtp_verify_window<F>(
    proposal_tokens: &[i32],
    predicted_tokens: &[i32],
    generated_len: usize,
    max_new_tokens: usize,
    mut token_is_eog: F,
) -> OpenAiResult<NativeMtpVerifyWindowDecision>
where
    F: FnMut(i32) -> OpenAiResult<bool>,
{
    let required_predictions = proposal_tokens.len().saturating_add(1);
    if predicted_tokens.len() < required_predictions {
        return Err(OpenAiError::backend(format!(
            "native MTP verify window returned too few tokens: got {} expected {}",
            predicted_tokens.len(),
            required_predictions
        )));
    }

    let mut accepted_proposal_tokens = 0usize;
    for (index, proposal_token) in proposal_tokens.iter().enumerate() {
        let predicted = predicted_tokens[index];
        let commit_count = index + 1;
        if predicted != *proposal_token {
            return Ok(NativeMtpVerifyWindowDecision {
                accepted_proposal_tokens,
                commit_count,
                rejected: true,
            });
        }

        accepted_proposal_tokens += 1;
        if token_is_eog(predicted)? || generated_len + commit_count >= max_new_tokens {
            return Ok(NativeMtpVerifyWindowDecision {
                accepted_proposal_tokens,
                commit_count,
                rejected: false,
            });
        }
    }

    Ok(NativeMtpVerifyWindowDecision {
        accepted_proposal_tokens,
        commit_count: required_predictions.min(max_new_tokens.saturating_sub(generated_len)),
        rejected: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> NativeMtpDecodeOptions {
        NativeMtpDecodeOptions {
            max_draft_tokens: 1,
            min_draft_tokens: 0,
            reject_cooldown_tokens: 0,
            suppress_cooldown_drafts: false,
            suppress_cooldown_draft_limit: 0,
            ngram_hybrid: true,
            ngram_proposer: "cache",
            ngram_size: 2,
            ngram_max_proposal_tokens: 4,
            verify_window_min_tokens: 1,
            verify_window_max_tokens: 4,
        }
    }

    #[test]
    fn mtp_only_provider_preserves_native_proposals() {
        let mut options = options();
        options.ngram_hybrid = false;
        options.ngram_max_proposal_tokens = 0;
        let provider = CompositeProposalProvider::from_options(options);

        let proposal = provider
            .propose_with_ngram_extension(&[9, 10, 11], &[], 2, 2, None)
            .unwrap();

        assert_eq!(proposal.tokens(), &[9, 10]);
        assert_eq!(proposal.native_mtp_token_count(), 2);
        assert_eq!(proposal.ngram_token_count(), 0);
    }

    #[test]
    fn ngram_limit_does_not_truncate_the_native_prefix() {
        let mut options = options();
        options.ngram_max_proposal_tokens = 1;
        let provider = CompositeProposalProvider::from_options(options);

        let proposal = provider
            .propose_with_ngram_extension(&[9, 10, 11], &[], 3, 1, None)
            .unwrap();

        assert_eq!(proposal.tokens(), &[9, 10, 11]);
        assert_eq!(proposal.native_mtp_token_count(), 3);
        assert_eq!(proposal.ngram_token_count(), 0);
    }

    #[test]
    fn retains_native_mtp_prefix_when_no_ngram_span_exists() {
        let provider = CompositeProposalProvider::from_options(options());
        let proposal = provider
            .propose_with_ngram_extension(&[9, 10], &[1, 2, 3, 4], 4, 4, None)
            .unwrap();

        assert_eq!(proposal.tokens(), &[9, 10]);
        assert_eq!(proposal.native_mtp_token_count(), 2);
        assert_eq!(proposal.ngram_token_count(), 0);
        assert!(!proposal.ngram_span_available());
    }

    #[test]
    fn cache_extends_native_mtp_without_requiring_a_matching_prefix() {
        let provider = CompositeProposalProvider::from_options(options());
        let mut cache = HistoryNgramProposer::new_cache(2, 2).unwrap();
        let context = [1, 9, 7, 1, 9, 7, 1];

        let proposal = provider
            .propose_with_ngram_extension(&[9], &context, 3, 2, Some(&mut cache))
            .unwrap();

        assert_eq!(proposal.tokens(), &[9, 7, 1]);
        assert_eq!(proposal.native_mtp_token_count(), 1);
        assert_eq!(proposal.ngram_token_count(), 2);
        assert!(proposal.ngram_span_available());
    }

    #[test]
    fn one_token_horizon_keeps_one_exact_ngram_token() {
        let provider = CompositeProposalProvider::from_options(options());
        let mut cache = HistoryNgramProposer::new_cache(1, 1).unwrap();
        let block = (1..=12).collect::<Vec<_>>();
        let mut context = block.clone();
        context.extend_from_slice(&block);
        context.extend_from_slice(&block[..8]);

        let proposal = provider
            .propose_with_ngram_extension(&[], &context, 1, 1, Some(&mut cache))
            .unwrap();

        assert_eq!(proposal.tokens(), &[9]);
        assert_eq!(proposal.ngram_token_count(), 1);
        assert!(proposal.ngram_span_available());
    }

    #[test]
    fn tail_rejection_does_not_count_as_native_mtp_rejection() {
        let proposal = NativeMtpHybridProposal::from_parts(vec![9, 10, 11], 1, true);

        assert!(!proposal.native_mtp_prefix_rejected(1));
        assert!(proposal.native_mtp_prefix_rejected(0));
        assert!(proposal.ngram_tail_rejected(1));
        assert!(proposal.ngram_tail_rejected(2));
        assert!(!proposal.ngram_tail_rejected(3));
    }

    #[test]
    fn exact_match_uses_the_full_configured_horizon_immediately() {
        let proposal = NativeMtpHybridProposal::from_parts(vec![9, 10, 11], 1, true);
        let controller = NgramSidecarController::new(4);

        assert_eq!(controller.extension_limit(&[9], 3), 3);
        assert_eq!(controller.extension_limit(&[], 4), 4);
        assert_eq!(controller.refill_limit(4), 4);
        assert!(controller.permit_pipeline_start());
        assert!(controller.observe_tail_outcome(&proposal, 1));
    }

    #[test]
    fn rejection_does_not_disable_a_later_exact_match() {
        let accepted = NativeMtpHybridProposal::from_parts(vec![9, 1, 2], 1, true);
        let rejected = NativeMtpHybridProposal::from_parts(vec![7, 8, 9], 0, true);
        let controller = NgramSidecarController::new(6);

        assert!(!controller.observe_tail_outcome(&accepted, 3));
        assert!(controller.observe_tail_outcome(&rejected, 1));
        assert_eq!(controller.extension_limit(&[], 6), 6);
        assert_eq!(controller.refill_limit(6), 6);
        assert!(controller.permit_pipeline_start());
    }

    #[test]
    fn serial_fallback_preserves_full_verification_width() {
        let controller = NgramSidecarController::new(6);

        assert_eq!(controller.verify_width(1), 1);
        assert_eq!(controller.verify_width(3), 3);
    }

    #[test]
    fn disabled_ngram_has_no_extension_or_pipeline() {
        let controller = NgramSidecarController::new(0);

        assert_eq!(controller.extension_limit(&[], 4), 0);
        assert_eq!(controller.refill_limit(4), 0);
        assert_eq!(controller.verify_width(4), 4);
        assert!(!controller.permit_pipeline_start());
    }

    #[test]
    fn native_prefix_rejection_is_not_an_ngram_tail_rejection() {
        let proposal = NativeMtpHybridProposal::from_parts(vec![9, 1, 2], 1, true);
        let controller = NgramSidecarController::new(6);

        assert!(!controller.observe_tail_outcome(&proposal, 0));
    }

    #[test]
    fn positional_pipeline_requires_two_candidates_and_depth() {
        let too_shallow = NativeMtpHybridProposal::from_parts(vec![1], 1, false);
        let ready = NativeMtpHybridProposal::from_parts(vec![1, 2], 1, true);

        assert!(!too_shallow.supports_positional_pipeline(2));
        assert!(ready.supports_positional_pipeline(2));
        assert!(!ready.supports_positional_pipeline(1));
    }

    #[test]
    fn buffer_reuses_tail_only_when_target_advances_along_it() {
        let mut buffer = BufferedCompositeProposal::new(NativeMtpHybridProposal::from_parts(
            vec![9, 1, 2, 3],
            1,
            true,
        ));

        assert!(!buffer.accept_window(&[9, 1], Some(2)));
        assert_eq!(buffer.verify_tokens(4), vec![3]);
        assert_eq!(buffer.accepted_tokens(), 3);

        buffer.reject_window(0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn buffer_keeps_the_matching_prefix_when_the_tail_rejects() {
        let mut buffer = BufferedCompositeProposal::new(NativeMtpHybridProposal::from_parts(
            vec![9, 10, 11, 12],
            1,
            true,
        ));

        buffer.reject_window(3);

        assert!(buffer.is_empty());
        assert_eq!(buffer.accepted_tokens(), 3);
    }

    #[test]
    fn later_tail_rejection_does_not_reject_an_accepted_native_prefix() {
        let mut buffer = BufferedCompositeProposal::new(NativeMtpHybridProposal::from_parts(
            vec![9, 10, 11, 12],
            1,
            true,
        ));

        assert!(!buffer.accept_window(&[9, 10], Some(11)));

        assert!(!buffer.native_mtp_prefix_rejected_after(0));
    }

    #[test]
    fn buffer_reports_a_rejected_dependent_free_target() {
        let mut buffer = BufferedCompositeProposal::new(NativeMtpHybridProposal::from_parts(
            vec![9, 10],
            1,
            true,
        ));

        assert!(buffer.accept_window(&[9], Some(42)));
        assert!(buffer.is_empty());
        assert_eq!(buffer.accepted_tokens(), 1);
    }

    #[test]
    fn native_only_buffer_consumes_verified_token_in_release_builds() {
        let mut buffer =
            BufferedCompositeProposal::new(NativeMtpHybridProposal::from_parts(vec![9], 1, false));

        assert!(!buffer.accept_window(&[9], Some(10)));
        assert!(buffer.is_empty());
        assert_eq!(buffer.accepted_tokens(), 1);
    }

    #[test]
    fn verify_window_commits_the_extra_target_after_full_accept() {
        let decision =
            classify_native_mtp_verify_window(&[11, 12, 13], &[11, 12, 13, 14], 0, 8, |_| {
                Ok(false)
            })
            .unwrap();

        assert_eq!(decision.accepted_proposal_tokens, 3);
        assert_eq!(decision.commit_count, 4);
        assert!(!decision.rejected);
    }

    #[test]
    fn verify_window_commits_the_target_correction_after_rejection() {
        let decision =
            classify_native_mtp_verify_window(&[11, 12], &[11, 42, 99], 0, 8, |_| Ok(false))
                .unwrap();

        assert_eq!(decision.accepted_proposal_tokens, 1);
        assert_eq!(decision.commit_count, 2);
        assert!(decision.rejected);
    }
}
