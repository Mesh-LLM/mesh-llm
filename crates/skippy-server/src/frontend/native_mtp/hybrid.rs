use std::collections::VecDeque;

use openai_frontend::{OpenAiError, OpenAiResult};

use super::NativeMtpDecodeOptions;
use crate::frontend::speculative::{CachedNgramProposer, propose_ngram_tokens};

const MIN_NGRAM_EXTENSION_TOKENS: usize = 2;

/// Builds one speculative candidate from a native-MTP prefix and an optional
/// N-gram continuation. The N-gram proposal must independently predict the
/// native-MTP prefix before its remaining tokens may extend that prefix.
#[derive(Debug, Clone, Copy)]
pub(in crate::frontend) struct CompositeProposalProvider {
    enabled: bool,
    ngram_size: usize,
    max_proposal_tokens: usize,
}

impl CompositeProposalProvider {
    pub(in crate::frontend) fn from_options(options: NativeMtpDecodeOptions) -> Self {
        Self {
            enabled: options.ngram_hybrid,
            ngram_size: options.ngram_size,
            max_proposal_tokens: options.ngram_max_proposal_tokens,
        }
    }
}

impl CompositeProposalProvider {
    #[cfg(test)]
    pub(in crate::frontend) fn propose(
        &self,
        native_mtp_tokens: &[i32],
        context_tokens: &[i32],
        max_proposal_tokens: usize,
    ) -> NativeMtpHybridProposal {
        self.propose_with_ngram_extension(
            native_mtp_tokens,
            context_tokens,
            max_proposal_tokens,
            max_proposal_tokens,
            None,
        )
        .expect("llama.cpp N-gram proposal succeeds")
    }

    pub(in crate::frontend) fn propose_with_ngram_extension(
        &self,
        native_mtp_tokens: &[i32],
        context_tokens: &[i32],
        max_proposal_tokens: usize,
        max_ngram_extension_tokens: usize,
        cached_ngram_proposer: Option<&mut CachedNgramProposer>,
    ) -> OpenAiResult<NativeMtpHybridProposal> {
        let native_mtp_tokens =
            &native_mtp_tokens[..native_mtp_tokens.len().min(max_proposal_tokens)];
        if !self.enabled || self.max_proposal_tokens == 0 || max_ngram_extension_tokens == 0 {
            return Ok(NativeMtpHybridProposal::from_native_mtp_tokens(
                native_mtp_tokens.to_vec(),
            ));
        }

        let ngram_limit = max_proposal_tokens
            .saturating_sub(native_mtp_tokens.len())
            .min(self.max_proposal_tokens)
            .min(max_ngram_extension_tokens);
        let (
            ngram_tokens,
            ngram_span_available,
            ngram_mtp_prefix_agreed,
            ngram_mtp_prefix_disagreed,
        ) = if let Some(cache) = cached_ngram_proposer {
            // The cache sees only committed target history. Native MTP is
            // an optional read-only continuation, so this returns the
            // sidecar tail directly rather than trying to re-predict it.
            let tail = cache.propose(context_tokens, native_mtp_tokens, ngram_limit)?;
            let available = !tail.is_empty();
            (tail, available, false, false)
        } else {
            let candidates = if native_mtp_tokens.is_empty() {
                propose_ngram_tokens(context_tokens, self.ngram_size, ngram_limit)?
            } else {
                // Preserve the #875 anchor rule for native MTP: the most
                // recent earlier occurrence of the configured context
                // N-gram supplies a complete span. Its leading tokens
                // must agree with MTP before its remaining tokens may
                // become the sidecar tail below.
                propose_ngram_tokens(
                    context_tokens,
                    self.ngram_size,
                    native_mtp_tokens.len().saturating_add(ngram_limit),
                )?
            };
            if native_mtp_tokens.is_empty() {
                let available = !candidates.is_empty();
                (candidates, available, false, false)
            } else {
                let available = candidates.len() > native_mtp_tokens.len();
                let agreed = available && candidates.starts_with(native_mtp_tokens);
                let disagreed = available && !agreed;
                let tail = if agreed {
                    candidates[native_mtp_tokens.len()..].to_vec()
                } else {
                    Vec::new()
                };
                (tail, available, agreed, disagreed)
            }
        };
        let ngram_tokens = if ngram_tokens.len() >= MIN_NGRAM_EXTENSION_TOKENS {
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
            ngram_mtp_prefix_agreed,
            ngram_mtp_prefix_disagreed,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpHybridProposal {
    tokens: Vec<i32>,
    native_mtp_token_count: usize,
    ngram_token_count: usize,
    ngram_span_available: bool,
    ngram_mtp_prefix_agreed: bool,
    ngram_mtp_prefix_disagreed: bool,
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
            ngram_mtp_prefix_agreed: native_mtp_token_count > 0 && ngram_span_available,
            ngram_mtp_prefix_disagreed: false,
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

    pub(in crate::frontend) fn ngram_mtp_prefix_agreed(&self) -> bool {
        self.ngram_mtp_prefix_agreed
    }

    pub(in crate::frontend) fn ngram_mtp_prefix_disagreed(&self) -> bool {
        self.ngram_mtp_prefix_disagreed
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
    /// It must not penalize native MTP, but it should temporarily stop extending
    /// healthy MTP candidates with another unprofitable tail.
    pub(in crate::frontend) fn ngram_tail_rejected(&self, accepted_proposal_tokens: usize) -> bool {
        self.native_mtp_token_count > 0
            && self.ngram_token_count > 0
            && accepted_proposal_tokens >= self.native_mtp_token_count
            && accepted_proposal_tokens < self.tokens.len()
    }

    /// The first pipelined verify may consume a wider prefix, while a later
    /// in-flight window safely consumes the remaining suffix. Reserve one
    /// optimistic target plus one remaining candidate so depth actually buys
    /// overlap for short MTP-plus-N-gram spans.
    pub(in crate::frontend) fn parallel_verify_width(
        &self,
        adaptive_verify_width: usize,
        pipeline_depth: usize,
    ) -> Option<usize> {
        if pipeline_depth < 2 || self.tokens.len() < 3 {
            return None;
        }
        Some(
            adaptive_verify_width
                .min(self.tokens.len().saturating_sub(2))
                .max(1),
        )
    }
}

/// Request-local adaptive control for the N-gram extension.
///
/// MTP is already a useful proposer on its own, so the N-gram sidecar starts
/// with the smallest useful tail. It widens only after the target accepted an
/// entire tail and resets after a mismatch. This keeps one accidental repeated
/// context from turning a healthy MTP candidate into an expensive long span.
/// Pure N-gram proposals retain the configured maximum because they have no
/// native prefix to protect.
#[derive(Debug)]
pub(in crate::frontend) struct NgramSidecarController {
    remaining_proposals: usize,
    initial_extension_tokens: usize,
    current_extension_tokens: usize,
    max_extension_tokens: usize,
}

impl NgramSidecarController {
    pub(in crate::frontend) fn new(
        initial_extension_tokens: usize,
        max_extension_tokens: usize,
    ) -> Self {
        let initial_extension_tokens = initial_extension_tokens.min(max_extension_tokens);
        Self {
            remaining_proposals: 0,
            initial_extension_tokens,
            current_extension_tokens: initial_extension_tokens,
            max_extension_tokens,
        }
    }

    /// Returns the N-gram token budget for this proposal. A zero budget means
    /// use the native MTP prefix alone while the sidecar cools down.
    pub(in crate::frontend) fn extension_limit(
        &mut self,
        native_mtp_tokens: &[i32],
        available_tokens: usize,
    ) -> usize {
        if native_mtp_tokens.is_empty() {
            return available_tokens.min(self.max_extension_tokens);
        }
        if self.remaining_proposals > 0 {
            self.remaining_proposals -= 1;
            return 0;
        }
        self.current_extension_tokens.min(available_tokens)
    }

    /// Applies the result of a completed composite candidate. Returns true
    /// only when a rejected tail entered cooldown, so existing rejection
    /// telemetry remains a direct count of sidecar failures.
    pub(in crate::frontend) fn observe_tail_outcome(
        &mut self,
        proposal: &NativeMtpHybridProposal,
        accepted_proposal_tokens: usize,
        cooldown_proposals: usize,
    ) -> bool {
        if proposal.native_mtp_token_count() == 0 || proposal.ngram_token_count() == 0 {
            return false;
        }
        if accepted_proposal_tokens >= proposal.tokens().len() {
            self.current_extension_tokens = self
                .current_extension_tokens
                .saturating_add(1)
                .min(self.max_extension_tokens);
            return false;
        }
        if !proposal.ngram_tail_rejected(accepted_proposal_tokens) {
            return false;
        }
        self.current_extension_tokens = self.initial_extension_tokens;
        self.remaining_proposals = cooldown_proposals;
        cooldown_proposals > 0
    }

    #[cfg(test)]
    pub(in crate::frontend) fn remaining_proposals(&self) -> usize {
        self.remaining_proposals
    }

    #[cfg(test)]
    pub(in crate::frontend) fn current_extension_tokens(&self) -> usize {
        self.current_extension_tokens
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

    pub(in crate::frontend) fn expected_free_target(&self, width: usize) -> Option<i32> {
        self.remaining_tokens.get(width).copied()
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
    ) {
        for expected in verified_tokens {
            debug_assert_eq!(self.remaining_tokens.pop_front(), Some(*expected));
        }
        self.accepted_tokens += verified_tokens.len();
        if let Some(next_target_token) = next_target_token {
            if self.remaining_tokens.front() == Some(&next_target_token) {
                self.remaining_tokens.pop_front();
                self.accepted_tokens += 1;
            } else {
                self.remaining_tokens.clear();
            }
        }
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

    fn context_with_upstream_span() -> [i32; 10] {
        [0, 0, 2, 3, 9, 1, 7, 8, 2, 3]
    }

    fn options() -> NativeMtpDecodeOptions {
        NativeMtpDecodeOptions {
            max_draft_tokens: 1,
            min_draft_tokens: 0,
            reject_cooldown_tokens: 0,
            suppress_cooldown_drafts: false,
            suppress_cooldown_draft_limit: 0,
            ngram_hybrid: true,
            ngram_size: 2,
            ngram_initial_extension_tokens: 2,
            ngram_max_proposal_tokens: 4,
            ngram_tail_backoff_proposals: 2,
            verify_window_min_tokens: 1,
            verify_window_max_tokens: 4,
        }
    }

    #[test]
    fn appends_ngram_tail_after_native_mtp_prefix() {
        let provider = CompositeProposalProvider::from_options(options());
        let proposal = provider.propose(&[9], &context_with_upstream_span(), 4);

        assert_eq!(proposal.tokens(), &[9, 1, 7, 8]);
        assert_eq!(proposal.native_mtp_token_count(), 1);
        assert_eq!(proposal.ngram_token_count(), 3);
        assert!(proposal.ngram_span_available());
        assert!(proposal.ngram_mtp_prefix_agreed());
        assert!(!proposal.ngram_mtp_prefix_disagreed());
    }

    #[test]
    fn match_length_does_not_starve_a_short_adaptive_extension() {
        let mut options = options();
        options.ngram_size = 8;
        options.ngram_max_proposal_tokens = 10;
        let provider = CompositeProposalProvider::from_options(options);
        let context = [
            99, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 77, 1, 2, 3, 4, 5, 6, 7, 8,
        ];

        let proposal = provider
            .propose_with_ngram_extension(&[11], &context, 10, 2, None)
            .unwrap();

        assert_eq!(proposal.tokens(), &[11, 12, 13]);
        assert_eq!(proposal.native_mtp_token_count(), 1);
        assert_eq!(proposal.ngram_token_count(), 2);
        assert!(proposal.ngram_mtp_prefix_agreed());
    }

    #[test]
    fn mtp_only_provider_preserves_native_proposals() {
        let mut options = options();
        options.ngram_hybrid = false;
        options.ngram_max_proposal_tokens = 0;
        let provider = CompositeProposalProvider::from_options(options);

        let proposal = provider.propose(&[9, 10, 11], &[], 2);

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
    fn keeps_native_mtp_prefix_when_ngram_prefix_disagrees() {
        let provider = CompositeProposalProvider::from_options(options());
        let proposal = provider.propose(&[8], &context_with_upstream_span(), 4);

        assert_eq!(proposal.tokens(), &[8]);
        assert!(proposal.ngram_span_available());
        assert!(!proposal.ngram_mtp_prefix_agreed());
        assert!(proposal.ngram_mtp_prefix_disagreed());
    }

    #[test]
    fn requires_every_native_mtp_token_to_match_before_extending() {
        let provider = CompositeProposalProvider::from_options(options());
        let proposal = provider.propose(&[9, 1], &context_with_upstream_span(), 4);

        assert_eq!(proposal.tokens(), &[9, 1, 7, 8]);
        assert_eq!(proposal.native_mtp_token_count(), 2);
        assert_eq!(proposal.ngram_token_count(), 2);
        assert!(proposal.ngram_mtp_prefix_agreed());
    }

    #[test]
    fn falls_back_to_pure_ngram_without_native_mtp_tokens() {
        let provider = CompositeProposalProvider::from_options(options());
        let proposal = provider.propose(&[], &context_with_upstream_span(), 4);

        assert_eq!(proposal.tokens(), &[9, 1, 7, 8]);
        assert_eq!(proposal.native_mtp_token_count(), 0);
        assert!(proposal.is_pure_ngram());
        assert!(proposal.ngram_span_available());
    }

    #[test]
    fn retains_native_mtp_prefix_when_no_ngram_span_exists() {
        let provider = CompositeProposalProvider::from_options(options());
        let proposal = provider.propose(&[9, 10], &[1, 2, 3, 4], 4);

        assert_eq!(proposal.tokens(), &[9, 10]);
        assert_eq!(proposal.native_mtp_token_count(), 2);
        assert_eq!(proposal.ngram_token_count(), 0);
        assert!(!proposal.ngram_span_available());
    }

    #[test]
    fn cache_extends_native_mtp_without_requiring_a_matching_prefix() {
        let provider = CompositeProposalProvider::from_options(options());
        let mut cache = CachedNgramProposer::new(2, 2).unwrap();
        let context = [1, 9, 7, 1, 9, 7, 1];

        let proposal = provider
            .propose_with_ngram_extension(&[9], &context, 3, 2, Some(&mut cache))
            .unwrap();

        assert_eq!(proposal.tokens(), &[9, 7, 1]);
        assert_eq!(proposal.native_mtp_token_count(), 1);
        assert_eq!(proposal.ngram_token_count(), 2);
        assert!(proposal.ngram_span_available());
        assert!(!proposal.ngram_mtp_prefix_agreed());
        assert!(!proposal.ngram_mtp_prefix_disagreed());
    }

    #[test]
    fn uses_a_single_prior_fixed_ngram_anchor_for_native_mtp() {
        let provider = CompositeProposalProvider::from_options(options());
        let proposal = provider.propose(&[9], &context_with_upstream_span(), 4);

        assert_eq!(proposal.tokens(), &[9, 1, 7, 8]);
        assert_eq!(proposal.native_mtp_token_count(), 1);
        assert_eq!(proposal.ngram_token_count(), 3);
        assert!(proposal.ngram_span_available());
        assert!(proposal.ngram_mtp_prefix_agreed());
    }

    #[test]
    fn ignores_a_one_token_ngram_tail() {
        let provider = CompositeProposalProvider::from_options(options());
        let proposal = provider.propose(&[9], &context_with_upstream_span(), 2);

        assert_eq!(proposal.tokens(), &[9]);
        assert_eq!(proposal.native_mtp_token_count(), 1);
        assert_eq!(proposal.ngram_token_count(), 0);
        assert!(proposal.ngram_span_available());
        assert!(proposal.ngram_mtp_prefix_agreed());
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
    fn tail_rejection_resets_and_backs_off_only_mtp_extensions() {
        let proposal = NativeMtpHybridProposal::from_parts(vec![9, 10, 11], 1, true);
        let mut controller = NgramSidecarController::new(2, 4);

        assert!(controller.observe_tail_outcome(&proposal, 1, 2));
        assert_eq!(controller.remaining_proposals(), 2);
        assert_eq!(controller.current_extension_tokens(), 2);
        assert_eq!(controller.extension_limit(&[9], 3), 0);
        assert_eq!(controller.extension_limit(&[9], 3), 0);
        assert_eq!(controller.extension_limit(&[9], 3), 2);
        assert_eq!(controller.extension_limit(&[], 4), 4);
    }

    #[test]
    fn backoff_preserves_the_native_mtp_prefix() {
        let provider = CompositeProposalProvider::from_options(options());
        let mut controller = NgramSidecarController::new(2, 4);
        let rejected_tail = NativeMtpHybridProposal::from_parts(vec![9, 1, 2], 1, true);
        assert!(controller.observe_tail_outcome(&rejected_tail, 1, 1));

        let native_only = provider
            .propose_with_ngram_extension(
                &[9],
                &context_with_upstream_span(),
                4,
                controller.extension_limit(&[9], 3),
                None,
            )
            .unwrap();
        let pure_ngram = provider
            .propose_with_ngram_extension(
                &[],
                &context_with_upstream_span(),
                4,
                controller.extension_limit(&[], 4),
                None,
            )
            .unwrap();

        assert_eq!(native_only.tokens(), &[9]);
        assert_eq!(pure_ngram.tokens(), &[9, 1, 7, 8]);
    }

    #[test]
    fn fully_accepted_tail_grows_the_next_extension_budget() {
        let proposal = NativeMtpHybridProposal::from_parts(vec![9, 1, 2], 1, true);
        let mut controller = NgramSidecarController::new(3, 6);

        assert_eq!(controller.extension_limit(&[9], 5), 3);
        assert!(!controller.observe_tail_outcome(&proposal, 3, 4));
        assert_eq!(controller.current_extension_tokens(), 4);
        assert_eq!(controller.extension_limit(&[9], 5), 4);
    }

    #[test]
    fn caps_parallel_verify_width_to_the_available_candidate_depth() {
        let too_shallow = NativeMtpHybridProposal::from_parts(vec![1, 2], 1, true);
        let deep_enough = NativeMtpHybridProposal::from_parts(vec![1, 2, 3], 1, true);
        let four_tokens = NativeMtpHybridProposal::from_parts(vec![1, 2, 3, 4], 1, true);
        let wider = NativeMtpHybridProposal::from_parts(vec![1, 2, 3, 4, 5], 1, true);

        assert_eq!(too_shallow.parallel_verify_width(4, 2), None);
        assert_eq!(deep_enough.parallel_verify_width(4, 2), Some(1));
        assert_eq!(four_tokens.parallel_verify_width(4, 2), Some(2));
        assert_eq!(wider.parallel_verify_width(4, 2), Some(3));
        assert_eq!(wider.parallel_verify_width(4, 1), None);
    }

    #[test]
    fn buffer_reuses_tail_only_when_target_advances_along_it() {
        let mut buffer = BufferedCompositeProposal::new(NativeMtpHybridProposal::from_parts(
            vec![9, 1, 2, 3],
            1,
            true,
        ));

        buffer.accept_window(&[9, 1], Some(2));
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
    fn buffer_exposes_only_a_real_dependent_free_target() {
        let buffer = BufferedCompositeProposal::new(NativeMtpHybridProposal::from_parts(
            vec![9, 1, 2],
            1,
            true,
        ));

        assert_eq!(buffer.expected_free_target(1), Some(1));
        assert_eq!(buffer.expected_free_target(2), Some(2));
        assert_eq!(buffer.expected_free_target(3), None);
    }

    #[test]
    fn later_tail_rejection_does_not_reject_an_accepted_native_prefix() {
        let mut buffer = BufferedCompositeProposal::new(NativeMtpHybridProposal::from_parts(
            vec![9, 10, 11, 12],
            1,
            true,
        ));

        buffer.accept_window(&[9, 10], Some(11));

        assert!(!buffer.native_mtp_prefix_rejected_after(0));
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
