use super::{
    NativeMtpDraft, NativeMtpDraftOrigin, NativeMtpStats, NativeMtpVerification,
    PendingNativeMtpDraft,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingDraft {
    origin: NativeMtpDraftOrigin,
}

#[derive(Default)]
pub(in crate::frontend) struct NativeMtpVerifier {
    pending_tokens: Vec<i32>,
    pending: Option<PendingDraft>,
    stats: NativeMtpStats,
}

impl NativeMtpVerifier {
    pub(in crate::frontend) fn take_pending_draft(&mut self) -> Option<PendingNativeMtpDraft> {
        let pending = self.pending.take()?;
        let tokens = std::mem::take(&mut self.pending_tokens);
        Some(PendingNativeMtpDraft {
            tokens,
            origin: pending.origin,
        })
    }

    pub(in crate::frontend) fn restore_pending_draft(&mut self, pending: PendingNativeMtpDraft) {
        debug_assert!(self.pending.is_none());
        debug_assert!(self.pending_tokens.is_empty());
        self.pending_tokens = pending.tokens;
        self.pending = Some(PendingDraft {
            origin: pending.origin,
        });
    }

    pub(in crate::frontend) fn clear_pending_draft(&mut self) {
        self.pending = None;
        self.pending_tokens.clear();
    }

    #[cfg(test)]
    pub(in crate::frontend) fn observe_taken_draft_verification(
        &mut self,
        draft_token: i32,
        target_token: i32,
        verification_compute_us: i64,
    ) -> NativeMtpVerification {
        self.record_verification(draft_token, target_token, verification_compute_us)
    }

    pub(in crate::frontend) fn observe_taken_draft_span(
        &mut self,
        draft_tokens: &[i32],
        target_tokens: &[i32],
        verification_compute_us: i64,
    ) -> NativeMtpSpanVerification {
        let mut accepted_count = 0usize;
        let mut rejected = false;
        let mut first_decision = NativeMtpVerification::NoPending;
        for (index, draft_token) in draft_tokens.iter().copied().enumerate() {
            let Some(target_token) = target_tokens.get(index).copied() else {
                break;
            };
            let compute_us = if index == 0 {
                verification_compute_us
            } else {
                0
            };
            let decision = self.record_verification(draft_token, target_token, compute_us);
            if index == 0 {
                first_decision = decision;
            }
            match decision {
                NativeMtpVerification::Accepted { .. } => accepted_count += 1,
                NativeMtpVerification::Rejected { .. } => {
                    rejected = true;
                    break;
                }
                NativeMtpVerification::NoPending => {}
            }
        }
        NativeMtpSpanVerification {
            accepted_count,
            rejected,
            first_decision,
        }
    }

    pub(in crate::frontend) fn observe_target_token(
        &mut self,
        target_token: i32,
        verification_compute_us: i64,
        next_draft: Option<NativeMtpDraft>,
        next_draft_origin: NativeMtpDraftOrigin,
    ) -> NativeMtpVerification {
        let verification = self.verify_pending(target_token, verification_compute_us);
        self.observe_next_draft(next_draft, next_draft_origin);
        verification
    }

    pub(in crate::frontend) fn stats(&self) -> NativeMtpStats {
        let mut stats = self.stats;
        stats.pending_tokens = self.pending_tokens.len() as u64;
        stats
    }

    fn verify_pending(
        &mut self,
        target_token: i32,
        verification_compute_us: i64,
    ) -> NativeMtpVerification {
        let Some(_pending) = self.pending.take() else {
            return NativeMtpVerification::NoPending;
        };

        let Some(pending_token) = self.pending_tokens.first().copied() else {
            return NativeMtpVerification::NoPending;
        };
        self.pending_tokens.clear();
        self.record_verification(pending_token, target_token, verification_compute_us)
    }

    fn record_verification(
        &mut self,
        draft_token: i32,
        target_token: i32,
        verification_compute_us: i64,
    ) -> NativeMtpVerification {
        self.stats.verification_count += 1;
        self.stats.verification_compute_us = self
            .stats
            .verification_compute_us
            .saturating_add(verification_compute_us);
        if draft_token == target_token {
            self.stats.accepted_tokens += 1;
            NativeMtpVerification::Accepted {
                draft: draft_token,
                target: target_token,
            }
        } else {
            self.stats.rejected_tokens += 1;
            NativeMtpVerification::Rejected {
                draft: draft_token,
                target: target_token,
            }
        }
    }

    pub(in crate::frontend) fn observe_next_draft(
        &mut self,
        next_draft: Option<NativeMtpDraft>,
        origin: NativeMtpDraftOrigin,
    ) {
        let Some(next_draft) = next_draft else {
            return;
        };
        self.stats.drafted_tokens = self
            .stats
            .drafted_tokens
            .saturating_add(next_draft.tokens.len() as u64);
        self.stats.proposal_compute_us = self
            .stats
            .proposal_compute_us
            .saturating_add(next_draft.proposal_compute_us);
        self.pending_tokens = next_draft.tokens;
        self.pending = Some(PendingDraft { origin });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpSpanVerification {
    pub(in crate::frontend) accepted_count: usize,
    pub(in crate::frontend) rejected: bool,
    pub(in crate::frontend) first_decision: NativeMtpVerification,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(token: i32) -> NativeMtpDraft {
        NativeMtpDraft {
            tokens: vec![token],
            proposal_compute_us: 7,
        }
    }

    fn observe(
        verifier: &mut NativeMtpVerifier,
        target_token: i32,
        verification_compute_us: i64,
        next_draft: Option<NativeMtpDraft>,
    ) -> NativeMtpVerification {
        verifier.observe_target_token(
            target_token,
            verification_compute_us,
            next_draft,
            NativeMtpDraftOrigin::InitialSerial,
        )
    }

    #[test]
    fn no_draft_behaves_like_baseline() {
        let mut verifier = NativeMtpVerifier::default();

        let decision = observe(&mut verifier, 11, 5, None);

        assert_eq!(decision, NativeMtpVerification::NoPending);
        assert_eq!(verifier.stats(), NativeMtpStats::default());
    }

    #[test]
    fn first_draft_is_pending_until_next_target_decode() {
        let mut verifier = NativeMtpVerifier::default();

        let decision = observe(&mut verifier, 11, 5, Some(draft(12)));

        assert_eq!(decision, NativeMtpVerification::NoPending);
        assert_eq!(
            verifier.stats(),
            NativeMtpStats {
                drafted_tokens: 1,
                pending_tokens: 1,
                proposal_compute_us: 7,
                ..NativeMtpStats::default()
            }
        );
    }

    #[test]
    fn restoring_a_taken_draft_preserves_its_origin_without_counting_it_twice() {
        let mut verifier = NativeMtpVerifier::default();
        verifier.observe_next_draft(Some(draft(12)), NativeMtpDraftOrigin::VerifyNext);

        let pending = verifier.take_pending_draft().unwrap();
        verifier.restore_pending_draft(pending);
        let restored = verifier.take_pending_draft().unwrap();

        assert_eq!(restored.tokens, vec![12]);
        assert_eq!(restored.origin, NativeMtpDraftOrigin::VerifyNext);
        assert_eq!(verifier.stats().drafted_tokens, 1);
    }

    #[test]
    fn matching_next_target_accepts_pending_draft() {
        let mut verifier = NativeMtpVerifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let decision = observe(&mut verifier, 12, 9, None);

        assert_eq!(
            decision,
            NativeMtpVerification::Accepted {
                draft: 12,
                target: 12,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpStats {
                drafted_tokens: 1,
                accepted_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 7,
                verification_compute_us: 9,
                ..NativeMtpStats::default()
            }
        );
    }

    #[test]
    fn different_next_target_rejects_pending_draft() {
        let mut verifier = NativeMtpVerifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let decision = observe(&mut verifier, 13, 9, None);

        assert_eq!(
            decision,
            NativeMtpVerification::Rejected {
                draft: 12,
                target: 13,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpStats {
                drafted_tokens: 1,
                rejected_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 7,
                verification_compute_us: 9,
                ..NativeMtpStats::default()
            }
        );
    }

    #[test]
    fn verifies_previous_draft_before_storing_next_draft() {
        let mut verifier = NativeMtpVerifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let decision = observe(&mut verifier, 12, 9, Some(draft(14)));

        assert_eq!(
            decision,
            NativeMtpVerification::Accepted {
                draft: 12,
                target: 12,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpStats {
                drafted_tokens: 2,
                accepted_tokens: 1,
                pending_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 14,
                verification_compute_us: 9,
                ..NativeMtpStats::default()
            }
        );
    }

    #[test]
    fn taken_pending_draft_can_be_recorded_as_batched_accept() {
        let mut verifier = NativeMtpVerifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let pending = verifier.take_pending_draft().unwrap();
        assert_eq!(pending.origin, NativeMtpDraftOrigin::InitialSerial);
        assert!(verifier.take_pending_draft().is_none());
        let decision = verifier.observe_taken_draft_verification(pending.tokens[0], 12, 9);

        assert_eq!(
            decision,
            NativeMtpVerification::Accepted {
                draft: 12,
                target: 12,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpStats {
                drafted_tokens: 1,
                accepted_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 7,
                verification_compute_us: 9,
                ..NativeMtpStats::default()
            }
        );
    }

    #[test]
    fn taken_pending_draft_can_be_recorded_as_batched_reject() {
        let mut verifier = NativeMtpVerifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        let pending = verifier.take_pending_draft().unwrap();
        assert_eq!(pending.origin, NativeMtpDraftOrigin::InitialSerial);
        let decision = verifier.observe_taken_draft_verification(pending.tokens[0], 13, 9);

        assert_eq!(
            decision,
            NativeMtpVerification::Rejected {
                draft: 12,
                target: 13,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpStats {
                drafted_tokens: 1,
                rejected_tokens: 1,
                verification_count: 1,
                proposal_compute_us: 7,
                verification_compute_us: 9,
                ..NativeMtpStats::default()
            }
        );
    }

    #[test]
    fn clear_pending_draft_drops_unverified_draft_without_changing_stats() {
        let mut verifier = NativeMtpVerifier::default();
        observe(&mut verifier, 11, 5, Some(draft(12)));

        verifier.clear_pending_draft();

        assert_eq!(
            verifier.stats(),
            NativeMtpStats {
                drafted_tokens: 1,
                proposal_compute_us: 7,
                ..NativeMtpStats::default()
            }
        );
        assert_eq!(
            observe(&mut verifier, 12, 9, None),
            NativeMtpVerification::NoPending
        );
    }

    #[test]
    fn verification_compute_time_saturates_instead_of_overflowing() {
        let mut verifier = NativeMtpVerifier::default();

        observe(&mut verifier, 11, i64::MAX, Some(draft(12)));
        observe(&mut verifier, 13, i64::MAX, Some(draft(14)));
        observe(&mut verifier, 15, 1, None);

        assert_eq!(verifier.stats().verification_compute_us, i64::MAX);
    }

    #[test]
    fn empty_span_returns_no_pending_first_decision_and_zero_counts() {
        let mut verifier = NativeMtpVerifier::default();
        let result = verifier.observe_taken_draft_span(&[], &[], 0);
        assert_eq!(result.accepted_count, 0);
        assert!(!result.rejected);
        assert_eq!(result.first_decision, NativeMtpVerification::NoPending);
        assert_eq!(verifier.stats(), NativeMtpStats::default());
    }

    #[test]
    fn span_with_all_accepts_increments_accepted_count() {
        let mut verifier = NativeMtpVerifier::default();
        let result = verifier.observe_taken_draft_span(&[10, 11, 12], &[10, 11, 12], 5);
        assert_eq!(result.accepted_count, 3);
        assert!(!result.rejected);
        assert_eq!(
            result.first_decision,
            NativeMtpVerification::Accepted {
                draft: 10,
                target: 10,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpStats {
                verification_count: 3,
                verification_compute_us: 5,
                accepted_tokens: 3,
                ..NativeMtpStats::default()
            }
        );
    }

    #[test]
    fn span_with_rejected_at_first_position_marks_rejected() {
        let mut verifier = NativeMtpVerifier::default();
        let result = verifier.observe_taken_draft_span(&[10], &[99], 7);
        assert_eq!(result.accepted_count, 0);
        assert!(result.rejected);
        assert_eq!(
            result.first_decision,
            NativeMtpVerification::Rejected {
                draft: 10,
                target: 99,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpStats {
                verification_count: 1,
                verification_compute_us: 7,
                rejected_tokens: 1,
                ..NativeMtpStats::default()
            }
        );
    }

    #[test]
    fn span_with_mixed_accept_then_reject_breaks_after_reject() {
        let mut verifier = NativeMtpVerifier::default();
        let result = verifier.observe_taken_draft_span(&[10, 11, 12], &[10, 11, 99], 4);
        assert_eq!(result.accepted_count, 2);
        assert!(result.rejected);
        assert_eq!(
            result.first_decision,
            NativeMtpVerification::Accepted {
                draft: 10,
                target: 10,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpStats {
                verification_count: 3,
                verification_compute_us: 4,
                accepted_tokens: 2,
                rejected_tokens: 1,
                ..NativeMtpStats::default()
            }
        );
    }

    #[test]
    fn span_breaks_when_targets_run_out_before_drafts() {
        let mut verifier = NativeMtpVerifier::default();
        let result = verifier.observe_taken_draft_span(&[10, 11, 12], &[10], 3);
        assert_eq!(result.accepted_count, 1);
        assert!(!result.rejected);
        assert_eq!(
            result.first_decision,
            NativeMtpVerification::Accepted {
                draft: 10,
                target: 10,
            }
        );
        assert_eq!(
            verifier.stats(),
            NativeMtpStats {
                verification_count: 1,
                verification_compute_us: 3,
                accepted_tokens: 1,
                ..NativeMtpStats::default()
            }
        );
    }

    #[test]
    fn span_attributes_compute_time_to_first_iteration_only() {
        let mut verifier = NativeMtpVerifier::default();
        let result = verifier.observe_taken_draft_span(&[10, 11], &[10, 11], 42);
        assert_eq!(result.accepted_count, 2);
        assert!(!result.rejected);
        assert_eq!(
            verifier.stats(),
            NativeMtpStats {
                verification_count: 2,
                verification_compute_us: 42,
                accepted_tokens: 2,
                ..NativeMtpStats::default()
            }
        );
    }
}
