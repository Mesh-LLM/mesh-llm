use std::collections::VecDeque;

use super::{NativeMtpDraft, NativeMtpDraftOrigin, NativeMtpHybridProposal};

/// One contiguous candidate span verified by one target traversal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::frontend) struct PipelinedCandidateChunk {
    proposal_tokens: Vec<i32>,
    native_mtp_token_count: usize,
    starts_epoch: bool,
}

impl PipelinedCandidateChunk {
    pub(in crate::frontend) fn proposal_tokens(&self) -> &[i32] {
        &self.proposal_tokens
    }

    pub(in crate::frontend) fn native_mtp_token_count(&self) -> usize {
        self.native_mtp_token_count
    }

    pub(in crate::frontend) fn starts_epoch(&self) -> bool {
        self.starts_epoch
    }

    pub(in crate::frontend) fn advance_tokens(&self) -> usize {
        self.proposal_tokens.len()
    }
}

/// A full target traversal produces one prediction beyond its proposal span.
/// Keep that free token at the boundary while any later chunk depends on the
/// last proposal token as its input anchor.
pub(in crate::frontend) fn pipelined_target_commit_count(
    planned_advance_tokens: usize,
    target_commit_count: usize,
    fully_accepted: bool,
    dependent_work_exists: bool,
) -> usize {
    if fully_accepted && dependent_work_exists {
        planned_advance_tokens
    } else {
        target_commit_count
    }
}

/// Owns a composite candidate while positional windows consume it.
///
/// The first chunk is anchored by the current target token. Later chunks are
/// contiguous continuations: their first candidate is validated by the prior
/// chunk's free boundary prediction, and the target consumes only new inputs.
/// A fully accepted epoch therefore advances monotonically without trimming KV.
#[derive(Debug)]
pub(in crate::frontend) struct CompositeProposalPipeline {
    proposal: NativeMtpHybridProposal,
    origin: Option<NativeMtpDraftOrigin>,
    candidates: VecDeque<i32>,
    dispatched_tokens: usize,
    dispatched_native_mtp_token_count: usize,
    accepted_tokens: usize,
    next_draft: Option<NativeMtpDraft>,
}

impl CompositeProposalPipeline {
    pub(in crate::frontend) fn new(
        proposal: NativeMtpHybridProposal,
        origin: Option<NativeMtpDraftOrigin>,
    ) -> Self {
        Self {
            candidates: proposal.tokens().iter().copied().collect(),
            proposal,
            origin,
            dispatched_tokens: 0,
            dispatched_native_mtp_token_count: 0,
            accepted_tokens: 0,
            next_draft: None,
        }
    }

    pub(in crate::frontend) fn next_chunk(
        &mut self,
        max_tokens: usize,
    ) -> Option<PipelinedCandidateChunk> {
        let chunk_len = max_tokens.max(1).min(self.candidates.len());
        if chunk_len == 0 {
            return None;
        }
        let starts_epoch = self.dispatched_tokens == 0;
        let proposal_tokens = self.candidates.drain(..chunk_len).collect::<Vec<_>>();
        let native_mtp_token_count = self
            .proposal
            .native_mtp_token_count()
            .saturating_sub(self.dispatched_native_mtp_token_count)
            .min(proposal_tokens.len());
        self.dispatched_native_mtp_token_count += native_mtp_token_count;
        self.dispatched_tokens += proposal_tokens.len();
        Some(PipelinedCandidateChunk {
            proposal_tokens,
            native_mtp_token_count,
            starts_epoch,
        })
    }

    pub(in crate::frontend) fn proposal(&self) -> &NativeMtpHybridProposal {
        &self.proposal
    }

    pub(in crate::frontend) fn origin(&self) -> Option<NativeMtpDraftOrigin> {
        self.origin
    }

    pub(in crate::frontend) fn has_remaining_candidates(&self) -> bool {
        !self.candidates.is_empty()
    }

    pub(in crate::frontend) fn candidate_len(&self) -> usize {
        self.candidates.len()
    }

    /// The uncommitted optimistic suffix, including tokens already dispatched
    /// but not yet committed. The N-gram cache may read this suffix while its
    /// index remains restricted to committed target history.
    pub(in crate::frontend) fn optimistic_suffix(&self) -> &[i32] {
        &self.proposal.tokens()[self.accepted_tokens.min(self.proposal.tokens().len())..]
    }

    pub(in crate::frontend) fn append_ngram_candidates(&mut self, tokens: &[i32]) -> usize {
        self.proposal.append_ngram_tokens(tokens);
        self.candidates.extend(tokens.iter().copied());
        tokens.len()
    }

    pub(in crate::frontend) fn observe_accepted(&mut self, count: usize) {
        self.accepted_tokens += count;
    }

    pub(in crate::frontend) fn accepted_tokens(&self) -> usize {
        self.accepted_tokens
    }

    pub(in crate::frontend) fn set_next_draft(
        &mut self,
        native_mtp_enabled: bool,
        draft: Option<NativeMtpDraft>,
    ) {
        self.next_draft = native_mtp_enabled.then_some(draft).flatten();
    }

    pub(in crate::frontend) fn next_draft(&self) -> Option<&NativeMtpDraft> {
        self.next_draft.as_ref()
    }

    pub(in crate::frontend) fn take_next_draft(&mut self) -> Option<NativeMtpDraft> {
        self.next_draft.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposal(tokens: Vec<i32>, native_mtp_tokens: usize) -> NativeMtpHybridProposal {
        let ngram_span_available = native_mtp_tokens < tokens.len();
        NativeMtpHybridProposal::from_parts(tokens, native_mtp_tokens, ngram_span_available)
    }

    #[test]
    fn marks_only_the_first_full_chunk_as_epoch_start() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2, 3, 4], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
        );

        let first = pipeline.next_chunk(3).unwrap();
        assert_eq!(first.proposal_tokens(), [9, 1, 2]);
        assert_eq!(first.native_mtp_token_count(), 1);
        assert_eq!(first.advance_tokens(), 3);
        assert!(first.starts_epoch());

        let second = pipeline.next_chunk(3).unwrap();
        assert_eq!(second.proposal_tokens(), [3, 4]);
        assert_eq!(second.native_mtp_token_count(), 0);
        assert!(!second.starts_epoch());
    }

    #[test]
    fn dependent_chunk_keeps_the_free_target_at_the_boundary() {
        let mut pipeline = CompositeProposalPipeline::new(proposal(vec![9, 1, 2, 3], 1), None);
        let first = pipeline.next_chunk(2).unwrap();

        assert_eq!(
            pipelined_target_commit_count(first.advance_tokens(), 3, true, true),
            2
        );
        assert_eq!(pipeline.next_chunk(2).unwrap().proposal_tokens(), [2, 3]);
    }

    #[test]
    fn terminal_chunk_commits_the_free_target() {
        let mut pipeline = CompositeProposalPipeline::new(proposal(vec![9, 1], 1), None);
        let chunk = pipeline.next_chunk(2).unwrap();

        assert_eq!(
            pipelined_target_commit_count(chunk.advance_tokens(), 3, true, false),
            3
        );
    }

    #[test]
    fn supports_a_pure_ngram_candidate() {
        let mut pipeline = CompositeProposalPipeline::new(proposal(vec![1, 2, 3], 0), None);

        let chunk = pipeline.next_chunk(2).unwrap();
        assert_eq!(chunk.proposal_tokens(), [1, 2]);
        assert_eq!(chunk.native_mtp_token_count(), 0);
        assert!(pipeline.has_remaining_candidates());
    }

    #[test]
    fn pure_ngram_pipeline_discards_verify_next_native_mtp_drafts() {
        let mut pipeline = CompositeProposalPipeline::new(proposal(vec![1, 2, 3], 0), None);

        pipeline.set_next_draft(
            false,
            Some(NativeMtpDraft {
                tokens: vec![4],
                proposal_compute_us: 12,
            }),
        );

        assert!(pipeline.next_draft().is_none());
    }

    #[test]
    fn records_the_matching_prefix_of_a_rejected_window() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2, 3], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
        );

        let _ = pipeline.next_chunk(1).unwrap();
        pipeline.observe_accepted(1);

        assert_eq!(pipeline.accepted_tokens(), 1);
        assert!(
            pipeline
                .proposal()
                .ngram_tail_rejected(pipeline.accepted_tokens())
        );
    }

    #[test]
    fn later_ngram_rejection_does_not_reject_an_accepted_native_prefix() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2, 3], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
        );

        let first = pipeline.next_chunk(1).unwrap();
        assert_eq!(first.proposal_tokens(), [9]);
        pipeline.observe_accepted(1);

        let second = pipeline.next_chunk(1).unwrap();
        assert_eq!(second.proposal_tokens(), [1]);
        pipeline.observe_accepted(0);

        assert!(
            !pipeline
                .proposal()
                .native_mtp_prefix_rejected(pipeline.accepted_tokens())
        );
        assert!(
            pipeline
                .proposal()
                .ngram_tail_rejected(pipeline.accepted_tokens())
        );
    }

    #[test]
    fn appends_an_optimistic_ngram_suffix_without_committing_it() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2, 3], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
        );

        let first = pipeline.next_chunk(1).unwrap();
        assert_eq!(first.proposal_tokens(), [9]);
        assert_eq!(pipeline.optimistic_suffix(), &[9, 1, 2, 3]);

        pipeline.observe_accepted(1);
        assert_eq!(pipeline.optimistic_suffix(), &[1, 2, 3]);
        assert_eq!(pipeline.append_ngram_candidates(&[4, 5]), 2);
        assert_eq!(pipeline.optimistic_suffix(), &[1, 2, 3, 4, 5]);
        assert_eq!(pipeline.proposal().tokens(), &[9, 1, 2, 3, 4, 5]);
        assert_eq!(pipeline.proposal().ngram_token_count(), 5);

        let second = pipeline.next_chunk(1).unwrap();
        assert_eq!(second.proposal_tokens(), [1]);
    }
}
