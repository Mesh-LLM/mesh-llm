use crate::frontend::native_mtp::pipelined_target_commit_count;
use crate::frontend::speculative::{acceptance_threshold_met, split_draft_len};

/// The policy outcome for one pipelined speculative verification window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PipelineAcceptancePolicy {
    pub(super) threshold_met: bool,
    pub(super) fully_accepted: bool,
    pub(super) accepted_candidate_tokens: usize,
}

/// Derive pipeline acceptance from target agreement and the configured
/// acceptance threshold. A below-threshold window commits only its first
/// target token and cannot advance the optimistic pipeline.
pub(super) fn pipeline_acceptance_policy(
    accepted_proposal_tokens: usize,
    proposal_len: usize,
    rejected: bool,
    acceptance_threshold: f64,
) -> PipelineAcceptancePolicy {
    let threshold_met =
        acceptance_threshold_met(accepted_proposal_tokens, proposal_len, acceptance_threshold);
    PipelineAcceptancePolicy {
        threshold_met,
        fully_accepted: threshold_met && !rejected && accepted_proposal_tokens == proposal_len,
        accepted_candidate_tokens: if threshold_met {
            accepted_proposal_tokens
        } else {
            0
        },
    }
}

/// Pick the next pipelined verification width after applying draft splitting.
pub(super) fn pipeline_chunk_width(
    max_tokens: usize,
    split_probability: f64,
    seed: usize,
) -> usize {
    split_draft_len(max_tokens.max(1), split_probability, seed)
}

/// Apply deterministic draft-window splitting to a proposal before target
/// verification. Keeping this policy here makes serial and pipelined callers
/// use the same configured probability semantics.
pub(super) fn draft_split_len(draft_len: usize, split_probability: f64, seed: usize) -> usize {
    split_draft_len(draft_len, split_probability, seed)
}

/// Compute the number of target predictions that may be committed from one
/// pipelined window after applying threshold and dependency policy.
pub(super) fn pipeline_commit_count(
    planned_advance_tokens: usize,
    target_commit_count: usize,
    threshold_met: bool,
    fully_accepted: bool,
    dependent_work_exists: bool,
) -> usize {
    let policy_commit_count = if threshold_met {
        target_commit_count
    } else {
        1
    };
    pipelined_target_commit_count(
        planned_advance_tokens,
        policy_commit_count,
        fully_accepted,
        dependent_work_exists,
    )
}
