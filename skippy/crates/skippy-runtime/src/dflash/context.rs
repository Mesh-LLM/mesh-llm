//! How much of a session's target context its DFlash draft holds.
//!
//! A draft reads target features for the positions before the token it
//! drafts after. When rows go missing (a capture failed, or an earlier
//! stage's rows did not arrive), the draft's cache no longer describes the
//! recent target, and its blocks are guesses. This tracks the run of target
//! positions the draft holds without a gap, counts what was lost, and tells
//! the session when it may not propose.
//!
//! A target restored from a cache without a decode leaves the draft without
//! the restored prefix. Its rows are not coming back, but the rows decoded
//! after the restore point are whole, so the draft drafts from those, as a
//! shorter context. That is the expected cost of a cache hit, not a loss, so
//! it is not counted as one. A row lost after that stops drafting until the
//! target restarts at position 0.

/// Feature rows a session's draft lost, and how many times its context
/// stopped matching the target. Both only grow until the session is reused
/// for another request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DFlashHealth {
    /// Target positions whose features never reached the draft.
    pub feature_rows_missing: u64,
    /// Times the draft's context was cleared or left with a gap while the
    /// target still held those positions.
    pub draft_context_resets: u64,
}

impl DFlashHealth {
    pub fn is_clean(self) -> bool {
        self == Self::default()
    }
}

/// The target positions a lane's draft holds, as `from..fed` with no gap.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DraftContext {
    /// The first position of the run: 0, or where a restore left the target.
    from: u64,
    /// `None` once a row is missing: the draft cannot draft again until the
    /// target restarts from position 0.
    fed: Option<u64>,
    health: DFlashHealth,
    /// `health` as of the last [`DraftContext::take_new_losses`].
    reported: DFlashHealth,
}

/// What a draft context needs done before it ingests a decode's rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IngestPlan {
    /// Inject the rows; drop cached rows at and after `truncate_from` first
    /// when set, since the target re-decoded those positions.
    Ingest { truncate_from: Option<u64> },
    /// The context already has a gap; injecting more rows cannot fix it.
    Skip,
}

impl DraftContext {
    /// An empty draft for a target at position 0.
    pub(crate) fn empty() -> Self {
        Self {
            fed: Some(0),
            ..Self::default()
        }
    }

    pub(crate) fn health(&self) -> DFlashHealth {
        self.health
    }

    /// Starts a new request's counters on a reused lane.
    pub(crate) fn reset_health(&mut self) {
        self.health = DFlashHealth::default();
        self.reported = DFlashHealth::default();
    }

    /// Whether the draft holds every target position from the start of its
    /// run up to `anchor_position`, and at least one.
    pub(crate) fn can_propose(&self, anchor_position: u64) -> bool {
        anchor_position > self.from && self.fed.is_some_and(|fed| fed >= anchor_position)
    }

    /// Plans the injection of target positions `start..end`. A gap after the
    /// rows the draft holds makes the context incomplete.
    pub(crate) fn plan_ingest(&mut self, start: u64, end: u64) -> IngestPlan {
        match self.fed {
            None => {
                self.health.feature_rows_missing += end.saturating_sub(start);
                IngestPlan::Skip
            }
            Some(fed) if start > fed => {
                self.lose(start - fed + end.saturating_sub(start));
                IngestPlan::Skip
            }
            Some(fed) => {
                self.from = self.from.min(start);
                IngestPlan::Ingest {
                    truncate_from: (start < fed).then_some(start),
                }
            }
        }
    }

    /// Rows for `start..end` reached the draft.
    pub(crate) fn ingested(&mut self, end: u64) {
        if self.fed.is_some() {
            self.fed = Some(end);
        }
    }

    /// `rows` target positions did not reach the draft, which was cleared.
    pub(crate) fn lose(&mut self, rows: u64) {
        self.health.feature_rows_missing += rows;
        if self.fed.is_some() {
            self.health.draft_context_resets += 1;
        }
        self.fed = None;
    }

    /// The draft was cleared while the target holds `0..target_position`,
    /// restored without a decode. The draft starts a new run there.
    pub(crate) fn cleared(&mut self, target_position: u64) {
        if target_position > 0 {
            self.health.feature_rows_missing += target_position;
            self.health.draft_context_resets += 1;
        }
        self.from = target_position;
        self.fed = Some(target_position);
    }

    /// The target was deliberately replaced without a decode (a reset, a
    /// cache restore, or imported state) and holds `0..target_position`; the
    /// draft, cleared with it, starts a new run there. Not a loss: the
    /// restored rows were never sent to this draft.
    pub(crate) fn restarted_at(&mut self, target_position: u64) {
        self.from = target_position;
        self.fed = Some(target_position);
    }

    /// The draft dropped its rows at and after `position`, with the target.
    pub(crate) fn truncated(&mut self, position: u64) {
        if position == 0 {
            self.from = 0;
            self.fed = Some(0);
            return;
        }
        self.from = self.from.min(position);
        self.fed = self.fed.map(|fed| fed.min(position));
    }

    /// The counters, when they grew since the last call.
    pub(crate) fn take_new_losses(&mut self) -> Option<DFlashHealth> {
        if self.health == self.reported {
            return None;
        }
        self.reported = self.health;
        Some(self.health)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fed_through(end: u64) -> DraftContext {
        let mut context = DraftContext::empty();
        assert_eq!(
            context.plan_ingest(0, end),
            IngestPlan::Ingest {
                truncate_from: None
            }
        );
        context.ingested(end);
        context
    }

    #[test]
    fn an_empty_draft_never_proposes() {
        let context = DraftContext::empty();
        assert!(!context.can_propose(0));
        assert!(!context.can_propose(1));
    }

    #[test]
    fn contiguous_rows_let_the_draft_propose_at_or_before_their_end() {
        let context = fed_through(12);
        assert!(context.can_propose(12));
        assert!(context.can_propose(5));
        assert!(!context.can_propose(13));
        assert!(context.health().is_clean());
    }

    #[test]
    fn a_failed_ingest_blocks_proposals_and_counts_one_reset() {
        let mut context = fed_through(8);
        context.lose(4);
        assert!(!context.can_propose(12));
        // Later decodes cannot repair the gap, and add rows but no reset.
        assert_eq!(context.plan_ingest(12, 13), IngestPlan::Skip);
        assert_eq!(
            context.health(),
            DFlashHealth {
                feature_rows_missing: 5,
                draft_context_resets: 1
            }
        );
    }

    #[test]
    fn a_restore_drafts_only_from_rows_decoded_after_it_without_a_loss() {
        let mut context = fed_through(3);
        context.restarted_at(40);
        // Nothing after the restore point yet: an empty context.
        assert!(!context.can_propose(40));
        assert_eq!(
            context.plan_ingest(40, 47),
            IngestPlan::Ingest {
                truncate_from: None
            }
        );
        context.ingested(47);
        assert!(context.can_propose(47));
        assert!(!context.can_propose(40));
        // A cache hit is not a feature-delivery failure.
        assert_eq!(context.health(), DFlashHealth::default());
        assert_eq!(context.take_new_losses(), None);
        // A row lost after the restore still stops drafting, and counts.
        context.lose(1);
        assert!(!context.can_propose(48));
        assert_eq!(context.health().feature_rows_missing, 1);
    }

    #[test]
    fn a_failed_draft_cleared_mid_context_counts_the_rows_it_lost() {
        let mut context = fed_through(3);
        context.cleared(40);
        // Nothing after the restore point yet: an empty context.
        assert!(!context.can_propose(40));
        assert_eq!(
            context.plan_ingest(40, 47),
            IngestPlan::Ingest {
                truncate_from: None
            }
        );
        context.ingested(47);
        assert!(context.can_propose(47));
        assert!(!context.can_propose(40));
        assert_eq!(
            context.health(),
            DFlashHealth {
                feature_rows_missing: 40,
                draft_context_resets: 1
            }
        );
        // A row lost after the restore still stops drafting.
        context.lose(1);
        assert!(!context.can_propose(48));
    }

    #[test]
    fn a_gap_before_new_rows_is_a_loss() {
        let mut context = fed_through(3);
        assert_eq!(context.plan_ingest(10, 11), IngestPlan::Skip);
        assert!(!context.can_propose(11));
        assert_eq!(context.health().feature_rows_missing, 8);
        assert_eq!(context.health().draft_context_resets, 1);
    }

    #[test]
    fn a_reset_to_position_zero_recovers() {
        let mut context = fed_through(3);
        context.lose(1);
        context.cleared(0);
        assert_eq!(
            context.plan_ingest(0, 4),
            IngestPlan::Ingest {
                truncate_from: None
            }
        );
        context.ingested(4);
        assert!(context.can_propose(4));
        context.lose(1);
        context.truncated(0);
        assert!(context.plan_ingest(0, 1) != IngestPlan::Skip);
    }

    #[test]
    fn a_rewind_drops_held_rows_and_a_redecode_truncates_them() {
        let mut context = fed_through(10);
        context.truncated(7);
        assert!(!context.can_propose(8));
        assert_eq!(
            context.plan_ingest(7, 9),
            IngestPlan::Ingest {
                truncate_from: None
            }
        );
        context.ingested(9);
        // A verify window re-decoding rows the draft already holds.
        assert_eq!(
            context.plan_ingest(8, 10),
            IngestPlan::Ingest {
                truncate_from: Some(8)
            }
        );
        context.ingested(10);
        assert!(context.can_propose(10));
        assert!(context.health().is_clean());
    }

    #[test]
    fn new_losses_are_reported_once() {
        let mut context = fed_through(2);
        assert_eq!(context.take_new_losses(), None);
        context.lose(2);
        assert_eq!(
            context.take_new_losses(),
            Some(DFlashHealth {
                feature_rows_missing: 2,
                draft_context_resets: 1
            })
        );
        assert_eq!(context.take_new_losses(), None);
        context.reset_health();
        assert!(context.health().is_clean());
    }
}
