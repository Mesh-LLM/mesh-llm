//! Releasing a lane's native cells requires proof that no retainable
//! checkpoint still depends on them.
//!
//! A parked hybrid lane keeps its attention KV in native cells and its
//! recurrent snapshots as owned bytes. Before those cells can be reset for
//! another request, every snapshot that matters must reach the recorder.
//! `RecyclableLane` is the only type with a `reset`, and this module is the
//! only one that can construct it: by queuing every unrecorded checkpoint,
//! by finding none, or by a named `CheckpointDiscard` policy. A transient
//! recorder refusal is a `RetainedLane`, which still owns everything and
//! goes back to the pool; retention is bounded (`MAX_RETAINED_REFUSAL`) and
//! a stopped recorder ends it at once, so back-pressure never becomes a
//! stage that refuses requests forever.
use std::time::Duration;

use super::*;
use crate::kv_integration::DemotionRefusal;
use crate::kv_integration::exact_state_diagnostics::emit_exact_state_diagnostic;

pub(super) use super::recurrent_checkpoints::CheckpointDiscard;

/// How long a lane may sit in the pool with the recorder refusing its
/// export before the refusal is treated as a stuck recorder. Generous
/// against one record of several GiB being hashed and written to L3; short
/// against a request waiting on "lanes busy".
pub(super) const MAX_RETAINED_REFUSAL: Duration = Duration::from_secs(60);

/// What a recycle does with a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RefusalPolicy {
    /// Transient: keep the lane parked and try again next recycle.
    Retain,
    /// Terminal: nothing will ever record again; discard and recycle.
    Discard(CheckpointDiscard),
}

/// A refusal from a stopped recorder is terminal; every other refusal is
/// back-pressure until `MAX_RETAINED_REFUSAL` has elapsed since the first.
pub(super) fn refusal_policy(
    reason: DemotionRefusal,
    refused_since: Option<Instant>,
    now: Instant,
) -> RefusalPolicy {
    match reason {
        DemotionRefusal::WorkerStopped => {
            RefusalPolicy::Discard(CheckpointDiscard::RecorderStopped)
        }
        DemotionRefusal::CreditFull | DemotionRefusal::ByteFull | DemotionRefusal::QueueFull => {
            match refused_since {
                Some(since) if now.duration_since(since) >= MAX_RETAINED_REFUSAL => {
                    RefusalPolicy::Discard(CheckpointDiscard::RetentionExpired)
                }
                _ => RefusalPolicy::Retain,
            }
        }
    }
}

/// A lane whose checkpoints have all reached the recorder or been
/// deliberately discarded. Only this module can build one.
#[must_use = "a recyclable lane must be reset or returned to the pool"]
pub(super) struct RecyclableLane(RuntimeLaneSession);

/// The recorder refused the lane's export. The lane keeps its cells and
/// every unqueued checkpoint, and belongs back in the idle pool.
#[must_use = "a retained lane still owns native cells and checkpoints"]
pub(super) struct RetainedLane {
    pub lane: RuntimeLaneSession,
    pub reason: DemotionRefusal,
}

/// `StageSession::reset` failed. The lane has been dropped, which freed its
/// native cells; only the index survives for the free list.
pub(super) struct LaneResetFailed {
    pub index: usize,
    pub error: anyhow::Error,
}

/// Outcome of asking a lane to become recyclable.
#[must_use = "both outcomes carry the lane"]
pub(super) enum Recycle {
    Ready(RecyclableLane),
    Retained(RetainedLane),
}

/// Outcome of asking a checkpoint-free lane to become recyclable.
#[must_use = "both outcomes carry the lane"]
pub(super) enum EmptyRecycle {
    Ready(RecyclableLane),
    /// The lane holds checkpoints after all; it is returned untouched.
    Occupied(RuntimeLaneSession),
}

impl RuntimeLaneSession {
    /// Queue every unrecorded checkpoint, or keep them all resident. Nothing
    /// here touches the native cells except to copy them out. A refusal that
    /// `refusal_policy` calls terminal is discarded here and still yields a
    /// recyclable lane, so the pool can never fill with lanes nothing will
    /// ever record.
    pub(super) fn into_recyclable(mut self, start: u32, end: u32) -> Recycle {
        let reason = match self.release_checkpoints(start, end) {
            Ok(()) => {
                self.refused_since = None;
                return Recycle::Ready(RecyclableLane(self));
            }
            Err(reason) => reason,
        };
        let now = Instant::now();
        match refusal_policy(reason, self.refused_since, now) {
            RefusalPolicy::Retain => {
                self.refused_since.get_or_insert(now);
                Recycle::Retained(RetainedLane { lane: self, reason })
            }
            RefusalPolicy::Discard(policy) => {
                tracing::warn!(
                    lane_index = self.index,
                    reason = ?reason,
                    policy = ?policy,
                    "recorder refusal is terminal for this lane; discarding its checkpoints"
                );
                Recycle::Ready(self.into_discarded(policy))
            }
        }
    }

    /// A lane with nothing to demote needs no recorder. Hands the lane back
    /// untouched when it does hold checkpoints.
    pub(super) fn into_recyclable_if_empty(self) -> EmptyRecycle {
        if self.recurrent_prefixes.is_empty() {
            EmptyRecycle::Ready(RecyclableLane(self))
        } else {
            EmptyRecycle::Occupied(self)
        }
    }

    /// Sanctioned loss at a named call site.
    pub(super) fn into_discarded(mut self, reason: CheckpointDiscard) -> RecyclableLane {
        self.recurrent_prefixes.discard(reason);
        self.refused_since = None;
        RecyclableLane(self)
    }

    /// Offer every unrecorded checkpoint in rank order. A refusal leaves it
    /// and every unoffered entry in the lane; this never yields a proof, so
    /// callers that must free the cells go through `into_recyclable`.
    pub(super) fn release_checkpoints(
        &mut self,
        start: u32,
        end: u32,
    ) -> Result<(), DemotionRefusal> {
        self.recurrent_prefixes
            .release(&mut self.session, start, end)
    }

    /// Record every unrecorded checkpoint straight from this lane's cells,
    /// leaving everything resident. Only meaningful on a host-visible stage;
    /// elsewhere the recorder reports it unsupported and the checkpoints
    /// wait for recycle time.
    pub(super) fn record_in_place(&mut self, start: u32, end: u32) -> usize {
        let Self {
            session,
            recurrent_prefixes,
            leases,
            ..
        } = self;
        match recurrent_prefixes.record_in_place(session, start, end, leases) {
            Ok(recorded) => recorded,
            Err(reason) => {
                tracing::debug!(reason = ?reason, "in-place record refused at park; will retry at recycle");
                0
            }
        }
    }
}

impl RecyclableLane {
    /// The one `StageSession::reset` on a lane. See the contract test below.
    /// A recorder reader still copying the cells finishes first; after a
    /// successful reset no checkpoint can refer to the cells, so none stays.
    pub(super) fn reset(mut self) -> Result<RuntimeLaneSession, LaneResetFailed> {
        let index = self.0.index;
        self.0.leases.wait_idle();
        match self.0.session.reset() {
            Ok(()) => {
                self.0.recurrent_prefixes.forget_after_reset();
                Ok(self.0)
            }
            Err(error) => Err(LaneResetFailed { index, error }),
        }
    }
}

impl RuntimeState {
    /// Reset a recyclable lane for reuse. A failed reset drops the lane and
    /// returns its index to the free list, matching every existing caller.
    pub(super) fn reset_recycled(&mut self, lane: RecyclableLane) -> Result<RuntimeLaneSession> {
        lane.reset().map_err(|failed| {
            self.free_lane_indices.push(failed.index);
            failed.error
        })
    }

    /// Oldest idle lane whose checkpoints the recorder accepted. Refused
    /// lanes keep their cells and return to the pool; the caller then falls
    /// back to a fresh lane or to back-pressure.
    pub(super) fn recycle_idle_session(&mut self) -> Result<Option<RuntimeLaneSession>> {
        let mut retained = Vec::new();
        let recycled = loop {
            let Some(lane) = self.take_idle_session() else {
                break None;
            };
            match lane.into_recyclable(self.layer_start, self.layer_end) {
                Recycle::Ready(lane) => break Some(lane),
                Recycle::Retained(RetainedLane { lane, reason }) => {
                    emit_exact_state_diagnostic(
                        "lane_recycle_retained",
                        &[("lane_index", lane.index as u64)],
                    );
                    tracing::debug!(
                        lane_index = lane.index,
                        reason = ?reason,
                        "recorder refused the idle lane's export; lane stays parked"
                    );
                    retained.push(lane);
                }
            }
        };
        self.idle_sessions.extend(retained);
        recycled.map(|lane| self.reset_recycled(lane)).transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stopped_recorder_ends_retention_at_once() {
        let now = Instant::now();
        assert_eq!(
            refusal_policy(DemotionRefusal::WorkerStopped, None, now),
            RefusalPolicy::Discard(CheckpointDiscard::RecorderStopped)
        );
        assert_eq!(
            refusal_policy(DemotionRefusal::WorkerStopped, Some(now), now),
            RefusalPolicy::Discard(CheckpointDiscard::RecorderStopped)
        );
    }

    #[test]
    fn transient_refusals_retain_until_the_bound_then_discard() {
        let first = Instant::now();
        for reason in [
            DemotionRefusal::CreditFull,
            DemotionRefusal::ByteFull,
            DemotionRefusal::QueueFull,
        ] {
            assert_eq!(refusal_policy(reason, None, first), RefusalPolicy::Retain);
            assert_eq!(
                refusal_policy(reason, Some(first), first + MAX_RETAINED_REFUSAL / 2),
                RefusalPolicy::Retain
            );
            assert_eq!(
                refusal_policy(reason, Some(first), first + MAX_RETAINED_REFUSAL),
                RefusalPolicy::Discard(CheckpointDiscard::RetentionExpired)
            );
        }
    }

    /// `StageSession::reset` through `lane.session` bypasses the proof. The
    /// borrow checker cannot see that, so this keeps every reset in the
    /// runtime state behind `RecyclableLane::reset`.
    #[test]
    fn lane_reset_only_happens_through_recyclable_lane() {
        let sources = [
            ("runtime_state.rs", include_str!("../runtime_state.rs")),
            ("frame_operations.rs", include_str!("frame_operations.rs")),
            (
                "iteration_execution.rs",
                include_str!("iteration_execution.rs"),
            ),
            ("lane_lifecycle.rs", include_str!("lane_lifecycle.rs")),
            ("panic_recovery.rs", include_str!("panic_recovery.rs")),
            (
                "resident_recurrent.rs",
                include_str!("resident_recurrent.rs"),
            ),
            (
                "restore_transaction.rs",
                include_str!("restore_transaction.rs"),
            ),
            ("state_transfer.rs", include_str!("state_transfer.rs")),
        ];
        for (name, source) in sources {
            assert!(
                !source.contains("session.reset()"),
                "{name} resets a lane without a RecyclableLane"
            );
        }
    }
}
