//! Drain wake entries in ingress-sequence order and apply each through the
//! transactional reducer: only an accepted fact appends a replay frame and
//! fans out to subscribers, so a rejected input never appears on the
//! stream. The reservation is released after the reducer has settled the
//! outcome either way, matching the release-after-ack contract.
//!
//! Every `drain`/`drain_up_to` call also drains the state-transition lane
//! and the diagnostic queue fully, and flushes any progress slots due
//! under the 100 ms export interval (task 4,
//! `.omo/plans/event-system-fixes.md`) -- fixing review defect D2, where
//! only terminal-class facts ever reached the reducer. `engine.drain()`
//! stays the SAME single stable entry point the task-3 driver
//! (`runtime_events::driver`) calls on every `Notify`/fallback tick; this
//! module is the only place that decides what "drain" now does.
//!
//! Task 5 (review defect D8) additionally fixes how a root's release
//! interacts with its still-occupied children: [`release_or_defer`]
//! replaces the old `cascade_children`/`force_complete_child` pair, which
//! force-released every outstanding child the instant the root's own
//! terminal drained WITHOUT ever writing a terminal for it -- so a real,
//! still-in-flight child terminal arriving moments later was rejected as
//! stale (`TerminalDeliveryFailed`) instead of accepted. A root's own
//! terminal still applies and publishes immediately either way; only the
//! ROOT's slot release is now deferred while a child remains occupied,
//! bounded by [`settle_pending_root_releases`]'s `CHILD_SETTLE_GRACE`.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use mesh_llm_runtime_event_contracts::{EventSequence, OperationId, OperationScope, RuntimeFact};

use super::{ChildSlot, PendingRootRelease, RuntimeEventEngine};
use crate::runtime_events::config::{CHILD_SETTLE_GRACE, PROGRESS_EXPORT_INTERVAL};
use crate::runtime_events::reducer::{ReduceOutcome, ReducerInput, apply};
use crate::runtime_events::replay::ReplayFrame;
use crate::runtime_events::reservation::{SlotHandle, TerminalRecord};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrainReport {
    pub applied: usize,
    pub left_queued: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShutdownReport {
    pub applied: usize,
    pub started_with: usize,
    pub remaining_after_deadline: usize,
}

impl RuntimeEventEngine {
    /// Drain and apply every currently queued wake entry, the full
    /// state-transition lane, and the full diagnostic queue; flush any
    /// progress slots due under the 100 ms export interval.
    pub fn drain(&self) -> DrainReport {
        self.drain_up_to(None)
    }

    /// Drain and apply at most `max` wake entries, leaving the rest
    /// queued (`None` drains everything currently queued). The
    /// state-transition lane and diagnostic queue are always drained in
    /// full regardless of `max` -- both are independently bounded (4,096
    /// distinct keys, 2,048 entries), so draining either fully is always
    /// bounded work, never unbounded.
    pub fn drain_up_to(&self, max: Option<usize>) -> DrainReport {
        self.drain_up_to_inner(max, Instant::now())
    }

    /// Test-only seam for the 100 ms progress-flush gate: identical to
    /// [`Self::drain_up_to`] but takes an explicit `now` instead of
    /// reading the wall clock, so a test can prove "at most one frame per
    /// 100 ms" with pure `Instant` arithmetic -- no real sleep, and no
    /// dependency on `tokio::time::pause` (which does not virtualize
    /// `std::time::Instant::now()`). Mirrors `EngineHealth::publish_at`'s
    /// identical caller-supplied-`now` pattern.
    #[cfg(test)]
    pub(crate) fn drain_up_to_at(&self, max: Option<usize>, now: Instant) -> DrainReport {
        self.drain_up_to_inner(max, now)
    }

    fn drain_up_to_inner(&self, max: Option<usize>, now: Instant) -> DrainReport {
        let entries = match max {
            Some(limit) => self.wake().drain_up_to(limit),
            None => self.wake().drain_all(),
        };
        // Every fact pulled out of the wake list, state lane, or
        // diagnostic queue this pass is collected here BEFORE any of them
        // is applied, and then applied in ONE ingress-sequence-sorted
        // pass below. Applying per-lane batches back to back (all
        // terminals, then all state-transitions) would let a scope's
        // terminal settle the reducer's per-scope state before an
        // earlier-minted state-transition or diagnostic for that SAME
        // scope -- sitting in a different lane, drained a moment later in
        // program order -- ever applies, spuriously rejecting it as
        // `OperationSettled` even though it was never actually stale.
        let mut pending: Vec<(u64, OperationScope, RuntimeFact, bool)> = Vec::new();
        let mut applied = 0;
        for entry in entries {
            let handle = entry.handle;
            if !self.table().is_current(handle) {
                continue;
            }
            let Some(scope) = self.table().is_occupied(handle.index) else {
                continue;
            };
            if let Some(record) = self.table().terminal_record(handle) {
                pending.push((
                    entry.ingress_sequence,
                    scope,
                    record.fact,
                    record.synthesized,
                ));
            }
            release_or_defer(self, scope, handle, now);
            applied += 1;
        }

        let state_entries = self.state_lane().drain();
        applied += state_entries.len();
        pending.extend(
            state_entries
                .into_iter()
                .map(|(scope, fact, sequence)| (sequence, scope, fact, false)),
        );

        let diagnostic_entries = self.diagnostic_lane().drain();
        applied += diagnostic_entries.len();
        pending.extend(
            diagnostic_entries
                .into_iter()
                .map(|(scope, fact, sequence)| (sequence, scope, fact, false)),
        );

        pending.sort_by_key(|(sequence, ..)| *sequence);
        for (sequence, scope, fact, synthesized) in pending {
            self.apply_and_publish_fact(scope, sequence, fact, synthesized);
        }

        applied += self.maybe_flush_progress(now);
        settle_pending_root_releases(self, now);
        DrainReport {
            applied,
            left_queued: self.wake().len(),
        }
    }

    /// Apply one fact through the transactional reducer and, on
    /// acceptance, append the replay frame and fan it out to subscribers.
    /// Shared by every lane's drain step (terminal, state-transition,
    /// diagnostic, progress) so all four delivery classes go through
    /// EXACTLY the same publication path -- there is no second reducer
    /// path anywhere in the engine.
    fn apply_and_publish_fact(
        &self,
        scope: OperationScope,
        ingress_sequence: u64,
        fact: RuntimeFact,
        synthesized: bool,
    ) {
        let fact_arc = Arc::new(fact.clone());
        let input = ReducerInput {
            scope,
            ingress_sequence,
            native_sequence: None,
            wall_clock_hint: None,
            synthesized,
            fact,
        };
        let mut reducer_state = self
            .reducer_state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ReduceOutcome::Applied(next) = apply(&reducer_state, input) else {
            self.health.bump_reducer_rejected();
            return;
        };
        *reducer_state = next;
        drop(reducer_state);

        let frame = ReplayFrame {
            sequence: EventSequence::new(ingress_sequence),
            rebuild_generation: self.rebuild_generation.load(Ordering::Acquire),
            scope,
            fact: fact_arc,
            recorded_at: Instant::now(),
        };
        if self.replay.push(frame.clone()) {
            self.health.bump_replay_evicted();
        }
        self.subscribers.publish(frame);
    }

    /// Flush every slot with pending progress if at least
    /// `PROGRESS_EXPORT_INTERVAL` (100 ms) has elapsed since the last
    /// flush; a no-op otherwise. The first ever call always flushes
    /// (matches `EngineHealth::publish_at`'s identical `None => true`
    /// convention), so an idle engine's very first drain establishes the
    /// baseline instant rather than waiting a full interval. Returns the
    /// number of slots flushed (0 when not due).
    fn maybe_flush_progress(&self, now: Instant) -> usize {
        let mut last = self
            .progress_last_flush
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let due = match *last {
            None => true,
            Some(previous) => now.duration_since(previous) >= PROGRESS_EXPORT_INTERVAL,
        };
        if !due {
            return 0;
        }
        *last = Some(now);
        drop(last);
        let entries = self.table().take_all_progress();
        let processed = entries.len();
        for (scope, fact, sequence) in entries {
            self.apply_and_publish_fact(scope, sequence, fact, false);
        }
        processed
    }

    /// Increment `rebuild_generation` and evict every retained replay frame,
    /// simulating a reducer crash/restart recovering into a fresh window.
    pub fn rebuild(&self) -> u64 {
        let generation = self.rebuild_generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.health.set_rebuild_generation(generation);
        let evicted = self.replay.evict_all();
        for _ in 0..evicted {
            self.health.bump_replay_evicted();
        }
        let mut reducer_state = self
            .reducer_state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let crate::runtime_events::reducer::RebuildOutcome::Rebuilt(next) =
            crate::runtime_events::reducer::rebuild(&reducer_state, generation)
        {
            *reducer_state = next;
        }
        generation
    }

    /// Begin shutdown: block new admission, then drain at most `budget`
    /// wake entries (`None` drains fully). Entries left queued past the
    /// budget are recorded as shutdown-degraded rather than silently
    /// dropped, matching the deadline-degradation rule. A root release
    /// still deferred at this point (task 5's child-settle grace) cannot
    /// wait out its real `CHILD_SETTLE_GRACE` here -- shutdown is
    /// synchronous and the driver task is aborted immediately after this
    /// call returns -- so it is force-settled now (treated as already past
    /// its deadline) and drained once more, exactly like a normal grace
    /// expiry, rather than left occupying its slot for the rest of process
    /// life.
    pub fn shutdown(&self, budget: Option<usize>) -> ShutdownReport {
        self.shutting_down.store(true, Ordering::Release);
        let started_with = self.wake().len();
        let mut report = self.drain_up_to(budget);
        if !self.pending_root_releases_is_empty() {
            let forced_now = Instant::now() + CHILD_SETTLE_GRACE;
            settle_pending_root_releases(self, forced_now);
            let follow_up = self.drain_up_to(budget);
            report.applied += follow_up.applied;
        }
        let remaining = self.wake().len();
        if remaining > 0 {
            self.health.bump_shutdown_degraded();
            for _ in 0..remaining {
                self.health.bump_terminal_delivery_failed();
            }
        }
        ShutdownReport {
            applied: report.applied,
            started_with,
            remaining_after_deadline: remaining,
        }
    }

    fn pending_root_releases_is_empty(&self) -> bool {
        self.pending_root_releases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    }
}

/// Release `handle`'s now-settled slot for `scope` -- or, for a `Root`
/// scope with at least one still-occupied child, DEFER the release
/// instead (review defect D8). A `Child` scope, or a `Root` with no
/// occupied children right now, releases immediately exactly as a plain
/// release always did. Shared by the per-entry drain loop above and
/// `OperationReservation::cancel` (`engine/mod.rs`), so a root released
/// via explicit pre-work cancellation gets the identical deferred-release
/// contract as one released by its own terminal draining.
pub(super) fn release_or_defer(
    engine: &RuntimeEventEngine,
    scope: OperationScope,
    handle: SlotHandle,
    now: Instant,
) {
    let OperationScope::Root(root) = scope else {
        engine.table().release(handle);
        return;
    };
    if has_occupied_children(engine, root) {
        engine
            .pending_root_releases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                root,
                PendingRootRelease {
                    handle,
                    deadline: now + CHILD_SETTLE_GRACE,
                },
            );
        return;
    }
    engine.table().release(handle);
    forget_children(engine, root);
}

fn has_occupied_children(engine: &RuntimeEventEngine, root: OperationId) -> bool {
    engine
        .children_by_root
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&root)
        .is_some_and(|children| {
            children
                .iter()
                .any(|child| engine.table().is_occupied(child.index).is_some())
        })
}

fn forget_children(engine: &RuntimeEventEngine, root: OperationId) {
    engine
        .children_by_root
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&root);
}

/// Resolve every root whose own terminal has settled but whose slot
/// release was deferred by [`release_or_defer`]. Called on EVERY drain
/// pass -- the task-3 engine-owned driver ticks this at least every
/// `TUI_RENDER_TICK`, plus immediately on `Notify` -- so a root's grace
/// deadline is enforced without a second background task; the driver's
/// own cadence is the only clock this needs. A root whose children have
/// ALL settled since (drained through the ordinary per-entry loop above,
/// exactly like any other terminal) releases immediately, however much
/// grace time is left. A root still short a child past `deadline` gets
/// each remaining child's OWN synthesized `terminal_not_delivered`
/// written and enqueued through the SAME `write_terminal` + `push_next`
/// mechanism `OperationReservation::drop` already uses for a genuinely-
/// dropped guard, so it is picked up and applied+published by the
/// ordinary per-entry loop on this engine's very next drain call -- there
/// is no second reducer path here, and no fact is applied synchronously
/// inside this function.
fn settle_pending_root_releases(engine: &RuntimeEventEngine, now: Instant) {
    let candidates: Vec<(OperationId, SlotHandle, Instant)> = engine
        .pending_root_releases
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .map(|(root, entry)| (*root, entry.handle, entry.deadline))
        .collect();

    for (root, handle, deadline) in candidates {
        let outstanding = occupied_children(engine, root);
        if outstanding.is_empty() {
            release_pending_root(engine, root, handle);
            continue;
        }
        if now < deadline {
            continue;
        }
        for child in outstanding {
            synthesize_child_not_delivered(engine, child);
        }
        release_pending_root(engine, root, handle);
    }
}

fn occupied_children(engine: &RuntimeEventEngine, root: OperationId) -> Vec<ChildSlot> {
    engine
        .children_by_root
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&root)
        .map(|children| {
            children
                .iter()
                .copied()
                .filter(|child| engine.table().is_occupied(child.index).is_some())
                .collect()
        })
        .unwrap_or_default()
}

fn release_pending_root(engine: &RuntimeEventEngine, root: OperationId, handle: SlotHandle) {
    engine.table().release(handle);
    engine
        .pending_root_releases
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&root);
    forget_children(engine, root);
}

/// Write and enqueue `child`'s own synthesized `terminal_not_delivered`
/// terminal -- a no-op if it already settled on its own (a real
/// submission, or a caller-side guard drop) between the outstanding-
/// children snapshot in [`settle_pending_root_releases`] and this call.
fn synthesize_child_not_delivered(engine: &RuntimeEventEngine, child: ChildSlot) {
    if engine.table().is_occupied(child.index).is_none() {
        return;
    }
    let handle = SlotHandle {
        index: child.index,
        generation: engine.table().current_generation(child.index),
    };
    let record = TerminalRecord {
        fact: (child.synthetic_terminal)(),
        synthesized: true,
    };
    if engine.table().write_terminal(handle, record) {
        engine.wake().push_next(handle);
    }
}
