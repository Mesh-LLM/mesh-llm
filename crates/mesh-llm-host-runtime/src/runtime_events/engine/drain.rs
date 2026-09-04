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
//!
//! Task 6-fix defect A (`.omo/plans/event-system-fixes.md`): a scope whose
//! reservation is actually released here is ALSO evicted from the
//! reducer's `operations` map (`RuntimeEventEngine::evict_operation`,
//! `reducer::evict`), so the map tracks in-flight operations only instead
//! of every settled one forever. [`release_or_defer`] now reports whether
//! it released THIS call (`Some(scope)`) or deferred (`None`); the
//! per-entry loop in [`RuntimeEventEngine::drain_up_to_inner`] batches
//! every scope released this pass and evicts them only AFTER this pass's
//! `pending` facts -- including that scope's own just-drained terminal --
//! have already been applied, so eviction can never race an application
//! that would just re-insert the entry a moment later. [`release_pending_root`]
//! evicts immediately: by the time a deferred root's slot is finally
//! released, its own terminal was already applied in whichever earlier
//! pass first drained it, so there is no such race there.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use mesh_llm_runtime_event_contracts::{EventSequence, OperationId, OperationScope, RuntimeFact};

use super::{ChildSlot, PendingRootRelease, RuntimeEventEngine};
use crate::runtime_events::config::{
    CHILD_SETTLE_GRACE, PROGRESS_EXPORT_INTERVAL, TOTAL_OPERATION_BOUND,
};
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
        // 5th element `reserved` (R1 fix, task 6-fix,
        // `.omo/plans/event-system-fixes.md`): whether this fact arrived
        // through a reservation-bound submission -- always `true` for a
        // drained Terminal record (a terminal is only ever written into an
        // OCCUPIED reservation-table slot) and for a flushed progress slot
        // (progress coalescing itself requires a live `SlotHandle`); comes
        // from the lane's own stored value for state-transition/diagnostic
        // entries, which MAY be `false` (`unreserved_ingress`).
        let mut pending: Vec<(u64, OperationScope, RuntimeFact, bool, bool)> = Vec::new();
        let mut released_now: Vec<OperationScope> = Vec::new();
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
                    true,
                ));
            }
            if let Some(released_scope) = release_or_defer(self, scope, handle, now) {
                released_now.push(released_scope);
            }
            applied += 1;
        }

        let state_entries = self.state_lane().drain();
        applied += state_entries.len();
        pending.extend(
            state_entries
                .into_iter()
                .map(|(scope, fact, sequence, reserved)| (sequence, scope, fact, false, reserved)),
        );

        let diagnostic_entries = self.diagnostic_lane().drain();
        applied += diagnostic_entries.len();
        pending.extend(
            diagnostic_entries
                .into_iter()
                .map(|(scope, fact, sequence, reserved)| (sequence, scope, fact, false, reserved)),
        );

        pending.sort_by_key(|(sequence, ..)| *sequence);
        for (sequence, scope, fact, synthesized, reserved) in pending {
            self.apply_and_publish_fact(scope, sequence, fact, synthesized, reserved);
        }

        // Defect A (task 6-fix): evict every scope released THIS pass only
        // now that every fact drained this pass has already been applied
        // above -- see the module doc comment for why the ordering matters.
        for scope in released_now {
            self.evict_operation(scope);
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
        reserved: bool,
    ) {
        let fact_arc = Arc::new(fact.clone());
        let input = ReducerInput {
            scope,
            ingress_sequence,
            native_sequence: None,
            wall_clock_hint: None,
            synthesized,
            reserved,
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
        // Also-required observability fix (task 6-fix, review finding on
        // top of defect A): `with_operation`'s settled-only capacity
        // backstop used to silently `break` out of its sweep when nothing
        // settled was left to evict, restoring unbounded growth with no
        // counter and no log.
        //
        // R1 CORRECTION (task 6-fix, `.omo/plans/event-system-fixes.md`):
        // the comment that used to sit here claimed release-triggered
        // eviction made that stall "structurally unreachable in the
        // steady state" -- false: six production call sites
        // (`unreserved_ingress` with a fresh `OperationId` per event, no
        // reservation ever backing them) could genuinely drive the
        // settled-only sweep's "nothing left to evict" branch forever.
        // `ReducerSnapshot`'s new `unreserved_order` bounded LRU
        // (`reducer/state.rs`) fixes that by bounding those scopes
        // independently, so the check below is now against
        // `TOTAL_OPERATION_BOUND` (`RESERVATION_TABLE_CAPACITY +
        // UNRESERVED_OPERATION_BOUND`) -- the TRUE combined ceiling both
        // mechanisms together guarantee -- rather than the old
        // reservation-only `RESERVATION_TABLE_CAPACITY`, which legitimate
        // unreserved traffic can now exceed without anything being
        // "stalled". This bump should stay unreachable in practice again,
        // for the right reason this time.
        if next.operation_count() > TOTAL_OPERATION_BOUND {
            self.health.bump_reducer_eviction_stalled();
        }
        *reducer_state = next;
        drop(reducer_state);

        let mut frame = ReplayFrame {
            sequence: EventSequence::new(ingress_sequence),
            rebuild_generation: self.rebuild_generation.load(Ordering::Acquire),
            scope,
            fact: fact_arc,
            recorded_at: Instant::now(),
            // Placeholder, overwritten immediately below. `event_frame`
            // only reads `sequence`/`rebuild_generation`/`scope`/`fact` --
            // never `wire_bytes` itself -- so computing the real bytes
            // against this not-yet-filled `frame` is safe.
            wire_bytes: Arc::from(Vec::new()),
        };
        // Task 9 (`.omo/plans/event-system-fixes.md`, defect D11):
        // serialize this frame's `runtime_event` wire bytes ONCE, here, at
        // push -- not once per subscriber delivery. `frames::event_frame`
        // is the exact byte-for-byte SSE encoder the v1 stream has always
        // used; calling it from here (the one call site granted to
        // engine/drain.rs for this seam) instead of duplicating its logic
        // guarantees these bytes are identical to what a fresh
        // `event_frame` call would have produced (pinned by
        // `runtime_event_api_tests::sample_frames_fixture_is_byte_exact_for_every_frame_type`).
        let encoded = crate::api::routes::runtime_events::frames::event_frame(self, &frame);
        frame.wire_bytes = Arc::from(encoded.into_bytes());

        // Task 8-fix E1 (`.omo/plans/event-system-fixes.md`): `push` now
        // reports the real number of frames it evicted -- a single push can
        // evict more than one (see `replay::ReplayBuffer::push`'s doc
        // comment) -- so every evicted frame is credited here, not one
        // bump per push. `bump_replay_evicted_by` is itself a no-op
        // (including no version bump) when `evicted == 0`.
        let evicted = self.replay.push(frame.clone());
        self.health.bump_replay_evicted_by(evicted as u64);
        self.subscribers.publish(frame);
    }

    /// Evict `scope`'s tracked reducer state -- the release-triggered
    /// eviction path (task 6-fix defect A). Callers must only invoke this
    /// once `scope`'s reservation-table slot has ACTUALLY been released
    /// AND every fact drained in the same pass has already been applied
    /// (see the module doc comment and [`release_or_defer`] /
    /// [`release_pending_root`] below, the only two call sites).
    pub(super) fn evict_operation(&self, scope: OperationScope) {
        let mut reducer_state = self
            .reducer_state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *reducer_state = crate::runtime_events::reducer::evict(&reducer_state, scope);
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
            // Always `reserved: true`: progress coalescing itself
            // (`ReservationTable::coalesce_progress`) requires a live
            // `SlotHandle`, so a slot can only ever appear here via a
            // reservation-bound submission (R1 fix, task 6-fix).
            self.apply_and_publish_fact(scope, sequence, fact, false, true);
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
///
/// Returns `Some(scope)` when this call released the slot immediately --
/// the caller then owns evicting `scope` from the reducer (task 6-fix
/// defect A) once it is safe to (see the module doc comment); returns
/// `None` when the release was deferred, in which case
/// [`release_pending_root`] below evicts once the deferred release
/// actually happens.
pub(super) fn release_or_defer(
    engine: &RuntimeEventEngine,
    scope: OperationScope,
    handle: SlotHandle,
    now: Instant,
) -> Option<OperationScope> {
    let OperationScope::Root(root) = scope else {
        engine.table().release(handle);
        return Some(scope);
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
        return None;
    }
    engine.table().release(handle);
    forget_children(engine, root);
    Some(scope)
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
    // Task 6-fix defect A: safe to evict immediately here, unlike the
    // per-entry loop's immediate-release path -- this root's OWN operation
    // state was already durably applied by the `pending` loop in whichever
    // earlier `drain_up_to_inner` pass first drained its terminal (task 5's
    // child-settle-grace defers only the SLOT release, never the reducer
    // apply), so there is no pending-apply-would-re-insert-it race here.
    engine.evict_operation(OperationScope::root_only(root));
}

/// Write and enqueue `child`'s own synthesized `terminal_not_delivered`
/// terminal -- a no-op if it already settled on its own (a real
/// submission, or a caller-side guard drop), OR if its slot was released
/// and reused by a DIFFERENT operation, between the outstanding-children
/// snapshot in [`settle_pending_root_releases`] and this call.
///
/// Uses `child.generation` -- captured at RESERVE time -- rather than
/// re-reading `current_generation(child.index)` here: re-reading would
/// return whatever generation currently occupies that index, which
/// `write_terminal`'s own generation check would then always match
/// (having just been read from the same slot), landing this child's stale
/// synthesized terminal in a slot a completely different, currently
/// in-flight operation now legitimately owns -- and rejecting THAT
/// operation's real terminal afterward as a duplicate. The reserve-time
/// generation makes `write_terminal` correctly detect the mismatch and
/// no-op instead.
fn synthesize_child_not_delivered(engine: &RuntimeEventEngine, child: ChildSlot) {
    if engine.table().is_occupied(child.index).is_none() {
        return;
    }
    let handle = SlotHandle {
        index: child.index,
        generation: child.generation,
    };
    let record = TerminalRecord {
        fact: (child.synthetic_terminal)(),
        synthesized: true,
    };
    if engine.table().write_terminal(handle, record) {
        engine.wake().push_next(handle);
    }
}

// Task 8-fix E1 (`.omo/plans/event-system-fixes.md`): the engine-level
// proof that `apply_and_publish_fact` above credits every frame a single
// `ReplayBuffer::push` evicts, not one bump per push. `engine/mod.rs` has
// no production seam to shrink the replay buffer's `max_age` below the
// frozen 300s `REPLAY_MAX_AGE` (and task 8-fix's grant does not extend to
// adding one there), so `engine_with_tiny_replay_age` below builds a
// `RuntimeEventEngine` by struct literal -- every field matches
// `RuntimeEventEngine::with_capacities` exactly except `replay`, which
// uses a millisecond-scale age bound so the test can force a genuine
// same-push multi-eviction without a real 300s wait.
//
// Task 9 CORRECTION (`.omo/plans/event-system-fixes.md`, defect D11): this
// comment used to claim "only the age dimension can [evict more than one
// frame per push]; the count/byte dimensions can never... since each
// push's own eviction loop already restores the invariant before the next
// push can violate it again" -- true of the COUNT dimension always, and
// was true of the BYTE dimension only while `ReplayBuffer` charged every
// frame the same fixed `APPROX_FRAME_BYTE_COST`. Task 9 replaced that
// fixed cost with each frame's REAL, variable pre-serialized wire-byte
// length (`replay::ReplayFrame::wire_bytes`), so the byte dimension can
// now ALSO evict more than one frame per push -- proven at the
// `ReplayBuffer` level (not here; see the ownership note below) by
// `replay::tests::a_single_large_frame_can_evict_multiple_smaller_frames_via_the_byte_bound`.
// The test below still isolates the AGE dimension specifically
// (`max_bytes: usize::MAX` means the byte bound can never fire here), so
// it remains a valid, UNCHANGED proof of age-driven multi-eviction; it is
// simply no longer the only dimension capable of it.
//
// This is legal because `engine::drain::tests` is a descendant module of
// `engine`, where every field of `RuntimeEventEngine` is defined
// (module-private, not `pub`) -- the same visibility rule that already
// lets this file's own `apply_and_publish_fact` read
// `self.replay`/`self.health` directly.
#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    use mesh_llm_runtime_event_contracts::{
        FamilyFact, NativeRuntimeEventKind, ProcessInstanceId, RuntimeEventIngress, SubmitOutcome,
    };
    use tokio::sync::Notify;

    use super::*;
    use crate::runtime_events::engine::lanes::{DiagnosticLane, StateLane};
    use crate::runtime_events::health::EngineHealth;
    use crate::runtime_events::reducer::ReducerSnapshot;
    use crate::runtime_events::replay::ReplayBuffer;
    use crate::runtime_events::reservation::ReservationTable;
    use crate::runtime_events::subscribers::SubscriberRegistry;
    use crate::runtime_events::wake::WakeList;

    fn engine_with_tiny_replay_age(max_age: Duration) -> Arc<RuntimeEventEngine> {
        Arc::new(RuntimeEventEngine {
            table: ReservationTable::new(64),
            wake: WakeList::new(),
            replay: ReplayBuffer::with_bounds(1_000, usize::MAX, max_age),
            subscribers: SubscriberRegistry::with_capacity(64),
            health: EngineHealth::default(),
            children_by_root: Mutex::new(HashMap::new()),
            pending_root_releases: Mutex::new(HashMap::new()),
            shutting_down: AtomicBool::new(false),
            rebuild_generation: AtomicU64::new(0),
            state_lane: StateLane::default(),
            diagnostic_lane: DiagnosticLane::default(),
            reducer_state: Mutex::new(ReducerSnapshot::empty()),
            process_instance: ProcessInstanceId::new(),
            telemetry: OnceLock::new(),
            progress_diagnostic_class_bypass: AtomicBool::new(false),
            notify: Notify::new(),
            progress_last_flush: Mutex::new(None),
        })
    }

    fn distinct_state_transition_fact() -> RuntimeFact {
        RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeInitialized))
    }

    /// A fresh `OperationScope` each call, submitted unreserved (bypassing
    /// the reservation table entirely) so coalescing never merges it with
    /// a sibling call: the state lane keys on `(OperationScope, kind)`, and
    /// every call here mints a brand new `OperationId`.
    fn submit_one(engine: &Arc<RuntimeEventEngine>) {
        let scope = OperationScope::root_only(OperationId::new());
        let outcome = engine
            .unreserved_ingress(scope)
            .try_submit(distinct_state_transition_fact());
        assert_eq!(outcome, SubmitOutcome::Accepted);
    }

    /// Fails at the parent commit (`apply_and_publish_fact` bumping
    /// `EngineHealth::replay_evicted` by exactly one per push regardless of
    /// the real eviction count) and passes once it reports the real count.
    /// Three distinct facts are drained together so each publishes its own
    /// replay frame within microseconds of the others (well under the tiny
    /// age bound); after real wall-clock time passes that bound, a fourth
    /// push must evict all three in ONE `ReplayBuffer::push` call.
    #[test]
    fn a_single_push_that_evicts_several_stale_frames_credits_every_one_at_the_engine_level() {
        let engine = engine_with_tiny_replay_age(Duration::from_millis(5));

        submit_one(&engine);
        submit_one(&engine);
        submit_one(&engine);
        engine.drain();
        assert_eq!(
            engine.health().snapshot().replay_evicted,
            0,
            "all three frames were recorded within microseconds of each \
             other, well under the 5ms age bound -- nothing is stale yet"
        );
        assert_eq!(engine.replay().len(), 3);

        // Real wall-clock time passing the tiny age bound, well past it for
        // safety margin against scheduling jitter on a loaded machine.
        std::thread::sleep(Duration::from_millis(50));

        submit_one(&engine);
        engine.drain();

        assert_eq!(
            engine.health().snapshot().replay_evicted,
            3,
            "one push evicted three stale frames; the engine-level EngineHealth \
             counter must credit every one of them, not bump by one per push"
        );
        assert_eq!(engine.replay().len(), 1, "only the fresh frame remains");
    }
}
