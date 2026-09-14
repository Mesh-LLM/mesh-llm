//! Pop the ingress ring in submission order and apply each fact through
//! the transactional reducer: only an accepted fact appends a replay frame
//! and fans out to subscribers, so a rejected input never appears on the
//! stream.
//!
//! ## Why this is short now
//!
//! It used to have to rebuild a total order. Four bounded containers were
//! written by producers under a shared gate that also minted the sequence,
//! and a pass had to select one global sequence prefix across all four,
//! drain each to that boundary, merge, and sort -- while holding the gate
//! every producer needed.
//!
//! With one ring there is nothing to reconstruct. Pop order is submission
//! order, and a sequence is assigned when a fact publishes, so the
//! published order is the pop order by construction. `collect_sequence_prefix`,
//! the per-lane `drain_before_limit` cutoffs, the full-table progress scan,
//! and the merge-and-sort are all gone with it, and a pass holds nothing a
//! producer ever takes.
//!
//! ## What a pass does
//!
//! 1. Pop up to `max` items, returning ring credits as it goes.
//! 2. Route by class ([`super::lanes`]): terminals, state transitions, and
//!    diagnostics into this pass's batch; progress into the one lane that
//!    persists, to be exported on its own 100 ms cadence.
//! 3. Apply the batch in arrival order, assigning each published fact the
//!    next sequence.
//! 4. Release the slot of each terminal that applied, and evict its
//!    reducer state -- after every fact in the pass has applied, so
//!    eviction cannot race an application that would re-create the entry.
//!
//! A root whose own terminal has drained but which still has occupied
//! children has only its *slot release* deferred ([`release_or_defer`]),
//! bounded by [`settle_pending_root_releases`]'s `CHILD_SETTLE_GRACE`. Its
//! terminal still applies and publishes immediately. Force-releasing the
//! children instead would reject their real, still-in-flight terminals as
//! stale moments later.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use mesh_llm_runtime_event_contracts::{
    DeliveryClass, EventSequence, OperationId, OperationScope, RuntimeFact,
};

use super::lanes::{PassBatch, PendingFact};
use super::{ChildSlot, PendingRootRelease, RuntimeEventEngine};
use crate::runtime_events::config::{
    CHILD_SETTLE_GRACE, SHUTDOWN_DRAIN_DEADLINE, TOTAL_OPERATION_BOUND,
};
use crate::runtime_events::ingress::{IngressFact, IngressItem};
use crate::runtime_events::reducer::{ReduceOutcome, ReducerInput, apply};
use crate::runtime_events::replay::ReplayFrame;
use crate::runtime_events::reservation::{SlotHandle, TerminalClaim};

// Shutdown work is split into small batches so the elapsed deadline is
// observed between bounded pieces of reducer work. This is an
// implementation chunk size, not a public queue capacity.
const SHUTDOWN_WORK_CHUNK: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrainReport {
    /// Facts successfully reduced and published during this pass.
    pub applied: usize,
    pub left_queued: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct DrainPass {
    report: DrainReport,
    /// Items physically popped, before any liveness or reducer filtering.
    /// Shutdown continues on this rather than on the published count, so a
    /// rejected prefix cannot stop it early.
    consumed: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ShutdownReport {
    pub applied: usize,
    pub started_with: usize,
    pub remaining_after_deadline: usize,
}

/// Whether a pass should export whatever progress it is holding regardless
/// of the 100 ms window. Shutdown does; a normal pass does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProgressFlush {
    WhenDue,
    Everything,
}

impl RuntimeEventEngine {
    /// Drain and apply everything currently queued.
    pub fn drain(&self) -> DrainReport {
        self.drain_up_to(None)
    }

    /// Drain and apply at most `max` queued items, leaving the rest for the
    /// next pass. Held progress still exports only when its own window is
    /// due.
    pub fn drain_up_to(&self, max: Option<usize>) -> DrainReport {
        self.drain_at(max, Instant::now())
    }

    /// Test seam for the 100 ms progress window: identical to
    /// [`Self::drain_up_to`] but takes an explicit `now` instead of reading
    /// the wall clock, so a test can prove "at most one frame per 100 ms"
    /// with pure `Instant` arithmetic -- no real sleep, and no dependency
    /// on `tokio::time::pause` (which does not virtualize
    /// `std::time::Instant::now()`).
    #[cfg(test)]
    pub(crate) fn drain_up_to_at(&self, max: Option<usize>, now: Instant) -> DrainReport {
        self.drain_at(max, now)
    }

    fn drain_at(&self, max: Option<usize>, now: Instant) -> DrainReport {
        let _audit = crate::runtime_events::lock_audit::scope(
            crate::runtime_events::lock_audit::Context::Drain,
        );
        let _drain = self.drain_gate().lock();
        self.drain_pass(max, now, ProgressFlush::WhenDue).report
    }

    /// One pass: pop, route, apply, release.
    ///
    /// Holds `drain_gate` (taken by the caller) and, briefly, the progress
    /// lane and the publication/reducer locks. It holds nothing a producer
    /// takes, which is what the stall tests in
    /// `engine::tests::nonblocking` assert against a pass deliberately
    /// parked mid-flight.
    fn drain_pass(&self, max: Option<usize>, now: Instant, flush: ProgressFlush) -> DrainPass {
        // Test seam (`runtime_events::drain_hold`): park here, inside
        // everything this pass is holding, so a producer that is coupled to
        // the drain waits the full hold duration and one that is not returns
        // immediately.
        #[cfg(test)]
        if let Some(hold) = self.drain_hold() {
            hold.hold();
        }

        let mut popped = Vec::new();
        self.ingress()
            .pop_up_to(max.unwrap_or(usize::MAX), &mut popped);
        let consumed = popped.len();

        let mut batch = PassBatch::default();
        // Slots to release once every fact in this pass has applied. A
        // release taken mid-pass would invalidate facts submitted BEFORE
        // it that are still waiting their turn in the same batch.
        let mut releases: Vec<(OperationScope, SlotHandle)> = Vec::new();

        // Scopes whose slot was already released by a cancelling thread,
        // waiting only for their reducer state to be evicted.
        let mut cancelled: Vec<OperationScope> = Vec::new();
        for item in popped {
            match item {
                IngressItem::Released { scope } => {
                    // A cancelled operation must not publish the progress
                    // snapshot it was holding.
                    self.progress_lane().lock().forget(scope);
                    cancelled.push(scope);
                }
                IngressItem::Fact(entry) => self.route(entry, &mut batch),
            }
        }

        let due = match flush {
            ProgressFlush::WhenDue => self.progress_lane().lock().take_due(now),
            ProgressFlush::Everything => self.progress_lane().lock().take_all(),
        };
        for entry in due {
            batch.push(entry, DeliveryClass::Progress);
        }

        for _ in 0..batch.superseded_progress {
            self.health.bump_dropped_progress();
        }

        let mut applied = 0;
        for pending in batch.drain() {
            if !pending.live {
                self.count_stale(&pending);
                continue;
            }
            if pending.fact.delivery_class() == DeliveryClass::Terminal
                && let Some(handle) = pending.handle
            {
                releases.push((pending.scope, handle));
            }
            if let Some(handle) = pending.handle {
                // Retaining typed identities is a payload-lock write, so it
                // happens here rather than on the submitting thread.
                self.table()
                    .remember_scope(handle, pending.fact.data().scope.clone());
            }
            if self.apply_and_publish(pending) {
                applied += 1;
            }
        }

        let mut released_now = cancelled;
        for (scope, handle) in releases {
            if let Some(released) = release_or_defer(self, scope, handle, now) {
                released_now.push(released);
            }
        }

        // Evict only now that every fact this pass drained has applied:
        // evicting earlier could race an application that would just
        // re-insert the entry a moment later.
        for scope in released_now {
            self.progress_lane().lock().forget(scope);
            self.evict_operation(scope);
        }

        settle_pending_root_releases(self, now);
        DrainPass {
            report: DrainReport {
                applied,
                left_queued: self.ingress().len(),
            },
            consumed,
        }
    }

    /// Send one popped fact to its class's destination, recording whether
    /// its reservation is still valid.
    ///
    /// Liveness is decided HERE, while routing, not at apply time: every
    /// slot this pass releases is released after the apply loop, so a fact
    /// submitted before a terminal cannot be invalidated by that terminal's
    /// own release. What survives this check is then the reducer's call,
    /// and the reducer counts what it rejects.
    fn route(&self, entry: IngressFact, batch: &mut PassBatch) {
        let class = entry.class;
        let live = self.reservation_is_valid(&entry);
        let mut pending = PendingFact::from(entry);
        pending.live = live;
        if class == DeliveryClass::Progress {
            // Progress is rate-limited, not queued: only the latest value
            // per operation survives to the next export window.
            if self.progress_lane().lock().record(pending) {
                batch.superseded_progress += 1;
            }
            return;
        }
        batch.push(pending, class);
    }

    /// Whether `entry`'s reservation is still the one it was submitted
    /// against. A terminal is exempt from the scope and cancellation
    /// checks: winning the write-once claim is what admitted it.
    fn reservation_is_valid(&self, entry: &IngressFact) -> bool {
        if entry.class == DeliveryClass::Terminal {
            return entry
                .handle
                .is_some_and(|handle| self.table().is_current(handle));
        }
        if !entry.reserved {
            return true;
        }
        entry.handle.is_some_and(|handle| {
            self.table().is_current(handle)
                && self.table().occupant(handle) == Some(entry.scope)
                && !self.table().is_cancelled(handle)
        })
    }

    fn count_stale(&self, pending: &PendingFact) {
        match pending.fact.delivery_class() {
            DeliveryClass::Progress => self.health.bump_dropped_progress(),
            DeliveryClass::Diagnostic => self.health.bump_dropped_diagnostic(),
            DeliveryClass::Terminal | DeliveryClass::StateTransition => {}
        }
    }

    fn apply_and_publish(&self, pending: PendingFact) -> bool {
        let sequence = self.next_publication_sequence();
        self.apply_and_publish_fact(
            pending.scope,
            sequence,
            pending.fact,
            pending.synthesized,
            pending.reserved,
        )
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
    ) -> bool {
        let _publication = self.publication_gate().lock();
        let fact_arc = Arc::new(fact.clone());
        let metadata = fact.metadata();
        let input = ReducerInput {
            scope,
            ingress_sequence,
            native_sequence: metadata.and_then(|metadata| {
                metadata
                    .native_sequence
                    .map(|observation| observation.sequence)
                    .or_else(|| {
                        metadata
                            .native_source
                            .as_ref()
                            .map(|source| source.sequence)
                    })
            }),
            wall_clock_hint: metadata
                .and_then(|metadata| metadata.wall_clock_unix_ns)
                .map(|timestamp| i64::try_from(timestamp).unwrap_or(i64::MAX)),
            synthesized,
            reserved,
            fact,
        };
        let mut reducer_state = self.reducer_state().lock();
        let ReduceOutcome::Applied(next) = apply(&reducer_state, input) else {
            self.health.bump_reducer_rejected();
            return false;
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
        self.set_published_frontier(ingress_sequence);
        true
    }

    /// Evict `scope`'s tracked reducer state -- the release-triggered
    /// eviction path (task 6-fix defect A). Callers must only invoke this
    /// once `scope`'s reservation-table slot has ACTUALLY been released
    /// AND every fact drained in the same pass has already been applied
    /// (see the module doc comment and [`release_or_defer`] /
    /// [`release_pending_root`] below, the only two call sites).
    pub(super) fn evict_operation(&self, scope: OperationScope) {
        // Eviction mutates the reducer snapshot that `attach` captures next
        // to replay and the published frontier. Keep the mutation inside the
        // same publication boundary so a reconnect cannot observe a settled
        // scope in one snapshot and its removal in the next at one cursor.
        let _publication = self.publication_gate().lock();
        let mut reducer_state = self.reducer_state().lock();
        *reducer_state = crate::runtime_events::reducer::evict(&reducer_state, scope);
    }

    /// Increment `rebuild_generation` and evict every retained replay frame,
    /// simulating a reducer crash/restart recovering into a fresh window.
    pub fn rebuild(&self) -> u64 {
        let _audit = crate::runtime_events::lock_audit::scope(
            crate::runtime_events::lock_audit::Context::Drain,
        );
        let _drain = self.drain_gate().lock();
        let _publication = self.publication_gate().lock();
        let generation = self.rebuild_generation.fetch_add(1, Ordering::AcqRel) + 1;
        let previous_frontier = self.published_frontier();
        self.rebuild_invalidated_through
            .store(previous_frontier, Ordering::Release);
        self.has_rebuild_invalidated_through
            .store(true, Ordering::Release);
        self.health.set_rebuild_generation(generation);
        let evicted = self.replay.evict_all();
        for _ in 0..evicted {
            self.health.bump_replay_evicted();
        }
        let mut reducer_state = self.reducer_state().lock();
        if let crate::runtime_events::reducer::RebuildOutcome::Rebuilt(next) =
            crate::runtime_events::reducer::rebuild(&reducer_state, generation)
        {
            *reducer_state = next;
        }
        generation
    }

    /// Begin shutdown: block new admission, then drain at most `budget`
    /// accepted work items (`None` drains until the deadline). Entries left
    /// queued past the budget or cooperative deadline are recorded as
    /// shutdown-degraded rather than silently dropped. A root release still
    /// deferred at this point is force-settled after the driver has stopped,
    /// then drained once more instead of retaining its slot for process life.
    pub fn shutdown(&self, budget: Option<usize>) -> ShutdownReport {
        self.shutdown_until(budget, Instant::now() + SHUTDOWN_DRAIN_DEADLINE)
    }

    /// Shutdown variant used by the async driver owner so driver cancellation
    /// and the exclusive final drain share one deadline.
    pub(crate) fn shutdown_until(
        &self,
        budget: Option<usize>,
        deadline: Instant,
    ) -> ShutdownReport {
        let _audit = crate::runtime_events::lock_audit::scope(
            crate::runtime_events::lock_audit::Context::Drain,
        );
        let _drain = self.drain_gate().lock();
        self.close_admission();
        let started_with = self.ingress().len();
        let mut report = DrainReport::default();
        let mut remaining_budget = budget;
        while Instant::now() < deadline && remaining_budget.is_none_or(|remaining| remaining > 0) {
            if !self.drain_shutdown_chunk(deadline, budget, &mut remaining_budget, &mut report) {
                break;
            }
        }
        if !self.pending_root_releases_is_empty() && Instant::now() < deadline {
            let forced_now = Instant::now() + CHILD_SETTLE_GRACE;
            settle_pending_root_releases(self, forced_now);
            while Instant::now() < deadline
                && remaining_budget.is_none_or(|remaining| remaining > 0)
            {
                if !self.drain_shutdown_chunk(deadline, budget, &mut remaining_budget, &mut report)
                {
                    break;
                }
            }
        }
        let (remaining, terminal_remainder) = self.pending_work_counts();
        if remaining > 0 {
            self.health.bump_shutdown_degraded();
            for _ in 0..terminal_remainder {
                self.health.bump_terminal_delivery_failed();
            }
        }
        ShutdownReport {
            applied: report.applied,
            started_with,
            remaining_after_deadline: remaining,
        }
    }

    /// Apply one bounded shutdown chunk. Both shutdown phases use this
    /// helper so synthesis, deadline checks, and budget accounting cannot
    /// drift apart when a deferred root is force-settled.
    ///
    /// Held progress is exported unconditionally here: the 100 ms window
    /// will never come due again, and a stranded snapshot would be a
    /// silent loss rather than a counted one.
    fn drain_shutdown_chunk(
        &self,
        deadline: Instant,
        budget: Option<usize>,
        remaining_budget: &mut Option<usize>,
        report: &mut DrainReport,
    ) -> bool {
        if Instant::now() >= deadline || remaining_budget.is_some_and(|remaining| remaining == 0) {
            return false;
        }
        let chunk = remaining_budget.map_or(SHUTDOWN_WORK_CHUNK, |remaining| {
            remaining.min(SHUTDOWN_WORK_CHUNK)
        });
        // Drain BEFORE synthesizing. Applying a fact is what retains its
        // typed identities on the slot (`remember_scope`), and synthesis
        // reads those identities to give a synthesized terminal the right
        // scope. Synthesizing first would settle a reservation whose
        // identities were still sitting unread in the ring.
        let pass = self.drain_pass(
            budget.map(|limit| limit.min(chunk)).or(Some(chunk)),
            Instant::now(),
            ProgressFlush::Everything,
        );
        let synthesized = self.synthesize_unsettled_reservations();
        report.applied += pass.report.applied;
        if let Some(remaining) = remaining_budget.as_mut() {
            *remaining = (*remaining).saturating_sub(pass.report.applied);
        }
        pass.consumed > 0 || synthesized > 0
    }

    /// Synthesize a terminal for every occupied slot whose write-once claim
    /// is still available. The original family synthesizer is retained in
    /// the reservation table, so a live guard is settled with the same
    /// terminal kind a normal guard drop would produce.
    ///
    /// Admission is already closed by the caller. The claim CAS is still
    /// what arbitrates: a guard dropping concurrently either wins and this
    /// skips the slot, or loses and publishes nothing.
    fn synthesize_unsettled_reservations(&self) -> usize {
        let mut synthesized = 0;
        for unsettled in self.table().unsettled() {
            if self.table().claim_terminal(unsettled.handle) == TerminalClaim::Refused {
                continue;
            }
            let synthetic = (unsettled.synthetic_terminal)();
            let synthetic = match unsettled.scope_identities.as_ref() {
                Some(scope) => synthetic.with_scope(scope),
                None => synthetic,
            };
            let placed = self.place(IngressItem::Fact(IngressFact {
                scope: unsettled.scope,
                fact: self.fill_ingress_metadata(synthetic, Instant::now()),
                handle: Some(unsettled.handle),
                reserved: true,
                synthesized: true,
                class: DeliveryClass::Terminal,
            }));
            if placed.is_ok() {
                synthesized += 1;
            } else {
                self.health.bump_terminal_delivery_failed();
            }
        }
        synthesized
    }

    fn pending_root_releases_is_empty(&self) -> bool {
        self.pending_root_releases.lock().is_empty()
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
    if let OperationScope::Child { root, .. } = scope {
        engine.table().release(handle);
        // The child is no longer a candidate for root settlement. Remove
        // only this exact generation: the index may already have been
        // reused by an unrelated operation by the time this lock is taken.
        remove_child_slot(engine, root, handle);
        return Some(scope);
    }
    let OperationScope::Root(root) = scope else {
        unreachable!("child scope returned above");
    };
    if has_occupied_children(engine, root) {
        engine.pending_root_releases.lock().insert(
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
    child_slots(engine, root).into_iter().any(|child| {
        engine
            .table()
            .occupant(SlotHandle {
                index: child.index,
                generation: child.generation,
            })
            .is_some()
    })
}

fn forget_children(engine: &RuntimeEventEngine, root: OperationId) {
    engine.children_by_root.lock().remove(&root);
}

fn remove_child_slot(engine: &RuntimeEventEngine, root: OperationId, handle: SlotHandle) {
    let mut children_by_root = engine.children_by_root.lock();
    let Some(children) = children_by_root.get_mut(&root) else {
        return;
    };
    children.retain(|child| child.index != handle.index || child.generation != handle.generation);
    let empty = children.is_empty();
    if empty {
        children_by_root.remove(&root);
    }
}

fn child_slots(engine: &RuntimeEventEngine, root: OperationId) -> Vec<ChildSlot> {
    engine
        .children_by_root
        .lock()
        .get(&root)
        .cloned()
        .unwrap_or_default()
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
/// written and placed through the SAME write-once claim
/// `OperationReservation::drop` already uses for a genuinely-dropped
/// guard, so it is picked up and applied+published by the ordinary apply
/// loop on this engine's very next pass -- there is no second reducer
/// path here, and no fact is applied synchronously inside this
/// function.
fn settle_pending_root_releases(engine: &RuntimeEventEngine, now: Instant) {
    let mut released_roots = Vec::new();
    {
        let candidates: Vec<(OperationId, SlotHandle, Instant)> = engine
            .pending_root_releases
            .lock()
            .iter()
            .map(|(root, entry)| (*root, entry.handle, entry.deadline))
            .collect();

        for (root, handle, deadline) in candidates {
            let outstanding = occupied_children(engine, root);
            if outstanding.is_empty() {
                if release_pending_root(engine, root, handle) {
                    released_roots.push(root);
                }
                continue;
            }
            if now < deadline {
                continue;
            }
            for child in outstanding {
                synthesize_child_not_delivered(engine, child);
            }
            if release_pending_root(engine, root, handle) {
                released_roots.push(root);
            }
        }
    }
    // Reducer eviction is intentionally outside ingress admission. The drain
    // gate still excludes a concurrent drain/cancel, so no later apply can
    // resurrect one of these released scopes before eviction.
    for root in released_roots {
        engine.evict_operation(OperationScope::root_only(root));
    }
}

fn occupied_children(engine: &RuntimeEventEngine, root: OperationId) -> Vec<ChildSlot> {
    child_slots(engine, root)
        .into_iter()
        .filter(|child| {
            engine
                .table()
                .occupant(SlotHandle {
                    index: child.index,
                    generation: child.generation,
                })
                .is_some()
        })
        .collect()
}

fn release_pending_root(
    engine: &RuntimeEventEngine,
    root: OperationId,
    handle: SlotHandle,
) -> bool {
    if engine.table().occupant(handle).is_none() {
        return false;
    }
    engine.table().release(handle);
    engine.pending_root_releases.lock().remove(&root);
    forget_children(engine, root);
    true
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
/// the claim's own generation check would then always match (having just
/// been read from the same slot), landing this child's stale synthesized
/// terminal in a slot a completely different, currently in-flight
/// operation now legitimately owns -- and refusing THAT operation's real
/// terminal afterward as a duplicate. The reserve-time generation makes
/// `claim_terminal` correctly detect the mismatch and refuse instead.
fn synthesize_child_not_delivered(engine: &RuntimeEventEngine, child: ChildSlot) {
    let handle = SlotHandle {
        index: child.index,
        generation: child.generation,
    };
    if engine.table().occupant(handle).is_none() {
        return;
    }
    let synthetic = (child.synthetic_terminal)();
    let synthetic = match engine.table().scope_identities(handle) {
        Some(scope) => synthetic.with_scope(&scope),
        None => synthetic,
    };
    let Some(scope) = engine.table().occupant(handle) else {
        return;
    };
    if engine.table().claim_terminal(handle) == TerminalClaim::Refused {
        return;
    }
    let placed = engine.place(IngressItem::Fact(IngressFact {
        scope,
        fact: engine.fill_ingress_metadata(synthetic, Instant::now()),
        handle: Some(handle),
        reserved: true,
        synthesized: true,
        class: DeliveryClass::Terminal,
    }));
    if placed.is_err() {
        engine.health().bump_terminal_delivery_failed();
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
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::time::Duration;

    use mesh_llm_runtime_event_contracts::{
        FamilyFact, NativeRuntimeEventKind, ProcessInstanceId, RuntimeEventIngress, SubmitOutcome,
    };
    use tokio::sync::Notify;

    use super::*;
    use crate::runtime_events::health::EngineHealth;
    use crate::runtime_events::ingress::Ingress;
    use crate::runtime_events::lock_audit::{AuditedMutex, LockClass};
    use crate::runtime_events::reducer::ReducerSnapshot;
    use crate::runtime_events::replay::ReplayBuffer;
    use crate::runtime_events::reservation::ReservationTable;
    use crate::runtime_events::subscribers::SubscriberRegistry;

    fn engine_with_tiny_replay_age(max_age: Duration) -> Arc<RuntimeEventEngine> {
        Arc::new(RuntimeEventEngine {
            table: ReservationTable::new(64),
            ingress: Ingress::new(),
            next_sequence: AtomicU64::new(1),
            drain_gate: AuditedMutex::new(LockClass::DrainGate, ()),
            publication_gate: AuditedMutex::new(LockClass::PublicationGate, ()),
            published_frontier: AtomicU64::new(0),
            rebuild_invalidated_through: AtomicU64::new(0),
            has_rebuild_invalidated_through: AtomicBool::new(false),
            replay: ReplayBuffer::with_bounds(1_000, usize::MAX, max_age),
            subscribers: SubscriberRegistry::with_capacity(64),
            health: EngineHealth::default(),
            children_by_root: AuditedMutex::new(LockClass::ChildrenByRoot, HashMap::new()),
            pending_root_releases: AuditedMutex::new(
                LockClass::PendingRootReleases,
                HashMap::new(),
            ),
            shutting_down: AtomicBool::new(false),
            rebuild_generation: AtomicU64::new(0),
            progress_lane: AuditedMutex::new(
                LockClass::ProgressLane,
                crate::runtime_events::engine::lanes::ProgressLane::default(),
            ),
            reducer_state: AuditedMutex::new(LockClass::ReducerState, ReducerSnapshot::empty()),
            process_instance: ProcessInstanceId::new(),
            process_started: Instant::now(),
            telemetry: OnceLock::new(),
            progress_diagnostic_class_bypass: AtomicBool::new(false),
            notify: Notify::new(),
            ingress_latency: crate::runtime_events::ingress_latency::IngressLatencyReservoir::new(),
            drain_hold: OnceLock::new(),
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
