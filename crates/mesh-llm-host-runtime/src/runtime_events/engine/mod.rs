//! The host runtime-event engine: admission, the write-once terminal slot,
//! and the minimal acknowledgement seam a reducer (task 4) drains.

mod drain;
mod lanes;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use mesh_llm_runtime_event_contracts::{
    DeliveryClass, DiagnosticEventKind, FactMetadata, OperationId, OperationScope, Outcome,
    ProcessInstanceId, ProducerSource, ReasonCode, RuntimeEventIngress, RuntimeFact, Severity,
    SubmitOutcome,
};
use tokio::sync::Notify;

use super::config::{CHILD_MULTIPLIER, RESERVATION_TABLE_CAPACITY};
use super::health::{EngineHealth, EngineHealthSnapshot};
use super::ingress::{Ingress, IngressFact, IngressItem, PushError};
use super::ingress_latency::IngressLatencyReservoir;
use super::lock_audit::{AuditedMutex, LockClass};
use super::reducer::ReducerSnapshot;
use super::replay::ReplayBuffer;
use super::reservation::{ReservationTable, SlotHandle, TerminalClaim};
use super::subscribers::{SubscribeError, SubscriberRegistry, SubscriptionHandle};
use super::telemetry::RuntimeEventTelemetryQueue;

use lanes::ProgressLane;

/// Builds the family-correct synthesized terminal fact for a dropped guard
/// or shutdown. Engine layer stays family-agnostic; callers (family
/// adapters in later tasks) supply the right `Terminal`-class kind with
/// `outcome: Unknown` and `reason: TerminalNotDelivered` already set.
pub type SyntheticTerminal = fn() -> RuntimeFact;

/// The immutable handoff from the engine to a live presentation consumer.
///
/// `RuntimeEventEngine::attach` registers the subscription and captures every
/// snapshot field while holding the publication guard. The stream can then
/// write the captured replay/state/health frames and start receiving from the
/// already-registered subscription without a publication gap between those
/// steps. The attachment owns the subscription so malformed cursors or failed
/// initial writes release it normally when the value is dropped.
pub struct RuntimeEventAttachment {
    pub subscription: SubscriptionHandle,
    pub replay: Vec<super::replay::ReplayFrame>,
    pub reducer: Arc<ReducerSnapshot>,
    pub health: EngineHealthSnapshot,
    /// Ingress p99 captured under the same publication guard as `health`, so
    /// the initial health frame does not combine one snapshot with a later
    /// reservoir read.
    pub ingress_p99_us: Option<u64>,
    /// The last ingress sequence that was accepted by the reducer and
    /// published. Sequence zero is the empty-snapshot sentinel; real ingress
    /// starts at one.
    pub published_frontier: u64,
    /// Highest sequence known to have been evicted from replay. This lets a
    /// reconnect distinguish a genuine replay gap from a cursor that simply
    /// falls between published ingress sequences.
    pub replay_evicted_through: Option<u64>,
    /// Inclusive frontier invalidated by the most recent rebuild. A cursor
    /// equal to the pre-rebuild frontier must receive a gap/state refresh even
    /// when no new event has minted a larger sequence yet.
    pub rebuild_invalidated_through: Option<u64>,
    pub rebuild_generation: u64,
}

/// One admitted child slot under a root: its reservation-table index and
/// the GENERATION it was admitted at, plus the family-supplied
/// synthesizer. The SAME synthesizer a genuinely-dropped guard uses
/// (`OperationReservation::drop` below) is reused by `engine::drain::
/// settle_pending_root_releases` at child-settle-grace expiry (task 5,
/// `.omo/plans/event-system-fixes.md`), so a forgotten child gets the
/// identical family-correct `terminal_not_delivered` fact either way.
/// `generation` is captured at RESERVE time (not re-read from the table at
/// synthesis time): re-reading `current_generation(index)` at synthesis
/// time would return whatever generation currently occupies that index --
/// including a DIFFERENT operation's, if this child's slot was already
/// released and reused between the outstanding-children snapshot and the
/// synthesis call -- landing a stale synthesized terminal in a reused
/// slot instead of safely no-op'ing.
#[derive(Clone, Copy)]
struct ChildSlot {
    index: usize,
    generation: u64,
    synthetic_terminal: SyntheticTerminal,
}

/// A root whose own terminal settled while at least one child was still
/// occupied: its slot release is deferred until `engine::drain::
/// settle_pending_root_releases` either finds every child has settled on
/// its own, or `deadline` (`CHILD_SETTLE_GRACE` past the moment the root
/// settled) has passed.
struct PendingRootRelease {
    handle: SlotHandle,
    deadline: Instant,
}

pub struct RuntimeEventEngine {
    table: ReservationTable,
    /// The producer boundary: one bounded lock-free ring, popped only by
    /// the driver's drain pass. There is no admission gate -- see
    /// `runtime_events::ingress`.
    ingress: Ingress,
    /// The next sequence a published fact will receive. Assigned at
    /// publication, not at submission, so published sequences are
    /// contiguous and monotonic by construction.
    next_sequence: AtomicU64,
    /// Serializes drain passes and rebuilds. No producer ever takes this,
    /// which is what lets a pass hold it for as long as reducer work and
    /// subscriber fan-out take.
    drain_gate: AuditedMutex<()>,
    /// Publication boundary for replay, subscribers, reducer state, and the
    /// monotonic published frontier. Attachment captures under this lock so a
    /// new stream cannot fall between replay and live delivery.
    publication_gate: AuditedMutex<()>,
    published_frontier: AtomicU64,
    rebuild_invalidated_through: AtomicU64,
    has_rebuild_invalidated_through: AtomicBool,
    replay: ReplayBuffer,
    subscribers: SubscriberRegistry,
    health: EngineHealth,
    children_by_root: AuditedMutex<HashMap<OperationId, Vec<ChildSlot>>>,
    /// Roots deferred by `engine::drain::release_or_defer` -- review
    /// defect D8 -- and resolved by `engine::drain::
    /// settle_pending_root_releases` on every drain pass. See
    /// [`PendingRootRelease`].
    pending_root_releases: AuditedMutex<HashMap<OperationId, PendingRootRelease>>,
    shutting_down: AtomicBool,
    rebuild_generation: AtomicU64,
    /// The one lane that survives a drain pass: the latest progress value
    /// per operation, exported on the 100 ms cadence. Taken only in drain
    /// context.
    progress_lane: AuditedMutex<ProgressLane>,
    reducer_state: AuditedMutex<Arc<ReducerSnapshot>>,
    process_instance: ProcessInstanceId,
    process_started: Instant,
    telemetry: OnceLock<Arc<RuntimeEventTelemetryQueue>>,
    /// The plan's `event-disabled` trial-mode class bypass (task 19).
    /// `false` by default -- production startup sets this from
    /// `mesh_llm_config::event_system_progress_diagnostic_bypass_enabled()`.
    /// See `set_progress_diagnostic_class_bypass` and `submit` below for
    /// the single contract boundary this flag gates.
    progress_diagnostic_class_bypass: AtomicBool,
    /// Signaled once per `SubmitOutcome::Accepted` submission (never on
    /// `Coalesced`/`Dropped*`/`Rejected*`/`TerminalDeliveryFailed`)
    /// -- the engine-owned driver's (`runtime_events::driver`, task 3) wake
    /// source, alongside its own fallback tick. See [`Self::notified`].
    notify: Notify,
    /// The fixed, always-present ingress-latency ring backing
    /// `Self::ingress_p99_us`, written unconditionally in `submit` below,
    /// independent of whether a telemetry queue was ever installed.
    ingress_latency: IngressLatencyReservoir,
    /// Test seam: when installed, every drain pass parks inside its
    /// critical section for the hold's duration. See
    /// `runtime_events::drain_hold`.
    #[cfg(test)]
    drain_hold: OnceLock<Arc<super::drain_hold::DrainHold>>,
}

fn inferred_rust_severity(fact: &RuntimeFact) -> Severity {
    if let RuntimeFact::Diagnostic(diagnostic) = fact {
        return match diagnostic.kind() {
            DiagnosticEventKind::FatalNativeFailure => Severity::Fatal,
            DiagnosticEventKind::WarningCleared | DiagnosticEventKind::DegradedOperationExited => {
                Severity::Info
            }
            DiagnosticEventKind::WarningRaised
            | DiagnosticEventKind::RecoverableNativeFailure
            | DiagnosticEventKind::FallbackApplied
            | DiagnosticEventKind::DegradedOperationEntered
            | DiagnosticEventKind::InvariantProtocolViolation => Severity::Warning,
        };
    }

    match fact.data().outcome {
        Some(Outcome::Failure) => Severity::Error,
        Some(Outcome::Rejected | Outcome::Cancelled | Outcome::Unknown) => Severity::Warning,
        Some(Outcome::Success) => Severity::Info,
        None => match fact.data().reason.as_ref() {
            Some(
                ReasonCode::InvalidConfiguration
                | ReasonCode::UnsupportedCapability
                | ReasonCode::MissingArtifact
                | ReasonCode::ArtifactIoFailure
                | ReasonCode::ModelFormatOrLoadFailure
                | ReasonCode::BackendInitializationFailure
                | ReasonCode::DeviceUnavailable
                | ReasonCode::ResourceAllocationFailure
                | ReasonCode::OutOfMemory
                | ReasonCode::ContextExhausted
                | ReasonCode::StageUnavailable
                | ReasonCode::ProcessCrash
                | ReasonCode::IncompatibleAbiOrFeatureSet
                | ReasonCode::InternalRuntimeFailure
                | ReasonCode::UnknownFailure,
            ) => Severity::Error,
            Some(
                ReasonCode::Timeout
                | ReasonCode::Cancellation
                | ReasonCode::TerminalNotDelivered
                | ReasonCode::ReservationExhausted
                | ReasonCode::Unknown(_),
            ) => Severity::Warning,
            None if fact.kind_id().contains("fatal") => Severity::Fatal,
            None if fact.kind_id().contains("degraded")
                || fact.kind_id().contains("warning")
                || fact.kind_id().contains("unavailable") =>
            {
                Severity::Warning
            }
            None => Severity::Info,
        },
    }
}

impl RuntimeEventEngine {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Self::with_capacity(RESERVATION_TABLE_CAPACITY)
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Arc<Self> {
        Self::with_capacities(capacity, super::config::SUBSCRIBER_LAG_MAX_FRAMES)
    }

    /// Full constructor: reservation-table capacity plus an explicit
    /// subscriber frame-count lag capacity. `with_capacity` delegates here
    /// using the frozen `SUBSCRIBER_LAG_MAX_FRAMES` value; tests use this
    /// directly to build a "slow subscriber" scenario reachable in a
    /// handful of publishes instead of the frozen 1,024.
    #[must_use]
    pub fn with_capacities(capacity: usize, subscriber_lag_frames: usize) -> Arc<Self> {
        Arc::new(Self {
            table: ReservationTable::new(capacity),
            ingress: Ingress::new(),
            // Sequence zero is reserved for the empty snapshot cursor;
            // real publication starts at one.
            next_sequence: AtomicU64::new(1),
            drain_gate: AuditedMutex::new(LockClass::DrainGate, ()),
            publication_gate: AuditedMutex::new(LockClass::PublicationGate, ()),
            published_frontier: AtomicU64::new(0),
            rebuild_invalidated_through: AtomicU64::new(0),
            has_rebuild_invalidated_through: AtomicBool::new(false),
            replay: ReplayBuffer::new(),
            subscribers: SubscriberRegistry::with_capacity(subscriber_lag_frames),
            health: EngineHealth::default(),
            children_by_root: AuditedMutex::new(
                LockClass::ChildrenByRoot,
                HashMap::with_capacity(capacity),
            ),
            pending_root_releases: AuditedMutex::new(
                LockClass::PendingRootReleases,
                HashMap::with_capacity(capacity),
            ),
            shutting_down: AtomicBool::new(false),
            rebuild_generation: AtomicU64::new(0),
            progress_lane: AuditedMutex::new(LockClass::ProgressLane, ProgressLane::default()),
            reducer_state: AuditedMutex::new(LockClass::ReducerState, ReducerSnapshot::empty()),
            process_instance: ProcessInstanceId::new(),
            process_started: Instant::now(),
            telemetry: OnceLock::new(),
            progress_diagnostic_class_bypass: AtomicBool::new(false),
            notify: Notify::new(),
            ingress_latency: IngressLatencyReservoir::new(),
            #[cfg(test)]
            drain_hold: OnceLock::new(),
        })
    }

    /// Install this engine's telemetry sample queue: the live wiring seam
    /// for the certification ingress-latency instrument. Every producer
    /// already funnels its `try_submit` calls through `submit` below (via
    /// `ScopedIngress`/`UnreservedIngress`, minted by `reserve_root`/
    /// `reserve_child`/`unreserved_ingress`), so installing the queue here
    /// -- once, at the same startup site the engine itself is installed --
    /// observes every real producer's traffic without touching any task
    /// 9-12 producer file. Idempotent: a second call is a silent no-op
    /// (`OnceLock` refuses a second write), matching "install once at
    /// startup" alongside `install_runtime_event_engine`. Before this is
    /// called, or if it is never called, `submit` behaves identically with
    /// zero telemetry overhead.
    pub fn install_telemetry_queue(&self, queue: Arc<RuntimeEventTelemetryQueue>) {
        let _ = self.telemetry.set(queue);
    }

    /// Sets the plan's `event-disabled` trial-mode class bypass: when
    /// `true`, `submit` below bypasses ONLY `Progress` and `Diagnostic`
    /// class facts, at this single contract boundary, before they ever
    /// reach a lane -- `Terminal` and `StateTransition` facts (and
    /// therefore reservations and the reducer) are completely unaffected
    /// regardless of this flag. Idempotent and safe to call repeatedly;
    /// production startup calls this once from
    /// `mesh_llm_config::event_system_progress_diagnostic_bypass_enabled()`.
    /// Defaults to `false` (full production pipeline).
    pub fn set_progress_diagnostic_class_bypass(&self, enabled: bool) {
        self.progress_diagnostic_class_bypass
            .store(enabled, Ordering::Relaxed);
    }

    /// Current class-bypass state (see `set_progress_diagnostic_class_bypass`).
    #[must_use]
    pub fn progress_diagnostic_class_bypass(&self) -> bool {
        self.progress_diagnostic_class_bypass
            .load(Ordering::Relaxed)
    }

    /// This engine's process-local identity: the first component of the
    /// wire cursor grammar `rt1:<process-instance-uuid>:<sequence>`. Minted
    /// once per engine instance and never changes for its lifetime.
    #[must_use]
    pub fn process_instance(&self) -> ProcessInstanceId {
        self.process_instance
    }

    /// The last ingress sequence that was accepted by the reducer and
    /// published. This is the only cursor frontier exposed to stream clients;
    /// merely minted, queued, or rejected sequences are never acknowledged by
    /// the API. Sequence zero is the wire-level empty snapshot sentinel.
    #[must_use]
    pub fn highest_known_sequence(&self) -> Option<u64> {
        Some(self.published_frontier())
    }

    /// The publication cursor captured by [`Self::attach`]. Real ingress
    /// sequences begin at one, so zero means that no event has been published.
    #[must_use]
    pub fn published_frontier(&self) -> u64 {
        self.published_frontier.load(Ordering::Acquire)
    }

    /// Register a live subscriber and capture replay, reducer state, health,
    /// generation, and the publication frontier as one coherent handoff.
    /// Publication cannot interleave between registration and capture because
    /// both this method and `engine::drain`'s publication path use the same
    /// short-lived gate. The gate is released before the caller performs any
    /// socket writes.
    pub fn attach(&self) -> Result<RuntimeEventAttachment, SubscribeError> {
        let _publication = self.publication_gate.lock();
        let subscription = self.subscribers.subscribe()?;
        let replay = self.replay.snapshot();
        let reducer = self.reducer_snapshot();
        let health = self.health.snapshot();
        let ingress_p99_us = self.ingress_p99_us();
        let published_frontier = self.published_frontier();
        let replay_evicted_through = self.replay.evicted_through();
        let rebuild_invalidated_through = self
            .has_rebuild_invalidated_through
            .load(Ordering::Acquire)
            .then(|| self.rebuild_invalidated_through.load(Ordering::Acquire));
        Ok(RuntimeEventAttachment {
            subscription,
            replay,
            reducer,
            health,
            ingress_p99_us,
            published_frontier,
            replay_evicted_through,
            rebuild_invalidated_through,
            rebuild_generation: health.rebuild_generation,
        })
    }

    /// Snapshot of the reducer's current, fully-applied state. Cheap: an
    /// `Arc` clone, never a copy of the underlying map.
    #[must_use]
    pub fn reducer_snapshot(&self) -> Arc<ReducerSnapshot> {
        Arc::clone(&self.reducer_state.lock())
    }

    pub(super) fn reducer_state(&self) -> &AuditedMutex<Arc<ReducerSnapshot>> {
        &self.reducer_state
    }

    pub(super) fn progress_lane(&self) -> &AuditedMutex<ProgressLane> {
        &self.progress_lane
    }

    /// Test-only: drain, then report every kind this engine has published.
    ///
    /// Replaces the old `published_kinds()`, which read a producer-side
    /// latest-value lane that no longer exists. This asserts something
    /// strictly stronger: the fact reached the reducer and the stream,
    /// rather than merely landing in a container.
    #[cfg(test)]
    #[must_use]
    pub fn published_kinds(&self) -> Vec<&'static str> {
        self.drain();
        self.replay
            .snapshot()
            .iter()
            .map(|frame| frame.fact.kind_id())
            .collect()
    }

    #[must_use]
    pub fn health(&self) -> &EngineHealth {
        &self.health
    }

    /// The current p99 (99th percentile) ingress duration in whole
    /// microseconds, over the fixed in-process reservoir `submit` writes
    /// unconditionally -- `None` before `INGRESS_LATENCY_MIN_SAMPLES`
    /// samples have ever been recorded. Never gated on OTLP telemetry
    /// configuration (task 13, `.omo/plans/event-system-fixes.md`).
    #[must_use]
    pub fn ingress_p99_us(&self) -> Option<u64> {
        self.ingress_latency.p99_micros()
    }

    #[must_use]
    pub fn replay(&self) -> &ReplayBuffer {
        &self.replay
    }

    #[must_use]
    pub fn subscribers(&self) -> &SubscriberRegistry {
        &self.subscribers
    }

    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::Acquire)
    }

    /// Await the engine's next `SubmitOutcome::Accepted` signal (or return
    /// immediately if one arrived since the caller's last await). Backs the
    /// engine-owned driver's (`runtime_events::driver`, task 3) `select!`
    /// wake condition; a signal missed by a race is bounded by the
    /// driver's own fallback tick, so no precise check-then-wait ordering
    /// is required here.
    pub(crate) async fn notified(&self) {
        self.notify.notified().await;
    }

    pub fn reserve_root(
        self: &Arc<Self>,
        operation: OperationId,
        synthetic_terminal: SyntheticTerminal,
    ) -> Option<OperationReservation> {
        self.reserve_scope(OperationScope::root_only(operation), synthetic_terminal)
    }

    pub fn reserve_child(
        self: &Arc<Self>,
        root: OperationId,
        child: mesh_llm_runtime_event_contracts::ChildOperationId,
        synthetic_terminal: SyntheticTerminal,
    ) -> Option<OperationReservation> {
        self.reserve_scope(OperationScope::with_child(root, child), synthetic_terminal)
    }

    fn reserve_scope(
        self: &Arc<Self>,
        scope: OperationScope,
        synthetic_terminal: SyntheticTerminal,
    ) -> Option<OperationReservation> {
        let _audit = super::lock_audit::scope(super::lock_audit::Context::Reserve);
        if self.is_shutting_down() {
            return None;
        }
        match self
            .table
            .reserve_with_synthesizer(scope, synthetic_terminal)
        {
            Ok(handle) => {
                if let OperationScope::Child { root, .. } = scope {
                    self.children_by_root
                        .lock()
                        .entry(root)
                        .or_insert_with(|| Vec::with_capacity(CHILD_MULTIPLIER))
                        .push(ChildSlot {
                            index: handle.index,
                            generation: handle.generation,
                            synthetic_terminal,
                        });
                }
                Some(OperationReservation {
                    engine: Arc::clone(self),
                    scope,
                    handle,
                    synthetic_terminal,
                    cancelled: false,
                })
            }
            Err(super::reservation::ReserveError::Exhausted) => {
                self.health.bump_reservation_exhausted();
                None
            }
        }
    }

    /// The one producer boundary. Takes no lock and never blocks: every
    /// question it asks about a reservation is answered by an atomic read,
    /// and placing the fact is a lock-free ring push.
    fn submit(
        &self,
        scope: OperationScope,
        handle: Option<SlotHandle>,
        fact: RuntimeFact,
    ) -> SubmitOutcome {
        // Every lock taken from here until this function returns is
        // attributed to `Context::Producer`, which is permitted to take
        // none. The guard covers the metadata fill and the telemetry tail
        // as well, because a producer thread pays for all of it inline
        // with its real work.
        let _audit = super::lock_audit::scope(super::lock_audit::Context::Producer);
        let started_at = Instant::now();
        let fact = self.fill_ingress_metadata(fact, started_at);
        let class = fact.delivery_class();
        let telemetry = self.telemetry.get();

        let outcome = self.admit(scope, handle, fact, class);

        let elapsed = started_at.elapsed();
        if self.ingress_latency.record(elapsed) {
            self.health.bump_for_ingress_latency_milestone();
        }
        if let Some(queue) = telemetry {
            queue.record_class_outcome(class, outcome, elapsed);
        }
        if outcome == SubmitOutcome::Accepted {
            self.notify.notify_one();
        }
        outcome
    }

    /// Decide whether `fact` enters the ring, and under which outcome.
    ///
    /// Every branch here is atomic reads plus at most one lock-free push.
    /// The class bypass (`event-disabled`) is the SINGLE contract boundary
    /// for that trial mode: only `Progress`/`Diagnostic` short-circuit,
    /// before anything downstream sees the fact.
    fn admit(
        &self,
        scope: OperationScope,
        handle: Option<SlotHandle>,
        fact: RuntimeFact,
        class: DeliveryClass,
    ) -> SubmitOutcome {
        if self.is_shutting_down() {
            return SubmitOutcome::RejectedShuttingDown;
        }
        if self
            .progress_diagnostic_class_bypass
            .load(Ordering::Relaxed)
        {
            match class {
                DeliveryClass::Progress => {
                    self.health.bump_dropped_progress();
                    return SubmitOutcome::DroppedProgress;
                }
                DeliveryClass::Diagnostic => {
                    self.health.bump_dropped_diagnostic();
                    return SubmitOutcome::DroppedDiagnostic;
                }
                DeliveryClass::Terminal | DeliveryClass::StateTransition => {}
            }
        }
        match class {
            DeliveryClass::Terminal => self.admit_terminal(scope, handle, fact),
            DeliveryClass::Progress => self.admit_progress(scope, handle, fact),
            DeliveryClass::StateTransition => self.admit_state_transition(scope, handle, fact),
            DeliveryClass::Diagnostic => self.admit_diagnostic(scope, handle, fact),
        }
    }

    /// Terminals are admitted by winning the slot's write-once claim, so
    /// at most one can exist per occupied slot. That bound is what lets
    /// them skip the ring's credit budget and never be lost to pressure.
    fn admit_terminal(
        &self,
        scope: OperationScope,
        handle: Option<SlotHandle>,
        fact: RuntimeFact,
    ) -> SubmitOutcome {
        let Some(handle) = handle else {
            self.health.bump_terminal_delivery_failed();
            return SubmitOutcome::TerminalDeliveryFailed;
        };
        if self.table.claim_terminal(handle) == TerminalClaim::Refused {
            self.health.bump_terminal_delivery_failed();
            return SubmitOutcome::TerminalDeliveryFailed;
        }
        self.place(IngressItem::Fact(IngressFact {
            scope,
            fact,
            handle: Some(handle),
            reserved: true,
            synthesized: false,
            class: DeliveryClass::Terminal,
        }))
        .map_or_else(
            |PushError::Full| {
                // Unreachable by the ring's capacity arithmetic, and
                // counted rather than assumed. The claim is already spent,
                // so this operation now has no terminal.
                self.health.bump_terminal_delivery_failed();
                SubmitOutcome::TerminalDeliveryFailed
            },
            |()| SubmitOutcome::Accepted,
        )
    }

    fn admit_progress(
        &self,
        scope: OperationScope,
        handle: Option<SlotHandle>,
        fact: RuntimeFact,
    ) -> SubmitOutcome {
        let Some(handle) = handle else {
            self.health.bump_dropped_progress();
            return SubmitOutcome::DroppedProgress;
        };
        if !self.table.is_live(handle) {
            self.health.bump_dropped_progress();
            return SubmitOutcome::DroppedProgress;
        }
        self.place(IngressItem::Fact(IngressFact {
            scope,
            fact,
            handle: Some(handle),
            reserved: true,
            synthesized: false,
            class: DeliveryClass::Progress,
        }))
        .map_or_else(
            |PushError::Full| {
                self.health.bump_dropped_progress();
                SubmitOutcome::DroppedProgress
            },
            |()| SubmitOutcome::Accepted,
        )
    }

    fn admit_state_transition(
        &self,
        scope: OperationScope,
        handle: Option<SlotHandle>,
        fact: RuntimeFact,
    ) -> SubmitOutcome {
        // A ScopedIngress may outlive an explicit reservation cancellation.
        // Never let that stale handle resurrect reducer state.
        if handle.is_some_and(|handle| !self.table.is_live(handle)) {
            self.health.bump_cancelled_reservation_rejected();
            return SubmitOutcome::RejectedCancelled;
        }
        self.place(IngressItem::Fact(IngressFact {
            scope,
            fact,
            handle,
            reserved: handle.is_some(),
            synthesized: false,
            class: DeliveryClass::StateTransition,
        }))
        .map_or_else(
            |PushError::Full| {
                self.health.bump_state_transition_rejected();
                SubmitOutcome::RejectedCapacity
            },
            |()| SubmitOutcome::Accepted,
        )
    }

    fn admit_diagnostic(
        &self,
        scope: OperationScope,
        handle: Option<SlotHandle>,
        fact: RuntimeFact,
    ) -> SubmitOutcome {
        if handle.is_some_and(|handle| !self.table.is_live(handle)) {
            self.health.bump_dropped_diagnostic();
            return SubmitOutcome::DroppedDiagnostic;
        }
        self.place(IngressItem::Fact(IngressFact {
            scope,
            fact,
            handle,
            reserved: handle.is_some(),
            synthesized: false,
            class: DeliveryClass::Diagnostic,
        }))
        .map_or_else(
            |PushError::Full| {
                self.health.bump_dropped_diagnostic();
                SubmitOutcome::DroppedDiagnostic
            },
            |()| SubmitOutcome::Accepted,
        )
    }

    /// Place one item in the ring. The single call site for every push, so
    /// there is exactly one place a producer can touch the ring.
    pub(super) fn place(&self, item: IngressItem) -> Result<(), PushError> {
        self.ingress.push(item)
    }

    pub(super) fn ingress(&self) -> &Ingress {
        &self.ingress
    }

    /// The next sequence a published fact will receive, without consuming
    /// it. Pure observation, so calling it repeatedly is side-effect-free.
    #[must_use]
    pub fn peek_next_sequence(&self) -> u64 {
        self.next_sequence.load(Ordering::Acquire)
    }

    /// Assign the next publication sequence. Called only from a drain
    /// pass, which `drain_gate` serializes, so the values it hands out are
    /// both unique and issued in publication order.
    pub(super) fn next_publication_sequence(&self) -> u64 {
        self.next_sequence.fetch_add(1, Ordering::AcqRel)
    }

    /// Fill missing Rust metadata and producer timestamps at the synchronous
    /// ingress boundary. Native source identity, explicit severity, sequence
    /// evidence, and any timestamps supplied by a producer are preserved
    /// verbatim. The metadata work happens before `ingress_gate`, so producer
    /// admission is still limited to the shutdown check, sequence mint, and
    /// bounded lane insertion.
    fn fill_ingress_metadata(&self, fact: RuntimeFact, submitted_at: Instant) -> RuntimeFact {
        let mut metadata = fact.metadata().cloned().unwrap_or_else(|| FactMetadata {
            producer: ProducerSource::Rust,
            severity: inferred_rust_severity(&fact),
            ..FactMetadata::rust_defaults()
        });
        if metadata.wall_clock_unix_ns.is_none() {
            metadata.wall_clock_unix_ns = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .and_then(|duration| u64::try_from(duration.as_nanos()).ok());
        }
        if metadata.process_monotonic_time.is_none() {
            metadata.process_monotonic_time =
                Some(submitted_at.duration_since(self.process_started));
        }
        fact.with_metadata(metadata)
    }

    fn scoped_synthetic_terminal(
        &self,
        handle: SlotHandle,
        synthetic_terminal: SyntheticTerminal,
    ) -> RuntimeFact {
        let fact = synthetic_terminal();
        match self.table.scope_identities(handle) {
            Some(scope) => fact.with_scope(&scope),
            None => fact,
        }
    }

    /// A `RuntimeEventIngress` bound to `scope` with no slot. Used by an
    /// exhaustion-degraded caller (`reserve_*` returned `None`) so primary
    /// work still proceeds; a `Terminal`-class fact submitted here always
    /// reports `TerminalDeliveryFailed` because there is no slot to own it.
    #[must_use]
    pub fn unreserved_ingress(self: &Arc<Self>, scope: OperationScope) -> UnreservedIngress {
        UnreservedIngress {
            engine: Arc::clone(self),
            scope,
        }
    }

    pub(super) fn table(&self) -> &ReservationTable {
        &self.table
    }

    /// Count of currently-occupied reservation slots. Test-only: a linear
    /// scan over the table's full capacity, fine for the small capacities
    /// used in tests but never a production hot path.
    #[cfg(test)]
    #[must_use]
    pub fn occupied_count(&self) -> usize {
        (0..self.table.capacity())
            .filter(|&index| self.table.is_occupied(index).is_some())
            .count()
    }

    /// Install the drain-hold seam. Returns the installed hold, which is
    /// the same one a second call would get back: a hold is installed once
    /// per engine so a test cannot silently replace another test's.
    #[cfg(test)]
    pub(crate) fn install_drain_hold(
        &self,
        hold: Arc<super::drain_hold::DrainHold>,
    ) -> Arc<super::drain_hold::DrainHold> {
        Arc::clone(self.drain_hold.get_or_init(|| hold))
    }

    #[cfg(test)]
    pub(super) fn drain_hold(&self) -> Option<&Arc<super::drain_hold::DrainHold>> {
        self.drain_hold.get()
    }

    pub(super) fn drain_gate(&self) -> &AuditedMutex<()> {
        &self.drain_gate
    }

    pub(super) fn publication_gate(&self) -> &AuditedMutex<()> {
        &self.publication_gate
    }

    pub(super) fn set_published_frontier(&self, sequence: u64) {
        self.published_frontier
            .fetch_max(sequence, Ordering::Release);
    }

    /// Count accepted work that still has no publication/release outcome.
    /// Unsettled slots and deferred root releases are counted separately
    /// from queued facts, because a reserved operation can have both a
    /// queued fact and an unclaimed terminal.
    pub(super) fn pending_work_counts(&self) -> (usize, usize) {
        let unsettled = self.table.unsettled().len() + self.pending_root_releases.lock().len();
        // A terminal still in the ring is as undelivered as a reservation
        // that never got one: the operation has no outcome on the stream
        // either way.
        let terminal_remainder = self.ingress.queued_terminals() + unsettled;
        let total = self.ingress.len() + self.progress_lane.lock().len() + unsettled;
        (total, terminal_remainder)
    }

    /// Close admission. One release store: there is no gate to hold, so a
    /// producer that already passed its shutdown check can still land one
    /// push afterwards. That is deliberate -- shutdown drains cooperatively
    /// and reports whatever remains, and the alternative is a lock every
    /// producer would pay for on every submit to close a window the drain
    /// already tolerates.
    pub(super) fn close_admission(&self) {
        self.shutting_down.store(true, Ordering::Release);
    }
}

/// An operation-ID-bound admission guard: the only way to obtain a
/// [`ScopedIngress`]. Dropping this guard before a terminal fact was
/// submitted synthesizes one with `terminal_not_delivered`/`unknown`.
#[must_use = "dropping without submitting a terminal synthesizes terminal_not_delivered"]
pub struct OperationReservation {
    engine: Arc<RuntimeEventEngine>,
    scope: OperationScope,
    handle: SlotHandle,
    synthetic_terminal: SyntheticTerminal,
    cancelled: bool,
}

impl OperationReservation {
    #[must_use]
    pub fn scope(&self) -> OperationScope {
        self.scope
    }

    #[must_use]
    pub fn ingress(&self) -> ScopedIngress {
        ScopedIngress {
            engine: Arc::clone(&self.engine),
            scope: self.scope,
            handle: self.handle,
        }
    }

    /// Explicit pre-work cancellation: the reservation is released without
    /// a terminal, no synthesis and nothing published.
    ///
    /// The cancellation is marked here with one atomic flag flip -- which
    /// immediately stops any surviving `ScopedIngress` from admitting a
    /// fact -- and then queued through the ring for the drain to act on.
    /// Two reasons it is not done inline:
    ///
    /// * Releasing the slot and evicting reducer state need `drain_gate`
    ///   and `publication_gate`. Taking either here would make cancelling
    ///   an operation wait out a whole drain pass, on the calling
    ///   producer's thread.
    /// * Ordering. A cancel applied inline could overtake facts submitted
    ///   for this same scope moments earlier that are still in the ring,
    ///   evicting reducer state those facts would then re-create. Going
    ///   through the ring puts the cancel behind them, where it belongs.
    ///
    /// A root with at least one still-occupied child still has its release
    /// deferred exactly like a terminal-driven one
    /// (`engine::drain::release_or_defer`).
    pub fn cancel(mut self) {
        self.cancelled = true;
        // `Reserve`, not `Producer`: cancelling is the teardown counterpart
        // to `reserve_scope`, performed once per operation rather than once
        // per event. Everything it touches here is a bounded reservation-
        // table operation, never a lock a drain pass holds.
        let _audit = super::lock_audit::scope(super::lock_audit::Context::Reserve);
        if !self.engine.table().mark_cancelled(self.handle) {
            // Already cancelled, already released, or a stale generation.
            // Whoever won that race owns the release.
            return;
        }
        let Some(released) =
            drain::release_or_defer(&self.engine, self.scope, self.handle, Instant::now())
        else {
            // Deferred behind a live child. `settle_pending_root_releases`
            // owns both the release and the eviction from here.
            return;
        };
        // Eviction, unlike the release, needs the publication and reducer
        // locks a drain pass holds. Queue it.
        let queued = self.engine.place(IngressItem::Released { scope: released });
        if queued.is_ok() {
            self.engine.notify.notify_one();
        }
    }
}

impl Drop for OperationReservation {
    /// A guard dropped without a terminal synthesizes one, through the
    /// exact same write-once claim a real terminal submission uses. The CAS
    /// is the arbiter: if a producer already submitted a terminal, or
    /// shutdown already synthesized one, this claim is refused and the drop
    /// publishes nothing.
    fn drop(&mut self) {
        if self.cancelled {
            return;
        }
        // `Reserve`: synthesis happens once per operation, and reading the
        // slot's remembered identities is a payload-lock read the producer
        // path never performs.
        let _audit = super::lock_audit::scope(super::lock_audit::Context::Reserve);
        if self.engine.table().claim_terminal(self.handle) == TerminalClaim::Refused {
            return;
        }
        let synthetic = self.engine.fill_ingress_metadata(
            self.engine
                .scoped_synthetic_terminal(self.handle, self.synthetic_terminal),
            Instant::now(),
        );
        let queued = self.engine.place(IngressItem::Fact(IngressFact {
            scope: self.scope,
            fact: synthetic,
            handle: Some(self.handle),
            reserved: true,
            synthesized: true,
            class: DeliveryClass::Terminal,
        }));
        if queued.is_ok() {
            self.engine.notify.notify_one();
        } else {
            self.engine.health.bump_terminal_delivery_failed();
        }
    }
}

pub struct ScopedIngress {
    engine: Arc<RuntimeEventEngine>,
    scope: OperationScope,
    handle: SlotHandle,
}

impl ScopedIngress {
    #[must_use]
    pub fn scope(&self) -> OperationScope {
        self.scope
    }
}

impl RuntimeEventIngress for ScopedIngress {
    fn try_submit(&self, fact: RuntimeFact) -> SubmitOutcome {
        self.engine.submit(self.scope, Some(self.handle), fact)
    }
}

pub struct UnreservedIngress {
    engine: Arc<RuntimeEventEngine>,
    scope: OperationScope,
}

impl RuntimeEventIngress for UnreservedIngress {
    fn try_submit(&self, fact: RuntimeFact) -> SubmitOutcome {
        self.engine.submit(self.scope, None, fact)
    }
}
