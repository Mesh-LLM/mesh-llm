//! Lock-class audit for the producer boundary.
//!
//! The governing rule for this subsystem is that emitting an event never
//! blocks the thread that is doing real work. A lock is the only way a
//! producer can be made to wait, so this module makes every mutex in
//! `runtime_events/` declare which class it belongs to, and makes every
//! acquisition record the calling context it happened under.
//!
//! Two things fall out of that:
//!
//! * [`AuditedMutex`] replaces bare `std::sync::Mutex` for every lock this
//!   module owns. A source-shape test asserts no bare `Mutex<` field
//!   declaration survives in `runtime_events/`, so a future lock cannot be
//!   added without picking a class for it.
//! * [`scope`] marks a thread as running in [`Context::Producer`],
//!   [`Context::Reserve`], or [`Context::Drain`] for the duration of a
//!   guard. Every acquisition is OR-ed into a per-thread bitset that a
//!   test can read back with [`thread_record`].
//!
//! This commit runs the audit in **record-only** mode: it observes and
//! never fails. [`thread_record`] reads the observation back so a test can
//! pin what the producer path locks today, which is the baseline the
//! ingress rework has to move. A later commit turns [`note`] into a panic
//! when a producer takes anything outside [`PRODUCER_ALLOWED`].
//!
//! Everything here compiles away outside `debug_assertions`: the release
//! build of [`AuditedMutex`] is a `#[repr(transparent)]` newtype whose
//! `lock` is the inner `lock`, and [`scope`] returns an empty guard.
//!
//! Nothing in the recording path allocates or locks: both the context and
//! the record are const-initialized thread-local `Cell`s, so reading and
//! writing them is a direct TLS slot access with no lazy-initialization
//! branch. `tests/ingress_reservoir_no_alloc.rs` measures allocation calls
//! on the submit path with this audit compiled in, so a recording step
//! that allocated would show up there as a failure.

use std::sync::{Mutex, MutexGuard, PoisonError, TryLockResult};

/// Every mutex `runtime_events/` owns, named by what it protects.
///
/// The discriminants are stable bit positions in the per-thread record;
/// adding a class means adding a bit, never renumbering an existing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum LockClass {
    /// The process-global producer admission gate.
    IngressGate = 0,
    /// Serializes drain passes and rebuilds.
    DrainGate = 1,
    /// Publication boundary for replay, subscribers, and reducer state.
    PublicationGate = 2,
    /// One reservation-table slot.
    ReservationSlot = 3,
    /// The reservation table's free-index stack.
    ReservationFreeList = 4,
    /// Child slots indexed by root operation.
    ChildrenByRoot = 5,
    /// Roots whose release is deferred behind a live child.
    PendingRootReleases = 6,
    /// The reducer's published snapshot.
    ReducerState = 7,
    /// The last progress-flush instant.
    ProgressLastFlush = 8,
    /// The state-transition lane's key order.
    StateLaneEntries = 9,
    /// The state-transition lane's latest-value map.
    StateLaneLatest = 10,
    /// The diagnostic FIFO.
    DiagnosticLane = 11,
    /// The wake list and its ingress-sequence counter.
    WakeList = 12,
    /// The bounded replay buffer.
    ReplayBuffer = 13,
    /// One subscriber's bounded queue.
    SubscriberQueue = 14,
    /// The subscriber registry map.
    SubscriberRegistry = 15,
    /// The telemetry sample ring.
    TelemetrySamples = 16,
    /// The presentation subscriber's coalescing map.
    PresentationCoalescer = 17,
    /// The presentation subscriber's last-flush instant.
    PresentationLastFlush = 18,
}

impl LockClass {
    const fn bit(self) -> u32 {
        1 << (self as u8)
    }

    /// Every class, for a test that wants to name what it observed.
    #[cfg(test)]
    pub(crate) const ALL: [LockClass; 19] = [
        LockClass::IngressGate,
        LockClass::DrainGate,
        LockClass::PublicationGate,
        LockClass::ReservationSlot,
        LockClass::ReservationFreeList,
        LockClass::ChildrenByRoot,
        LockClass::PendingRootReleases,
        LockClass::ReducerState,
        LockClass::ProgressLastFlush,
        LockClass::StateLaneEntries,
        LockClass::StateLaneLatest,
        LockClass::DiagnosticLane,
        LockClass::WakeList,
        LockClass::ReplayBuffer,
        LockClass::SubscriberQueue,
        LockClass::SubscriberRegistry,
        LockClass::TelemetrySamples,
        LockClass::PresentationCoalescer,
        LockClass::PresentationLastFlush,
    ];
}

/// What the current thread is doing while it takes a lock.
///
/// `Producer` is the one that matters: it covers
/// `RuntimeEventEngine::submit` from entry to return, which is the call a
/// decode loop or a request task makes inline with its real work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Context {
    /// Not inside any marked runtime-event operation.
    Unmarked = 0,
    /// Inside `RuntimeEventEngine::submit`.
    Producer = 1,
    /// Inside `RuntimeEventEngine::reserve_scope`.
    Reserve = 2,
    /// Inside a drain pass.
    Drain = 3,
}

/// Classes a reservation is permitted to take once the audit is enforcing.
///
/// Reserving is a setup step a producer performs once per operation, not
/// per event, so it is allowed to touch the table it is reserving in. It
/// is still never allowed to touch a lane, the wake list, or a gate.
#[cfg(test)]
pub(crate) const RESERVE_ALLOWED: u32 = LockClass::ReservationSlot.bit()
    | LockClass::ReservationFreeList.bit()
    | LockClass::ChildrenByRoot.bit();

/// Classes a producer is permitted to take once the audit is enforcing:
/// none. `try_submit` must reach the ingress structure through atomics.
#[cfg(test)]
pub(crate) const PRODUCER_ALLOWED: u32 = 0;

#[cfg(debug_assertions)]
mod recording {
    use std::cell::Cell;

    use super::{Context, LockClass};

    thread_local! {
        /// Const-initialized so reading either slot is a direct TLS access
        /// with no lazy-initialization branch and no allocation.
        pub(super) static CONTEXT: Cell<u8> = const { Cell::new(Context::Unmarked as u8) };
        /// Bitset of classes this thread has acquired since the last
        /// [`super::clear_thread_record`]. Per-thread, not global, so a
        /// test that drives its producer on a dedicated thread reads an
        /// exact set regardless of what the rest of the suite is doing in
        /// parallel.
        pub(super) static TAKEN: Cell<u32> = const { Cell::new(0) };
    }

    pub(super) fn note(class: LockClass) {
        TAKEN.with(|taken| taken.set(taken.get() | class.bit()));
    }

    #[cfg(test)]
    pub(super) fn context() -> Context {
        match CONTEXT.with(Cell::get) {
            1 => Context::Producer,
            2 => Context::Reserve,
            3 => Context::Drain,
            _ => Context::Unmarked,
        }
    }

    #[cfg(test)]
    pub(super) fn taken() -> u32 {
        TAKEN.with(Cell::get)
    }

    #[cfg(test)]
    pub(super) fn clear() {
        TAKEN.with(|taken| taken.set(0));
    }
}

/// Record that `class` was acquired in the current context.
#[inline(always)]
pub(crate) fn note(class: LockClass) {
    #[cfg(debug_assertions)]
    recording::note(class);
    #[cfg(not(debug_assertions))]
    let _ = class;
}

/// The context the calling thread is currently running in.
#[cfg(test)]
#[must_use]
pub(crate) fn current_context() -> Context {
    #[cfg(debug_assertions)]
    {
        recording::context()
    }
    #[cfg(not(debug_assertions))]
    {
        Context::Unmarked
    }
}

/// Restores the previous context when dropped.
pub(crate) struct ContextGuard {
    #[cfg(debug_assertions)]
    previous: u8,
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        recording::CONTEXT.with(|cell| cell.set(self.previous));
    }
}

/// Mark the current thread as running in `context` until the returned
/// guard drops. Nesting restores the outer context, so a `Reserve` inside
/// a `Producer` call reports as `Reserve` for its own duration only.
#[inline(always)]
#[must_use]
pub(crate) fn scope(context: Context) -> ContextGuard {
    #[cfg(debug_assertions)]
    {
        let previous = recording::CONTEXT.with(|cell| cell.replace(context as u8));
        ContextGuard { previous }
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = context;
        ContextGuard {}
    }
}

/// Classes the calling thread has acquired since its last
/// [`clear_thread_record`], in declaration order. Debug builds only;
/// returns an empty set otherwise.
#[cfg(test)]
#[must_use]
pub(crate) fn thread_record() -> Vec<LockClass> {
    #[cfg(debug_assertions)]
    {
        let bits = recording::taken();
        LockClass::ALL
            .into_iter()
            .filter(|class| bits & class.bit() != 0)
            .collect()
    }
    #[cfg(not(debug_assertions))]
    {
        Vec::new()
    }
}

/// Forget every class the calling thread has acquired so far, so a test
/// can scope a measurement to exactly the calls it is about to make.
#[cfg(test)]
pub(crate) fn clear_thread_record() {
    #[cfg(debug_assertions)]
    recording::clear();
}

/// A `std::sync::Mutex` that records its class on every acquisition.
///
/// Poisoning is handled the same way every call site in this module
/// already handled it: the guard is taken regardless, because none of
/// these locks protect an invariant that a panicking holder could have
/// half-broken in a way a later reader must not see.
#[derive(Debug)]
pub(crate) struct AuditedMutex<T> {
    class: LockClass,
    inner: Mutex<T>,
}

impl<T> AuditedMutex<T> {
    pub(crate) const fn new(class: LockClass, value: T) -> Self {
        Self {
            class,
            inner: Mutex::new(value),
        }
    }

    /// Acquire, recording the class against the calling context first so
    /// an acquisition that then blocks forever is still observed.
    #[inline]
    pub(crate) fn lock(&self) -> MutexGuard<'_, T> {
        note(self.class);
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Non-blocking acquire, passed straight through to the inner mutex so
    /// callers keep std's poison handling. Still recorded: a `try_lock`
    /// that succeeds held the lock, and one that fails proves contention.
    #[inline]
    pub(crate) fn try_lock(&self) -> TryLockResult<MutexGuard<'_, T>> {
        note(self.class);
        self.inner.try_lock()
    }
}
