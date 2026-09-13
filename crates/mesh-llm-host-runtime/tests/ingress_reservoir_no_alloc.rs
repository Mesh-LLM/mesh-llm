//! Task 13 (`.omo/plans/event-system-fixes.md`, "Must NOT: allocate on the
//! submit path") -- proves `RuntimeEventEngine::submit`'s ingress-latency
//! recording is allocation-free, via a thread-local counting
//! `#[global_allocator]`. Safe to install here ONLY because integration
//! tests under `tests/*.rs` each compile as their OWN separate binary
//! crate: this allocator applies to this one binary alone, never to the
//! crate's `--lib` unit-test binary or any other integration test.
//!
//! ## Task 13 alloc-fix (post-review strengthening)
//!
//! An adversarial review of the original version of this file found that
//! it asserted only `NET_ALLOCS` -- outstanding allocations, incremented
//! by `alloc`/`realloc` and decremented by `dealloc` -- returning to its
//! starting value. That is a strictly WEAKER property than "no heap
//! allocation": a transient allocation that is both allocated AND freed
//! inside the very same measured `try_submit` call (a stray `format!()`,
//! `to_string()`, or scratch `Vec`) nets to zero and sails straight
//! through a net-only assertion. The review proved this empirically by
//! planting exactly such a `format!()` on the submit path and observing
//! every test in this file stay green. Re-run that mutation against this
//! file to reproduce it: the net-only assertion still passes, the
//! `TOTAL_ALLOC_CALLS` assertions below do not.
//!
//! `TOTAL_ALLOC_CALLS` below fixes this: a monotonic counter incremented
//! on every `alloc`/`realloc` call and NEVER decremented, so a
//! same-window allocate-then-free still moves it even though it nets to
//! zero on `NET_ALLOCS`. It is the counter that actually discharges "no
//! heap allocation happened" for a measured window. `NET_ALLOCS` is kept
//! alongside it -- it is still a useful, independent signal (a real
//! *leak* across many iterations shows up as steady growth there, which
//! is not quite what a same-window total-calls count highlights) -- but
//! every test below now asserts BOTH. `alloc_zeroed` needs no separate
//! counting: `GlobalAlloc`'s default `alloc_zeroed` implementation calls
//! back into this allocator's own (counted) `alloc`, so any zeroed
//! allocation is already covered by the `alloc` override below without
//! duplicating that logic.
//!
//! ## Delivery-class coverage
//!
//! The original file only ever exercised the `Terminal` delivery class,
//! and its many-call measured loops only ever hit the DUPLICATE-REJECTION
//! path (`SubmitOutcome::TerminalDeliveryFailed`) -- the one successful,
//! `Accepted` write on each reservation was the excluded warm-up call.
//! This file now additionally proves a genuinely DELIVERED (never
//! rejected, never dropped) submission is allocation-free for all four
//! `DeliveryClass` values: `Terminal`, `StateTransition`, `Progress`, and
//! `Diagnostic`. Each new test documents exactly how its measured window
//! is scoped, in its own doc comment.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use mesh_llm_host_runtime::runtime_events::engine::RuntimeEventEngine;
use mesh_llm_runtime_event_contracts::{
    DiagnosticEventKind, FamilyFact, ModelLoadingEventKind, NativeRuntimeEventKind, OperationId,
    OperationScope, RuntimeEventIngress, RuntimeFact, SubmitOutcome,
};

thread_local! {
    /// NET outstanding allocations on this thread: `alloc`/`realloc`
    /// increment, `dealloc` decrements. Returning to its starting value
    /// proves nothing LEAKED across a measured window -- it says nothing
    /// about a transient allocate-then-free WITHIN that window. Kept
    /// alongside `TOTAL_ALLOC_CALLS` -- see the module doc comment.
    static NET_ALLOCS: Cell<i64> = const { Cell::new(0) };
    /// TOTAL allocation CALLS on this thread: incremented by `alloc` and
    /// `realloc`, NEVER decremented. This is the counter that actually
    /// proves "no heap allocation happened" for a measured window -- a
    /// same-window allocate-then-free still moves it even though it nets
    /// to zero on `NET_ALLOCS`. See the module doc comment.
    static TOTAL_ALLOC_CALLS: Cell<u64> = const { Cell::new(0) };
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        NET_ALLOCS.with(|count| count.set(count.get() + 1));
        TOTAL_ALLOC_CALLS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        NET_ALLOCS.with(|count| count.set(count.get() - 1));
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A realloc that moves counts as one alloc; this file cares about
        // BOTH whether the net count of outstanding allocations moves
        // during the measured window (a moved reallocation) AND whether
        // any allocation call happened at all (the total-calls counter),
        // so counting it toward both is correct for either measurement.
        NET_ALLOCS.with(|count| count.set(count.get() + 1));
        TOTAL_ALLOC_CALLS.with(|count| count.set(count.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    // `alloc_zeroed` is intentionally NOT overridden: `GlobalAlloc`'s
    // default implementation calls back into `Self::alloc` (above) and
    // then zeroes the result, so it is already counted on both counters
    // without duplicating that logic here.
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn terminal_fact() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped))
}

/// A `StateTransition`-class fact from the SAME family as `terminal_fact`
/// (`NativeRuntimeEventKind`), cross-checked against
/// `crates/mesh-llm-runtime-event-contracts/src/delivery/lifecycle.rs`.
fn state_transition_fact() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeInitialized))
}

/// A `Progress`-class fact (`delivery/lifecycle.rs`).
fn progress_fact() -> RuntimeFact {
    RuntimeFact::ModelLoading(FamilyFact::new(ModelLoadingEventKind::ModelLoadProgress))
}

/// A `Diagnostic`-class fact (`delivery/execution.rs`).
fn diagnostic_fact() -> RuntimeFact {
    RuntimeFact::Diagnostic(FamilyFact::new(DiagnosticEventKind::WarningRaised))
}

fn net_allocs() -> i64 {
    NET_ALLOCS.with(Cell::get)
}

fn total_alloc_calls() -> u64 {
    TOTAL_ALLOC_CALLS.with(Cell::get)
}

/// Drives ONE `try_submit(fact)` call and asserts it (a) reports the
/// expected DELIVERED outcome -- never a rejection or a drop, so this
/// helper actually measures a success path, not a fast-reject shortcut --
/// and (b) costs ZERO total allocation calls. The counting window opens
/// at `before` and closes at `after`, covering ONLY this one call: every
/// caller below builds its engine, its reservation, and any lane warm-up
/// strictly BEFORE calling this helper, so nothing outside `try_submit`
/// itself can be mistaken for what it costs.
fn assert_try_submit_is_alloc_free(
    ingress: &dyn RuntimeEventIngress,
    fact: RuntimeFact,
    expected_outcome: SubmitOutcome,
    class_name: &str,
) {
    let before = total_alloc_calls();
    let outcome = ingress.try_submit(fact);
    let after = total_alloc_calls();

    assert_eq!(
        outcome, expected_outcome,
        "{class_name}: expected a delivered ({expected_outcome:?}) outcome, got \
         {outcome:?} -- a wrong outcome here would mean this test isn't \
         measuring the success path it claims to"
    );
    assert_eq!(
        after, before,
        "{class_name}: a delivered ({expected_outcome:?}) submission must perform \
         zero TOTAL heap allocation calls, not merely net-zero outstanding \
         (before={before}, after={after})"
    );
}

/// Submits `count` state transitions before a measured window opens.
///
/// This used to warm a producer-side lane's backing collections, which no
/// longer exist -- the ring is allocated once at construction, so there is
/// nothing left to grow. It is kept because the tests below are
/// deliberately about the WARMED steady state, and
/// `the_first_submission_on_a_fresh_engine_allocates_nothing` covers the
/// cold path separately. A fresh `OperationId` per call keeps every
/// submission a distinct key, so none of them coalesce away.
fn warm_up_state_transition_lane(engine: &Arc<RuntimeEventEngine>, count: usize) {
    for _ in 0..count {
        let scope = OperationScope::root_only(OperationId::new());
        let outcome = engine
            .unreserved_ingress(scope)
            .try_submit(state_transition_fact());
        assert_eq!(
            outcome,
            SubmitOutcome::Accepted,
            "warm-up state-transition submission (fresh scope) must itself be accepted"
        );
    }
}

/// Submits `count` diagnostics before a measured window opens, for the
/// same reason as `warm_up_state_transition_lane`. Diagnostics never
/// coalesce by key, so this reuses one scope.
fn warm_up_diagnostic_lane(engine: &Arc<RuntimeEventEngine>, count: usize) {
    let scope = OperationScope::root_only(OperationId::new());
    for _ in 0..count {
        let outcome = engine
            .unreserved_ingress(scope)
            .try_submit(diagnostic_fact());
        assert_eq!(
            outcome,
            SubmitOutcome::Accepted,
            "warm-up diagnostic submission must itself be accepted (diagnostics never coalesce)"
        );
    }
}

#[test]
fn submit_records_ingress_latency_with_no_heap_allocation() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    let ingress = reservation.ingress();

    // Warm up: the FIRST `try_submit` on this handle is the only one that
    // can succeed, because the terminal claim is write-once. Excluding it
    // isolates the reservoir write from anything a first call might do.
    // The cold path is covered directly by
    // `the_first_submission_on_a_fresh_engine_allocates_nothing`.
    let _ = ingress.try_submit(terminal_fact());

    let before_net = net_allocs();
    let before_total = total_alloc_calls();
    // Every call after the first is a refused duplicate terminal
    // (`TerminalDeliveryFailed`): it still runs the full `submit` body,
    // including the reservoir's `record` call, but its claim CAS loses, so
    // nothing reaches the ring.
    for _ in 0..1_000 {
        let _ = ingress.try_submit(terminal_fact());
    }
    let after_net = net_allocs();
    let after_total = total_alloc_calls();

    assert_eq!(
        after_net, before_net,
        "submit's ingress-latency recording must add no net heap allocation \
         across 1,000 calls (before={before_net}, after={after_net})"
    );
    // The TOTAL-calls assertion (task 13 alloc-fix): the strictly
    // stronger property. A net-only assertion cannot distinguish "zero
    // allocation" from "an equal number of same-window allocate/free
    // pairs"; this one can, because it never decrements.
    assert_eq!(
        after_total, before_total,
        "submit's ingress-latency recording must perform zero TOTAL heap \
         allocation calls across 1,000 calls, not merely net-zero outstanding \
         (before={before_total}, after={after_total})"
    );
}

#[test]
fn submit_crosses_the_reservoir_milestone_with_no_heap_allocation() {
    // A second, independent proof at a scale that actually exercises the
    // 100-sample health-version-bump milestone (`IngressLatencyReservoir::record`'s
    // return value), not just the reservoir's plain ring write.
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    let ingress = reservation.ingress();
    let _ = ingress.try_submit(terminal_fact());

    let before_net = net_allocs();
    let before_total = total_alloc_calls();
    for _ in 0..250 {
        let _ = ingress.try_submit(terminal_fact());
    }
    let after_net = net_allocs();
    let after_total = total_alloc_calls();

    assert_eq!(
        after_net, before_net,
        "crossing the 100-sample milestone (twice, over 250 calls) must still add \
         no net heap allocation (before={before_net}, after={after_net})"
    );
    assert_eq!(
        after_total, before_total,
        "crossing the 100-sample milestone (twice, over 250 calls) must still \
         perform zero TOTAL heap allocation calls, not merely net-zero \
         outstanding (before={before_total}, after={after_total})"
    );
}

/// Delivery-class coverage: `Terminal`. Unlike the two tests above (which
/// measure only REFUSED duplicate submissions after their one excluded
/// warm-up write), this measures a genuinely fresh, `Accepted` terminal --
/// a second reservation's first-and-only submission.
#[test]
fn submit_delivers_an_accepted_terminal_fact_with_zero_allocation_calls() {
    let engine = RuntimeEventEngine::with_capacity(4);
    // Warm-up (excluded from the window): a throwaway reservation's own
    // terminal, so this test measures the warmed steady state. The cold
    // path has its own test above.
    let warm_up = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    let warm_outcome = warm_up.ingress().try_submit(terminal_fact());
    assert_eq!(warm_outcome, SubmitOutcome::Accepted);

    // The MEASURED reservation: a fresh scope whose terminal has never
    // been written, so `try_submit` below is a genuinely delivered
    // (`Accepted`) submission, not a rejected duplicate.
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    assert_try_submit_is_alloc_free(
        &reservation.ingress(),
        terminal_fact(),
        SubmitOutcome::Accepted,
        "Terminal",
    );
}

/// Task 13 alloc-fix, delivery-class coverage: `StateTransition`. The
/// measured call is the lane's 201st distinct `(scope, kind)` key
/// (`warm_up_state_transition_lane` above installs 200 first), so it is
/// `Accepted` -- never `Coalesced` -- inside the measured window.
#[test]
fn submit_delivers_an_accepted_state_transition_fact_with_zero_allocation_calls() {
    let engine = RuntimeEventEngine::with_capacity(4);
    warm_up_state_transition_lane(&engine, 200);

    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    assert_try_submit_is_alloc_free(
        &reservation.ingress(),
        state_transition_fact(),
        SubmitOutcome::Accepted,
        "StateTransition",
    );
}

/// Task 13 alloc-fix, delivery-class coverage: `Progress`. `Coalesced` is
/// deliberately the expected outcome here, not `Accepted`:
/// `lanes::submit_progress` can only ever return `Coalesced` (a live
/// handle) or `DroppedProgress` (no handle) -- `SubmitOutcome::Accepted`
/// is not a reachable outcome for a `Progress`-class fact at all -- so
/// `Coalesced` here IS progress's own delivered, never-dropped success
/// case, not a weaker stand-in for it.
///
/// Progress no longer coalesces at the boundary -- the latest-value
/// decision belongs to the drain now -- so the measured call is an
/// ordinary accepted ring push, the same shape as every other class. The
/// one warm-up submission remains because the first-ever call on a fresh
/// engine can still pay a one-time cost that the steady state does not.
#[test]
fn submit_delivers_an_accepted_progress_fact_with_zero_allocation_calls() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let warm_up = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    let warm_outcome = warm_up.ingress().try_submit(progress_fact());
    assert_eq!(warm_outcome, SubmitOutcome::Accepted);

    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    assert_try_submit_is_alloc_free(
        &reservation.ingress(),
        progress_fact(),
        SubmitOutcome::Accepted,
        "Progress",
    );
}

/// Task 13 alloc-fix, delivery-class coverage: `Diagnostic`. The measured
/// call is the lane's 201st submission (`warm_up_diagnostic_lane` above
/// installs 200 first); diagnostics never coalesce, so it is `Accepted`
/// regardless of key reuse.
#[test]
fn submit_delivers_an_accepted_diagnostic_fact_with_zero_allocation_calls() {
    let engine = RuntimeEventEngine::with_capacity(4);
    warm_up_diagnostic_lane(&engine, 200);

    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    assert_try_submit_is_alloc_free(
        &reservation.ingress(),
        diagnostic_fact(),
        SubmitOutcome::Accepted,
        "Diagnostic",
    );
}

/// COLD: the very first submission on a freshly built engine allocates
/// nothing.
///
/// Every other test in this file warms something first, which is honest
/// about what it measures but leaves the first-use path unmeasured -- and
/// the first use is exactly where a lazily-initialized container or a
/// mutex's first-ever acquisition would show up. There is nothing to warm
/// now: the ring is allocated at construction and the submit path takes no
/// lock, so the first call has to cost what the ten-thousandth does.
#[test]
fn the_first_submission_on_a_fresh_engine_allocates_nothing() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("reserve");
    let ingress = reservation.ingress();

    // No warm-up of any kind between here and the measured window.
    let before = total_alloc_calls();
    let outcome = ingress.try_submit(state_transition_fact());
    let after = total_alloc_calls();

    assert_eq!(outcome, SubmitOutcome::Accepted);
    assert_eq!(
        after, before,
        "the FIRST submission on a fresh engine must allocate nothing \
         (before={before}, after={after}); a cost that only the first call \
         pays is still a cost a request pays"
    );
}

/// COLD, every class: the first submission of each delivery class on its
/// own fresh engine allocates nothing.
///
/// One engine per class, so no class can be warmed by another's traffic.
#[test]
fn the_first_submission_of_every_class_allocates_nothing() {
    for (class_name, fact) in [
        ("Terminal", terminal_fact as fn() -> RuntimeFact),
        ("StateTransition", state_transition_fact),
        ("Progress", progress_fact),
        ("Diagnostic", diagnostic_fact),
    ] {
        let engine = RuntimeEventEngine::with_capacity(4);
        let reservation = engine
            .reserve_root(OperationId::new(), terminal_fact)
            .expect("reserve");
        let ingress = reservation.ingress();

        let before = total_alloc_calls();
        let outcome = ingress.try_submit(fact());
        let after = total_alloc_calls();

        assert_eq!(
            outcome,
            SubmitOutcome::Accepted,
            "{class_name}: expected a delivered outcome on a fresh engine"
        );
        assert_eq!(
            after, before,
            "{class_name}: the first submission on a fresh engine allocated \
             (before={before}, after={after})"
        );
    }
}

/// CONTENDED: eight producer threads submitting against a live drain each
/// allocate nothing, across a thousand submissions apiece.
///
/// The counters are thread-local, so each producer measures only its own
/// window -- the drain thread's allocations, which are real and expected,
/// cannot be mistaken for a producer's. This is the shape the warmed
/// single-threaded tests above deliberately do not cover: a submit racing
/// a consumer that is actively popping, routing, reducing, and publishing.
#[test]
fn contended_submissions_against_a_live_drain_allocate_nothing() {
    const PRODUCERS: usize = 8;
    const PER_PRODUCER: usize = 1_000;

    let engine = Arc::new(RuntimeEventEngine::new());
    let draining = Arc::clone(&engine);
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let drain_stop = Arc::clone(&stop);

    // A real consumer, running the whole time: popping, routing, reducing,
    // publishing, and returning ring credits.
    let drain = std::thread::spawn(move || {
        while !drain_stop.load(std::sync::atomic::Ordering::Relaxed) {
            draining.drain();
            std::thread::yield_now();
        }
        draining.drain();
    });

    let accepted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    std::thread::scope(|threads| {
        for _ in 0..PRODUCERS {
            let engine = Arc::clone(&engine);
            let accepted = Arc::clone(&accepted);
            threads.spawn(move || {
                let reservation = engine
                    .reserve_root(OperationId::new(), terminal_fact)
                    .expect("a full-capacity table has room for 8 roots");
                let ingress = reservation.ingress();
                // One warm-up submission: this thread's first touch of its
                // own thread-locals is not what this test is about, and
                // `the_first_submission_on_a_fresh_engine_allocates_nothing`
                // covers the cold path directly.
                let _ = ingress.try_submit(state_transition_fact());

                let before = total_alloc_calls();
                let mut delivered = 0usize;
                for index in 0..PER_PRODUCER {
                    let fact = match index % 3 {
                        0 => state_transition_fact(),
                        1 => progress_fact(),
                        _ => diagnostic_fact(),
                    };
                    if ingress.try_submit(fact) == SubmitOutcome::Accepted {
                        delivered += 1;
                    }
                }
                let after = total_alloc_calls();
                accepted.fetch_add(delivered, std::sync::atomic::Ordering::Relaxed);

                assert_eq!(
                    after,
                    before,
                    "a producer racing a live drain allocated {} times across \
                     {PER_PRODUCER} submissions",
                    after - before
                );
                reservation.cancel();
            });
        }
    });

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    drain.join().expect("drain thread panicked");

    let accepted = accepted.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        accepted > PRODUCERS * PER_PRODUCER / 2,
        "only {accepted} of {} submissions were delivered; a test that mostly \
         measured the refusal path would not be measuring the submit path",
        PRODUCERS * PER_PRODUCER
    );
}
