//! The governing property: emitting an event never blocks the thread doing
//! real work.
//!
//! Two instruments prove it, and both are designed so a regression is
//! unmissable rather than a latency number someone has to argue about.
//!
//! **The drain-hold seam.** A test parks a whole drain pass inside
//! everything it holds, for 500 ms, and then submits against it. If any
//! coupling between producer and consumer comes back, a producer waits the
//! full 500 ms against a 50 ms budget -- two orders of magnitude, not
//! jitter.
//!
//! **The lock audit.** Every mutex in `runtime_events/` declares a class,
//! and a blocking acquisition in producer context panics. Because
//! essentially every test in this crate submits events, a lock
//! reintroduced on the submit path fails hundreds of tests, not just
//! these.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use mesh_llm_runtime_event_contracts::{OperationId, RuntimeEventIngress, SubmitOutcome};

use super::fixtures::{
    diagnostic_fact, progress_fact, state_transition_fact, synthetic_unknown, terminal_success,
};
use crate::runtime_events::drain_hold::DrainHold;
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::lock_audit::{
    Context, LockClass, PRODUCER_ALLOWED, RESERVE_ALLOWED, clear_thread_record, current_context,
    thread_record,
};

/// How long an installed hold parks a drain pass.
const HOLD: Duration = Duration::from_millis(500);

/// The bound a producer must meet while a pass is parked for [`HOLD`].
/// Two orders of magnitude below it, so "blocked behind the drain" and
/// "not blocked" cannot be confused for each other on a loaded machine.
const PRODUCER_BUDGET: Duration = Duration::from_millis(50);

/// Producer threads in the stall tests, spread across every class.
const PRODUCERS: usize = 32;

/// The seam itself works: an installed hold parks a drain pass for its
/// full duration, and the test side can observe the pass arriving.
///
/// Everything below depends on this, so it is asserted directly rather
/// than assumed.
#[test]
fn an_installed_hold_parks_a_drain_pass_for_its_full_duration() {
    let engine = Arc::new(RuntimeEventEngine::with_capacity(4));
    let hold = engine.install_drain_hold(Arc::new(DrainHold::new(HOLD)));

    let draining = Arc::clone(&engine);
    let started = Instant::now();
    let drain = std::thread::spawn(move || draining.drain());

    assert!(
        hold.wait_until_entered(Duration::from_secs(5)),
        "the drain pass never reached the hold point"
    );
    drain.join().expect("drain thread panicked");

    assert!(
        started.elapsed() >= hold.duration(),
        "the drain returned in {:?}, faster than the {:?} hold -- the seam \
         is not actually parking the pass",
        started.elapsed(),
        hold.duration()
    );
}

/// Every delivery class returns immediately while a drain pass is parked.
///
/// This is the regression that matters. Under the old ingress every one of
/// these waited the full hold, because the drain held the same
/// `ingress_gate` a producer needed to submit.
#[test]
fn every_class_submits_inside_the_budget_while_a_pass_is_parked() {
    let engine = Arc::new(RuntimeEventEngine::with_capacity(64));
    let hold = engine.install_drain_hold(Arc::new(DrainHold::new(HOLD)));

    // Reserve before parking the pass: a live reservation is what lets
    // progress and terminals be admitted at all.
    let reservations: Vec<_> = (0..PRODUCERS)
        .map(|_| {
            engine
                .reserve_root(OperationId::new(), synthetic_unknown)
                .expect("a 64-slot table has room for 32 roots")
        })
        .collect();

    let draining = Arc::clone(&engine);
    let drain = std::thread::spawn(move || draining.drain());
    assert!(
        hold.wait_until_entered(Duration::from_secs(5)),
        "the drain pass never reached the hold point"
    );

    let slowest = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|threads| {
        for (index, reservation) in reservations.iter().enumerate() {
            let slowest = Arc::clone(&slowest);
            threads.spawn(move || {
                let ingress = reservation.ingress();
                let fact = match index % 4 {
                    0 => terminal_success(),
                    1 => state_transition_fact(),
                    2 => progress_fact(),
                    _ => diagnostic_fact(),
                };
                let started = Instant::now();
                let _ = ingress.try_submit(fact);
                let waited = started.elapsed();
                slowest.fetch_max(
                    usize::try_from(waited.as_micros()).unwrap_or(usize::MAX),
                    Ordering::Relaxed,
                );
            });
        }
    });

    drain.join().expect("drain thread panicked");

    let slowest = Duration::from_micros(slowest.load(Ordering::Relaxed) as u64);
    assert!(
        slowest < PRODUCER_BUDGET,
        "the slowest of {PRODUCERS} producers waited {slowest:?} against a \
         {PRODUCER_BUDGET:?} budget while a pass was parked for {HOLD:?}. \
         A figure near {HOLD:?} means producer and consumer are coupled again."
    );
}

/// Reserving and cancelling are also producer-thread work, and also must
/// not wait out a parked pass. Cancelling used to take `drain_gate` and
/// then `ingress_gate`, so it waited for the entire pass.
#[test]
fn reserve_and_cancel_stay_inside_the_budget_while_a_pass_is_parked() {
    let engine = Arc::new(RuntimeEventEngine::with_capacity(16));
    let hold = engine.install_drain_hold(Arc::new(DrainHold::new(HOLD)));
    let existing = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");

    let draining = Arc::clone(&engine);
    let drain = std::thread::spawn(move || draining.drain());
    assert!(
        hold.wait_until_entered(Duration::from_secs(5)),
        "the drain pass never reached the hold point"
    );

    let started = Instant::now();
    let fresh = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let reserved_in = started.elapsed();

    let started = Instant::now();
    existing.cancel();
    let cancelled_in = started.elapsed();

    drain.join().expect("drain thread panicked");
    fresh.cancel();

    assert!(
        reserved_in < PRODUCER_BUDGET,
        "reserve_root waited {reserved_in:?} for a parked pass"
    );
    assert!(
        cancelled_in < PRODUCER_BUDGET,
        "cancel waited {cancelled_in:?} for a parked pass"
    );
}

/// The audit's own policy, asserted directly: a producer may block on
/// nothing, and a reservation may block only on the table it is reserving
/// in.
#[test]
fn the_permitted_sets_are_what_the_policy_says() {
    assert_eq!(
        PRODUCER_ALLOWED, 0,
        "a producer may not block on any lock at all"
    );
    for permitted in [
        LockClass::ReservationSlot,
        LockClass::ReservationFreeList,
        LockClass::ChildrenByRoot,
    ] {
        assert!(RESERVE_ALLOWED & (1 << (permitted as u8)) != 0);
    }
    for forbidden in [
        LockClass::DrainGate,
        LockClass::PublicationGate,
        LockClass::ReducerState,
        LockClass::ProgressLane,
        LockClass::ReplayBuffer,
    ] {
        assert!(
            RESERVE_ALLOWED & (1 << (forbidden as u8)) == 0,
            "{forbidden:?} must not be reachable from a reservation"
        );
    }
}

/// `try_submit` blocks on nothing, for every class.
///
/// The enforcing audit would already have panicked, so this asserts the
/// stronger, exact statement: the recorded set is empty, not merely legal.
#[test]
fn a_submit_records_no_blocking_acquisition_at_all() {
    let recorded = std::thread::spawn(|| {
        let engine = RuntimeEventEngine::with_capacity(4);
        let reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        let ingress = reservation.ingress();
        // Warm anything one-time before measuring.
        let _ = ingress.try_submit(state_transition_fact());

        clear_thread_record();
        for fact in [
            state_transition_fact(),
            progress_fact(),
            diagnostic_fact(),
            terminal_success(),
        ] {
            let _ = ingress.try_submit(fact);
        }
        let taken = thread_record();
        drop(reservation);
        taken
    })
    .join()
    .expect("recording thread panicked");

    assert!(
        recorded.is_empty(),
        "submitting took {recorded:?}; the producer path must reach ingress \
         through atomics only"
    );
}

/// Reserving blocks only on what [`RESERVE_ALLOWED`] permits. The
/// process-global admission gate it used to take is gone.
#[test]
fn a_reservation_blocks_only_on_the_table_it_reserves_in() {
    let recorded = std::thread::spawn(|| {
        let engine = Arc::new(RuntimeEventEngine::with_capacity(4));
        // Warm the free list and slot mutexes.
        engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve")
            .cancel();

        clear_thread_record();
        let reservation = engine.reserve_root(OperationId::new(), synthetic_unknown);
        let taken = thread_record();
        assert!(reservation.is_some(), "a fresh table has capacity");
        reservation.expect("checked above").cancel();
        taken
    })
    .join()
    .expect("recording thread panicked");

    for class in &recorded {
        assert!(
            RESERVE_ALLOWED & (1 << (*class as u8)) != 0,
            "reserving blocked on {class:?}, which RESERVE_ALLOWED forbids; \
             recorded {recorded:?}"
        );
    }
}

/// The audit attributes locks to the right context and restores it:
/// neither scope outlives the call it marks.
#[test]
fn context_does_not_outlive_the_call_it_marks() {
    let engine = RuntimeEventEngine::with_capacity(4);
    assert_eq!(current_context(), Context::Unmarked);

    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("a fresh table has capacity");
    assert_eq!(
        current_context(),
        Context::Unmarked,
        "the Reserve scope must not outlive reserve_scope"
    );

    let outcome = reservation.ingress().try_submit(state_transition_fact());
    assert_eq!(outcome, SubmitOutcome::Accepted);
    assert_eq!(
        current_context(),
        Context::Unmarked,
        "the Producer scope must not outlive submit"
    );

    reservation.cancel();
    assert_eq!(current_context(), Context::Unmarked);
}

/// The producer-visible admission gate is gone from the source, not just
/// unused. A field that still existed could be reached again.
#[test]
fn no_ingress_gate_remains_in_the_source() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/runtime_events");
    let mut offenders = Vec::new();
    visit(&root, &mut |path: &std::path::Path, source: &str| {
        if is_test_source(path) {
            // This file names the gate to assert its absence.
            return;
        }
        for (index, line) in source.lines().enumerate() {
            // Prose may name the gate to explain why it is gone; code may
            // not name it at all.
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            if line.contains("ingress_gate") {
                offenders.push(format!("{}:{}", path.display(), index + 1));
            }
        }
    });
    assert!(
        offenders.is_empty(),
        "ingress_gate is back at:\n{}",
        offenders.join("\n")
    );
}

/// Every mutex `runtime_events/` owns on a production path must be an
/// `AuditedMutex`, so adding a lock forces picking a class for it and the
/// enforcing mode cannot be sidestepped by declaring a bare `Mutex`.
///
/// Scoped to production source: `lock_audit.rs` wraps the real mutex, and
/// `drain_hold.rs` plus the test modules are `#[cfg(test)]` and never on a
/// producer path.
#[test]
fn no_production_lock_in_runtime_events_escapes_the_audit() {
    const EXEMPT: [&str; 2] = ["lock_audit.rs", "drain_hold.rs"];

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/runtime_events");
    let mut offenders = Vec::new();
    visit(&root, &mut |path: &std::path::Path, source: &str| {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if EXEMPT.contains(&name) || is_test_source(path) {
            return;
        }
        for (index, line) in source.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            // A field declaration, not a use/type/expression.
            let is_field = trimmed.ends_with(',') && trimmed.contains(": Mutex<");
            let is_construction =
                trimmed.contains("Mutex::new(") && !trimmed.contains("AuditedMutex::new(");
            if is_field || is_construction {
                offenders.push(format!(
                    "{}:{}: {}",
                    path.display(),
                    index + 1,
                    trimmed.trim_end()
                ));
            }
        }
    });

    assert!(
        offenders.is_empty(),
        "these production locks are not AuditedMutex:\n{}",
        offenders.join("\n")
    );
}

/// Whether `path` is a `#[cfg(test)]` module rather than production
/// source. Test code may legitimately name what it is asserting about.
fn is_test_source(path: &std::path::Path) -> bool {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    name == "tests.rs"
        || name.starts_with("tests_")
        || path.components().any(|part| part.as_os_str() == "tests")
}

fn visit(dir: &std::path::Path, each: &mut dyn FnMut(&std::path::Path, &str)) {
    let entries = std::fs::read_dir(dir).expect("runtime_events source tree is readable");
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            visit(&path, each);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let source = std::fs::read_to_string(&path).expect("source file is readable");
            each(&path, &source);
        }
    }
}
