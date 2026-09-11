//! Baseline for the producer boundary: what the drain holds, and what a
//! producer has to wait for to get past it.
//!
//! These tests do not assert the target behavior. They pin the behavior
//! that exists today so the ingress rework has a measured starting point
//! and so a regression back to it is caught rather than argued about. Each
//! one names, in its own doc comment, the assertion that replaces it once
//! ingress no longer has a producer-visible gate.

use std::sync::Arc;
use std::time::{Duration, Instant};

use mesh_llm_runtime_event_contracts::{
    OperationId, OperationScope, RuntimeEventIngress, SubmitOutcome,
};

use super::fixtures::{diagnostic_fact, state_transition_fact};
use crate::runtime_events::drain_hold::DrainHold;
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::lock_audit::{
    Context, LockClass, PRODUCER_ALLOWED, RESERVE_ALLOWED, clear_thread_record, current_context,
    thread_record,
};

/// How long an installed hold parks a drain pass. Long enough that a
/// producer blocked behind it is unmistakable against scheduling noise,
/// short enough to keep the suite fast.
const HOLD: Duration = Duration::from_millis(500);

/// The bound a producer must meet. Two orders of magnitude below [`HOLD`],
/// so "blocked behind the drain" and "not blocked" cannot be confused for
/// each other on a loaded machine.
const PRODUCER_BUDGET: Duration = Duration::from_millis(50);

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

/// BASELINE, not the target: a producer submitting while a drain pass is
/// parked waits for that pass to finish.
///
/// This is the defect the ingress rework exists to remove. Once producers
/// reach ingress through atomics instead of `ingress_gate`, this test is
/// replaced by its inverse -- every producer returns inside
/// [`PRODUCER_BUDGET`] while the same hold is in effect.
#[test]
fn baseline_a_producer_waits_for_a_parked_drain_pass() {
    let engine = Arc::new(RuntimeEventEngine::with_capacity(8));
    let hold = engine.install_drain_hold(Arc::new(DrainHold::new(HOLD)));

    let draining = Arc::clone(&engine);
    let drain = std::thread::spawn(move || draining.drain());
    assert!(
        hold.wait_until_entered(Duration::from_secs(5)),
        "the drain pass never reached the hold point"
    );

    let scope = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);
    let started = Instant::now();
    let outcome = ingress.try_submit(diagnostic_fact());
    let waited = started.elapsed();

    drain.join().expect("drain thread panicked");

    assert_eq!(outcome, SubmitOutcome::Accepted);
    assert!(
        waited >= PRODUCER_BUDGET,
        "a producer returned in {waited:?} while a drain pass was parked for \
         {HOLD:?}. If this now passes quickly the coupling is gone -- replace \
         this baseline with the inverse assertion rather than loosening it."
    );
}

/// BASELINE, not the target: `try_submit` takes the process-global
/// ingress gate, the wake-list mutex, and a lane mutex.
///
/// [`crate::runtime_events::lock_audit::PRODUCER_ALLOWED`] is empty, which
/// is what this set has to become. The record is per-thread, so this is
/// exact regardless of what the rest of the suite is doing in parallel.
#[test]
fn baseline_records_every_lock_class_a_producer_takes() {
    // A dedicated thread so the measurement covers this submission only.
    let recorded = std::thread::spawn(|| {
        let engine = RuntimeEventEngine::with_capacity(4);
        let scope = OperationScope::root_only(OperationId::new());
        let ingress = engine.unreserved_ingress(scope);

        clear_thread_record();
        let outcome = ingress.try_submit(state_transition_fact());
        let taken = thread_record();

        assert_eq!(outcome, SubmitOutcome::Accepted);
        taken
    })
    .join()
    .expect("recording thread panicked");

    for expected in [
        LockClass::IngressGate,
        LockClass::WakeList,
        LockClass::StateLaneEntries,
        LockClass::StateLaneLatest,
    ] {
        assert!(
            recorded.contains(&expected),
            "expected the producer path to take {expected:?} today; recorded {recorded:?}"
        );
    }

    assert!(
        !within(&recorded, PRODUCER_ALLOWED),
        "the producer path is already inside PRODUCER_ALLOWED ({recorded:?}). \
         That is the goal, not the baseline -- turn the audit on and delete \
         this test rather than relaxing it."
    );
}

/// BASELINE, not the target: `reserve_root` takes the process-global
/// ingress gate on top of the table locks it legitimately needs.
///
/// [`RESERVE_ALLOWED`] is the set a reservation is permitted once the
/// audit is enforcing: its own slot, the free list, and the child index.
/// The gate is not in it.
#[test]
fn baseline_records_every_lock_class_a_reservation_takes() {
    let recorded = std::thread::spawn(|| {
        let engine = Arc::new(RuntimeEventEngine::with_capacity(4));

        clear_thread_record();
        let reservation =
            engine.reserve_root(OperationId::new(), super::fixtures::synthetic_unknown);
        let taken = thread_record();

        assert!(reservation.is_some(), "a fresh table has capacity");
        reservation.expect("checked above").cancel();
        taken
    })
    .join()
    .expect("recording thread panicked");

    assert!(
        recorded.contains(&LockClass::IngressGate),
        "expected reserve_root to take the ingress gate today; recorded {recorded:?}"
    );
    assert!(
        !within(&recorded, RESERVE_ALLOWED),
        "the reserve path is already inside RESERVE_ALLOWED ({recorded:?}). \
         That is the goal, not the baseline."
    );
}

/// Whether every recorded class is permitted by `allowed`.
fn within(recorded: &[LockClass], allowed: u32) -> bool {
    recorded
        .iter()
        .all(|class| allowed & (1 << (*class as u8)) != 0)
}

/// The audit attributes locks to the right context and restores it: a
/// submit runs as `Producer`, a reservation runs as `Reserve`, and
/// neither scope outlives its call.
///
/// This is the mechanism the enforcing mode depends on, so it is pinned
/// independently of what either path currently locks.
#[test]
fn context_does_not_outlive_the_call_it_marks() {
    let engine = RuntimeEventEngine::with_capacity(4);
    assert_eq!(current_context(), Context::Unmarked);

    let reservation = engine
        .reserve_root(OperationId::new(), super::fixtures::synthetic_unknown)
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
        if EXEMPT.contains(&name)
            || name == "tests.rs"
            || path.components().any(|part| part.as_os_str() == "tests")
        {
            return;
        }
        for (index, line) in source.lines().enumerate() {
            let trimmed = line.trim_start();
            // A field declaration, not a use/type/expression: `name: Mutex<..>,`
            let is_field = trimmed.ends_with(',')
                && trimmed.contains(": Mutex<")
                && !trimmed.starts_with("//");
            let is_construction = trimmed.contains("Mutex::new(")
                && !trimmed.contains("AuditedMutex::new(")
                && !trimmed.starts_with("//");
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
