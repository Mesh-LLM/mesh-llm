//! The declared callback ingress budget, actually asserted.
//!
//! `CALLBACK_INGRESS_P99_BUDGET` has existed as a constant with no Rust
//! reader: nothing compared a measured p99 against it, and the
//! benchmark comparator carried its own independent copy of the same
//! number. A budget nothing checks is a comment.
//!
//! This file makes it load-bearing. It is deliberately a separate
//! integration binary from `ingress_reservoir_no_alloc.rs`, which installs
//! a counting global allocator: measuring latency under an instrumented
//! allocator would measure the instrument.
//!
//! **Policy for this gate, stated up front.** It is one of only two timing
//! assertions in the subsystem. If it fails, the runner gets fixed or the
//! regression gets fixed. The bound does not get loosened -- a budget that
//! moves to accommodate whatever the code currently does is back to being
//! a comment.

use std::time::Duration;

use mesh_llm_host_runtime::runtime_events::config::CALLBACK_INGRESS_P99_BUDGET;
use mesh_llm_host_runtime::runtime_events::engine::RuntimeEventEngine;
use mesh_llm_runtime_event_contracts::{
    DiagnosticEventKind, FamilyFact, GenerationEventKind, NativeRuntimeEventKind, OperationId,
    Outcome, RequestEventKind, RuntimeEventIngress, RuntimeFact, SubmitOutcome,
};

/// Submissions per class. Ten thousand total, so the p99 is the
/// hundredth-worst sample rather than a single outlier.
const PER_CLASS: usize = 2_500;

fn terminal_fact() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestCompleted,
        mesh_llm_runtime_event_contracts::FactData {
            outcome: Some(Outcome::Success),
            ..mesh_llm_runtime_event_contracts::FactData::default()
        },
    ))
}

fn state_transition_fact() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeInitialized))
}

fn progress_fact() -> RuntimeFact {
    RuntimeFact::Generation(FamilyFact::new(GenerationEventKind::GenerationProgress))
}

fn diagnostic_fact() -> RuntimeFact {
    RuntimeFact::Diagnostic(FamilyFact::new(DiagnosticEventKind::WarningRaised))
}

/// A mixed 10,000-submission workload's measured p99 must fit the declared
/// budget.
///
/// `ingress_p99_us` times `RuntimeEventEngine::submit` end to end: the
/// metadata fill, the class decision, the reservation reads, the ring
/// push, and the telemetry tail. That is what a producer thread pays to
/// emit one event. It does not cover a caller's own locking before it
/// reaches `submit` -- see `docs/design/RUNTIME_EVENT_ARCHITECTURE_REPAIRS.md`.
#[test]
fn a_mixed_workload_meets_the_declared_ingress_p99_budget() {
    let engine = RuntimeEventEngine::new();
    // One long-lived reservation per class that needs one. Reserving is
    // once-per-operation work and is not what this budget covers.
    let progress_owner = engine
        .reserve_root(OperationId::new(), terminal_fact)
        .expect("a full-capacity table has room");

    for _ in 0..PER_CLASS {
        let scope = mesh_llm_runtime_event_contracts::OperationScope::root_only(OperationId::new());
        let unreserved = engine.unreserved_ingress(scope);
        let _ = unreserved.try_submit(state_transition_fact());
        let _ = unreserved.try_submit(diagnostic_fact());
        let _ = progress_owner.ingress().try_submit(progress_fact());

        // A terminal needs its own reservation, since the claim is
        // write-once. Drain as we go so the table and the ring both stay
        // in their steady state rather than filling up.
        if let Some(reservation) = engine.reserve_root(OperationId::new(), terminal_fact) {
            let _ = reservation.ingress().try_submit(terminal_fact());
        }
        engine.drain();
    }

    let measured = engine
        .ingress_p99_us()
        .expect("10,000 submissions is far past the minimum sample count");
    let budget = u64::try_from(CALLBACK_INGRESS_P99_BUDGET.as_micros()).expect("budget fits u64");

    assert!(
        measured <= budget,
        "ingress p99 was {measured} us against a {budget} us budget. This gate \
         does not get loosened: either the submit path regressed, or the \
         machine running this is too loaded to measure it. Fix whichever it \
         is."
    );
}

/// The budget is a real bound, not a number so large it can never fail.
///
/// A gate that would pass for any implementation proves nothing, so this
/// pins the declared budget at the order of magnitude it was chosen at.
#[test]
fn the_declared_budget_is_tight_enough_to_mean_something() {
    assert!(
        CALLBACK_INGRESS_P99_BUDGET <= Duration::from_micros(1_000),
        "a budget above a millisecond would not distinguish a lock-free push \
         from a contended mutex"
    );
}

/// A submission that is refused is still on the producer's clock, so it is
/// measured too. This proves the reservoir records rejected outcomes
/// rather than only the happy path.
#[test]
fn refused_submissions_are_measured_as_well() {
    let engine = RuntimeEventEngine::with_capacity(1);
    let scope = mesh_llm_runtime_event_contracts::OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    // An unreserved terminal has no slot to own it: always refused.
    for _ in 0..200 {
        assert_eq!(
            ingress.try_submit(terminal_fact()),
            SubmitOutcome::TerminalDeliveryFailed
        );
    }

    assert!(
        engine.ingress_p99_us().is_some(),
        "refused submissions must still be recorded, or the p99 would only \
         ever describe the path that succeeds"
    );
}
