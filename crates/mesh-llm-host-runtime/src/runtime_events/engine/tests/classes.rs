//! Task 4 acceptance: every delivery class (terminal, state-transition,
//! progress, diagnostic) is applied through the reducer and published,
//! coalescing is per `(OperationScope, kind)` -- never globally by kind --
//! and the ONE shared engine-level ingress-sequence counter (`wake.rs`) is
//! consumed by every submit outcome, so a coalesced/dropped/failed input
//! leaves a permanent gap in the published sequence space rather than
//! silently vanishing. Fixes review defect D2
//! (`.omo/plans/event-system-fixes.md` task 4): before this module,
//! `engine::drain` only ever processed the terminal wake list, so state,
//! progress, and diagnostic facts were accepted at the lane but never
//! reached the reducer or a subscriber.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use mesh_llm_runtime_event_contracts::{
    DiagnosticEventKind, FactData, FamilyFact, OperationId, OperationScope, Outcome,
    RequestEventKind, RuntimeEventIngress, RuntimeFact, Severity, SubmitOutcome,
};

use super::fixtures::{
    diagnostic_fact, progress_fact, state_transition_fact, synthetic_unknown, terminal_success,
};
use crate::runtime_events::config::PROGRESS_EXPORT_INTERVAL;
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::ingress::NON_TERMINAL_CREDITS;

/// The core D2 regression: two DIFFERENT operations reporting the SAME
/// state-transition kind must both reach the reducer and publish -- a
/// global-by-kind lane would silently drop one of them.
#[test]
fn state_transitions_from_different_scopes_never_coalesce_across_scopes() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope_a = OperationScope::root_only(OperationId::new());
    let scope_b = OperationScope::root_only(OperationId::new());

    assert_eq!(
        engine
            .unreserved_ingress(scope_a)
            .try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        engine
            .unreserved_ingress(scope_b)
            .try_submit(state_transition_fact()),
        SubmitOutcome::Accepted,
        "the same kind for a DIFFERENT scope must never coalesce with scope_a's pending entry"
    );

    let report = engine.drain();
    assert_eq!(
        report.applied, 2,
        "both scopes' state-transition facts must be drained"
    );
    let frames = engine.replay().snapshot();
    assert_eq!(
        frames.len(),
        2,
        "both scopes' facts must be applied and published, not just one"
    );
    let published_scopes: HashSet<OperationScope> =
        frames.iter().map(|frame| frame.scope).collect();
    assert!(published_scopes.contains(&scope_a));
    assert!(published_scopes.contains(&scope_b));
}

/// Same scope, same kind coalesces, and the value that publishes is the
/// LATEST submission's, at the latest arrival position.
///
/// Both submissions are `Accepted`: the coalescing decision now belongs to
/// the consumer, so a producer is never told its fact replaced another's.
#[test]
fn state_transition_repeat_key_coalesces_to_the_latest_value() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
    let other = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    // An unrelated scope between the two, so "latest position" is
    // observable: the coalesced frame must publish AFTER this one.
    assert_eq!(
        engine
            .unreserved_ingress(other)
            .try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );

    let report = engine.drain();
    assert_eq!(
        report.applied, 2,
        "the repeated key drains as exactly one entry, alongside the other scope's"
    );
    let frames = engine.replay().snapshot();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].scope, other);
    assert_eq!(
        frames[1].scope, scope,
        "the coalesced value publishes at its newest arrival position, not its first"
    );
    assert_eq!(
        frames
            .iter()
            .map(|frame| frame.sequence.get())
            .collect::<Vec<_>>(),
        vec![1, 2],
        "sequences are assigned at publication, so they are contiguous"
    );
}

/// A key drained by one `drain()` call is gone from the lane: the next
/// submission for the same `(scope, kind)` starts fresh (`Accepted`, not
/// `Coalesced`).
#[test]
fn a_drained_state_transition_key_starts_fresh_on_the_next_submission() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    engine.drain();

    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted,
        "the lane must be empty again after a full drain, so the same key is fresh, not a repeat"
    );
    assert_eq!(engine.drain().applied, 1);
    assert_eq!(
        engine.replay().snapshot().len(),
        2,
        "both drains published their own frame"
    );
}

/// Diagnostics currently sit in their bounded queue forever (D2): this
/// proves a submitted diagnostic is actually applied through the reducer
/// and published once `drain()` runs.
#[test]
fn diagnostic_facts_drain_through_the_reducer_and_publish() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    for _ in 0..3 {
        assert_eq!(
            ingress.try_submit(diagnostic_fact()),
            SubmitOutcome::Accepted
        );
    }

    let report = engine.drain();
    assert_eq!(report.applied, 3);
    assert_eq!(
        engine.replay().snapshot().len(),
        3,
        "every accepted diagnostic must reach the stream"
    );
}

/// A diagnostic refused for exceeding the ring's non-terminal budget must
/// never reach the reducer -- only what was actually accepted publishes.
#[test]
fn diagnostics_dropped_past_the_budget_never_reach_the_reducer() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    for _ in 0..NON_TERMINAL_CREDITS {
        assert_eq!(
            ingress.try_submit(diagnostic_fact()),
            SubmitOutcome::Accepted
        );
    }
    assert_eq!(
        ingress.try_submit(diagnostic_fact()),
        SubmitOutcome::DroppedDiagnostic
    );

    assert_eq!(
        engine.drain().applied,
        NON_TERMINAL_CREDITS,
        "the refused entry must never have been applied or published"
    );
}

#[test]
fn rust_ingress_defaults_preserve_failure_unknown_and_fatal_severity() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let failure_scope = OperationScope::root_only(OperationId::new());
    let unknown_scope = OperationScope::root_only(OperationId::new());
    let failure = RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestExecutionStarted,
        FactData {
            outcome: Some(Outcome::Failure),
            ..FactData::default()
        },
    ));
    let unknown = RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestExecutionStarted,
        FactData {
            outcome: Some(Outcome::Unknown),
            ..FactData::default()
        },
    ));
    assert_eq!(
        engine.unreserved_ingress(failure_scope).try_submit(failure),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        engine.unreserved_ingress(unknown_scope).try_submit(unknown),
        SubmitOutcome::Accepted
    );

    let fatal_reservation = engine
        .reserve_root(OperationId::new(), super::fixtures::synthetic_unknown)
        .expect("reserve fatal diagnostic");
    assert_eq!(
        fatal_reservation
            .ingress()
            .try_submit(RuntimeFact::Diagnostic(FamilyFact::new(
                DiagnosticEventKind::FatalNativeFailure,
            ))),
        SubmitOutcome::Accepted
    );
    engine.drain();

    let frames = engine.replay().snapshot();
    assert_eq!(
        frames[0].fact.metadata().map(|metadata| metadata.severity),
        Some(Severity::Error)
    );
    assert_eq!(
        frames[1].fact.metadata().map(|metadata| metadata.severity),
        Some(Severity::Warning)
    );
    assert_eq!(
        frames[2].fact.metadata().map(|metadata| metadata.severity),
        Some(Severity::Fatal)
    );
}

/// Progress facts flush at most once per the frozen 100 ms export
/// interval, carrying only the latest coalesced value -- proven with pure
/// `Instant` arithmetic (`drain_up_to_at`), never a real sleep.
#[test]
fn progress_flushes_at_most_once_per_hundred_milliseconds_with_the_latest_value() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let ingress = reservation.ingress();
    let t0 = Instant::now();

    // Consume the "first call always flushes" baseline with nothing
    // pending yet, so the 100 ms gate below measures from a known instant.
    assert_eq!(engine.drain_up_to_at(None, t0).applied, 0);

    assert_eq!(ingress.try_submit(progress_fact()), SubmitOutcome::Accepted);
    let mid_flush = engine.drain_up_to_at(None, t0 + Duration::from_millis(40));
    assert_eq!(
        mid_flush.applied, 0,
        "under the 100ms interval, progress must not flush yet"
    );
    assert!(engine.replay().is_empty());

    // A second progress update before the interval elapses supersedes the
    // first -- still only one value pending per operation.
    assert_eq!(ingress.try_submit(progress_fact()), SubmitOutcome::Accepted);

    let due_flush = engine.drain_up_to_at(None, t0 + PROGRESS_EXPORT_INTERVAL);
    assert_eq!(
        due_flush.applied, 1,
        "exactly one progress frame publishes once the interval elapses"
    );
    let frames = engine.replay().snapshot();
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].sequence.get(),
        1,
        "sequences are assigned at publication, so the first published frame is 1 \
         however many submissions were superseded to produce it"
    );
    assert_eq!(
        engine.health().snapshot().dropped_progress,
        1,
        "the superseded snapshot is counted"
    );

    reservation.cancel();
}

/// Terminal, state-transition, and diagnostic facts submitted for the SAME
/// scope must all apply through the SAME transactional reducer, in
/// ingress-sequence order -- the reducer's own per-scope ordering
/// invariant is unaffected by which lane a fact travels through.
#[test]
fn mixed_classes_for_one_scope_apply_in_ingress_sequence_order() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());

    assert_eq!(
        engine
            .unreserved_ingress(scope)
            .try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        engine
            .unreserved_ingress(scope)
            .try_submit(diagnostic_fact()),
        SubmitOutcome::Accepted
    );

    let report = engine.drain();
    assert_eq!(report.applied, 2);
    let frames = engine.replay().snapshot();
    assert_eq!(frames.len(), 2);
    let sequences: Vec<u64> = frames.iter().map(|frame| frame.sequence.get()).collect();
    let mut sorted = sequences.clone();
    sorted.sort_unstable();
    assert_eq!(
        sequences, sorted,
        "mixed classes for one scope must publish in ingress-sequence order"
    );
    assert_eq!(sequences, vec![1, 2]);
}

#[test]
fn reserved_state_before_its_terminal_survives_same_pass_release() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let ingress = reservation.ingress();
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        ingress.try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );

    engine.drain();
    let sequences: Vec<_> = engine
        .replay()
        .snapshot()
        .into_iter()
        .map(|frame| frame.sequence.get())
        .collect();
    assert_eq!(sequences, vec![1, 2]);
}

#[test]
fn drain_report_excludes_state_rejected_by_reservation_validation() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let ingress = reservation.ingress();
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    reservation.cancel();

    let report = engine.drain();
    assert_eq!(report.applied, 0);
    assert_eq!(report.left_queued, 0);
    assert!(engine.replay().is_empty());
}

#[test]
fn drain_report_excludes_facts_rejected_by_the_reducer() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let ingress = reservation.ingress();
    assert_eq!(
        ingress.try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );

    let report = engine.drain();
    assert_eq!(report.applied, 1);
    assert_eq!(engine.replay().snapshot().len(), 1);
    assert_eq!(engine.health().snapshot().reducer_rejected, 1);
}

/// Progress held across a pass is not stranded behind facts that
/// published in the meantime.
///
/// The old drain assigned a progress fact its sequence at submission and
/// then had to discard it if an unrelated fact had published past that
/// number before its export window came due -- a value dropped purely for
/// the order it was numbered in. Assigning the sequence at publication
/// removes the problem rather than compensating for it: the held value
/// publishes at its own window, with a sequence that reflects where it
/// actually published.
#[test]
fn progress_held_across_a_pass_still_publishes_at_its_window() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let ingress = reservation.ingress();
    let t0 = Instant::now();
    assert_eq!(engine.drain_up_to_at(None, t0).applied, 0);

    assert_eq!(ingress.try_submit(progress_fact()), SubmitOutcome::Accepted);
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );

    // Inside the window: the state transition publishes, the progress does
    // not.
    assert_eq!(
        engine
            .drain_up_to_at(None, t0 + Duration::from_millis(40))
            .applied,
        1
    );

    let report = engine.drain_up_to_at(None, t0 + PROGRESS_EXPORT_INTERVAL);
    assert_eq!(
        report.applied, 1,
        "the held progress publishes once its window comes due"
    );
    let sequences: Vec<u64> = engine
        .replay()
        .snapshot()
        .iter()
        .map(|frame| frame.sequence.get())
        .collect();
    assert_eq!(
        sequences,
        vec![1, 2],
        "the progress frame is numbered where it published, after the state transition"
    );
    assert_eq!(engine.health().snapshot().dropped_progress, 0);

    reservation.cancel();
}

/// Submitting consumes no sequence at all.
///
/// This inverts the previous contract, where every outcome -- including a
/// coalesce or a drop -- burned a number under the admission gate. Minting
/// at submission is exactly what forced that gate to exist: assigning a
/// number and placing the fact had to be one atomic step for the drain to
/// be able to reconstruct an order across four separate containers.
/// Sequences are now assigned at publication instead, so there is nothing
/// to mint and nothing to synchronize.
#[test]
fn submitting_consumes_no_sequence() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);
    let before = engine.peek_next_sequence();

    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    // Repeat kind for the same scope: coalescing now happens in the
    // consumer, so the producer is simply told the fact was accepted.
    assert_eq!(
        ingress.try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    assert_eq!(
        ingress.try_submit(progress_fact()),
        SubmitOutcome::DroppedProgress
    );
    assert_eq!(
        ingress.try_submit(terminal_success()),
        SubmitOutcome::TerminalDeliveryFailed
    );
    assert_eq!(
        ingress.try_submit(diagnostic_fact()),
        SubmitOutcome::Accepted
    );

    assert_eq!(
        engine.peek_next_sequence(),
        before,
        "no submit outcome may advance the publication sequence"
    );
}

/// The client-visible cursor contract: published sequences are contiguous
/// and monotonic. A dropped input leaves no hole, because it never
/// received a number to leave one with.
///
/// The drops themselves are not lost evidence -- `runtime_health` counts
/// them per class, which says both what was dropped and how many, where an
/// anonymous gap said neither.
#[test]
fn published_sequences_are_contiguous_across_a_drop() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope_a = OperationScope::root_only(OperationId::new());
    let scope_b = OperationScope::root_only(OperationId::new());

    assert_eq!(
        engine
            .unreserved_ingress(scope_a)
            .try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );
    // Dropped at the boundary: never queued, never published.
    assert_eq!(
        engine
            .unreserved_ingress(scope_a)
            .try_submit(progress_fact()),
        SubmitOutcome::DroppedProgress
    );
    assert_eq!(
        engine
            .unreserved_ingress(scope_b)
            .try_submit(state_transition_fact()),
        SubmitOutcome::Accepted
    );

    engine.drain();
    let sequences: Vec<u64> = engine
        .replay()
        .snapshot()
        .iter()
        .map(|frame| frame.sequence.get())
        .collect();
    assert_eq!(sequences, vec![1, 2]);
    assert_eq!(engine.health().snapshot().dropped_progress, 1);
}

/// Distinct state-transition keys are lossless up to the ring's shared
/// non-terminal budget; the first past it is refused and counted, and no
/// already-accepted key is evicted to make room for it.
///
/// The budget is the sum of the two frozen per-class lane depths, so one
/// shared budget cannot admit less than the separate lanes promised.
#[test]
fn a_state_transition_past_the_non_terminal_budget_is_rejected_without_evicting() {
    let engine = RuntimeEventEngine::with_capacity(4);
    for _ in 0..NON_TERMINAL_CREDITS {
        let scope = OperationScope::root_only(OperationId::new());
        assert_eq!(
            engine
                .unreserved_ingress(scope)
                .try_submit(state_transition_fact()),
            SubmitOutcome::Accepted
        );
    }

    let overflow = OperationScope::root_only(OperationId::new());
    assert_eq!(
        engine
            .unreserved_ingress(overflow)
            .try_submit(state_transition_fact()),
        SubmitOutcome::RejectedCapacity
    );
    assert_eq!(engine.health().snapshot().state_transition_rejected, 1);

    // Asserted on what the pass published, not on replay length: replay
    // is separately bounded at REPLAY_MAX_FRAMES and would cap the count
    // long before the budget does.
    assert_eq!(
        engine.drain().applied,
        NON_TERMINAL_CREDITS,
        "every accepted key must still publish; the rejection evicts nothing"
    );
}
