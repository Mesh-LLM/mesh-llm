//! Prefill/generation execution state on in-flight request rows.
//!
//! Correlation mirrors production: the request row key is the request
//! root's uuid text, and Skippy prefill/generation facts arrive on CHILD
//! scopes of that root with no `scope.request_id`.

use std::sync::Arc;

use mesh_llm_runtime_event_contracts::{
    ChildOperationId, FactData, FamilyFact, GenerationEventKind, OperationId, OperationScope,
    Outcome, PrefillEventKind, RequestEventKind, RuntimeFact,
};

use super::super::fixtures::{apply_all, request_fact, summaries, tokens};
use crate::runtime_events::reducer::{
    ReducerSnapshot, RequestDomainState, RequestGenerationState, RequestPrefillState,
};

struct Request {
    root: OperationId,
}

impl Request {
    fn new() -> Self {
        Self {
            root: OperationId::new(),
        }
    }

    fn id(&self) -> String {
        self.root.to_string()
    }

    fn root_scope(&self) -> OperationScope {
        OperationScope::root_only(self.root)
    }

    fn child(&self) -> OperationScope {
        OperationScope::with_child(self.root, ChildOperationId::new())
    }

    fn received(&self) -> Arc<ReducerSnapshot> {
        apply_all(
            &ReducerSnapshot::empty(),
            self.root_scope(),
            0,
            vec![request_fact(RequestEventKind::RequestReceived, &self.id())],
        )
    }

    fn row(&self, snapshot: &ReducerSnapshot) -> RequestDomainState {
        snapshot
            .domain()
            .requests()
            .into_iter()
            .find(|row| row.id == self.id())
            .expect("request row is in flight")
    }

    fn generation(&self, snapshot: &ReducerSnapshot) -> RequestGenerationState {
        self.row(snapshot).generation.expect("generation state")
    }

    fn prefill(&self, snapshot: &ReducerSnapshot) -> RequestPrefillState {
        self.row(snapshot).prefill.expect("prefill state")
    }
}

fn generation(kind: GenerationEventKind) -> RuntimeFact {
    RuntimeFact::Generation(FamilyFact::new(kind))
}

fn generated(kind: GenerationEventKind, count: u64) -> RuntimeFact {
    RuntimeFact::Generation(FamilyFact::with_data(
        kind,
        FactData {
            numeric_summaries: summaries(&[("generated_token_count", count)]),
            ..FactData::default()
        },
    ))
}

fn generation_terminal(kind: GenerationEventKind, outcome: Outcome, count: u64) -> RuntimeFact {
    RuntimeFact::Generation(FamilyFact::with_data(
        kind,
        FactData {
            outcome: Some(outcome),
            numeric_summaries: summaries(&[("generated_token_count", count)]),
            ..FactData::default()
        },
    ))
}

fn prefill(kind: PrefillEventKind, data: FactData) -> RuntimeFact {
    RuntimeFact::Prefill(FamilyFact::with_data(kind, data))
}

fn computed(count: u64) -> FactData {
    FactData {
        numeric_summaries: summaries(&[("computed_tokens", count)]),
        ..FactData::default()
    }
}

#[test]
fn child_scope_updates_execution_state_of_root_row() {
    let request = Request::new();

    let snapshot = apply_all(
        &request.received(),
        request.child(),
        1,
        vec![
            generation(GenerationEventKind::GenerationStarted),
            generation(GenerationEventKind::FirstTokenProduced),
        ],
    );

    let row = request.row(&snapshot);
    assert_eq!(
        row.state.as_deref(),
        Some("received"),
        "root status is untouched"
    );
    let generation = row
        .generation
        .expect("child fact attaches generation state");
    assert_eq!(generation.phase, Some("streaming"));
    assert!(generation.first_token);
}

#[test]
fn a_later_root_request_fact_keeps_the_rows_execution_state() {
    let request = Request::new();
    let snapshot = apply_all(
        &request.received(),
        request.child(),
        1,
        vec![generation(GenerationEventKind::GenerationStarted)],
    );

    let snapshot = apply_all(
        &snapshot,
        request.root_scope(),
        2,
        vec![request_fact(
            RequestEventKind::RequestExecutionStarted,
            &request.id(),
        )],
    );

    assert_eq!(request.generation(&snapshot).phase, Some("started"));
}

#[test]
fn progress_is_monotonic_and_regressions_ignored() {
    let request = Request::new();

    let snapshot = apply_all(
        &request.received(),
        request.child(),
        1,
        vec![
            generation(GenerationEventKind::GenerationStarted),
            generated(GenerationEventKind::GenerationProgress, 10),
            generated(GenerationEventKind::GenerationProgress, 4),
            generated(GenerationEventKind::GenerationProgress, 12),
            generated(GenerationEventKind::GenerationProgress, 11),
        ],
    );

    assert_eq!(request.generation(&snapshot).generated_tokens, Some(12));
}

#[test]
fn prefill_progress_is_monotonic_and_regressions_ignored() {
    let request = Request::new();

    let snapshot = apply_all(
        &request.received(),
        request.child(),
        1,
        vec![
            prefill(PrefillEventKind::PrefillStarted, FactData::default()),
            prefill(PrefillEventKind::PrefillProgress, computed(64)),
            prefill(PrefillEventKind::PrefillProgress, computed(32)),
        ],
    );

    let prefill = request.prefill(&snapshot);
    assert_eq!(prefill.phase, Some("progress"));
    assert_eq!(prefill.computed_tokens, Some(64));
}

#[test]
fn terminal_overrides_dropped_progress() {
    let request = Request::new();

    let snapshot = apply_all(
        &request.received(),
        request.child(),
        1,
        vec![
            generation(GenerationEventKind::GenerationStarted),
            generation_terminal(
                GenerationEventKind::GenerationCompleted,
                Outcome::Success,
                42,
            ),
        ],
    );

    let generation = request.generation(&snapshot);
    assert_eq!(generation.phase, Some("completed"));
    assert_eq!(generation.generated_tokens, Some(42));
    assert_eq!(generation.outcome, Some("success"));
}

#[test]
fn terminal_pins_final_counts_even_below_the_progress_high_water_mark() {
    let request = Request::new();

    let snapshot = apply_all(
        &request.received(),
        request.child(),
        1,
        vec![
            generated(GenerationEventKind::GenerationProgress, 30),
            generation_terminal(
                GenerationEventKind::GenerationCancelled,
                Outcome::Cancelled,
                25,
            ),
        ],
    );

    let generation = request.generation(&snapshot);
    assert_eq!(generation.phase, Some("cancelled"));
    assert_eq!(generation.generated_tokens, Some(25));
}

#[test]
fn late_progress_after_terminal_ignored() {
    let request = Request::new();
    let settled = apply_all(
        &request.received(),
        request.child(),
        1,
        vec![generation_terminal(
            GenerationEventKind::GenerationCompleted,
            Outcome::Success,
            8,
        )],
    );

    let snapshot = apply_all(
        &settled,
        request.child(),
        2,
        vec![
            generated(GenerationEventKind::GenerationProgress, 99),
            generation(GenerationEventKind::StopConditionReached),
        ],
    );

    let generation = request.generation(&snapshot);
    assert_eq!(generation.phase, Some("completed"));
    assert_eq!(generation.generated_tokens, Some(8));
    assert!(!generation.stop_condition_reached);
}

#[test]
fn stop_condition_before_the_terminal_is_recorded() {
    let request = Request::new();

    let snapshot = apply_all(
        &request.received(),
        request.child(),
        1,
        vec![
            generation(GenerationEventKind::StopConditionReached),
            generation_terminal(
                GenerationEventKind::GenerationCompleted,
                Outcome::Success,
                3,
            ),
        ],
    );

    assert!(request.generation(&snapshot).stop_condition_reached);
}

#[test]
fn prefill_terminal_and_cache_restore_are_recorded() {
    let request = Request::new();

    let snapshot = apply_all(
        &request.received(),
        request.child(),
        1,
        vec![
            prefill(
                PrefillEventKind::PromptCacheRestorePartial,
                FactData::default(),
            ),
            prefill(
                PrefillEventKind::PrefillCompleted,
                FactData {
                    outcome: Some(Outcome::Success),
                    numeric_summaries: summaries(&[("cached_tokens", 16), ("computed_tokens", 48)]),
                    progress: tokens(48),
                    ..FactData::default()
                },
            ),
        ],
    );

    let prefill = request.prefill(&snapshot);
    assert_eq!(prefill.phase, Some("completed"));
    assert_eq!(prefill.cache_restore, Some("partial"));
    assert_eq!(prefill.cached_tokens, Some(16));
    assert_eq!(prefill.computed_tokens, Some(48));
    assert_eq!(prefill.outcome, Some("success"));
}

#[test]
fn facts_without_a_root_request_row_are_ignored() {
    let orphan_root = OperationScope::root_only(OperationId::new());

    let snapshot = apply_all(
        &ReducerSnapshot::empty(),
        orphan_root,
        0,
        vec![
            generation(GenerationEventKind::GenerationStarted),
            prefill(PrefillEventKind::PrefillStarted, FactData::default()),
        ],
    );

    assert!(snapshot.domain().requests().is_empty());
}

#[test]
fn execution_facts_after_the_root_request_settles_do_not_resurrect_it() {
    let request = Request::new();
    let completed = apply_all(
        &request.received(),
        request.root_scope(),
        1,
        vec![request_fact(
            RequestEventKind::RequestCompleted,
            &request.id(),
        )],
    );

    let snapshot = apply_all(
        &completed,
        request.child(),
        2,
        vec![generation_terminal(
            GenerationEventKind::GenerationCompleted,
            Outcome::Success,
            5,
        )],
    );

    assert!(snapshot.domain().requests().is_empty());
}
