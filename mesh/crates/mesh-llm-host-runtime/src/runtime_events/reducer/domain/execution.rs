//! Prefill and generation execution state attached to an existing
//! in-flight request row.
//!
//! Correlation: prefill/generation facts carry no `scope.request_id`. The
//! Skippy adapter reserves them as children of the OpenAI request root,
//! whose `OperationId` is byte-equal to the logging request id, and the
//! request row key is that same uuid's text
//! (`network::openai::runtime_events::scope_for_request`). So the row key
//! is `scope.request_id` when present, else `scope.root()`'s uuid text.
//! These facts never create a row: a generation without a frontend request
//! (its own freshly minted root) or one arriving after the root request
//! settled is ignored.
//!
//! Rules, per family: a terminal kind overrides the phase and pins final
//! counts from the terminal's own progress/summaries; anything after the
//! terminal is ignored; non-terminal counts only move up.

use mesh_llm_runtime_event_contracts::{
    FactData, GenerationEventKind, OperationScope, PrefillEventKind, ProgressUnit,
};

use super::{DomainState, RequestDomainState, outcome_label, unsigned_summary};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RequestPrefillState {
    pub phase: Option<&'static str>,
    pub cached_tokens: Option<u64>,
    pub computed_tokens: Option<u64>,
    pub cache_restore: Option<&'static str>,
    pub outcome: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RequestGenerationState {
    pub phase: Option<&'static str>,
    pub generated_tokens: Option<u64>,
    pub first_token: bool,
    pub stop_condition_reached: bool,
    pub outcome: Option<&'static str>,
}

const PREFILL_TERMINAL_PHASES: [&str; 3] = ["completed", "cancelled", "failed"];
const GENERATION_TERMINAL_PHASES: [&str; 4] = ["completed", "cancelled", "timed_out", "failed"];

fn is_settled(phase: Option<&'static str>, terminals: &[&str]) -> bool {
    phase.is_some_and(|phase| terminals.contains(&phase))
}

fn row_key(scope: OperationScope, data: &FactData) -> String {
    data.scope
        .request_id
        .as_ref()
        .map_or_else(|| scope.root().to_string(), |id| id.as_str().to_string())
}

fn existing_row<'a>(
    state: &'a mut DomainState,
    scope: OperationScope,
    data: &FactData,
) -> Option<&'a mut RequestDomainState> {
    state.requests.get_mut(&row_key(scope, data))
}

fn token_progress(data: &FactData) -> Option<u64> {
    data.progress
        .filter(|progress| progress.unit == ProgressUnit::Tokens)
        .map(|progress| progress.current)
}

fn raise(slot: &mut Option<u64>, candidate: Option<u64>) {
    if let Some(candidate) = candidate {
        *slot = Some(slot.map_or(candidate, |current| current.max(candidate)));
    }
}

fn pin(slot: &mut Option<u64>, terminal_value: Option<u64>) {
    if terminal_value.is_some() {
        *slot = terminal_value;
    }
}

enum PrefillStep {
    Phase(&'static str),
    CacheRestore(&'static str),
    Terminal(&'static str),
}

const fn prefill_step(kind: PrefillEventKind) -> PrefillStep {
    use PrefillEventKind::{
        MediaPrefillCompleted, MediaPrefillFailed, MediaPrefillStarted, PrefillCancelled,
        PrefillCompleted, PrefillFailed, PrefillProgress, PrefillStarted, PromptCacheRestoreError,
        PromptCacheRestoreHit, PromptCacheRestoreMiss, PromptCacheRestorePartial,
        PromptProcessingStarted, TokenizationCompleted, TokenizationFailed,
    };
    match kind {
        PromptProcessingStarted | PrefillStarted | MediaPrefillStarted => {
            PrefillStep::Phase("started")
        }
        // Sub-step completions: prefill as a whole keeps running.
        PrefillProgress | TokenizationCompleted | MediaPrefillCompleted => {
            PrefillStep::Phase("progress")
        }
        PromptCacheRestoreHit => PrefillStep::CacheRestore("hit"),
        PromptCacheRestoreMiss => PrefillStep::CacheRestore("miss"),
        PromptCacheRestorePartial => PrefillStep::CacheRestore("partial"),
        PromptCacheRestoreError => PrefillStep::CacheRestore("error"),
        PrefillCompleted => PrefillStep::Terminal("completed"),
        PrefillCancelled => PrefillStep::Terminal("cancelled"),
        PrefillFailed | TokenizationFailed | MediaPrefillFailed => PrefillStep::Terminal("failed"),
    }
}

fn prefill_counts(data: &FactData) -> (Option<u64>, Option<u64>) {
    let computed = unsigned_summary(data, "computed_tokens").or_else(|| token_progress(data));
    (unsigned_summary(data, "cached_tokens"), computed)
}

pub(super) fn apply_prefill(
    state: &mut DomainState,
    scope: OperationScope,
    kind: PrefillEventKind,
    data: &FactData,
) {
    let Some(row) = existing_row(state, scope, data) else {
        return;
    };
    let prefill = row.prefill.get_or_insert_with(RequestPrefillState::default);
    if is_settled(prefill.phase, &PREFILL_TERMINAL_PHASES) {
        return;
    }
    let (cached, computed) = prefill_counts(data);
    match prefill_step(kind) {
        PrefillStep::Phase(phase) => {
            prefill.phase = Some(phase);
            raise(&mut prefill.cached_tokens, cached);
            raise(&mut prefill.computed_tokens, computed);
        }
        PrefillStep::CacheRestore(result) => prefill.cache_restore = Some(result),
        PrefillStep::Terminal(phase) => {
            prefill.phase = Some(phase);
            pin(&mut prefill.cached_tokens, cached);
            pin(&mut prefill.computed_tokens, computed);
            prefill.outcome = data.outcome.map(outcome_label);
        }
    }
}

const fn generation_terminal_phase(kind: GenerationEventKind) -> Option<&'static str> {
    match kind {
        GenerationEventKind::GenerationCompleted => Some("completed"),
        GenerationEventKind::GenerationCancelled => Some("cancelled"),
        GenerationEventKind::GenerationTimedOut => Some("timed_out"),
        GenerationEventKind::GenerationFailed => Some("failed"),
        GenerationEventKind::GenerationStarted
        | GenerationEventKind::FirstTokenProduced
        | GenerationEventKind::GenerationProgress
        | GenerationEventKind::StopConditionReached => None,
    }
}

fn generated_tokens(data: &FactData) -> Option<u64> {
    unsigned_summary(data, "generated_token_count").or_else(|| token_progress(data))
}

fn advance_generation(
    generation: &mut RequestGenerationState,
    kind: GenerationEventKind,
    data: &FactData,
) {
    match kind {
        GenerationEventKind::GenerationStarted => {
            generation.phase.get_or_insert("started");
        }
        GenerationEventKind::FirstTokenProduced => {
            generation.first_token = true;
            generation.phase = Some("streaming");
        }
        GenerationEventKind::GenerationProgress => generation.phase = Some("streaming"),
        GenerationEventKind::StopConditionReached => generation.stop_condition_reached = true,
        GenerationEventKind::GenerationCompleted
        | GenerationEventKind::GenerationCancelled
        | GenerationEventKind::GenerationTimedOut
        | GenerationEventKind::GenerationFailed => {}
    }
    raise(&mut generation.generated_tokens, generated_tokens(data));
}

pub(super) fn apply_generation(
    state: &mut DomainState,
    scope: OperationScope,
    kind: GenerationEventKind,
    data: &FactData,
) {
    let Some(row) = existing_row(state, scope, data) else {
        return;
    };
    let generation = row
        .generation
        .get_or_insert_with(RequestGenerationState::default);
    if is_settled(generation.phase, &GENERATION_TERMINAL_PHASES) {
        return;
    }
    match generation_terminal_phase(kind) {
        Some(phase) => {
            generation.phase = Some(phase);
            pin(&mut generation.generated_tokens, generated_tokens(data));
            generation.outcome = data.outcome.map(outcome_label);
        }
        None => advance_generation(generation, kind, data),
    }
}
