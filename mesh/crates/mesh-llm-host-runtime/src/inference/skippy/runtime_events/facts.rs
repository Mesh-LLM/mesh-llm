//! Pure `RuntimeFact` construction for the generation-lifecycle adapter.
//!
//! Every function here is a pure projection: given inputs already flowing
//! through [`super::SkippyGenerationRuntimeEventAdapter`], build the fact the
//! observation backs. No reservation, engine, or I/O concerns live here --
//! see `mod.rs` for those. Kept deliberately privacy-first: nothing in this
//! module reads `generated_token_ids`, `prompt_token_ids`, or
//! `prompt_token_digest` into a fact.

use mesh_llm_runtime_event_contracts::{
    BoundedNumericSummaries, FactData, GenerationEventKind, GenerationFact, NumericSummary,
    NumericSummaryKey, NumericValue, Outcome, PrefillEventKind, PrefillFact, ReasonCode,
    RuntimeFact, SessionEventKind, SessionFact,
};
use skippy_serving::frontend::{GenerationReceipt, GenerationTermination};

pub(super) fn empty_data() -> FactData {
    FactData::default()
}

pub(super) fn generation_fact(kind: GenerationEventKind, data: FactData) -> RuntimeFact {
    RuntimeFact::Generation(GenerationFact::with_data(kind, data))
}

pub(super) fn prefill_fact(kind: PrefillEventKind, data: FactData) -> RuntimeFact {
    RuntimeFact::Prefill(PrefillFact::with_data(kind, data))
}

pub(super) fn session_fact(kind: SessionEventKind, data: FactData) -> RuntimeFact {
    RuntimeFact::Session(SessionFact::with_data(kind, data))
}

pub(super) fn synthetic_generation_terminal() -> RuntimeFact {
    generation_fact(
        GenerationEventKind::GenerationFailed,
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..empty_data()
        },
    )
}

pub(super) fn synthetic_prefill_terminal() -> RuntimeFact {
    prefill_fact(
        PrefillEventKind::PrefillFailed,
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..empty_data()
        },
    )
}

fn unsigned_summaries(pairs: &[(&str, u64)]) -> BoundedNumericSummaries {
    let summaries = pairs
        .iter()
        .filter_map(|(key, value)| {
            let key = NumericSummaryKey::new(key).ok()?;
            Some(NumericSummary::new(key, NumericValue::Unsigned(*value)))
        })
        .collect();
    BoundedNumericSummaries::new(summaries).unwrap_or_default()
}

fn token_count_summaries(pairs: &[(&str, usize)]) -> BoundedNumericSummaries {
    let converted: Vec<(&str, u64)> = pairs
        .iter()
        .map(|(key, count)| (*key, u64::try_from(*count).unwrap_or(u64::MAX)))
        .collect();
    unsigned_summaries(&converted)
}

fn micros_summary(key: &str, micros: u64) -> BoundedNumericSummaries {
    NumericSummaryKey::new(key)
        .ok()
        .and_then(|key| {
            BoundedNumericSummaries::new(vec![NumericSummary::new(
                key,
                NumericValue::Unsigned(micros),
            )])
            .ok()
        })
        .unwrap_or_default()
}

/// Bounded native decode timing for the terminal per-request summary. `None`
/// on a field means the generation could not support it (fewer than two
/// tokens, or a zero-length decode span), so the summary reports only what was
/// actually observed. The terminal fact is the durable carrier (#2199): the
/// client-side usage frame is best-effort, so decode timing must not depend on
/// it or on droppable per-token diagnostics.
#[derive(Clone, Copy, Default)]
pub(super) struct DecodeTiming {
    pub(super) elapsed_us: Option<u64>,
    pub(super) tokens_per_second_milli: Option<u64>,
}

impl DecodeTiming {
    /// Derives decode timing from the first and last token availability
    /// (microseconds since request start) and the generated token count: the
    /// decode span is first to last token, and the rate is the token
    /// intervals over that span.
    pub(super) fn from_token_span(
        request_to_first_token_us: Option<u64>,
        request_to_last_token_us: Option<u64>,
        generated_token_count: usize,
    ) -> Self {
        let (Some(first), Some(last)) = (request_to_first_token_us, request_to_last_token_us)
        else {
            return Self::default();
        };
        let intervals = u64::try_from(generated_token_count.saturating_sub(1)).unwrap_or(u64::MAX);
        if intervals == 0 {
            return Self::default();
        }
        let elapsed_us = last.saturating_sub(first);
        Self {
            elapsed_us: Some(elapsed_us),
            tokens_per_second_milli: (elapsed_us > 0)
                .then(|| intervals.saturating_mul(1_000_000_000) / elapsed_us),
        }
    }
}

/// Redacted, bounded projection of a [`GenerationReceipt`]: counts and
/// timing only. Never carries token IDs, digests, or prompt content.
struct RedactedReceiptSummary {
    generated_token_count: usize,
    prompt_token_count: usize,
    decode: DecodeTiming,
}

fn redact_receipt(receipt: &GenerationReceipt) -> RedactedReceiptSummary {
    RedactedReceiptSummary {
        generated_token_count: receipt.generated_token_ids.len(),
        prompt_token_count: receipt.prompt_token_count,
        decode: DecodeTiming::from_token_span(
            receipt.request_to_token_emission_us.first().copied(),
            receipt.request_to_token_emission_us.last().copied(),
            receipt.request_to_token_emission_us.len(),
        ),
    }
}

/// Maps a receipt or lifecycle-only completion's termination onto the
/// family's terminal kinds. `GenerationReceiptSink::record` is only reachable
/// on a successful local generation (`GenerationReceipt` is target-authoritative
/// success evidence); `Aborted` covers the failure/cancellation path separately,
/// while a lifecycle-only completion can carry cancellation directly. In both
/// cases `outcome` reflects the ACTUAL termination, not a blanket `Success`:
/// `GenerationReceiptSink::record` fires whenever local generation stops
/// producing a target-authoritative receipt, including a mid-stream
/// cancellation that still committed a partial suffix -- see
/// `GenerationTermination::Cancelled`'s doc ("The local generation loop
/// observed request cancellation"). A `GenerationCancelled` kind reporting
/// `Outcome::Success` would misclassify a cancelled request as successful
/// in every derived readiness/health projection; `Outcome::Cancelled`
/// matches the exact same disposition `abort_terminal_fact` already reports
/// for the `GenerationAbort` path, so the two cancellation paths agree.
pub(super) fn generation_completion_terminal_fact(
    termination: GenerationTermination,
    prompt_token_count: usize,
    generated_token_count: usize,
    decode: DecodeTiming,
) -> RuntimeFact {
    let (kind, outcome) = match termination {
        GenerationTermination::Cancelled => {
            (GenerationEventKind::GenerationCancelled, Outcome::Cancelled)
        }
        _ => (GenerationEventKind::GenerationCompleted, Outcome::Success),
    };
    let mut pairs: Vec<(&str, u64)> = vec![
        (
            "generated_token_count",
            u64::try_from(generated_token_count).unwrap_or(u64::MAX),
        ),
        (
            "prompt_token_count",
            u64::try_from(prompt_token_count).unwrap_or(u64::MAX),
        ),
    ];
    if let Some(elapsed_us) = decode.elapsed_us {
        pairs.push(("decode_elapsed_us", elapsed_us));
    }
    if let Some(milli) = decode.tokens_per_second_milli {
        pairs.push(("decode_tokens_per_second_milli", milli));
    }
    let data = FactData {
        outcome: Some(outcome),
        numeric_summaries: unsigned_summaries(&pairs),
        ..empty_data()
    };
    generation_fact(kind, data)
}

pub(super) fn receipt_terminal_fact(receipt: &GenerationReceipt) -> RuntimeFact {
    let redacted = redact_receipt(receipt);
    generation_completion_terminal_fact(
        receipt.termination,
        redacted.prompt_token_count,
        redacted.generated_token_count,
        redacted.decode,
    )
}

/// §8.10's `first token produced`, backed by the real existing signal this
/// adapter already observes: the FIRST [`skippy_serving::frontend::
/// GenerationCommit`] for a generation IS the moment its first token
/// became available. StateTransition-class, `unreserved_ingress`-safe.
pub(super) fn first_token_produced_fact() -> RuntimeFact {
    generation_fact(GenerationEventKind::FirstTokenProduced, empty_data())
}

/// §8.10's `stop condition reached`, backed by the real existing
/// [`GenerationTermination::CallbackStop`] variant: "the token callback
/// requested a stop, including for an end-of-generation token" IS the
/// existing signal that a stop CONDITION (as opposed to the max-token
/// budget) ended generation. Co-emitted alongside `GenerationCompleted`,
/// never in place of it. StateTransition-class, `unreserved_ingress`-safe.
pub(super) fn stop_condition_reached_fact() -> RuntimeFact {
    generation_fact(GenerationEventKind::StopConditionReached, empty_data())
}

pub(super) fn abort_terminal_fact() -> RuntimeFact {
    generation_fact(
        GenerationEventKind::GenerationCancelled,
        FactData {
            outcome: Some(Outcome::Cancelled),
            reason: Some(ReasonCode::Cancellation),
            ..empty_data()
        },
    )
}

pub(super) fn progress_fact(generated_token_count: usize) -> RuntimeFact {
    generation_fact(
        GenerationEventKind::GenerationProgress,
        FactData {
            numeric_summaries: token_count_summaries(&[(
                "generated_token_count",
                generated_token_count,
            )]),
            ..empty_data()
        },
    )
}

/// §8's prefill family, backed by the real existing
/// `GenerationReceipt::request_to_first_token_us` field or its
/// lifecycle-only equivalent: the target-authoritative
/// backend-request-start-to-first-token latency IS the existing signal for
/// "prefill completed" (decode begins at first token). A generation that never
/// emitted a token (`None`) reports the same successful terminal without the
/// timing summary rather than inventing a failure this layer cannot actually
/// distinguish from "zero tokens requested". Explicit cancellation before a
/// first token remains a cancelled prefill terminal; once first-token timing
/// exists, prefill has already completed even if decode is later cancelled.
pub(super) fn prefill_terminal_fact(receipt: &GenerationReceipt) -> RuntimeFact {
    prefill_completion_terminal_fact(receipt.termination, receipt.request_to_first_token_us)
}

pub(super) fn prefill_completion_terminal_fact(
    termination: GenerationTermination,
    request_to_first_token_us: Option<u64>,
) -> RuntimeFact {
    if matches!(termination, GenerationTermination::Cancelled)
        && request_to_first_token_us.is_none()
    {
        return prefill_cancelled_fact();
    }
    let data = match request_to_first_token_us {
        Some(micros) => FactData {
            outcome: Some(Outcome::Success),
            numeric_summaries: micros_summary("request_to_first_token_us", micros),
            ..empty_data()
        },
        None => FactData {
            outcome: Some(Outcome::Success),
            ..empty_data()
        },
    };
    prefill_fact(PrefillEventKind::PrefillCompleted, data)
}

pub(super) fn prefill_cancelled_fact() -> RuntimeFact {
    prefill_fact(
        PrefillEventKind::PrefillCancelled,
        FactData {
            outcome: Some(Outcome::Cancelled),
            reason: Some(ReasonCode::Cancellation),
            ..empty_data()
        },
    )
}

/// §9's session family. `SessionActive`/`SessionIdle` are StateTransition-
/// class (never Terminal), so they are safe to submit through
/// `unreserved_ingress` -- no reservation lifecycle applies. Backed by the
/// same generation-lifecycle observation stream this adapter already
/// receives: a session becomes active exactly when a generation starts on
/// it and idle exactly when that generation's terminal resolves. This
/// adapter never emits `SessionClosed`/`SessionFailed` (Terminal-class):
/// Skippy's generation lifecycle has no signal for "this KV session will
/// never be reused again", only "no generation is active on it right now".
pub(super) fn session_active_fact() -> RuntimeFact {
    session_fact(SessionEventKind::SessionActive, empty_data())
}

pub(super) fn session_idle_fact() -> RuntimeFact {
    session_fact(SessionEventKind::SessionIdle, empty_data())
}
