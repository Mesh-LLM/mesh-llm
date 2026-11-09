//! Per-generation performance figures for an embedder to observe.
//!
//! The OpenAI frontend already measures each generation for its telemetry
//! spans, but those are level-gated and shaped for an OTLP collector. This is
//! the same measurement handed to one process-wide observer as a plain value,
//! so a host can aggregate or report real serving behavior (prefill, decode,
//! speculation, lanes) without parsing span attributes.
//!
//! Only counts and timings cross this boundary. Prompt text, generated text,
//! and token IDs never do.

use std::sync::{Arc, OnceLock};

/// One finished text generation, as measured by the frontend that served it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationStats {
    /// The served model identifier. The embedder decides whether it may be
    /// reported anywhere; it can be a name derived from a local file.
    pub model_id: String,
    /// Whether this frontend is stage 0 of a split rather than a whole model.
    pub split: bool,
    /// Persistent downstream lanes to the next stage. `None` when local.
    pub downstream_lanes: Option<usize>,
    /// Generations this frontend admits at once at the time of completion.
    pub concurrency_limit: usize,
    /// Configured context window, in tokens.
    pub context_size: usize,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Prompt tokens served from the KV cache rather than prefilled.
    pub cached_prompt_tokens: u32,
    /// Prompt tokens actually prefilled for this request.
    pub prefill_tokens: u32,
    /// How the prompt cache hit, or `none`.
    pub cache_hit_kind: &'static str,
    pub queue_wait_ms: f64,
    /// Time spent restoring cached state before prefill.
    pub restore_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    /// Whole generation, admission through the last token.
    pub total_ms: f64,
    pub finish_reason: &'static str,
    /// The resolved speculation strategy, e.g. `dflash` or `ngram-suffix`.
    pub speculative_strategy: String,
    /// Whether the speculation gate let this request speculate.
    pub speculation_allowed: bool,
    /// Draft-model and N-gram verify windows.
    pub draft_windows: u64,
    pub draft_proposed_tokens: u64,
    pub draft_accepted_tokens: u64,
    /// Native MTP and split DFlash drafts, which stage 0 verifies alike.
    pub mtp_verifications: u64,
    pub mtp_proposed_tokens: u64,
    pub mtp_accepted_tokens: u64,
    /// Activation codec on the first split hop, when split.
    pub activation_codec: Option<&'static str>,
    pub activation_prefill_bytes: u64,
    pub activation_decode_bytes: u64,
}

/// Receives every finished generation. Called on the generating thread, so
/// an implementation must not block: hand the value off and return.
pub trait GenerationStatsObserver: Send + Sync {
    fn observe(&self, stats: &GenerationStats);
}

static OBSERVER: OnceLock<Arc<dyn GenerationStatsObserver>> = OnceLock::new();

/// Install the process-wide observer. Returns `false` if one is already set,
/// in which case the first stays in place.
pub fn set_generation_stats_observer(observer: Arc<dyn GenerationStatsObserver>) -> bool {
    OBSERVER.set(observer).is_ok()
}

pub(crate) fn observer_installed() -> bool {
    OBSERVER.get().is_some()
}

pub(crate) fn notify(stats: &GenerationStats) {
    if let Some(observer) = OBSERVER.get() {
        observer.observe(stats);
    }
}
