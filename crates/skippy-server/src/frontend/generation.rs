mod cache_hints;
mod draft_runner;
mod incremental_text;
mod parsing;
mod persistent_lanes;
mod queue;
mod server;
mod streaming;
mod timeouts;
mod tool_call_stream;
mod types;

pub use cache_hints::{CONTEXT_BUDGET_MAX_TOKENS, DEFAULT_EMBEDDED_MAX_TOKENS};
pub(crate) use server::serve_embedded_openai_with_scheduler;
pub use server::{
    EmbeddedOpenAiArgs, EmbeddedOpenAiBackend, EmbeddedOpenAiRequestDefaults, EmbeddedOpenAiRouter,
    EmbeddedReasoningBudget, EmbeddedReasoningEnabled, EmbeddedReasoningFormat,
};
pub use server::{
    embedded_openai_backend, embedded_openai_router, serve_embedded_openai,
    serve_embedded_openai_with_shutdown, serve_openai,
};

pub(in crate::frontend) use cache_hints::{
    ChainPrefixRestore, GENERATION_RETRY_AFTER_SECS, GENERATION_TOKEN_BUDGET_TIMEOUT,
    GenerationCacheStats, MAX_EXACT_REPLAY_TOKENS, OpenAiCacheHints, OpenAiGenerationIds,
};
pub(in crate::frontend) use draft_runner::*;
#[cfg(test)]
pub(in crate::frontend) use incremental_text::recorded_fixture;
pub(in crate::frontend) use parsing::*;
pub(in crate::frontend) use persistent_lanes::*;
pub(in crate::frontend) use queue::*;
pub(in crate::frontend) use streaming::*;
pub(in crate::frontend) use timeouts::*;
pub(in crate::frontend) use types::*;

pub(crate) fn default_generation_queue_capacity(generation_concurrency: usize) -> usize {
    generation_concurrency.saturating_mul(8).clamp(16, 256)
}

#[cfg(test)]
mod admission_defaults_tests {
    use super::default_generation_queue_capacity;

    #[test]
    fn queue_capacity_tracks_lane_waves_with_bounds() {
        assert_eq!(default_generation_queue_capacity(1), 16);
        assert_eq!(default_generation_queue_capacity(2), 16);
        assert_eq!(default_generation_queue_capacity(4), 32);
        assert_eq!(default_generation_queue_capacity(32), 256);
        assert_eq!(default_generation_queue_capacity(usize::MAX), 256);
    }
}
