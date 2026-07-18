mod cache_hints;
mod draft_runner;
mod parsing;
mod persistent_lanes;
mod queue;
mod server;
mod streaming;
mod types;

pub use cache_hints::{CONTEXT_BUDGET_MAX_TOKENS, DEFAULT_EMBEDDED_MAX_TOKENS};
pub use server::{
    EmbeddedOpenAiArgs, EmbeddedOpenAiBackend, EmbeddedOpenAiRequestDefaults, EmbeddedOpenAiRouter,
    EmbeddedReasoningBudget, EmbeddedReasoningEnabled, EmbeddedReasoningFormat,
};
pub use server::{
    embedded_openai_backend, embedded_openai_router, serve_embedded_openai,
    serve_embedded_openai_with_shutdown, serve_openai,
};

pub(in crate::frontend) use cache_hints::{
    ChainPrefixRestore, GENERATION_ADMISSION_TIMEOUT, GENERATION_RETRY_AFTER_SECS,
    GenerationCacheStats, MAX_EXACT_REPLAY_TOKENS, OpenAiCacheHints, OpenAiGenerationIds,
};
pub(in crate::frontend) use draft_runner::*;
pub(in crate::frontend) use parsing::*;
pub(in crate::frontend) use persistent_lanes::*;
pub(in crate::frontend) use queue::*;
pub(in crate::frontend) use streaming::*;
pub(in crate::frontend) use types::*;
