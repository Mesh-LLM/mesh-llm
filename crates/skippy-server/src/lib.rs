//! skippy-server library interface.
//!
//! Exposes the stage serving loop for in-process embedding by mesh-llm
//! or other host runtimes.

pub mod binary_transport;
pub mod cli;
pub mod config;
mod decode_batch_policy;
pub mod embedded;
pub mod frontend;
pub mod http;
pub mod kv_integration;
pub mod kv_proto;
pub mod package;
pub mod runtime_state;
pub mod telemetry;

// Re-export key types for consumers
pub use binary_transport::serve_binary;
pub use cli::ServeBinaryArgs;
pub use embedded::{
    EmbeddedRuntimeOptions, EmbeddedRuntimeStatus, EmbeddedServerHandle, EmbeddedServerStatus,
    EmbeddedState, SkippyRuntimeHandle, start_binary_stage, start_embedded_openai,
    start_openai_backend, start_stage_http,
};
pub use frontend::{
    CONTEXT_BUDGET_MAX_TOKENS, DEFAULT_EMBEDDED_MAX_TOKENS, EmbeddedOpenAiArgs,
    EmbeddedOpenAiBackend, EmbeddedOpenAiRequestDefaults, EmbeddedReasoningBudget,
    EmbeddedReasoningEnabled, EmbeddedReasoningFormat, OpenAiGuardrailsConfig,
    OpenAiGuardrailsStatus, OpenAiGuardrailsTarget, embedded_openai_backend,
};
pub use skippy_protocol::StageConfig;
