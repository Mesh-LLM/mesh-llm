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
pub mod tokenizer;

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
    EmbeddedReasoningEnabled, EmbeddedReasoningFormat, GenerationReceipt, GenerationReceiptConfig,
    GenerationReceiptSink, GenerationStateDigest, GenerationTermination, LinearProposal,
    LinearProposalDiscardReason, LinearProposalDisposition, LinearProposalIngress,
    LinearProposalIngressConfig, LinearProposalQuery, LinearProposalReceipt,
    NativeMtpProposalConfig, NgramExtensionConfig, NgramProposalConfig, NgramProposerKind,
    OpaqueProposalDecisionId, OpenAiGuardrailsConfig, OpenAiGuardrailsStatus,
    OpenAiGuardrailsTarget, SpeculativeDecodeConfig, VerifyWindowConfig, embedded_openai_backend,
    generation_token_id_digest,
};
pub use skippy_protocol::StageConfig;
pub use tokenizer::{MAX_TOKENIZE_TOKENS, TokenizerCapability, TokenizerCapabilityError};
