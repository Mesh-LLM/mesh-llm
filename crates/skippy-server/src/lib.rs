//! skippy-server library interface.
//!
//! Exposes the stage serving loop for in-process embedding by mesh-llm
//! or other host runtimes.

pub mod binary_transport;
pub mod cli;
pub mod config;
pub mod embedded;
pub mod frontend;
pub mod http;
pub mod kv_integration;
pub mod kv_proto;
pub mod package;
pub mod runtime_state;

#[cfg(test)]
mod legacy_scheduler_absence_tests {
    use std::path::Path;

    #[test]
    fn removed_serving_scheduler_modules_cannot_reappear() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for removed in [
            "decode_batch_policy.rs",
            "frontend/decode_batcher.rs",
            "binary_transport/decode_batcher.rs",
        ] {
            assert!(
                !source.join(removed).exists(),
                "legacy serving scheduler module reappeared: {removed}"
            );
        }
    }
}
pub mod serving_hooks;
mod stage_performance;
pub mod telemetry;
pub use stage_performance::{StageDecodeTimingHint, stage_decode_timing_hints};
pub mod tokenizer;

// Re-export key types for consumers
pub use binary_transport::serve_binary;
pub use cli::ServeBinaryArgs;
pub use embedded::{
    EmbeddedRuntimeOptions, EmbeddedRuntimeStatus, EmbeddedServerHandle, EmbeddedServerStatus,
    EmbeddedState, SkippyRuntimeHandle, start_binary_stage, start_embedded_openai,
    start_openai_backend, start_openai_backend_with_lifecycle_observer,
    start_openai_backend_with_tokenizer,
    start_openai_backend_with_tokenizer_and_lifecycle_observer, start_stage_http,
};
pub use frontend::{
    CONTEXT_BUDGET_MAX_TOKENS, DECODE_BATCH_HEADROOM_TOKENS, DEFAULT_EMBEDDED_MAX_TOKENS,
    EmbeddedOpenAiArgs, EmbeddedOpenAiBackend, EmbeddedOpenAiRequestDefaults,
    EmbeddedReasoningBudget, EmbeddedReasoningEnabled, EmbeddedReasoningFormat, LinearProposal,
    LinearProposalDiscardReason, LinearProposalDisposition, LinearProposalIngress,
    LinearProposalQuery, LinearProposalReceipt, LinearProposalSourceOutcome,
    LinearProposalSourceResponse, LinearProposalSourceTelemetry, NativeMtpProposalConfig,
    NgramExtensionConfig, NgramProposalConfig, NgramProposerKind, OpaqueProposalDecisionId,
    OpenAiGuardrailsConfig, OpenAiGuardrailsStatus, OpenAiGuardrailsTarget,
    SpeculativeDecodeConfig, VerifyWindowConfig, embedded_openai_backend,
};
pub use skippy_protocol::StageConfig;
pub use tokenizer::{MAX_TOKENIZE_TOKENS, TokenizerCapability, TokenizerCapabilityError};
