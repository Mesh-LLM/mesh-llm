use skippy_protocol::binary::StageSamplingConfig as WireSamplingConfig;

mod admission;
mod backend;
mod decode_batcher;
mod decode_scheduler;
mod embedded_execution;
mod embedded_generation;
mod generation;
mod generation_flow;
mod guardrails;
mod local_generation;
mod native_mtp;
mod prefill;
mod prefix_cache;
mod prompting;
mod request;
mod speculative;
mod tool_emulation;
mod util;

#[cfg(test)]
use prompting::parse_emulated_chat_output;
mod wire_messages;

use self::{
    decode_scheduler::*, native_mtp::*, request::*, speculative::*, util::*, wire_messages::*,
};

use self::generation::*;
pub use self::generation::{
    CONTEXT_BUDGET_MAX_TOKENS, DEFAULT_EMBEDDED_MAX_TOKENS, EmbeddedOpenAiArgs,
    EmbeddedOpenAiBackend, EmbeddedOpenAiRequestDefaults, EmbeddedOpenAiRouter,
    EmbeddedReasoningBudget, EmbeddedReasoningEnabled, EmbeddedReasoningFormat,
    embedded_openai_backend, embedded_openai_router, serve_embedded_openai,
    serve_embedded_openai_with_shutdown, serve_openai,
};
pub use self::guardrails::{
    OpenAiGuardrailsConfig, OpenAiGuardrailsStatus, OpenAiGuardrailsTarget,
};
pub use self::speculative::{
    NativeMtpProposalConfig, NgramExtensionConfig, NgramProposalConfig, NgramProposerKind,
    SpeculativeDecodeConfig, VerifyWindowConfig,
};

#[cfg(test)]
mod tests;
