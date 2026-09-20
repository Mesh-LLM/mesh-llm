//! Anthropic Messages API (`/v1/messages`) frontend.
//!
//! Translates the Anthropic Messages wire protocol onto the same
//! [`crate::backend::OpenAiBackend`] pipeline that serves the OpenAI-compatible
//! routes, so both protocols share generation, tool plumbing, mesh routing,
//! and MoA behavior. Claude Code speaks this protocol natively; serving it
//! directly removes the OpenAI bridge hop for Anthropic-protocol clients.
//!
//! Scope notes:
//! - One `POST /v1/messages` handler covers streaming (Anthropic SSE events)
//!   and non-streaming responses, selected by the body's `stream` field.
//! - `POST /v1/messages/count_tokens` uses the selected backend tokenizer
//!   and chat template. Backends without counting support return an explicit error.
//! - Errors use the Anthropic envelope (`{"type":"error","error":{...}}`).

mod normalize;
mod protocol;
mod routes;
mod translate;
mod wire;

pub use protocol::{
    AnthropicContentBlock, AnthropicError, AnthropicErrorMessage, AnthropicMessagesRequest,
    AnthropicMessagesResponse, AnthropicMessagesStreamEvent, AnthropicSystemPrompt, AnthropicTool,
    AnthropicToolChoice, AnthropicToolDefinition, AnthropicToolResultContent,
    AnthropicToolUseContent, AnthropicUsage,
};
pub(crate) use routes::{AnthropicRejection, messages, messages_count_tokens};
pub use translate::{
    messages_request_to_chat_request, messages_response_from_chat_response,
    translate_stream_error_body,
};

pub use normalize::normalize_messages_request;

pub use wire::{MessagesWireStream, completion_events, translate_chat_value};

mod validation;
