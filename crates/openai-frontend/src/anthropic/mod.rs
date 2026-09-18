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
//! - `POST /v1/messages/count_tokens` returns a token estimate for the
//!   request's prompt. The frontend has no tokenizer handle, so the estimate
//!   uses the same plain-text approximation as the OpenAI path.
//! - Errors use the Anthropic envelope (`{"type":"error","error":{...}}`).

mod protocol;
mod routes;
mod translate;

pub use protocol::{
    AnthropicContentBlock, AnthropicError, AnthropicErrorMessage, AnthropicMessagesRequest,
    AnthropicMessagesResponse, AnthropicMessagesStreamEvent, AnthropicSystemPrompt, AnthropicTool,
    AnthropicToolChoice, AnthropicToolDefinition, AnthropicToolResultContent,
    AnthropicToolUseContent, AnthropicUsage,
};
pub(crate) use routes::{messages, messages_count_tokens};
pub use translate::{
    messages_request_to_chat_request, messages_response_from_chat_response,
    translate_stream_error_body,
};
