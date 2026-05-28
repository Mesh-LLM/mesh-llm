pub mod backend;
pub mod chat;
pub mod common;
pub mod completions;
pub mod errors;
mod guardrails;
pub mod hooks;
pub mod models;
pub mod responses;
pub mod router;
pub mod sse;

pub use backend::{
    CancellationToken, ChatCompletionStream, CompletionStream, OpenAiBackend, OpenAiRequestContext,
    OpenAiResult,
};
pub use chat::{
    AssistantMessage, ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
    ChatCompletionDelta, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    MessageContent, MessageContentPart, message_content_to_text, messages_to_plain_prompt,
};
pub use common::{
    FinishReason, PromptCacheRetention, ReasoningConfig, ReasoningEffort, ReasoningTemplateOptions,
    StopSequence, StreamOptions, THINKING_BOOLEAN_ALIASES, Usage, completion_id,
    normalize_reasoning_template_options, now_unix_secs,
};
pub use completions::{
    CompletionChoice, CompletionChunk, CompletionChunkChoice, CompletionPrompt, CompletionRequest,
    CompletionResponse,
};
pub use errors::{OpenAiError, OpenAiErrorKind, already_openai_error, map_upstream_error_body};
pub use guardrails::{
    CompactingOpenAiBackend, CompactionConfig, CompactionDecision, CompactionOverride,
    CompactionReport, GuardedOpenAiBackend, GuardrailMode, GuardrailPolicy, GuardrailPolicyHandle,
    GuardrailTelemetrySink, MESH_COMPACT_FIELD, MESH_RESPOND_TOOL_NAME, StreamingGuardrailMode,
};
pub use hooks::{
    ChatHookAction, ChatHookOutcome, ChatMediaKind, ChatMediaRef, GenerationHookSignals,
    HookedOpenAiBackend, MESH_HOOKS_FIELD, OpenAiHookPolicy, PrefillHookSignals,
    apply_chat_hook_outcome, chat_mesh_hooks_enabled, first_chat_media,
    inject_text_into_chat_messages, set_chat_mesh_hooks_enabled,
};
pub use models::{ModelId, ModelIdError, ModelObject, ModelsResponse};
pub use responses::{
    NormalizationOutcome, ResponseAdapterMode, ResponsesRequest, StreamUsage,
    chat_usage_to_responses_usage, normalize_openai_compat_request, parse_chat_stream_chunk,
    responses_stream_completed_event, responses_stream_completed_event_with_sequence,
    responses_stream_content_part_added_event, responses_stream_content_part_done_event,
    responses_stream_created_event, responses_stream_created_event_with_sequence,
    responses_stream_delta_event, responses_stream_delta_event_with_logprobs,
    responses_stream_delta_event_with_logprobs_and_sequence,
    responses_stream_output_item_added_event, responses_stream_output_item_done_event,
    responses_stream_reasoning_delta_event_with_sequence, responses_stream_text_done_event,
    responses_stream_text_done_event_with_sequence, stream_usage_to_responses_usage,
    translate_chat_completion_response_to_responses, translate_chat_completion_to_responses,
};
pub use router::{
    OpenAiFrontendConfig, router, router_for, router_for_with_config, router_with_config,
};
