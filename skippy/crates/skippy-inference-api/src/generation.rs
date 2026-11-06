//! Protocol-neutral boundary between frontend adapters and generation backends.
//! Requests, results and events are owned here; adapters in `chat` and
//! `anthropic` preserve their respective wire contracts. SystemOne remains
//! a separate backend operation.

pub mod content;
mod request;
pub mod response;
pub(crate) mod validation;

pub use content::{
    GenerationMessage, MessageContent, MessageContentPart, ensure_tool_call_ids,
    message_content_to_text, messages_to_plain_prompt,
};
pub use request::GenerationRequest;
pub use response::{
    CapsuleMarker, GeneratedMessage, GenerationChoice, GenerationDelta, GenerationEvent,
    GenerationEventChoice, GenerationResponse, capsule_id_is_valid,
};

pub type GenerationEventStream = std::pin::Pin<
    Box<
        dyn futures_core::Stream<Item = crate::backend::InferenceResult<GenerationEvent>>
            + Send
            + 'static,
    >,
>;
