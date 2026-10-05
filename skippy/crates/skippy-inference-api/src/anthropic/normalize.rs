//! Protocol normalization shared by embedded serving and raw host ingress.
use super::{AnthropicMessagesRequest, messages_request_to_chat_request};
use crate::errors::OpenAiError;
use serde_json::Value;

pub fn normalize_messages_request(body: &mut Value) -> Result<(), OpenAiError> {
    let request: AnthropicMessagesRequest =
        serde_json::from_value(body.clone()).map_err(|error| {
            OpenAiError::invalid_request(format!("invalid Messages request: {error}"))
        })?;
    let chat = messages_request_to_chat_request(request)?;
    chat.validate()?;
    *body = serde_json::to_value(chat).map_err(|error| OpenAiError::internal(error.to_string()))?;
    // Null optional fields need not cross internal hops.
    if let Some(object) = body.as_object_mut() {
        object.retain(|_, value| !value.is_null());
    }
    Ok(())
}
