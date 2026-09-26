//! Shared wire adapter for OpenAI upstreams (local, mesh, plugin and MoA).
use super::protocol::AnthropicMessagesStreamEvent;
use super::translate::{
    MessagesStreamAssembler, message_start_event, messages_response_from_chat_response,
    translate_stream_error_body,
};
use crate::{
    ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionDelta, ChatCompletionResponse,
    FinishReason, OpenAiError, Usage,
};
use serde_json::{Value, json};

fn usage(value: &Value) -> Usage {
    let count = |key| value[key].as_u64().unwrap_or(0).min(u64::from(u32::MAX)) as u32;
    let mut usage = Usage::new(count("prompt_tokens"), count("completion_tokens"));
    if let Some(cached) = value["prompt_tokens_details"]["cached_tokens"].as_u64() {
        usage = usage.with_cached_tokens(cached.min(u64::from(u32::MAX)) as u32);
    }
    usage
}

pub fn translate_chat_value(value: &Value) -> Result<Value, OpenAiError> {
    if value.get("error").is_some() {
        let event = translate_stream_error_body(value);
        return serde_json::to_value(event)
            .map_err(|error| OpenAiError::internal(error.to_string()));
    }
    let choice = value["choices"]
        .as_array()
        .and_then(|choices| choices.first())
        .ok_or_else(|| OpenAiError::internal("upstream completion has no choices"))?;
    let finish: Option<FinishReason> = serde_json::from_value(choice["finish_reason"].clone())
        .map_err(|error| OpenAiError::internal(error.to_string()))?;
    let mut response = ChatCompletionResponse::new(
        value["model"].as_str().unwrap_or_default(),
        choice["message"]["content"].as_str().unwrap_or_default(),
        usage(&value["usage"]),
    );
    response.id = value["id"].as_str().unwrap_or("upstream").into();
    response.choices[0].finish_reason = finish;
    response.choices[0].message.tool_calls = choice["message"].get("tool_calls").cloned();
    serde_json::to_value(messages_response_from_chat_response(&response)?)
        .map_err(|error| OpenAiError::internal(error.to_string()))
}

#[derive(Default)]
pub struct MessagesWireStream {
    assembler: MessagesStreamAssembler,
    started: bool,
    ended: bool,
}

impl MessagesWireStream {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, data: &str) -> Result<Vec<AnthropicMessagesStreamEvent>, OpenAiError> {
        if self.ended {
            return Ok(Vec::new());
        }
        if data == "[DONE]" {
            self.ended = true;
            let mut events = Vec::new();
            if !self.started {
                events.push(message_start_event("upstream", ""));
                self.started = true;
            }
            events.extend(self.assembler.finish(None));
            return Ok(events);
        }
        let value: Value =
            serde_json::from_str(data).map_err(|error| OpenAiError::internal(error.to_string()))?;
        if value.get("error").is_some() {
            self.assembler.fail();
            self.ended = true;
            return Ok(vec![translate_stream_error_body(&value)]);
        }
        let mut events = Vec::new();
        let model = value["model"].as_str().unwrap_or_default();
        if !self.started {
            events.push(message_start_event(
                value["id"].as_str().unwrap_or("upstream"),
                model,
            ));
            self.started = true;
        }
        let mut chunk = ChatCompletionChunk::delta(model, "");
        chunk.choices.clear();
        if let Some(choices) = value["choices"].as_array() {
            for choice in choices {
                chunk.choices.push(ChatCompletionChunkChoice {
                    index: choice["index"].as_u64().unwrap_or(0) as u32,
                    delta: ChatCompletionDelta {
                        role: choice["delta"]["role"].as_str().map(|_| "assistant"),
                        content: choice["delta"]["content"].as_str().map(str::to_owned),
                        reasoning_content: None,
                        tool_calls: choice["delta"].get("tool_calls").cloned(),
                    },
                    finish_reason: serde_json::from_value(choice["finish_reason"].clone())
                        .map_err(|error| OpenAiError::internal(error.to_string()))?,
                    logprobs: None,
                });
            }
        }
        if value["usage"].is_object() {
            chunk.usage = Some(usage(&value["usage"]));
        }
        events.extend(self.assembler.absorb(&chunk));
        Ok(events)
    }
    pub fn truncated(&mut self) -> Vec<AnthropicMessagesStreamEvent> {
        if self.ended {
            return Vec::new();
        }
        self.ended = true;
        self.assembler.fail();
        vec![translate_stream_error_body(
            &json!({"error":{"message":"upstream stream ended before completion"}}),
        )]
    }
}

/// Adapt a completed chat response using the same stream state as live chunks.
pub fn completion_events(value: &Value) -> Result<Vec<AnthropicMessagesStreamEvent>, OpenAiError> {
    let mut stream = MessagesWireStream::new();
    if value.get("error").is_some() {
        return stream.push(&value.to_string());
    }
    let choice = &value["choices"][0];
    let mut delta = choice["message"].clone();
    if let Some(calls) = delta.get_mut("tool_calls").and_then(Value::as_array_mut) {
        for (index, call) in calls.iter_mut().enumerate() {
            call["index"] = json!(index);
        }
    }
    let chunk = json!({"id":value["id"], "model":value["model"], "usage":value["usage"],
        "choices":[{"index":0,"delta":delta,"finish_reason":choice["finish_reason"]}]});
    let mut events = stream.push(&chunk.to_string())?;
    events.extend(stream.push("[DONE]")?);
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_done_still_emits_a_complete_anthropic_stream() {
        let events = MessagesWireStream::new().push("[DONE]").unwrap();
        assert!(matches!(
            events.first(),
            Some(AnthropicMessagesStreamEvent::MessageStart { .. })
        ));
        assert!(matches!(
            events.last(),
            Some(AnthropicMessagesStreamEvent::MessageStop {})
        ));
    }
}
