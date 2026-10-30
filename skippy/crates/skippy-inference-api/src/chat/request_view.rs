//! OpenAI serialization at the plugin observation boundary.

use crate::generation::GenerationRequest;
use serde::{Serialize, Serializer, ser::SerializeMap};

/// An OpenAI wire projection of a backend request, without taking ownership.
/// The backend type itself deliberately remains independent of serialization.
pub struct ChatCompletionRequestRef<'a>(pub &'a GenerationRequest);

impl Serialize for ChatCompletionRequestRef<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Exhaustive destructuring keeps this projection in step with the
        // backend contract. Preserve ChatCompletionRequest's field order,
        // explicit nulls and flattened extensions for plugin wire digests.
        let GenerationRequest {
            model,
            messages,
            stream,
            max_tokens,
            max_completion_tokens,
            temperature,
            top_p,
            n,
            logprobs,
            top_logprobs,
            presence_penalty,
            frequency_penalty,
            logit_bias,
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls,
            user,
            stop,
            seed,
            reasoning,
            reasoning_effort,
            prompt_cache_key,
            prompt_cache_retention,
            stream_options,
            extra,
        } = self.0;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("model", model)?;
        map.serialize_entry("messages", messages)?;
        map.serialize_entry("stream", stream)?;
        map.serialize_entry("max_tokens", max_tokens)?;
        map.serialize_entry("max_completion_tokens", max_completion_tokens)?;
        map.serialize_entry("temperature", temperature)?;
        map.serialize_entry("top_p", top_p)?;
        map.serialize_entry("n", n)?;
        map.serialize_entry("logprobs", logprobs)?;
        map.serialize_entry("top_logprobs", top_logprobs)?;
        map.serialize_entry("presence_penalty", presence_penalty)?;
        map.serialize_entry("frequency_penalty", frequency_penalty)?;
        map.serialize_entry("logit_bias", logit_bias)?;
        map.serialize_entry("response_format", response_format)?;
        map.serialize_entry("tools", tools)?;
        map.serialize_entry("tool_choice", tool_choice)?;
        map.serialize_entry("parallel_tool_calls", parallel_tool_calls)?;
        map.serialize_entry("user", user)?;
        map.serialize_entry("stop", stop)?;
        map.serialize_entry("seed", seed)?;
        map.serialize_entry("reasoning", reasoning)?;
        map.serialize_entry("reasoning_effort", reasoning_effort)?;
        map.serialize_entry("prompt_cache_key", prompt_cache_key)?;
        map.serialize_entry("prompt_cache_retention", prompt_cache_retention)?;
        map.serialize_entry("stream_options", stream_options)?;
        for (key, value) in extra {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}
