//! Lossless OpenAI projections at the HTTP/native-template compatibility edge.
use super::*;
use crate::generation::{
    GeneratedMessage, GenerationChoice, GenerationDelta, GenerationEvent, GenerationEventChoice,
    GenerationRequest, GenerationResponse,
};

impl From<ChatCompletionRequest> for GenerationRequest {
    fn from(value: ChatCompletionRequest) -> Self {
        Self {
            model: value.model,
            messages: value.messages,
            stream: value.stream,
            max_tokens: value.max_tokens,
            max_completion_tokens: value.max_completion_tokens,
            temperature: value.temperature,
            top_p: value.top_p,
            n: value.n,
            logprobs: value.logprobs,
            top_logprobs: value.top_logprobs,
            presence_penalty: value.presence_penalty,
            frequency_penalty: value.frequency_penalty,
            logit_bias: value.logit_bias,
            response_format: value.response_format,
            tools: value.tools,
            tool_choice: value.tool_choice,
            parallel_tool_calls: value.parallel_tool_calls,
            user: value.user,
            stop: value.stop,
            seed: value.seed,
            reasoning: value.reasoning,
            reasoning_effort: value.reasoning_effort,
            prompt_cache_key: value.prompt_cache_key,
            prompt_cache_retention: value.prompt_cache_retention,
            stream_options: value.stream_options,
            extra: value.extra,
        }
    }
}

impl From<GenerationRequest> for ChatCompletionRequest {
    fn from(value: GenerationRequest) -> Self {
        Self {
            model: value.model,
            messages: value.messages,
            stream: value.stream,
            max_tokens: value.max_tokens,
            max_completion_tokens: value.max_completion_tokens,
            temperature: value.temperature,
            top_p: value.top_p,
            n: value.n,
            logprobs: value.logprobs,
            top_logprobs: value.top_logprobs,
            presence_penalty: value.presence_penalty,
            frequency_penalty: value.frequency_penalty,
            logit_bias: value.logit_bias,
            response_format: value.response_format,
            tools: value.tools,
            tool_choice: value.tool_choice,
            parallel_tool_calls: value.parallel_tool_calls,
            user: value.user,
            stop: value.stop,
            seed: value.seed,
            reasoning: value.reasoning,
            reasoning_effort: value.reasoning_effort,
            prompt_cache_key: value.prompt_cache_key,
            prompt_cache_retention: value.prompt_cache_retention,
            stream_options: value.stream_options,
            extra: value.extra,
        }
    }
}

impl From<AssistantMessage> for GeneratedMessage {
    fn from(value: AssistantMessage) -> Self {
        Self {
            role: value.role,
            content: value.content,
            reasoning_content: value.reasoning_content,
            tool_calls: value.tool_calls,
        }
    }
}

impl From<GeneratedMessage> for AssistantMessage {
    fn from(value: GeneratedMessage) -> Self {
        Self {
            role: value.role,
            content: value.content,
            reasoning_content: value.reasoning_content,
            tool_calls: value.tool_calls,
        }
    }
}

impl From<ChatCompletionChoice> for GenerationChoice {
    fn from(value: ChatCompletionChoice) -> Self {
        Self {
            index: value.index,
            message: value.message.into(),
            logprobs: value.logprobs,
            finish_reason: value.finish_reason,
        }
    }
}

impl From<GenerationChoice> for ChatCompletionChoice {
    fn from(value: GenerationChoice) -> Self {
        Self {
            index: value.index,
            message: value.message.into(),
            logprobs: value.logprobs,
            finish_reason: value.finish_reason,
        }
    }
}

impl From<ChatCompletionResponse> for GenerationResponse {
    fn from(value: ChatCompletionResponse) -> Self {
        Self {
            id: value.id,
            created: value.created,
            model: value.model,
            choices: value.choices.into_iter().map(Into::into).collect(),
            usage: value.usage,
            timings: value.timings,
            capsule_marker: value.capsule_marker,
            exchange_id: value.exchange_id,
        }
    }
}

impl From<GenerationResponse> for ChatCompletionResponse {
    fn from(value: GenerationResponse) -> Self {
        Self {
            id: value.id,
            object: "chat.completion",
            created: value.created,
            model: value.model,
            choices: value.choices.into_iter().map(Into::into).collect(),
            usage: value.usage,
            timings: value.timings,
            capsule_marker: value.capsule_marker,
            exchange_id: value.exchange_id,
        }
    }
}

impl From<ChatCompletionDelta> for GenerationDelta {
    fn from(value: ChatCompletionDelta) -> Self {
        Self {
            role: value.role,
            content: value.content,
            reasoning_content: value.reasoning_content,
            tool_calls: value.tool_calls,
        }
    }
}

impl From<GenerationDelta> for ChatCompletionDelta {
    fn from(value: GenerationDelta) -> Self {
        Self {
            role: value.role,
            content: value.content,
            reasoning_content: value.reasoning_content,
            tool_calls: value.tool_calls,
        }
    }
}

impl From<ChatCompletionChunkChoice> for GenerationEventChoice {
    fn from(value: ChatCompletionChunkChoice) -> Self {
        Self {
            index: value.index,
            delta: value.delta.into(),
            logprobs: value.logprobs,
            finish_reason: value.finish_reason,
        }
    }
}

impl From<GenerationEventChoice> for ChatCompletionChunkChoice {
    fn from(value: GenerationEventChoice) -> Self {
        Self {
            index: value.index,
            delta: value.delta.into(),
            logprobs: value.logprobs,
            finish_reason: value.finish_reason,
        }
    }
}

impl From<ChatCompletionChunk> for GenerationEvent {
    fn from(value: ChatCompletionChunk) -> Self {
        Self {
            id: value.id,
            created: value.created,
            model: value.model,
            choices: value.choices.into_iter().map(Into::into).collect(),
            usage: value.usage,
            timings: value.timings,
        }
    }
}

impl From<GenerationEvent> for ChatCompletionChunk {
    fn from(value: GenerationEvent) -> Self {
        Self {
            id: value.id,
            object: "chat.completion.chunk",
            created: value.created,
            model: value.model,
            choices: value.choices.into_iter().map(Into::into).collect(),
            usage: value.usage,
            timings: value.timings,
        }
    }
}
