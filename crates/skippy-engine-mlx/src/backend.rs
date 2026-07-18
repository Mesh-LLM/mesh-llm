//! `OpenAiBackend` implementation driving the MLX worker.
//!
//! This is the adapter between mesh-llm's real OpenAI-compatible frontend and
//! the MLX engine: it converts `ChatCompletionRequest`s into engine jobs and
//! turns the worker's `TokenMsg` stream into OpenAI chat responses / SSE chunks.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use openai_frontend::backend::{
    ChatCompletionStream, OpenAiBackend, OpenAiRequestContext, OpenAiResult,
};
use openai_frontend::chat::{
    AssistantMessage, ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
    ChatCompletionDelta, ChatCompletionRequest, ChatCompletionResponse, message_content_to_text,
};
use openai_frontend::common::{FinishReason, Usage, completion_id};
use openai_frontend::errors::OpenAiError;
use openai_frontend::models::ModelObject;

use crate::distributed::MlxDistributedEngine;
use crate::engine::{ChatTurn, FinishReason as EngineFinish, GenerateRequest, MlxEngine, TokenMsg};

enum BackendEngine {
    Local(MlxEngine),
    Distributed(MlxDistributedEngine),
}

impl BackendEngine {
    fn model_id(&self) -> &str {
        match self {
            Self::Local(engine) => engine.model_id(),
            Self::Distributed(engine) => engine.model_id(),
        }
    }

    fn clamp_max_tokens(&self, requested: Option<usize>) -> usize {
        match self {
            Self::Local(engine) => engine.clamp_max_tokens(requested),
            Self::Distributed(engine) => engine.clamp_max_tokens(requested),
        }
    }

    fn submit(&self, request: GenerateRequest) -> tokio::sync::mpsc::UnboundedReceiver<TokenMsg> {
        match self {
            Self::Local(engine) => engine.submit(request),
            Self::Distributed(engine) => engine.submit(request),
        }
    }
}

pub struct MlxBackend {
    engine: Arc<BackendEngine>,
}

impl MlxBackend {
    pub fn new(engine: MlxEngine) -> Self {
        Self {
            engine: Arc::new(BackendEngine::Local(engine)),
        }
    }

    pub fn new_distributed(engine: MlxDistributedEngine) -> Self {
        Self {
            engine: Arc::new(BackendEngine::Distributed(engine)),
        }
    }

    fn build_request(&self, request: &ChatCompletionRequest) -> OpenAiResult<GenerateRequest> {
        let messages: Vec<ChatTurn> = request
            .messages
            .iter()
            .map(|m| ChatTurn {
                role: m.role.clone(),
                content: m
                    .content
                    .as_ref()
                    .and_then(message_content_to_text)
                    .unwrap_or_default(),
            })
            .collect();
        if messages.is_empty() {
            return Err(OpenAiError::invalid_request("no messages in request"));
        }
        let requested = request
            .max_completion_tokens
            .or(request.max_tokens)
            .map(|n| n as usize);
        Ok(GenerateRequest {
            messages,
            raw_prompt: None,
            max_tokens: self.engine.clamp_max_tokens(requested),
        })
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn map_finish(reason: EngineFinish) -> FinishReason {
    match reason {
        EngineFinish::Stop => FinishReason::Stop,
        EngineFinish::Length => FinishReason::Length,
    }
}

fn role_chunk(id: &str, created: u64, model: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionDelta {
                role: Some("assistant"),
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: None,
        }],
        usage: None,
    }
}

fn content_chunk(id: &str, created: u64, model: &str, text: String) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionDelta {
                role: None,
                content: Some(text),
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: None,
        }],
        usage: None,
    }
}

fn final_chunk(
    id: &str,
    created: u64,
    model: &str,
    finish: FinishReason,
    usage: Usage,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: Some(finish),
        }],
        usage: Some(usage),
    }
}

fn usage(prompt_tokens: u32, completion_tokens: u32) -> Usage {
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        prompt_tokens_details: None,
    }
}

#[async_trait]
impl OpenAiBackend for MlxBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject {
            id: self.engine.model_id().to_string(),
            object: "model",
            created: now_secs(),
            owned_by: "mesh-llm-mlx".to_string(),
        }])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        let job = self.build_request(&request)?;
        let model = self.engine.model_id().to_string();
        let mut rx = self.engine.submit(job);

        let mut text = String::new();
        let mut finish = FinishReason::Stop;
        let mut used = usage(0, 0);

        while let Some(msg) = rx.recv().await {
            match msg {
                TokenMsg::Delta(delta) => text.push_str(&delta),
                TokenMsg::Done {
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                } => {
                    finish = map_finish(finish_reason);
                    used = usage(prompt_tokens, completion_tokens);
                    break;
                }
                TokenMsg::Error(e) => return Err(OpenAiError::internal(e)),
            }
        }

        Ok(ChatCompletionResponse {
            id: completion_id("chatcmpl"),
            object: "chat.completion",
            created: now_secs(),
            model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content: Some(text),
                    reasoning_content: None,
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: Some(finish),
            }],
            usage: used,
            timings: None,
        })
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        let job = self.build_request(&request)?;
        let model = self.engine.model_id().to_string();
        let mut rx = self.engine.submit(job);
        let id = completion_id("chatcmpl");
        let created = now_secs();

        let stream = async_stream::stream! {
            yield Ok(role_chunk(&id, created, &model));
            while let Some(msg) = rx.recv().await {
                match msg {
                    TokenMsg::Delta(delta) => {
                        yield Ok(content_chunk(&id, created, &model, delta));
                    }
                    TokenMsg::Done { finish_reason, prompt_tokens, completion_tokens } => {
                        let used = usage(prompt_tokens, completion_tokens);
                        yield Ok(final_chunk(&id, created, &model, map_finish(finish_reason), used));
                        break;
                    }
                    TokenMsg::Error(e) => {
                        yield Err(OpenAiError::internal(e));
                        break;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}
