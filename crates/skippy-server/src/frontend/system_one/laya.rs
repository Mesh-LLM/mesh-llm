//! A System One backend for a model that serves nothing else, such as Laya.

use std::sync::Arc;

use async_trait::async_trait;
use openai_frontend::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream, ModelObject,
    OpenAiBackend, OpenAiError, OpenAiRequestContext, OpenAiResult, SystemOneRequest,
    SystemOneResponse,
};
use skippy_runtime::LayaModel;
use tokio::task;

use super::run_on_model;

/// Serves `POST /systemone` from a loaded Laya model. Every other OpenAI
/// surface is refused: Laya generates no text.
#[derive(Clone)]
pub struct LayaSystemOneBackend {
    model_id: String,
    model: Arc<LayaModel>,
}

impl LayaSystemOneBackend {
    pub fn new(model_id: impl Into<String>, model: Arc<LayaModel>) -> Self {
        Self {
            model_id: model_id.into(),
            model,
        }
    }
}

#[async_trait]
impl OpenAiBackend for LayaSystemOneBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new(self.model_id.clone())])
    }

    async fn system_one(&self, request: SystemOneRequest) -> OpenAiResult<SystemOneResponse> {
        let backend = self.clone();
        task::spawn_blocking(move || {
            run_on_model(backend.model.as_ref(), &backend.model_id, request)
        })
        .await
        .map_err(|error| {
            OpenAiError::backend(format!("System One execution task failed: {error}"))
        })?
    }

    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        Err(decision_only())
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        Err(decision_only())
    }
}

fn decision_only() -> OpenAiError {
    OpenAiError::unsupported("this Laya decision model only serves POST /systemone")
}
