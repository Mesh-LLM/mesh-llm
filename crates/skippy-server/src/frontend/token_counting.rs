//! Token counting shares the generation prompt renderer and loaded tokenizer.
use super::generation::StageOpenAiBackend;
use super::request::{
    apply_chat_request_defaults, chat_template_options, ensure_chat_runtime_features_supported,
};
use openai_frontend::{ChatCompletionRequest, OpenAiError, OpenAiResult};

impl StageOpenAiBackend {
    pub(super) async fn count_prompt_tokens(
        &self,
        mut request: ChatCompletionRequest,
    ) -> OpenAiResult<u32> {
        self.ensure_model(&request.model)?;
        apply_chat_request_defaults(&mut request, &self.request_defaults)?;
        ensure_chat_runtime_features_supported(&request)?;
        let options = chat_template_options(&request, &self.request_defaults)?;
        let prompt = self
            .prepare_chat_prompt_offloaded(&request, options)
            .await?;
        if !prompt.media.is_empty() {
            return Err(OpenAiError::unsupported(
                "media token counting is unavailable",
            ));
        }
        let backend = self.clone();
        tokio::task::spawn_blocking(move || {
            let tokens = backend.tokenize(&prompt.text)?;
            u32::try_from(tokens.len()).map_err(|_| OpenAiError::internal("token count overflow"))
        })
        .await
        .map_err(|error| OpenAiError::backend(error.to_string()))?
    }
}
