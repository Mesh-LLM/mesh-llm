//! Token counting shares the generation prompt renderer and loaded tokenizer.
use super::generation::StageOpenAiBackend;
use super::request::{
    apply_chat_request_defaults, chat_template_options, ensure_chat_runtime_features_supported,
};
use skippy_inference_api::{ChatCompletionRequest, OpenAiError, OpenAiResult};

impl StageOpenAiBackend {
    pub(super) async fn count_prompt_tokens(
        &self,
        mut request: ChatCompletionRequest,
    ) -> OpenAiResult<u32> {
        self.ensure_model(&request.model)?;
        apply_chat_request_defaults(&mut request, &self.request_defaults)?;
        ensure_chat_runtime_features_supported(&request)?;
        let options = chat_template_options(&request, &self.request_defaults)?;
        let admission = self.acquire_token_count_admission().await?;
        let backend = self.clone();
        // Keep admission inside the blocking task: dropping the HTTP future
        // must not free capacity while native preparation is still running.
        tokio::task::spawn_blocking(move || {
            let _admission = admission;
            let prompt = backend.prepare_chat_prompt(&request, options)?;
            if !prompt.media.is_empty() {
                return Err(OpenAiError::unsupported(
                    "media token counting is unavailable",
                ));
            }
            let tokens = backend.tokenize(&prompt.text)?;
            u32::try_from(tokens.len()).map_err(|_| OpenAiError::internal("token count overflow"))
        })
        .await
        .map_err(|error| OpenAiError::backend(error.to_string()))?
    }
}
