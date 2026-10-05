use std::time::Instant;

use skippy_inference_api::{ChatCompletionRequest, InferenceError, InferenceResult};
use skippy_runtime::{ModelWorkload, SamplingConfig};

use crate::frontend::generation::{
    GenerationCacheStats, OpenAiGenerationIds, StageOpenAiBackend, TokenControl,
    tool_calls_requested,
};
use crate::frontend::util::openai_backend_error;

use super::LocalSessionCleanupGuard;

impl StageOpenAiBackend {
    #[allow(clippy::too_many_arguments)]
    /// Encode once, decode bounded tokens, and clean up the exclusive session on every exit.
    pub(super) fn generate_encoder_decoder_tokens(
        &self,
        prompt_token_ids: &[i32],
        max_tokens: u32,
        sampling: &SamplingConfig,
        chat_request: Option<&ChatCompletionRequest>,
        cancellation: Option<&skippy_inference_api::CancellationToken>,
        ids: &OpenAiGenerationIds,
        mut emit_token: impl FnMut(i32) -> InferenceResult<TokenControl>,
    ) -> InferenceResult<GenerationCacheStats> {
        if !self.has_unsplit_full_model_topology() {
            return Err(InferenceError::unsupported(
                "encoder-decoder models currently require an unsplit local runtime",
            ));
        }
        self.ensure_local_workload(ModelWorkload::EncoderDecoder)?;
        if chat_request.is_some_and(tool_calls_requested) {
            return Err(InferenceError::unsupported(
                "tool calls are not supported by encoder-decoder models",
            ));
        }
        let session_id = ids.session_label.clone();
        let (result, mut cleanup) = LocalSessionCleanupGuard::run(
            || self.cleanup_local_generation_session(&session_id, ids),
            || {
                let mut runtime = self
                    .runtime
                    .lock()
                    .map_err(|_| InferenceError::backend("runtime lock poisoned"))?;
                let prompt_started = Instant::now();
                let decoder_start = runtime
                    .encode_prompt(&session_id, prompt_token_ids)
                    .map_err(openai_backend_error)?;
                let prompt_ms = prompt_started.elapsed().as_secs_f64() * 1_000.0;
                let predicted_started = Instant::now();
                if max_tokens > 0 {
                    let mut predicted = runtime
                        .decode_frame_sampled(
                            &session_id,
                            decoder_start,
                            sampling.enabled.then_some(sampling),
                            None,
                            0,
                        )
                        .map_err(openai_backend_error)?
                        .0;
                    for generated in 0..max_tokens {
                        if cancellation
                            .is_some_and(skippy_inference_api::CancellationToken::is_cancelled)
                        {
                            return Err(InferenceError::backend("request cancelled"));
                        }
                        if emit_token(predicted)? == TokenControl::Stop {
                            break;
                        }
                        if generated + 1 < max_tokens {
                            predicted = runtime
                                .decode_frame_sampled(
                                    &session_id,
                                    predicted,
                                    sampling.enabled.then_some(sampling),
                                    None,
                                    0,
                                )
                                .map_err(openai_backend_error)?
                                .0;
                        }
                    }
                }
                Ok(GenerationCacheStats {
                    prompt_ms,
                    predicted_ms: predicted_started.elapsed().as_secs_f64() * 1_000.0,
                    ..GenerationCacheStats::default()
                })
            },
        );
        cleanup.cleanup();
        result
    }
}
