use super::*;

impl StageOpenAiBackend {
    pub(in crate::frontend) async fn acquire_token_count_admission(
        &self,
    ) -> OpenAiResult<impl Send + use<>> {
        let ids =
            OpenAiGenerationIds::new_with_trust(OpenAiCacheHints::default(), None, false, None);
        self.acquire_generation_admission(
            &ids,
            &skippy_inference_api::CancellationToken::new(),
            GenerationAdmissionWork::default(),
            GenerationAdmissionScheduling::default(),
        )
        .await
    }
}
