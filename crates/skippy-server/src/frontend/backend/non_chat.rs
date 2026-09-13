use super::*;

/// Reject an invalid model descriptor before allocating any embedding sessions.
pub(super) fn embedding_output_dimensions(dimensions: u32) -> OpenAiResult<usize> {
    if dimensions == 0 {
        return Err(OpenAiError::backend(
            "model did not report an embedding output dimension",
        ));
    }
    usize::try_from(dimensions)
        .map_err(|_| OpenAiError::backend("embedding dimensions exceed usize"))
}

/// Validate the same documents that execution consumes before estimating admission.
pub(super) fn rerank_prompt_tokens_estimate(request: &RerankRequest) -> OpenAiResult<usize> {
    request.validate()?;
    request
        .documents
        .iter()
        .map(|document| {
            document
                .text()
                .map(|text| request.query.len().saturating_add(text.len()).div_ceil(3))
        })
        .try_fold(1, |estimate, next| next.map(|tokens| estimate.max(tokens)))
}

/// Stop a native batch between items when its owning HTTP request is cancelled.
pub(super) fn collect_workload_batch<I, T, F>(
    items: I,
    cancellation: &openai_frontend::CancellationToken,
    mut run_item: F,
) -> anyhow::Result<Vec<T>>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> anyhow::Result<T>,
{
    let mut results = Vec::new();
    for item in items {
        // Native calls cannot be interrupted mid-item; stop before the next
        // one so a disconnected client does not hold the runtime for a batch.
        if cancellation.is_cancelled() {
            return Err(request_cancelled_error().into());
        }
        results.push(run_item(item)?);
    }
    Ok(results)
}

/// Preserve structured frontend errors while adding context to native failures.
fn workload_error(error: anyhow::Error) -> OpenAiError {
    if let Some(openai_error) = error.downcast_ref::<OpenAiError>() {
        return openai_error.clone();
    }
    OpenAiError::backend(format!("workload execution failed: {error:#}"))
}

/// Accept only the implemented default speaker; do not reinterpret voice as language.
pub(super) fn validate_speech_voice(voice: &str) -> OpenAiResult<()> {
    if voice == "default" {
        Ok(())
    } else {
        Err(OpenAiError::unsupported(
            "native speech synthesis currently supports only voice 'default'; speaker selection is not implemented",
        )
        .with_param("voice"))
    }
}

impl StageOpenAiBackend {
    /// Recognize complete local models in both standalone and embedded serving modes.
    pub(in crate::frontend) fn has_unsplit_full_model_topology(&self) -> bool {
        fn is_unsplit(config: &skippy_protocol::StageConfig) -> bool {
            config.stage_index == 0
                && config.layer_start == 0
                && config.layer_end > 0
                && !config.filter_tensors_on_load
                && config.upstream.is_none()
                && config.downstream.is_none()
        }

        // The mesh serves a complete local GGUF through EmbeddedStageZero as
        // well as through LocalRuntime. The mode name alone cannot tell us
        // whether model execution is distributed.
        is_unsplit(&self.config)
            && match &self.mode {
                OpenAiBackendMode::LocalRuntime => true,
                OpenAiBackendMode::EmbeddedStageZero { config, .. } => is_unsplit(config),
            }
    }

    /// Run audio transcription or English translation through a full-model projector.
    pub(super) async fn audio_to_text(
        &self,
        request: AudioTranscriptionRequest,
        translate_to_english: bool,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<AudioTranscriptionResponse> {
        self.ensure_model(&request.model)?;
        if !self.has_unsplit_full_model_topology() {
            return Err(OpenAiError::unsupported(
                "audio transcription currently requires an unsplit local runtime",
            ));
        }
        {
            let runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            if runtime.input_activation_boundary().is_some()
                || runtime.output_activation_boundary().is_some()
                || !runtime.has_media_projector()
            {
                return Err(OpenAiError::unsupported(
                    "model does not expose full-model multimodal audio input",
                ));
            }
        }

        let instruction = audio_text_instruction(&request, translate_to_english);
        let mut chat_request = ChatCompletionRequest {
            model: request.model.clone(),
            messages: vec![audio_text_user_message(instruction)],
            temperature: request.temperature,
            ..Default::default()
        };
        apply_chat_request_defaults(&mut chat_request, &self.request_defaults)?;
        let template_options = chat_template_options(&chat_request, &self.request_defaults)?;
        let mut prompt = self
            .prepare_chat_prompt_offloaded(&chat_request, template_options)
            .await?;
        prompt.media = vec![MediaInput {
            bytes: request.file,
        }];
        let sampling = sampling_config(
            chat_request.temperature,
            chat_request.top_p,
            chat_request.presence_penalty,
            chat_request.frequency_penalty,
            chat_request.seed,
            chat_request.logit_bias.as_ref(),
            &chat_request.extra,
        )?;
        let ids = generation_ids(OpenAiCacheHints::default(), None, &context);
        let output = self
            .run_generation(
                prompt,
                GenerationTokenLimit::from_request(None, self.default_max_tokens),
                None,
                sampling,
                None,
                context,
                ids,
            )
            .await?;
        Ok(AudioTranscriptionResponse {
            text: audio_transcript_text(&output.text),
        })
    }

    /// Reject split execution and mismatched native workload descriptors before work.
    pub(in crate::frontend) fn ensure_local_workload(
        &self,
        expected: ModelWorkload,
    ) -> OpenAiResult<WorkloadInfo> {
        if !self.has_unsplit_full_model_topology() {
            return Err(OpenAiError::unsupported(
                "non-chat workloads currently require an unsplit local runtime",
            ));
        }
        let runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        if runtime.input_activation_boundary().is_some()
            || runtime.output_activation_boundary().is_some()
        {
            return Err(OpenAiError::unsupported(
                "non-chat workloads currently require an unsplit full model",
            ));
        }
        let info = runtime
            .workload_info()
            .map_err(|error| OpenAiError::backend(format!("read model workload: {error:#}")))?;
        if info.kind != expected {
            return Err(OpenAiError::unsupported(format!(
                "model workload is {:?}; endpoint requires {:?}",
                info.kind, expected
            )));
        }
        Ok(info)
    }

    /// Tokenize text batches with the loaded vocabulary, preserving supplied token IDs.
    pub(super) fn prepare_embedding_inputs(
        &self,
        request: EmbeddingsRequest,
    ) -> OpenAiResult<Vec<Vec<i32>>> {
        let reader = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?
            .model
            .reader();
        match request.input {
            EmbeddingInput::Text(text) => reader
                .tokenize(&text, true)
                .map(|tokens| vec![tokens])
                .map_err(|error| OpenAiError::backend(format!("tokenize embedding: {error:#}"))),
            EmbeddingInput::Texts(texts) => texts
                .into_iter()
                .map(|text| {
                    reader.tokenize(&text, true).map_err(|error| {
                        OpenAiError::backend(format!("tokenize embedding: {error:#}"))
                    })
                })
                .collect(),
            EmbeddingInput::Tokens(tokens) => Ok(vec![tokens]),
            EmbeddingInput::TokenArrays(tokens) => Ok(tokens),
        }
    }

    /// Execute admitted blocking work with cancellation, slot ownership, and cleanup.
    pub(super) async fn run_local_workload<T, F>(
        &self,
        context: OpenAiRequestContext,
        ids: OpenAiGenerationIds,
        prompt_tokens: usize,
        work: F,
    ) -> OpenAiResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut crate::runtime_state::RuntimeState, &str) -> anyhow::Result<T>
            + Send
            + 'static,
    {
        let cancellation = context.cancellation_token();
        let (permit, session_permit) = self
            .acquire_generation_admission(
                &ids,
                &cancellation,
                GenerationAdmissionWork::new(prompt_tokens.max(1), 0),
                GenerationAdmissionScheduling::default(),
            )
            .await?;
        let runtime = Arc::clone(&self.runtime);
        let session_id = ids.session_id_string();
        let worker_context = context.clone();
        let result = run_blocking_generation_worker(permit, context, move |token| {
            let _session_permit = session_permit;
            if token.is_cancelled() {
                return Err(request_cancelled_error());
            }
            let mut runtime = runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            let result = work(&mut runtime, &session_id).map_err(workload_error);
            let cleanup = runtime.drop_session_timed(&session_id).map_err(|error| {
                OpenAiError::backend(format!("workload session cleanup failed: {error:#}"))
            });
            match (result, cleanup) {
                (Ok(value), Ok(_)) => Ok(value),
                (Err(error), _) => Err(error),
                (Ok(_), Err(error)) => Err(error),
            }
        })
        .await
        .map_err(|error| OpenAiError::backend(format!("workload task failed: {error}")))?;
        if worker_context.is_cancelled() {
            Err(request_cancelled_error())
        } else {
            result
        }
    }
}

fn audio_transcript_text(raw: &str) -> String {
    let text = raw.trim();
    for prefix in ["The text is:", "The audio is:"] {
        if let Some(quoted) = text.strip_prefix(prefix).map(str::trim)
            && let Some(inner) = quoted
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
        {
            return inner.to_string();
        }
    }
    text.to_string()
}

fn audio_text_instruction(
    request: &AudioTranscriptionRequest,
    translate_to_english: bool,
) -> String {
    let mut instruction = if translate_to_english {
        "Translate the supplied audio into English. Return only the English translation."
            .to_string()
    } else {
        let language = request
            .language
            .as_deref()
            .filter(|language| !language.trim().is_empty())
            .map_or_else(String::new, |language| format!(" (language: {language})"));
        format!("Transcribe audio to text{language}")
    };
    if let Some(prompt) = request
        .prompt
        .as_deref()
        .filter(|prompt| !prompt.trim().is_empty())
    {
        instruction.push_str(" Use this context when resolving names or terminology: ");
        instruction.push_str(prompt);
    }
    instruction
}

fn audio_text_user_message(instruction: String) -> openai_frontend::ChatMessage {
    openai_frontend::ChatMessage {
        role: "user".to_string(),
        // The audio marker must follow the instruction. Upstream llama.cpp's
        // transcription route appends the media marker to the user prompt;
        // reversing these parts can make Ultravox ignore the supplied audio.
        content: Some(openai_frontend::MessageContent::Parts(vec![
            openai_frontend::MessageContentPart {
                content_type: "text".to_string(),
                text: Some(instruction),
                extra: BTreeMap::new(),
            },
            openai_frontend::MessageContentPart {
                content_type: "input_audio".to_string(),
                text: None,
                extra: BTreeMap::from([(
                    "input_audio".to_string(),
                    json!({"data": "AA==", "format": "wav"}),
                )]),
            },
        ])),
        extra: BTreeMap::new(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn invalid_embedding_dimensions_are_rejected_at_admission() {
        assert!(
            super::embedding_output_dimensions(0)
                .unwrap_err()
                .body()
                .error
                .message
                .contains("did not report")
        );
        assert_eq!(super::embedding_output_dimensions(768).unwrap(), 768);
    }

    #[test]
    fn rerank_estimate_rejects_invalid_documents_before_workload_admission() {
        let mut request: openai_frontend::RerankRequest =
            serde_json::from_value(serde_json::json!({
                "model": "rank", "query": "query", "documents": ["text", {"title": "no text"}]
            }))
            .unwrap();
        assert!(super::rerank_prompt_tokens_estimate(&request).is_err());
        request.documents.pop();
        assert_eq!(super::rerank_prompt_tokens_estimate(&request).unwrap(), 3);
    }

    use super::{
        audio_text_instruction, audio_text_user_message, audio_transcript_text,
        collect_workload_batch, validate_speech_voice, workload_error,
    };
    use openai_frontend::{AudioTranscriptionRequest, MessageContent};

    fn audio_request() -> AudioTranscriptionRequest {
        AudioTranscriptionRequest {
            model: "ultravox".to_string(),
            file: vec![1],
            filename: Some("sample.wav".to_string()),
            language: None,
            prompt: None,
            response_format: "json".to_string(),
            temperature: Some(0.0),
        }
    }

    #[test]
    fn transcription_uses_upstream_default_prompt_before_audio_marker() {
        let instruction = audio_text_instruction(&audio_request(), false);
        assert_eq!(instruction, "Transcribe audio to text");

        let message = audio_text_user_message(instruction);
        let Some(MessageContent::Parts(parts)) = message.content else {
            panic!("audio request must use multipart chat content");
        };
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].content_type, "text");
        assert_eq!(parts[0].text.as_deref(), Some("Transcribe audio to text"));
        assert_eq!(parts[1].content_type, "input_audio");
        assert!(parts[1].extra.contains_key("input_audio"));
    }

    #[test]
    fn transcription_removes_only_confirmed_quoted_ultravox_wrappers() {
        assert_eq!(
            audio_transcript_text("The text is: \"The mesh is ready\""),
            "The mesh is ready"
        );
        assert_eq!(
            audio_transcript_text("The audio is: \"The mesh is ready\""),
            "The mesh is ready"
        );
        assert_eq!(
            audio_transcript_text("  The mesh is ready  "),
            "The mesh is ready"
        );
        assert_eq!(
            audio_transcript_text("The text is: The mesh is ready"),
            "The text is: The mesh is ready"
        );
        assert_eq!(
            audio_transcript_text("I heard: \"The mesh is ready\""),
            "I heard: \"The mesh is ready\""
        );
    }

    #[test]
    fn transcription_preserves_empty_text_without_an_artificial_wrapper() {
        assert_eq!(audio_transcript_text("   "), "");
        assert_eq!(audio_transcript_text("The text is: \"\""), "");
        assert_eq!(audio_transcript_text("The audio is: \"\""), "");
    }

    #[test]
    fn translation_and_explicit_context_keep_audio_marker_last() {
        let mut request = audio_request();
        request.language = Some("German".to_string());
        request.prompt = Some("mesh-llm".to_string());
        assert_eq!(
            audio_text_instruction(&request, false),
            "Transcribe audio to text (language: German) Use this context when resolving names or terminology: mesh-llm"
        );

        let instruction = audio_text_instruction(&request, true);
        assert!(instruction.starts_with("Translate the supplied audio into English."));
        let message = audio_text_user_message(instruction);
        let Some(MessageContent::Parts(parts)) = message.content else {
            panic!("audio translation must use multipart chat content");
        };
        assert_eq!(parts[0].content_type, "text");
        assert_eq!(parts[1].content_type, "input_audio");
    }

    #[test]
    fn speech_voice_rejects_unsupported_speakers_without_reinterpreting_language() {
        validate_speech_voice("default").expect("the default native speaker is supported");

        for voice in ["alloy", "english", ""] {
            let error = validate_speech_voice(voice).expect_err("speaker selection is unsupported");
            assert_eq!(
                error.body().error.code.as_deref(),
                Some("unsupported_model_feature")
            );
            assert_eq!(error.body().error.param.as_deref(), Some("voice"));
            assert!(error.body().error.message.contains("voice 'default'"));
        }
    }

    #[test]
    fn workload_batch_stops_before_the_next_native_call_after_cancellation() {
        let cancellation = openai_frontend::CancellationToken::new();
        let mut calls = 0;
        let result = collect_workload_batch(0..3, &cancellation, |item| {
            calls += 1;
            cancellation.cancel();
            Ok(item)
        });
        let error = result.expect_err("the remaining batch must stop after cancellation");
        assert_eq!(calls, 1);
        assert_eq!(
            workload_error(error).body().error.code.as_deref(),
            Some("request_cancelled")
        );
    }

    #[test]
    fn workload_error_keeps_structured_cancellation() {
        let error = workload_error(anyhow::Error::new(super::request_cancelled_error()));
        assert_eq!(
            error.body().error.code.as_deref(),
            Some("request_cancelled")
        );
    }
}
