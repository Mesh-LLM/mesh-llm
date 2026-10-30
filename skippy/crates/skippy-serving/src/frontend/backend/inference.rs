//! Generation and other typed operations exposed by the serving backend.
use super::*;

#[async_trait]
impl InferenceBackend for StageOpenAiBackend {
    fn http_exchange_policy(
        &self,
    ) -> Option<Arc<dyn skippy_inference_api::http_exchange::HttpExchangePolicy>> {
        self.hook_policy
            .as_ref()
            .and_then(|hooks| hooks.http_exchange_policy())
    }
    async fn count_chat_tokens(&self, request: GenerationRequest) -> InferenceResult<u32> {
        self.count_prompt_tokens(request).await
    }

    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        Ok(vec![
            ModelObject::new(self.model_id.clone()).with_thinking(self.thinking.clone()),
        ])
    }

    /// Serve System One requests through the backend's selected execution path.
    async fn system_one(&self, request: SystemOneRequest) -> InferenceResult<SystemOneResponse> {
        self.serve_system_one(request).await
    }

    async fn chat_completion(
        &self,
        request: GenerationRequest,
    ) -> InferenceResult<GenerationResponse> {
        self.chat_completion_with_context(request, InferenceRequestContext::new())
            .await
    }

    async fn chat_completion_with_context(
        &self,
        request: GenerationRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<GenerationResponse> {
        let ids = generation_ids(
            InferenceCacheHints::from_chat_request(&request),
            request.agent_session(),
            &context,
        );
        let request_timer = PhaseTimer::start();
        self.chat_completion_with_prepared_hooks_for_exchange(
            request,
            None,
            move |mut request, admission| async move {
                self.ensure_model(&request.model)?;
                let (request_defaults, defaults_diagnostics) =
                    resolve_chat_request_defaults(&request, &self.request_defaults)?;
                apply_chat_request_defaults(&mut request, &request_defaults)?;
                ensure_chat_runtime_features_supported(&request)?;
                let sampling = chat_sampling_config(&request, &request_defaults)?
                    .with_penalty_windows_within(self.ctx_size);
                let template_options = chat_template_options(&request, &request_defaults)?;
                let parse_chat_output = chat_output_parser_required(&request, &template_options);
                let template_timer = PhaseTimer::start();
                let prompt = self
                    .prepare_chat_prompt_offloaded(&request, template_options.clone())
                    .await?;
                let mut template_attrs = self.openai_attrs(&ids);
                template_attrs.insert(
                    "llama_stage.openai_operation".to_string(),
                    json!("chat_completion"),
                );
                template_attrs.insert(
                    "llama_stage.chat_message_count".to_string(),
                    json!(request.messages.len()),
                );
                template_attrs.insert(
                    "llama_stage.prompt_chars".to_string(),
                    json!(prompt.text.len()),
                );
                template_attrs.insert(
                    "llama_stage.media_item_count".to_string(),
                    json!(prompt.media.len()),
                );
                template_attrs.insert(
                    "llama_stage.generation_profile".to_string(),
                    json!(defaults_diagnostics.selected_package_profile),
                );
                template_attrs.insert(
                    "llama_stage.max_tokens_source".to_string(),
                    json!(defaults_diagnostics.max_tokens_source),
                );
                template_attrs.insert(
                    "llama_stage.reasoning_budget_source".to_string(),
                    json!(defaults_diagnostics.reasoning_budget_source),
                );
                template_attrs.insert(
                    "llama_stage.generation_default_sources".to_string(),
                    json!(defaults_diagnostics.field_sources),
                );
                self.emit_openai_phase(
                    "stage.openai_chat_template",
                    template_timer,
                    template_attrs,
                );
                let max_tokens = GenerationTokenLimit::from_request(
                    request.effective_max_tokens(),
                    self.default_max_tokens,
                );
                let chat_parse_metadata = prompt.chat_parse_metadata.clone();
                admission.admit(&request).await?;
                let output = self
                    .run_generation(
                        prompt,
                        max_tokens,
                        request.stop.clone(),
                        sampling,
                        Some(request.clone()),
                        context,
                        ids.clone(),
                    )
                    .await?;
                let response_timer = PhaseTimer::start();
                let parsed_message = if parse_chat_output {
                    self.parse_chat_output(
                        &output.text,
                        &request,
                        chat_parse_metadata.as_deref(),
                        false,
                    )?
                } else {
                    None
                };
                crate::frontend::generation::tool_output_validation::validate(
                    &request,
                    output.finish_reason,
                    parsed_message.as_ref(),
                )?;
                let parsed_message = apply_reasoning_visibility(parsed_message, &template_options);
                let response = generation_response_from_generated_text(
                    request.model.clone(),
                    &output,
                    parsed_message,
                );
                let mut response_attrs = self.openai_attrs(&ids);
                response_attrs.insert(
                    "llama_stage.openai_operation".to_string(),
                    json!("chat_completion"),
                );
                response_attrs.insert(
                    "llama_stage.prompt_token_count".to_string(),
                    json!(output.prompt_tokens),
                );
                response_attrs.insert(
                    "llama_stage.completion_token_count".to_string(),
                    json!(output.completion_tokens),
                );
                self.emit_openai_phase(
                    "stage.openai_response_build",
                    response_timer,
                    response_attrs,
                );
                let mut summary_attrs = self.openai_attrs(&ids);
                summary_attrs.insert(
                    "llama_stage.openai_operation".to_string(),
                    json!("chat_completion"),
                );
                summary_attrs.insert("llama_stage.status".to_string(), json!("ok"));
                summary_attrs.insert(
                    "llama_stage.prompt_token_count".to_string(),
                    json!(output.prompt_tokens),
                );
                summary_attrs.insert(
                    "llama_stage.completion_token_count".to_string(),
                    json!(output.completion_tokens),
                );
                self.emit_openai_summary(
                    "stage.openai_request_summary",
                    request_timer,
                    summary_attrs,
                );
                Ok(response)
            },
        )
        .await
    }

    async fn chat_completion_stream(
        &self,
        request: GenerationRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<GenerationEventStream> {
        let ids = generation_ids(
            InferenceCacheHints::from_chat_request(&request),
            request.agent_session(),
            &context,
        );
        let exchange_context = context.clone();
        self.chat_completion_stream_with_prepared_hooks(
            request,
            &exchange_context,
            move |mut request, admission| async move {
                self.ensure_model(&request.model)?;
                let (request_defaults, defaults_diagnostics) =
                    resolve_chat_request_defaults(&request, &self.request_defaults)?;
                apply_chat_request_defaults(&mut request, &request_defaults)?;
                ensure_chat_runtime_features_supported(&request)?;
                let sampling = chat_sampling_config(&request, &request_defaults)?
                    .with_penalty_windows_within(self.ctx_size);
                let include_usage = request.include_usage();
                let template_options = chat_template_options(&request, &request_defaults)?;
                let parse_chat_output = chat_output_parser_required(&request, &template_options);
                let emit_reasoning = template_exposes_reasoning(&template_options);
                let template_timer = PhaseTimer::start();
                let prompt = self
                    .prepare_chat_prompt_offloaded(&request, template_options)
                    .await?;
                let mut template_attrs = self.openai_attrs(&ids);
                template_attrs.insert(
                    "llama_stage.openai_operation".to_string(),
                    json!("chat_completion_stream"),
                );
                template_attrs.insert(
                    "llama_stage.chat_message_count".to_string(),
                    json!(request.messages.len()),
                );
                template_attrs.insert(
                    "llama_stage.prompt_chars".to_string(),
                    json!(prompt.text.len()),
                );
                template_attrs.insert(
                    "llama_stage.media_item_count".to_string(),
                    json!(prompt.media.len()),
                );
                template_attrs.insert(
                    "llama_stage.generation_profile".to_string(),
                    json!(defaults_diagnostics.selected_package_profile),
                );
                template_attrs.insert(
                    "llama_stage.max_tokens_source".to_string(),
                    json!(defaults_diagnostics.max_tokens_source),
                );
                template_attrs.insert(
                    "llama_stage.reasoning_budget_source".to_string(),
                    json!(defaults_diagnostics.reasoning_budget_source),
                );
                template_attrs.insert(
                    "llama_stage.generation_default_sources".to_string(),
                    json!(defaults_diagnostics.field_sources),
                );
                self.emit_openai_phase(
                    "stage.openai_chat_template",
                    template_timer,
                    template_attrs,
                );
                let max_tokens = GenerationTokenLimit::from_request(
                    request.effective_max_tokens(),
                    self.default_max_tokens,
                );
                let model = request.model.clone();
                admission.admit(&request).await?;
                let stream = self
                    .run_generation_stream(
                        prompt,
                        max_tokens,
                        request.stop.clone(),
                        sampling,
                        include_usage,
                        Some(request),
                        parse_chat_output,
                        emit_reasoning,
                        context,
                        ids,
                    )
                    .await?;
                let stream: GenerationEventStream = Box::pin(
                    stream.map(move |event| generation_event_to_inference_event(event, &model)),
                );
                Ok(stream)
            },
        )
        .await
    }

    async fn completion(&self, request: CompletionRequest) -> InferenceResult<CompletionResponse> {
        self.completion_with_context(request, InferenceRequestContext::new())
            .await
    }

    async fn completion_with_context(
        &self,
        mut request: CompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<CompletionResponse> {
        let ids = generation_ids(
            InferenceCacheHints::from_completion_request(&request),
            request.agent_session(),
            &context,
        );
        let request_timer = PhaseTimer::start();
        self.ensure_model(&request.model)?;
        let (request_defaults, defaults_diagnostics) =
            resolve_completion_request_defaults(&request, &self.request_defaults);
        apply_completion_request_defaults(&mut request, &request_defaults);
        ensure_completion_runtime_features_supported(&request)?;
        let sampling =
            completion_sampling_config(&request)?.with_penalty_windows_within(self.ctx_size);
        let max_tokens =
            GenerationTokenLimit::from_request(request.max_tokens, self.default_max_tokens);
        let prompt_timer = PhaseTimer::start();
        let prompt = PreparedGenerationPrompt::text(request.prompt.text_lossy());
        let mut prompt_attrs = self.openai_attrs(&ids);
        prompt_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        prompt_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        prompt_attrs.insert(
            "llama_stage.generation_profile".to_string(),
            json!(defaults_diagnostics.selected_package_profile),
        );
        prompt_attrs.insert(
            "llama_stage.max_tokens_source".to_string(),
            json!(defaults_diagnostics.max_tokens_source),
        );
        prompt_attrs.insert(
            "llama_stage.generation_default_sources".to_string(),
            json!(defaults_diagnostics.field_sources),
        );
        self.emit_openai_phase("stage.openai_prompt_prepare", prompt_timer, prompt_attrs);
        let mut completion_guard = self.admit_prepared_completion(&request, &context).await?;
        let output = self
            .run_generation(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                None,
                context,
                ids.clone(),
            )
            .await
            .inspect_err(|error| {
                if let Some(guard) = completion_guard.as_mut() {
                    guard.finish(completion_exchange::error_outcome(error));
                }
            })?;
        if let Some(guard) = completion_guard.as_mut() {
            guard.finish("completed");
        }
        let response_timer = PhaseTimer::start();
        let response = completion_response_from_generated_text(request.model, &output);
        let mut response_attrs = self.openai_attrs(&ids);
        response_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        response_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        response_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_phase(
            "stage.openai_response_build",
            response_timer,
            response_attrs,
        );
        let mut summary_attrs = self.openai_attrs(&ids);
        summary_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        summary_attrs.insert("llama_stage.status".to_string(), json!("ok"));
        summary_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        summary_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_summary("stage.openai_request_summary", request_timer, summary_attrs);
        Ok(response)
    }

    async fn completion_stream(
        &self,
        mut request: CompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<CompletionStream> {
        let ids = generation_ids(
            InferenceCacheHints::from_completion_request(&request),
            request.agent_session(),
            &context,
        );
        self.ensure_model(&request.model)?;
        let (request_defaults, defaults_diagnostics) =
            resolve_completion_request_defaults(&request, &self.request_defaults);
        apply_completion_request_defaults(&mut request, &request_defaults);
        ensure_completion_runtime_features_supported(&request)?;
        let sampling =
            completion_sampling_config(&request)?.with_penalty_windows_within(self.ctx_size);
        let include_usage = request.include_usage();
        let max_tokens =
            GenerationTokenLimit::from_request(request.max_tokens, self.default_max_tokens);
        let model = request.model.clone();
        let prompt_timer = PhaseTimer::start();
        let prompt = PreparedGenerationPrompt::text(request.prompt.text_lossy());
        let mut prompt_attrs = self.openai_attrs(&ids);
        prompt_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion_stream"),
        );
        prompt_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        prompt_attrs.insert(
            "llama_stage.generation_profile".to_string(),
            json!(defaults_diagnostics.selected_package_profile),
        );
        prompt_attrs.insert(
            "llama_stage.max_tokens_source".to_string(),
            json!(defaults_diagnostics.max_tokens_source),
        );
        prompt_attrs.insert(
            "llama_stage.generation_default_sources".to_string(),
            json!(defaults_diagnostics.field_sources),
        );
        self.emit_openai_phase("stage.openai_prompt_prepare", prompt_timer, prompt_attrs);
        let mut completion_guard = self.admit_prepared_completion(&request, &context).await?;
        let stream = self
            .run_generation_stream(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                include_usage,
                None,
                false,
                false,
                context,
                ids,
            )
            .await
            .inspect_err(|error| {
                if let Some(guard) = completion_guard.as_mut() {
                    guard.finish(completion_exchange::error_outcome(error));
                }
            })?;
        Ok(completion_exchange::guarded(
            Box::pin(stream.map(move |event| generation_event_to_completion_chunk(event, &model))),
            completion_guard,
        ))
    }

    /// Execute embeddings only through an admitted local full-model workload.
    async fn embeddings(
        &self,
        request: EmbeddingsRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<EmbeddingResponse> {
        self.ensure_model(&request.model)?;
        let info = self.ensure_local_workload(ModelWorkload::Embedding)?;
        let expected_dimensions = non_chat::embedding_output_dimensions(info.output_dimensions)?;
        if request
            .dimensions
            .is_some_and(|requested| requested != expected_dimensions)
        {
            return Err(InferenceError::unsupported(format!(
                "model exposes {expected_dimensions} embedding dimensions; dimensionality reduction is not supported"
            )));
        }

        let model = request.model.clone();
        let encoding_format = request.encoding_format.clone();
        let backend = self.clone();
        let token_inputs = task::spawn_blocking(move || backend.prepare_embedding_inputs(request))
            .await
            .map_err(|error| {
                InferenceError::backend(format!("embedding tokenization task failed: {error}"))
            })??;
        let prompt_tokens = token_inputs.iter().map(Vec::len).sum::<usize>();
        let max_input_tokens = token_inputs.iter().map(Vec::len).max().unwrap_or_default();
        let ids = generation_ids(InferenceCacheHints::default(), None, &context);
        let cancellation = context.cancellation_token();
        let embeddings = self
            .run_local_workload(
                context,
                ids,
                max_input_tokens,
                move |runtime, session_id| {
                    non_chat::collect_workload_batch(
                        token_inputs.iter().enumerate(),
                        &cancellation,
                        |(index, tokens)| {
                            runtime
                                .embed(session_id, tokens, expected_dimensions)
                                .map(|values| Embedding { values, index })
                        },
                    )
                },
            )
            .await?;
        Ok(EmbeddingResponse::from_embeddings(
            model,
            embeddings,
            u32::try_from(prompt_tokens).unwrap_or(u32::MAX),
            &encoding_format,
        ))
    }

    /// Execute reranking only through an admitted local full-model workload.
    async fn rerank(
        &self,
        request: RerankRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<RerankResponse> {
        self.ensure_model(&request.model)?;
        self.ensure_local_workload(ModelWorkload::Rerank)?;
        let prompt_tokens_estimate = non_chat::rerank_prompt_tokens_estimate(&request)?;
        let ids = generation_ids(InferenceCacheHints::default(), None, &context);
        let cancellation = context.cancellation_token();
        let query = request.query.clone();
        let documents = request.documents.clone();
        let mut scored = self
            .run_local_workload(
                context,
                ids,
                prompt_tokens_estimate,
                move |runtime, session_id| {
                    non_chat::collect_workload_batch(
                        documents.iter().enumerate(),
                        &cancellation,
                        |(index, document)| {
                            let (relevance_score, token_count) =
                                runtime.rerank(session_id, &query, document.text()?)?;
                            Ok((index, relevance_score, token_count))
                        },
                    )
                },
            )
            .await?;
        let prompt_tokens = scored.iter().map(|(_, _, count)| count).sum::<usize>();
        scored.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        scored.truncate(request.top_n.unwrap_or(scored.len()).min(scored.len()));
        let results = scored
            .into_iter()
            .map(|(index, relevance_score, _)| RerankResult {
                index,
                relevance_score,
                document: request
                    .return_documents
                    .then(|| request.documents[index].clone()),
            })
            .collect();
        Ok(RerankResponse {
            id: format!("rerank-{}", uuid::Uuid::new_v4().simple()),
            results,
            usage: skippy_inference_api::Usage {
                prompt_tokens: u32::try_from(prompt_tokens).unwrap_or(u32::MAX),
                completion_tokens: 0,
                total_tokens: u32::try_from(prompt_tokens).unwrap_or(u32::MAX),
                ..skippy_inference_api::Usage::default()
            },
        })
    }

    /// Execute native speech synthesis with the configured local projector.
    async fn audio_speech(
        &self,
        request: AudioSpeechRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioResponse> {
        self.ensure_model(&request.model)?;
        if !self.has_unsplit_full_model_topology() {
            return Err(InferenceError::unsupported(
                "speech synthesis currently requires an unsplit local runtime",
            ));
        }
        if (request.speed - 1.0).abs() > f32::EPSILON {
            return Err(InferenceError::unsupported(
                "speech speed control is not supported by the native model",
            ));
        }
        non_chat::validate_speech_voice(&request.voice)?;
        let output_format = match request.response_format {
            AudioFormat::Wav => SpeechOutputFormat::Wav,
            AudioFormat::Pcm => SpeechOutputFormat::PcmS16Le,
            _ => {
                return Err(InferenceError::unsupported(
                    "native speech synthesis currently supports wav and pcm output",
                ));
            }
        };
        {
            let runtime = lock_runtime(&self.runtime);
            if runtime.input_activation_boundary().is_some()
                || runtime.output_activation_boundary().is_some()
                || !runtime.supports_speech_synthesis()
            {
                return Err(InferenceError::unsupported(
                    "model does not expose full-model speech synthesis",
                ));
            }
        }
        let content_type = request.response_format.content_type().to_string();
        let prompt_tokens_estimate = request.input.len().div_ceil(3).max(1);
        let ids = generation_ids(InferenceCacheHints::default(), None, &context);
        let cancellation = context.clone();
        let config = SpeechSynthesisConfig {
            prompt: request.input,
            language: None,
            top_k: 20,
            top_p: 0.8,
            seed: u32::MAX,
            output_format,
            max_frames: 512,
        };
        let audio = self
            .run_local_workload(
                context,
                ids,
                prompt_tokens_estimate,
                move |runtime, session_id| {
                    runtime.synthesize_speech(session_id, &config, || cancellation.is_cancelled())
                },
            )
            .await?;
        AudioResponse::new(audio.bytes, content_type)
    }

    /// Decode uploaded audio using the local speech-recognition runtime.
    async fn audio_transcription(
        &self,
        request: AudioTranscriptionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        self.audio_to_text(request, false, context).await
    }

    /// Translate uploaded audio using the local speech-recognition runtime.
    async fn audio_translation(
        &self,
        request: AudioTranscriptionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<AudioTranscriptionResponse> {
        self.audio_to_text(request, true, context).await
    }
}
