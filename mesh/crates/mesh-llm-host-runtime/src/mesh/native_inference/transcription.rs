use super::*;
use crate::sdk::native::transcription::{RealtimeTranscriptionConfig, TranscriptionSession};

struct SessionAdmission {
    _slot: tokio::sync::OwnedSemaphorePermit,
    _instance: crate::runtime::InstanceRequestGuard,
    _inflight: InflightRequestGuard,
    _cancel: CancelNativeOnDrop,
}

impl Node {
    pub(crate) async fn native_live_transcription(
        &self,
        mut config: RealtimeTranscriptionConfig,
    ) -> Result<TranscriptionSession> {
        validate(&config)?;
        if let AdmissionResult::Paused { reason, .. } = self
            .activity_policy_guard
            .check_admission(IngressType::LocalOpenAi)
        {
            anyhow::bail!("inference paused: {reason}");
        }
        if let Some(manager) = self.plugin_manager().await {
            anyhow::ensure!(
                !manager.has_exchange_hooks().await,
                "live transcription is unavailable on a node with exchange admission plugins"
            );
        }
        let models = self.native_models().await?;
        let local: Vec<_> = models
            .into_iter()
            .filter(|model| {
                model.target.provider_node_id == self.endpoint.id().to_string()
                    && model
                        .descriptor
                        .metadata
                        .as_ref()
                        .and_then(|meta| meta.workload_class.as_ref())
                        == Some(&ModelWorkloadClass::SpeechRecognition)
            })
            .collect();
        let callable: Vec<_> = local
            .iter()
            .map(|model| model.target.serving_id.clone())
            .collect();
        config.model = if config.model.is_empty() {
            callable
                .first()
                .context("no local model supports live transcription")?
                .clone()
        } else {
            let requested = self
                .native_model_lifecycle
                .get()
                .map(|service| service.execution_model(&config.model))
                .unwrap_or(config.model);
            let descriptors = self.served_model_descriptors().await;
            let resolved = crate::network::openai::request_parse::internal_model_for_public_id(
                &requested,
                &callable,
                &descriptors,
            )
            .unwrap_or(requested);
            resolved.split('#').next().unwrap_or_default().to_string()
        };
        anyhow::ensure!(
            callable.contains(&config.model),
            "no local model supports live transcription for {:?}; remote models require an explicit network session",
            config.model
        );
        let slot = self
            .native_inference
            .stream_slots
            .clone()
            .try_acquire_owned()
            .context("native live transcription capacity exhausted")?;
        let candidates = self
            .native_inference
            .backends
            .read()
            .await
            .get(&config.model)
            .cloned()
            .context("native speech model is no longer loaded")?;
        for (id, candidate) in candidates {
            let Ok(Some(instance)) = self.begin_runtime_instance_request(candidate.port).await
            else {
                continue;
            };
            let current = self.native_inference.backends.read().await;
            if !current
                .get(&config.model)
                .and_then(|instances| instances.get(&id))
                .is_some_and(|active| Arc::ptr_eq(&active.backend, &candidate.backend))
            {
                continue;
            }
            drop(current);
            let context = InferenceRequestContext::new();
            let admission = SessionAdmission {
                _slot: slot,
                _instance: instance,
                _inflight: self.begin_inflight_request(),
                _cancel: CancelNativeOnDrop(context.cancellation_token()),
            };
            let session = candidate
                .backend
                .realtime_transcription(config, context.clone())
                .await
                .map_err(|error| anyhow::anyhow!("start native live transcription: {error}"))?;
            return Ok(TranscriptionSession::new(session, context, admission));
        }
        anyhow::bail!("native speech model is draining or unavailable")
    }
}

fn validate(config: &RealtimeTranscriptionConfig) -> Result<()> {
    anyhow::ensure!(
        (8_000..=96_000).contains(&config.input_sample_rate),
        "input sample rate must be 8000 to 96000 Hz"
    );
    if let Some(step) = config.step_ms {
        anyhow::ensure!((200..=5000).contains(&step), "step_ms must be 200 to 5000");
    }
    if let Some(vad) = config.turn_detection {
        anyhow::ensure!(
            (0.0..=1.0).contains(&vad.threshold),
            "turn detection threshold must be 0 to 1"
        );
        anyhow::ensure!(
            vad.prefix_padding_ms <= 5000,
            "prefix padding must be at most 5000 ms"
        );
        anyhow::ensure!(
            (100..=30_000).contains(&vad.silence_duration_ms),
            "silence duration must be 100 to 30000 ms"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{InstanceLifecycleRecord, InstanceLifecycleState};
    use crate::sdk::native::transcription::{RealtimeInput, RealtimeOutput};
    use async_trait::async_trait;
    use skippy_inference_api::generation::{
        GenerationEventStream, GenerationRequest, GenerationResponse,
    };
    use skippy_inference_api::{InferenceError, InferenceResult, ModelObject};
    use std::sync::Mutex;

    #[test]
    fn native_options_reject_the_same_out_of_range_audio_and_cadence_settings() {
        let base = RealtimeTranscriptionConfig::new("speech");
        for config in [
            RealtimeTranscriptionConfig {
                input_sample_rate: 0,
                ..base.clone()
            },
            RealtimeTranscriptionConfig {
                input_sample_rate: 96_001,
                ..base.clone()
            },
            RealtimeTranscriptionConfig {
                step_ms: Some(199),
                ..base.clone()
            },
            RealtimeTranscriptionConfig {
                turn_detection: Some(crate::sdk::native::transcription::RealtimeTurnDetection {
                    threshold: f32::NAN,
                    ..Default::default()
                }),
                ..base.clone()
            },
        ] {
            assert!(validate(&config).is_err());
        }
        assert!(
            validate(&RealtimeTranscriptionConfig {
                input_sample_rate: 48_000,
                step_ms: Some(400),
                language: Some("de".into()),
                prompt: Some("jargon".into()),
                turn_detection: None,
                tentative: true,
                ..base
            })
            .is_ok()
        );
    }
    #[derive(Default)]
    struct SpeechBackend {
        options: Mutex<Option<RealtimeTranscriptionConfig>>,
    }

    #[async_trait]
    impl InferenceBackend for SpeechBackend {
        async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
            Ok(vec![ModelObject::new("speech")])
        }
        async fn chat_completion(
            &self,
            _: GenerationRequest,
        ) -> InferenceResult<GenerationResponse> {
            Err(InferenceError::unsupported("speech only"))
        }
        async fn chat_completion_stream(
            &self,
            _: GenerationRequest,
            _: InferenceRequestContext,
        ) -> InferenceResult<GenerationEventStream> {
            Err(InferenceError::unsupported("speech only"))
        }
        async fn realtime_transcription(
            &self,
            config: RealtimeTranscriptionConfig,
            context: InferenceRequestContext,
        ) -> InferenceResult<skippy_inference_api::realtime::RealtimeTranscriptionSession> {
            *self.options.lock().unwrap() = Some(config);
            let (session, mut input, output) =
                skippy_inference_api::realtime::RealtimeTranscriptionSession::channels();
            tokio::spawn(async move {
                assert_eq!(input.recv().await, Some(RealtimeInput::Audio(vec![3, -4])));
                output
                    .send(RealtimeOutput::Delta {
                        turn: 0,
                        text: "hello".into(),
                    })
                    .await
                    .unwrap();
                assert_eq!(input.recv().await, Some(RealtimeInput::Commit));
                output
                    .send(RealtimeOutput::Completed {
                        turn: 0,
                        transcript: "hello world".into(),
                    })
                    .await
                    .unwrap();
                context.cancellation_token().cancelled().await;
            });
            Ok(session)
        }
    }

    #[tokio::test]
    async fn a_local_speech_backend_executes_without_any_http_server() {
        let node = Node::new_for_tests(NodeRole::Host { http_port: 9 })
            .await
            .unwrap();
        node.set_hosted_models(vec!["speech".into()]).await;
        node.set_serving_models(vec!["speech".into()]).await;
        node.upsert_served_model_descriptor(ServedModelDescriptor {
            identity: ServedModelIdentity {
                model_name: "speech".into(),
                is_primary: true,
                ..Default::default()
            },
            capabilities_known: true,
            metadata: Some(ServedModelMetadata {
                workload_class: Some(ModelWorkloadClass::SpeechRecognition),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await;
        let lifecycle = Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
            InstanceLifecycleState::Serving,
            1,
        )));
        node.register_runtime_instance_lifecycle(9, lifecycle);
        let backend = Arc::new(SpeechBackend::default());
        node.register_native_backend("speech", "speech-instance", 9, Some(backend.clone()), None)
            .await;
        let options = RealtimeTranscriptionConfig {
            model: "speech#fast".into(),
            input_sample_rate: 48_000,
            language: Some("de".into()),
            prompt: Some("jargon".into()),
            turn_detection: None,
            step_ms: Some(400),
            tentative: true,
        };
        let mut live = node
            .native_live_transcription(options.clone())
            .await
            .unwrap();
        let mut expected = options;
        expected.model = "speech".into();
        assert_eq!(*backend.options.lock().unwrap(), Some(expected));
        live.audio
            .send(RealtimeInput::Audio(vec![3, -4]))
            .await
            .unwrap();
        assert_eq!(
            live.events.next_output().await,
            Some(RealtimeOutput::Delta {
                turn: 0,
                text: "hello".into()
            })
        );
        live.audio.send(RealtimeInput::Commit).await.unwrap();
        assert_eq!(
            live.events.next_output().await,
            Some(RealtimeOutput::Completed {
                turn: 0,
                transcript: "hello world".into()
            })
        );
        drop(live);
        let error = node
            .native_live_transcription(RealtimeTranscriptionConfig::new("remote-only"))
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("no local model"));
        node.endpoint.close().await;
    }
}
