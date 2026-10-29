use super::*;
use crate::runtime::{InstanceLifecycleRecord, InstanceLifecycleState};
use async_trait::async_trait;
use futures_util::StreamExt;
use skippy_inference_api::common::ReasoningEffort;
use skippy_inference_api::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream, InferenceResult,
    ModelObject, Usage,
};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Instant;

#[test]
fn native_chat_requires_peer_admission_under_every_trust_policy() {
    for policy in [
        crate::crypto::TrustPolicy::Off,
        crate::crypto::TrustPolicy::PreferOwned,
        crate::crypto::TrustPolicy::RequireOwned,
        crate::crypto::TrustPolicy::Allowlist,
    ] {
        assert!(!stream_allowed_before_admission(STREAM_NATIVE_CHAT, policy));
    }
}

#[derive(Default)]
struct RecordingBackend {
    requests: Mutex<Vec<ChatCompletionRequest>>,
    stream_tokens: Mutex<Vec<skippy_inference_api::CancellationToken>>,
    generated_chunks: Arc<AtomicUsize>,
    stream_forever: bool,
    pacing: Option<Pacing>,
}

/// Shapes a test stream: either slow enough that the whole generation outlives
/// a short inactivity window, or silent after the first chunk.
#[derive(Clone, Copy)]
enum Pacing {
    Slow { every: Duration, chunks: usize },
    Stalled,
}

#[async_trait]
impl InferenceBackend for RecordingBackend {
    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("native-test")])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> InferenceResult<ChatCompletionResponse> {
        let answer = format!("effort={:?}", request.reasoning_effort);
        let model = request.model.clone();
        self.requests.lock().unwrap().push(request);
        Ok(ChatCompletionResponse::new(model, answer, Usage::new(2, 1)))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<ChatCompletionStream> {
        self.stream_tokens
            .lock()
            .unwrap()
            .push(context.cancellation_token());
        let model = request.model.clone();
        self.requests.lock().unwrap().push(request);
        if self.stream_forever {
            let generated = self.generated_chunks.clone();
            return Ok(Box::pin(futures_util::stream::repeat_with(move || {
                generated.fetch_add(1, Ordering::Relaxed);
                Ok(skippy_inference_api::ChatCompletionChunk::delta(
                    model.clone(),
                    "x".repeat(64 * 1024),
                ))
            })));
        }
        if let Some(pacing) = self.pacing {
            return Ok(match pacing {
                Pacing::Slow { every, chunks } => {
                    Box::pin(futures_util::stream::unfold(0usize, move |index| {
                        let model = model.clone();
                        async move {
                            if index == chunks {
                                return None;
                            }
                            tokio::time::sleep(every).await;
                            Some((
                                Ok(skippy_inference_api::ChatCompletionChunk::delta(
                                    model, "tok",
                                )),
                                index + 1,
                            ))
                        }
                    }))
                }
                Pacing::Stalled => Box::pin(
                    futures_util::stream::iter(vec![Ok(
                        skippy_inference_api::ChatCompletionChunk::delta(model, "tok"),
                    )])
                    .chain(futures_util::stream::pending()),
                ),
            });
        }
        Ok(Box::pin(futures_util::stream::iter(vec![
            Ok(skippy_inference_api::ChatCompletionChunk::delta(
                model.clone(),
                "first",
            )),
            Ok(skippy_inference_api::ChatCompletionChunk::delta(
                model.clone(),
                "second",
            )),
            Ok(skippy_inference_api::ChatCompletionChunk::done(model)),
        ])))
    }
}

async fn serving_node(
    efforts: Option<Vec<String>>,
) -> (
    Node,
    Arc<RecordingBackend>,
    Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>,
) {
    let node = Node::new_for_tests(NodeRole::Host { http_port: 9 })
        .await
        .unwrap();
    node.set_serving_models(vec!["native-test".into()]).await;
    node.set_hosted_models(vec!["native-test".into()]).await;
    node.upsert_served_model_descriptor(ServedModelDescriptor {
        identity: super::super::ServedModelIdentity {
            model_name: "native-test".into(),
            is_primary: true,
            ..Default::default()
        },
        capabilities_known: true,
        metadata: Some(super::super::ServedModelMetadata {
            thinking_efforts: efforts.clone(),
            workload_class: Some(super::super::ModelWorkloadClass::CausalGeneration),
            ..Default::default()
        }),
        ..Default::default()
    })
    .await;
    let lifecycle = Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
        InstanceLifecycleState::Serving,
        8,
    )));
    node.register_runtime_instance_lifecycle(31337, lifecycle.clone());
    let backend = Arc::new(RecordingBackend::default());
    node.register_native_backend(
        "native-test",
        "instance-1",
        31337,
        Some(backend.clone()),
        efforts,
    )
    .await;
    (node, backend, lifecycle)
}

fn call(node: &Node, effort: Option<ReasoningEffort>) -> NativeChatRequest {
    NativeChatRequest {
        target: ModelTarget {
            provider_node_id: node.id().to_string(),
            serving_id: "native-test".into(),
        },
        request: ChatCompletionRequest {
            reasoning_effort: effort,
            ..Default::default()
        },
    }
}

async fn connect(client: &Node, provider: &Node) {
    client.start_accepting();
    provider.start_accepting();
    for node in [client.clone(), provider.clone()] {
        tokio::spawn(async move { node.accept_loop().await });
    }
    client
        .connect_to_peer(provider.endpoint_addr_for_advertisement())
        .await
        .unwrap();
}

#[tokio::test]
async fn local_native_inference_preserves_effort_and_rejects_stale_or_draining_targets() {
    let (node, backend, lifecycle) = serving_node(Some(vec!["high".into()])).await;
    let models = node.native_models().await.unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].thinking_efforts, Some(vec!["high".into()]));
    let response = node
        .native_chat_completion(call(&node, Some(ReasoningEffort::High)))
        .await
        .unwrap();
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "effort=Some(High)"
    );
    assert_eq!(backend.requests.lock().unwrap()[0].model, "native-test");
    assert!(
        node.native_chat_completion(call(&node, Some(ReasoningEffort::Low)))
            .await
            .is_err()
    );
    lifecycle
        .lock()
        .await
        .mark_draining(std::time::Instant::now())
        .unwrap();
    assert!(
        node.native_chat_completion(call(&node, None))
            .await
            .is_err()
    );
    assert_eq!(backend.requests.lock().unwrap().len(), 1);
    node.unregister_native_backend("native-test", "instance-1")
        .await;
    assert!(
        node.native_chat_completion(call(&node, None))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn absent_instance_lifecycle_cannot_execute_a_retired_backend_snapshot() {
    let (node, backend, _) = serving_node(Some(vec!["high".into()])).await;
    node.unregister_runtime_instance_lifecycle(31337);
    assert!(
        node.native_chat_completion(call(&node, Some(ReasoningEffort::High)))
            .await
            .is_err()
    );
    assert!(backend.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn reload_during_admission_cannot_execute_an_old_backend_on_a_reused_port() {
    let (node, old_backend, lifecycle) = serving_node(Some(vec!["high".into()])).await;
    let guard = lifecycle.lock().await;
    let request = node.native_chat_completion(call(&node, Some(ReasoningEffort::High)));
    tokio::pin!(request);
    // Poll through the backend snapshot and suspend at lifecycle admission.
    assert!(futures_util::poll!(&mut request).is_pending());
    let new_backend = Arc::new(RecordingBackend::default());
    node.register_native_backend(
        "native-test",
        "instance-1",
        31337,
        Some(new_backend.clone()),
        Some(vec!["low".into()]),
    )
    .await;
    drop(guard);
    assert!(request.await.is_err());
    assert!(old_backend.requests.lock().unwrap().is_empty());
    assert!(new_backend.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn native_catalog_preserves_unknown_and_observed_empty_efforts() {
    for efforts in [None, Some(vec![])] {
        let (node, backend, _) = serving_node(efforts.clone()).await;
        assert_eq!(
            node.native_models().await.unwrap()[0].thinking_efforts,
            efforts
        );
        assert!(
            node.native_chat_completion(call(&node, Some(ReasoningEffort::High)))
                .await
                .is_err()
        );
        node.native_chat_completion(call(&node, None))
            .await
            .unwrap();
        assert_eq!(backend.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_mesh_discovery_keeps_provider_controls_and_executes_selected_remote_model() {
    let (provider, remote_backend, _) = serving_node(Some(vec!["high".into()])).await;
    let (client, local_backend, _) = serving_node(Some(vec!["low".into()])).await;
    provider
        .update_served_model_descriptor("native-test", |descriptor| {
            let mut descriptor = descriptor.unwrap();
            descriptor.identity.source_kind = ModelSourceKind::Catalog;
            descriptor.identity.canonical_ref = Some("publisher/native-model:Q4_K_M".into());
            descriptor
        })
        .await;
    connect(&client, &provider).await;
    let peer = client.peers().await.pop().unwrap();
    assert_ne!(http_routable_models(&peer), vec!["native-test".to_string()]);
    let models = client.native_models().await.unwrap();
    assert_eq!(
        models.len(),
        2,
        "same serving name must retain separate providers"
    );
    let remote = models
        .iter()
        .find(|model| model.target.provider_node_id == provider.id().to_string())
        .unwrap();
    assert_eq!(remote.thinking_efforts, Some(vec!["high".into()]));
    let local = models
        .iter()
        .find(|model| model.target.provider_node_id == client.id().to_string())
        .unwrap();
    assert_eq!(local.thinking_efforts, Some(vec!["low".into()]));
    let response = client
        .native_chat_completion(call(&provider, Some(ReasoningEffort::High)))
        .await
        .unwrap();
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "effort=Some(High)"
    );
    assert_eq!(remote_backend.requests.lock().unwrap().len(), 1);
    assert_eq!(
        remote_backend.requests.lock().unwrap()[0].reasoning_effort,
        Some(ReasoningEffort::High)
    );
    assert!(local_backend.requests.lock().unwrap().is_empty());
    // A locally supported level must not cause fallback from the chosen provider.
    assert!(
        client
            .native_chat_completion(call(&provider, Some(ReasoningEffort::Low)))
            .await
            .is_err()
    );
    assert!(local_backend.requests.lock().unwrap().is_empty());
    provider
        .unregister_native_backend("native-test", "instance-1")
        .await;
    assert!(
        client
            .native_chat_completion(call(&provider, Some(ReasoningEffort::High)))
            .await
            .is_err()
    );
    assert!(local_backend.requests.lock().unwrap().is_empty());
    client.endpoint.close().await;
    provider.endpoint.close().await;
}

async fn exchange_frame(
    client: &Node,
    provider: &Node,
    body: &[u8],
) -> crate::proto::node::NativeChatFrame {
    let connection = client.connection_to_peer(provider.id()).await.unwrap();
    let (mut send, mut recv) = connection.open_bi().await.unwrap();
    send.write_all(&[STREAM_NATIVE_CHAT]).await.unwrap();
    send.write_all(body).await.unwrap();
    send.finish().unwrap();
    let bytes = tokio::time::timeout(Duration::from_secs(5), read_len_prefixed(&mut recv))
        .await
        .unwrap()
        .unwrap();
    decode_frame(&bytes).unwrap()
}

fn framed(frame: crate::proto::node::NativeChatFrame) -> Vec<u8> {
    let bytes = frame.encode_to_vec();
    let mut body = (bytes.len() as u32).to_le_bytes().to_vec();
    body.extend_from_slice(&bytes);
    body
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_wire_rejects_wrong_provider_generation_and_oversize_without_execution() {
    let (provider, backend, _) = serving_node(Some(vec!["high".into()])).await;
    let client = Node::new_for_tests(NodeRole::Client).await.unwrap();
    connect(&client, &provider).await;
    // Dial the provider while naming the client as the execution target.
    let wrong_target = crate::proto::node::NativeChatFrame {
        generation: 1,
        payload_json: serde_json::to_vec(&call(&client, None)).unwrap(),
        error: None,
    };
    let response = exchange_frame(&client, &provider, &framed(wrong_target)).await;
    assert!(response.error.unwrap().contains("different provider"));
    let bad_generation = crate::proto::node::NativeChatFrame {
        generation: 99,
        payload_json: serde_json::to_vec(&call(&provider, None)).unwrap(),
        error: None,
    };
    let response = exchange_frame(&client, &provider, &framed(bad_generation)).await;
    assert!(response.error.unwrap().contains("generation"));
    let oversize = ((crate::protocol::MAX_CONTROL_FRAME_BYTES + 1) as u32).to_le_bytes();
    let response = exchange_frame(&client, &provider, &oversize).await;
    assert!(response.error.is_some());
    assert!(backend.requests.lock().unwrap().is_empty());
    client.endpoint.close().await;
    provider.endpoint.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_mesh_stream_delivers_deltas_and_preserves_effort() {
    let (provider, backend, _) = serving_node(Some(vec!["high".into()])).await;
    let client = Node::new_for_tests(NodeRole::Client).await.unwrap();
    connect(&client, &provider).await;
    let mut stream = client
        .native_chat_completion_stream(call(&provider, Some(ReasoningEffort::High)))
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(5), stream.next_chunk())
        .await
        .unwrap()
    {
        let chunk = chunk.unwrap();
        if let Some(delta) = chunk["choices"][0]["delta"]["content"].as_str() {
            text.push_str(delta);
        }
    }
    assert_eq!(text, "firstsecond");
    {
        let requests = backend.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].stream);
        assert_eq!(requests[0].reasoning_effort, Some(ReasoningEffort::High));
    }
    client.endpoint.close().await;
    provider.endpoint.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_native_stream_cancels_local_and_remote_generation_under_backpressure() {
    for remote in [false, true] {
        let (provider, _, lifecycle) = serving_node(Some(vec!["high".into()])).await;
        let backend = Arc::new(RecordingBackend {
            stream_forever: true,
            ..Default::default()
        });
        provider
            .register_native_backend(
                "native-test",
                "instance-1",
                31337,
                Some(backend.clone()),
                Some(vec!["high".into()]),
            )
            .await;
        let client = Node::new_for_tests(NodeRole::Client).await.unwrap();
        if remote {
            connect(&client, &provider).await;
        }
        let consumer = if remote { &client } else { &provider };
        let stream = consumer
            .native_chat_completion_stream(call(&provider, Some(ReasoningEffort::High)))
            .await
            .unwrap();
        // With no user reads, the local bounded queue stalls after 16 chunks.
        // Remote execution additionally exercises QUIC receive cancellation.
        tokio::time::timeout(Duration::from_secs(5), async {
            while backend.generated_chunks.load(Ordering::Relaxed) < 17 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap_or_else(|error| panic!("native queue did not fill (remote={remote}): {error}"));
        drop(stream);
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if backend
                    .stream_tokens
                    .lock()
                    .unwrap()
                    .first()
                    .is_some_and(|token| token.is_cancelled())
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("consumer drop must cancel backend context");
        assert_eq!(lifecycle.lock().await.in_flight_tracker().get(), 0);
        client.endpoint.close().await;
        provider.endpoint.close().await;
    }
}

/// Paces a backend whose total generation outlives a short inactivity window.
/// Both the consumer and the provider must keep it alive: the window is per
/// chunk, not per request, wherever the chunk is produced.
async fn slow_generation_survives_the_window(remote: bool) {
    let window = Duration::from_millis(150);
    let chunks = 15usize;
    let (provider, _, _) = serving_node(Some(vec!["high".into()])).await;
    provider.set_native_stream_idle(window);
    let backend = Arc::new(RecordingBackend {
        pacing: Some(Pacing::Slow {
            every: Duration::from_millis(20),
            chunks,
        }),
        ..Default::default()
    });
    provider
        .register_native_backend(
            "native-test",
            "instance-1",
            31337,
            Some(backend),
            Some(vec!["high".into()]),
        )
        .await;
    let client = Node::new_for_tests(NodeRole::Client).await.unwrap();
    client.set_native_stream_idle(window);
    if remote {
        connect(&client, &provider).await;
    }
    let consumer = if remote { &client } else { &provider };
    let started = Instant::now();
    let mut stream = consumer
        .native_chat_completion_stream(call(&provider, Some(ReasoningEffort::High)))
        .await
        .unwrap();
    let mut seen = 0usize;
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(5), stream.next_chunk())
        .await
        .unwrap_or_else(|error| panic!("slow stream stalled (remote={remote}): {error}"))
    {
        chunk.unwrap();
        seen += 1;
    }
    let elapsed = started.elapsed();
    assert_eq!(seen, chunks, "remote={remote}");
    assert!(
        elapsed > window,
        "the generation must outlive the inactivity window (remote={remote}): {elapsed:?}"
    );
    client.endpoint.close().await;
    provider.endpoint.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_native_stream_outlives_the_inactivity_window_local_and_remote() {
    slow_generation_survives_the_window(false).await;
    slow_generation_survives_the_window(true).await;
}

/// A generation that stops producing must fail inside the inactivity window
/// instead of waiting for a total-time limit, wherever it runs.
async fn stalled_generation_fails_inside_the_window(remote: bool) {
    let window = Duration::from_millis(150);
    let (provider, _, _) = serving_node(Some(vec!["high".into()])).await;
    provider.set_native_stream_idle(window);
    let backend = Arc::new(RecordingBackend {
        pacing: Some(Pacing::Stalled),
        ..Default::default()
    });
    provider
        .register_native_backend(
            "native-test",
            "instance-1",
            31337,
            Some(backend),
            Some(vec!["high".into()]),
        )
        .await;
    let client = Node::new_for_tests(NodeRole::Client).await.unwrap();
    client.set_native_stream_idle(window);
    if remote {
        connect(&client, &provider).await;
    }
    let consumer = if remote { &client } else { &provider };
    let mut stream = consumer
        .native_chat_completion_stream(call(&provider, Some(ReasoningEffort::High)))
        .await
        .unwrap();
    let first = tokio::time::timeout(Duration::from_secs(5), stream.next_chunk())
        .await
        .unwrap()
        .expect("first chunk")
        .expect("first chunk succeeds");
    assert!(first["choices"][0]["delta"]["content"].is_string());
    let error = tokio::time::timeout(Duration::from_secs(5), stream.next_chunk())
        .await
        .unwrap_or_else(|error| panic!("stalled stream never failed (remote={remote}): {error}"))
        .expect("a stalled stream reports an error instead of completing")
        .expect_err("a stalled stream must not complete");
    assert!(
        error.to_string().contains("timed out"),
        "remote={remote}: {error}"
    );
    client.endpoint.close().await;
    provider.endpoint.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stalled_native_stream_fails_inside_the_inactivity_window_local_and_remote() {
    stalled_generation_fails_inside_the_window(false).await;
    stalled_generation_fails_inside_the_window(true).await;
}
