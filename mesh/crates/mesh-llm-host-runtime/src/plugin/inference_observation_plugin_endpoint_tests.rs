//! End-to-end coverage for `local_inference_observation.v1` on plugin-hosted
//! OpenAI inference endpoints: readiness (`ready_models`, `model_ready`,
//! `model_unloaded`) and execution lifecycle (`started`, `first_output`,
//! `usage`, `compute_finished`, `ended`).

use super::*;
use crate::logging::OpenAiRouteObserver;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::request_parse::read_http_request;
use crate::network::openai::transport::{RouteSelectionMetadata, route_http_endpoint_request};
use crate::plugin::{InProcessPluginRunner, InProcessPlugins, config};
use mesh_llm_plugin::{PluginMetadata, PluginRuntime, SimplePlugin};
use proto::inference_observation_event::Kind;
use skippy_inference_api::generate_request_id;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

struct TestCleanupGuard {
    hub: Arc<inference_observation::hub::ObservationHub>,
    subscriber: inference_observation::hub::SubscriberId,
    server_task: tokio::task::JoinHandle<()>,
}

impl Drop for TestCleanupGuard {
    fn drop(&mut self) {
        self.hub.unsubscribe(self.subscriber);
        self.server_task.abort();
    }
}

async fn read_full_http_request(stream: &mut TcpStream) -> String {
    let mut raw = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut header_end = None;
    let mut content_length = 0usize;
    while let Ok(n) = stream.read(&mut chunk).await {
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&chunk[..n]);
        if header_end.is_none()
            && let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n")
        {
            let end = pos + 4;
            header_end = Some(end);
            let headers = String::from_utf8_lossy(&raw[..pos]);
            for line in headers.lines() {
                if let Some((name, val)) = line.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = val.trim().parse().unwrap_or(0);
                }
            }
        }
        if let Some(end) = header_end
            && raw.len() >= end + content_length
        {
            break;
        }
    }
    String::from_utf8_lossy(&raw).into_owned()
}

async fn spawn_plugin_openai_server(
    model: String,
    fail_next_chat: Arc<AtomicBool>,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}/v1", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let model = model.clone();
            let fail_next_chat = fail_next_chat.clone();
            tokio::spawn(async move {
                let req = read_full_http_request(&mut stream).await;
                let first_line = req.lines().next().unwrap_or("");
                if first_line.starts_with("GET ") && first_line.contains("/models") {
                    let body = format!(r#"{{"data":[{{"id":"{model}"}}]}}"#);
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.shutdown().await;
                    return;
                }
                if fail_next_chat.swap(false, Ordering::SeqCst) {
                    let body = r#"{"error":{"message":"overloaded","type":"server_error"}}"#;
                    let response = format!(
                        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.shutdown().await;
                    return;
                }
                if first_line.contains("/completions")
                    && !first_line.contains("/chat/completions")
                    && req.contains("\"stream\":true")
                {
                    let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
                    let chunk = format!(
                        "data: {{\"id\":\"cmpl-1\",\"object\":\"text_completion\",\"model\":\"{model}\",\"choices\":[{{\"text\":\"hi\",\"index\":0,\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n"
                    );
                    let _ = stream.write_all(headers.as_bytes()).await;
                    let _ = stream.flush().await;
                    let _ = stream.write_all(chunk.as_bytes()).await;
                    let _ = stream.shutdown().await;
                    return;
                }
                if req.contains("\"stream\":true") {
                    let chunk1 = format!(
                        "data: {{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"{model}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"hi\"}},\"finish_reason\":null}}]}}\n\n"
                    );
                    let chunk2 = format!(
                        "data: {{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"{model}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":11,\"completion_tokens\":7,\"total_tokens\":18}}}}\n\ndata: [DONE]\n\n"
                    );
                    let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
                    let _ = stream.write_all(headers.as_bytes()).await;
                    let _ = stream.write_all(chunk1.as_bytes()).await;
                    let _ = stream.flush().await;
                    let _ = stream.write_all(chunk2.as_bytes()).await;
                    let _ = stream.shutdown().await;
                    return;
                }
                let body = format!(
                    r#"{{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"{model}","choices":[{{"index":0,"message":{{"role":"assistant","content":"hi"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}}}"#
                );
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            });
        }
    });
    (address, handle)
}

fn inference_plugin(name: &'static str, address: String) -> InProcessPluginRunner {
    Arc::new(move |stream| {
        let manifest = mesh_llm_plugin::plugin_manifest()
            .item(mesh_llm_plugin::openai_http_inference_endpoint(
                "openai",
                address.clone(),
            ))
            .build();
        let plugin = SimplePlugin::new(
            PluginMetadata::new(
                name,
                "1.0.0",
                mesh_llm_plugin::plugin_server_info(
                    name,
                    "1.0.0",
                    "Inference",
                    "Serves a plugin OpenAI endpoint",
                    None::<String>,
                ),
            )
            .with_manifest(manifest),
        );
        Box::pin(PluginRuntime::run_with_stream(plugin, stream))
    })
}

async fn run_plugin_http_request(
    node: &crate::mesh::Node,
    model: &str,
    base_url: &str,
    raw_request: &str,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let raw = raw_request.to_owned();
    let client = tokio::spawn(async move {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(raw.as_bytes()).await.unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).await.unwrap();
        response
    });
    let (mut server_stream, _) = listener.accept().await.unwrap();
    let request = read_http_request(&mut server_stream).await.unwrap();
    let mut client_stream = ClientStream::from(server_stream);
    let _outcome = route_http_endpoint_request(
        node,
        Some(model),
        RouteSelectionMetadata {
            provider: Some("obs-plugin"),
            engine: Some("openai"),
        },
        &mut client_stream,
        base_url,
        &request,
        OpenAiRouteObserver::default(),
    )
    .await;
    drop(client_stream);
    client.await.unwrap()
}

fn drain_model_events(
    hub: &inference_observation::hub::ObservationHub,
    subscriber: inference_observation::hub::SubscriberId,
    model: &str,
) -> Vec<proto::InferenceObservationEvent> {
    let drained = hub.drain(subscriber, usize::MAX).unwrap();
    let mut matching_execs = std::collections::HashSet::new();
    let mut events = Vec::new();
    for event in drained.events {
        match event.kind.as_ref() {
            Some(Kind::ModelReady(m)) | Some(Kind::ModelUnloaded(m)) if m == model => {
                events.push(event);
            }
            Some(Kind::Started(started)) if started.model == model => {
                matching_execs.insert(event.execution_id);
                events.push(event);
            }
            _ if matching_execs.contains(&event.execution_id) => {
                events.push(event);
            }
            _ => {}
        }
    }
    events
}

fn event_labels(
    events: &[proto::InferenceObservationEvent],
    expected_op: proto::InferenceOperation,
    expected_streaming: bool,
    expected_outcome: proto::InferenceExecutionOutcome,
    expected_usage: Option<(u32, u32)>,
) -> Vec<&'static str> {
    events
        .iter()
        .map(|e| match e.kind.as_ref().unwrap() {
            Kind::Started(s) => {
                assert_eq!(s.operation(), expected_op);
                assert_eq!(s.streaming, expected_streaming);
                "started"
            }
            Kind::FirstOutput(_) => "first_output",
            Kind::Usage(u) => {
                if let Some((prompt, completion)) = expected_usage {
                    assert_eq!(u.prompt_tokens, prompt);
                    assert_eq!(u.completion_tokens, completion);
                }
                "usage"
            }
            Kind::ComputeFinished(_) => "compute_finished",
            Kind::Ended(end) => {
                assert_eq!(end.outcome(), expected_outcome);
                "ended"
            }
            _ => "other",
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plugin_openai_endpoint_emits_readiness_and_execution_observations() {
    let model = format!("obs-plugin-{}", generate_request_id().as_ref());
    let fail_next_chat = Arc::new(AtomicBool::new(false));
    let (address, server_task) =
        spawn_plugin_openai_server(model.clone(), fail_next_chat.clone()).await;

    let hub = inference_observation::hub();
    let (subscriber, _, initial_snapshot) = hub.subscribe();
    let _cleanup = TestCleanupGuard {
        hub: hub.clone(),
        subscriber,
        server_task,
    };
    assert!(
        !initial_snapshot.ready_models.contains(&model),
        "model should not be ready before plugin starts"
    );

    let (mesh_tx, _mesh_rx) = mpsc::channel(16);
    let mut in_process = InProcessPlugins::default();
    in_process = in_process.with(
        "obs-plugin",
        inference_plugin("obs-plugin", address.clone()),
    );
    let mut spec = config::in_process_builtin_spec("obs-plugin");
    spec.startup.optional = false;
    let manager = PluginManager::start_with_in_process(
        &config::ResolvedPlugins {
            externals: vec![spec],
            inactive: Vec::new(),
        },
        config::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
        in_process,
    )
    .await
    .unwrap();

    // Readiness: startup health probe must publish model_ready and list the model in snapshots.
    let (ready_sub, _, ready_snapshot) = hub.subscribe();
    hub.unsubscribe(ready_sub);
    assert!(
        ready_snapshot.ready_models.contains(&model),
        "expected plugin model '{model}' in ready_models, got {:?}",
        ready_snapshot.ready_models
    );
    let startup_events = drain_model_events(&hub, subscriber, &model);
    assert!(
        startup_events
            .iter()
            .any(|event| matches!(event.kind.as_ref(), Some(Kind::ModelReady(m)) if m == &model)),
        "expected ModelReady for '{model}', got {startup_events:?}"
    );

    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();

    // 1. Non-streaming chat completion emits started -> usage -> compute_finished -> ended(Completed).
    let non_stream_body = format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"hello"}}],"stream":false}}"#
    );
    let non_stream_req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{non_stream_body}",
        non_stream_body.len()
    );
    let response = run_plugin_http_request(&node, &model, &address, &non_stream_req).await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "unexpected response: {response}"
    );
    let events = drain_model_events(&hub, subscriber, &model);
    assert_eq!(
        event_labels(
            &events,
            proto::InferenceOperation::ChatCompletion,
            false,
            proto::InferenceExecutionOutcome::Completed,
            Some((11, 7)),
        ),
        ["started", "usage", "compute_finished", "ended"]
    );

    // 2. Streaming chat completion emits started -> first_output -> usage -> compute_finished -> ended(Completed).
    let stream_body = format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"hello"}}],"stream":true}}"#
    );
    let stream_req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{stream_body}",
        stream_body.len()
    );
    let stream_response = run_plugin_http_request(&node, &model, &address, &stream_req).await;
    assert!(
        stream_response.starts_with("HTTP/1.1 200"),
        "unexpected stream response: {stream_response}"
    );
    let stream_events = drain_model_events(&hub, subscriber, &model);
    assert_eq!(
        event_labels(
            &stream_events,
            proto::InferenceOperation::ChatCompletion,
            true,
            proto::InferenceExecutionOutcome::Completed,
            Some((11, 7)),
        ),
        [
            "started",
            "first_output",
            "usage",
            "compute_finished",
            "ended"
        ]
    );

    // 3. Raw SSE streaming (/v1/completions, ResponseAdapter::None) also emits first_output.
    let cmpl_body = format!(r#"{{"model":"{model}","prompt":"hi","stream":true}}"#);
    let cmpl_req = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{cmpl_body}",
        cmpl_body.len()
    );
    let cmpl_response = run_plugin_http_request(&node, &model, &address, &cmpl_req).await;
    assert!(
        cmpl_response.starts_with("HTTP/1.1 200"),
        "unexpected completion stream response: {cmpl_response}"
    );
    let cmpl_events = drain_model_events(&hub, subscriber, &model);
    assert_eq!(
        event_labels(
            &cmpl_events,
            proto::InferenceOperation::Completion,
            true,
            proto::InferenceExecutionOutcome::Completed,
            None,
        ),
        ["started", "first_output", "compute_finished", "ended"]
    );

    // 4. Failing chat completion (503) emits started -> compute_finished -> ended(Failed).
    fail_next_chat.store(true, Ordering::SeqCst);
    let _fail_response = run_plugin_http_request(&node, &model, &address, &non_stream_req).await;
    let fail_events = drain_model_events(&hub, subscriber, &model);
    assert_eq!(
        event_labels(
            &fail_events,
            proto::InferenceOperation::ChatCompletion,
            false,
            proto::InferenceExecutionOutcome::Failed,
            None,
        ),
        ["started", "compute_finished", "ended"]
    );

    // 5. Non-execution routes and non-POST requests do not start an execution.
    for req in [
        "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
        "GET /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n",
    ] {
        let _ = run_plugin_http_request(&node, &model, &address, req).await;
        assert!(
            drain_model_events(&hub, subscriber, &model).is_empty(),
            "request must not emit execution observations: {req}"
        );
    }

    // 6. Shutting the plugin down unloads the model from ready_models.
    manager.shutdown().await;
    let shutdown_events = drain_model_events(&hub, subscriber, &model);
    assert!(
        shutdown_events
            .iter()
            .any(|e| matches!(e.kind.as_ref(), Some(Kind::ModelUnloaded(m)) if m == &model)),
        "expected ModelUnloaded for '{model}' on shutdown, got {shutdown_events:?}"
    );
    let (after_sub, _, after_shutdown) = hub.subscribe();
    hub.unsubscribe(after_sub);
    assert!(!after_shutdown.ready_models.contains(&model));
}

fn observation_watcher_plugin(
    name: &'static str,
    ready_tx: mpsc::Sender<Vec<String>>,
    batches_tx: mpsc::Sender<proto::InferenceObservationBatch>,
) -> InProcessPluginRunner {
    Arc::new(move |stream| {
        let manifest = mesh_llm_plugin::plugin_manifest()
            .item(mesh_llm_plugin::inference_observation::inference_observation())
            .build();
        let ready_tx = ready_tx.clone();
        let batches_tx = batches_tx.clone();
        let plugin = SimplePlugin::new(
            PluginMetadata::new(
                name,
                "1.0.0",
                mesh_llm_plugin::plugin_server_info(
                    name,
                    "1.0.0",
                    "Watcher",
                    "Observes plugin inference over protobuf",
                    None::<String>,
                ),
            )
            .with_manifest(manifest),
        )
        .on_initialized(move |context| {
            let ready_tx = ready_tx.clone();
            let batches_tx = batches_tx.clone();
            Box::pin(async move {
                let mut feed = context
                    .subscribe_inference_observation()
                    .await
                    .expect("watcher subscription granted");
                let _ = ready_tx.send(feed.snapshot().ready_models.clone()).await;
                tokio::spawn(async move {
                    while let Some(
                        mesh_llm_plugin::inference_observation::InferenceObservationItem::Batch(
                            batch,
                        ),
                    ) = feed.next().await
                    {
                        if batches_tx.send(batch).await.is_err() {
                            break;
                        }
                    }
                });
                Ok(())
            })
        });
        Box::pin(PluginRuntime::run_with_stream(plugin, stream))
    })
}

async fn spawn_disconnect_capable_server(
    model: String,
    hold_second_chunk: Arc<tokio::sync::Notify>,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}/v1", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let model = model.clone();
            let hold_second_chunk = hold_second_chunk.clone();
            tokio::spawn(async move {
                let req = read_full_http_request(&mut stream).await;
                let first_line = req.lines().next().unwrap_or("");
                if first_line.starts_with("GET ") && first_line.contains("/models") {
                    let body = format!(r#"{{"data":[{{"id":"{model}"}}]}}"#);
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.shutdown().await;
                    return;
                }
                if req.contains("\"stream\":true") {
                    let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
                    let chunk1 = format!(
                        "data: {{\"id\":\"chatcmpl-drop\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"{model}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"part1\"}},\"finish_reason\":null}}]}}\n\n"
                    );
                    let chunk2 = format!(
                        "data: {{\"id\":\"chatcmpl-drop\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"{model}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"part2\"}},\"finish_reason\":null}}]}}\n\n"
                    );
                    let chunk3 = format!(
                        "data: {{\"id\":\"chatcmpl-drop\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"{model}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n"
                    );
                    let _ = stream.write_all(headers.as_bytes()).await;
                    let _ = stream.write_all(chunk1.as_bytes()).await;
                    let _ = stream.flush().await;
                    hold_second_chunk.notified().await;
                    let _ = stream.write_all(chunk2.as_bytes()).await;
                    let _ = stream.flush().await;
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    let _ = stream.write_all(chunk3.as_bytes()).await;
                    let _ = stream.shutdown().await;
                    return;
                }
                let body = format!(
                    r#"{{"id":"chatcmpl-ok","object":"chat.completion","created":1,"model":"{model}","choices":[{{"index":0,"message":{{"role":"assistant","content":"ok"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}}}"#
                );
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            });
        }
    });
    (address, handle)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sdk_watcher_receives_plugin_observations_over_wire_including_disconnect_and_crash() {
    let model = format!("obs-wire-{}", generate_request_id().as_ref());
    let hold_second_chunk = Arc::new(tokio::sync::Notify::new());
    let (address, server_task) =
        spawn_disconnect_capable_server(model.clone(), hold_second_chunk.clone()).await;
    struct AbortOnDrop(tokio::task::JoinHandle<()>);
    impl Drop for AbortOnDrop {
        fn drop(&mut self) {
            self.0.abort();
        }
    }
    let _server_guard = AbortOnDrop(server_task);

    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    let config_dir = tempfile::tempdir().unwrap();
    let config_path = config_dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        "version = 1\n\n[[plugin]]\nname = \"obs-watcher\"\nallow_inference_observation = true\n",
    )
    .unwrap();
    node.replace_config_state_for_test(&config_path)
        .await
        .unwrap();
    let (mesh_tx, mesh_rx) = mpsc::channel(32);
    node.start_plugin_channel_forwarder(mesh_rx);

    let (ready_tx, mut ready_rx) = mpsc::channel(4);
    let (batches_tx, mut batches_rx) = mpsc::channel(128);
    let in_process = InProcessPlugins::default()
        .with(
            "obs-watcher",
            observation_watcher_plugin("obs-watcher", ready_tx, batches_tx),
        )
        .with(
            "obs-provider",
            inference_plugin("obs-provider", address.clone()),
        );
    let mut watcher_spec = config::in_process_builtin_spec("obs-watcher");
    watcher_spec.startup.optional = false;
    let mut provider_spec = config::in_process_builtin_spec("obs-provider");
    provider_spec.startup.optional = false;

    let manager = PluginManager::start_with_in_process(
        &config::ResolvedPlugins {
            externals: vec![watcher_spec, provider_spec],
            inactive: Vec::new(),
        },
        config::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
        in_process,
    )
    .await
    .unwrap();

    let initial_ready = tokio::time::timeout(std::time::Duration::from_secs(10), ready_rx.recv())
        .await
        .expect("watcher snapshot timeout")
        .expect("watcher snapshot channel closed");

    // 1. Non-streaming chat completion.
    let ok_body = format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}],"stream":false}}"#
    );
    let ok_req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{ok_body}",
        ok_body.len()
    );
    let ok_resp = run_plugin_http_request(&node, &model, &address, &ok_req).await;
    assert!(ok_resp.starts_with("HTTP/1.1 200"), "{ok_resp}");

    // 2. Streaming chat completion where the client drops mid-stream after the first chunk.
    let drop_body = format!(
        r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}],"stream":true}}"#
    );
    let drop_req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{drop_body}",
        drop_body.len()
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let ingress_addr = listener.local_addr().unwrap();
    let notify_after_drop = hold_second_chunk.clone();
    let client_drop = tokio::spawn(async move {
        let mut client = TcpStream::connect(ingress_addr).await.unwrap();
        client.write_all(drop_req.as_bytes()).await.unwrap();
        let mut buf = [0u8; 4096];
        let mut received = String::new();
        while let Ok(n) = client.read(&mut buf).await {
            if n == 0 {
                break;
            }
            received.push_str(&String::from_utf8_lossy(&buf[..n]));
            if received.contains("part1") {
                break;
            }
        }
        drop(client);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        notify_after_drop.notify_one();
    });
    let (mut server_stream, _) = listener.accept().await.unwrap();
    let parsed_drop_req = read_http_request(&mut server_stream).await.unwrap();
    let mut client_stream = ClientStream::from(server_stream);
    let _drop_outcome = route_http_endpoint_request(
        &node,
        Some(&model),
        RouteSelectionMetadata {
            provider: Some("obs-provider"),
            engine: Some("openai"),
        },
        &mut client_stream,
        &address,
        &parsed_drop_req,
        OpenAiRouteObserver::default(),
    )
    .await;
    client_drop.await.unwrap();

    // 3. Simulated provider crash must immediately unload the model.
    manager.inner.plugins["obs-provider"]
        .handle_runtime_failure(None, "simulated crash".into())
        .await;

    // Collect wire events delivered to the SDK watcher plugin until ModelUnloaded arrives.
    let mut saw_ready = initial_ready.contains(&model);
    let mut saw_unloaded = false;
    let mut exec_outcomes: Vec<(bool, bool, bool, proto::InferenceExecutionOutcome)> = Vec::new();
    let mut active_execs: std::collections::HashMap<u64, (bool, bool, bool)> =
        std::collections::HashMap::new();
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        while let Some(batch) = batches_rx.recv().await {
            for event in batch.events {
                match event.kind.unwrap() {
                    Kind::ModelReady(m) if m == model => saw_ready = true,
                    Kind::ModelUnloaded(m) if m == model => {
                        saw_unloaded = true;
                        return;
                    }
                    Kind::Started(s) if s.model == model => {
                        active_execs.insert(event.execution_id, (s.streaming, false, false));
                    }
                    Kind::FirstOutput(_) => {
                        if let Some(entry) = active_execs.get_mut(&event.execution_id) {
                            entry.1 = true;
                        }
                    }
                    Kind::Usage(_) => {
                        if let Some(entry) = active_execs.get_mut(&event.execution_id) {
                            entry.2 = true;
                        }
                    }
                    Kind::Ended(end) => {
                        if let Some((streaming, first_out, saw_usage)) =
                            active_execs.remove(&event.execution_id)
                        {
                            exec_outcomes.push((streaming, first_out, saw_usage, end.outcome()));
                        }
                    }
                    _ => {}
                }
            }
        }
    })
    .await
    .expect("watcher should receive ModelReady, executions, and immediate ModelUnloaded");

    assert!(saw_ready, "watcher should see ModelReady for '{model}'");
    assert!(
        saw_unloaded,
        "watcher should see immediate ModelUnloaded on crash"
    );
    assert_eq!(
        exec_outcomes,
        vec![
            (
                false,
                false,
                true,
                proto::InferenceExecutionOutcome::Completed
            ),
            (true, true, false, proto::InferenceExecutionOutcome::Dropped),
        ]
    );

    manager.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_flight_health_probe_does_not_restore_models_after_runtime_failure() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let model = format!("obs-race-{}", generate_request_id().as_ref());
    let probe_count = Arc::new(AtomicUsize::new(0));
    let probe_entered = Arc::new(tokio::sync::Notify::new());
    let release_probe = Arc::new(tokio::sync::Notify::new());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}/v1", listener.local_addr().unwrap());
    let server_model = model.clone();
    let server_count = probe_count.clone();
    let server_entered = probe_entered.clone();
    let server_release = release_probe.clone();
    let server_task = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let model = server_model.clone();
            let count = server_count.clone();
            let entered = server_entered.clone();
            let release = server_release.clone();
            tokio::spawn(async move {
                let _req = read_full_http_request(&mut stream).await;
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n >= 1 {
                    entered.notify_one();
                    release.notified().await;
                }
                let body = format!(r#"{{"data":[{{"id":"{model}"}}]}}"#);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            });
        }
    });

    let (mesh_tx, _mesh_rx) = mpsc::channel(32);
    let in_process =
        InProcessPlugins::default().with("obs-race", inference_plugin("obs-race", address));
    let mut spec = config::in_process_builtin_spec("obs-race");
    spec.startup.optional = false;
    let manager = PluginManager::start_with_in_process(
        &config::ResolvedPlugins {
            externals: vec![spec],
            inactive: Vec::new(),
        },
        config::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
        in_process,
    )
    .await
    .unwrap();

    let hub = super::inference_observation::hub();
    let (sub, _notify, initial_ready) = hub.subscribe();
    assert!(
        initial_ready.ready_models.contains(&model),
        "model should be ready after startup"
    );

    let refresh_manager = manager.clone();
    let refresh_task = tokio::spawn(async move {
        refresh_manager
            .refresh_plugin_endpoints("obs-race")
            .await
            .unwrap();
    });
    tokio::time::timeout(std::time::Duration::from_secs(5), probe_entered.notified())
        .await
        .expect("in-flight health probe should reach server");

    let plugin = manager.inner.plugins.get("obs-race").unwrap();
    let generation = plugin
        .current_generation()
        .await
        .expect("active generation");
    plugin
        .handle_runtime_failure(Some(generation), "simulated crash".to_string())
        .await;

    release_probe.notify_one();
    refresh_task.await.unwrap();

    let drained = hub.drain(sub, usize::MAX).unwrap();
    let mut ready = true;
    for event in drained.events {
        match event.kind {
            Some(Kind::ModelUnloaded(m)) if m == model => ready = false,
            Some(Kind::ModelReady(m)) if m == model => ready = true,
            _ => {}
        }
    }
    assert!(
        !ready,
        "stale in-flight health probe must not re-publish ModelReady after crash"
    );

    server_task.abort();
    manager.shutdown().await;
}
