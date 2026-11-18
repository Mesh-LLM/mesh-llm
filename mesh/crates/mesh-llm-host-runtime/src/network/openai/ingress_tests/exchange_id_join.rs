//! One exchange id joins what a plugin sees across the host's lifecycle
//! callbacks and its `openai.exchange.v1` channel events, from the ingress
//! request through the plugin callbacks to the terminal channel event.

use super::*;
use crate::plugin::selected_attempt_test_support::{AttemptProbe, attempt_policy_probe};
use serde_json::Value;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const MODEL: &str = "qwen";

/// Wait for a recorded message the plugin receives asynchronously.
async fn wait_for(
    what: &str,
    log: &std::sync::Mutex<Vec<Value>>,
    ready: impl Fn(&[Value]) -> bool,
) -> Vec<Value> {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            {
                let events = log.lock().unwrap();
                if ready(&events) {
                    return events.clone();
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {what}"))
}

/// Two local backends that each answer one request with a JSON completion.
async fn backends(contacted: &Arc<AtomicUsize>) -> Vec<u16> {
    let mut ports = Vec::new();
    for _ in 0..2 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        ports.push(listener.local_addr().unwrap().port());
        let contacted = contacted.clone();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            contacted.fetch_add(1, Ordering::SeqCst);
            let mut seen = Vec::new();
            let mut buffer = [0u8; 4096];
            while let Ok(read) = stream.read(&mut buffer).await {
                if read == 0 {
                    break;
                }
                seen.extend_from_slice(&buffer[..read]);
                if seen.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let body = r#"{"id":"c1","model":"qwen","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
        });
    }
    ports
}

/// Send one chat request through the real ingress handler and return the
/// client's response.
async fn chat_through_ingress(probe: &AttemptProbe, ports: &[u16]) -> String {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    node.set_plugin_manager(probe.manager.clone()).await;
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        MODEL.to_string(),
        ports
            .iter()
            .map(|port| election::InferenceTarget::Local(*port))
            .collect(),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let manager = probe.manager.clone();
    let handler = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = proxy::read_http_request(&mut stream).await.unwrap();
        let affinity = affinity::AffinityRouter::new();
        handle_buffered_api_request(
            stream.into(),
            request,
            ProxyConnectionContext {
                route: IngressRouteContext {
                    node: &node,
                    targets: &targets,
                    affinity: &affinity,
                    plugin_manager: Some(&manager),
                    requested_by_node_id: None,
                    automatic_retry: None,
                    exchange_channel: None,
                },
            },
            None,
            crate::runtime::IngressType::LocalOpenAi,
        )
        .await;
    });
    let body = r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}]}"#;
    let mut client = TcpStream::connect(address).await.unwrap();
    client
        .write_all(
            format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    let mut response = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(20),
        client.read_to_string(&mut response),
    )
    .await
    .expect("the client must receive a response")
    .unwrap();
    handler.await.unwrap();
    response
}

fn phase<'a>(events: &'a [Value], phase: &str) -> Vec<&'a Value> {
    events.iter().filter(|e| e["phase"] == phase).collect()
}

/// A paid exchange that fails over from a denied peer to the next one is one
/// exchange: every lifecycle callback the plugin receives (admission, each
/// peer attempt, the terminal callback) and both channel events (effective and
/// terminal) carry the same id, and another request gets another id.
#[tokio::test]
async fn peer_failover_joins_lifecycle_callbacks_and_channel_events_on_one_id() {
    let contacted = Arc::new(AtomicUsize::new(0));
    let ports = backends(&contacted).await;
    // Attempt 1 is denied (first peer), attempt 2 is admitted (next peer).
    let (probe, seen) = attempt_policy_probe(Some(1), false).await;

    let response = chat_through_ingress(&probe, &ports).await;
    assert!(response.starts_with("HTTP/1.1 200"), "got: {response}");
    assert_eq!(
        contacted.load(Ordering::SeqCst),
        1,
        "only the next peer ran"
    );
    assert_eq!(seen.lock().unwrap().len(), 2, "both peers were asked");

    let lifecycle = wait_for("the terminal callback", &probe.lifecycle, |events| {
        events.iter().any(|e| e["phase"] == "exchange_finished")
    })
    .await;
    let channel = wait_for("the terminal channel event", &probe.channel, |events| {
        events.iter().any(|e| e["phase"] == "terminal")
    })
    .await;

    let id = phase(&lifecycle, "request_received")[0]["exchange_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(!id.is_empty());
    let selected = phase(&lifecycle, "backend_selected");
    assert_eq!(selected.len(), 2, "one callback per peer attempt");
    assert_eq!(
        selected
            .iter()
            .map(|e| e["attempt"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [1, 2]
    );
    for event in &lifecycle {
        assert_eq!(event["exchange_id"], id.as_str(), "lifecycle: {event}");
    }
    assert_eq!(channel.len(), 2, "effective and terminal: {channel:?}");
    for event in &channel {
        assert_eq!(event["exchange_id"], id.as_str(), "channel: {event}");
    }
    assert_eq!(channel[0]["phase"], "effective_request");
    assert_eq!(channel[1]["phase"], "terminal");
    // The terminal callback names the served peer and the denied attempt too.
    let finished = phase(&lifecycle, "exchange_finished")[0];
    assert_eq!(finished["attempt_history"].as_array().unwrap().len(), 2);
    assert_eq!(finished["attempt_history"][0]["exchange_id"], id.as_str());
    assert_eq!(finished["attempt_history"][1]["exchange_id"], id.as_str());

    // A second request is a different exchange.
    let contacted = Arc::new(AtomicUsize::new(0));
    let ports = backends(&contacted).await;
    probe.lifecycle.lock().unwrap().clear();
    probe.channel.lock().unwrap().clear();
    chat_through_ingress(&probe, &ports).await;
    let other = wait_for(
        "the second terminal channel event",
        &probe.channel,
        |events| events.iter().any(|e| e["phase"] == "terminal"),
    )
    .await;
    assert_ne!(other[0]["exchange_id"], id.as_str());
}

/// A typed chat completion served by this node: the lifecycle callbacks
/// (`request_received`, `backend_selected`, `exchange_finished`) and the
/// channel's effective and terminal events are one exchange under one id.
#[tokio::test]
async fn typed_chat_joins_lifecycle_callbacks_and_channel_events_on_one_id() {
    use skippy_inference_api::{
        HookedInferenceBackend, InferenceBackend, InferenceRequestContext, InferenceResult,
        ModelObject, Usage, generation::GenerationEvent, generation::GenerationEventStream,
        generation::GenerationRequest, generation::GenerationResponse,
    };

    struct TypedBackend;
    #[async_trait::async_trait]
    impl InferenceBackend for TypedBackend {
        async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
            Ok(vec![ModelObject::new(MODEL)])
        }
        async fn chat_completion(
            &self,
            request: GenerationRequest,
        ) -> InferenceResult<GenerationResponse> {
            Ok(GenerationResponse::new(
                request.model,
                "typed",
                Usage::new(2, 1),
            ))
        }
        async fn chat_completion_stream(
            &self,
            request: GenerationRequest,
            _context: InferenceRequestContext,
        ) -> InferenceResult<GenerationEventStream> {
            Ok(Box::pin(futures_util::stream::iter(vec![
                Ok(GenerationEvent::delta(request.model.clone(), "typed")),
                Ok(GenerationEvent::done(request.model)),
            ])))
        }
    }

    let (probe, _) = attempt_policy_probe(None, false).await;
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    node.set_plugin_manager(probe.manager.clone()).await;
    let hooks = crate::plugin::exchange_policy::compose_node_hooks(node);
    let backend = Arc::new(HookedInferenceBackend::new(Arc::new(TypedBackend), hooks));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, skippy_inference_api::router(backend))
            .await
            .unwrap();
    });
    for stream in [true, false] {
        probe.lifecycle.lock().unwrap().clear();
        probe.channel.lock().unwrap().clear();
        let response = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": stream,
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let _ = response.bytes().await.unwrap();

        let lifecycle = wait_for("the terminal callback", &probe.lifecycle, |events| {
            events.iter().any(|e| e["phase"] == "exchange_finished")
        })
        .await;
        let channel = wait_for("the terminal channel event", &probe.channel, |events| {
            events.iter().any(|e| e["phase"] == "terminal")
        })
        .await;
        let id = phase(&lifecycle, "request_received")[0]["exchange_id"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(!id.is_empty());
        for event in &lifecycle {
            assert_eq!(
                event["exchange_id"],
                id.as_str(),
                "stream={stream} lifecycle: {event}"
            );
        }
        assert_eq!(phase(&lifecycle, "backend_selected").len(), 1);
        assert_eq!(channel.len(), 2, "effective and terminal: {channel:?}");
        for event in &channel {
            assert_eq!(
                event["exchange_id"],
                id.as_str(),
                "stream={stream} channel: {event}"
            );
        }
    }
    server.abort();
}
