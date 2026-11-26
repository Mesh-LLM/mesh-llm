//! One exchange id joins what a plugin sees across the host's lifecycle
//! callbacks and its `openai.exchange.v1` channel events, from the ingress
//! request through the plugin callbacks to the terminal channel event.

use super::*;
use crate::plugin::selected_attempt_test_support::{AttemptProbe, attempt_policy_probe};
use futures_util::StreamExt as _;
use serde_json::Value;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const MODEL: &str = "qwen";

/// Wait for a recorded message the plugin receives asynchronously.
async fn wait_for<T: Clone>(
    what: &str,
    log: &std::sync::Mutex<Vec<T>>,
    ready: impl Fn(&[T]) -> bool,
) -> Vec<T> {
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
    // Both attempts share `id`, so the terminal names the attempt that served:
    // the second `backend_selected` (the first was denied), not the first.
    assert_eq!(channel[1]["served_attempt"], 2, "terminal: {}", channel[1]);
    assert_eq!(selected[1]["attempt"], channel[1]["served_attempt"]);
    // The terminal callback names the served peer and the denied attempt too.
    let finished = phase(&lifecycle, "exchange_finished")[0];
    assert_eq!(finished["attempt_history"].as_array().unwrap().len(), 2);
    assert_eq!(finished["attempt_history"][0]["exchange_id"], id.as_str());
    assert_eq!(finished["attempt_history"][1]["exchange_id"], id.as_str());
    assert_eq!(finished["served_exchange_id"], id.as_str());
    let request_id = phase(&lifecycle, "request_received")[0]["request_id"].clone();
    for event in phase(&lifecycle, "backend_selected") {
        assert_eq!(event["request_id"], request_id);
    }

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

/// A typed backend behind the node's production hooks and the real HTTP
/// router. With `hang`, a streamed completion sends one delta and then never
/// finishes, so a test can drop the client mid-stream.
async fn typed_server(probe: &AttemptProbe, hang: bool) -> (String, tokio::task::JoinHandle<()>) {
    let (base, server, _) = typed_server_logged(probe, hang).await;
    (base, server)
}

/// The exchange ids the frontend's request log recorded: what a persisted
/// request summary is keyed by.
type LoggedExchangeIds = Arc<std::sync::Mutex<Vec<String>>>;

/// [`typed_server`] with request logging on: the returned list collects every
/// exchange id the frontend's lifecycle events carry.
async fn typed_server_logged(
    probe: &AttemptProbe,
    hang: bool,
) -> (String, tokio::task::JoinHandle<()>, LoggedExchangeIds) {
    use skippy_inference_api::{
        HookedInferenceBackend, InferenceBackend, InferenceRequestContext, InferenceResult,
        ModelObject, Usage, generation::GenerationEvent, generation::GenerationEventStream,
        generation::GenerationRequest, generation::GenerationResponse,
    };

    struct TypedBackend {
        hang: bool,
    }
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
            let delta = Ok(GenerationEvent::delta(request.model.clone(), "typed"));
            if self.hang {
                return Ok(Box::pin(
                    futures_util::stream::iter(vec![delta]).chain(futures_util::stream::pending()),
                ));
            }
            Ok(Box::pin(futures_util::stream::iter(vec![
                delta,
                Ok(GenerationEvent::done(request.model)),
            ])))
        }
    }

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    node.set_plugin_manager(probe.manager.clone()).await;
    let hooks = crate::plugin::exchange_policy::compose_node_hooks(node);
    let backend = Arc::new(HookedInferenceBackend::new(
        Arc::new(TypedBackend { hang }),
        hooks,
    ));
    struct Log(LoggedExchangeIds);
    impl skippy_inference_api::InferenceLifecycleObserver for Log {
        fn observe(&self, event: &skippy_inference_api::InferenceLifecycleEvent) {
            use skippy_inference_api::InferenceLifecycleEvent as Event;
            let id = match event {
                Event::ExchangeIdentified { exchange_id, .. } => Some(exchange_id.clone()),
                Event::NonStreamTerminal { exchange_id, .. } => exchange_id.clone(),
                _ => None,
            };
            self.0.lock().unwrap().extend(id);
        }
    }
    let logged = LoggedExchangeIds::default();
    let config = skippy_inference_api::InferenceFrontendConfig::default()
        .with_lifecycle_observer(Arc::new(Log(logged.clone())));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            skippy_inference_api::router_with_config(backend, config),
        )
        .await
        .unwrap();
    });
    (base, server, logged)
}

/// Every lifecycle callback and both channel events of one typed exchange
/// carry one id.
fn assert_one_typed_exchange(lifecycle: &[Value], channel: &[Value], context: &str) -> String {
    let id = phase(lifecycle, "request_received")[0]["exchange_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(!id.is_empty());
    for event in lifecycle {
        assert_eq!(
            event["exchange_id"],
            id.as_str(),
            "{context} lifecycle: {event}"
        );
    }
    assert_eq!(phase(lifecycle, "backend_selected").len(), 1);
    // One request id on every callback: `backend_selected` must not report the
    // typed backend's own dispatch id in its place.
    let request_id = phase(lifecycle, "request_received")[0]["request_id"].clone();
    for event in lifecycle {
        assert_eq!(
            event["request_id"], request_id,
            "{context} request_id: {event}"
        );
    }
    // A served dispatch is named once the exchange completed or a 2xx
    // response reached the client (a stream the client then dropped).
    let finished = phase(lifecycle, "exchange_finished")[0];
    let began_2xx = finished["status"]
        .as_u64()
        .is_some_and(|status| (200..300).contains(&status))
        && finished["any_response_bytes_emitted"] == true;
    let served = if finished["execution_outcome"] == "completed" || began_2xx {
        Value::from(id.as_str())
    } else {
        Value::Null
    };
    assert_eq!(
        finished["served_exchange_id"], served,
        "{context}: {finished}"
    );
    assert_eq!(
        channel.len(),
        2,
        "{context}: effective and terminal: {channel:?}"
    );
    for event in channel {
        assert_eq!(
            event["exchange_id"],
            id.as_str(),
            "{context} channel: {event}"
        );
    }
    id
}

/// A typed chat completion served by this node, streamed or not: the
/// lifecycle callbacks (`request_received`, `backend_selected`,
/// `exchange_finished`) and the channel's effective and terminal events are
/// one exchange under one id, and the next request is another exchange.
#[tokio::test]
async fn typed_chat_joins_lifecycle_callbacks_and_channel_events_on_one_id() {
    let (probe, _) = attempt_policy_probe(None, false).await;
    let (base, server, logged) = typed_server_logged(&probe, false).await;
    let mut ids = Vec::new();
    for (path, stream) in [
        ("/v1/chat/completions", true),
        ("/v1/chat/completions", false),
        ("/v1/responses", true),
        ("/v1/responses", false),
    ] {
        probe.lifecycle.lock().unwrap().clear();
        probe.channel.lock().unwrap().clear();
        logged.lock().unwrap().clear();
        let body = if path == "/v1/responses" {
            serde_json::json!({"model": MODEL, "input": "hi", "stream": stream})
        } else {
            serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": stream,
            })
        };
        let response = reqwest::Client::new()
            .post(format!("{base}{path}"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK, "{path}");
        let header = response
            .headers()
            .get("x-exchange-id")
            .map(|value| value.to_str().unwrap().to_owned());
        let _ = response.bytes().await.unwrap();

        let lifecycle = wait_for("the terminal callback", &probe.lifecycle, |events| {
            events.iter().any(|e| e["phase"] == "exchange_finished")
        })
        .await;
        let channel = wait_for("the terminal channel event", &probe.channel, |events| {
            events.iter().any(|e| e["phase"] == "terminal")
        })
        .await;
        let context = format!("{path} stream={stream}");
        let id = assert_one_typed_exchange(&lifecycle, &channel, &context);
        // The response header and the request log name the exchange as the
        // channel does, so a receipt keyed by the channel id finds both.
        assert_eq!(header.as_deref(), Some(id.as_str()), "{context} header");
        let logged = wait_for("the logged exchange id", &logged, |ids| !ids.is_empty()).await;
        assert!(
            logged.iter().all(|logged| *logged == id),
            "{context} request log: {logged:?}"
        );
        ids.push(id);
    }
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 4, "each request is its own exchange");
    server.abort();
}

/// A client that drops a streamed typed chat mid-response still gets a
/// terminal channel event under the exchange's id: the cancellation terminal
/// runs after the session is gone, so the id cannot be looked up then.
#[tokio::test]
async fn cancelled_typed_chat_keeps_the_one_id_on_its_terminal_channel_event() {
    let (probe, _) = attempt_policy_probe(None, false).await;
    let (base, server) = typed_server(&probe, true).await;
    let mut response = reqwest::Client::new()
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    // The first chunk arrived, so the exchange is in flight; drop the client.
    assert!(response.chunk().await.unwrap().is_some());
    drop(response);

    let lifecycle = wait_for("the terminal callback", &probe.lifecycle, |events| {
        events.iter().any(|e| e["phase"] == "exchange_finished")
    })
    .await;
    let channel = wait_for("the terminal channel event", &probe.channel, |events| {
        events.iter().any(|e| e["phase"] == "terminal")
    })
    .await;
    let id = assert_one_typed_exchange(&lifecycle, &channel, "cancelled");
    // The client dropped a stream it had started receiving: the dispatch
    // served what it sent, so the terminal still names it.
    let finished = phase(&lifecycle, "exchange_finished")[0];
    assert_eq!(finished["served_exchange_id"], id.as_str(), "{finished}");
    server.abort();
}
