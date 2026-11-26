//! Edge-case regression tests for plugin inference observation:
//! - trusted locality rule + `plugin.<name>.local_inference_endpoints` override
//!   across readiness and execution
//! - actual plugin connection-close path (`plugin/transport.rs`) clearing
//!   readiness and re-advertising on restart
//! - HTTP 200 SSE error frames (bounded & unbounded) reporting `Failed` while
//!   forwarding original bytes
//! - premature EOF on chunked responses and `/v1/completions` SSE before `[DONE]`
//! - SSE keep-alive comments not emitting `FirstOutput`
//! - refused upstream connections emitting zero observation events
//! - loopback readiness probe isolation from inherited `HTTP_PROXY`

use super::*;
use crate::logging::OpenAiRouteObserver;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::request_parse::read_http_request;
use crate::network::openai::transport::{RouteSelectionMetadata, route_http_endpoint_request};
use crate::plugin::{InProcessPluginRunner, InProcessPlugins, config};
use mesh_llm_plugin::{PluginMetadata, PluginRuntime, SimplePlugin};
use proto::inference_observation_event::Kind;
use skippy_inference_api::generate_request_id;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn read_http_head(stream: &mut TcpStream) -> String {
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

async fn send_routed_request(
    node: &crate::mesh::Node,
    model: &str,
    base_url: &str,
    observe_local_inference: bool,
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
            provider: Some("obs-edge"),
            engine: Some("openai"),
            observe_local_inference,
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

fn drain_events_for_model(
    hub: &inference_observation::hub::ObservationHub,
    subscriber: inference_observation::hub::SubscriberId,
    model: &str,
) -> Vec<proto::InferenceObservationEvent> {
    let drained = hub.drain(subscriber, usize::MAX).unwrap();
    let mut matching_execs = std::collections::HashSet::new();
    let mut events = Vec::new();
    for event in drained.events {
        let matches = match event.kind.as_ref() {
            Some(Kind::ModelReady(m) | Kind::ModelUnloaded(m)) => m == model,
            Some(Kind::Started(s)) if s.model == model => {
                matching_execs.insert(event.execution_id);
                true
            }
            _ => event.execution_id != 0 && matching_execs.contains(&event.execution_id),
        };
        if matches {
            events.push(event);
        }
    }
    events
}

fn endpoint_plugin(
    name: &'static str,
    address: String,
    managed_by_plugin: bool,
) -> InProcessPluginRunner {
    Arc::new(move |stream| {
        let manifest = mesh_llm_plugin::plugin_manifest()
            .item(
                mesh_llm_plugin::openai_http_inference_endpoint("openai", address.clone())
                    .managed_by_plugin(managed_by_plugin),
            )
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

async fn spawn_simple_models_and_chat_server(
    model: String,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}/v1", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let model = model.clone();
            tokio::spawn(async move {
                let req = read_http_head(&mut stream).await;
                let first_line = req.lines().next().unwrap_or("");
                let body = if first_line.starts_with("GET ") && first_line.contains("/models") {
                    format!(r#"{{"data":[{{"id":"{model}"}}]}}"#)
                } else {
                    format!(
                        r#"{{"id":"chatcmpl-loc","object":"chat.completion","created":1,"model":"{model}","choices":[{{"index":0,"message":{{"role":"assistant","content":"ok"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}}}"#
                    )
                };
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
async fn locality_rule_applies_consistently_to_readiness_and_execution_with_operator_override() {
    let model = format!("obs-loc-{}", generate_request_id().as_ref());
    let (address, server_task) = spawn_simple_models_and_chat_server(model.clone()).await;
    let hub = inference_observation::hub();
    let (sub, _, _) = hub.subscribe();
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();

    // 1. Unmanaged loopback endpoint (ambiguous proxy) with no operator override:
    // neither readiness nor execution is reported as local.
    let (mesh_tx, _mesh_rx) = mpsc::channel(16);
    let in_process = InProcessPlugins::default().with(
        "ambiguous-proxy",
        endpoint_plugin("ambiguous-proxy", address.clone(), false),
    );
    let mut spec = config::in_process_builtin_spec("ambiguous-proxy");
    spec.startup.optional = false;
    spec.local_inference_endpoints = None;
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

    let routes = manager.inference_endpoints().await.unwrap();
    let route = routes
        .iter()
        .find(|r| r.plugin_name == "ambiguous-proxy")
        .expect("unmanaged endpoint should still route");
    assert!(
        !route.observe_local_inference,
        "ambiguous unmanaged proxy must not be marked local by default"
    );
    let (snap_sub, _, snapshot) = hub.subscribe();
    hub.unsubscribe(snap_sub);
    assert!(
        !snapshot.ready_models.contains(&model),
        "ambiguous unmanaged proxy must not publish local model readiness by default"
    );

    let chat_body =
        format!(r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}]}}"#);
    let chat_req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{chat_body}",
        chat_body.len()
    );
    let resp = send_routed_request(
        &node,
        &model,
        &route.address,
        route.observe_local_inference,
        &chat_req,
    )
    .await;
    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    assert!(
        drain_events_for_model(&hub, sub, &model).is_empty(),
        "non-local proxy execution must emit zero local observation events"
    );
    manager.shutdown().await;

    // 2. Same unmanaged loopback proxy with `local_inference_endpoints = Some(true)`:
    // both readiness and execution are reported as local.
    let (mesh_tx, _mesh_rx) = mpsc::channel(16);
    let in_process = InProcessPlugins::default().with(
        "local-proxy",
        endpoint_plugin("local-proxy", address.clone(), false),
    );
    let mut spec = config::in_process_builtin_spec("local-proxy");
    spec.startup.optional = false;
    spec.local_inference_endpoints = Some(true);
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

    let routes = manager.inference_endpoints().await.unwrap();
    let route = routes
        .iter()
        .find(|r| r.plugin_name == "local-proxy")
        .expect("opted-in endpoint should route");
    assert!(route.observe_local_inference);
    let (snap_sub, _, snapshot) = hub.subscribe();
    hub.unsubscribe(snap_sub);
    assert!(snapshot.ready_models.contains(&model));

    let resp = send_routed_request(
        &node,
        &model,
        &route.address,
        route.observe_local_inference,
        &chat_req,
    )
    .await;
    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    let events = drain_events_for_model(&hub, sub, &model);
    assert!(
        events
            .iter()
            .any(|e| matches!(e.kind.as_ref(), Some(Kind::ModelReady(_))))
    );
    assert!(events.iter().any(|e| matches!(
        e.kind.as_ref(),
        Some(Kind::Ended(end)) if end.outcome() == proto::InferenceExecutionOutcome::Completed
    )));
    manager.shutdown().await;

    // 3. Managed loopback plugin with `local_inference_endpoints = Some(false)`:
    // suppressed for both readiness and execution.
    let _ = drain_events_for_model(&hub, sub, &model);
    let (mesh_tx, _mesh_rx) = mpsc::channel(16);
    let in_process = InProcessPlugins::default().with(
        "remote-forwarding-managed",
        endpoint_plugin("remote-forwarding-managed", address, true),
    );
    let mut spec = config::in_process_builtin_spec("remote-forwarding-managed");
    spec.startup.optional = false;
    spec.local_inference_endpoints = Some(false);
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

    let routes = manager.inference_endpoints().await.unwrap();
    let route = routes
        .iter()
        .find(|r| r.plugin_name == "remote-forwarding-managed")
        .expect("opted-out endpoint should still route");
    assert!(!route.observe_local_inference);
    let (snap_sub, _, snapshot) = hub.subscribe();
    hub.unsubscribe(snap_sub);
    assert!(!snapshot.ready_models.contains(&model));

    manager.shutdown().await;
    hub.unsubscribe(sub);
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plugin_self_close_clears_readiness_and_reemits_ready_after_restart() {
    let model = format!("obs-close-{}", generate_request_id().as_ref());
    let (address, server_task) = spawn_simple_models_and_chat_server(model.clone()).await;
    let hub = inference_observation::hub();
    let (sub, _, _) = hub.subscribe();

    let close_first_conn = Arc::new(tokio::sync::Notify::new());
    let (allow_restart_tx, allow_restart_rx) = tokio::sync::watch::channel(false);
    let launch_count = Arc::new(AtomicUsize::new(0));
    let runner_close = close_first_conn.clone();
    let runner_launches = launch_count.clone();
    let runner_addr = address.clone();
    let runner: InProcessPluginRunner = Arc::new(move |stream| {
        let generation_idx = runner_launches.fetch_add(1, Ordering::SeqCst);
        let close_trigger = runner_close.clone();
        let mut restart_rx = allow_restart_rx.clone();
        let manifest = mesh_llm_plugin::plugin_manifest()
            .item(
                mesh_llm_plugin::openai_http_inference_endpoint("openai", runner_addr.clone())
                    .managed_by_plugin(true),
            )
            .build();
        let plugin = SimplePlugin::new(
            PluginMetadata::new(
                "self-closing-plugin",
                "1.0.0",
                mesh_llm_plugin::plugin_server_info(
                    "self-closing-plugin",
                    "1.0.0",
                    "Inference",
                    "Drops its own IPC connection on demand",
                    None::<String>,
                ),
            )
            .with_manifest(manifest),
        );
        Box::pin(async move {
            if generation_idx == 0 {
                tokio::select! {
                    res = PluginRuntime::run_with_stream(plugin, stream) => res,
                    _ = close_trigger.notified() => Ok(()),
                }
            } else {
                let _ = restart_rx.wait_for(|allowed| *allowed).await;
                PluginRuntime::run_with_stream(plugin, stream).await
            }
        })
    });

    let (mesh_tx, _mesh_rx) = mpsc::channel(16);
    let in_process = InProcessPlugins::default().with("self-closing-plugin", runner);
    let mut spec = config::in_process_builtin_spec("self-closing-plugin");
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

    let startup = drain_events_for_model(&hub, sub, &model);
    assert!(
        startup
            .iter()
            .any(|e| matches!(e.kind.as_ref(), Some(Kind::ModelReady(_)))),
        "startup must publish ModelReady"
    );

    // Close the plugin's own IPC connection without calling handle_runtime_failure.
    close_first_conn.notify_one();

    // Wait for transport.rs's connection_loop to clear the runtime and emit ModelUnloaded.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut saw_unloaded = false;
    while tokio::time::Instant::now() < deadline {
        let events = drain_events_for_model(&hub, sub, &model);
        if events
            .iter()
            .any(|e| matches!(e.kind.as_ref(), Some(Kind::ModelUnloaded(_))))
        {
            saw_unloaded = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(
        saw_unloaded,
        "plugin self-close in transport.rs must emit ModelUnloaded"
    );
    let (snap_sub, _, snapshot) = hub.subscribe();
    hub.unsubscribe(snap_sub);
    assert!(
        !snapshot.ready_models.contains(&model),
        "ready_models must be cleared after plugin connection closes"
    );

    // Restart the plugin advertising the exact same model; subscribers must see ModelReady again.
    let _ = allow_restart_tx.send(true);
    let plugin = manager.inner.plugins.get("self-closing-plugin").unwrap();
    plugin.ensure_running().await.unwrap();
    manager
        .refresh_plugin_endpoints("self-closing-plugin")
        .await
        .unwrap();
    let restart_events = drain_events_for_model(&hub, sub, &model);
    assert!(
        restart_events
            .iter()
            .any(|e| matches!(e.kind.as_ref(), Some(Kind::ModelReady(_)))),
        "restarted plugin advertising the same model must emit ModelReady again"
    );

    manager.shutdown().await;
    hub.unsubscribe(sub);
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_sse_error_chunked_truncation_and_refused_connection() {
    let hub = inference_observation::hub();
    let (sub, _, _) = hub.subscribe();
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();

    // 1. Refused upstream connection emits zero observation events (no Started / ComputeFinished).
    let dead_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_addr = format!("http://{}/v1", dead_listener.local_addr().unwrap());
    drop(dead_listener);
    let refused_model = format!("obs-refused-{}", generate_request_id().as_ref());
    let body = format!(r#"{{"model":"{refused_model}","prompt":"hi","stream":true}}"#);
    let req = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = send_routed_request(&node, &refused_model, &dead_addr, true, &req).await;
    assert!(
        drain_events_for_model(&hub, sub, &refused_model).is_empty(),
        "refused upstream connection must not emit Started or ComputeFinished"
    );

    // 2. HTTP 200 completion SSE stream with explicit error forwards original bytes and reports Failed.
    let error_payload = "data: {\"error\":{\"message\":\"backend exploded\",\"type\":\"server_error\"}}\n\ndata: [DONE]\n\n";
    for (label, bounded) in [("unbounded", false), ("bounded", true)] {
        let model = format!("obs-sse-err-{label}-{}", generate_request_id().as_ref());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_head(&mut stream).await;
            let response = if bounded {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{error_payload}",
                    error_payload.len()
                )
            } else {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{error_payload}"
                )
            };
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.shutdown().await;
        });
        let body = format!(r#"{{"model":"{model}","prompt":"hi","stream":true}}"#);
        let req = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let resp = send_routed_request(&node, &model, &addr, true, &req).await;
        server.await.unwrap();
        assert!(
            resp.contains("backend exploded"),
            "{label}: original SSE error bytes must be forwarded verbatim: {resp}"
        );
        let events = drain_events_for_model(&hub, sub, &model);
        assert!(
            !events
                .iter()
                .any(|e| matches!(e.kind.as_ref(), Some(Kind::FirstOutput(_)))),
            "{label}: error-only SSE must not emit FirstOutput"
        );
        let ended = events.iter().find_map(|e| match e.kind.as_ref() {
            Some(Kind::Ended(end)) => Some(end.outcome()),
            _ => None,
        });
        assert_eq!(
            ended,
            Some(proto::InferenceExecutionOutcome::Failed),
            "{label}: explicit HTTP 200 SSE error must report Failed"
        );
    }

    // 3. Chunked response closing before terminating `0\r\n\r\n` reports Failed (Truncated).
    {
        let model = format!("obs-chunk-eof-{}", generate_request_id().as_ref());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_head(&mut stream).await;
            let part = "data: {\"choices\":[{\"text\":\"partial\"}]}\n\n";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{part}\r\n",
                part.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.shutdown().await;
        });
        let body = format!(r#"{{"model":"{model}","prompt":"hi","stream":true}}"#);
        let req = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = send_routed_request(&node, &model, &addr, true, &req).await;
        server.await.unwrap();
        let events = drain_events_for_model(&hub, sub, &model);
        let ended = events.iter().find_map(|e| match e.kind.as_ref() {
            Some(Kind::Ended(end)) => Some(end.outcome()),
            _ => None,
        });
        assert_eq!(
            ended,
            Some(proto::InferenceExecutionOutcome::Failed),
            "premature EOF before terminating HTTP chunk must report Failed"
        );
    }

    // 4. Unchunked /v1/completions SSE closing without [DONE] reports Failed (Truncated).
    {
        let model = format!("obs-nodone-{}", generate_request_id().as_ref());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_head(&mut stream).await;
            let response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"text\":\"cut off\"}]}\n\n";
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.shutdown().await;
        });
        let body = format!(r#"{{"model":"{model}","prompt":"hi","stream":true}}"#);
        let req = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = send_routed_request(&node, &model, &addr, true, &req).await;
        server.await.unwrap();
        let events = drain_events_for_model(&hub, sub, &model);
        let ended = events.iter().find_map(|e| match e.kind.as_ref() {
            Some(Kind::Ended(end)) => Some(end.outcome()),
            _ => None,
        });
        assert_eq!(
            ended,
            Some(proto::InferenceExecutionOutcome::Failed),
            "/v1/completions SSE EOF before [DONE] must report Failed"
        );
    }

    hub.unsubscribe(sub);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_completions_sse_without_done_reports_completed() {
    let hub = inference_observation::hub();
    let (sub, _, _) = hub.subscribe();
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();

    // Non-/v1/completions SSE endpoint (e.g. /v1/audio/transcriptions) ending cleanly
    // without [DONE] still reports Completed (not Truncated).
    let model = format!("obs-audio-sse-{}", generate_request_id().as_ref());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = format!("http://{}/v1", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let _ = read_http_head(&mut stream).await;
        let sse = "event: transcript.text.delta\ndata: {\"type\":\"transcript.text.delta\",\"delta\":\"hi\"}\n\nevent: transcript.text.done\ndata: {\"type\":\"transcript.text.done\",\"text\":\"hi\"}\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{sse}\r\n0\r\n\r\n",
            sse.len()
        );
        let _ = stream.write_all(response.as_bytes()).await;
        let _ = stream.shutdown().await;
    });
    let body = format!(r#"{{"model":"{model}","stream":true}}"#);
    let req = format!(
        "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let resp = send_routed_request(&node, &model, &addr, true, &req).await;
    server.await.unwrap();
    assert!(resp.contains("transcript.text.done"), "{resp}");
    let events = drain_events_for_model(&hub, sub, &model);
    let ended = events.iter().find_map(|e| match e.kind.as_ref() {
        Some(Kind::Ended(end)) => Some(end.outcome()),
        _ => None,
    });
    assert_eq!(
        ended,
        Some(proto::InferenceExecutionOutcome::Completed),
        "non-/v1/completions SSE stream without [DONE] must report Completed"
    );

    hub.unsubscribe(sub);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_sse_keepalive_delays_first_output_until_real_data_frame() {
    let hub = inference_observation::hub();
    let (sub, _, _) = hub.subscribe();
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();

    // SSE keep-alive comment does not emit FirstOutput; first real data frame does.
    {
        let model = format!("obs-keepalive-{}", generate_request_id().as_ref());
        let keepalive_sent = Arc::new(tokio::sync::Notify::new());
        let send_output = Arc::new(tokio::sync::Notify::new());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = format!("http://{}/v1", listener.local_addr().unwrap());
        let server_sent = keepalive_sent.clone();
        let server_gate = send_output.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_head(&mut stream).await;
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
            let keepalive = ": keep-alive\n\n";
            let _ = stream.write_all(head.as_bytes()).await;
            let _ = stream
                .write_all(format!("{:x}\r\n{keepalive}\r\n", keepalive.len()).as_bytes())
                .await;
            let _ = stream.flush().await;
            server_sent.notify_one();
            server_gate.notified().await;
            let data = "data: {\"choices\":[{\"text\":\"hello\"}]}\n\ndata: [DONE]\n\n";
            let _ = stream
                .write_all(format!("{:x}\r\n{data}\r\n0\r\n\r\n", data.len()).as_bytes())
                .await;
            let _ = stream.shutdown().await;
        });

        let body = format!(r#"{{"model":"{model}","prompt":"hi","stream":true}}"#);
        let req = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let client_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client_addr = client_listener.local_addr().unwrap();
        let client_read_keepalive = Arc::new(tokio::sync::Notify::new());
        let client_got_keepalive = client_read_keepalive.clone();
        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(client_addr).await.unwrap();
            stream.write_all(req.as_bytes()).await.unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 256];
            loop {
                let n = stream.read(&mut tmp).await.unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                if String::from_utf8_lossy(&buf).contains(": keep-alive") {
                    client_got_keepalive.notify_one();
                }
            }
            String::from_utf8_lossy(&buf).into_owned()
        });
        let (mut server_stream, _) = client_listener.accept().await.unwrap();
        let request = read_http_request(&mut server_stream).await.unwrap();
        let mut client_stream = ClientStream::from(server_stream);
        let route_model = model.clone();
        let route_node = node.clone();
        let route_task = tokio::spawn(async move {
            route_http_endpoint_request(
                &route_node,
                Some(&route_model),
                RouteSelectionMetadata {
                    provider: Some("obs-edge"),
                    engine: Some("openai"),
                    observe_local_inference: true,
                },
                &mut client_stream,
                &addr,
                &request,
                OpenAiRouteObserver::default(),
            )
            .await
        });

        tokio::time::timeout(std::time::Duration::from_secs(5), keepalive_sent.notified())
            .await
            .expect("server should send keep-alive");
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client_read_keepalive.notified(),
        )
        .await
        .expect("client should receive keep-alive");
        let early_events = drain_events_for_model(&hub, sub, &model);
        let exec_id = early_events
            .iter()
            .find_map(|e| {
                matches!(e.kind.as_ref(), Some(Kind::Started(_))).then_some(e.execution_id)
            })
            .expect("execution must have started once forwarded");
        assert!(
            !early_events
                .iter()
                .any(|e| matches!(e.kind.as_ref(), Some(Kind::FirstOutput(_)))),
            "keep-alive comment must not emit FirstOutput before real model output"
        );

        send_output.notify_one();
        let _ = route_task.await.unwrap();
        let _ = client.await.unwrap();
        server.await.unwrap();

        let later_events: Vec<_> = hub
            .drain(sub, usize::MAX)
            .unwrap()
            .events
            .into_iter()
            .filter(|e| e.execution_id == exec_id)
            .collect();
        assert!(
            later_events
                .iter()
                .any(|e| matches!(e.kind.as_ref(), Some(Kind::FirstOutput(_)))),
            "real data frame must emit FirstOutput"
        );
        assert!(later_events.iter().any(|e| matches!(
            e.kind.as_ref(),
            Some(Kind::Ended(end)) if end.outcome() == proto::InferenceExecutionOutcome::Completed
        )));
    }

    hub.unsubscribe(sub);
}

#[test]
fn loopback_readiness_probe_ignores_inherited_http_proxy() {
    if std::env::var_os("MESH_LLM_TEST_LOOPBACK_PROXY_CHILD").is_some() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let model = format!("obs-proxy-iso-{}", generate_request_id().as_ref());
            let (address, server_task) = spawn_simple_models_and_chat_server(model.clone()).await;
            let (mesh_tx, _mesh_rx) = mpsc::channel(16);
            let in_process = InProcessPlugins::default().with(
                "obs-proxy-iso",
                endpoint_plugin("obs-proxy-iso", address, true),
            );
            let mut spec = config::in_process_builtin_spec("obs-proxy-iso");
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
            let hub = inference_observation::hub();
            let (snap_sub, _, snapshot) = hub.subscribe();
            hub.unsubscribe(snap_sub);
            assert!(
                snapshot.ready_models.contains(&model),
                "loopback readiness probe must succeed even with HTTP_PROXY set"
            );
            manager.shutdown().await;
            server_task.abort();
        });
        return;
    }

    let exe = std::env::current_exe().unwrap();
    let output = std::process::Command::new(exe)
        .arg("--exact")
        .arg("plugin::inference_observation_edge_tests::loopback_readiness_probe_ignores_inherited_http_proxy")
        .arg("--nocapture")
        .env("MESH_LLM_TEST_LOOPBACK_PROXY_CHILD", "1")
        .env("HTTP_PROXY", "http://127.0.0.1:1")
        .env("http_proxy", "http://127.0.0.1:1")
        .env("ALL_PROXY", "http://127.0.0.1:1")
        .env("all_proxy", "http://127.0.0.1:1")
        .env_remove("NO_PROXY")
        .env_remove("no_proxy")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child test failed under HTTP_PROXY:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
