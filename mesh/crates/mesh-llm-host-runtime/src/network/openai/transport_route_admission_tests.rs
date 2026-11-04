//! A plugin admission NO at `backend_selected` skips to the next candidate.
use super::*;
use crate::plugin::selected_attempt_test_support::{DENY_HEADER, attempt_policy_manager};
use mesh_llm_events::logging::identifiers::RequestId;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MODEL: &str = "qwen";

/// Route one request across two healthy backends under a plugin that denies
/// attempts `<= deny_through` (or fails). Returns the client's raw response,
/// how many backends were contacted, and the plugin's `(target, attempt)` log.
async fn route(deny_through: u64, fail: bool) -> (String, usize, Vec<(String, u64)>) {
    route_observed(deny_through, fail, false).await.0
}

/// `route`, optionally inside a real `ExchangeSession`; also returns the
/// terminal events the plugin received.
async fn route_observed(
    deny_through: u64,
    fail: bool,
    observed: bool,
) -> ((String, usize, Vec<(String, u64)>), Vec<serde_json::Value>) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let contacted = Arc::new(AtomicUsize::new(0));
    let mut ports = Vec::new();
    let mut tasks = Vec::new();
    for _ in 0..2 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        ports.push(listener.local_addr().unwrap().port());
        let contacted = contacted.clone();
        tasks.push(tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            contacted.fetch_add(1, Ordering::SeqCst);
            let mut buffer = [0u8; 4096];
            let mut seen = Vec::new();
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
        }));
    }
    let downstream = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = downstream.local_addr().unwrap();
    let client = tokio::spawn(async move {
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        String::from_utf8_lossy(&response).into_owned()
    });
    let (downstream_stream, _) = downstream.accept().await.unwrap();

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    let (manager, seen, terminals) = attempt_policy_manager(deny_through, fail).await;
    let mut session = None;
    let mut observation_id = None;
    if observed {
        let (started, admission) = crate::plugin::ExchangeSession::begin(
            &manager,
            crate::plugin::request_event(
                "skip-e2e".into(),
                "chat_completions",
                "POST",
                "/v1/chat/completions",
                b"{}",
                Default::default(),
                false,
            ),
        )
        .await;
        assert!(admission.error_status().is_none());
        observation_id = Some(started.observation_id().to_owned());
        session = Some(started);
    }
    node.set_plugin_manager(manager).await;
    let affinity = AffinityRouter::new();
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        MODEL.to_string(),
        ports
            .iter()
            .map(|port| election::InferenceTarget::Local(*port))
            .collect(),
    );
    let body = r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}]}"#;
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes();
    let mut request = BufferedHttpRequest {
        raw,
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: body.len(),
        completion_tokens: None,
        stream: None,
        model_name: Some(MODEL.to_owned()),
        request_object_request_ids: Vec::new(),
        response_adapter: ResponseAdapter::None,
        correlation_id: None,
        exchange_observation_id: None,
        automatic_route: false,
        trust: crate::network::request_trust::RequestTrust::UNRESTRICTED,
    };
    request.exchange_observation_id = observation_id;
    route_model_request(
        node,
        downstream_stream.into(),
        &targets,
        MODEL,
        &request,
        RouteModelRequestContext {
            local_execution_only: false,
            automatic_retry: None,
            required_tokens: None,
            affinity: &affinity,
            route_observer: OpenAiRouteObserver::default(),
            served_by_header: None,
            peer_capsule_id: None,
            served_by_node_id: None,
        },
    )
    .await;
    let response = tokio::time::timeout(std::time::Duration::from_secs(20), client)
        .await
        .expect("the client must receive a response")
        .unwrap();
    let contacted = contacted.load(Ordering::SeqCst);
    for task in tasks {
        task.abort();
    }
    let seen = seen.lock().unwrap().clone();
    if let Some(mut session) = session {
        session
            .observer()
            .finish(skippy_inference_api::wire_bytes::commit_wire_bytes(b""));
        session.finish("completed").await;
    }
    let finished = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while observed && terminals.lock().unwrap().is_empty() {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        terminals.lock().unwrap().clone()
    })
    .await
    .expect("the plugin must receive the terminal event");
    ((response, contacted, seen), finished)
}

#[tokio::test]
async fn a_no_on_the_first_candidate_routes_to_the_next_peer() {
    let (response, contacted, seen) = route(1, false).await;
    assert!(response.starts_with("HTTP/1.1 200"), "got: {response}");
    assert_eq!(contacted, 1, "the denied candidate is never contacted");
    assert_eq!(seen.iter().map(|(_, a)| *a).collect::<Vec<_>>(), [1, 2]);
    assert_ne!(
        seen[0].0, seen[1].0,
        "the next peer, not a retry of the first"
    );
}

#[tokio::test]
async fn a_no_on_every_candidate_returns_the_deny_once_none_admits() {
    let (response, contacted, seen) = route(u64::MAX, false).await;
    assert!(response.starts_with("HTTP/1.1 403"), "got: {response}");
    // The last deny's status and body, exactly as a single-candidate route sends.
    let body: serde_json::Value =
        serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        body,
        serde_json::json!({"error":{
            "message":"OpenAI plugin admission rejected the selected route",
            "type":"permission_error","code":"plugin_policy_denied"}})
    );
    assert_eq!(response.matches("HTTP/1.1").count(), 1);
    assert_eq!(contacted, 0);
    assert_eq!(seen.len(), 2, "every candidate was asked before giving up");
}

#[tokio::test]
async fn a_failed_admission_hook_does_not_skip_to_the_next_peer() {
    let (response, contacted, seen) = route(0, true).await;
    assert!(response.starts_with("HTTP/1.1 503"), "got: {response}");
    assert_eq!(contacted, 0);
    assert_eq!(seen.len(), 1, "fail closed on the first candidate");
}

#[tokio::test]
async fn a_skipped_peers_deny_is_recorded_but_does_not_mark_the_admitted_exchange() {
    let ((response, contacted, seen), terminals) = route_observed(1, false, true).await;
    assert!(response.starts_with("HTTP/1.1 200"), "got: {response}");
    assert!(
        !response.to_ascii_lowercase().contains(DENY_HEADER),
        "the skipped peer's deny headers must not reach the client"
    );
    assert_eq!(contacted, 1);
    assert_eq!(seen.len(), 2);
    let terminal = &terminals[0];
    assert_eq!(terminal["admission_denied"], false);
    let history = terminal["attempt_history"].as_array().unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0]["admission_denied"], true);
    assert!(history[1].get("admission_denied").is_none());
}
