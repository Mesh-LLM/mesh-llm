use super::*;
use crate::network::target_health::TargetHealthOutcome;
use iroh::SecretKey;

#[test]
fn reservation_does_not_reintroduce_target_cooled_during_cache_lookup() {
    let affinity = AffinityRouter::with_config(true, true);
    let first = election::InferenceTarget::Local(9001);
    let second = election::InferenceTarget::Local(9002);
    let ranked = RankedCandidates::with_equivalent_prefix(vec![first.clone(), second.clone()], 2);
    let (_, pressure) = affinity
        .reserve_route("qwen", &ranked.ordered, 2, &second, true)
        .unwrap();
    // This is the snapshot passed to the asynchronous cache lookup.
    let before_lookup = affinity.route_eligible_candidates("qwen", &ranked.ordered);
    assert_eq!(before_lookup, ranked.ordered);
    // Another request times out while that lookup is suspended.
    affinity.record_target_outcome(Some("qwen"), &first, TargetHealthOutcome::Timeout);
    for cache_target in [None, Some(first.clone())] {
        let route = select_and_reserve_model_route(
            &election::ModelTargets::default(),
            &ranked,
            "qwen",
            None,
            &affinity,
            cache_target,
        )
        .unwrap();
        assert_eq!(route.selection.target, second);
        assert_eq!(route.ordered, vec![second.clone()]);
        assert!(!route.selection.affinity_applied);
        assert_eq!(affinity.stats_snapshot().reservation_active, 2);
    }
    drop(pressure);
    assert_eq!(affinity.stats_snapshot().reservation_active, 0);
}

#[test]
fn equivalent_local_targets_route_around_the_busy_one_in_any_list_order() {
    // Three equivalent targets; the busy one may sit in any slot, including
    // the first. New sessions reach only the idle two, about evenly.
    let targets: Vec<_> = (9001..=9003)
        .map(election::InferenceTarget::Local)
        .collect();
    for busy_slot in 0..3 {
        let affinity = AffinityRouter::new();
        let ranked = RankedCandidates::with_equivalent_prefix(targets.clone(), 3);
        let (_, held) = affinity
            .reserve_route(
                "qwen",
                std::slice::from_ref(&targets[busy_slot]),
                1,
                &targets[busy_slot],
                true,
            )
            .unwrap();
        let mut counts = std::collections::HashMap::new();
        for _ in 0..300 {
            let route = select_and_reserve_model_route(
                &election::ModelTargets::default(),
                &ranked,
                "qwen",
                None,
                &affinity,
                None,
            )
            .unwrap();
            assert_eq!(route.ordered[0], route.selection.target);
            *counts
                .entry(route.selection.target.clone())
                .or_insert(0usize) += 1;
        }
        assert_eq!(
            counts.get(&targets[busy_slot]),
            None,
            "busy slot {busy_slot} took new sessions: {counts:?}"
        );
        for (slot, target) in targets.iter().enumerate() {
            if slot != busy_slot {
                let count = counts.get(target).copied().unwrap_or(0);
                assert!(
                    (110..=190).contains(&count),
                    "idle slot {slot} should take about half: {counts:?}"
                );
            }
        }
        drop(held);
        assert_eq!(affinity.stats_snapshot().reservation_active, 0);
    }
}

#[test]
fn refreshed_spread_window_does_not_admit_lower_ranked_fallback() {
    let first = election::InferenceTarget::Local(9001);
    let second = election::InferenceTarget::Local(9002);
    let fallback = election::InferenceTarget::Local(9003);
    // Cover both a shortened equivalent run and one removed entirely.
    for equivalent_prefix in [2, 1] {
        let affinity = AffinityRouter::new();
        let ranked = RankedCandidates::with_equivalent_prefix(
            vec![first.clone(), second.clone(), fallback.clone()],
            equivalent_prefix,
        );
        let (_, _pressure) = affinity
            .reserve_route("qwen", &ranked.ordered, 3, &second, true)
            .unwrap();
        affinity.record_target_outcome(Some("qwen"), &first, TargetHealthOutcome::Timeout);
        let route = select_and_reserve_model_route(
            &election::ModelTargets::default(),
            &ranked,
            "qwen",
            None,
            &affinity,
            None,
        )
        .unwrap();
        assert_eq!(route.selection.target, second);
        assert_eq!(route.ordered, vec![second.clone(), fallback.clone()]);
    }
}

#[test]
fn refreshed_route_preserves_healthy_cache_and_session_affinity_under_pressure() {
    let affinity = AffinityRouter::with_config(true, true);
    let ranked = RankedCandidates::with_equivalent_prefix(
        vec![
            election::InferenceTarget::Local(9001),
            election::InferenceTarget::Local(9002),
        ],
        2,
    );
    let body = serde_json::json!({"user": "same-session"});
    for (parsed_body, cache_target) in
        [(None, Some(ranked.ordered[0].clone())), (Some(&body), None)]
    {
        let select = || {
            select_and_reserve_model_route(
                &election::ModelTargets::default(),
                &ranked,
                "qwen",
                parsed_body,
                &affinity,
                cache_target.clone(),
            )
            .unwrap()
        };
        let held = select();
        let next = select();
        assert!(held.selection.affinity_applied);
        assert!(next.selection.affinity_applied);
        assert_eq!(next.selection.target, held.selection.target);
    }
    assert_eq!(affinity.stats_snapshot().reservation_active, 0);
}

#[test]
fn refreshed_route_keeps_all_cooling_availability_fallback() {
    let affinity = AffinityRouter::new();
    let ranked = RankedCandidates::with_equivalent_prefix(
        vec![
            election::InferenceTarget::Local(9001),
            election::InferenceTarget::Local(9002),
        ],
        2,
    );
    for target in &ranked.ordered {
        affinity.record_target_outcome(Some("qwen"), target, TargetHealthOutcome::Timeout);
    }
    let route = select_and_reserve_model_route(
        &election::ModelTargets::default(),
        &ranked,
        "qwen",
        None,
        &affinity,
        None,
    )
    .unwrap();
    assert!(ranked.ordered.contains(&route.selection.target));
    assert_eq!(route.ordered.len(), 2);
}

#[test]
fn empty_refreshed_route_does_not_reserve() {
    let affinity = AffinityRouter::new();
    assert!(
        select_and_reserve_model_route(
            &election::ModelTargets::default(),
            &RankedCandidates::with_equivalent_prefix(vec![], 0,),
            "qwen",
            None,
            &affinity,
            Some(election::InferenceTarget::Local(9001)),
        )
        .is_none()
    );
    assert_eq!(affinity.stats_snapshot().reservation_active, 0);
}

/// A request whose shared scaffold covers the 512 cached tokens.
fn cached_request() -> mesh_llm_routing::request_cost::RequestCostInput {
    mesh_llm_routing::request_cost::RequestCostInput::heuristic(1_024, 512)
}

async fn cache_context(
    prefix_hash: u64,
    target: election::InferenceTarget,
) -> (
    mesh::Node,
    AffinityRouter,
    election::InferenceTarget,
    TargetSelection,
    RouteModelState,
) {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Host { http_port: 9337 })
        .await
        .expect("test node");
    let affinity = AffinityRouter::with_config(true, true);
    let selection = TargetSelection {
        target: target.clone(),
        prefix_hash: Some(prefix_hash),
        cache_target: Some(target.clone()),
        affinity_applied: true,
    };
    let state = RouteModelState {
        route_started: Instant::now(),
        attempts: 1,
        timeout_attempts: 0,
        refreshed: false,
        prompt_bytes: None,
    };
    (node, affinity, target, selection, state)
}

#[tokio::test]
async fn authoritative_local_miss_invalidates_inventory_and_lease() {
    let prefix_hash = 0xfeed_beef;
    let (node, affinity, target, selection, state) =
        cache_context(prefix_hash, election::InferenceTarget::Local(9337)).await;
    node.record_local_cache_hit("qwen", prefix_hash, 512, 24, 0);
    node.observe_local_prefill_cost("qwen", Some(1_000));
    affinity.remember_cache_lease("qwen", prefix_hash, &target);
    let context = DeliveredRouteModelContext {
        node: &node,
        model: "qwen",
        target: &target,
        selection: &selection,
        affinity: &affinity,
        state: &state,
    };

    update_local_cache_evidence(
        &context,
        200,
        Some(&TokenUsage {
            prompt_tokens: Some(536),
            cached_prompt_tokens: Some(0),
            ..TokenUsage::default()
        }),
        None,
    );

    assert_eq!(
        node.select_cache_target(
            "qwen",
            prefix_hash,
            std::slice::from_ref(&target),
            cached_request(),
        )
        .await,
        None
    );
    assert_eq!(
        affinity.lookup_cache_lease("qwen", prefix_hash, std::slice::from_ref(&target)),
        None
    );
}

#[tokio::test]
async fn authoritative_remote_miss_invalidates_lease_but_not_local_inventory() {
    let prefix_hash = 0xfeed_beef;
    let mut bytes = [0u8; 32];
    bytes[0] = 1;
    let remote = election::InferenceTarget::Remote(SecretKey::from_bytes(&bytes).public());
    let (node, affinity, target, selection, state) = cache_context(prefix_hash, remote).await;
    let local = election::InferenceTarget::Local(9337);
    node.record_local_cache_hit("qwen", prefix_hash, 512, 24, 0);
    node.observe_local_prefill_cost("qwen", Some(1_000));
    affinity.remember_cache_lease("qwen", prefix_hash, &target);
    let context = DeliveredRouteModelContext {
        node: &node,
        model: "qwen",
        target: &target,
        selection: &selection,
        affinity: &affinity,
        state: &state,
    };

    update_local_cache_evidence(
        &context,
        200,
        Some(&TokenUsage {
            prompt_tokens: Some(536),
            cached_prompt_tokens: Some(0),
            ..TokenUsage::default()
        }),
        None,
    );

    assert_eq!(
        affinity.lookup_cache_lease("qwen", prefix_hash, std::slice::from_ref(&target)),
        None
    );
    assert_eq!(
        node.select_cache_target(
            "qwen",
            prefix_hash,
            std::slice::from_ref(&local),
            cached_request(),
        )
        .await,
        Some(local)
    );
}

#[tokio::test]
async fn missing_cache_usage_does_not_refute_positive_evidence() {
    let prefix_hash = 0xfeed_beef;
    let (node, affinity, target, selection, state) =
        cache_context(prefix_hash, election::InferenceTarget::Local(9337)).await;
    node.record_local_cache_hit("qwen", prefix_hash, 512, 24, 0);
    node.observe_local_prefill_cost("qwen", Some(1_000));
    affinity.remember_cache_lease("qwen", prefix_hash, &target);
    let context = DeliveredRouteModelContext {
        node: &node,
        model: "qwen",
        target: &target,
        selection: &selection,
        affinity: &affinity,
        state: &state,
    };

    update_local_cache_evidence(
        &context,
        200,
        Some(&TokenUsage {
            prompt_tokens: Some(536),
            cached_prompt_tokens: None,
            ..TokenUsage::default()
        }),
        None,
    );

    assert_eq!(
        node.select_cache_target(
            "qwen",
            prefix_hash,
            std::slice::from_ref(&target),
            cached_request(),
        )
        .await,
        Some(target.clone())
    );
    assert_eq!(
        affinity.lookup_cache_lease("qwen", prefix_hash, std::slice::from_ref(&target)),
        Some(target)
    );
}

#[tokio::test]
async fn timing_without_cache_usage_still_calibrates_local_prefill_cost() {
    let prefix_hash = 0xfeed_beef;
    let calibration_probe = prefix_hash + 1;
    let (node, affinity, target, selection, state) =
        cache_context(prefix_hash, election::InferenceTarget::Local(9337)).await;
    let context = DeliveredRouteModelContext {
        node: &node,
        model: "qwen",
        target: &target,
        selection: &selection,
        affinity: &affinity,
        state: &state,
    };

    update_local_cache_evidence(
        &context,
        200,
        Some(&TokenUsage {
            prompt_tokens: Some(536),
            cached_prompt_tokens: None,
            ..TokenUsage::default()
        }),
        Some(CacheCostObservation {
            queue_delay_micros: 4_000,
            restore_micros: 8_000,
            prefill_micros_per_token: Some(250),
        }),
    );

    node.record_local_cache_hit("qwen", calibration_probe, 512, 24, 0);
    let entry = node
        .cache_affinity_inventory
        .lock()
        .unwrap()
        .probe_local("qwen", calibration_probe)
        .expect("calibrated local evidence");
    assert_eq!(entry.prefill_micros_per_token, 250);
}

#[tokio::test]
async fn local_hit_records_measured_queue_restore_and_prefill_costs() {
    let prefix_hash = 0xfeed_beef;
    let (node, affinity, target, selection, state) =
        cache_context(prefix_hash, election::InferenceTarget::Local(9337)).await;
    let context = DeliveredRouteModelContext {
        node: &node,
        model: "qwen",
        target: &target,
        selection: &selection,
        affinity: &affinity,
        state: &state,
    };

    update_local_cache_evidence(
        &context,
        200,
        Some(&TokenUsage {
            prompt_tokens: Some(536),
            cached_prompt_tokens: Some(512),
            ..TokenUsage::default()
        }),
        Some(CacheCostObservation {
            queue_delay_micros: 4_000,
            restore_micros: 8_000,
            prefill_micros_per_token: Some(250),
        }),
    );

    let entry = node
        .cache_affinity_inventory
        .lock()
        .unwrap()
        .probe_local("qwen", prefix_hash)
        .expect("measured local evidence");
    assert_eq!(entry.queue_delay_micros, 4_000);
    assert_eq!(entry.restore_micros, 8_000);
    assert_eq!(entry.prefill_micros_per_token, 250);
}

/// Regression for the sink-drain gap `ndizazzo` flagged on PR #1944: the
/// FIRST target's response carries an `X-Capsule-Id` header but is itself
/// retryable (a context-overflow 400), so the loop moves on to a SECOND
/// target that delivers successfully without ever asserting a
/// capsule_id. Without draining the sink before the next attempt, the
/// terminal event would misattribute the first (failed) peer's asserted
/// capsule_id to the second (actually serving) peer.
#[tokio::test]
async fn a_retry_to_a_different_target_does_not_inherit_the_previous_peers_stale_capsule_id() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn drain_request_headers(stream: &mut tokio::net::TcpStream) {
        let mut buffer = [0u8; 4096];
        let mut accumulated = Vec::new();
        loop {
            let read = stream.read(&mut buffer).await.expect("backend request");
            if read == 0 {
                break;
            }
            accumulated.extend_from_slice(&buffer[..read]);
            if accumulated.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
    }

    let model = "qwen";
    let connect_order = Arc::new(AtomicUsize::new(0));

    let mut backend_ports = Vec::new();
    let mut backend_tasks = Vec::new();
    for _ in 0..2 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        backend_ports.push(listener.local_addr().unwrap().port());
        let connect_order = Arc::clone(&connect_order);
        backend_tasks.push(tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            drain_request_headers(&mut stream).await;
            // Whichever backend the loop dials FIRST plays the failed,
            // capsule_id-asserting peer; whichever it dials second plays
            // the eventual, silent success -- this pins the scenario to
            // connection order rather than to which port the router
            // happens to rank first.
            let response = if connect_order.fetch_add(1, Ordering::SeqCst) == 0 {
                let body = r#"{"error":"prompt exceeds the context window limit"}"#;
                format!(
                    "HTTP/1.1 400 Bad Request\r\nX-Capsule-Id: cap-from-the-first-peer\r\nx-mesh-weights-digest: digest-from-the-first-peer\r\nx-mesh-model-canonical-ref: ref-from-the-first-peer\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                )
            } else {
                let body = r#"{"id":"c1","model":"qwen","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                )
            };
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
        }));
    }

    let downstream = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let downstream_address = downstream.local_addr().unwrap();
    let client = tokio::spawn(async move {
        let mut stream = tokio::net::TcpStream::connect(downstream_address)
            .await
            .unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
    });
    let (downstream_stream, _) = downstream.accept().await.unwrap();

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    let affinity = AffinityRouter::new();
    let sink = PeerCapsuleIdSink::new();
    let served_by = ServedByNodeIdSink::new();

    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        model.to_string(),
        backend_ports
            .iter()
            .map(|port| election::InferenceTarget::Local(*port))
            .collect(),
    );

    let request_body = r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}]}"#;
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        request_body.len(),
        request_body
    )
    .into_bytes();
    let request = BufferedHttpRequest {
        raw,
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: request_body.len(),
        completion_tokens: None,
        stream: None,
        model_name: Some(model.to_owned()),
        request_object_request_ids: Vec::new(),
        response_adapter: ResponseAdapter::None,
        correlation_id: None,
        exchange_observation_id: None,
        automatic_route: false,
    };

    let outcome = route_model_request(
        node,
        downstream_stream.into(),
        &targets,
        model,
        &request,
        RouteModelRequestContext {
            local_execution_only: false,
            automatic_retry: None,
            required_tokens: None,
            affinity: &affinity,
            route_observer: OpenAiRouteObserver::default(),
            served_by_header: None,
            peer_capsule_id: Some(&sink),
            served_by_node_id: Some(&served_by),
        },
    )
    .await;

    assert!(
        matches!(
            outcome,
            RouteDispatchOutcome::RespondedWithUsage {
                status_code: 200,
                ..
            }
        ),
        "expected the surviving target to deliver 200, got {outcome:?}"
    );

    for task in backend_tasks {
        task.await.unwrap();
    }
    client.await.unwrap();

    assert_eq!(
        sink.take(),
        None,
        "the sink must not carry the FIRST (retried-away-from) peer's \
         capsule_id onto a later attempt that never asserted one itself"
    );
    assert!(sink.take_served_model().is_empty(), "nor its served model");
    assert_eq!(
        served_by.take(),
        None,
        "a local target delivered, so no peer is named as the server"
    );
}

/// Drive one `route_model_request` across one mock backend per entry in
/// `backends` and return the router's outcome together with the raw
/// response bytes the downstream client actually received.
///
/// Responses are assigned by connection order, so target ranking cannot
/// make a scenario skip its intended first response. Backend tasks are
/// deliberately not awaited: a router that stops at the first target
/// leaves the later backends parked in `accept`, and awaiting them would
/// hang instead of reporting the missing failover.
async fn route_across_mock_backends(
    backends: &[(u16, &str)],
) -> (RouteDispatchOutcome, String, usize) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const MODEL: &str = "qwen";

    async fn drain_request_headers(stream: &mut tokio::net::TcpStream) {
        let mut buffer = [0u8; 4096];
        let mut accumulated = Vec::new();
        while let Ok(read) = stream.read(&mut buffer).await {
            if read == 0 {
                break;
            }
            accumulated.extend_from_slice(&buffer[..read]);
            if accumulated.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
    }

    let mut listeners = Vec::new();
    let mut ports = Vec::new();
    for _ in backends {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock backend must bind");
        ports.push(listener.local_addr().expect("mock backend address").port());
        listeners.push(listener);
    }

    let connect_order = Arc::new(AtomicUsize::new(0));
    let response_plan = Arc::new(
        backends
            .iter()
            .map(|(status, body)| (*status, (*body).to_owned()))
            .collect::<Vec<_>>(),
    );
    let mut backend_tasks = Vec::new();
    for listener in listeners {
        let connect_order = Arc::clone(&connect_order);
        let response_plan = Arc::clone(&response_plan);
        backend_tasks.push(tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("mock backend accept");
            let response_index = connect_order.fetch_add(1, Ordering::SeqCst);
            let Some((status, body)) = response_plan.get(response_index) else {
                return;
            };
            let reason = match status {
                200 => "OK",
                402 => "Payment Required",
                _ => "Internal Server Error",
            };
            drain_request_headers(&mut stream).await;
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len(),
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("mock backend write");
            stream.shutdown().await.expect("mock backend shutdown");
        }));
    }

    let downstream = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("downstream must bind");
    let downstream_address = downstream.local_addr().expect("downstream address");
    let client = tokio::spawn(async move {
        let mut stream = tokio::net::TcpStream::connect(downstream_address)
            .await
            .expect("client must connect");
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .await
            .expect("client must read the response");
        response
    });
    let (downstream_stream, _) = downstream.accept().await.expect("downstream accept");

    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node must start");
    let affinity = AffinityRouter::new();

    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        MODEL.to_string(),
        ports
            .iter()
            .map(|port| election::InferenceTarget::Local(*port))
            .collect(),
    );

    let request_body = r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}]}"#;
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        request_body.len(),
        request_body
    )
    .into_bytes();
    let request = BufferedHttpRequest {
        raw,
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: request_body.len(),
        completion_tokens: None,
        stream: None,
        model_name: Some(MODEL.to_owned()),
        request_object_request_ids: Vec::new(),
        response_adapter: ResponseAdapter::None,
        correlation_id: None,
        exchange_observation_id: None,
        automatic_route: false,
    };

    let outcome = route_model_request(
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

    let raw_response = tokio::time::timeout(std::time::Duration::from_secs(20), client)
        .await
        .expect("the client must receive a response")
        .expect("the client task must not panic");
    let backend_attempts = connect_order.load(Ordering::SeqCst);
    for task in backend_tasks {
        task.abort();
    }
    (
        outcome,
        String::from_utf8_lossy(&raw_response).into_owned(),
        backend_attempts,
    )
}

#[tokio::test]
async fn busy_or_unavailable_replica_fails_over_before_relay() {
    for status in [429, 502, 503, 504] {
        let (_, response, attempts) =
            route_across_mock_backends(&[(status, "busy"), (200, served_completion_body())]).await;
        assert_eq!(attempts, 2, "status {status}");
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    }
}

#[tokio::test]
async fn internal_server_error_does_not_retry_another_replica() {
    let (_, response, attempts) =
        route_across_mock_backends(&[(500, "internal error"), (200, served_completion_body())])
            .await;
    assert_eq!(attempts, 1);
    assert!(response.starts_with("HTTP/1.1 500"), "{response}");
}

fn gated_replica_body() -> &'static str {
    r#"{"error":{"message":"a gate plugin on this host refused the request","type":"gate_unsatisfied"}}"#
}

fn served_completion_body() -> &'static str {
    r#"{"id":"c1","model":"qwen","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#
}

/// #2059, the failover half. A replica whose gate plugin answers 402 is
/// refusing *this* serving endpoint, not the request: another replica of
/// the same model can still serve it. The route loop treats a peer
/// timeout, unavailability and context-overflow 400 as "try the next
/// target"; a 402 before any relayed byte gets the same treatment.
#[tokio::test]
async fn a_replica_whose_gate_refuses_fails_over_to_the_next_replica() {
    let (outcome, raw_response, backend_attempts) =
        route_across_mock_backends(&[(402, gated_replica_body()), (200, served_completion_body())])
            .await;

    assert_eq!(
        backend_attempts, 2,
        "the gated backend must be tried before the standby"
    );

    assert!(
        raw_response.starts_with("HTTP/1.1 200"),
        "the standby must have served the request, client saw: {raw_response}"
    );
    assert!(
        matches!(
            outcome,
            RouteDispatchOutcome::RespondedWithUsage {
                status_code: 200,
                ..
            }
        ),
        "expected the surviving replica to deliver 200, got {outcome:?}"
    );
}

/// The other half of the same decision: when no other replica exists the
/// client must still see the gate's 402 — with the terms the gate put in
/// its response — not an unavailable-model error.
#[tokio::test]
async fn a_model_served_only_by_a_gated_replica_still_relays_the_402() {
    let (outcome, raw_response, backend_attempts) =
        route_across_mock_backends(&[(402, gated_replica_body())]).await;

    assert_eq!(backend_attempts, 1, "the only backend must be attempted");

    assert!(
        raw_response.starts_with("HTTP/1.1 402"),
        "the gate's 402 must reach the client, client saw: {raw_response}"
    );
    assert!(
        matches!(
            outcome,
            RouteDispatchOutcome::Responded(402)
                | RouteDispatchOutcome::RespondedWithUsage {
                    status_code: 402,
                    ..
                }
                | RouteDispatchOutcome::RespondedWithDigests {
                    status_code: 402,
                    ..
                }
        ),
        "expected a terminal 402, got {outcome:?}"
    );
}

#[test]
fn automatic_reservation_never_restores_all_cooling_placements() {
    let affinity = AffinityRouter::with_config(true, true);
    let ranked =
        RankedCandidates::with_equivalent_prefix(vec![election::InferenceTarget::Local(9001)], 1);
    // The candidate snapshot was healthy before the asynchronous cache lookup.
    assert_eq!(
        affinity.route_strict_eligible_candidates("qwen", &ranked.ordered),
        ranked.ordered
    );
    affinity.record_target_outcome(
        Some("qwen"),
        &ranked.ordered[0],
        TargetHealthOutcome::Shedding {
            retry_after: std::time::Duration::from_secs(30),
        },
    );
    for cache_target in [None, Some(ranked.ordered[0].clone())] {
        assert!(
            select_and_reserve_model_route_with_health(
                &election::ModelTargets::default(),
                &ranked,
                "qwen",
                None,
                &affinity,
                cache_target,
                true,
                &Default::default(),
            )
            .is_none()
        );
        assert_eq!(affinity.stats_snapshot().reservation_active, 0);
    }
}

#[test]
fn two_choice_keeps_penalized_and_deprioritized_hosts_as_spillover() {
    // Equal weights: before, the draw ignored health order, so a penalized or
    // deprioritized host still took about half the new sessions.
    let healthy = election::InferenceTarget::Local(9001);
    let penalized = election::InferenceTarget::Local(9002);
    let affinity = AffinityRouter::new();
    // A timeout then a success: no cooldown left, but a reputation penalty.
    affinity.record_target_outcome(Some("qwen"), &penalized, TargetHealthOutcome::Timeout);
    affinity.record_target_outcome(Some("qwen"), &penalized, TargetHealthOutcome::Success);
    let ranked =
        RankedCandidates::with_equivalent_prefix(vec![healthy.clone(), penalized.clone()], 2);
    for _ in 0..50 {
        let route = select_and_reserve_model_route(
            &election::ModelTargets::default(),
            &ranked,
            "qwen",
            None,
            &affinity,
            None,
        )
        .unwrap();
        assert_eq!(route.selection.target, healthy);
    }

    let peer = SecretKey::generate().public();
    let near = election::InferenceTarget::Local(9003);
    let spill = election::InferenceTarget::Remote(peer);
    let deprioritized = std::collections::HashSet::from([peer]);
    let ranked = RankedCandidates::with_equivalent_prefix(vec![spill.clone(), near.clone()], 2);
    for _ in 0..50 {
        let route = select_and_reserve_model_route_with_health(
            &election::ModelTargets::default(),
            &ranked,
            "qwen",
            None,
            &AffinityRouter::new(),
            None,
            false,
            &deprioritized,
        )
        .unwrap();
        assert_eq!(route.selection.target, near);
    }
}
