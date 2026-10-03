use super::*;

mod audit_sanitization;

#[tokio::test]
async fn log_routes_reject_methods_and_invalid_paths_with_bounded_json_errors() {
    let requests = [
        "POST /api/logs/requests HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n".to_string(),
        "GET /api/logs/artifacts/not-a-uuid/extra HTTP/1.1\r\nHost: localhost\r\n\r\n"
            .to_string(),
        "GET /api/logs/requests/00000000-0000-4000-8000-000000000001?limit=1 HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    ];
    for request in requests {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(address, request).await;
        server.await.unwrap().unwrap();
        assert!(
            response.starts_with("HTTP/1.1 405 Method Not Allowed")
                || response.starts_with("HTTP/1.1 404 Not Found")
                || response.starts_with("HTTP/1.1 400 Bad Request")
        );
        let body = json_body(&response);
        assert!(body["error"]["code"].is_string());
        assert!(response.len() < 1024);
        assert!(!response.contains("/Users/") && !response.contains("sqlite"));
    }
}

#[tokio::test]
async fn existing_runtime_event_routes_remain_sse_routes() {
    for path in ["/api/events", "/api/runtime/events"] {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream
            .write_all(format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes())
            .await
            .unwrap();
        let response = read_until_contains(
            &mut stream,
            b"Content-Type: text/event-stream",
            Duration::from_secs(2),
        )
        .await;
        assert!(String::from_utf8_lossy(&response).contains("Content-Type: text/event-stream"));
        drop(stream);
        server.abort();
    }
}

#[tokio::test]
#[serial]
async fn logs_events_sends_semantic_replay_and_fans_out_live_updates() {
    let _temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().unwrap();
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let replay_id = push_sse_event(
        &logging,
        mesh_llm_events::logging::replay::ReplayChannel::Requests,
        request_id,
    );

    let state = build_test_mesh_api().await;
    let (first_addr, first_server) = spawn_management_test_server(state.clone()).await;
    let (second_addr, second_server) = spawn_management_test_server(state).await;
    let request = b"GET /api/logs/events?channel=requests HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n";
    let mut first = TcpStream::connect(first_addr).await.unwrap();
    let mut second = TcpStream::connect(second_addr).await.unwrap();
    first.write_all(request).await.unwrap();
    second.write_all(request).await.unwrap();

    for stream in [&mut first, &mut second] {
        let replay =
            read_until_contains(stream, replay_id.as_bytes(), Duration::from_secs(2)).await;
        let replay = String::from_utf8_lossy(&replay);
        assert!(replay.contains("HTTP/1.1 200 OK"));
        assert!(replay.contains("Content-Type: text/event-stream"));
        assert!(replay.contains("event: log_event"));
        assert!(replay.contains(&replay_id));
        assert!(!replay.contains("private/operator") && !replay.contains("token=secret"));
    }

    let live_id = push_sse_event(
        &logging,
        mesh_llm_events::logging::replay::ReplayChannel::Requests,
        request_id,
    );
    for stream in [&mut first, &mut second] {
        let live = read_until_contains(stream, live_id.as_bytes(), Duration::from_secs(2)).await;
        assert!(String::from_utf8_lossy(&live).contains("event: log_event"));
    }

    drop(first);
    drop(second);
    first_server.abort();
    second_server.abort();
    disable_sse_logging().await;
}

#[tokio::test]
#[serial]
async fn logs_events_subscribes_before_exposing_sse_headers() {
    let _temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().unwrap();
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let mut stream = TcpStream::connect(address).await.unwrap();
    stream
        .write_all(b"GET /api/logs/events?channel=requests HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n")
        .await
        .unwrap();

    let headers = read_until_contains(&mut stream, b"\r\n\r\n", Duration::from_secs(2)).await;
    assert!(String::from_utf8_lossy(&headers).contains("200 OK"));
    let live_id = push_sse_event(
        &logging,
        mesh_llm_events::logging::replay::ReplayChannel::Requests,
        mesh_llm_events::logging::identifiers::RequestId::new(),
    );
    let live = read_until_contains(&mut stream, live_id.as_bytes(), Duration::from_secs(2)).await;
    assert!(String::from_utf8_lossy(&live).contains("event: log_event"));

    drop(stream);
    server.abort();
    disable_sse_logging().await;
}

#[tokio::test]
#[serial]
async fn logs_events_merges_cursors_filters_and_reports_eviction_gaps() {
    let _temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().unwrap();
    let bus = logging.replay_bus().unwrap();
    bus.set_capacity(1_024);
    let wanted = mesh_llm_events::logging::identifiers::RequestId::new();
    let other = mesh_llm_events::logging::identifiers::RequestId::new();
    let initial_request_id = push_sse_event(
        &logging,
        mesh_llm_events::logging::replay::ReplayChannel::Requests,
        wanted,
    );
    push_sse_event(
        &logging,
        mesh_llm_events::logging::replay::ReplayChannel::Operations,
        other,
    );

    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state.clone()).await;
    let mut stream = TcpStream::connect(address).await.unwrap();
    let request = format!(
        "GET /api/logs/events?channel=requests&filter=request_id%3A{}&cursor=v1%3A0.0.0 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\nLast-Event-ID: {}\r\n\r\n",
        wanted.as_uuid(),
        initial_request_id.trim_start_matches("id: "),
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    let headers = read_until_contains(&mut stream, b"\r\n\r\n", Duration::from_secs(2)).await;
    assert!(String::from_utf8_lossy(&headers).contains("200 OK"));
    assert_no_stream_bytes_within(&mut stream, Duration::from_millis(100)).await;
    let live_id = push_sse_event(
        &logging,
        mesh_llm_events::logging::replay::ReplayChannel::Requests,
        wanted,
    );
    let live = read_until_contains(&mut stream, live_id.as_bytes(), Duration::from_secs(2)).await;
    assert!(String::from_utf8_lossy(&live).contains("event: log_event"));
    drop(stream);
    server.abort();

    bus.set_capacity(1);
    push_sse_event(
        &logging,
        mesh_llm_events::logging::replay::ReplayChannel::Requests,
        wanted,
    );
    let (address, server) = spawn_management_test_server(state).await;
    let mut stream = TcpStream::connect(address).await.unwrap();
    stream
        .write_all(b"GET /api/logs/events?channel=requests&cursor=v1%3A0.0.0 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n")
        .await
        .unwrap();
    let gap = read_until_contains(&mut stream, b"event: replay_gap", Duration::from_secs(2)).await;
    let gap = String::from_utf8_lossy(&gap);
    assert!(gap.contains("/api/logs/requests"));
    assert!(!gap.contains("private/operator") && !gap.contains("token=secret"));
    drop(stream);
    server.abort();
    disable_sse_logging().await;
}

#[tokio::test]
async fn logs_events_rejects_invalid_raw_requests_before_sse_headers() {
    let oversized = std::iter::repeat_n("unknown=x", 33)
        .collect::<Vec<_>>()
        .join("&");
    let requests = [
        "GET /api/logs/events?channel=requests HTTP/1.1\r\nHost: localhost\r\nAccept: application/json\r\n\r\n".to_string(),
        "GET /api/logs/events?channel=requests HTTP/1.1\r\nAccept: text/event-stream\r\n\r\n".to_string(),
        "POST /api/logs/events?channel=requests HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\nContent-Length: 0\r\n\r\n".to_string(),
        "GET /api/logs/events?channel=requests&filter=request_id%ZZ HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n".to_string(),
        format!("GET /api/logs/events?{oversized} HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n"),
        "GET /api/logs/events?channel=requests HTTP/1.1\r\nHost: hostile.example\r\nAccept: text/event-stream\r\n\r\n".to_string(),
        "GET /api/logs/events?channel=requests HTTP/1.1\r\nHost: localhost\r\nOrigin: https://hostile.example\r\nAccept: text/event-stream\r\n\r\n".to_string(),
    ];
    for request in requests {
        let state = build_test_mesh_api().await;
        let (address, server) = spawn_management_test_server(state).await;
        let response = send_management_request(address, request).await;
        server.await.unwrap().unwrap();
        assert!(
            response.starts_with("HTTP/1.1 400")
                || response.starts_with("HTTP/1.1 403")
                || response.starts_with("HTTP/1.1 405")
                || response.starts_with("HTTP/1.1 406"),
            "{response}"
        );
        assert!(!response.contains("Content-Type: text/event-stream"));
        assert!(response.len() < 1024);
        assert!(json_body(&response)["error"].is_object());
    }
}

#[tokio::test]
#[serial]
async fn logs_events_audit_reconcile_failure_writes_one_terminal_frame_then_eof() {
    let _temporary_directory = install_sse_logging().await;
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let mut stream = TcpStream::connect(address).await.unwrap();
    stream
        .write_all(
            b"GET /api/logs/events?audit=true&cursor=a1:18446744073709551615 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n",
        )
        .await
        .unwrap();

    let mut response = read_until_contains(
        &mut stream,
        b"\"code\":\"audit_reconcile_failed\"",
        Duration::from_secs(2),
    )
    .await;
    let mut remainder = Vec::new();
    tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut remainder))
        .await
        .expect("audit reconcile failure stream must reach EOF")
        .unwrap();
    response.extend_from_slice(&remainder);
    let response = String::from_utf8(response).unwrap();
    assert_eq!(response.matches("event: stream_error").count(), 1);
    assert_eq!(
        response
            .matches("\"code\":\"audit_reconcile_failed\"")
            .count(),
        1
    );

    server.await.unwrap().unwrap();
    disable_sse_logging().await;
}

#[tokio::test]
#[serial]
async fn logs_events_stops_after_a_disconnected_tcp_client() {
    let _temporary_directory = install_sse_logging().await;
    let logging = crate::logging_runtime_state().unwrap();
    let state = build_test_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let mut stream = TcpStream::connect(address).await.unwrap();
    stream
        .write_all(b"GET /api/logs/events?channel=requests HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n")
        .await
        .unwrap();
    let _headers = read_until_contains(&mut stream, b"\r\n\r\n", Duration::from_secs(2)).await;
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    drop(stream);
    push_sse_event(
        &logging,
        mesh_llm_events::logging::replay::ReplayChannel::Requests,
        request_id,
    );
    tokio::time::sleep(Duration::from_millis(10)).await;
    push_sse_event(
        &logging,
        mesh_llm_events::logging::replay::ReplayChannel::Requests,
        request_id,
    );
    let completed = tokio::time::timeout(Duration::from_secs(2), server).await;
    assert!(
        completed.is_ok(),
        "SSE server did not release disconnected client"
    );
    assert!(completed.unwrap().unwrap().is_ok());
    disable_sse_logging().await;
}
