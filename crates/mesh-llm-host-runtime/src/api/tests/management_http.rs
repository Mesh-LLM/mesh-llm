#[tokio::test]
async fn test_management_request_parser_handles_fragmented_post_body() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let body = br#"{"text":"fragmented"}"#;
    let headers = format!(
        "POST /api/plugins/demo/http/post HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    let header_split = headers.find("\r\n\r\n").unwrap() + 2;
    let body_split = 8;
    let (server_ready_tx, server_ready_rx) = oneshot::channel();

    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        server_ready_tx.send(()).unwrap();
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            proxy::read_http_request(&mut stream),
        )
        .await
        .unwrap()
        .unwrap()
    });

    let client = tokio::spawn(async move {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.set_nodelay(true).unwrap();
        server_ready_rx.await.unwrap();
        stream
            .write_all(&headers.as_bytes()[..header_split])
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        stream
            .write_all(&headers.as_bytes()[header_split..])
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        stream.write_all(&body[..body_split]).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        stream.write_all(&body[body_split..]).await.unwrap();
        let mut sink = [0u8; 1];
        let _ = stream.read(&mut sink).await;
    });

    client.await.unwrap();
    let request = server.await.unwrap();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/api/plugins/demo/http/post");
    assert_eq!(http_body_text(&request.raw), "{\"text\":\"fragmented\"}");
}

#[tokio::test]
async fn test_api_events_sends_initial_payload_and_updates() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state.clone()).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();

    let initial = read_until_contains(&mut stream, b"data: {", Duration::from_secs(2)).await;
    let initial_text = String::from_utf8_lossy(&initial);
    assert!(initial_text.contains("HTTP/1.1 200 OK"));
    assert!(initial_text.contains("Content-Type: text/event-stream"));
    assert!(initial_text.contains("\"llama_ready\":false"));

    state.update(true, true).await;
    let updated =
        read_until_contains(&mut stream, b"\"llama_ready\":true", Duration::from_secs(2)).await;
    let updated_text = String::from_utf8_lossy(&updated);
    assert!(updated_text.contains("\"llama_ready\":true"));
    assert!(updated_text.contains("\"is_host\":true"));

    drop(stream);
    handle.abort();
}

#[tokio::test]
async fn test_api_events_push_publication_state_updates() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state.clone()).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();

    let _initial = read_until_contains(
        &mut stream,
        b"\"publication_state\":\"private\"",
        Duration::from_secs(2),
    )
    .await;

    state
        .set_publication_state(crate::api::PublicationState::PublishFailed)
        .await;
    let updated = read_until_contains(
        &mut stream,
        b"\"publication_state\":\"publish_failed\"",
        Duration::from_secs(2),
    )
    .await;
    let updated_text = String::from_utf8_lossy(&updated);
    assert!(updated_text.contains("\"publication_state\":\"publish_failed\""));

    drop(stream);
    handle.abort();
}
