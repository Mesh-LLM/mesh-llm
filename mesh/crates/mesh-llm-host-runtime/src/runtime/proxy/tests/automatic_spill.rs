// Exercise directive selection and bounded spill through ordinary HTTP ingress.

#[tokio::test]
async fn automatic_ingress_caps_spill_at_four_distinct_models() {
    for directive in ["mesh", "auto"] {
        let mut entries = Vec::new();
        let mut upstreams = Vec::new();
        for name in ["worker-a", "worker-b", "worker-c", "worker-d", "worker-e"] {
            let (port, captured, task) =
                spawn_status_upstream("429 Too Many Requests", r#"{"error":"busy"}"#).await;
            entries.push((name, port));
            upstreams.push((name, captured, task));
        }
        let (address, proxy) = spawn_api_proxy_test_harness(local_targets(&entries)).await;
        let body = json!({
            "model": directive,
            "messages": [{"role":"user", "content":"hello"}]
        }).to_string();
        let wire = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let response = tokio::time::timeout(
            Duration::from_secs(30),
            send_request_and_read_response(address, vec![wire.into_bytes()]),
        ).await.unwrap();
        assert!(response.starts_with("HTTP/1.1 503"), "{response}");
        assert!(response.contains("exhausted after 4 model(s)"), "{response}");
        let mut called = 0;
        for (name, mut captured, task) in upstreams {
            if let Ok(raw) = captured.try_recv() {
                called += 1;
                let raw = String::from_utf8(raw).unwrap();
                assert!(raw.contains(&format!("\"model\":\"{name}\"")), "{raw}");
                task.await.unwrap();
            } else {
                task.abort();
                assert!(task.await.unwrap_err().is_cancelled());
            }
        }
        assert_eq!(called, 4, "{directive} must stop before the fifth model");
        proxy.abort();
        assert!(proxy.await.unwrap_err().is_cancelled());
    }
}

#[tokio::test]
async fn automatic_ingress_without_plugin_selects_concrete_model() {
    for directive in ["mesh", "auto"] {
        let (port, captured, upstream) = spawn_capturing_upstream(
            r#"{"id":"core","model":"concrete-worker","choices":[]}"#,
        ).await;
        let (address, proxy) =
            spawn_api_proxy_test_harness(local_targets(&[("concrete-worker", port)])).await;
        let body = json!({
            "model": directive,
            "messages": [{"role":"user", "content":"hello"}]
        }).to_string();
        let wire = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let response = tokio::time::timeout(
            Duration::from_secs(30),
            send_request_and_read_response(address, vec![wire.into_bytes()]),
        ).await.unwrap();
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.contains("concrete-worker"), "{response}");
        assert!(!response.to_ascii_lowercase().contains("x-moa-"), "{response}");
        let raw = String::from_utf8(captured.await.unwrap()).unwrap();
        assert!(raw.contains(r#""model":"concrete-worker""#), "{raw}");
        upstream.await.unwrap();
        proxy.abort();
        assert!(proxy.await.unwrap_err().is_cancelled());
    }
}

// A served answer the quality probe dislikes is still the answer when no other
// host of that model remains: the spill is for uncommitted exhaustion, not for a
// reply truncated by the client's own `max_tokens` (CI smoke, 2026-11-08).
#[tokio::test]
async fn automatic_ingress_delivers_length_truncated_reply_from_sole_model() {
    for directive in ["mesh", "auto"] {
        let (port, captured, upstream) = spawn_capturing_upstream(
            r#"{"id":"chatcmpl-short","object":"chat.completion","model":"sole-worker","choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","content":"Hello, how are"}}]}"#,
        ).await;
        let (address, proxy) =
            spawn_api_proxy_test_harness(local_targets(&[("sole-worker", port)])).await;
        let body = json!({
            "model": directive,
            "messages": [{"role":"user", "content":"Say hi."}],
            "max_tokens": 4
        }).to_string();
        let wire = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let response = tokio::time::timeout(
            Duration::from_secs(30),
            send_request_and_read_response(address, vec![wire.into_bytes()]),
        ).await.unwrap();
        assert!(response.starts_with("HTTP/1.1 200"), "{directive}: {response}");
        assert!(response.contains("chatcmpl-short"), "{directive}: {response}");
        assert!(!response.contains("exhausted"), "{directive}: {response}");
        let raw = String::from_utf8(captured.await.unwrap()).unwrap();
        assert!(raw.contains(r#""model":"sole-worker""#), "{raw}");
        upstream.await.unwrap();
        proxy.abort();
        assert!(proxy.await.unwrap_err().is_cancelled());
    }
}

/// An upstream that answers each connection with the next `(status, body)`
/// in `replies`, counting every request it reads.
async fn spawn_scripted_upstream(
    replies: &[(&str, &str)],
) -> (
    u16,
    Arc<std::sync::atomic::AtomicUsize>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let replies = replies
        .iter()
        .map(|(status, body)| ((*status).to_string(), (*body).to_string()))
        .collect::<Vec<_>>();
    let requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let request_count = Arc::clone(&requests);
    let handle = tokio::spawn(async move {
        for (status, body) in replies {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let _ = read_raw_http_request(&mut stream).await;
            request_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let reply = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(reply.as_bytes()).await;
            let _ = stream.shutdown().await;
        }
    });
    (port, requests, handle)
}

// One model on one host: a single failure cools that host, but the cooling
// must not lock automatic requests out of the only placement while naming the
// model would still reach it. The next automatic request must try the host.
#[tokio::test]
async fn automatic_ingress_retries_sole_host_after_one_failure() {
    for directive in ["mesh", "auto"] {
        let (port, requests, upstream) = spawn_scripted_upstream(&[
            ("503 Service Unavailable", r#"{"error":"warming"}"#),
            (
                "200 OK",
                r#"{"id":"chatcmpl-sole","object":"chat.completion","model":"sole-worker","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}]}"#,
            ),
        ])
        .await;
        let (address, proxy) =
            spawn_api_proxy_test_harness(local_targets(&[("sole-worker", port)])).await;
        let body = json!({
            "model": directive,
            "messages": [{"role":"user", "content":"hello"}]
        })
        .to_string();
        let wire = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let first = tokio::time::timeout(
            Duration::from_secs(30),
            send_request_and_read_response(address, vec![wire.clone().into_bytes()]),
        )
        .await
        .unwrap();
        assert!(!first.starts_with("HTTP/1.1 200"), "{directive}: {first}");
        assert_eq!(
            requests.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "{directive}: the first request reaches the host once"
        );
        let second = tokio::time::timeout(
            Duration::from_secs(30),
            send_request_and_read_response(address, vec![wire.into_bytes()]),
        )
        .await
        .unwrap();
        assert_eq!(
            requests.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "{directive}: the second request must reach the sole host; got {second}"
        );
        assert!(second.starts_with("HTTP/1.1 200"), "{directive}: {second}");
        assert!(second.contains("chatcmpl-sole"), "{directive}: {second}");
        upstream.await.unwrap();
        proxy.abort();
        assert!(proxy.await.unwrap_err().is_cancelled());
    }
}

// A request that names no model is routed automatically, so it spills across
// models like `mesh` instead of relaying the first model's 429.
#[tokio::test]
async fn model_less_ingress_spills_to_another_model() {
    let mut entries = Vec::new();
    let mut upstreams = Vec::new();
    for name in ["worker-a", "worker-b"] {
        let (port, captured, task) =
            spawn_status_upstream("429 Too Many Requests", r#"{"error":"busy"}"#).await;
        entries.push((name, port));
        upstreams.push((captured, task));
    }
    let (address, proxy) = spawn_api_proxy_test_harness(local_targets(&entries)).await;
    let body = json!({"messages": [{"role":"user", "content":"hello"}]}).to_string();
    let wire = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let response = tokio::time::timeout(
        Duration::from_secs(30),
        send_request_and_read_response(address, vec![wire.into_bytes()]),
    )
    .await
    .unwrap();
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    assert!(response.contains("exhausted after 2 model(s)"), "{response}");
    for (mut captured, task) in upstreams {
        assert!(captured.try_recv().is_ok(), "both models must be tried");
        task.await.unwrap();
    }
    proxy.abort();
    assert!(proxy.await.unwrap_err().is_cancelled());
}
