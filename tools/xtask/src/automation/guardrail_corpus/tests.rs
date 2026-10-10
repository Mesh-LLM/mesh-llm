use super::*;
#[test]
fn fixed_request_corpus_preserves_tools_schema_modes_and_declared_retry() {
    let cases = corpus::cases();
    assert_eq!(cases.len(), 5);
    assert_eq!(cases.iter().map(|c| c.declared_retries).sum::<u64>(), 1);
    for mode in ["off", "metrics", "enforce"] {
        for case in &cases {
            let body = case.request("fixture", mode);
            assert_eq!(body["mesh_guardrails"], mode != "off");
            assert_eq!(body["messages"][0]["content"], case.prompt);
        }
    }
    assert_eq!(
        cases[1].overrides["tools"][0]["function"]["parameters"]["required"],
        json!(["left", "right"])
    );
    assert_eq!(
        cases[3].overrides["response_format"]["json_schema"]["schema"]["required"],
        json!(["status", "count", "note"])
    );
    assert!(!cases[4].supported());
}
#[test]
fn sse_refuses_missing_finish_done_malformed_error_and_postterminal_data() {
    let valid=b"data: {\"choices\":[{\"delta\":{\"content\":\"pass\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    assert!(transport::successful(valid, true).unwrap());
    for bad in [
        b"data: {\"choices\":[{\"delta\":{\"content\":\"pass\"}}]}\n".as_slice(),
        b"data: [DONE]\n",
        b"data: malformed\ndata: [DONE]\n",
        b"data: {\"error\":{\"message\":\"failed\"}}\ndata: [DONE]\n",
        b"data: [DONE]\ndata: {}\n",
    ] {
        assert!(transport::successful(bad, true).is_err());
    }
}
#[test]
fn interpolation_and_synthetic_latency_are_explicit() {
    assert_eq!(latency_summary(vec![1.0, 3.0])["p50"], 2.0);
    assert!((latency_summary(vec![1.0, 3.0])["p95"].as_f64().unwrap() - 2.9).abs() < 1e-12);
    let case = corpus::cases().remove(0);
    let value = synthetic(&case, 0, "off");
    assert_eq!(value, synthetic(&case, 0, "off"));
    assert_eq!(
        value["latency_origin"],
        "deterministic_synthetic_not_measured"
    );
}
#[test]
fn pre_cancelled_run_publishes_no_synthetic_requests() {
    let root = tempfile::tempdir().unwrap();
    let input = Options {
        base: "fake://fixture".into(),
        model: "fixture".into(),
        mode: "metrics".into(),
        trials: 2,
        out: root.path().join("out.json"),
        seconds: 2,
    };
    let cancellation = Cancellation::default();
    cancellation.cancel();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (report, failed) = runtime.block_on(execute(&input, &cancellation)).unwrap();
    assert!(failed);
    assert_eq!(report["total_requests"], 0);
    assert_eq!(report["status"], "incomplete");
    assert!(report["latency_ms"]["mean"].is_null());
}
#[test]
fn local_http_inflight_cancel_and_deadline_stop_owned_future_without_fake_rows() {
    for cancelled in [false, true] {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            use tokio::io::AsyncReadExt;
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let cancellation = Cancellation::default();
            let peer_cancel = cancellation.clone();
            let (tx, rx) = tokio::sync::oneshot::channel();
            let peer = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = [0; 4096];
                let n = socket.read(&mut bytes).await.unwrap();
                assert!(n > 0);
                let _ = tx.send(());
                if cancelled {
                    peer_cancel.cancel();
                }
                std::future::pending::<()>().await;
            });
            let until = Instant::now() + Duration::from_millis(600);
            let result = bounded(
                transport::exchange(
                    &format!("http://127.0.0.1:{port}/v1"),
                    "models",
                    None,
                    until,
                    &cancellation,
                ),
                until,
                &cancellation,
            )
            .await;
            peer.abort();
            let joined = peer.await;
            let admitted = rx.await.is_ok();
            assert!(joined.unwrap_err().is_cancelled());
            assert!(admitted);
            assert_eq!(cancellation.is_cancelled(), cancelled);
            assert!(result.is_err());
        });
    }
}
#[test]
fn local_http_body_cap_refuses_large_completed_response() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let peer = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = [0; 4096];
            assert!(socket.read(&mut bytes).await.unwrap() > 0);
            let body = vec![b'x'; 1024 * 1024 + 1];
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(header.as_bytes()).await;
            let _ = socket.write_all(&body).await;
        });
        let cancellation = Cancellation::default();
        let until = Instant::now() + Duration::from_secs(2);
        let result = bounded(
            transport::exchange(
                &format!("http://127.0.0.1:{port}/v1"),
                "models",
                None,
                until,
                &cancellation,
            ),
            until,
            &cancellation,
        )
        .await;
        peer.abort();
        let joined = peer.await;
        assert!(joined.is_ok() || joined.unwrap_err().is_cancelled());
        assert!(matches!(result,Err(e) if e.to_string().contains("1 MiB")));
    });
}
