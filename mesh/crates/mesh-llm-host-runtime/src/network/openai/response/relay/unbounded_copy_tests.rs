use super::super::common::CommittedStreamFailureCause;
use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[tokio::test]
async fn unbounded_sse_copy_returns_committed_stream_failure_on_upstream_error() {
    use std::pin::Pin;
    use std::task::{Context, Poll};

    struct FailingReader;
    impl tokio::io::AsyncRead for FailingReader {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _buf: &mut tokio::io::ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            Poll::Ready(Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "upstream reset",
            )))
        }
    }

    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n";
    let parsed = ParsedResponseHeaders {
        header_end: header.len(),
        status_code: 200,
        content_length: None,
        content_type: Some("text/event-stream".to_owned()),
        chunked: false,
        client_nonce: None,
        nonce_origin: None,
        served_model: Default::default(),
    };
    let probe = ResponseProbe {
        buffered: header.as_bytes().to_vec(),
        header_end: header.len(),
        status_code: 200,
        retryable_context_overflow: false,
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (client, _) = listener.accept().await.unwrap();
        let mut client: ClientStream = client.into();
        let mut reader = FailingReader;
        relay_success_response(
            &mut client,
            &mut reader,
            probe,
            parsed,
            ResponseRetryPolicy::next_target_available(false),
            None,
            OpenAiRouteObserver::default(),
            RequestId::new(),
        )
        .await
        .unwrap()
    });
    let mut socket = ClientStream::connect(address).await.unwrap();
    let mut response = Vec::new();
    let _ = socket.read_to_end(&mut response).await;
    let result = task.await.unwrap();
    assert_eq!(
        result,
        RouteAttemptResult::CommittedStreamFailure {
            status_code: 200,
            cause: CommittedStreamFailureCause::Truncated,
        }
    );
}

#[tokio::test]
async fn unbounded_sse_copy_returns_client_disconnected_on_downstream_drop() {
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n";
    let parsed = ParsedResponseHeaders {
        header_end: header.len(),
        status_code: 200,
        content_length: None,
        content_type: Some("text/event-stream".to_owned()),
        chunked: false,
        client_nonce: None,
        nonce_origin: None,
        served_model: Default::default(),
    };
    let probe = ResponseProbe {
        buffered: header.as_bytes().to_vec(),
        header_end: header.len(),
        status_code: 200,
        retryable_context_overflow: false,
    };
    let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (client, _) = listener.accept().await.unwrap();
        let mut client: ClientStream = client.into();
        relay_success_response(
            &mut client,
            &mut upstream_reader,
            probe,
            parsed,
            ResponseRetryPolicy::next_target_available(false),
            None,
            OpenAiRouteObserver::default(),
            RequestId::new(),
        )
        .await
        .unwrap()
    });
    let mut socket = ClientStream::connect(address).await.unwrap();
    let mut head_buf = [0u8; 256];
    let _ = socket.read(&mut head_buf).await.unwrap();
    drop(socket);

    let chunk = vec![b'a'; 16 * 1024];
    for _ in 0..32 {
        if upstream_writer.write_all(&chunk).await.is_err() {
            break;
        }
        tokio::task::yield_now().await;
    }
    drop(upstream_writer);
    let result = task.await.unwrap();
    assert_eq!(result, RouteAttemptResult::ClientDisconnected);
}

#[tokio::test]
async fn unbounded_sse_stream_returns_usage_from_final_frame() {
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n";
    let initial_chunk = "data: {\"choices\":[{\"text\":\"hello\"}]}\n\n";
    let mut buffered = header.as_bytes().to_vec();
    buffered.extend_from_slice(initial_chunk.as_bytes());
    let parsed = ParsedResponseHeaders {
        header_end: header.len(),
        status_code: 200,
        content_length: None,
        content_type: Some("text/event-stream".to_owned()),
        chunked: false,
        client_nonce: None,
        nonce_origin: None,
        served_model: Default::default(),
    };
    let probe = ResponseProbe {
        buffered,
        header_end: header.len(),
        status_code: 200,
        retryable_context_overflow: false,
    };
    let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (client, _) = listener.accept().await.unwrap();
        let mut client: ClientStream = client.into();
        relay_success_response(
            &mut client,
            &mut upstream_reader,
            probe,
            parsed,
            ResponseRetryPolicy::next_target_available(false),
            None,
            OpenAiRouteObserver::default(),
            RequestId::new(),
        )
        .await
        .unwrap()
    });
    upstream_writer
        .write_all(
            b"data: {\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":4,\"total_tokens\":11}}\n\ndata: [DONE]\n\n",
        )
        .await
        .unwrap();
    drop(upstream_writer);
    let mut socket = ClientStream::connect(address).await.unwrap();
    let mut response = Vec::new();
    socket.read_to_end(&mut response).await.unwrap();
    let result = task.await.unwrap();
    match result {
        RouteAttemptResult::Delivered {
            status_code, usage, ..
        } => {
            assert_eq!(status_code, 200);
            let usage = usage.expect("SSE usage frame should be parsed");
            assert_eq!(usage.prompt_tokens, Some(7));
            assert_eq!(usage.completion_tokens, Some(4));
            assert_eq!(usage.total_tokens, Some(11));
        }
        other => panic!("expected Delivered with usage, got {other:?}"),
    }
}

#[tokio::test]
async fn large_declared_content_length_short_eof_returns_committed_stream_failure() {
    let declared_len = 2 * 1024 * 1024;
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {declared_len}\r\n\r\n"
    );
    let parsed = ParsedResponseHeaders {
        header_end: header.len(),
        status_code: 200,
        content_length: Some(declared_len),
        content_type: Some("application/json".to_owned()),
        chunked: false,
        client_nonce: None,
        nonce_origin: None,
        served_model: Default::default(),
    };
    let probe = ResponseProbe {
        buffered: header.as_bytes().to_vec(),
        header_end: header.len(),
        status_code: 200,
        retryable_context_overflow: false,
    };
    let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (client, _) = listener.accept().await.unwrap();
        let mut client: ClientStream = client.into();
        relay_success_response(
            &mut client,
            &mut upstream_reader,
            probe,
            parsed,
            ResponseRetryPolicy::next_target_available(false),
            None,
            OpenAiRouteObserver::default(),
            RequestId::new(),
        )
        .await
        .unwrap()
    });
    upstream_writer
        .write_all(b"{\"partial\":true")
        .await
        .unwrap();
    drop(upstream_writer);
    let mut socket = ClientStream::connect(address).await.unwrap();
    let mut response = Vec::new();
    socket.read_to_end(&mut response).await.unwrap();
    let result = task.await.unwrap();
    assert_eq!(
        result,
        RouteAttemptResult::CommittedStreamFailure {
            status_code: 200,
            cause: CommittedStreamFailureCause::Truncated,
        }
    );
}

#[tokio::test]
async fn chunked_response_ignores_conflicting_content_length_per_rfc9112() {
    let header = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nContent-Length: 999999\r\n\r\n";
    let parsed = ParsedResponseHeaders {
        header_end: header.len(),
        status_code: 200,
        content_length: Some(999_999),
        content_type: Some("application/json".to_owned()),
        chunked: true,
        client_nonce: None,
        nonce_origin: None,
        served_model: Default::default(),
    };
    let probe = ResponseProbe {
        buffered: header.as_bytes().to_vec(),
        header_end: header.len(),
        status_code: 200,
        retryable_context_overflow: false,
    };
    let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (client, _) = listener.accept().await.unwrap();
        let mut client: ClientStream = client.into();
        relay_success_response(
            &mut client,
            &mut upstream_reader,
            probe,
            parsed,
            ResponseRetryPolicy::next_target_available(false),
            None,
            OpenAiRouteObserver::default(),
            RequestId::new(),
        )
        .await
        .unwrap()
    });
    upstream_writer
        .write_all(b"2\r\n{}\r\n0\r\n\r\n")
        .await
        .unwrap();
    drop(upstream_writer);
    let mut socket = ClientStream::connect(address).await.unwrap();
    let mut response = Vec::new();
    socket.read_to_end(&mut response).await.unwrap();
    let result = task.await.unwrap();
    assert!(
        matches!(
            result,
            RouteAttemptResult::Delivered {
                status_code: 200,
                ..
            }
        ),
        "complete chunked framing must ignore conflicting Content-Length, got {result:?}"
    );
}

#[tokio::test]
async fn shared_raw_relay_reports_backend_error_on_http_200_sse_error_frame() {
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n";
    let parsed = ParsedResponseHeaders {
        header_end: header.len(),
        status_code: 200,
        content_length: None,
        content_type: Some("text/event-stream".to_owned()),
        chunked: false,
        client_nonce: None,
        nonce_origin: None,
        served_model: Default::default(),
    };
    let probe = ResponseProbe {
        buffered: header.as_bytes().to_vec(),
        header_end: header.len(),
        status_code: 200,
        retryable_context_overflow: false,
    };
    let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (client, _) = listener.accept().await.unwrap();
        let mut client: ClientStream = client.into();
        relay_success_response(
            &mut client,
            &mut upstream_reader,
            probe,
            parsed,
            ResponseRetryPolicy::next_target_available(false),
            Some("peer-node"),
            OpenAiRouteObserver::default(),
            RequestId::new(),
        )
        .await
        .unwrap()
    });
    let payload = b"data: {\"error\":{\"message\":\"peer failed\"}}\n\ndata: [DONE]\n\n";
    upstream_writer.write_all(payload).await.unwrap();
    drop(upstream_writer);
    let mut socket = ClientStream::connect(address).await.unwrap();
    let mut response = Vec::new();
    socket.read_to_end(&mut response).await.unwrap();
    let result = task.await.unwrap();
    assert!(String::from_utf8_lossy(&response).contains("peer failed"));
    assert_eq!(
        result,
        RouteAttemptResult::CommittedStreamFailure {
            status_code: 200,
            cause: CommittedStreamFailureCause::BackendError,
        }
    );
}
