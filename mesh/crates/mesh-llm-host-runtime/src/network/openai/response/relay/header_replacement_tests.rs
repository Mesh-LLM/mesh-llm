use super::*;
use tokio::io::AsyncReadExt;

async fn assert_replacement_preserves_body(bounded: bool) {
    let body = b"the body must remain byte-for-byte intact\r\nx-request-id: body-data";
    let length = if bounded {
        format!("Content-Length: {}\r\n", body.len())
    } else {
        String::new()
    };
    let header = format!(
        "HTTP/1.1 200 OK\r\n{length}x-mesh-served-by: a-very-long-stale-peer-identity\r\nx-request-id: stale-request\r\n\r\n"
    );
    let parsed = try_parse_response_headers(header.as_bytes())
        .unwrap()
        .unwrap();
    let mut buffered = header.into_bytes();
    buffered.extend_from_slice(body);
    let probe = ResponseProbe {
        buffered,
        header_end: parsed.header_end,
        status_code: 200,
        retryable_context_overflow: false,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let request_id = RequestId::new();
    let relay = tokio::spawn(async move {
        let (client, _) = listener.accept().await.unwrap();
        let mut client: ClientStream = client.into();
        relay_success_response(
            &mut client,
            &mut tokio::io::empty(),
            probe,
            parsed,
            ResponseRetryPolicy::next_target_available(false),
            Some("new"),
            OpenAiRouteObserver::default(),
            request_id,
        )
        .await
        .unwrap();
    });
    let mut client = ClientStream::connect(address).await.unwrap();
    let mut response = Vec::new();
    client.read_to_end(&mut response).await.unwrap();
    relay.await.unwrap();
    let parsed = try_parse_response_headers(&response).unwrap().unwrap();
    let head = std::str::from_utf8(&response[..parsed.header_end]).unwrap();
    assert_eq!(head.matches("x-mesh-served-by:").count(), 1);
    assert_eq!(head.matches("x-request-id:").count(), 1);
    assert!(head.contains("x-mesh-served-by: new\r\n"));
    assert!(head.contains(&format!("x-request-id: {}\r\n", request_id.as_uuid())));
    assert_eq!(&response[parsed.header_end..], body);
}

#[tokio::test]
async fn bounded_relay_replaces_shorter_headers_without_corrupting_body() {
    assert_replacement_preserves_body(true).await;
}

#[tokio::test]
async fn unbounded_relay_replaces_shorter_headers_without_corrupting_body() {
    assert_replacement_preserves_body(false).await;
}
