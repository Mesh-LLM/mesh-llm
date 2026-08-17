use super::super::{
    MeshApi,
    http::{respond_error, write_managed_response_head},
};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use tokio::{io::AsyncWriteExt, net::TcpStream};

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    raw_request: &[u8],
) -> anyhow::Result<()> {
    let request = match parse_request(raw_request) {
        Ok(request) => request,
        Err(err) => {
            respond_error(stream, 400, &err.to_string()).await?;
            return Ok(());
        }
    };
    let endpoint = {
        let inner = state.inner.lock().await;
        inner.mcp_http.clone()
    };
    let response = endpoint.handle(request).await;
    write_response(stream, response).await
}

fn parse_request(raw_request: &[u8]) -> anyhow::Result<http::Request<Full<Bytes>>> {
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut parsed = httparse::Request::new(&mut headers);
    let header_len = match parsed.parse(raw_request)? {
        httparse::Status::Complete(header_len) => header_len,
        httparse::Status::Partial => anyhow::bail!("Incomplete HTTP request"),
    };

    let method = parsed.method.unwrap_or("GET");
    let path = parsed.path.unwrap_or("/mcp");
    let mut builder = http::Request::builder()
        .method(method)
        .uri(path)
        .version(http_version(parsed.version));
    for header in parsed.headers.iter() {
        builder = builder.header(header.name, header.value);
    }
    builder
        .body(Full::new(Bytes::copy_from_slice(
            &raw_request[header_len..],
        )))
        .map_err(Into::into)
}

fn http_version(version: Option<u8>) -> http::Version {
    match version {
        Some(0) => http::Version::HTTP_10,
        Some(1) => http::Version::HTTP_11,
        Some(2) => http::Version::HTTP_2,
        Some(3) => http::Version::HTTP_3,
        _ => http::Version::HTTP_11,
    }
}

async fn write_response(
    stream: &mut TcpStream,
    response: http::Response<http_body_util::combinators::BoxBody<Bytes, std::convert::Infallible>>,
) -> anyhow::Result<()> {
    let status = response.status();
    let reason = status.canonical_reason().unwrap_or("");
    let mut head = format!("HTTP/1.1 {} {}\r\n", status.as_u16(), reason);
    let has_connection_header = response.headers().contains_key(http::header::CONNECTION);
    for (name, value) in response.headers() {
        head.push_str(name.as_str());
        head.push_str(": ");
        head.push_str(value.to_str().unwrap_or(""));
        head.push_str("\r\n");
    }
    if !has_connection_header {
        head.push_str("Connection: close\r\n");
    }
    head.push_str("\r\n");
    write_managed_response_head(stream, head.into_bytes()).await?;

    let mut body = response.into_body();
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|err| anyhow::anyhow!("MCP response body error: {err}"))?;
        if let Some(chunk) = frame.data_ref() {
            stream.write_all(chunk).await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use mesh_llm_events::logging::identifiers::RequestId;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    use super::write_response;

    #[tokio::test]
    async fn mcp_success_response_uses_the_scoped_id_and_completes() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
        let (mut server, _) = listener.accept().await.unwrap();
        let request_id = RequestId::new();
        let service = Arc::new(crate::logging::LoggingService::new_disabled(
            Default::default(),
        ));
        let lifecycle = crate::logging::ManagementRequestLifecycle::register(
            Arc::clone(&service),
            request_id,
            "management_post",
        );
        let response = http::Response::builder()
            .status(http::StatusCode::CREATED)
            .header("X-Request-Id", "plugin-supplied-id")
            .body(Full::new(Bytes::from_static(b"{} ")).boxed())
            .unwrap();

        super::super::super::management_lifecycle::scope(lifecycle, async {
            write_response(&mut server, response)
                .await
                .expect("write MCP response");
        })
        .await;
        server.shutdown().await.unwrap();

        let mut response = Vec::new();
        client.read_to_end(&mut response).await.unwrap();
        let response = String::from_utf8(response).unwrap();
        assert!(response.starts_with("HTTP/1.1 201 Created\r\n"));
        assert!(response.contains(&format!("x-request-id: {}\r\n", request_id.as_uuid())));
        assert!(!response.contains("plugin-supplied-id"));
        assert_eq!(response.matches("x-request-id:").count(), 1);
        assert_eq!(
            service
                .registry_ref()
                .get_recent(&request_id.as_uuid().to_string())
                .expect("MCP lifecycle terminal entry")
                .state,
            "completed"
        );
    }
}
