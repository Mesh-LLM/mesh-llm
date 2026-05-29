use super::super::{MeshApi, http::respond_error};
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
    stream.write_all(head.as_bytes()).await?;

    let mut body = response.into_body();
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|err| anyhow::anyhow!("MCP response body error: {err}"))?;
        if let Some(chunk) = frame.data_ref() {
            stream.write_all(chunk).await?;
        }
    }
    Ok(())
}
