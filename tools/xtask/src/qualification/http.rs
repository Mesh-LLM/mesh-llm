//! Minimal blocking loopback HTTP for qualification drivers: one
//! current-thread Tokio runtime per call, returning status and body without
//! rejecting non-2xx responses (negative cases need them).

use crate::command::DynResult;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use std::time::Duration;

/// HTTP status and raw body.
pub(crate) struct Response {
    pub(crate) status: u16,
    pub(crate) body: Vec<u8>,
}

pub(crate) fn request(
    url: &str,
    body: Option<Vec<u8>>,
    method: &str,
    timeout: Duration,
) -> DynResult<Response> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let outcome = runtime
        .block_on(async { tokio::time::timeout(timeout, exchange(url, body, method)).await });
    let response = match outcome {
        Ok(result) => result?,
        Err(_) => return Err(format!("qualification HTTP request timed out: {url}").into()),
    };
    runtime.shutdown_timeout(Duration::from_millis(100));
    Ok(response)
}

/// POST/GET JSON, requiring an object body in the response.
pub(crate) fn request_json(
    url: &str,
    body: Option<serde_json::Value>,
    method: &str,
    timeout: Duration,
) -> DynResult<(u16, serde_json::Value)> {
    let payload = body.map(|body| serde_json::to_vec(&body)).transpose()?;
    let response = request(url, payload, method, timeout)?;
    let value: serde_json::Value = serde_json::from_slice(&response.body)?;
    if !value.is_object() {
        return Err(format!("{url} returned a non-object").into());
    }
    Ok((response.status, value))
}

async fn exchange(url: &str, body: Option<Vec<u8>>, method: &str) -> DynResult<Response> {
    let uri: hyper::Uri = url.parse()?;
    if uri.scheme_str() != Some("http") {
        return Err("qualification drivers require an HTTP endpoint".into());
    }
    let stream = tokio::net::TcpStream::connect((
        uri.host().ok_or("missing HTTP host")?,
        uri.port_u16().unwrap_or(80),
    ))
    .await?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(stream)).await?;
    let request = hyper::Request::builder()
        .method(method)
        .uri(uri.path_and_query().ok_or("missing HTTP path")?.as_str())
        .header(
            "host",
            uri.authority().ok_or("missing HTTP authority")?.as_str(),
        )
        .header("content-type", "application/json")
        .header("connection", "close")
        .body(Full::new(Bytes::from(body.unwrap_or_default())))?;
    let response = async {
        let response = sender.send_request(request).await?;
        let status = response.status().as_u16();
        let mut body = response.into_body();
        let mut bytes = Vec::new();
        while let Some(frame) = body.frame().await {
            if let Some(data) = frame?.data_ref() {
                if data.len() > (16 * 1024 * 1024_usize).saturating_sub(bytes.len()) {
                    return Err("qualification response exceeds 16 MiB".into());
                }
                bytes.extend_from_slice(data);
            }
        }
        Ok(Response {
            status,
            body: bytes,
        })
    };
    tokio::pin!(response);
    tokio::select! {
        result = &mut response => result,
        result = connection => { result?; response.await }
    }
}
