//! Consumer-specific bounded response projection using existing native Hyper owners.
use crate::command::DynResult;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use serde_json::Value;

const BODY_LIMIT: usize = 1024 * 1024;
#[derive(Debug)]
pub(in crate::automation) struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}
pub(super) async fn exchange(
    base: &str,
    suffix: &str,
    body: Option<&Value>,
    deadline: std::time::Instant,
    cancellation: &crate::process::Cancellation,
) -> DynResult<Response> {
    exchange_with_authorization(
        base,
        suffix,
        body,
        deadline,
        cancellation,
        Some("mesh-llm-ci"),
    )
    .await
}
pub(in crate::automation) async fn exchange_with_authorization(
    base: &str,
    suffix: &str,
    body: Option<&Value>,
    deadline: std::time::Instant,
    cancellation: &crate::process::Cancellation,
    authorization: Option<&str>,
) -> DynResult<Response> {
    let uri: hyper::Uri = format!("{}/{}", base.trim_end_matches('/'), suffix).parse()?;
    if uri.scheme_str() == Some("https")
        || uri
            .host()
            .is_none_or(|h| h.parse::<std::net::Ipv4Addr>().is_err())
    {
        return super::curl_transport::exchange(
            &uri.to_string(),
            body,
            deadline,
            cancellation,
            authorization,
        );
    }
    let socket = tokio::net::TcpStream::connect((
        uri.host().ok_or("missing HTTP host")?,
        uri.port_u16().unwrap_or(80),
    ))
    .await?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(socket)).await?;
    let bytes = body
        .map(serde_json::to_vec)
        .transpose()?
        .unwrap_or_default();
    let mut request = hyper::Request::builder()
        .method(if body.is_some() { "POST" } else { "GET" })
        .uri(uri.path_and_query().ok_or("missing HTTP path")?.as_str())
        .header(
            "host",
            uri.authority().ok_or("missing HTTP authority")?.as_str(),
        )
        .header("content-type", "application/json")
        .header("connection", "close");
    if let Some(token) = authorization {
        request = request.header("authorization", format!("Bearer {token}"));
    }
    let request = request.body(Full::new(Bytes::from(bytes)))?;
    let response = async {
        let mut response = sender.send_request(request).await?;
        let status = response.status().as_u16();
        let mut bytes = Vec::new();
        while let Some(frame) = response.body_mut().frame().await {
            if let Some(data) = frame?.data_ref() {
                if data.len() > BODY_LIMIT.saturating_sub(bytes.len()) {
                    return Err("guardrail HTTP body exceeds 1 MiB".into());
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
    tokio::select! { result=&mut response=>result, result=connection=>{result?;response.await} }
}
pub(super) fn successful(body: &[u8], streaming: bool) -> DynResult<bool> {
    if !streaming {
        let value: Value = serde_json::from_slice(body)?;
        if value.get("error").is_some_and(|e| !e.is_null()) {
            return Ok(false);
        }
        let message = &value["choices"][0]["message"];
        return Ok(message["content"].as_str().is_some_and(|s| !s.is_empty())
            || message["tool_calls"]
                .as_array()
                .is_some_and(|a| !a.is_empty())
            || value["output_text"].as_str().is_some_and(|s| !s.is_empty()));
    }
    let text = std::str::from_utf8(body)?;
    let mut done = false;
    let mut content = false;
    let mut finished = false;
    for line in text.lines() {
        let Some(data) = line.trim().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if done {
            return Err("guardrail SSE data after terminal marker".into());
        }
        if data == "[DONE]" {
            done = true;
            continue;
        }
        let event: Value = serde_json::from_str(data)?;
        if event.get("error").is_some_and(|e| !e.is_null()) {
            return Err("guardrail SSE server error".into());
        }
        if let Some(choices) = event["choices"].as_array() {
            for choice in choices {
                content |= choice["delta"]["content"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty());
                finished |= choice["finish_reason"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty());
            }
        }
    }
    if !done || !finished {
        return Err("guardrail SSE missing finish reason or terminal marker".into());
    }
    Ok(content)
}

/// Await supervised curl off the async coordinator, including its cancellation cleanup.
/// Callers must await this owned path rather than race/drop it with an outer timeout.
pub(in crate::automation) async fn exchange_owned_with_authorization(
    base: &str,
    suffix: &str,
    body: Option<&Value>,
    deadline: std::time::Instant,
    cancellation: &crate::process::Cancellation,
    authorization: Option<&str>,
) -> DynResult<Response> {
    let endpoint = format!("{}/{}", base.trim_end_matches('/'), suffix);
    let uri: hyper::Uri = endpoint.parse()?;
    if uri.scheme_str() != Some("https")
        && uri
            .host()
            .is_some_and(|h| h.parse::<std::net::Ipv4Addr>().is_ok())
    {
        return exchange_with_authorization(
            base,
            suffix,
            body,
            deadline,
            cancellation,
            authorization,
        )
        .await;
    }
    let body = body.cloned();
    let token = cancellation.clone();
    let authorization = authorization.map(str::to_owned);
    tokio::task::spawn_blocking(move || {
        super::curl_transport::exchange(
            &endpoint,
            body.as_ref(),
            deadline,
            &token,
            authorization.as_deref(),
        )
        .map_err(|_| "bounded curl request refused".to_owned())
    })
    .await?
    .map_err(Into::into)
}
