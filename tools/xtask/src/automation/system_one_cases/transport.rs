use super::{Failure, Options, Result};
use crate::{command::DynResult, process::Cancellation};
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use serde_json::Value;
use std::time::Duration;
pub(super) struct Client<'a> {
    pub options: &'a Options,
    pub cancellation: Cancellation,
}
pub(super) fn validate_endpoint(value: &str) -> DynResult<()> {
    if value.len() > 8192 || value.contains(['#', '\r', '\n', '\0']) {
        return Err("unsupported System One endpoint".into());
    }
    let uri: hyper::Uri = value.parse()?;
    if uri.scheme_str() != Some("http")
        || uri.host().is_none()
        || uri.authority().is_none_or(|a| a.as_str().contains('@'))
        || uri.query().is_some()
    {
        return Err(
            "System One requires an HTTP endpoint without credentials, query or fragment".into(),
        );
    }
    socket_host(&uri).map_err(|_| "invalid System One host")?;
    socket_port(&uri).map_err(|_| "invalid System One port")?;
    Ok(())
}
impl Client<'_> {
    pub(super) async fn json(&self, payload: &Value) -> Result<(u16, Value)> {
        let body = serde_json::to_vec(payload)
            .map_err(|_| Failure::Case("request serialization failed"))?;
        self.send("POST", body).await
    }
    pub(super) async fn send(&self, method: &str, body: Vec<u8>) -> Result<(u16, Value)> {
        if body.len() > 65536 {
            return Err(Failure::Case("System One request exceeds 64 KiB"));
        }
        let exchange = tokio::time::timeout(self.options.timeout, self.exchange(method, body));
        let cancellation = async {
            loop {
                if self.cancellation.is_cancelled() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        };
        let bytes = tokio::select! { result=exchange=>result.map_err(|_|Failure::Transport("System One request deadline exceeded"))??,()=cancellation=>return Err(Failure::Transport("System One operation cancelled")) };
        let json = serde_json::from_slice(&bytes.1)
            .map_err(|_| Failure::Case("System One response is not JSON"))?;
        Ok((bytes.0, json))
    }
    async fn exchange(&self, method: &str, body: Vec<u8>) -> Result<(u16, Vec<u8>)> {
        let uri: hyper::Uri = self
            .options
            .endpoint
            .parse()
            .map_err(|_| Failure::Transport("invalid endpoint"))?;
        let stream = tokio::net::TcpStream::connect((socket_host(&uri)?, socket_port(&uri)?))
            .await
            .map_err(|_| Failure::Transport("System One connection failed"))?;
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .map_err(|_| Failure::Transport("System One handshake failed"))?;
        let request = hyper::Request::builder()
            .method(method)
            .uri(uri.path())
            .header(
                "host",
                uri.authority()
                    .ok_or(Failure::Transport("missing authority"))?
                    .as_str(),
            )
            .header("content-type", "application/json")
            .header("connection", "close")
            .body(Full::new(Bytes::from(body)))
            .map_err(|_| Failure::Transport("invalid HTTP request"))?;
        let response = async {
            let mut response = sender
                .send_request(request)
                .await
                .map_err(|_| Failure::Transport("System One request failed"))?;
            let status = response.status().as_u16();
            if response.status().is_redirection() {
                return Err(Failure::Transport("System One redirects are not admitted"));
            }
            let mut bytes = Vec::new();
            while let Some(frame) = response.body_mut().frame().await {
                let frame =
                    frame.map_err(|_| Failure::Transport("System One response body incomplete"))?;
                if let Some(data) = frame.data_ref() {
                    if data.len() > 1048576usize.saturating_sub(bytes.len()) {
                        return Err(Failure::Transport("System One response exceeds 1 MiB"));
                    }
                    bytes.extend_from_slice(data);
                }
            }
            Ok((status, bytes))
        };
        tokio::pin!(response);
        tokio::select! {result=&mut response=>result,result=connection=>{result.map_err(|_|Failure::Transport("System One connection failed"))?;response.await}}
    }
}

pub(super) fn socket_host(uri: &hyper::Uri) -> Result<&str> {
    let host = uri.host().ok_or(Failure::Transport("missing host"))?;
    if let Some(address) = host.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        address
            .parse::<std::net::Ipv6Addr>()
            .map_err(|_| Failure::Transport("invalid IPv6 endpoint"))?;
        Ok(address)
    } else {
        Ok(host)
    }
}

// http::Authority::port() returns None for invalid/overflowing explicit ports.
// Inspect the suffix after its already parsed host before choosing default 80.
pub(super) fn socket_port(uri: &hyper::Uri) -> Result<u16> {
    let authority = uri
        .authority()
        .ok_or(Failure::Transport("missing authority"))?;
    let suffix = authority
        .as_str()
        .strip_prefix(authority.host())
        .ok_or(Failure::Transport("invalid authority"))?;
    if suffix.is_empty() {
        return Ok(80);
    }
    let value = suffix
        .strip_prefix(':')
        .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .ok_or(Failure::Transport("invalid explicit port"))?;
    value
        .parse()
        .map_err(|_| Failure::Transport("invalid explicit port"))
}
