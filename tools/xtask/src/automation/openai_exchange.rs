//! Shared bounded HTTP completion transport for repository automation.
use crate::command::DynResult;
pub(super) mod json_completion;
pub(super) mod native_completion;
pub(super) mod stream;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use std::time::{Duration, Instant};
use stream::Stream;

type TransportResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

trait Decoder {
    type Evidence;
    fn accepts(&self, status: hyper::StatusCode) -> bool;
    fn consume(&mut self, bytes: &[u8], elapsed: Duration) -> Result<(), String>;
    fn terminal(&self) -> bool;
    fn finish(self, elapsed: Duration) -> Result<Self::Evidence, String>;
}

struct OpenAiDecoder {
    stream: Stream,
    probe: bool,
    exact_ok: bool,
    maximum: usize,
    bytes_seen: usize,
}

impl Decoder for OpenAiDecoder {
    type Evidence = stream::Evidence;
    fn accepts(&self, status: hyper::StatusCode) -> bool {
        if self.exact_ok {
            status == hyper::StatusCode::OK
        } else {
            status.is_success()
        }
    }
    fn consume(&mut self, bytes: &[u8], elapsed: Duration) -> Result<(), String> {
        if self.maximum < 64 * 1024 * 1024
            && bytes.len() > self.maximum.saturating_sub(self.bytes_seen)
        {
            return Err("OpenAI completion response exceeds caller bound".into());
        }
        self.bytes_seen += bytes.len();
        self.stream.consume(bytes, elapsed)
    }
    fn terminal(&self) -> bool {
        self.stream.terminal()
    }
    fn finish(self, elapsed: Duration) -> Result<Self::Evidence, String> {
        self.stream.finish(elapsed, self.probe)
    }
}

pub(super) async fn request(
    base: &str,
    body: &serde_json::Value,
    qualification_probe: bool,
) -> TransportResult<stream::Evidence> {
    let started = Instant::now();
    let uri = format!("{}/chat/completions", base.trim_end_matches('/')).parse()?;
    post(
        uri,
        body,
        started,
        true,
        OpenAiDecoder {
            stream: Stream::default(),
            probe: qualification_probe,
            exact_ok: false,
            maximum: 64 * 1024 * 1024,
            bytes_seen: 0,
        },
    )
    .await
}

/// Cache throughput workers bound each observed response to 1 MiB.
pub(super) async fn cache_request(
    base: &str,
    body: &serde_json::Value,
) -> TransportResult<stream::Evidence> {
    let started = Instant::now();
    let uri = format!("{}/chat/completions", base.trim_end_matches('/')).parse()?;
    post(
        uri,
        body,
        started,
        true,
        OpenAiDecoder {
            stream: Stream::default(),
            probe: false,
            exact_ok: true,
            maximum: 1024 * 1024,
            bytes_seen: 0,
        },
    )
    .await
}

// This private transport owns the connection future in the same cancellation scope as
// response decoding. Callers supply deadline/interrupt ownership around the future.
async fn post<D: Decoder>(
    uri: hyper::Uri,
    body: &serde_json::Value,
    started: Instant,
    dummy_authorization: bool,
    mut decoder: D,
) -> TransportResult<D::Evidence> {
    if uri.scheme_str() != Some("http") {
        return Err("replay execution requires HTTP".into());
    }
    let connection = tokio::net::TcpStream::connect((
        uri.host().ok_or("missing host")?,
        uri.port_u16().unwrap_or(80),
    ))
    .await?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(connection)).await?;
    let mut request = hyper::Request::builder()
        .method("POST")
        .uri(uri.path_and_query().ok_or("missing request path")?.as_str())
        .header("host", uri.authority().ok_or("missing authority")?.as_str())
        .header("content-type", "application/json");
    if dummy_authorization {
        request = request.header("authorization", "Bearer EMPTY");
    }
    let request = request.body(Full::new(Bytes::from(serde_json::to_vec(body)?)))?;
    let response = async {
        let mut response = sender.send_request(request).await?;
        if !decoder.accepts(response.status()) {
            return Err(format!("HTTP {}", response.status()).into());
        }
        while let Some(frame) = response.body_mut().frame().await {
            if let Some(bytes) = frame?.data_ref() {
                decoder.consume(bytes, started.elapsed())?;
                if decoder.terminal() {
                    break;
                }
            }
        }
        Ok(decoder.finish(started.elapsed())?)
    };
    tokio::pin!(response);
    tokio::select! { result = &mut response => result, result = connection => { result?; response.await } }
}

pub(super) async fn get(url: &str) -> DynResult<Vec<u8>> {
    let uri: hyper::Uri = url.parse()?;
    if uri.scheme_str() != Some("http") {
        return Err("replay readiness requires HTTP".into());
    }
    let stream = tokio::net::TcpStream::connect((
        uri.host().ok_or("missing readiness host")?,
        uri.port_u16().unwrap_or(80),
    ))
    .await?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(stream)).await?;
    let request = hyper::Request::builder()
        .uri(
            uri.path_and_query()
                .ok_or("missing readiness path")?
                .as_str(),
        )
        .header(
            "host",
            uri.authority()
                .ok_or("missing readiness authority")?
                .as_str(),
        )
        .header("connection", "close")
        .body(Full::new(Bytes::new()))?;
    let response = async {
        let mut response = sender.send_request(request).await?;
        if !response.status().is_success() {
            return Err("readiness HTTP error".into());
        }
        let mut bytes = Vec::new();
        while let Some(frame) = response.body_mut().frame().await {
            if let Some(data) = frame?.data_ref() {
                if data.len() > (1024 * 1024_usize).saturating_sub(bytes.len()) {
                    return Err("readiness response exceeds 1 MiB".into());
                }
                bytes.extend_from_slice(data);
            }
        }
        Ok(bytes)
    };
    tokio::pin!(response);
    tokio::select! {result = &mut response => result, result = connection => { result?; response.await }}
}
