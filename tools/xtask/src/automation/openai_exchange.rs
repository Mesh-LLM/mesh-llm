//! Shared bounded OpenAI streaming transport for repository automation.
pub(super) mod stream;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use std::time::Instant;
use stream::Stream;

pub(super) async fn request(
    base: &str,
    body: &serde_json::Value,
    qualification_probe: bool,
) -> Result<stream::Evidence, Box<dyn std::error::Error + Send + Sync>> {
    let started = Instant::now();
    let uri: hyper::Uri = format!("{}/chat/completions", base.trim_end_matches('/')).parse()?;
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
    let request = hyper::Request::builder()
        .method("POST")
        .uri(uri.path_and_query().ok_or("missing request path")?.as_str())
        .header("host", uri.authority().ok_or("missing authority")?.as_str())
        .header("content-type", "application/json")
        .header("authorization", "Bearer EMPTY")
        .body(Full::new(Bytes::from(serde_json::to_vec(body)?)))?;
    let response = async {
        let mut response = sender.send_request(request).await?;
        if !response.status().is_success() {
            return Err(format!("HTTP {}", response.status()).into());
        }
        let mut stream = Stream::default();
        while let Some(frame) = response.body_mut().frame().await {
            if let Some(bytes) = frame?.data_ref() {
                stream.consume(bytes, started.elapsed())?;
                if stream.terminal() {
                    break;
                }
            }
        }
        Ok(stream.finish(started.elapsed(), qualification_probe)?)
    };
    tokio::pin!(response);
    tokio::select! { result = &mut response => result, result = connection => { result?; response.await } }
}
