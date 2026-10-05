//! Loopback HTTP streaming transport using the benchmark's minimal usage contract.
use super::stream_metrics::{Measurement, Stream};
use crate::command::DynResult;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use std::time::Instant;

pub(super) async fn request(port: u16, body: &serde_json::Value) -> DynResult<Measurement> {
    if port == 0 {
        return Err("benchmark HTTP port must be nonzero".into());
    }
    let started = Instant::now();
    let socket = tokio::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port)).await?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(socket)).await?;
    let request = hyper::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("host", format!("127.0.0.1:{port}"))
        .header("content-type", "application/json")
        .header("connection", "close")
        .header("authorization", "Bearer EMPTY")
        .body(Full::new(Bytes::from(serde_json::to_vec(body)?)))?;
    let response = async {
        let mut response = sender.send_request(request).await?;
        if !response.status().is_success() {
            return Err(format!("benchmark HTTP {}", response.status()).into());
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
        Ok(stream.finish(started.elapsed()))
    };
    tokio::pin!(response);
    tokio::select! {
        result = &mut response => result,
        result = connection => { result?; response.await }
    }
}

#[cfg(test)]
#[path = "http_measurement_tests.rs"]
pub(super) mod tests;
