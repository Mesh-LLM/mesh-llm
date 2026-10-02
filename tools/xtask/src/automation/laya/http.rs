use crate::command::DynResult;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use std::time::Duration;

pub(super) fn request(url: &str, body: Option<Vec<u8>>, timeout: Duration) -> DynResult<Vec<u8>> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async { tokio::time::timeout(timeout, exchange(url, body)).await? })
}

async fn exchange(url: &str, body: Option<Vec<u8>>) -> DynResult<Vec<u8>> {
    let uri: hyper::Uri = url.parse()?;
    if uri.scheme_str() != Some("http") {
        return Err("Laya smoke requires an HTTP endpoint".into());
    }
    let stream = tokio::net::TcpStream::connect((
        uri.host().ok_or("missing HTTP host")?,
        uri.port_u16().unwrap_or(80),
    ))
    .await?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(stream)).await?;
    let request = hyper::Request::builder()
        .method(if body.is_some() { "POST" } else { "GET" })
        .uri(uri.path_and_query().ok_or("missing HTTP path")?.as_str())
        .header(
            "host",
            uri.authority().ok_or("missing HTTP authority")?.as_str(),
        )
        .header("content-type", "application/json")
        .header("connection", "close")
        .body(Full::new(Bytes::from(body.unwrap_or_default())))?;
    let response = async {
        let mut response = sender.send_request(request).await?;
        if !response.status().is_success() {
            return Err("Laya HTTP request failed".into());
        }
        let mut bytes = Vec::new();
        while let Some(frame) = response.body_mut().frame().await {
            if let Some(data) = frame?.data_ref() {
                if data.len() > (16 * 1024 * 1024_usize).saturating_sub(bytes.len()) {
                    return Err("Laya response exceeds 16 MiB".into());
                }
                bytes.extend_from_slice(data);
            }
        }
        Ok(bytes)
    };
    tokio::pin!(response);
    tokio::select! {
        result = &mut response => result,
        result = connection => { result?; response.await }
    }
}
