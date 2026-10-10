use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use std::time::Duration;

pub(in crate::automation) struct Request {
    pub port: u16,
    pub path: String,
    pub body: Vec<u8>,
    pub headers: Vec<(String, String)>,
    pub timeout: Duration,
    pub partial: bool,
    pub method: Option<String>,
}

pub(in crate::automation) struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

pub(in crate::automation) fn transfer(request: Request) -> Result<Response, String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|_| "logging HTTP runtime")?;
    runtime.block_on(exchange(request))
}

async fn exchange(request: Request) -> Result<Response, String> {
    let operation = async {
        let stream =
            tokio::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, request.port)).await?;
        let (mut sender, connection) = hyper::client::conn::http1::Builder::new()
            .max_buf_size(16384)
            .max_headers(4096)
            .handshake::<_, Full<Bytes>>(TokioIo::new(stream))
            .await?;
        let mut message = hyper::Request::builder()
            .uri(&request.path)
            .method(
                request
                    .method
                    .as_deref()
                    .unwrap_or(if request.body.is_empty() {
                        "GET"
                    } else {
                        "POST"
                    }),
            )
            .header("host", format!("127.0.0.1:{}", request.port))
            .header("content-type", "application/json");
        for (key, value) in &request.headers {
            message = message.header(key, value);
        }
        let message = message.body(Full::new(Bytes::from(request.body)))?;
        let response = async {
            let mut response = sender.send_request(message).await?;
            let status = response.status().as_u16();
            let mut body = Vec::new();
            let mut limit = tokio::time::interval(request.timeout);
            limit.tick().await;
            loop {
                let frame = tokio::select! {
                    frame = response.body_mut().frame() => frame,
                    _ = limit.tick(), if request.partial => return Ok(Response { status, body }),
                };
                let Some(frame) = frame else { break };
                if let Some(data) = frame?.data_ref() {
                    if data.len() > 1_048_576 - body.len() {
                        return Err("logging HTTP body limit".into());
                    }
                    body.extend_from_slice(data);
                    if request.partial
                        && body
                            .windows(b"event: replay_gap".len())
                            .any(|bytes| bytes == b"event: replay_gap")
                        && body
                            .windows(b"/api/logs/requests".len())
                            .any(|bytes| bytes == b"/api/logs/requests")
                    {
                        return Ok(Response { status, body });
                    }
                }
            }
            Ok::<_, Box<dyn std::error::Error>>(Response { status, body })
        };
        tokio::pin!(response);
        tokio::select! {
            biased;
            result = &mut response => result,
            result = connection => { result?; response.await }
        }
    };
    tokio::time::timeout(request.timeout + Duration::from_secs(1), operation)
        .await
        .map_err(|_| "logging HTTP deadline".to_owned())?
        .map_err(|_: Box<dyn std::error::Error>| "logging HTTP transfer failed".to_owned())
}

/// Owned HTTP futures stop at cancellation or an absolute caller deadline.
pub(in crate::automation) fn transfer_cancelled(
    request: Request,
    cancellation: &crate::process::Cancellation,
    deadline: std::time::Instant,
) -> Result<Response, String> {
    if cancellation.is_cancelled() {
        return Err("logging HTTP cancelled".into());
    }
    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
    if remaining.is_zero() {
        return Err("logging HTTP deadline".into());
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|_| "logging HTTP runtime")?;
    runtime.block_on(async {
        let cancelled=async {loop {if cancellation.is_cancelled(){break;}tokio::time::sleep(Duration::from_millis(5)).await;}};
        tokio::select! {biased; ()=cancelled=>Err("logging HTTP cancelled".into()), result=tokio::time::timeout(remaining,exchange(request))=>result.map_err(|_|"logging HTTP deadline".to_owned())?}
    })
}
