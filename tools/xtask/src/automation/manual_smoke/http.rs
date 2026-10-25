//! Loopback-only bounded nonstream smoke exchange, driven by its retained owner.
use http_body_util::{BodyExt, Full};
use hyper::{Method, body::Bytes};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use std::{
    net::Ipv4Addr,
    time::{Duration, Instant},
};
async fn exchange(port: u16, path: &str, body: Option<Value>) -> Result<Value, String> {
    let stream = tokio::net::TcpStream::connect((Ipv4Addr::LOCALHOST, port))
        .await
        .map_err(|_| "HTTP connect failed")?;
    let (mut sender, connection) = hyper::client::conn::http1::Builder::new()
        .max_buf_size(16384)
        .max_headers(128)
        .handshake(TokioIo::new(stream))
        .await
        .map_err(|_| "HTTP handshake failed")?;
    let bytes = body
        .as_ref()
        .map(serde_json::to_vec)
        .transpose()
        .map_err(|_| "request JSON failed")?
        .unwrap_or_default();
    let request = hyper::Request::builder()
        .method(if body.is_some() {
            Method::POST
        } else {
            Method::GET
        })
        .uri(path)
        .header("host", format!("127.0.0.1:{port}"))
        .header("content-type", "application/json")
        .header("connection", "close")
        .body(Full::new(Bytes::from(bytes)))
        .map_err(|_| "HTTP request refused")?;
    let response = async {
        let mut response = sender
            .send_request(request)
            .await
            .map_err(|_| "HTTP response failed")?;
        if !response.status().is_success() {
            return Err(format!("HTTP status {}", response.status()));
        }
        let mut bytes = Vec::new();
        while let Some(frame) = response.body_mut().frame().await {
            let frame = frame.map_err(|_| "HTTP body failed")?;
            if let Some(data) = frame.data_ref() {
                if data.len() > (1024 * 1024_usize).saturating_sub(bytes.len()) {
                    return Err("HTTP body exceeds 1 MiB".to_owned());
                }
                bytes.extend_from_slice(data);
            }
        }
        serde_json::from_slice(&bytes).map_err(|_| "HTTP JSON refused".to_owned())
    };
    tokio::pin!(response);
    tokio::select! { result = &mut response => result, result = connection => { result.map_err(|_|"HTTP connection failed")?; response.await } }
}
async fn request(
    port: u16,
    path: &str,
    body: Option<Value>,
    deadline: Instant,
) -> Result<Value, String> {
    tokio::time::timeout_at(deadline.into(), exchange(port, path, body))
        .await
        .map_err(|_| "HTTP deadline")?
}
async fn ready(port: u16, path: &str, deadline: Instant, status: bool) -> Result<Value, String> {
    loop {
        if Instant::now() >= deadline {
            return Err(format!("{path} readiness deadline"));
        }
        let cap = deadline.min(Instant::now() + Duration::from_secs(5));
        if let Ok(value) = request(port, path, None, cap).await {
            let admitted = if status {
                value["llama_ready"].as_bool() == Some(true)
            } else {
                value["data"][0]["id"]
                    .as_str()
                    .is_some_and(|id| !id.is_empty())
            };
            if admitted {
                return Ok(value);
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
pub(super) async fn smoke(api: u16, console: u16, max_wait: Duration) -> Result<Value, String> {
    let status = ready(console, "/api/status", Instant::now() + max_wait, true).await?;
    let models = ready(
        api,
        "/v1/models",
        Instant::now() + Duration::from_secs(60),
        false,
    )
    .await?;
    let model = models["data"][0]["id"]
        .as_str()
        .ok_or("model publication refused")?;
    let payload = json!({"model":model,"messages":[{"role":"user","content":"Say hello in exactly three words."}],"max_tokens":4,"temperature":0});
    let chat = request(
        api,
        "/v1/chat/completions",
        Some(payload.clone()),
        Instant::now() + Duration::from_secs(90),
    )
    .await?;
    if !chat.is_object() || chat.get("error").is_some() {
        return Err("chat completion error/object refused".into());
    }
    Ok(json!({"STATUS_JSON":status,"MODELS_JSON":models,"CHAT_JSON":chat,"request":payload}))
}
