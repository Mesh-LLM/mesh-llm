use super::options::Options;
use crate::process::Cancellation;
use http_body_util::{BodyExt, Empty};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use serde::Deserialize;
use std::time::{Duration, Instant};

pub(super) struct Ready {
    pub token: String,
    pub model: String,
}
#[derive(Deserialize)]
struct Status {
    token: String,
}
#[derive(Deserialize)]
struct Models {
    data: Vec<Model>,
}
#[derive(Deserialize)]
struct Model {
    id: String,
}

pub(super) fn wait(options: &Options, cancel: &Cancellation) -> Result<Ready, String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|_| "SDK HTTP runtime failed")?;
    let until = Instant::now() + options.wait;
    while Instant::now() < until && !cancel.is_cancelled() {
        let deadline = until.min(Instant::now() + Duration::from_secs(2));
        let status = runtime.block_on(fetch(options.console, "/api/status", deadline));
        let models = runtime.block_on(fetch(options.api, "/v1/models", deadline));
        if let (Ok(status), Ok(models)) = (status, models)
            && let (Ok(status), Ok(models)) = (
                serde_json::from_slice::<Status>(&status),
                serde_json::from_slice::<Models>(&models),
            )
            && let Some(model) = models
                .data
                .into_iter()
                .next()
                .filter(|model| !model.id.is_empty())
            && !status.token.is_empty()
        {
            return Ok(Ready {
                token: status.token,
                model: model.id,
            });
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err("SDK fixture readiness deadline or cancellation".into())
}

async fn fetch(port: u16, path: &str, deadline: Instant) -> Result<Vec<u8>, String> {
    tokio::time::timeout_at(deadline.into(), exchange(port, path))
        .await
        .map_err(|_| "SDK HTTP deadline".to_owned())?
}

async fn exchange(port: u16, path: &str) -> Result<Vec<u8>, String> {
    let exchange = async {
        let stream = tokio::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port)).await?;
        let (mut sender, connection) = hyper::client::conn::http1::Builder::new()
            .max_buf_size(16384)
            .max_headers(4096)
            .handshake::<_, Empty<Bytes>>(TokioIo::new(stream))
            .await?;
        let request = hyper::Request::builder()
            .uri(path)
            .header("host", format!("127.0.0.1:{port}"))
            .header("connection", "close")
            .body(Empty::<Bytes>::new())?;
        let response = async {
            let mut response = sender.send_request(request).await?;
            if !response.status().is_success() {
                return Err("SDK HTTP status".into());
            }
            let mut bytes = Vec::new();
            while let Some(frame) = response.body_mut().frame().await {
                if let Some(data) = frame?.data_ref() {
                    if data.len() > 1_048_576 - bytes.len() {
                        return Err("SDK HTTP body limit".into());
                    }
                    bytes.extend_from_slice(data);
                }
            }
            Ok::<_, Box<dyn std::error::Error>>(bytes)
        };
        tokio::pin!(response);
        tokio::select! {
            biased;
            result = &mut response => result,
            result = connection => { result?; response.await }
        }
    };
    exchange
        .await
        .map_err(|_: Box<dyn std::error::Error>| "SDK HTTP transfer failed".into())
}
