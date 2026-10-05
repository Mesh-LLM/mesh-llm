//! Bounded loopback metrics-server lifecycle for one benchmark cell.
use super::{metrics_correlation, publish};
use crate::{command::DynResult, process::Cancellation};
use http_body_util::{BodyExt, Full};
use hyper::{Method, Uri, body::Bytes};
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Duration};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Endpoint {
    pub http: String,
    pub otlp_grpc: String,
    pub run_id: String,
    pub timeout_secs: u64,
}

impl Endpoint {
    pub fn validate(&self) -> DynResult<()> {
        for endpoint in [&self.http, &self.otlp_grpc] {
            let uri: Uri = endpoint.parse()?;
            let port = uri.port_u16().ok_or("collector endpoint requires a port")?;
            if uri.scheme_str() != Some("http")
                || uri.host() != Some("127.0.0.1")
                || port == 0
                || uri.authority().map(|authority| authority.as_str())
                    != Some(format!("127.0.0.1:{port}").as_str())
                || uri.path() != "/"
                || uri.query().is_some()
            {
                return Err("collector endpoints must be explicit loopback HTTP roots".into());
            }
        }
        if !(1..=3600).contains(&self.timeout_secs)
            || self.run_id.is_empty()
            || self.run_id.len() > 128
            || !self
                .run_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err("invalid collector run identity or deadline".into());
        }
        Ok(())
    }

    fn url(&self, suffix: &str) -> String {
        format!(
            "{}/v1/runs/{}{suffix}",
            self.http.trim_end_matches('/'),
            self.run_id
        )
    }
}

async fn cancelled(cancellation: &Cancellation) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn exchange(url: &str, method: Method, body: Vec<u8>, maximum: usize) -> DynResult<Vec<u8>> {
    let uri: Uri = url.parse()?;
    let socket = tokio::net::TcpStream::connect((
        uri.host().ok_or("missing collector host")?,
        uri.port_u16().ok_or("missing collector port")?,
    ))
    .await?;
    let (mut sender, connection) =
        hyper::client::conn::http1::handshake(TokioIo::new(socket)).await?;
    let request = hyper::Request::builder()
        .method(method)
        .uri(
            uri.path_and_query()
                .ok_or("missing collector path")?
                .as_str(),
        )
        .header(
            "host",
            uri.authority()
                .ok_or("missing collector authority")?
                .as_str(),
        )
        .header("content-type", "application/json")
        .header("connection", "close")
        .body(Full::new(Bytes::from(body)))?;
    let response = async {
        let mut response = sender.send_request(request).await?;
        if !response.status().is_success() {
            return Err(format!("collector HTTP {}", response.status()).into());
        }
        let mut bytes = Vec::new();
        while let Some(frame) = response.body_mut().frame().await {
            if let Some(data) = frame?.data_ref() {
                if data.len() > maximum.saturating_sub(bytes.len()) {
                    return Err("collector response exceeded its byte budget".into());
                }
                bytes.extend_from_slice(data);
            }
        }
        Ok(bytes)
    };
    tokio::pin!(response);
    tokio::select! {result = &mut response => result, result = connection => {result?; response.await}}
}

async fn bounded(
    endpoint: &Endpoint,
    cancellation: &Cancellation,
    url: &str,
    method: Method,
    body: Vec<u8>,
    maximum: usize,
) -> DynResult<Vec<u8>> {
    endpoint.validate()?;
    tokio::select! {
        biased;
        () = cancelled(cancellation) => Err("collector operation interrupted".into()),
        result = tokio::time::timeout(Duration::from_secs(endpoint.timeout_secs),
            exchange(url, method, body, maximum)) => result.map_err(|_| "collector operation deadline expired")?,
    }
}

pub(super) async fn create(
    endpoint: &Endpoint,
    config: &serde_json::Value,
    cancellation: &Cancellation,
) -> DynResult<()> {
    let mut body = config
        .as_object()
        .ok_or("collector config must be an object")?
        .clone();
    body.insert("run_id".into(), serde_json::json!(endpoint.run_id));
    let body = serde_json::to_vec(&body)?;
    if body.len() > 1024 * 1024 {
        return Err("collector create input exceeds 1 MiB".into());
    }
    let bytes = bounded(
        endpoint,
        cancellation,
        &format!("{}/v1/runs", endpoint.http.trim_end_matches('/')),
        Method::POST,
        body,
        1024 * 1024,
    )
    .await?;
    let response: serde_json::Value = serde_json::from_slice(&bytes)?;
    if response["run_id"] != endpoint.run_id || response["status"] != "running" {
        return Err("collector did not create the admitted run".into());
    }
    Ok(())
}

pub(super) async fn collect(
    endpoint: &Endpoint,
    measured: &[String],
    directory: &Path,
    cancellation: &Cancellation,
) -> DynResult<Vec<metrics_correlation::Timing>> {
    endpoint.validate()?;
    let operation = async {
        loop {
            let bytes = exchange(
                &endpoint.url("/report.json"),
                Method::GET,
                Vec::new(),
                64 * 1024 * 1024,
            )
            .await?;
            publish(&directory.join("metrics-report.json"), &bytes)?;
            if metrics_correlation::ready(&bytes, &endpoint.run_id, measured).is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let bytes = exchange(
            &endpoint.url("/finalize"),
            Method::POST,
            Vec::new(),
            1024 * 1024,
        )
        .await?;
        let response: serde_json::Value = serde_json::from_slice(&bytes)?;
        if response["run_id"] != endpoint.run_id || response["status"] != "completed" {
            return Err("collector did not finalize the admitted run".into());
        }
        let bytes = exchange(
            &endpoint.url("/report.json"),
            Method::GET,
            Vec::new(),
            64 * 1024 * 1024,
        )
        .await?;
        publish(&directory.join("metrics-report.json"), &bytes)?;
        let timings = metrics_correlation::correlate(&bytes, &endpoint.run_id, measured)?;
        publish(
            &directory.join("metrics-timing.json"),
            &serde_json::to_vec_pretty(&timings)?,
        )?;
        Ok(timings)
    };
    tokio::select! {
        biased;
        () = cancelled(cancellation) => Err("collector collection interrupted".into()),
        result = tokio::time::timeout(Duration::from_secs(endpoint.timeout_secs), operation) =>
            result.map_err(|_| "collector delivery/finalization deadline expired")?,
    }
}

#[cfg(test)]
#[path = "metrics_client_tests.rs"]
mod tests;
