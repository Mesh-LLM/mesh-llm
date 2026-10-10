use super::sse::Decoder;
use crate::process::Cancellation;
use hyper::Method;
use serde_json::Value;
use std::time::{Duration, Instant};
use url::Url;
#[path = "http_transport.rs"]
mod http;
#[path = "https_transport.rs"]
mod https;
const BODY_LIMIT: usize = 16 * 1024 * 1024;
pub(super) struct Http {
    tls: Option<https::Curl>,
    base: Url,
    timeout: Duration,
    cancellation: Cancellation,
    token: &'static str,
}
pub(super) struct Reply {
    pub status: u16,
    pub json: Option<Value>,
    pub events: Vec<Value>,
    pub first_event_ms: Option<u64>,
}
#[derive(Clone, Debug)]
pub(super) struct Failure {
    pub detail: String,
    pub status: Option<u16>,
}
impl Http {
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
    pub fn new(
        base: Url,
        timeout: Duration,
        cancellation: Cancellation,
        token: &'static str,
    ) -> Result<Self, String> {
        let tls = if base.scheme() == "https" {
            Some(https::Curl::discover()?)
        } else {
            None
        };
        Ok(Self {
            tls,
            base,
            timeout,
            cancellation,
            token,
        })
    }
    pub async fn models(&self) -> Result<Reply, Failure> {
        self.request(Method::GET, "/models", None, false).await
    }
    pub async fn chat(&self, payload: &Value, stream: bool) -> Result<Reply, Failure> {
        self.request(Method::POST, "/chat/completions", Some(payload), stream)
            .await
    }
    async fn request(
        &self,
        method: Method,
        suffix: &str,
        payload: Option<&Value>,
        stream: bool,
    ) -> Result<Reply, Failure> {
        let started = Instant::now();
        if self.cancellation.is_cancelled() {
            return Err(failure("stability operation cancelled", None));
        }
        if let Some(tls) = &self.tls {
            let body = payload
                .map(serde_json::to_vec)
                .transpose()
                .map_err(|_| failure("stability request encoding failed", None))?;
            if body
                .as_ref()
                .is_some_and(|bytes| bytes.len() > 2 * 1024 * 1024)
            {
                return Err(failure("stability request exceeds 2 MiB", None));
            }
            return tls
                .exchange(
                    https::Request {
                        endpoint: format!("{}{suffix}", self.base.as_str().trim_end_matches('/')),
                        method,
                        body,
                        stream,
                        timeout: self.timeout,
                        token: self.token,
                        started,
                    },
                    self.cancellation.clone(),
                )
                .await;
        }
        let operation = self.exchange_http(method, suffix, payload, stream, started);
        let cancelled = async {
            loop {
                if self.cancellation.is_cancelled() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        };
        tokio::select! {
            biased;
            () = cancelled => Err(failure("stability operation cancelled", None)),
            result = tokio::time::timeout(self.timeout, operation) =>
                result.map_err(|_| failure("stability request deadline exceeded", None))?,
        }
    }
}
fn failure(detail: &str, status: Option<u16>) -> Failure {
    Failure {
        detail: detail.into(),
        status,
    }
}
pub(super) fn millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}
