//! Credentialed HF Jobs transport. No alternate origin or redirect admission.
use anyhow::{Result, anyhow, bail};
use futures::StreamExt as _;
use serde::de::DeserializeOwned;
use std::time::{Duration, Instant};

/// Caps are shared across response phases, never renewed for each chunk.
#[derive(Clone, Copy, Debug)]
pub struct TransportLimits {
    pub request_timeout: Duration,
    pub log_timeout: Duration,
    pub json_bytes: usize,
    pub log_bytes: usize,
    pub log_line_bytes: usize,
}
impl Default for TransportLimits {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30),
            log_timeout: Duration::from_secs(900),
            json_bytes: 4 * 1024 * 1024,
            log_bytes: 32 * 1024 * 1024,
            log_line_bytes: 256 * 1024,
        }
    }
}
impl TransportLimits {
    pub(super) fn validate(self) -> Result<Self> {
        if self.request_timeout.is_zero()
            || self.request_timeout > Duration::from_secs(300)
            || self.log_timeout.is_zero()
            || self.log_timeout > Duration::from_secs(86400)
            || self.json_bytes == 0
            || self.json_bytes > 16 * 1024 * 1024
            || self.log_bytes == 0
            || self.log_bytes > 256 * 1024 * 1024
            || self.log_line_bytes == 0
            || self.log_line_bytes > self.log_bytes
        {
            bail!("HF Jobs transport limits refused");
        }
        Ok(self)
    }
}
pub(crate) fn origin(value: &str) -> Result<String> {
    if value.len() > 2048 || value.bytes().any(|b| b.is_ascii_control()) {
        bail!("HF Jobs origin refused");
    }
    let url = reqwest::Url::parse(value).map_err(|_| anyhow!("HF Jobs origin refused"))?;
    if url.scheme() != "https"
        || url.host_str() != Some("huggingface.co")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        bail!("HF Jobs requires the trusted https://huggingface.co origin");
    }
    Ok("https://huggingface.co".into())
}
pub(crate) fn client() -> Result<reqwest::Client> {
    let _ = skippy_model_hf::configure_hf_tls_provider();
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .map_err(|_| anyhow!("HF Jobs HTTP client initialization failed"))
}
pub(crate) fn token(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 4096 || value.bytes().any(|b| !b.is_ascii_graphic()) {
        bail!("HF Jobs credential grammar refused");
    }
    Ok(())
}
pub(super) fn url(endpoint: &str, parts: &[&str]) -> Result<reqwest::Url> {
    if parts.iter().any(|p| {
        p.is_empty()
            || p.len() > 128
            || *p == "."
            || *p == ".."
            || p.bytes()
                .any(|b| !(b.is_ascii_alphanumeric() || b"._-".contains(&b)))
    }) {
        bail!("HF Jobs path component refused");
    }
    let mut url = reqwest::Url::parse(endpoint).map_err(|_| anyhow!("HF Jobs origin refused"))?;
    url.path_segments_mut()
        .map_err(|_| anyhow!("HF Jobs origin refused"))?
        .extend(["api", "jobs"])
        .extend(parts.iter().copied());
    Ok(url)
}
pub(super) fn deadline(absolute: Instant, allowance: Duration) -> Result<Instant> {
    let now = Instant::now();
    if now >= absolute {
        bail!("HF Jobs deadline expired");
    }
    Ok(absolute.min(
        now.checked_add(allowance)
            .ok_or_else(|| anyhow!("HF Jobs deadline overflow"))?,
    ))
}
pub(super) async fn send(
    request: reqwest::RequestBuilder,
    until: Instant,
) -> Result<reqwest::Response> {
    deadline(until, Duration::from_secs(1))?;
    let response = tokio::time::timeout_at(until.into(), request.send())
        .await
        .map_err(|_| anyhow!("HF Jobs request deadline expired"))?
        .map_err(|_| anyhow!("HF Jobs HTTP request failed"))?;
    if !response.status().is_success() {
        // Neither server bodies nor reqwest URL/proxy diagnostics enter errors.
        bail!("HF Jobs API returned HTTP {}", response.status().as_u16());
    }
    deadline(until, Duration::from_secs(1))?;
    Ok(response)
}
pub(super) async fn json<T: DeserializeOwned>(
    response: reqwest::Response,
    until: Instant,
    cap: usize,
) -> Result<T> {
    if response.content_length().is_some_and(|n| n > cap as u64) {
        bail!("HF Jobs JSON body byte bound exceeded");
    }
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    loop {
        let chunk = tokio::time::timeout_at(until.into(), stream.next())
            .await
            .map_err(|_| anyhow!("HF Jobs response deadline expired"))?;
        let Some(chunk) = chunk else { break };
        let chunk = chunk.map_err(|_| anyhow!("HF Jobs response read failed"))?;
        if chunk.len() > cap.saturating_sub(bytes.len()) {
            bail!("HF Jobs JSON body byte bound exceeded");
        }
        bytes.extend_from_slice(&chunk);
    }
    deadline(until, Duration::from_secs(1))?;
    let value =
        serde_json::from_slice(&bytes).map_err(|_| anyhow!("HF Jobs JSON response invalid"))?;
    deadline(until, Duration::from_secs(1))?;
    Ok(value)
}
struct BoundedJson {
    bytes: Vec<u8>,
    cap: usize,
}
impl std::io::Write for BoundedJson {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.cap.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other("request byte bound"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
pub(super) fn encode(spec: &super::JobSpec, cap: usize) -> Result<Vec<u8>> {
    let mut output = BoundedJson {
        bytes: Vec::new(),
        cap,
    };
    serde_json::to_writer(&mut output, spec)
        .map_err(|_| anyhow!("HF Jobs request JSON serialization or byte bound refused"))?;
    Ok(output.bytes)
}
