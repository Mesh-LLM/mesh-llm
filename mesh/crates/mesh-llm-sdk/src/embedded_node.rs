use std::net::IpAddr;
use std::path::PathBuf;
use std::time::Duration;

pub use mesh_llm_embedded_runtime::{
    EmbeddedMeshAdmissionConfig, EmbeddedMeshDiscoveryMode, EmbeddedMeshHttpConfig,
    EmbeddedMeshLogFormat, EmbeddedMeshNetworkConfig, EmbeddedMeshNodeConfig, EmbeddedMeshNodeMode,
    EmbeddedMeshRequirementsConfig, EmbeddedMeshServingConfig, EmbeddedMeshStorageConfig,
    EmbeddedTrustPolicy, SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION,
};

pub type MeshNodeStatus = mesh_llm_embedded_runtime::EmbeddedMeshNodeStatus;

pub struct MeshNode {
    handle: mesh_llm_embedded_runtime::EmbeddedMeshNodeHandle,
    mode: EmbeddedMeshNodeMode,
}

impl MeshNode {
    pub fn builder() -> MeshNodeBuilder {
        MeshNodeBuilder::default()
    }

    pub fn api_base_url(&self) -> &str {
        self.handle.api_base_url()
    }

    pub fn console_url(&self) -> &str {
        self.handle.console_url()
    }

    pub fn invite_token(&self) -> Option<&str> {
        self.handle.invite_token()
    }

    pub fn mode(&self) -> &EmbeddedMeshNodeMode {
        &self.mode
    }

    pub fn openai_client(&self) -> anyhow::Result<OpenAiClient> {
        anyhow::ensure!(
            self.mode.allows_client_inference(),
            "client inference is disabled for a serve-only node"
        );
        Ok(OpenAiClient::new(self.api_base_url()))
    }

    pub async fn status(&self) -> anyhow::Result<MeshNodeStatus> {
        self.handle.status().await
    }

    pub async fn join_token(&self, token: impl Into<String>) -> anyhow::Result<()> {
        self.handle.join_token(token).await
    }

    pub async fn shutdown(self) -> anyhow::Result<()> {
        self.handle.stop().await
    }

    pub async fn stop(self) -> anyhow::Result<()> {
        self.shutdown().await
    }
}

#[derive(Clone, Debug, Default)]
pub struct MeshNodeBuilder {
    inner: mesh_llm_embedded_runtime::EmbeddedMeshNodeBuilder,
}

impl MeshNodeBuilder {
    pub fn mode(mut self, mode: EmbeddedMeshNodeMode) -> Self {
        self.inner = self.inner.mode(mode);
        self
    }

    pub fn serve(mut self) -> Self {
        self.inner = self.inner.serve();
        self
    }

    pub fn serve_only(mut self) -> Self {
        self.inner = self.inner.serve_only();
        self
    }

    pub fn client(mut self) -> Self {
        self.inner = self.inner.client();
        self
    }

    pub fn model(mut self, model_ref: impl Into<String>) -> Self {
        self.inner = self.inner.model(model_ref);
        self
    }

    pub fn models<I, S>(mut self, model_refs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner = self.inner.models(model_refs);
        self
    }

    pub fn max_vram_gb(mut self, max_vram_gb: f64) -> Self {
        self.inner = self.inner.max_vram_gb(max_vram_gb);
        self
    }

    pub fn api_port(mut self, port: u16) -> Self {
        self.inner = self.inner.api_port(port);
        self
    }

    pub fn console_port(mut self, port: u16) -> Self {
        self.inner = self.inner.console_port(port);
        self
    }

    pub fn console_ui(mut self, enabled: bool) -> Self {
        self.inner = self.inner.console_ui(enabled);
        self
    }

    pub fn join_token(mut self, token: impl Into<String>) -> Self {
        self.inner = self.inner.join_token(token);
        self
    }

    pub fn join_tokens<I, S>(mut self, tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner = self.inner.join_tokens(tokens);
        self
    }

    pub fn auto_join(mut self, enabled: bool) -> Self {
        self.inner = self.inner.auto_join(enabled);
        self
    }

    pub fn auto_join_public_mesh(mut self) -> Self {
        self.inner = self
            .inner
            .auto_join(true)
            .discovery_mode(EmbeddedMeshDiscoveryMode::Nostr);
        self
    }

    pub fn discovery_mode(mut self, mode: EmbeddedMeshDiscoveryMode) -> Self {
        self.inner = self.inner.discovery_mode(mode);
        self
    }

    pub fn publish(mut self, enabled: bool) -> Self {
        self.inner = self.inner.publish(enabled);
        self
    }

    pub fn mesh_name(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.mesh_name(name);
        self
    }

    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.inner = self.inner.region(region);
        self
    }

    pub fn node_name(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.node_name(name);
        self
    }

    pub fn iroh_relay(mut self, url: impl Into<String>) -> Self {
        self.inner = self.inner.iroh_relay(url);
        self
    }

    pub fn iroh_relays<I, S>(mut self, urls: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner = self.inner.iroh_relays(urls);
        self
    }

    pub fn iroh_relay_auth(
        mut self,
        relay_url: impl Into<String>,
        bearer_token: impl Into<String>,
    ) -> Self {
        self.inner = self.inner.iroh_relay_auth(relay_url, bearer_token);
        self
    }

    pub fn nostr_relay(mut self, url: impl Into<String>) -> Self {
        self.inner = self.inner.nostr_relay(url);
        self
    }

    pub fn nostr_relays<I, S>(mut self, urls: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner = self.inner.nostr_relays(urls);
        self
    }

    pub fn bind_ip(mut self, ip: IpAddr) -> Self {
        self.inner = self.inner.bind_ip(ip);
        self
    }

    pub fn bind_port(mut self, port: u16) -> Self {
        self.inner = self.inner.bind_port(port);
        self
    }

    pub fn listen_all(mut self, enabled: bool) -> Self {
        self.inner = self.inner.listen_all(enabled);
        self
    }

    pub fn enumerate_host(mut self, enabled: bool) -> Self {
        self.inner = self.inner.enumerate_host(enabled);
        self
    }

    pub fn owner_key(mut self, path: impl Into<PathBuf>) -> Self {
        self.inner = self.inner.owner_key(path);
        self
    }

    pub fn owner_required(mut self, required: bool) -> Self {
        self.inner = self.inner.owner_required(required);
        self
    }

    pub fn node_label(mut self, label: impl Into<String>) -> Self {
        self.inner = self.inner.node_label(label);
        self
    }

    pub fn trust_policy(mut self, policy: EmbeddedTrustPolicy) -> Self {
        self.inner = self.inner.trust_policy(policy);
        self
    }

    pub fn trust_owner(mut self, owner_id: impl Into<String>) -> Self {
        self.inner = self.inner.trust_owner(owner_id);
        self
    }

    pub fn trust_owners<I, S>(mut self, owner_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner = self.inner.trust_owners(owner_ids);
        self
    }

    pub fn min_node_version(mut self, version: impl Into<String>) -> Self {
        self.inner = self.inner.min_node_version(version);
        self
    }

    pub fn max_node_version(mut self, version: impl Into<String>) -> Self {
        self.inner = self.inner.max_node_version(version);
        self
    }

    pub fn min_protocol_version(mut self, version: u32) -> Self {
        self.inner = self.inner.min_protocol_version(version);
        self
    }

    pub fn signed_join_tokens(mut self, enabled: bool) -> Self {
        self.inner = self.inner.signed_join_tokens(enabled);
        self
    }

    pub fn max_protocol_version(mut self, version: u32) -> Self {
        self.inner = self.inner.max_protocol_version(version);
        self
    }

    pub fn require_release_attestation(mut self, required: bool) -> Self {
        self.inner = self.inner.require_release_attestation(required);
        self
    }

    pub fn release_signer_key(mut self, key: impl Into<String>) -> Self {
        self.inner = self.inner.release_signer_key(key);
        self
    }

    pub fn release_signer_keys<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner = self.inner.release_signer_keys(keys);
        self
    }

    pub fn config_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.inner = self.inner.config_path(path);
        self
    }

    pub fn isolated_config(mut self, enabled: bool) -> Self {
        self.inner = self.inner.isolated_config(enabled);
        self
    }

    pub fn log_format(mut self, format: EmbeddedMeshLogFormat) -> Self {
        self.inner = self.inner.log_format(format);
        self
    }

    pub fn startup_timeout(mut self, timeout: Duration) -> Self {
        self.inner = self.inner.startup_timeout(timeout);
        self
    }

    pub fn build(self) -> EmbeddedMeshNodeConfig {
        self.inner.build()
    }

    pub async fn start(self) -> anyhow::Result<MeshNode> {
        let config = self.build();
        let mode = config.mode.clone();
        let handle = mesh_llm_embedded_runtime::start_embedded_node(config).await?;
        Ok(MeshNode { handle, mode })
    }
}

#[derive(Clone)]
pub struct OpenAiClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl OpenAiClient {
    fn new(base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key: "mesh".to_string(),
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn http_client(&self) -> &reqwest::Client {
        &self.http
    }

    pub async fn request(
        &self,
        path: &str,
        body_json: String,
    ) -> anyhow::Result<RawOpenAiResponse> {
        let response = self
            .http
            .post(self.url(path))
            .bearer_auth(&self.api_key)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body_json)
            .send()
            .await?;
        let status_code = response.status().as_u16();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string);
        let body = response.text().await?;
        Ok(RawOpenAiResponse {
            status_code,
            content_type,
            body,
        })
    }

    pub async fn stream(&self, path: &str, body_json: String) -> anyhow::Result<reqwest::Response> {
        Ok(tokio::time::timeout(
            Duration::from_secs(120),
            self.http
                .post(self.url(path))
                .bearer_auth(&self.api_key)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body_json)
                .send(),
        )
        .await??)
    }

    pub async fn models(&self) -> anyhow::Result<serde_json::Value> {
        self.get_json("models").await
    }

    pub async fn chat_completions(
        &self,
        body: impl serde::Serialize,
    ) -> anyhow::Result<serde_json::Value> {
        self.post_json("chat/completions", body).await
    }

    pub async fn responses(
        &self,
        body: impl serde::Serialize,
    ) -> anyhow::Result<serde_json::Value> {
        self.post_json("responses", body).await
    }

    async fn get_json(&self, path: &str) -> anyhow::Result<serde_json::Value> {
        let response = self
            .http
            .get(self.url(path))
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?;
        Ok(response.json().await?)
    }

    async fn post_json(
        &self,
        path: &str,
        body: impl serde::Serialize,
    ) -> anyhow::Result<serde_json::Value> {
        let response = self
            .http
            .post(self.url(path))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        Ok(response.json().await?)
    }

    fn url(&self, path: &str) -> String {
        let path = path.trim_start_matches('/');
        let path = path.strip_prefix("v1/").unwrap_or(path);
        format!("{}/{}", self.base_url.trim_end_matches('/'), path)
    }
}

pub struct RawOpenAiResponse {
    pub status_code: u16,
    pub content_type: Option<String>,
    pub body: String,
}

pub struct SseFrame {
    pub event_type: Option<String>,
    pub data: String,
    pub raw: String,
}

#[derive(Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    pub fn push(&mut self, chunk: &[u8]) -> anyhow::Result<Vec<SseFrame>> {
        const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;
        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();
        while let Some(end) = sse_frame_end(&self.buffer) {
            anyhow::ensure!(end <= MAX_FRAME_BYTES, "SSE frame exceeds 8 MiB");
            let raw = self.buffer.drain(..end).collect::<Vec<_>>();
            if let Some(frame) = parse_sse_frame(raw)? {
                frames.push(frame);
            }
        }
        anyhow::ensure!(
            self.buffer.len() <= MAX_FRAME_BYTES,
            "SSE frame exceeds 8 MiB"
        );
        Ok(frames)
    }

    pub fn finish(&mut self) -> anyhow::Result<Option<SseFrame>> {
        if self.buffer.is_empty() {
            return Ok(None);
        }
        parse_sse_frame(std::mem::take(&mut self.buffer))
    }
}

fn sse_frame_end(buffer: &[u8]) -> Option<usize> {
    let lf = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| index + 2);
    let crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4);
    match (lf, crlf) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(end), None) | (None, Some(end)) => Some(end),
        (None, None) => None,
    }
}

fn parse_sse_frame(raw: Vec<u8>) -> anyhow::Result<Option<SseFrame>> {
    let raw = String::from_utf8(raw)?;
    let mut data = Vec::new();
    let mut event_type = None;
    for line in raw.lines().map(|line| line.trim_end_matches('\r')) {
        if let Some(value) = line.strip_prefix("data:") {
            data.push(value.strip_prefix(' ').unwrap_or(value).to_string());
        } else if let Some(value) = line.strip_prefix("event:") {
            event_type = Some(value.trim().to_string());
        }
    }
    if data.is_empty() {
        return Ok(None);
    }
    Ok(Some(SseFrame {
        event_type,
        data: data.join("\n"),
        raw,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_shapes_public_auto_join_serve_node() {
        let config = MeshNode::builder()
            .serve()
            .model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
            .auto_join_public_mesh()
            .api_port(19447)
            .console_port(13141)
            .build();

        assert_eq!(config.mode, EmbeddedMeshNodeMode::Serve);
        assert_eq!(
            config.serving.models,
            vec!["unsloth/Qwen3-0.6B-GGUF:Q4_K_M"]
        );
        assert!(config.network.auto_join);
        assert_eq!(
            config.network.discovery_mode,
            EmbeddedMeshDiscoveryMode::Nostr
        );
        assert_eq!(config.http.api_port, 19447);
        assert_eq!(config.http.console_port, 13141);
    }

    #[test]
    fn builder_configures_each_node_role() {
        let client = MeshNode::builder().client().build();
        let serve_only = MeshNode::builder().serve_only().build();
        let combined = MeshNode::builder().serve().build();

        assert_eq!(client.mode, EmbeddedMeshNodeMode::Client);
        assert_eq!(serve_only.mode, EmbeddedMeshNodeMode::ServeOnly);
        assert_eq!(combined.mode, EmbeddedMeshNodeMode::Serve);
        assert!(client.mode.allows_client_inference());
        assert!(!client.mode.allows_serving());
        assert!(!serve_only.mode.allows_client_inference());
        assert!(serve_only.mode.allows_serving());
        assert!(combined.mode.allows_client_inference());
        assert!(combined.mode.allows_serving());
    }

    #[test]
    fn openai_client_builds_v1_urls() {
        let client = OpenAiClient::new("http://127.0.0.1:9337/v1/");
        assert_eq!(client.url("models"), "http://127.0.0.1:9337/v1/models");
        assert_eq!(
            client.url("chat/completions"),
            "http://127.0.0.1:9337/v1/chat/completions"
        );
        for path in [
            "/v1/chat/completions",
            "v1/chat/completions",
            "/chat/completions",
        ] {
            assert_eq!(
                client.url(path),
                "http://127.0.0.1:9337/v1/chat/completions"
            );
        }
    }

    #[test]
    fn sse_decoder_preserves_split_utf8_and_crlf_frames() {
        let mut decoder = SseDecoder::default();
        let payload = "event: delta\r\ndata: café\r\n\r\n".as_bytes();
        let split = payload.iter().position(|byte| *byte == 0xc3).unwrap() + 1;
        assert!(decoder.push(&payload[..split]).unwrap().is_empty());
        let frames = decoder.push(&payload[split..]).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].event_type.as_deref(), Some("delta"));
        assert_eq!(frames[0].data, "café");
        assert_eq!(frames[0].raw, "event: delta\r\ndata: café\r\n\r\n");
    }

    #[test]
    fn sse_decoder_rejects_oversized_frame() {
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(&vec![b'x'; 8 * 1024 * 1024 + 1]).is_err());
    }
}
