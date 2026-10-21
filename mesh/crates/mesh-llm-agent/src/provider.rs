//! A small OpenAI chat-completions client for the mesh endpoint.
//!
//! Request and response mapping is goose's (`goose_provider_types::formats::openai`);
//! this only does the HTTP round trip, non-streaming, so mesh doesn't link
//! `goose-providers` and its full set of provider modules.

use async_trait::async_trait;
use goose_provider_types::base::{MessageStream, Provider, stream_from_single_message};
use goose_provider_types::conversation::message::Message;
use goose_provider_types::conversation::token_usage::ProviderUsage;
use goose_provider_types::errors::ProviderError;
use goose_provider_types::formats::openai::{create_request, get_usage, response_to_message};
use goose_provider_types::images::ImageFormat;
use goose_provider_types::model::ModelConfig;
use rmcp::model::Tool;
use serde_json::Value;

pub struct MeshProvider {
    url: String,
    http: reqwest::Client,
}

impl MeshProvider {
    /// `base_url` is the mesh API origin, e.g. `http://127.0.0.1:9337`.
    pub fn new(base_url: &str) -> Self {
        Self {
            url: format!("{}/v1/chat/completions", base_url.trim_end_matches('/')),
            http: reqwest::Client::new(),
        }
    }
}

/// Mesh can return HTTP 200 with `{"error": {...}}` (e.g. a peer's
/// `llama_decode failed`), so check the body as well as the status.
fn error_message(body: &Value) -> Option<String> {
    let error = body.get("error")?;
    Some(
        error
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| error.to_string()),
    )
}

#[async_trait]
impl Provider for MeshProvider {
    fn get_name(&self) -> &str {
        "mesh"
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let payload = create_request(
            model_config,
            system,
            messages,
            tools,
            &ImageFormat::OpenAi,
            false,
        )
        .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;
        let response = self
            .http
            .post(&self.url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| ProviderError::NetworkError(e.to_string()))?;
        let body: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        if let Some(message) = error_message(&body) {
            return Err(ProviderError::ServerError(format!("{status}: {message}")));
        }
        if !status.is_success() {
            return Err(ProviderError::ServerError(format!("{status}: {text}")));
        }
        let message =
            response_to_message(&body).map_err(|e| ProviderError::RequestFailed(e.to_string()))?;
        let usage = body.get("usage").map(get_usage).unwrap_or_default();
        Ok(stream_from_single_message(
            message,
            ProviderUsage::new(model_config.model_name.clone(), usage),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_in_200_body_is_reported() {
        let body = serde_json::json!({"error": {"code": "service_unavailable", "message": "RuntimeError: llama_decode failed"}});
        assert_eq!(
            error_message(&body).as_deref(),
            Some("RuntimeError: llama_decode failed")
        );
        assert!(error_message(&serde_json::json!({"choices": []})).is_none());
    }
}
