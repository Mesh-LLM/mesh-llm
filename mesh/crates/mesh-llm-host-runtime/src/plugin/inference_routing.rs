//! Model discovery and protocol-aware selection of plugin inference endpoints.

use super::{InferenceEndpointRoute, PluginManager};
use anyhow::Result;

impl PluginManager {
    pub async fn inference_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();
        for endpoint in self.inference_endpoints().await? {
            models.extend(endpoint.models);
        }
        models.sort();
        models.dedup();
        Ok(models)
    }

    /// Plugin models exposed by the chat-compatible `/v1/models` listing.
    pub async fn chat_inference_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();
        for endpoint in self.inference_endpoints().await? {
            if endpoint.protocol.as_deref() != Some("document_ocr_v1") {
                models.extend(endpoint.models);
            }
        }
        models.sort();
        models.dedup();
        Ok(models)
    }

    pub async fn inference_endpoint_for_model(
        &self,
        model: &str,
    ) -> Result<Option<InferenceEndpointRoute>> {
        let mut endpoints = self.inference_endpoints().await?;
        sort_endpoints(&mut endpoints);
        Ok(endpoints.into_iter().find(|endpoint| {
            endpoint.protocol.as_deref() != Some("document_ocr_v1")
                && endpoint.models.iter().any(|candidate| candidate == model)
        }))
    }

    pub async fn inference_endpoint_for_request(
        &self,
        model: &str,
        path: &str,
    ) -> Result<Option<InferenceEndpointRoute>> {
        let is_document_ocr = path.split('?').next() == Some("/v1/documents/ocr");
        let mut endpoints = self.inference_endpoints().await?;
        sort_endpoints(&mut endpoints);
        Ok(endpoints.into_iter().find(|endpoint| {
            let serves_model = endpoint.models.iter().any(|candidate| candidate == model);
            let is_ocr_provider = endpoint.protocol.as_deref() == Some("document_ocr_v1");
            serves_model && is_document_ocr == is_ocr_provider
        }))
    }
}

fn sort_endpoints(endpoints: &mut [InferenceEndpointRoute]) {
    endpoints.sort_by(|a, b| {
        a.plugin_name
            .cmp(&b.plugin_name)
            .then_with(|| a.endpoint_id.cmp(&b.endpoint_id))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn chat_models_exclude_ocr_but_discovery_keeps_all_models() {
        let manager = PluginManager::for_test_summaries(Vec::new());
        let endpoints = [
            (Some("document_ocr_v1"), vec!["ocr-only", "shared"]),
            (Some("openai_compatible"), vec!["chat", "shared", "chat"]),
            (None, vec!["legacy"]),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (protocol, models))| InferenceEndpointRoute {
            plugin_name: format!("plugin-{index}"),
            endpoint_id: "inference".into(),
            protocol: protocol.map(str::to_string),
            address: "http://127.0.0.1:1".into(),
            models: models.into_iter().map(str::to_string).collect(),
        })
        .collect();
        manager.set_test_inference_endpoints(endpoints).await;
        assert_eq!(
            manager.inference_models().await.unwrap(),
            ["chat", "legacy", "ocr-only", "shared"]
        );
        assert_eq!(
            manager.chat_inference_models().await.unwrap(),
            ["chat", "legacy", "shared"]
        );
    }

    #[tokio::test]
    async fn request_selects_only_matching_ocr_protocol() {
        let manager = PluginManager::for_test_summaries(Vec::new());
        manager
            .set_test_inference_endpoints(vec![
                InferenceEndpointRoute {
                    plugin_name: "chat".into(),
                    endpoint_id: "chat".into(),
                    protocol: Some("openai_compatible".into()),
                    address: "http://127.0.0.1:1".into(),
                    models: vec!["same-model".into()],
                },
                InferenceEndpointRoute {
                    plugin_name: "ocr".into(),
                    endpoint_id: "ocr".into(),
                    protocol: Some("document_ocr_v1".into()),
                    address: "http://127.0.0.1:2".into(),
                    models: vec!["same-model".into()],
                },
            ])
            .await;
        let ocr = manager
            .inference_endpoint_for_request("same-model", "/v1/documents/ocr")
            .await
            .expect("lookup")
            .expect("OCR endpoint");
        assert_eq!(ocr.plugin_name, "ocr");
        let chat = manager
            .inference_endpoint_for_request("same-model", "/v1/chat/completions")
            .await
            .expect("lookup")
            .expect("chat endpoint");
        assert_eq!(chat.plugin_name, "chat");
        let payment_backend = manager
            .inference_endpoint_for_model("same-model")
            .await
            .expect("lookup")
            .expect("chat backend");
        assert_eq!(payment_backend.plugin_name, "chat");
    }
}
