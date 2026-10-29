//! Model discovery and protocol-aware selection of plugin inference endpoints.

use super::{InferenceEndpointRoute, PluginManager};
use crate::mesh::ModelWorkloadClass;
use anyhow::Result;

/// The one plugin protocol that serves document OCR. Every other protocol is a
/// chat-compatible endpoint, and this is exactly how
/// [`PluginManager::inference_endpoint_for_request`] splits dispatch.
pub(crate) fn is_document_ocr_protocol(protocol: Option<&str>) -> bool {
    protocol == Some("document_ocr_v1")
}

/// Workload class a plugin endpoint serves, derived from the protocol it
/// declares.
///
/// A model is admitted on descriptor evidence, while dispatch selects an
/// endpoint by protocol, so the advertisement has to describe every endpoint
/// the dispatcher is willing to use. Anything the OCR check does not claim is
/// offered chat traffic by [`PluginManager::inference_endpoint_for_request`],
/// so it advertises the chat class.
pub(crate) fn endpoint_workload_class(protocol: Option<&str>) -> ModelWorkloadClass {
    if is_document_ocr_protocol(protocol) {
        ModelWorkloadClass::DocumentOcr
    } else {
        ModelWorkloadClass::CausalGeneration
    }
}

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
            if !is_document_ocr_protocol(endpoint.protocol.as_deref()) {
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
            !is_document_ocr_protocol(endpoint.protocol.as_deref())
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
            let is_ocr_provider = is_document_ocr_protocol(endpoint.protocol.as_deref());
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
        let named_backend = manager
            .inference_endpoint_for_model("same-model")
            .await
            .expect("lookup")
            .expect("chat backend");
        assert_eq!(named_backend.plugin_name, "chat");
    }

    /// Admission is descriptor-based and dispatch is protocol-based, so the
    /// class the advertisement carries has to match what dispatch offers the
    /// endpoint for.
    #[test]
    fn endpoint_workload_class_matches_dispatch_protocols() {
        assert_eq!(
            endpoint_workload_class(Some("document_ocr_v1")),
            ModelWorkloadClass::DocumentOcr
        );
        assert_eq!(
            endpoint_workload_class(Some("openai_compatible")),
            ModelWorkloadClass::CausalGeneration
        );
        assert_eq!(
            endpoint_workload_class(None),
            ModelWorkloadClass::CausalGeneration,
            "a protocol-less endpoint is a chat-compatible endpoint to the dispatcher"
        );
    }
}
