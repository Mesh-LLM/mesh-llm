//! Model discovery and protocol-aware selection of plugin inference endpoints.

use super::{InferenceEndpointRoute, PluginManager};
use crate::mesh::ModelWorkloadClass;
use anyhow::Result;

/// Whether an endpoint is dedicated to the document OCR contract.
pub(crate) fn is_document_ocr_protocol(protocol: Option<&str>) -> bool {
    protocol == Some("document_ocr_v1")
}

/// Map a declared task contract to its workload; never classify tasks as chat.
pub(crate) fn endpoint_workload_class(protocol: Option<&str>) -> ModelWorkloadClass {
    match protocol {
        Some("document_ocr_v1") => ModelWorkloadClass::DocumentOcr,
        Some("transcription_v1") => ModelWorkloadClass::SpeechRecognition,
        Some("image_text_similarity_v1") => ModelWorkloadClass::ImageTextSimilarity,
        Some("depth_estimation_v1") => ModelWorkloadClass::DepthEstimation,
        None => ModelWorkloadClass::CausalGeneration,
        Some(protocol) if protocol.eq_ignore_ascii_case("openai_compatible") => {
            ModelWorkloadClass::CausalGeneration
        }
        _ => ModelWorkloadClass::Unknown,
    }
}

fn is_chat_protocol(protocol: Option<&str>) -> bool {
    endpoint_workload_class(protocol) == ModelWorkloadClass::CausalGeneration
}

fn protocol_serves_path(protocol: Option<&str>, path: &str) -> bool {
    match path.split('?').next().unwrap_or(path) {
        "/mesh/v1/documents/ocr" => is_document_ocr_protocol(protocol),
        "/v1/audio/transcriptions" => {
            protocol == Some("transcription_v1") || is_chat_protocol(protocol)
        }
        "/mesh/v1/images/similarity" => protocol == Some("image_text_similarity_v1"),
        "/mesh/v1/images/depth" => protocol == Some("depth_estimation_v1"),
        _ => is_chat_protocol(protocol),
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
            if is_chat_protocol(endpoint.protocol.as_deref()) {
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
            is_chat_protocol(endpoint.protocol.as_deref())
                && endpoint.models.iter().any(|candidate| candidate == model)
        }))
    }

    pub async fn inference_endpoint_for_request(
        &self,
        model: &str,
        path: &str,
    ) -> Result<Option<InferenceEndpointRoute>> {
        let mut endpoints = self.inference_endpoints().await?;
        sort_endpoints(&mut endpoints);
        Ok(endpoints.into_iter().find(|endpoint| {
            let serves_model = endpoint.models.iter().any(|candidate| candidate == model);
            serves_model && protocol_serves_path(endpoint.protocol.as_deref(), path)
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
            .inference_endpoint_for_request("same-model", "/mesh/v1/documents/ocr")
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

    /// Admission is descriptor-based and dispatch is protocol-based, so the
    /// class the advertisement carries has to match what dispatch offers the
    /// endpoint for.
    #[test]
    fn endpoint_workload_class_matches_dispatch_protocols() {
        assert_eq!(
            endpoint_workload_class(Some("OPENAI_COMPATIBLE")),
            ModelWorkloadClass::CausalGeneration
        );
        assert!(protocol_serves_path(
            Some("OpenAI_Compatible"),
            "/v1/chat/completions"
        ));
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

    #[test]
    fn typed_model_contracts_do_not_admit_chat_or_other_tasks() {
        for (protocol, path, class) in [
            (
                "transcription_v1",
                "/v1/audio/transcriptions",
                ModelWorkloadClass::SpeechRecognition,
            ),
            (
                "image_text_similarity_v1",
                "/mesh/v1/images/similarity",
                ModelWorkloadClass::ImageTextSimilarity,
            ),
            (
                "depth_estimation_v1",
                "/mesh/v1/images/depth",
                ModelWorkloadClass::DepthEstimation,
            ),
        ] {
            assert_eq!(endpoint_workload_class(Some(protocol)), class);
            assert!(protocol_serves_path(Some(protocol), path));
            assert!(!protocol_serves_path(
                Some(protocol),
                "/v1/chat/completions"
            ));
            assert!(!protocol_serves_path(
                Some(protocol),
                "/mesh/v1/documents/ocr"
            ));
            assert!(!is_chat_protocol(Some(protocol)));
        }
        assert!(!protocol_serves_path(
            Some("transcription_v1"),
            "/v1/audio/translations"
        ));
        assert_eq!(
            endpoint_workload_class(Some("future_v1")),
            ModelWorkloadClass::Unknown
        );
    }
}
