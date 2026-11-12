use super::*;
use crate::mesh::{Node, NodeRole};

#[tokio::test]
async fn endpoint_outputs_reach_local_discovery_and_gossip_without_input_capabilities() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let manager = PluginManager::for_test_summaries(Vec::new());
    let (models, model_output_modalities) = super::super::endpoint_models::parse_models(
        &serde_json::json!({"data": [
            {"id": "image-model", "output_modalities": ["text", "image"]},
            {"id": "text-model", "output_modalities": ["text"]},
            {"id": "legacy-model"}
        ]}),
        Some("openai_compatible"),
    );
    manager
        .set_test_inference_endpoints(vec![InferenceEndpointRoute {
            plugin_name: "generic-provider".into(),
            endpoint_id: "inference".into(),
            protocol: Some("openai_compatible".into()),
            address: "http://127.0.0.1:1/v1".into(),
            models: models.clone(),
        }])
        .await;
    manager.inner.endpoint_health.lock().await.insert(
        endpoint_key("generic-provider", "inference"),
        EndpointHealthState {
            record: EndpointHealthRecord {
                state: "healthy".into(),
                available: true,
                detail: None,
                models,
                model_output_modalities,
            },
            first_checked_at: Instant::now(),
            consecutive_failures: 0,
        },
    );
    node.set_plugin_manager(manager).await;

    let descriptors = node.plugin_inference_descriptors().await;
    let catalog = crate::models::discovery::collect(&node).await.unwrap();
    for (name, expected) in [
        ("image-model", vec!["image", "text"]),
        ("text-model", vec!["text"]),
        ("legacy-model", Vec::new()),
    ] {
        let descriptor = descriptors
            .iter()
            .find(|d| d.identity.model_name == name)
            .unwrap();
        assert_eq!(
            descriptor.metadata.as_ref().unwrap().output_modalities,
            expected
        );
        assert!(!descriptor.capabilities.multimodal);
        assert_eq!(
            descriptor.capabilities.vision,
            crate::models::CapabilityLevel::None
        );
        let model = catalog.iter().find(|model| model.id == name).unwrap();
        assert_eq!(
            model.providers[0]
                .metadata
                .as_ref()
                .unwrap()
                .output_modalities,
            expected
        );
    }
    node.close_endpoint().await;
}
