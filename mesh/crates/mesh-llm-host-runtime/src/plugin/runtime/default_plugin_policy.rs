use super::super::proto;
use mesh_llm_plugin_manager::defaults::Surface;

/// Keep only what a default-managed plugin's entry allows from its initialize
/// answer: the host surfaces in `allows`, the capability strings in
/// `capabilities`. Filter the untrusted plugin declaration before any host
/// surface publishes it. Returns the names of what was removed.
pub(super) fn restrict_default_plugin(
    init: &mut proto::InitializeResponse,
    allows: &[Surface],
    capabilities: &[&str],
) -> Vec<&'static str> {
    let may = |surface: Surface| allows.contains(&surface);
    let keep_capability = |capability: &String| {
        capabilities.contains(&capability.as_str())
            || capability == mesh_llm_plugin::host_capabilities::INVOCATION_CALLBACKS
    };
    let mut removed = Vec::new();
    let before = init.capabilities.len();
    init.capabilities.retain(keep_capability);
    let mut capabilities_removed = init.capabilities.len() != before;
    if let Some(manifest) = init.manifest.as_mut() {
        let before = manifest.capabilities.len();
        manifest.capabilities.retain(keep_capability);
        capabilities_removed |= manifest.capabilities.len() != before;
        if !may(Surface::WebUi) && manifest.web_ui.take().is_some() {
            removed.push(Surface::WebUi.name());
        }
        if !may(Surface::Config) && manifest.config_schema.take().is_some() {
            removed.push(Surface::Config.name());
        }
        if !may(Surface::MeshChannels) && !manifest.mesh_channels.is_empty() {
            manifest.mesh_channels.clear();
            removed.push(Surface::MeshChannels.name());
        }
        if !may(Surface::MeshEvents) && !manifest.mesh_event_subscriptions.is_empty() {
            manifest.mesh_event_subscriptions.clear();
            removed.push(Surface::MeshEvents.name());
        }
        if !may(Surface::HttpRoutes) && !manifest.http_bindings.is_empty() {
            manifest.http_bindings.clear();
            removed.push(Surface::HttpRoutes.name());
        }
        if !may(Surface::McpOperations)
            && !(manifest.operations.is_empty()
                && manifest.resources.is_empty()
                && manifest.resource_templates.is_empty()
                && manifest.prompts.is_empty()
                && manifest.completions.is_empty())
        {
            manifest.operations.clear();
            manifest.resources.clear();
            manifest.resource_templates.clear();
            manifest.prompts.clear();
            manifest.completions.clear();
            removed.push(Surface::McpOperations.name());
        }
        let is_inference = |endpoint: &proto::EndpointManifest| {
            proto::EndpointKind::try_from(endpoint.kind) == Ok(proto::EndpointKind::Inference)
        };
        if !may(Surface::InferenceEndpoints) && manifest.endpoints.iter().any(is_inference) {
            manifest
                .endpoints
                .retain(|endpoint| !is_inference(endpoint));
            removed.push(Surface::InferenceEndpoints.name());
        }
        if !may(Surface::OtherEndpoints) && manifest.endpoints.iter().any(|e| !is_inference(e)) {
            manifest.endpoints.retain(is_inference);
            removed.push(Surface::OtherEndpoints.name());
        }
        if !may(Surface::VirtualModels) && !manifest.virtual_models.is_empty() {
            manifest.virtual_models.clear();
            removed.push(Surface::VirtualModels.name());
        }
        if !may(Surface::OpenAiExchangeHook) && manifest.openai_exchange_hook.take().is_some() {
            removed.push(Surface::OpenAiExchangeHook.name());
        }
        if !may(Surface::StatusContributions) && manifest.plugin_status.take().is_some() {
            removed.push(Surface::StatusContributions.name());
        }
        if !may(Surface::InferenceObservation) && manifest.inference_observation.take().is_some() {
            removed.push(Surface::InferenceObservation.name());
        }
    }
    if capabilities_removed {
        removed.push("capabilities");
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::super::super::proto;
    use super::restrict_default_plugin;
    use mesh_llm_plugin_manager::defaults::{Surface, UNLISTED_DEFAULT_ALLOWS};
    use std::sync::atomic::Ordering;

    #[test]
    fn a_default_plugin_keeps_only_what_its_entry_allows() {
        let mut init = proto::InitializeResponse {
            capabilities: vec!["admission_policy.v1".into(), "audit.v1".into()],
            manifest: Some(proto::PluginManifest {
                endpoints: vec![
                    proto::EndpointManifest {
                        kind: proto::EndpointKind::Inference as i32,
                        ..Default::default()
                    },
                    proto::EndpointManifest {
                        kind: proto::EndpointKind::Mcp as i32,
                        ..Default::default()
                    },
                ],
                capabilities: vec!["endpoint:inference".into(), "audit.v1".into()],
                virtual_models: vec![proto::VirtualModelManifest::default()],
                mesh_channels: vec![proto::MeshChannelManifest::default()],
                web_ui: Some(proto::PluginWebUiManifest::default()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let removed = restrict_default_plugin(
            &mut init,
            &[
                Surface::WebUi,
                Surface::MeshChannels,
                Surface::OtherEndpoints,
            ],
            &["audit.v1"],
        );
        assert_eq!(
            removed,
            ["inference endpoints", "virtual models", "capabilities"]
        );
        assert_eq!(init.capabilities, ["audit.v1"]);
        let manifest = init.manifest.unwrap();
        assert_eq!(manifest.capabilities, ["audit.v1"]);
        assert_eq!(manifest.endpoints.len(), 1);
        assert_eq!(manifest.endpoints[0].kind, proto::EndpointKind::Mcp as i32);
        assert!(
            manifest.virtual_models.is_empty(),
            "a default cannot list a model"
        );
        assert_eq!(manifest.mesh_channels.len(), 1);
        assert!(manifest.web_ui.is_some());
    }

    #[test]
    fn a_default_keeps_the_openai_exchange_hook_only_when_its_entry_allows_it() {
        let with_hook = || proto::InitializeResponse {
            manifest: Some(proto::PluginManifest {
                openai_exchange_hook: Some(
                    mesh_llm_plugin::openai_exchange::openai_exchange_hook("observe").into(),
                ),
                ..Default::default()
            }),
            ..Default::default()
        };

        let mut init = with_hook();
        let removed = restrict_default_plugin(&mut init, &[Surface::WebUi], &[]);
        assert_eq!(removed, ["OpenAI exchange hook"]);
        assert!(init.manifest.unwrap().openai_exchange_hook.is_none());

        let mut init = with_hook();
        let removed = restrict_default_plugin(&mut init, &[Surface::OpenAiExchangeHook], &[]);
        assert!(removed.is_empty());
        assert!(init.manifest.unwrap().openai_exchange_hook.is_some());

        // A default no longer on this build's list does not keep it either.
        let mut init = with_hook();
        let removed = restrict_default_plugin(&mut init, UNLISTED_DEFAULT_ALLOWS, &[]);
        assert_eq!(removed, ["OpenAI exchange hook"]);
    }

    #[test]
    fn a_default_keeps_its_status_contribution_only_when_its_entry_allows_it() {
        let with_status = || proto::InitializeResponse {
            manifest: Some(proto::PluginManifest {
                plugin_status: Some(proto::PluginStatusManifest {
                    schema_version: 1,
                    required: false,
                }),
                ..Default::default()
            }),
            ..Default::default()
        };

        let mut init = with_status();
        let removed = restrict_default_plugin(&mut init, &[Surface::WebUi], &[]);
        assert_eq!(removed, ["status contributions"]);
        assert!(init.manifest.unwrap().plugin_status.is_none());

        let mut init = with_status();
        let removed = restrict_default_plugin(&mut init, &[Surface::StatusContributions], &[]);
        assert!(removed.is_empty());
        assert!(init.manifest.unwrap().plugin_status.is_some());

        // Unlike the exchange hook, it shows only the plugin's own data, so an
        // unlisted default keeps it.
        let mut init = with_status();
        assert!(restrict_default_plugin(&mut init, UNLISTED_DEFAULT_ALLOWS, &[]).is_empty());
    }

    #[test]
    fn a_default_observes_local_inference_only_when_its_entry_allows_it() {
        let with_observation = || proto::InferenceObservationManifest {
            contract_version: 1,
            required: false,
        };
        let init = |observation| proto::InitializeResponse {
            manifest: Some(proto::PluginManifest {
                inference_observation: Some(observation),
                ..Default::default()
            }),
            ..Default::default()
        };

        let mut allowed = init(with_observation());
        assert!(
            restrict_default_plugin(&mut allowed, &[Surface::InferenceObservation], &[]).is_empty()
        );
        assert!(allowed.manifest.unwrap().inference_observation.is_some());

        // Not even an unlisted default keeps it: it sees every execution here.
        let mut unlisted = init(with_observation());
        assert_eq!(
            restrict_default_plugin(&mut unlisted, UNLISTED_DEFAULT_ALLOWS, &[]),
            ["local inference observation"]
        );
        assert!(unlisted.manifest.unwrap().inference_observation.is_none());
    }

    #[test]
    fn a_default_within_its_allowlist_loses_nothing_and_warns_of_nothing() {
        let mut init = proto::InitializeResponse {
            manifest: Some(proto::PluginManifest {
                mesh_channels: vec![proto::MeshChannelManifest::default()],
                http_bindings: vec![proto::HttpBindingManifest::default()],
                ..Default::default()
            }),
            ..Default::default()
        };
        let removed = restrict_default_plugin(
            &mut init,
            &[Surface::MeshChannels, Surface::HttpRoutes],
            &[],
        );
        assert!(removed.is_empty());
    }

    #[tokio::test]
    async fn a_disallowed_default_hook_is_removed_before_lifecycle_validation() {
        let temp = tempfile::tempdir().unwrap();
        let mut spec = super::super::tests::plugin_spec(
            &temp,
            None,
            mesh_llm_plugin_manager::store::InstalledPluginWebUiValidationStatus::Valid,
            None,
        );
        spec.name = "capsules".into();
        spec.installed_metadata.as_mut().unwrap().default_managed = true;
        let plugin = super::super::tests::plugin_for_spec(spec);
        plugin.authenticated_peer.store(false, Ordering::Release);

        let init = proto::InitializeResponse {
            plugin_id: "capsules".into(),
            plugin_protocol_version: super::super::super::PROTOCOL_VERSION,
            manifest: Some(proto::PluginManifest {
                openai_exchange_hook: Some(
                    mesh_llm_plugin::openai_exchange::openai_exchange_hook("observe").into(),
                ),
                ..Default::default()
            }),
            ..Default::default()
        };
        let response = proto::Envelope {
            payload: Some(proto::envelope::Payload::InitializeResponse(init)),
            ..Default::default()
        };

        let filtered = plugin
            .parse_initialize_response(0, response)
            .await
            .expect("the disallowed hook is removed before lifecycle validation");

        assert!(filtered.manifest.unwrap().openai_exchange_hook.is_none());
    }
}
