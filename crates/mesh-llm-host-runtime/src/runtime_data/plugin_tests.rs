use super::{
    PluginDataKey, PluginEndpointKey, RuntimeDataCollector, RuntimeDataProducer, RuntimeDataSource,
};
use crate::plugin::{
    PluginCapabilityProvider, PluginEndpointSummary, PluginManifestOverview, PluginSummary,
    PluginWebUiState, PluginWebUiStateKind,
};
use serde_json::json;

#[test]
fn runtime_data_plugin_reports_are_scoped_by_name_and_endpoint() {
    let collector = RuntimeDataCollector::new();
    publish_alpha_plugin_reports(&collector);
    publish_beta_plugin_reports(&collector);

    assert_plugin_collection_snapshot(&collector);
    assert_alpha_plugin_snapshot(&collector);
    assert_missing_plugin_snapshots(&collector);
}

fn plugin_data_producer(
    collector: &RuntimeDataCollector,
    plugin_name: &str,
) -> RuntimeDataProducer {
    collector.producer(RuntimeDataSource {
        scope: "plugin",
        plugin_data_key: Some(PluginDataKey {
            plugin_name: plugin_name.into(),
            data_key: "summary".into(),
        }),
        plugin_endpoint_key: None,
    })
}

fn plugin_endpoint_producer(
    collector: &RuntimeDataCollector,
    plugin_name: &str,
    endpoint_id: &str,
) -> RuntimeDataProducer {
    collector.producer(RuntimeDataSource {
        scope: "plugin",
        plugin_data_key: None,
        plugin_endpoint_key: Some(PluginEndpointKey {
            plugin_name: plugin_name.into(),
            endpoint_id: endpoint_id.into(),
        }),
    })
}

fn plugin_manifest_with_endpoint(capability: &str) -> PluginManifestOverview {
    PluginManifestOverview {
        operations: 1,
        resources: 0,
        resource_templates: 0,
        prompts: 0,
        completions: 0,
        http_bindings: 0,
        endpoints: 1,
        mesh_channels: 0,
        mesh_event_subscriptions: 0,
        capabilities: vec![capability.into()],
        web_ui: None,
    }
}

fn publish_alpha_plugin_reports(collector: &RuntimeDataCollector) {
    let alpha = plugin_data_producer(collector, "alpha");
    let alpha_endpoint = plugin_endpoint_producer(collector, "alpha", "chat");

    alpha.publish_plugin_summary(PluginSummary {
        name: "alpha".into(),
        kind: "external".into(),
        enabled: true,
        status: "running".into(),
        pid: Some(1001),
        version: Some("1.0.0".into()),
        capabilities: vec!["chat".into()],
        command: Some("alpha-plugin".into()),
        args: vec!["--serve".into()],
        tools: Vec::new(),
        manifest: Some(plugin_manifest_with_endpoint("chat")),
        web_ui: PluginWebUiState {
            state: PluginWebUiStateKind::Ready,
            declared: true,
            enabled: true,
            available: true,
            unavailable_reason: None,
            pages: Vec::new(),
            config_sections: Vec::new(),
            asset_base_url: Some("/api/plugins/alpha/web-ui/assets/".into()),
        },
        startup: None,
        error: None,
    });
    alpha.publish_plugin_manifest(plugin_manifest_with_endpoint("chat"));
    alpha.publish_plugin_providers(vec![PluginCapabilityProvider {
        capability: "chat".into(),
        plugin_name: "alpha".into(),
        plugin_status: "running".into(),
        endpoint_id: Some("chat".into()),
        available: true,
        detail: None,
    }]);
    alpha.publish_plugin_payload("metrics", json!({"requests": 2}));
    alpha_endpoint.publish_plugin_endpoint(PluginEndpointSummary {
        plugin_name: "alpha".into(),
        plugin_status: "running".into(),
        endpoint_id: "chat".into(),
        state: "healthy".into(),
        available: true,
        kind: "mcp".into(),
        transport_kind: "http".into(),
        protocol: Some("http".into()),
        address: Some("http://127.0.0.1:9000/mcp".into()),
        args: Vec::new(),
        namespace: Some("alpha.chat".into()),
        supports_streaming: true,
        managed_by_plugin: true,
        detail: None,
        models: vec!["alpha-model".into()],
    });
}

fn publish_beta_plugin_reports(collector: &RuntimeDataCollector) {
    let beta = plugin_data_producer(collector, "beta");
    let beta_endpoint = plugin_endpoint_producer(collector, "beta", "embed");

    beta.publish_plugin_summary(PluginSummary {
        name: "beta".into(),
        kind: "external".into(),
        enabled: true,
        status: "disabled".into(),
        pid: None,
        version: None,
        capabilities: vec!["embed".into()],
        command: Some("beta-plugin".into()),
        args: Vec::new(),
        tools: Vec::new(),
        manifest: None,
        web_ui: PluginWebUiState::default(),
        startup: None,
        error: Some("disabled".into()),
    });
    beta.publish_plugin_payload("metrics", json!({"requests": 5}));
    beta_endpoint.publish_plugin_endpoint(PluginEndpointSummary {
        plugin_name: "beta".into(),
        plugin_status: "disabled".into(),
        endpoint_id: "embed".into(),
        state: "unavailable".into(),
        available: false,
        kind: "inference".into(),
        transport_kind: "tcp".into(),
        protocol: None,
        address: Some("127.0.0.1:9444".into()),
        args: Vec::new(),
        namespace: None,
        supports_streaming: false,
        managed_by_plugin: false,
        detail: Some("disabled".into()),
        models: vec!["beta-model".into()],
    });
}

fn assert_plugin_collection_snapshot(collector: &RuntimeDataCollector) {
    let all = collector.plugins_snapshot();
    assert_eq!(
        all.plugins
            .iter()
            .map(|plugin| plugin.name.as_str())
            .collect::<Vec<_>>(),
        vec!["alpha", "beta"]
    );
    assert_eq!(
        all.endpoints
            .iter()
            .map(|endpoint| (endpoint.plugin_name.as_str(), endpoint.endpoint_id.as_str()))
            .collect::<Vec<_>>(),
        vec![("alpha", "chat"), ("beta", "embed")]
    );
}

fn assert_alpha_plugin_snapshot(collector: &RuntimeDataCollector) {
    let alpha_snapshot = collector.plugin_snapshot("alpha");
    assert_eq!(alpha_snapshot.plugin_name, "alpha");
    assert_eq!(
        alpha_snapshot
            .summary
            .as_ref()
            .map(|summary| summary.name.as_str()),
        Some("alpha")
    );
    assert_eq!(
        alpha_snapshot
            .manifest
            .as_ref()
            .map(|manifest| manifest.endpoints),
        Some(1)
    );
    assert_eq!(alpha_snapshot.providers.len(), 1);
    assert_eq!(
        alpha_snapshot
            .summary
            .as_ref()
            .map(|summary| summary.web_ui.state),
        Some(PluginWebUiStateKind::Ready)
    );
    assert_eq!(
        alpha_snapshot
            .summary
            .as_ref()
            .and_then(|summary| summary.web_ui.asset_base_url.as_deref()),
        Some("/api/plugins/alpha/web-ui/assets/")
    );
    assert_eq!(
        alpha_snapshot.payloads.get("metrics"),
        Some(&json!({"requests": 2}))
    );
    assert_eq!(alpha_snapshot.endpoints.len(), 1);
    assert_eq!(alpha_snapshot.endpoints[0].endpoint_id, "chat");
}

fn assert_missing_plugin_snapshots(collector: &RuntimeDataCollector) {
    assert!(collector.plugin_snapshot("gamma").summary.is_none());
    assert!(collector.plugin_snapshot("gamma").endpoints.is_empty());
    assert_eq!(
        collector
            .plugin_endpoint_snapshot("alpha", "chat")
            .as_ref()
            .map(|endpoint| endpoint.address.as_deref()),
        Some(Some("http://127.0.0.1:9000/mcp"))
    );
    assert!(
        collector
            .plugin_endpoint_snapshot("alpha", "embed")
            .is_none()
    );
    assert!(collector.plugin_endpoint_snapshot("beta", "chat").is_none());
}
