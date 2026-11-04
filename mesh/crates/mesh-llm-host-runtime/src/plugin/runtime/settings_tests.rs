//! Whether `[plugin.settings]` reaches the plugin process.
//!
//! The host validates those values against the plugin's `config_schema` and
//! shows them in the console. A setting whose `restart_scope` is
//! `plugin_process` can only take effect if the process is told about it.

use super::tests::plugin_for_spec;
use super::*;
use crate::plugin::{InProcessPluginRunner, PluginStartupOptions};
use mesh_llm_plugin::{LocalStream, read_envelope, write_envelope};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// Records the initialize request the host sends, then stays connected.
async fn record_initialize(mut stream: LocalStream, seen: Arc<Mutex<String>>) -> Result<()> {
    let incoming = read_envelope(&mut stream).await?;
    let Some(proto::envelope::Payload::InitializeRequest(request)) = incoming.payload else {
        bail!("fixture expected an initialize request");
    };
    *seen.lock().unwrap() = serde_json::to_string(&request)?;
    write_envelope(
        &mut stream,
        &proto::Envelope {
            protocol_version: crate::plugin::PROTOCOL_VERSION,
            plugin_id: incoming.plugin_id,
            request_id: incoming.request_id,
            payload: Some(proto::envelope::Payload::InitializeResponse(
                proto::InitializeResponse {
                    plugin_id: "settings-fixture".into(),
                    plugin_protocol_version: crate::plugin::PROTOCOL_VERSION,
                    plugin_version: "0.0.0".into(),
                    server_info_json: serde_json::to_string(&rmcp::model::ServerConfig::default())?,
                    ..Default::default()
                },
            )),
        },
    )
    .await?;
    loop {
        let incoming = read_envelope(&mut stream).await?;
        if matches!(
            incoming.payload,
            Some(proto::envelope::Payload::ShutdownRequest(_))
        ) {
            return Ok(());
        }
    }
}

fn settings_spec(settings: &[(&str, serde_json::Value)]) -> ExternalPluginSpec {
    ExternalPluginSpec {
        operation_bindings: Vec::new(),
        openai_exchange_grant: None,
        name: "settings-fixture".into(),
        command: String::new(),
        args: Vec::new(),
        url: None,
        env: BTreeMap::new(),
        settings: settings
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect(),
        startup: PluginStartupOptions {
            optional: true,
            ..PluginStartupOptions::default()
        },
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        installed_metadata: None,
    }
}

async fn initialize_request_seen_by(settings: &[(&str, serde_json::Value)]) -> String {
    let seen = Arc::new(Mutex::new(String::new()));
    let runner_stream = seen.clone();
    let runner: InProcessPluginRunner =
        Arc::new(move |stream| Box::pin(record_initialize(stream, runner_stream.clone())));
    let mut plugin = plugin_for_spec(settings_spec(settings));
    plugin.in_process = Some(runner);
    plugin.ensure_running().await.unwrap();
    let seen = seen.lock().unwrap().clone();
    plugin.shutdown().await;
    seen
}

#[tokio::test]
async fn operator_settings_reach_the_plugin_initialize_request() {
    let initialize =
        initialize_request_seen_by(&[("greeting", serde_json::json!("marker-7f3a"))]).await;
    assert!(
        initialize.contains("marker-7f3a"),
        "[plugin.settings] never reached the plugin process; initialize was: {initialize}"
    );
}

#[tokio::test]
async fn a_plugin_without_settings_gets_an_empty_settings_payload() {
    let initialize = initialize_request_seen_by(&[]).await;
    assert!(
        initialize.contains(r#""settings_json":"{}""#),
        "a plugin with no settings must still get a well-formed empty JSON object: {initialize}"
    );
}
