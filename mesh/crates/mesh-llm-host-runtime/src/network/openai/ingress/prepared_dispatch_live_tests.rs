//! Actual installed policy admission over planner and injected strong requests.
use super::lifecycle_live_tests::LiveHost;

use crate::plugin::{ExternalPluginSpec, PluginHostMode, PluginManager, ResolvedPlugins};
use mesh_llm_config::{OpenAiExchangeFailurePolicy, OpenAiExchangeGrant};

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

async fn select_policy(host: &mut LiveHost, deny_model: &str, virtual_echo: bool) {
    host.manager.shutdown().await;
    let mut args = vec!["--body".into(), "--admission".into()];
    if virtual_echo {
        args.push("--virtual-echo".into());
    }
    let spec = ExternalPluginSpec {
        operation_bindings: Vec::new(),
        name: "openai-exchange-observer".into(),
        command: host
            .installed_metadata
            .executable_path()
            .display()
            .to_string(),
        args,
        url: None,
        settings: Default::default(),
        startup: Default::default(),
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        endpoint_models: None,
        env: BTreeMap::from([
            (
                "MESH_LLM_EXEMPLAR_DENY_SELECTED_MODEL".into(),
                deny_model.into(),
            ),
            (
                "MESH_LLM_EXEMPLAR_EVENT_LOG".into(),
                host.root.path().join("events.jsonl").display().to_string(),
            ),
            (
                "MESH_LLM_EXEMPLAR_VIRTUAL_LOG".into(),
                host.root.path().join("virtual.json").display().to_string(),
            ),
        ]),
        openai_exchange_grant: Some(Box::new(OpenAiExchangeGrant {
            endpoints: vec![
                "chat_completions".into(),
                "completions".into(),
                "responses".into(),
            ],
            phases: vec![
                "request_received".into(),
                "backend_selected".into(),
                "exchange_finished".into(),
            ],
            request_body: true,
            effective_request_body: true,
            response_body: true,
            admission: true,
            metadata: true,
            deadline_ms: 2000,
            max_body_bytes: 1048576,
            max_queue_bytes: 4194304,
            max_in_flight: 8,
            failure_policy: OpenAiExchangeFailurePolicy::Required,
            ..Default::default()
        })),
        installed_metadata: Some(host.installed_metadata.clone()),
    };
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    host.manager = PluginManager::start(
        &ResolvedPlugins {
            externals: vec![spec],
            inactive: Vec::new(),
        },
        PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        tx,
    )
    .await
    .unwrap();
    assert_eq!(host.manager.list().await[0].status, "running");
    host.node.set_plugin_manager(host.manager.clone()).await;
    host.node.start_plugin_channel_forwarder(rx);
}

#[tokio::test]
#[ignore = "requires just package-openai-exchange-exemplar"]
async fn installed_lifecycle_virtual_admission_observes_final_invocation_before_backend_runs() {
    for (deny, path, endpoint) in [
        (true, "/v1/chat/completions", "chat_completions"),
        (false, "/v1/responses", "responses"),
    ] {
        let mut host = LiveHost::start(true, true).await;
        select_policy(&mut host, if deny { "exchange-echo" } else { "" }, true).await;
        let body: &[u8] = if path == "/v1/responses" {
            br#"{"model":"exchange-echo","input":"echo this","stream":false}"#
        } else {
            br#"{"model":"exchange-echo","messages":[{"role":"user","content":"echo this"}],"stream":false}"#
        };
        let response = host.request(path, body).await;
        let invocation_file = host.root.path().join("virtual.json");
        let events = host.events();
        let selected = events
            .iter()
            .find(|event| event["phase"] == "backend_selected")
            .expect("virtual dispatch must be observed");
        assert_eq!(selected["path"], path);
        assert_eq!(selected["endpoint"], endpoint);
        assert_eq!(
            selected["effective_request_encoding"],
            "plugin_invocation_json"
        );
        assert_eq!(selected["body"]["request"]["model"], "exchange-echo");
        assert!(selected["body"]["candidates"].is_array());
        if deny {
            assert!(response.starts_with(b"HTTP/1.1 403 "));
            assert!(
                !invocation_file.exists(),
                "denied virtual backend must not run"
            );
        } else {
            assert!(response.starts_with(b"HTTP/1.1 200 "));
            let independently_received = std::fs::read(&invocation_file).unwrap();
            assert_eq!(
                selected["effective_request_wire_digest"]["sha256"],
                hex::encode(Sha256::digest(&independently_received))
            );
            assert_eq!(
                selected["effective_request_wire_digest"]["byte_count"],
                independently_received.len()
            );
        }
        assert!(host.requests.lock().await.is_empty());
        host.stop().await;
    }
}
