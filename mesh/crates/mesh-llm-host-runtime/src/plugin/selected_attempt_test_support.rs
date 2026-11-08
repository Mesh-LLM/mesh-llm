//! Test-only plugin that admits or denies each `backend_selected` attempt by
//! its attempt number, for routing tests that need a NO on one candidate.
use super::{InProcessPluginRunner, InProcessPlugins, PluginManager, config};
use mesh_llm_config::{OpenAiExchangeFailurePolicy, OpenAiExchangeGrant};
use mesh_llm_plugin::openai_exchange::{OpenAiAdmissionDecision, OpenAiExchangeDecision};
use mesh_llm_plugin::{PluginMetadata, PluginRuntime, SimplePlugin};
use std::sync::{Arc, Mutex as StdMutex};

/// Targets and attempt numbers of every `backend_selected` event the plugin saw.
pub(crate) type SeenSelections = Arc<StdMutex<Vec<(String, u64)>>>;

/// Denies `backend_selected` attempts numbered `<= deny_through` (use
/// `u64::MAX` to deny all), or fails the hook when `fail` is set.
pub(crate) async fn attempt_policy_manager(
    deny_through: u64,
    fail: bool,
) -> (PluginManager, SeenSelections) {
    let seen: SeenSelections = Arc::default();
    let sink = seen.clone();
    let runner: InProcessPluginRunner = Arc::new(move |stream| {
        let mut hook = mesh_llm_plugin::openai_exchange::openai_exchange_hook("observe");
        hook.required = true;
        hook.admission = true;
        let metadata = PluginMetadata::new(
            "attempt-policy",
            "1.0.0",
            mesh_llm_plugin::plugin_server_info(
                "attempt-policy",
                "1.0.0",
                "Policy",
                "Test policy",
                None::<String>,
            ),
        )
        .with_manifest(mesh_llm_plugin::plugin_manifest![hook]);
        let sink = sink.clone();
        let plugin =
            SimplePlugin::new(metadata).with_openai_exchange_handler(move |_, event, _| {
                let sink = sink.clone();
                Box::pin(async move {
                    let decision = match event["phase"].as_str().unwrap() {
                        "backend_selected" => {
                            let attempt = event["attempt"].as_u64().unwrap();
                            sink.lock()
                                .unwrap()
                                .push((event["target"].as_str().unwrap().to_owned(), attempt));
                            if fail {
                                return Err(mesh_llm_plugin::PluginError::invalid_request(
                                    "selected policy failed",
                                ));
                            }
                            if attempt <= deny_through {
                                OpenAiAdmissionDecision::Deny
                            } else {
                                OpenAiAdmissionDecision::Allow
                            }
                        }
                        "request_received" => OpenAiAdmissionDecision::Allow,
                        _ => OpenAiAdmissionDecision::Abstain,
                    };
                    Ok(OpenAiExchangeDecision {
                        decision,
                        status: None,
                        reason: None,
                        annotations: Vec::new(),
                        response_headers: Vec::new(),
                    })
                })
            });
        Box::pin(PluginRuntime::run_with_stream(plugin, stream))
    });
    let mut spec = config::in_process_builtin_spec("attempt-policy");
    spec.startup.optional = false;
    spec.openai_exchange_grant = Some(Box::new(OpenAiExchangeGrant {
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
        admission: true,
        metadata: true,
        failure_policy: OpenAiExchangeFailurePolicy::Required,
        deadline_ms: 1_000,
        max_body_bytes: 1_048_576,
        max_queue_bytes: 4_194_304,
        max_in_flight: 32,
        ..Default::default()
    }));
    let (tx, _rx) = tokio::sync::mpsc::channel(4);
    let manager = PluginManager::start_with_in_process(
        &config::ResolvedPlugins {
            externals: vec![spec],
            inactive: Vec::new(),
        },
        config::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        tx,
        InProcessPlugins::default().with("attempt-policy", runner),
    )
    .await
    .unwrap();
    (manager, seen)
}
