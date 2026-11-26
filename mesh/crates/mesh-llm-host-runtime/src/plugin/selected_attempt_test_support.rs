//! Test-only plugin that admits or denies each `backend_selected` attempt by
//! its attempt number, for routing tests that need a NO on one candidate.
use super::{InProcessPluginRunner, InProcessPlugins, PluginManager, config};
use mesh_llm_config::{OpenAiExchangeFailurePolicy, OpenAiExchangeGrant};
use mesh_llm_plugin::openai_exchange::{OpenAiAdmissionDecision, OpenAiExchangeDecision};
use mesh_llm_plugin::{PluginMetadata, PluginRuntime, SimplePlugin};
use std::sync::{Arc, Mutex as StdMutex};

/// The response header the plugin sets on every deny (its required prefix is
/// `x-plugin-<hex plugin name>-`).
pub(crate) const DENY_HEADER: &str = "x-plugin-617474656d70742d706f6c696379-deny";

/// Targets and attempt numbers of every `backend_selected` event the plugin saw.
pub(crate) type SeenSelections = Arc<StdMutex<Vec<(String, u64)>>>;
/// Every `exchange_finished` event the plugin received.
pub(crate) type SeenTerminals = Arc<StdMutex<Vec<serde_json::Value>>>;

/// Every lifecycle callback, and every `openai.exchange.v1` channel message,
/// the plugin received, in arrival order.
pub(crate) type SeenJson = Arc<StdMutex<Vec<serde_json::Value>>>;

/// [`attempt_policy_manager`] plus the full record of what the plugin saw.
pub(crate) struct AttemptProbe {
    pub(crate) manager: PluginManager,
    pub(crate) terminals: SeenTerminals,
    /// Every lifecycle callback event, whatever its phase.
    pub(crate) lifecycle: SeenJson,
    /// Every `openai.exchange.v1` message the host published, decoded.
    pub(crate) channel: SeenJson,
}

/// Denies `backend_selected` attempts numbered `<= deny_through` (use
/// `u64::MAX` to deny all), or fails the hook when `fail` is set.
pub(crate) async fn attempt_policy_manager(
    deny_through: u64,
    fail: bool,
) -> (PluginManager, SeenSelections, SeenTerminals) {
    let (probe, seen) = attempt_policy_probe(Some(deny_through), fail).await;
    (probe.manager, seen, probe.terminals)
}

/// The same plugin as [`attempt_policy_manager`], which also declares the
/// `openai.exchange.v1` channel, so a test sees both what the host's lifecycle
/// callbacks and what its channel events say about one exchange. A
/// `deny_through` of `None` admits every attempt, the typed attempt 0 included.
pub(crate) async fn attempt_policy_probe(
    deny_through: Option<u64>,
    fail: bool,
) -> (AttemptProbe, SeenSelections) {
    let seen: SeenSelections = Arc::default();
    let terminals: SeenTerminals = Arc::default();
    let lifecycle: SeenJson = Arc::default();
    let channel: SeenJson = Arc::default();
    let lifecycle_sink = lifecycle.clone();
    let channel_sink = channel.clone();
    let sink = seen.clone();
    let terminal_sink = terminals.clone();
    let runner: InProcessPluginRunner = Arc::new(move |stream| {
        let mut hook = mesh_llm_plugin::openai_exchange::openai_exchange_hook("observe");
        hook.required = true;
        hook.admission = true;
        hook.headers = vec![DENY_HEADER.into()];
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
        .with_manifest(mesh_llm_plugin::plugin_manifest![
            hook,
            mesh_llm_plugin::mesh_channel("openai.exchange.v1")
        ]);
        let sink = sink.clone();
        let terminal_sink = terminal_sink.clone();
        let lifecycle_sink = lifecycle_sink.clone();
        let channel_sink = channel_sink.clone();
        let plugin = SimplePlugin::new(metadata)
            .on_channel_message(move |message, _| {
                let channel_sink = channel_sink.clone();
                Box::pin(async move {
                    channel_sink
                        .lock()
                        .unwrap()
                        .push(serde_json::from_slice(&message.body).unwrap());
                    Ok(())
                })
            })
            .with_openai_exchange_handler(move |_, event, _| {
                let sink = sink.clone();
                let terminal_sink = terminal_sink.clone();
                let lifecycle_sink = lifecycle_sink.clone();
                Box::pin(async move {
                    lifecycle_sink.lock().unwrap().push(event.clone());
                    let mut response_headers = Vec::new();
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
                            if deny_through.is_some_and(|through| attempt <= through) {
                                response_headers.push((DENY_HEADER.into(), "1".into()));
                                OpenAiAdmissionDecision::Deny
                            } else {
                                OpenAiAdmissionDecision::Allow
                            }
                        }
                        "request_received" => OpenAiAdmissionDecision::Allow,
                        "exchange_finished" => {
                            terminal_sink.lock().unwrap().push(event);
                            OpenAiAdmissionDecision::Abstain
                        }
                        _ => OpenAiAdmissionDecision::Abstain,
                    };
                    Ok(OpenAiExchangeDecision {
                        decision,
                        status: None,
                        reason: None,
                        annotations: Vec::new(),
                        response_headers,
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
        headers: vec![DENY_HEADER.into()],
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
    (
        AttemptProbe {
            manager,
            terminals,
            lifecycle,
            channel,
        },
        seen,
    )
}
