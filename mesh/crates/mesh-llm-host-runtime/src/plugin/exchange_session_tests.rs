//! Session phase budgets and prompt retention over the real plugin protocol.
use super::*;
use crate::plugin::{InProcessPluginRunner, InProcessPlugins, config};
use mesh_llm_plugin::{PluginMetadata, PluginRuntime, SimplePlugin, proto};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn receive_response(
    listener: mesh_llm_plugin::LocalListener,
    token: String,
    received: tokio::sync::mpsc::Sender<Vec<u8>>,
) {
    let stream = listener.accept().await.unwrap();
    let (mut read, mut write) = stream.into_split();
    let mut auth = [0u8; 36];
    read.read_exact(&mut auth).await.unwrap();
    assert_eq!(auth.as_slice(), token.as_bytes());
    let mut bytes = Vec::new();
    read.read_to_end(&mut bytes).await.unwrap();
    let commitment = commit_wire_bytes(&bytes);
    let receipt = json!({"sha256":commitment.sha256,"byte_count":commitment.byte_count});
    write
        .write_all(format!("{receipt}\n").as_bytes())
        .await
        .unwrap();
    write.shutdown().await.unwrap();
    received.send(bytes).await.unwrap();
}

fn delayed_observer(denied: bool, received: tokio::sync::mpsc::Sender<Vec<u8>>) -> SimplePlugin {
    use mesh_llm_plugin::openai_exchange::openai_exchange_hook;
    let mut hook = openai_exchange_hook("observe");
    hook.required = true;
    hook.admission = true;
    hook.response_body = true;
    hook.deadline_ms = 1_000;
    hook.endpoints = vec!["chat_completions".into()];
    hook.phases = vec!["request_received".into(), "exchange_finished".into()];
    let metadata = PluginMetadata::new(
        "delayed-observer",
        "1.0.0",
        mesh_llm_plugin::plugin_server_info(
            "delayed-observer",
            "1.0.0",
            "Observer",
            "Test observer",
            None::<String>,
        ),
    )
    .with_manifest(mesh_llm_plugin::plugin_manifest![hook]);
    SimplePlugin::new(metadata)
        .with_openai_exchange_handler(move |_, event, _| {
            Box::pin(async move {
                let request = event["phase"] == "request_received";
                if request {
                    tokio::time::sleep(Duration::from_millis(700)).await;
                }
                Ok(OpenAiExchangeDecision {
                    decision: if request && denied {
                        OpenAiAdmissionDecision::Deny
                    } else if request {
                        OpenAiAdmissionDecision::Allow
                    } else {
                        OpenAiAdmissionDecision::Abstain
                    },
                    reason: None,
                    annotations: Vec::new(),
                    response_headers: Vec::new(),
                })
            })
        })
        .on_open_stream(move |request: proto::OpenStreamRequest, _| {
            let received = received.clone();
            Box::pin(async move {
                // Each operation fits its budget; their combined latency does not.
                tokio::time::sleep(Duration::from_millis(700)).await;
                let id = uuid::Uuid::new_v4().simple().to_string();
                let listener = mesh_llm_plugin::bind_side_stream("bd", &id[..16])
                    .await
                    .map_err(|error| {
                        mesh_llm_plugin::PluginError::invalid_request(error.to_string())
                    })?;
                let mut response = listener.open_stream_response(&request);
                let token = uuid::Uuid::new_v4().to_string();
                response.token = Some(token.clone());
                tokio::spawn(receive_response(listener, token, received));
                Ok(Some(response))
            })
        })
}

async fn observer_manager(denied: bool) -> (PluginManager, tokio::sync::mpsc::Receiver<Vec<u8>>) {
    let (received, receiver) = tokio::sync::mpsc::channel(1);
    let runner: InProcessPluginRunner = Arc::new(move |stream| {
        Box::pin(PluginRuntime::run_with_stream(
            delayed_observer(denied, received.clone()),
            stream,
        ))
    });
    let mut spec = config::in_process_builtin_spec("delayed-observer");
    spec.startup.optional = false;
    spec.openai_exchange_grant = Some(Box::new(OpenAiExchangeGrant {
        endpoints: vec!["chat_completions".into()],
        phases: vec!["request_received".into(), "exchange_finished".into()],
        admission: true,
        response_body: true,
        metadata: true,
        deadline_ms: 1_000,
        max_body_bytes: 1_024,
        max_queue_bytes: 1_024,
        max_in_flight: 1,
        failure_policy: OpenAiExchangeFailurePolicy::Required,
        ..Default::default()
    }));
    let (mesh_tx, _mesh_rx) = tokio::sync::mpsc::channel(4);
    let manager = PluginManager::start_with_in_process(
        &config::ResolvedPlugins {
            externals: vec![spec],
            inactive: Vec::new(),
        },
        config::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
        InProcessPlugins::default().with("delayed-observer", runner),
    )
    .await
    .unwrap();
    (manager, receiver)
}

#[tokio::test]
async fn slow_allow_hook_preserves_independent_response_open_budget_and_releases_prompt() {
    let (manager, mut receiver) = observer_manager(false).await;
    let event = request_event(
        "slow-allow".into(),
        "chat_completions",
        "POST",
        "/v1/chat/completions",
        br#"{"messages":[{"content":"private prompt"}]}"#,
        Default::default(),
        false,
    );
    assert!(event.get("body").is_some());
    let (mut session, result) = ExchangeSession::begin(&manager, event).await;
    assert!(
        result.error_status().is_none(),
        "required response open must get its own budget"
    );
    assert!(!result.evidence_unavailable);
    assert!(session.event.get("body").is_none());
    assert!(session.event.get("body_hex").is_none());
    let observer = session.observer();
    assert!(observer.try_chunk(0, b"response"));
    observer.finish(commit_wire_bytes(b"response"));
    session.finish("completed").await;
    assert_eq!(receiver.recv().await.unwrap(), b"response");
    manager.shutdown().await;
}

#[tokio::test]
async fn slow_denial_never_copies_backend_response_bytes() {
    let (manager, mut receiver) = observer_manager(true).await;
    let event = request_event(
        "slow-deny".into(),
        "chat_completions",
        "POST",
        "/v1/chat/completions",
        b"{}",
        Default::default(),
        false,
    );
    let (mut session, result) = ExchangeSession::begin(&manager, event).await;
    assert_eq!(result.error_status(), Some(403));
    assert!(!result.required_failure);
    // Admission rejection does not dispatch a backend or enqueue its entity bytes.
    session.observer().finish(commit_wire_bytes(b""));
    session.finish("admission_denied").await;
    assert!(receiver.recv().await.unwrap().is_empty());
    manager.shutdown().await;
}
