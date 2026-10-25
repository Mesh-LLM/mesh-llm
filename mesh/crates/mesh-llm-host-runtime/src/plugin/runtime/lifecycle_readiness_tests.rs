use super::super::{PluginRuntime, tests::plugin_for_spec};
use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};

fn observer() -> ExternalPlugin {
    plugin_for_spec(super::super::ExternalPluginSpec {
        name: "observer".into(),
        command: "unused".into(),
        args: Vec::new(),
        url: None,
        env: Default::default(),
        startup: Default::default(),
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        installed_metadata: None,
        // Model the original ungranted spec with permissions applied later.
        openai_exchange_grant: None,
    })
}

async fn install_replacement(
    plugin: &ExternalPlugin,
    authenticated: bool,
) -> mpsc::Receiver<proto::Envelope> {
    let (outbound_tx, outbound_rx) = mpsc::channel(4);
    *plugin.runtime.lock().await = Some(PluginRuntime {
        generation: 8,
        authenticated_peer: authenticated,
        initialized_lifecycle: None,
        _child: None,
        connection_task: tokio::spawn(std::future::pending()),
        outbound_tx,
        pending: Arc::new(Mutex::new(HashMap::new())),
    });
    outbound_rx
}

fn body_request() -> proto::OpenStreamRequest {
    proto::OpenStreamRequest {
        stream_id: "body".into(),
        metadata_json: Some(serde_json::json!({"kind":"openai_exchange_original"}).to_string()),
        ..Default::default()
    }
}

fn initialization(handler: &str, request_body: bool) -> proto::InitializeResponse {
    let mut hook = mesh_llm_plugin::openai_exchange::openai_exchange_hook(handler);
    hook.request_body = request_body;
    proto::InitializeResponse {
        manifest: Some(proto::PluginManifest {
            openai_exchange_hook: Some(Box::new(hook)),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[tokio::test]
async fn cached_manifest_cannot_authorize_concurrent_callback_or_body_to_replacement() {
    for authenticated in [false, true] {
        let plugin = observer();
        *plugin.manifest.lock().await = initialization("observe", true).manifest;
        let cached = plugin.manifest_snapshot().await.unwrap();
        assert!(cached.openai_exchange_hook.unwrap().request_body);
        let mut outbound = install_replacement(&plugin, authenticated).await;
        let declaration = mesh_llm_plugin::openai_exchange::openai_exchange_hook("observe");
        let (callback, body) = tokio::join!(
            plugin.invoke_exchange_service(
                &declaration,
                "{\"body\":\"private\"}",
                Instant::now() + Duration::from_secs(1)
            ),
            plugin.open_lifecycle_stream(body_request(), &declaration),
        );
        assert!(callback.is_err());
        assert!(body.is_err());
        assert!(matches!(
            outbound.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert!(
            plugin
                .runtime
                .lock()
                .await
                .as_ref()
                .unwrap()
                .pending
                .lock()
                .await
                .is_empty()
        );
        plugin.shutdown().await;
    }
}

#[tokio::test]
async fn only_current_authenticated_initialized_declaration_authorizes_handles() {
    let plugin = observer();
    let mut outbound = install_replacement(&plugin, true).await;
    let init = initialization("new-handler", false);
    assert!(
        plugin
            .publish_initialized_state(7, &init, ServerConfig::default())
            .await
            .is_err()
    );
    assert!(
        plugin
            .lifecycle_runtime_handles(LifecycleOperation::Callback(
                init.manifest
                    .as_ref()
                    .unwrap()
                    .openai_exchange_hook
                    .as_deref()
                    .unwrap()
            ))
            .await
            .is_err()
    );
    plugin
        .publish_initialized_state(8, &init, ServerConfig::default())
        .await
        .unwrap();
    assert!(
        plugin
            .lifecycle_runtime_handles(LifecycleOperation::Callback(
                init.manifest
                    .as_ref()
                    .unwrap()
                    .openai_exchange_hook
                    .as_deref()
                    .unwrap()
            ))
            .await
            .is_ok()
    );
    assert!(
        plugin
            .invoke_exchange_service(
                &mesh_llm_plugin::openai_exchange::openai_exchange_hook("observe"),
                "{}",
                Instant::now() + Duration::from_secs(1)
            )
            .await
            .is_err()
    );
    assert!(
        plugin
            .open_lifecycle_stream(
                body_request(),
                &mesh_llm_plugin::openai_exchange::openai_exchange_hook("observe")
            )
            .await
            .is_err()
    );
    assert!(matches!(
        outbound.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    plugin.shutdown().await;

    let plugin = observer();
    let _outbound = install_replacement(&plugin, false).await;
    assert!(
        plugin
            .publish_initialized_state(8, &initialization("observe", true), ServerConfig::default())
            .await
            .is_err()
    );
    assert!(
        plugin
            .runtime
            .lock()
            .await
            .as_ref()
            .unwrap()
            .initialized_lifecycle
            .is_none()
    );
    plugin.shutdown().await;
}

#[tokio::test]
async fn same_handler_reduced_declaration_rejects_cached_callback_and_body_authority() {
    let original = initialization("observe", true);
    let expected = original
        .manifest
        .as_ref()
        .unwrap()
        .openai_exchange_hook
        .as_deref()
        .unwrap();
    for reduction in ["endpoint", "phase", "body", "admission"] {
        let plugin = observer();
        let mut outbound = install_replacement(&plugin, true).await;
        let mut replacement = original.clone();
        let hook = replacement
            .manifest
            .as_mut()
            .unwrap()
            .openai_exchange_hook
            .as_mut()
            .unwrap();
        match reduction {
            "endpoint" => hook.endpoints.clear(),
            "phase" => hook.phases.clear(),
            "body" => hook.request_body = false,
            "admission" => hook.admission = !hook.admission,
            _ => unreachable!(),
        }
        plugin
            .publish_initialized_state(8, &replacement, ServerConfig::default())
            .await
            .unwrap();
        let (callback, body) = tokio::join!(
            plugin.invoke_exchange_service(
                expected,
                "{\"body\":\"private\"}",
                Instant::now() + Duration::from_secs(1)
            ),
            plugin.open_lifecycle_stream(body_request(), expected),
        );
        assert!(callback.is_err(), "{reduction}");
        assert!(body.is_err(), "{reduction}");
        assert!(
            matches!(outbound.try_recv(), Err(mpsc::error::TryRecvError::Empty)),
            "{reduction}"
        );
        plugin.shutdown().await;
    }
}

#[tokio::test]
async fn verified_handles_remain_bound_to_their_generation_after_replacement() {
    let plugin = observer();
    let mut old_outbound = install_replacement(&plugin, true).await;
    let init = initialization("observe", true);
    plugin
        .publish_initialized_state(8, &init, ServerConfig::default())
        .await
        .unwrap();
    let declaration = init
        .manifest
        .as_ref()
        .unwrap()
        .openai_exchange_hook
        .as_deref()
        .unwrap();
    let (sender, _) = plugin
        .lifecycle_runtime_handles(LifecycleOperation::Callback(declaration))
        .await
        .unwrap();
    plugin
        .runtime
        .lock()
        .await
        .as_ref()
        .unwrap()
        .connection_task
        .abort();
    let mut replacement_outbound = install_replacement(&plugin, false).await;
    sender.send(proto::Envelope::default()).await.unwrap();
    assert!(old_outbound.try_recv().is_ok());
    assert!(matches!(
        replacement_outbound.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    assert!(
        plugin
            .lifecycle_runtime_handles(LifecycleOperation::Callback(declaration))
            .await
            .is_err()
    );
    plugin.shutdown().await;
}
