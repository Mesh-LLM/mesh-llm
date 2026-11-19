use super::status::publication_state_from_update;
use super::{RuntimeOptions, nostr_relays};
use crate::inference::skippy;
use crate::network::lan_bootstrap::LanBootstrapTasks;
use crate::network::{discovery as mesh_discovery, nostr};
use crate::runtime::survey;
use crate::{api, mesh, plugin};
use mesh_llm_events::{OutputEvent, emit_event};

pub(super) struct AutoRuntimeNodeSetup {
    pub(super) is_client: bool,
    pub(super) console_port: Option<u16>,
    pub(super) skippy_telemetry: skippy::SkippyTelemetryOptions,
    pub(super) local_models: Vec<String>,
    pub(super) node: mesh::Node,
    pub(super) channels: mesh::TunnelChannels,
    pub(super) plugin_manager: plugin::PluginManager,
    pub(super) survey_telemetry: survey::SurveyTelemetry,
    pub(super) lan_bootstrap_tasks: LanBootstrapTasks,
}

pub(super) fn bridge_publication_state(
    console_state: api::MeshApi,
    mut status_rx: tokio::sync::watch::Receiver<Option<nostr::PublishStateUpdate>>,
) {
    tokio::spawn(async move {
        let mut pending = *status_rx.borrow_and_update();
        loop {
            if let Some(update) = pending.take() {
                console_state
                    .set_publication_state(publication_state_from_update(update))
                    .await;
            }

            if status_rx.changed().await.is_err() {
                break;
            }
            pending = *status_rx.borrow_and_update();
        }
    });
}

pub(super) async fn unpublish_run_auto_nostr_listing(options: &RuntimeOptions) {
    if !options.publish || options.mesh_discovery_mode != mesh_discovery::MeshDiscoveryMode::Nostr {
        return;
    }
    let Ok(keys) = nostr::load_or_create_keys() else {
        return;
    };
    let relays = nostr_relays(&options.nostr_relay);
    let Ok(publisher) = nostr::Publisher::new(keys, &relays).await else {
        return;
    };
    let _ = publisher.unpublish().await;
    let _ = emit_event(OutputEvent::Info {
        message: "Removed Nostr listing".to_string(),
        context: None,
    });
}

pub(super) async fn shutdown_run_auto_services(
    node: &mesh::Node,
    plugin_manager: &plugin::PluginManager,
    api_proxy_handle: tokio::task::JoinHandle<()>,
    console_server_handle: Option<tokio::task::JoinHandle<()>>,
) {
    node.shutdown_control_listener().await;
    plugin_manager.shutdown().await;
    // close_endpoint releases the stopped registry after work transports close.
    api_proxy_handle.abort();
    let _ = api_proxy_handle.await;
    if let Some(handle) = console_server_handle {
        handle.abort();
        let _ = handle.await;
    }
}

pub(super) fn cleanup_run_auto_runtime_dir(
    runtime: Option<std::sync::Arc<crate::runtime::instance::InstanceRuntime>>,
) {
    let Some(rt) = runtime else {
        return;
    };
    let outstanding_refs = std::sync::Arc::strong_count(&rt);
    if outstanding_refs == 1 {
        let dir = rt.dir().to_path_buf();
        drop(rt);
        let _ = std::fs::remove_dir_all(&dir);
    } else {
        tracing::warn!(
            outstanding_refs,
            "skipping runtime directory removal during shutdown because runtime references remain"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::runtime_admission::RuntimeProviderDependencies;
    use crate::plugin::{BridgeFuture, PluginManager, PluginRpcBridge, RpcResult, proto};
    use crate::runtime::{InstanceLifecycleRecord, InstanceLifecycleState};
    use std::sync::Arc;

    #[tokio::test]
    async fn peer_evidence_shutdown_keeps_rejected_work_blocked_until_endpoint_closes()
    -> anyhow::Result<()> {
        struct Reject;
        impl PluginRpcBridge for Reject {
            fn handle_request(
                &self,
                _: String,
                _: String,
                _: String,
            ) -> BridgeFuture<anyhow::Result<RpcResult, proto::ErrorResponse>> {
                Box::pin(async {
                    Ok(RpcResult {
                        result_json:
                            serde_json::json!({"accepted":false,"cache_until_restart":true})
                                .to_string(),
                    })
                })
            }
            fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
                Box::pin(async {})
            }
        }
        let provider = mesh::Node::new_for_tests(mesh::NodeRole::Host { http_port: 9337 }).await?;
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Client).await?;
        let mut listeners = Vec::new();
        for endpoint in [&provider, &node] {
            endpoint.set_mesh_id("shutdown-evidence".into()).await;
            let endpoint = endpoint.clone();
            listeners.push(tokio::spawn(async move { endpoint.accept_loop().await }));
        }
        provider.start_accepting();
        node.start_accepting();
        node.connect_to_peer(provider.endpoint_addr_for_advertisement())
            .await?;
        let manager = PluginManager::for_test_peer_evidence(Arc::new(Reject), None);
        *node.plugin_manager.lock().await = Some(manager.clone());
        manager
            .verify_provider_evidence(
                "shutdown-evidence",
                &provider.id().to_string(),
                &Default::default(),
            )
            .await;
        let lifecycle = Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
            InstanceLifecycleState::Serving,
            10,
        )));
        let dependencies = Arc::new(RuntimeProviderDependencies::new(
            node.clone(),
            vec![provider.id()],
        ));
        node.register_runtime_instance_admission(9, lifecycle, Some(&dependencies));
        assert!(node.begin_runtime_instance_request(9).await.is_err());
        shutdown_run_auto_services(&node, &manager, tokio::spawn(std::future::pending()), None)
            .await;
        assert!(node.plugin_manager().await.is_some());
        assert!(node.begin_runtime_instance_request(9).await.is_err());
        node.close_endpoint().await;
        assert!(node.plugin_manager().await.is_none());
        provider.close_endpoint().await;
        for listener in listeners {
            listener.await?;
        }
        Ok(())
    }
}
