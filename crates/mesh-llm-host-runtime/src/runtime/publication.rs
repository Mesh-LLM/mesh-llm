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
    // Break the Node <-> in-process payments runner cycle so the engine and
    // its process lock are released before an embedded restart.
    drop(node.take_plugin_manager().await);
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
