use super::status::publication_state_from_update;
use super::{RuntimeOptions, check_unserved_model, emit_shutdown, nostr_relays};
use crate::inference::skippy;
use crate::network::lan_bootstrap::LanBootstrapTasks;
use crate::network::{discovery as mesh_discovery, nostr};
use crate::runtime::survey;
use crate::{api, mesh, models, plugin};
use mesh_llm_events::{OutputEvent, emit_event, flush_output};

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

#[derive(Default)]
pub(super) struct PassivePublicationSetup {
    pub(super) state: Option<api::PublicationState>,
    pub(super) status_rx: Option<tokio::sync::watch::Receiver<Option<nostr::PublishStateUpdate>>>,
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

pub(super) fn maybe_spawn_passive_promotion_task(
    is_client: bool,
    node: &mesh::Node,
    promote_tx: tokio::sync::mpsc::Sender<String>,
) {
    if is_client {
        return;
    }

    let watch_node = node.clone();
    let mut peer_rx = node.peer_change_rx.clone();
    let local_models = models::scan_local_models();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        let mut demand_interval = tokio::time::interval(std::time::Duration::from_secs(60));
        demand_interval.tick().await;
        loop {
            tokio::select! {
                res = peer_rx.changed() => {
                    if res.is_err() { break; }
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    while peer_rx.has_changed().unwrap_or(false) {
                        let _ = peer_rx.borrow_and_update();
                    }
                }
                _ = demand_interval.tick() => {}
            }
            if let Some(model_name) = check_unserved_model(&watch_node, &local_models).await {
                let _ = emit_event(OutputEvent::HostElected {
                    model: model_name.clone(),
                    host: watch_node.id().fmt_short().to_string(),
                    role: Some("host".to_string()),
                    capacity_gb: Some(watch_node.vram_bytes() as f64 / 1e9),
                });
                let _ = promote_tx.send(model_name).await;
                break;
            }
        }
    });
}

pub(super) async fn setup_passive_publication(
    options: &RuntimeOptions,
    node: &mesh::Node,
    is_client: bool,
) -> PassivePublicationSetup {
    let mut setup = PassivePublicationSetup::default();
    if options.publish && !is_client {
        let pub_node = node.clone();
        match options.mesh_discovery_mode {
            mesh_discovery::MeshDiscoveryMode::Nostr => match nostr::load_or_create_keys() {
                Ok(nostr_keys) => {
                    let relays = nostr_relays(&options.nostr_relay);
                    let pub_name = options.mesh_name.clone();
                    let pub_region = options.region.clone();
                    let pub_max_clients = options.max_clients;
                    let (status_tx, status_rx) = tokio::sync::watch::channel(None);
                    setup.status_rx = Some(status_rx);
                    tokio::spawn(Box::pin(nostr::publish_loop(
                        pub_node,
                        nostr_keys,
                        nostr::PublishLoopConfig {
                            relays,
                            name: pub_name,
                            region: pub_region,
                            max_clients: pub_max_clients,
                            interval_secs: 60,
                            status_tx: Some(status_tx),
                        },
                    )));
                }
                Err(e) => {
                    let _ = emit_event(OutputEvent::Warning {
                        message: format!(
                            "Publishing to Nostr failed: {e}. Standby node is running privately — add --publish after fixing the issue to make discoverable."
                        ),
                        context: options
                            .mesh_name
                            .as_ref()
                            .map(|mesh_name| format!("mesh={mesh_name}")),
                    });
                    tracing::warn!("Passive Nostr publish failed: {e}");
                    setup.state = Some(api::PublicationState::PublishFailed);
                }
            },
            mesh_discovery::MeshDiscoveryMode::Mdns => {
                let pub_name = options.mesh_name.clone();
                let pub_region = options.region.clone();
                let pub_max_clients = options.max_clients;
                let pub_api_port = options.console;
                let pub_details_reachable = options.listen_all;
                let (status_tx, status_rx) = tokio::sync::watch::channel(None);
                setup.status_rx = Some(status_rx);
                tokio::spawn(Box::pin(mesh_discovery::publish_lan_loop(
                    pub_node,
                    mesh_discovery::LanPublishConfig {
                        name: pub_name,
                        region: pub_region,
                        max_clients: pub_max_clients,
                        api_port: pub_api_port,
                        details_reachable: pub_details_reachable,
                        interval_secs: 60,
                        status_tx: Some(status_tx),
                    },
                )));
            }
        }
        return setup;
    }

    if options.mesh_discovery_mode == mesh_discovery::MeshDiscoveryMode::Nostr
        && (options.auto || options.discover.is_some())
        && !is_client
    {
        let relays = nostr_relays(&options.nostr_relay);
        let wd_node = node.clone();
        let wd_name = options.mesh_name.clone();
        let wd_region = options.region.clone();
        let (status_tx, status_rx) = tokio::sync::watch::channel(None);
        setup.status_rx = Some(status_rx);
        tokio::spawn(async move {
            nostr::publish_watchdog(wd_node, relays, wd_name, wd_region, 120, Some(status_tx))
                .await;
        });
    }

    setup
}

pub(super) async fn shutdown_passive_runtime(
    node: &mesh::Node,
    plugin_manager: &plugin::PluginManager,
    console_server_handle: &mut Option<tokio::task::JoinHandle<()>>,
    signal: &'static str,
) {
    let _ = emit_event(OutputEvent::ShutdownRequested { signal });
    let _ = flush_output().await;
    emit_shutdown(None).await;
    node.shutdown_control_listener().await;
    plugin_manager.shutdown().await;
    if let Some(handle) = console_server_handle.take() {
        handle.abort();
        let _ = handle.await;
    }
    node.broadcast_leaving().await;
}
