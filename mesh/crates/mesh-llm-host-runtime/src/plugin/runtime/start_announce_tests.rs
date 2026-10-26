//! A plugin that starts after the host's own start asks the node for the mesh
//! snapshot (Mesh-LLM/capsules#48).

use super::tests::test_host_mode;
use super::*;
use crate::runtime_data::{RuntimeDataCollector, RuntimeDataSource};
use std::collections::BTreeMap;

fn started_events(rx: &mut mpsc::Receiver<PluginMeshEvent>) -> Vec<String> {
    let mut out = Vec::new();
    while let Ok(event) = rx.try_recv() {
        if let PluginMeshEvent::PluginStarted { plugin_id, .. } = event {
            out.push(plugin_id);
        }
    }
    out
}

/// Answers `initialize`, then reads until the host closes the stream.
fn initializing_runner() -> crate::plugin::InProcessPluginRunner {
    Arc::new(|mut stream| {
        Box::pin(async move {
            loop {
                let incoming = mesh_llm_plugin::read_envelope(&mut stream).await?;
                let Some(proto::envelope::Payload::InitializeRequest(_)) = incoming.payload else {
                    continue;
                };
                mesh_llm_plugin::write_envelope(
                    &mut stream,
                    &proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: incoming.plugin_id.clone(),
                        request_id: incoming.request_id,
                        payload: Some(proto::envelope::Payload::InitializeResponse(
                            proto::InitializeResponse {
                                plugin_id: incoming.plugin_id,
                                plugin_protocol_version: PROTOCOL_VERSION,
                                plugin_version: "0.0.0".into(),
                                server_info_json: serde_json::to_string(&ServerConfig::default())?,
                                manifest: Some(proto::PluginManifest::default()),
                                ..Default::default()
                            },
                        )),
                    },
                )
                .await?;
            }
        })
    })
}

async fn spawn_in_process(
    name: &str,
    lazy_start: bool,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
) -> ExternalPlugin {
    let runtime_data = RuntimeDataCollector::new();
    ExternalPlugin::spawn(
        &ExternalPluginSpec {
            openai_exchange_grant: None,
            name: name.into(),
            command: String::new(),
            args: Vec::new(),
            url: None,
            env: BTreeMap::new(),
            startup: crate::plugin::PluginStartupOptions {
                lazy_start,
                ..crate::plugin::PluginStartupOptions::default()
            },
            web_ui_enabled: None,
            web_ui_primary_tab: None,
            installed_metadata: None,
        },
        "test-instance".into(),
        test_host_mode(),
        mesh_tx,
        Arc::new(Mutex::new(None)),
        runtime_data.producer(RuntimeDataSource {
            scope: "test",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        }),
        Some(initializing_runner()),
    )
    .await
    .expect("the plugin starts")
}

async fn next_started(
    rx: &mut mpsc::Receiver<PluginMeshEvent>,
) -> (String, u64, oneshot::Sender<()>) {
    loop {
        let event = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("the node is told in time")
            .expect("the channel stays open");
        if let PluginMeshEvent::PluginStarted {
            plugin_id,
            generation,
            delivered,
        } = event
        {
            return (plugin_id, generation, delivered);
        }
    }
}

/// Starts `plugin` the way a request would, playing the node: the start
/// names the new process and waits until the node says the snapshot was
/// sent. Returns the plugin id and generation it named.
async fn start_as_the_node_answers(
    plugin: &ExternalPlugin,
    rx: &mut mpsc::Receiver<PluginMeshEvent>,
) -> (String, u64) {
    let start = plugin.ensure_running();
    tokio::pin!(start);
    let (plugin_id, generation, delivered) = tokio::select! {
        started = &mut start => panic!("started before the node was told: {started:?}"),
        event = next_started(rx) => event,
    };
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(200), &mut start)
            .await
            .is_err(),
        "the start waits for the snapshot"
    );
    let (current, _, _) = plugin.runtime_handles().await.expect("the process runs");
    assert_eq!(generation, current, "the event names the new process");
    // A second request meanwhile finds the process installed, and waits too.
    let second = plugin.ensure_running();
    tokio::pin!(second);
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(200), &mut second)
            .await
            .is_err(),
        "a concurrent request waits for the snapshot as well"
    );
    delivered.send(()).expect("the start is waiting");
    for (caller, started) in [
        ("the start", start.await),
        ("the second request", second.await),
    ] {
        started.unwrap_or_else(|error| panic!("{caller} fails: {error:#}"));
    }
    (plugin_id, generation)
}

/// The node sends every plugin its mesh snapshot (this node's id among
/// it) when the manager is installed, and otherwise only on a mesh
/// change. A plugin that starts after that (restarted after it exited or
/// failed, or started lazily) tells the node which process started and
/// waits for the snapshot before it takes requests, or it runs without
/// this node's id until a peer joins or leaves (Mesh-LLM/capsules#48). The
/// start inside `spawn` does not ask: the manager's own snapshot covers it.
#[tokio::test]
async fn a_plugin_that_starts_after_the_host_gets_the_mesh_snapshot_before_requests() {
    let (mesh_tx, mut mesh_rx) = mpsc::channel(8);
    let plugin = spawn_in_process("eager", false, mesh_tx.clone()).await;
    assert_eq!(plugin.summary().await.status, "running");
    assert!(started_events(&mut mesh_rx).is_empty(), "spawn's own start");

    plugin
        .handle_runtime_failure(None, "plugin exited".into())
        .await;
    let (plugin_id, _) = start_as_the_node_answers(&plugin, &mut mesh_rx).await;
    assert_eq!(plugin_id, "eager");
    plugin.shutdown().await;

    let lazy = spawn_in_process("lazy", true, mesh_tx).await;
    assert!(
        started_events(&mut mesh_rx).is_empty(),
        "deferred, not started"
    );
    let (plugin_id, _) = start_as_the_node_answers(&lazy, &mut mesh_rx).await;
    assert_eq!(plugin_id, "lazy");
    lazy.shutdown().await;
}

/// A snapshot goes to the process it was meant for, and never starts one:
/// sent to a process that has since exited, it is dropped, so a plugin
/// that exits right after starting is not started again by its own
/// snapshot.
#[tokio::test]
async fn a_snapshot_for_a_process_that_exited_starts_nothing() {
    let (mesh_tx, mut mesh_rx) = mpsc::channel(8);
    let plugin = spawn_in_process("eager", false, mesh_tx).await;
    let (first, _, _) = plugin.runtime_handles().await.expect("the process runs");
    plugin
        .handle_runtime_failure(None, "plugin exited".into())
        .await;
    let (_, second) = start_as_the_node_answers(&plugin, &mut mesh_rx).await;
    assert_ne!(first, second);

    let event = proto::MeshEvent::default();
    plugin
        .send_mesh_event_to_generation(first, event.clone())
        .await
        .expect("an earlier process's event is dropped");
    plugin
        .handle_runtime_failure(None, "plugin exited".into())
        .await;
    plugin
        .send_mesh_event_to_generation(second, event)
        .await
        .expect("an exited process's event is dropped");
    assert!(
        plugin.runtime_handles().await.is_err(),
        "the send did not start the plugin"
    );
    assert!(started_events(&mut mesh_rx).is_empty());
    plugin.shutdown().await;
}

/// A node whose queue stays full past the bound does not hold the start
/// up, and does not lose the event: it is queued as soon as there is room,
/// and still names the process that started.
#[tokio::test]
async fn a_start_event_the_node_cannot_take_yet_is_queued_later() {
    let (mesh_tx, mut mesh_rx) = mpsc::channel(1);
    let plugin = spawn_in_process("eager", false, mesh_tx.clone()).await;
    plugin
        .handle_runtime_failure(None, "plugin exited".into())
        .await;
    mesh_tx
        .send(PluginMeshEvent::PluginStopped {
            plugin_id: "filler".into(),
        })
        .await
        .expect("the queue takes one event");

    tokio::time::timeout(std::time::Duration::from_secs(3), plugin.ensure_running())
        .await
        .expect("the start does not wait on a full queue past the bound")
        .expect("the plugin starts");
    let (current, _, _) = plugin.runtime_handles().await.expect("the process runs");
    assert!(matches!(
        mesh_rx.recv().await,
        Some(PluginMeshEvent::PluginStopped { .. })
    ));
    let (plugin_id, generation, _) = next_started(&mut mesh_rx).await;
    assert_eq!((plugin_id.as_str(), generation), ("eager", current));
    plugin.shutdown().await;
}
