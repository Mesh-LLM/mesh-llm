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
    gated_runner(Arc::new(tokio::sync::Semaphore::new(usize::MAX >> 4)))
}

/// [`initializing_runner`], answering each `initialize` only once `gate`
/// gives it a permit.
fn gated_runner(gate: Arc<tokio::sync::Semaphore>) -> crate::plugin::InProcessPluginRunner {
    Arc::new(move |mut stream| {
        let gate = gate.clone();
        Box::pin(async move {
            loop {
                let incoming = mesh_llm_plugin::read_envelope(&mut stream).await?;
                let Some(proto::envelope::Payload::InitializeRequest(_)) = incoming.payload else {
                    continue;
                };
                gate.acquire().await?.forget();
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
    spawn_with_runner(name, lazy_start, mesh_tx, initializing_runner()).await
}

async fn spawn_with_runner(
    name: &str,
    lazy_start: bool,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
    runner: crate::plugin::InProcessPluginRunner,
) -> ExternalPlugin {
    let runtime_data = RuntimeDataCollector::new();
    let plugin = ExternalPlugin::spawn(
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
        Some(runner),
    )
    .await
    .expect("the plugin starts");
    // As when a node takes the manager.
    plugin.expect_mesh_snapshots();
    plugin
}

async fn next_started(
    rx: &mut mpsc::Receiver<PluginMeshEvent>,
) -> (String, u64, oneshot::Sender<bool>) {
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
    delivered.send(true).expect("the start is waiting");
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
/// up, and does not let it serve without the snapshot: the start fails, so
/// the request can be retried, and the next one starts a new process that
/// asks again.
#[tokio::test]
async fn a_start_the_node_cannot_take_in_time_fails_and_the_next_one_starts() {
    let (mesh_tx, mut mesh_rx) = mpsc::channel(1);
    let plugin = spawn_in_process("eager", false, mesh_tx.clone()).await;
    let (first, _, _) = plugin.runtime_handles().await.expect("the process runs");
    plugin
        .handle_runtime_failure(None, "plugin exited".into())
        .await;
    // The failure announces the disconnect from a task; take it before
    // filling the queue, so the filler cannot wait behind it.
    let disconnected = tokio::time::timeout(std::time::Duration::from_secs(5), mesh_rx.recv())
        .await
        .expect("the disconnect is announced in time");
    assert!(matches!(
        disconnected,
        Some(PluginMeshEvent::PluginDisconnected { .. })
    ));
    mesh_tx
        .try_send(PluginMeshEvent::PluginStopped {
            plugin_id: "filler".into(),
        })
        .expect("the queue takes one event");

    tokio::time::timeout(std::time::Duration::from_secs(3), plugin.ensure_running())
        .await
        .expect("the start does not wait on a full queue past the bound")
        .expect_err("a start the node never heard of does not serve");
    assert!(plugin.runtime_handles().await.is_err(), "no process serves");

    assert!(matches!(
        mesh_rx.recv().await,
        Some(PluginMeshEvent::PluginStopped { .. })
    ));
    let (_, generation) = start_as_the_node_answers(&plugin, &mut mesh_rx).await;
    assert_ne!(generation, first);
    plugin.shutdown().await;
}

/// A start whose snapshot the node could not send whole, or did not send
/// in time, fails the same way.
#[tokio::test]
async fn a_start_without_the_whole_snapshot_fails() {
    let (mesh_tx, mut mesh_rx) = mpsc::channel(8);
    let plugin = spawn_in_process("eager", false, mesh_tx).await;
    for answer in [Some(false), None] {
        plugin
            .handle_runtime_failure(None, "plugin exited".into())
            .await;
        let mut start = Box::pin(plugin.ensure_running());
        let (_, _, delivered) = tokio::select! {
            started = &mut start => panic!("started before the node was told: {started:?}"),
            event = next_started(&mut mesh_rx) => event,
        };
        match answer {
            Some(sent) => delivered.send(sent).expect("the start is waiting"),
            None => drop(delivered),
        }
        tokio::time::timeout(std::time::Duration::from_secs(3), start)
            .await
            .expect("the start does not wait past the bound")
            .expect_err("a process without the snapshot does not serve");
        assert!(plugin.runtime_handles().await.is_err(), "no process serves");
    }
    start_as_the_node_answers(&plugin, &mut mesh_rx).await;
    plugin.shutdown().await;
}

/// Answers `initialize`, then stops reading while it keeps the stream open.
fn stalled_runner() -> crate::plugin::InProcessPluginRunner {
    Arc::new(move |mut stream| {
        Box::pin(async move {
            loop {
                let incoming = mesh_llm_plugin::read_envelope(&mut stream).await?;
                if let Some(proto::envelope::Payload::InitializeRequest(_)) = incoming.payload {
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
                                    server_info_json: serde_json::to_string(
                                        &ServerConfig::default(),
                                    )?,
                                    manifest: Some(proto::PluginManifest::default()),
                                    ..Default::default()
                                },
                            )),
                        },
                    )
                    .await?;
                    std::future::pending::<()>().await;
                }
            }
        })
    })
}

/// The node sends mesh events to every plugin from one task. A process
/// whose queue stays full does not hold that task: the send gives up after
/// its bound and fails the process, so the next request starts a new one.
#[tokio::test]
async fn a_process_that_stops_reading_mesh_events_is_failed() {
    let (mesh_tx, _mesh_rx) = mpsc::channel(8);
    let plugin = spawn_with_runner("eager", false, mesh_tx, stalled_runner()).await;
    let (generation, _, _) = plugin.runtime_handles().await.expect("the process runs");

    let filled = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            if let Err(error) = plugin
                .send_mesh_event_to_generation(generation, proto::MeshEvent::default())
                .await
            {
                return error;
            }
        }
    })
    .await
    .expect("a full queue fails the send within its bound");
    assert!(
        filled
            .to_string()
            .contains("did not take a mesh event in time"),
        "{filled:#}"
    );
    assert!(
        plugin.runtime_handles().await.is_err(),
        "the process is failed"
    );
    plugin.shutdown().await;
}

/// A start whose caller is cancelled after its process was installed, but
/// before the node sent it the mesh snapshot, leaves a process that never
/// got this node's id. The next request does not use it: it stops that
/// process and starts a whole one, which asks for the snapshot again.
#[tokio::test]
async fn a_start_cancelled_before_its_snapshot_is_not_served() {
    let (mesh_tx, mut mesh_rx) = mpsc::channel(8);
    let plugin = spawn_in_process("eager", false, mesh_tx).await;
    plugin
        .handle_runtime_failure(None, "plugin exited".into())
        .await;

    // Boxed, so dropping it cancels the start itself.
    let mut start = Box::pin(plugin.ensure_running());
    let (_, cancelled, _delivered) = tokio::select! {
        started = &mut start => panic!("started before the node was told: {started:?}"),
        event = next_started(&mut mesh_rx) => event,
    };
    drop(start);
    assert_eq!(plugin.runtime_started().await, Some(false));

    let (_, generation) = start_as_the_node_answers(&plugin, &mut mesh_rx).await;
    assert_ne!(
        generation, cancelled,
        "a new process, not the unannounced one"
    );
    assert_eq!(plugin.runtime_started().await, Some(true));
    plugin.shutdown().await;
}

/// The same for a start cancelled while its process was still initializing.
#[tokio::test]
async fn a_start_cancelled_during_initialize_is_not_served() {
    let (mesh_tx, mut mesh_rx) = mpsc::channel(8);
    let gate = Arc::new(tokio::sync::Semaphore::new(0));
    let plugin = spawn_with_runner("lazy", true, mesh_tx, gated_runner(gate.clone())).await;

    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(200),
            plugin.ensure_running()
        )
        .await
        .is_err(),
        "initialize is held back"
    );
    assert_eq!(plugin.runtime_started().await, Some(false));
    let (cancelled, _, _) = plugin.runtime_handles().await.expect("installed");

    gate.add_permits(2);
    let (_, generation) = start_as_the_node_answers(&plugin, &mut mesh_rx).await;
    assert_ne!(
        generation, cancelled,
        "a new process, not the half-started one"
    );
    assert!(started_events(&mut mesh_rx).is_empty(), "one announcement");
    plugin.shutdown().await;
}

/// Answers `initialize` with a manifest subscribing to `LocalStandby`, and
/// counts the mesh events it receives.
fn counting_runner(received: Arc<AtomicU64>) -> crate::plugin::InProcessPluginRunner {
    Arc::new(move |mut stream| {
        let received = received.clone();
        Box::pin(async move {
            loop {
                let incoming = mesh_llm_plugin::read_envelope(&mut stream).await?;
                match incoming.payload {
                    Some(proto::envelope::Payload::InitializeRequest(_)) => {
                        let manifest = proto::PluginManifest {
                            mesh_event_subscriptions: vec![proto::MeshEventSubscriptionManifest {
                                kind: proto::mesh_event::Kind::LocalStandby as i32,
                            }],
                            ..Default::default()
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
                                        server_info_json: serde_json::to_string(
                                            &ServerConfig::default(),
                                        )?,
                                        manifest: Some(manifest),
                                        ..Default::default()
                                    },
                                )),
                            },
                        )
                        .await?;
                    }
                    Some(proto::envelope::Payload::MeshEvent(_)) => {
                        received.fetch_add(1, Ordering::SeqCst);
                    }
                    _ => {}
                }
            }
        })
    })
}

/// The node sends the manager its mesh snapshot when it takes it, before it
/// reads the manager's channel. A plugin that the broadcast starts again
/// then cannot get its start answered, and fails it. That failure does not
/// keep the event from the plugins after it.
#[tokio::test]
async fn a_plugin_that_fails_to_take_a_broadcast_does_not_keep_it_from_the_rest() {
    let (mesh_tx, mut mesh_rx) = mpsc::channel(8);
    let steady = Arc::new(AtomicU64::new(0));
    let specs = crate::plugin::ResolvedPlugins {
        externals: vec![
            crate::plugin::in_process_builtin_spec("failing"),
            crate::plugin::in_process_builtin_spec("steady"),
        ],
        inactive: Vec::new(),
    };
    let in_process = crate::plugin::InProcessPlugins::default()
        .with("failing", counting_runner(Arc::default()))
        .with("steady", counting_runner(steady.clone()));
    let manager = crate::plugin::PluginManager::start_with_in_process(
        &specs,
        test_host_mode(),
        mesh_tx,
        in_process,
    )
    .await
    .expect("the plugins start");
    manager.expect_mesh_snapshots();

    // Leave "failing" with a process whose start was cancelled before its
    // snapshot, so the next event sent to it starts it again. "failing"
    // comes first in the broadcast.
    let failing = &manager.inner.plugins["failing"];
    failing
        .handle_runtime_failure(None, "plugin exited".into())
        .await;
    let mut start = Box::pin(failing.ensure_running());
    let (_, _, _delivered) = tokio::select! {
        started = &mut start => panic!("started before the node was told: {started:?}"),
        event = next_started(&mut mesh_rx) => event,
    };
    drop(start);
    assert_eq!(failing.runtime_started().await, Some(false));

    let error = manager
        .broadcast_mesh_event(proto::MeshEvent {
            kind: proto::mesh_event::Kind::LocalStandby as i32,
            ..Default::default()
        })
        .await
        .expect_err("no node answers the restart");
    assert!(
        error.to_string().contains("did not get the mesh snapshot"),
        "{error:#}"
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while steady.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the plugin after the failed one gets the event");
    manager.shutdown().await;
}
