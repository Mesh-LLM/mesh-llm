/// A plugin process that exits and is started again by a request
/// (Mesh-LLM/capsules#48) gets the mesh snapshot, this node's id among it,
/// before that request. A plugin that did not start again gets nothing new.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_plugin_started_again_by_a_request_gets_the_mesh_snapshot_before_it() {
    use crate::plugin::proto::{self, mesh_event::Kind};
    use mesh_llm_plugin::{read_envelope, write_envelope};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    // (plugin, process, what it received), in the order each process got it.
    type Seen = Arc<Mutex<Vec<(&'static str, usize, String)>>>;

    async fn reply(
        stream: &mut mesh_llm_plugin::LocalStream,
        incoming: &proto::Envelope,
        payload: proto::envelope::Payload,
    ) -> anyhow::Result<()> {
        write_envelope(
            stream,
            &proto::Envelope {
                protocol_version: crate::plugin::PROTOCOL_VERSION,
                plugin_id: incoming.plugin_id.clone(),
                request_id: incoming.request_id,
                payload: Some(payload),
            },
        )
        .await?;
        Ok(())
    }

    // Answers `initialize` with a manifest subscribing to the snapshot's
    // kinds, records each mesh event and operation, answers operations, and
    // exits after answering `exit`.
    fn recording(name: &'static str, seen: Seen) -> crate::plugin::InProcessPluginRunner {
        let processes = Arc::new(AtomicUsize::new(0));
        Arc::new(move |mut stream| {
            let seen = seen.clone();
            let process = processes.fetch_add(1, Ordering::SeqCst) + 1;
            Box::pin(async move {
                loop {
                    let incoming = read_envelope(&mut stream).await?;
                    match &incoming.payload {
                        Some(proto::envelope::Payload::InitializeRequest(_)) => {
                            let manifest = proto::PluginManifest {
                                mesh_event_subscriptions: [Kind::LocalStandby, Kind::PeerUp]
                                    .into_iter()
                                    .map(|kind| proto::MeshEventSubscriptionManifest {
                                        kind: kind as i32,
                                    })
                                    .collect(),
                                ..Default::default()
                            };
                            let init = proto::InitializeResponse {
                                plugin_id: name.into(),
                                plugin_protocol_version: crate::plugin::PROTOCOL_VERSION,
                                plugin_version: "0.0.0".into(),
                                server_info_json: serde_json::to_string(
                                    &rmcp::model::ServerConfig::default(),
                                )?,
                                manifest: Some(manifest),
                                ..Default::default()
                            };
                            let payload = proto::envelope::Payload::InitializeResponse(init);
                            reply(&mut stream, &incoming, payload).await?;
                        }
                        Some(proto::envelope::Payload::MeshEvent(event)) => {
                            let what = format!("event {} {}", event.kind, event.local_peer_id);
                            seen.lock().unwrap().push((name, process, what));
                        }
                        Some(proto::envelope::Payload::InvokeServiceRequest(request)) => {
                            let what = format!("call {}", request.service_name);
                            seen.lock().unwrap().push((name, process, what));
                            let response = proto::InvokeServiceResponse {
                                output_json: "{}".into(),
                                is_error: false,
                            };
                            let payload = proto::envelope::Payload::InvokeServiceResponse(response);
                            reply(&mut stream, &incoming, payload).await?;
                            if request.service_name == "exit" {
                                return Ok(());
                            }
                        }
                        _ => {}
                    }
                }
            })
        })
    }

    let a = make_test_node(super::NodeRole::Worker)
        .await
        .expect("test node must start");
    // Not accepting: `start_accepting` would race its own event in.
    a.insert_test_peer(make_test_peer_info(make_test_endpoint_id(42)))
        .await;

    let seen: Seen = Arc::default();
    let specs = crate::plugin::ResolvedPlugins {
        externals: vec![
            crate::plugin::in_process_builtin_spec("restarted"),
            crate::plugin::in_process_builtin_spec("steady"),
        ],
        inactive: Vec::new(),
    };
    let in_process = crate::plugin::InProcessPlugins::default()
        .with("restarted", recording("restarted", seen.clone()))
        .with("steady", recording("steady", seen.clone()));
    let (mesh_tx, mesh_rx) = tokio::sync::mpsc::channel(8);
    let manager = crate::plugin::PluginManager::start_with_in_process(
        &specs,
        crate::plugin::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
        in_process,
    )
    .await
    .expect("the plugins start");
    a.set_plugin_manager(manager.clone()).await;
    a.start_plugin_channel_forwarder(mesh_rx);

    let own_id = hex::encode(a.id().as_bytes());
    let snapshot = [
        format!("event {} {own_id}", Kind::LocalStandby as i32),
        format!("event {} {own_id}", Kind::PeerUp as i32),
    ];
    let of = |plugin: &'static str, process: usize| {
        seen.lock()
            .unwrap()
            .iter()
            .filter(|(name, number, _)| *name == plugin && *number == process)
            .map(|(_, _, what)| what.clone())
            .collect::<Vec<_>>()
    };
    let settle = || tokio::time::sleep(std::time::Duration::from_millis(200));

    // Installing the manager sends each plugin the snapshot once.
    settle().await;
    assert_eq!(of("restarted", 1), snapshot);
    assert_eq!(of("steady", 1), snapshot);

    // The process exits; the next request starts another one.
    manager
        .call_tool("restarted", "exit", "{}")
        .await
        .expect("the plugin answers before it exits");
    // Let the host see the stream close; a request sent before it does
    // fails its send and starts the plugin the same way.
    settle().await;
    manager
        .call_tool("restarted", "after", "{}")
        .await
        .expect("the request starts the plugin again");
    settle().await;

    let second = of("restarted", 2);
    assert_eq!(second.len(), 3, "{second:?}");
    assert_eq!(second[..2], snapshot, "the snapshot comes first");
    assert_eq!(second[2], "call after", "then the request that started it");
    assert!(of("restarted", 3).is_empty(), "one start, not a loop");
    assert_eq!(of("steady", 1), snapshot, "the other plugin got nothing new");
    manager.shutdown().await;
}
