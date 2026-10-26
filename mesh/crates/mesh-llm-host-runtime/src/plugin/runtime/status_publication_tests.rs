use super::super::tests as runtime_tests;
use super::*;
use std::collections::HashMap;

fn runtime_with(generation: u64, declaration: StatusDeclaration) -> PluginRuntime {
    let (outbound_tx, _outbound_rx) = mpsc::channel(4);
    PluginRuntime {
        generation,
        authenticated_peer: false,
        initialized_lifecycle: None,
        status_declaration: declaration,
        _child: None,
        connection_task: tokio::spawn(std::future::pending()),
        outbound_tx,
        pending: Arc::new(Mutex::new(HashMap::new())),
    }
}

struct Harness {
    runtime: Arc<Mutex<Option<PluginRuntime>>>,
    lane: StatusLane,
    mesh_rx: mpsc::Receiver<PluginMeshEvent>,
    outbound_rx: mpsc::Receiver<proto::Envelope>,
}

fn harness(lane_generation: u64, runtime: Option<PluginRuntime>) -> Harness {
    let runtime = Arc::new(Mutex::new(runtime));
    let (mesh_tx, mesh_rx) = mpsc::channel(8);
    let (outbound_tx, outbound_rx) = mpsc::channel(8);
    let lane = StatusLane::spawn(
        "demo".into(),
        lane_generation,
        runtime.clone(),
        mesh_tx,
        outbound_tx,
    );
    Harness {
        runtime,
        lane,
        mesh_rx,
        outbound_rx,
    }
}

fn publication(data_json: &str) -> proto::PublishPluginStatusRequest {
    proto::PublishPluginStatusRequest {
        data_json: data_json.into(),
        fresh_for_ms: 0,
    }
}

async fn reply(outbound_rx: &mut mpsc::Receiver<proto::Envelope>) -> proto::Envelope {
    tokio::time::timeout(Duration::from_secs(5), outbound_rx.recv())
        .await
        .expect("the lane answers")
        .expect("the lane is open")
}

fn error_message(envelope: proto::Envelope) -> String {
    match envelope.payload {
        Some(proto::envelope::Payload::ErrorResponse(error)) => error.message,
        other => panic!("expected an error, got {other:?}"),
    }
}

#[tokio::test]
async fn a_publication_sent_before_initialization_finishes_waits_for_it() {
    let mut h = harness(3, Some(runtime_with(3, StatusDeclaration::Pending)));
    h.lane.push(11, publication(r#"{"watts":8.2}"#));

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        h.mesh_rx.try_recv().is_err(),
        "nothing before initialization"
    );
    h.runtime.lock().await.as_mut().unwrap().status_declaration =
        StatusDeclaration::Declared { schema_version: 2 };

    let event = tokio::time::timeout(Duration::from_secs(5), h.mesh_rx.recv())
        .await
        .unwrap()
        .unwrap();
    let PluginMeshEvent::PluginStatus {
        plugin_id,
        generation,
        schema_version,
        request,
        response_tx,
    } = event
    else {
        panic!("expected a status publication");
    };
    assert_eq!(plugin_id, "demo", "the host's name for the connection");
    assert_eq!((generation, schema_version), (3, 2));
    assert_eq!(request.data_json, r#"{"watts":8.2}"#);
    response_tx
        .send(Ok(proto::PublishPluginStatusResponse {
            received_at_unix_ms: 1,
            fresh_until_unix_ms: 2,
        }))
        .unwrap();

    let answer = reply(&mut h.outbound_rx).await;
    assert_eq!(answer.request_id, 11);
    assert_eq!(answer.plugin_id, "demo");
    assert!(matches!(
        answer.payload,
        Some(proto::envelope::Payload::PublishPluginStatusResponse(ref response))
            if response.fresh_until_unix_ms == 2
    ));
}

#[tokio::test]
async fn a_plugin_that_does_not_declare_it_is_refused() {
    let mut h = harness(3, Some(runtime_with(3, StatusDeclaration::Undeclared)));
    h.lane.push(1, publication("{}"));
    let message = error_message(reply(&mut h.outbound_rx).await);
    assert!(
        message.contains("does not declare `plugin_status`"),
        "{message}"
    );
    assert!(h.mesh_rx.try_recv().is_err());
}

#[tokio::test]
async fn a_connection_that_is_no_longer_current_is_refused() {
    let declared = StatusDeclaration::Declared { schema_version: 1 };
    // The plugin restarted: the runtime is a later connection's.
    let mut h = harness(3, Some(runtime_with(4, declared)));
    h.lane.push(1, publication("{}"));
    let message = error_message(reply(&mut h.outbound_rx).await);
    assert!(message.contains("no longer current"), "{message}");

    // The connection is gone, and the runtime with it.
    let mut h = harness(3, None);
    h.lane.push(1, publication("{}"));
    let message = error_message(reply(&mut h.outbound_rx).await);
    assert!(message.contains("no longer current"), "{message}");
    assert!(h.mesh_rx.try_recv().is_err());
}

#[tokio::test]
async fn publications_reach_the_node_in_the_order_they_were_sent() {
    let declared = StatusDeclaration::Declared { schema_version: 1 };
    let mut h = harness(3, Some(runtime_with(3, declared)));
    for (id, data) in [(1, r#"{"n":1}"#), (2, r#"{"n":2}"#), (3, r#"{"n":3}"#)] {
        h.lane.push(id, publication(data));
    }
    for expected in [r#"{"n":1}"#, r#"{"n":2}"#, r#"{"n":3}"#] {
        let Some(PluginMeshEvent::PluginStatus {
            request,
            response_tx,
            ..
        }) = h.mesh_rx.recv().await
        else {
            panic!("expected a status publication");
        };
        assert_eq!(request.data_json, expected);
        response_tx
            .send(Ok(proto::PublishPluginStatusResponse::default()))
            .unwrap();
    }
    let ids = [
        reply(&mut h.outbound_rx).await.request_id,
        reply(&mut h.outbound_rx).await.request_id,
        reply(&mut h.outbound_rx).await.request_id,
    ];
    assert_eq!(ids, [1, 2, 3]);
}

#[tokio::test]
async fn a_full_lane_answers_busy_at_once() {
    // The first publication waits for initialization, so the next ones queue.
    let mut h = harness(3, Some(runtime_with(3, StatusDeclaration::Pending)));
    h.lane.push(1, publication("{}"));
    tokio::time::sleep(Duration::from_millis(20)).await;
    for id in 2..=(LANE_CAPACITY as u64 + 1) {
        h.lane.push(id, publication("{}"));
    }
    h.lane.push(99, publication("{}"));
    let answer = reply(&mut h.outbound_rx).await;
    assert_eq!(answer.request_id, 99);
    let message = error_message(answer);
    assert!(message.contains("retry later"), "{message}");
}

#[tokio::test]
async fn a_failed_connection_tells_the_node_its_status_is_unavailable() {
    let temp = tempfile::tempdir().unwrap();
    let (mesh_tx, mut mesh_rx) = mpsc::channel(8);
    let (plugin, _runtime_data) = runtime_tests::plugin_for_spec_with_mesh_tx(
        runtime_tests::plugin_spec(
            &temp,
            None,
            mesh_llm_plugin_manager::store::InstalledPluginWebUiValidationStatus::Valid,
            None,
        ),
        mesh_tx,
    );
    runtime_tests::mark_plugin_running(&plugin, 4).await;

    plugin
        .handle_runtime_failure(Some(4), "plugin exited".into())
        .await;
    let mut disconnected = Vec::new();
    while let Ok(event) = mesh_rx.try_recv() {
        if let PluginMeshEvent::PluginDisconnected {
            plugin_id,
            generation,
        } = event
        {
            disconnected.push((plugin_id, generation));
        }
    }
    assert_eq!(disconnected, vec![("demo".to_string(), 4)]);

    // No runtime, nothing to announce.
    plugin
        .handle_runtime_failure(None, "plugin exited".into())
        .await;
    assert!(mesh_rx.try_recv().is_err());
}

#[test]
fn a_declaration_must_name_a_schema_version() {
    let manifest = |schema_version| proto::PluginManifest {
        plugin_status: Some(proto::PluginStatusManifest {
            schema_version,
            required: false,
        }),
        ..Default::default()
    };
    let error = validate_declaration("demo", &manifest(0)).unwrap_err();
    assert!(error.to_string().contains("schema_version 0"), "{error}");
    validate_declaration("demo", &manifest(1)).unwrap();
    validate_declaration("demo", &proto::PluginManifest::default()).unwrap();
    assert_eq!(
        StatusDeclaration::from_manifest(Some(&manifest(3))),
        StatusDeclaration::Declared { schema_version: 3 }
    );
    assert_eq!(
        StatusDeclaration::from_manifest(Some(&proto::PluginManifest::default())),
        StatusDeclaration::Undeclared
    );
}
