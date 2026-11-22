use super::*;
use crate::api::RuntimeUnloadResponse;
use crate::runtime::ModelIntent;
use crate::runtime::native_model_lifecycle::NativeModelLifecycle;
use mesh_llm_node::serving::UnloadTarget;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const NATIVE_MODEL: &str = "native-4b";
const PLUGIN_NAME: &str = "layerstream";
const PLUGIN_MODEL: &str = "plugin-14b";

async fn node_with_plugin_and_on_demand_native(
    exclusive_gpu: Option<bool>,
) -> (
    mesh::Node,
    tokio::sync::mpsc::Receiver<ModelIntent>,
    tokio::sync::watch::Sender<election::ModelTargets>,
    tempfile::TempDir,
) {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node should start");

    let exclusive_line = match exclusive_gpu {
        Some(true) => "exclusive_gpu = true\n",
        Some(false) => "exclusive_gpu = false\n",
        None => "",
    };
    let toml_cfg = format!(
        r#"
[[models]]
model = "{NATIVE_MODEL}"
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = false

[[plugin]]
name = "{PLUGIN_NAME}"
{exclusive_line}"#
    );
    let config_dir = tempfile::tempdir().expect("tempdir for test config");
    let config_path = config_dir.path().join("config.toml");
    std::fs::write(&config_path, &toml_cfg).expect("write test config");
    node.replace_config_state_for_test(&config_path)
        .await
        .expect("load test config state");
    let config = node.config_state.lock().await.config().clone();

    let (lifecycle, intents, targets_tx) = NativeModelLifecycle::test_service(&config);
    targets_tx.send_modify(|t| {
        t.targets.insert(
            NATIVE_MODEL.into(),
            vec![election::InferenceTarget::Local(4321)],
        );
    });
    lifecycle.record_loaded(NATIVE_MODEL, "runtime-1", NATIVE_MODEL, 4321);
    let _ = node.native_model_lifecycle.set(lifecycle);

    (node, intents, targets_tx, config_dir)
}

async fn run_plugin_dispatch(
    node: &mesh::Node,
    backend_addr: std::net::SocketAddr,
) -> (String, proxy::RouteDispatchOutcome) {
    let targets = election::ModelTargets::default();
    let affinity = affinity::AffinityRouter::new();
    let plugin_manager = tests::empty_test_plugin_manager();
    plugin_manager
        .set_test_inference_endpoints(vec![crate::plugin::InferenceEndpointRoute {
            plugin_name: PLUGIN_NAME.to_string(),
            endpoint_id: "layerstream".to_string(),
            protocol: Some("openai_compatible".to_string()),
            address: format!("http://{backend_addr}"),
            models: vec![PLUGIN_MODEL.to_string()],
        }])
        .await;

    let ctx = IngressRouteContext {
        node,
        targets: &targets,
        affinity: &affinity,
        plugin_manager: Some(&plugin_manager),
        requested_by_node_id: None,
        automatic_retry: None,
        #[cfg(test)]
        exchange_channel: None,
    };

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (client_res, server_res) = tokio::join!(TcpStream::connect(addr), listener.accept());
    let mut client = client_res.unwrap();
    let (server, _) = server_res.unwrap();
    let request = tests::plugin_only_request(PLUGIN_MODEL);

    let outcome = try_route_plugin_model(
        &ctx,
        server.into(),
        &request,
        PLUGIN_MODEL,
        OpenAiRouteObserver::default(),
    )
    .await;

    let mut response = String::new();
    client.read_to_string(&mut response).await.unwrap();
    (response, outcome)
}

fn assert_responded_status(outcome: proxy::RouteDispatchOutcome, expected: u16) {
    let actual = match outcome {
        proxy::RouteDispatchOutcome::Responded(status)
        | proxy::RouteDispatchOutcome::RespondedWithDigests {
            status_code: status,
            ..
        }
        | proxy::RouteDispatchOutcome::RespondedWithUsage {
            status_code: status,
            ..
        } => Some(status),
        _ => None,
    };
    assert_eq!(
        actual,
        Some(expected),
        "unexpected route dispatch outcome: {outcome:?}"
    );
}

#[tokio::test]
async fn exclusive_gpu_plugin_unloads_native_model_before_forwarding_request() {
    let (node, mut intents, targets_tx, _config_dir) =
        node_with_plugin_and_on_demand_native(Some(true)).await;

    let plugin_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let plugin_addr = plugin_listener.local_addr().unwrap();

    let dispatch_node = node.clone();
    let dispatch =
        tokio::spawn(async move { run_plugin_dispatch(&dispatch_node, plugin_addr).await });

    let ModelIntent::Unload {
        target,
        completion: Some(completion),
        ..
    } = tokio::time::timeout(std::time::Duration::from_secs(5), intents.recv())
        .await
        .expect("unload intent should arrive")
        .expect("intent channel open")
    else {
        panic!("expected unload intent");
    };
    assert_eq!(target, UnloadTarget::Instance("runtime-1".into()));

    // Keep unload completion unsent and confirm dispatch has not connected to
    // the plugin listener while the native model is still unloading.
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(50),
            plugin_listener.accept()
        )
        .await
        .is_err(),
        "plugin endpoint must not receive a connection before unload completes"
    );

    targets_tx.send_modify(|t| t.targets.clear());
    completion
        .send(Ok(RuntimeUnloadResponse {
            model: NATIVE_MODEL.into(),
            instance_id: "runtime-1".into(),
            unloaded: true,
        }))
        .unwrap();

    let (mut stream, _) =
        tokio::time::timeout(std::time::Duration::from_secs(5), plugin_listener.accept())
            .await
            .expect("dispatch should connect to plugin after unload completes")
            .unwrap();
    let _req = proxy::read_http_request(&mut stream).await.unwrap();
    stream
        .write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}",
        )
        .await
        .unwrap();

    let (response, outcome) = dispatch.await.unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert_responded_status(outcome, 200);
    node.endpoint.close().await;
}

#[tokio::test]
async fn default_plugin_leaves_on_demand_native_model_loaded() {
    let (node, mut intents, _targets_tx, _config_dir) =
        node_with_plugin_and_on_demand_native(None).await;

    let plugin_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let plugin_addr = plugin_listener.local_addr().unwrap();
    let plugin_task = tokio::spawn(async move {
        let (mut stream, _) = plugin_listener.accept().await.unwrap();
        let _req = proxy::read_http_request(&mut stream).await.unwrap();
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}",
            )
            .await
            .unwrap();
    });

    let (response, outcome) = run_plugin_dispatch(&node, plugin_addr).await;
    plugin_task.await.unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert_responded_status(outcome, 200);
    assert!(
        intents.try_recv().is_err(),
        "non-exclusive plugin must not unload native models"
    );
    node.endpoint.close().await;
}

#[tokio::test]
async fn exclusive_gpu_handoff_failure_returns_503_without_calling_plugin() {
    let (node, mut intents, _targets_tx, _config_dir) =
        node_with_plugin_and_on_demand_native(Some(true)).await;

    let plugin_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let plugin_addr = plugin_listener.local_addr().unwrap();
    let dispatch_node = node.clone();
    let dispatch =
        tokio::spawn(async move { run_plugin_dispatch(&dispatch_node, plugin_addr).await });

    let ModelIntent::Unload {
        completion: Some(completion),
        ..
    } = tokio::time::timeout(std::time::Duration::from_secs(5), intents.recv())
        .await
        .expect("unload intent should arrive")
        .expect("intent channel open")
    else {
        panic!("expected unload intent");
    };
    completion
        .send(Err(anyhow::anyhow!("maintained owner intent")))
        .unwrap();

    let (response, outcome) = dispatch.await.unwrap();
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    assert_responded_status(outcome, 503);
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(50),
            plugin_listener.accept()
        )
        .await
        .is_err(),
        "plugin endpoint must not be dialed when exclusive_gpu handoff fails"
    );
    node.endpoint.close().await;
}
