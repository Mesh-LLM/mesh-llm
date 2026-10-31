//! Status contributions over the real plugin protocol: from an SDK plugin
//! publishing in `on_initialized` to the node's `/api/status` data.

use super::*;
use crate::api::status::PluginStatusState;
use crate::plugin::{InProcessPluginRunner, InProcessPlugins, config};
use mesh_llm_plugin::{PluginMetadata, PluginRuntime, SimplePlugin};
use std::time::Duration;

type PublishResults = mpsc::Sender<(String, std::result::Result<(), String>)>;

/// A plugin that publishes `{"from": <name>}` as soon as it is initialized and
/// reports how that went.
fn publisher(name: &'static str, declares: bool, results: PublishResults) -> InProcessPluginRunner {
    Arc::new(move |stream| {
        let mut manifest = mesh_llm_plugin::plugin_manifest();
        if declares {
            manifest = manifest.item(mesh_llm_plugin::plugin_status::plugin_status(1));
        }
        let results = results.clone();
        let plugin = SimplePlugin::new(
            PluginMetadata::new(
                name,
                "1.0.0",
                mesh_llm_plugin::plugin_server_info(
                    name,
                    "1.0.0",
                    "Publisher",
                    "Publishes its status",
                    None::<String>,
                ),
            )
            .with_manifest(manifest.build()),
        )
        .on_initialized(move |context| {
            let results = results.clone();
            Box::pin(async move {
                let published = context
                    .publish_status(&serde_json::json!({"from": name}), Duration::from_secs(30))
                    .await
                    .map(|_| ())
                    .map_err(|error| error.to_string());
                let _ = results.send((name.to_string(), published)).await;
                Ok(())
            })
        });
        Box::pin(PluginRuntime::run_with_stream(plugin, stream))
    })
}

async fn start(
    plugins: &[(&'static str, bool)],
) -> (
    PluginManager,
    crate::mesh::Node,
    mpsc::Receiver<(String, std::result::Result<(), String>)>,
) {
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    let (mesh_tx, mesh_rx) = mpsc::channel(16);
    node.start_plugin_channel_forwarder(mesh_rx);
    let (results_tx, results_rx) = mpsc::channel(8);
    let mut in_process = InProcessPlugins::default();
    let mut externals = Vec::new();
    for &(name, declares) in plugins {
        in_process = in_process.with(name, publisher(name, declares, results_tx.clone()));
        let mut spec = config::in_process_builtin_spec(name);
        spec.startup.optional = false;
        externals.push(spec);
    }
    let manager = PluginManager::start_with_in_process(
        &config::ResolvedPlugins {
            externals,
            inactive: Vec::new(),
        },
        config::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
        in_process,
    )
    .await
    .unwrap();
    (manager, node, results_rx)
}

async fn results(
    rx: &mut mpsc::Receiver<(String, std::result::Result<(), String>)>,
    count: usize,
) -> BTreeMap<String, std::result::Result<(), String>> {
    let mut out = BTreeMap::new();
    while out.len() < count {
        let (name, result) = tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("every plugin reports its publication")
            .unwrap();
        out.insert(name, result);
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plugins_publishing_on_initialization_show_up_in_the_node_status() {
    let (manager, node, mut results_rx) =
        start(&[("alpha", true), ("beta", true), ("gamma", false)]).await;

    let results = results(&mut results_rx, 3).await;
    assert_eq!(results["alpha"], Ok(()));
    assert_eq!(results["beta"], Ok(()));
    let refused = results["gamma"].as_ref().unwrap_err();
    assert!(
        refused.contains("does not declare `plugin_status`"),
        "{refused}"
    );

    let payloads = node.runtime_data_collector().plugin_status_payloads();
    assert_eq!(
        payloads.keys().collect::<Vec<_>>(),
        ["alpha", "beta"],
        "a plugin that does not declare it contributes nothing"
    );
    for name in ["alpha", "beta"] {
        assert_eq!(payloads[name].state, PluginStatusState::Ready);
        assert_eq!(payloads[name].data, serde_json::json!({"from": name}));
    }

    // Shutting the plugins down removes what they contributed.
    manager.shutdown().await;
    tokio::time::timeout(Duration::from_secs(5), async {
        while !node
            .runtime_data_collector()
            .plugin_status_payloads()
            .is_empty()
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("stopped plugins no longer contribute");
}
