//! The local inference observation feed over the real plugin protocol: SDK
//! plugins subscribe from `on_initialized`, and one granted by the operator
//! follows an execution reported by a local frontend's observer.

use super::*;
use crate::plugin::{InProcessPluginRunner, InProcessPlugins, config};
use mesh_llm_plugin::inference_observation::{InferenceObservationItem, inference_observation};
use mesh_llm_plugin::{PluginMetadata, PluginRuntime, SimplePlugin};
use proto::inference_observation_event::Kind;
use skippy_inference_api::{
    InferenceBackendOperation, InferenceFrontendRoute, InferenceLifecycleContext,
    InferenceLifecycleEvent, InferenceLifecycleObserver, InferenceRequestMethod,
    InferenceTerminalResult, generate_request_id,
};
use std::time::Duration;

type Subscribed = mpsc::Sender<(String, std::result::Result<(), String>)>;

/// Subscribes as soon as it is initialized, reports how that went, and hands
/// every batch it then reads to `batches`.
fn watcher(
    name: &'static str,
    declares: bool,
    subscribed: Subscribed,
    batches: mpsc::Sender<proto::InferenceObservationBatch>,
) -> InProcessPluginRunner {
    Arc::new(move |stream| {
        let mut manifest = mesh_llm_plugin::plugin_manifest();
        if declares {
            manifest = manifest.item(inference_observation());
        }
        let subscribed = subscribed.clone();
        let batches = batches.clone();
        let plugin = SimplePlugin::new(
            PluginMetadata::new(
                name,
                "1.0.0",
                mesh_llm_plugin::plugin_server_info(
                    name,
                    "1.0.0",
                    "Watcher",
                    "Follows local inference",
                    None::<String>,
                ),
            )
            .with_manifest(manifest.build()),
        )
        .on_initialized(move |context| {
            let subscribed = subscribed.clone();
            let batches = batches.clone();
            Box::pin(async move {
                match context.subscribe_inference_observation().await {
                    Ok(mut feed) => {
                        let _ = subscribed.send((name.to_string(), Ok(()))).await;
                        tokio::spawn(async move {
                            while let Some(InferenceObservationItem::Batch(batch)) =
                                feed.next().await
                            {
                                if batches.send(batch).await.is_err() {
                                    break;
                                }
                            }
                        });
                    }
                    Err(error) => {
                        let _ = subscribed
                            .send((name.to_string(), Err(error.to_string())))
                            .await;
                    }
                }
                Ok(())
            })
        });
        Box::pin(PluginRuntime::run_with_stream(plugin, stream))
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn only_a_declared_and_granted_plugin_follows_local_inference() {
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    let config_dir = tempfile::tempdir().unwrap();
    let config_path = config_dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        "version = 1\n\n[[plugin]]\nname = \"watcher\"\nallow_inference_observation = true\n\n\
         [[plugin]]\nname = \"plain\"\nallow_inference_observation = true\n",
    )
    .unwrap();
    node.replace_config_state_for_test(&config_path)
        .await
        .unwrap();
    let (mesh_tx, mesh_rx) = mpsc::channel(16);
    node.start_plugin_channel_forwarder(mesh_rx);

    let (subscribed_tx, mut subscribed_rx) = mpsc::channel(8);
    let (batches_tx, mut batches_rx) = mpsc::channel(256);
    let mut in_process = InProcessPlugins::default();
    let mut externals = Vec::new();
    // watcher: declares and granted. nosy: declares, not granted.
    // plain: granted, but does not declare it.
    for (name, declares) in [("watcher", true), ("nosy", true), ("plain", false)] {
        in_process = in_process.with(
            name,
            watcher(name, declares, subscribed_tx.clone(), batches_tx.clone()),
        );
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

    let mut results = BTreeMap::new();
    while results.len() < 3 {
        let (name, result) = tokio::time::timeout(Duration::from_secs(10), subscribed_rx.recv())
            .await
            .expect("every plugin reports its subscription")
            .unwrap();
        results.insert(name, result);
    }
    assert_eq!(results["watcher"], Ok(()));
    let nosy = results["nosy"].as_ref().unwrap_err();
    assert!(nosy.contains("allow_inference_observation"), "{nosy}");
    let plain = results["plain"].as_ref().unwrap_err();
    assert!(plain.contains("does not declare"), "{plain}");

    // A local frontend serving a model only this test uses runs one stream.
    let model = format!("observed-{}", generate_request_id().as_ref());
    let frontend = inference_observation::adapter::ObservationAdapter::new(
        inference_observation::hub(),
        model.clone(),
    );
    let request = generate_request_id();
    let context = InferenceLifecycleContext::new(
        request,
        InferenceRequestMethod::Post,
        InferenceFrontendRoute::ChatCompletions,
    );
    frontend.observe(&InferenceLifecycleEvent::BackendDispatched {
        context: context.clone(),
        operation: InferenceBackendOperation::ChatCompletionStream,
    });
    frontend.observe(&InferenceLifecycleEvent::StreamTerminal {
        context,
        result: InferenceTerminalResult::Completed { status_code: 200 },
    });

    let mut seen = Vec::new();
    let mut execution = None;
    tokio::time::timeout(Duration::from_secs(10), async {
        while let Some(batch) = batches_rx.recv().await {
            assert!(batch.watermark_unix_ns > 0);
            for event in batch.events {
                match event.kind.unwrap() {
                    Kind::ModelReady(ready) if ready == model => seen.push("model_ready"),
                    Kind::Started(started) if started.model == model => {
                        execution = Some(event.execution_id);
                        assert!(started.streaming);
                        seen.push("started");
                    }
                    Kind::ComputeFinished(_) if Some(event.execution_id) == execution => {
                        seen.push("compute_finished")
                    }
                    Kind::Ended(_) if Some(event.execution_id) == execution => {
                        seen.push("ended");
                        return;
                    }
                    _ => {}
                }
            }
        }
    })
    .await
    .expect("the granted plugin follows the execution to its end");
    assert_eq!(
        seen,
        ["model_ready", "started", "compute_finished", "ended"]
    );

    // Shutting the plugin down ends its subscription.
    manager.shutdown().await;
    drop(frontend);
}
