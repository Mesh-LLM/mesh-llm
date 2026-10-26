//! A plugin whose process goes away by itself is started again.

use super::*;
use crate::plugin::{InProcessPluginRunner, InProcessPlugins, config};
use mesh_llm_plugin::{PluginMetadata, PluginRuntime, SimplePlugin};
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

/// A plugin that exits right after its first initialization, as a crash
/// would, and keeps running when it is started again.
fn exits_once(starts: Arc<AtomicUsize>) -> InProcessPluginRunner {
    Arc::new(move |stream| {
        let start = starts.fetch_add(1, Ordering::SeqCst) + 1;
        let plugin = SimplePlugin::new(PluginMetadata::new(
            "flaky",
            "1.0.0",
            mesh_llm_plugin::plugin_server_info(
                "flaky",
                "1.0.0",
                "Flaky",
                "Exits once",
                None::<String>,
            ),
        ))
        .on_initialized(move |_context| {
            Box::pin(async move {
                if start == 1 {
                    // After the host has finished initializing it.
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    anyhow::bail!("gone");
                }
                Ok(())
            })
        });
        Box::pin(PluginRuntime::run_with_stream(plugin, stream))
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_plugin_that_goes_away_by_itself_is_started_again() {
    let starts = Arc::new(AtomicUsize::new(0));
    let mut spec = config::in_process_builtin_spec("flaky");
    spec.startup.optional = false;
    let (mesh_tx, _mesh_rx) = mpsc::channel(8);
    let manager = PluginManager::start_with_in_process(
        &config::ResolvedPlugins {
            externals: vec![spec],
            inactive: Vec::new(),
        },
        config::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
        InProcessPlugins::default().with("flaky", exits_once(starts.clone())),
    )
    .await
    .unwrap();
    let plugin = &manager.inner.plugins["flaky"];

    let gone = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let summary = plugin.summary().await;
            if summary.status != "running" {
                return summary;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the plugin's exit is noticed");
    assert_eq!(gone.status, "restarting", "not stopped for good");
    assert_eq!(gone.pid, None);

    // The supervisor's next round starts it again.
    plugin.supervise().await.unwrap();
    assert_eq!(starts.load(Ordering::SeqCst), 2);
    assert_eq!(plugin.summary().await.status, "running");

    manager.shutdown().await;
    assert_eq!(
        plugin.summary().await.status,
        "stopped",
        "a shutdown still stops it for good"
    );
}
