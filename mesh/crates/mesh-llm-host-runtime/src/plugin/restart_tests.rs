//! A plugin whose process goes away by itself is started again.

use super::*;
use crate::plugin::{InProcessPluginRunner, InProcessPlugins, config};
use mesh_llm_plugin::{LocalStream, PluginMetadata, PluginRuntime, SimplePlugin};
use std::sync::atomic::AtomicUsize;
use std::time::Duration;
use tokio::sync::Notify;

/// A plugin that counts its health checks. The first time it runs, its end of
/// the connection closes when `crash` is notified, as when its process dies;
/// when it is started again it keeps running.
fn crashes_once(
    starts: Arc<AtomicUsize>,
    health_checks: Arc<AtomicUsize>,
    crash: Arc<Notify>,
) -> InProcessPluginRunner {
    Arc::new(move |stream| {
        let start = starts.fetch_add(1, Ordering::SeqCst) + 1;
        let health_checks = health_checks.clone();
        let plugin = SimplePlugin::new(PluginMetadata::new(
            "flaky",
            "1.0.0",
            mesh_llm_plugin::plugin_server_info(
                "flaky",
                "1.0.0",
                "Flaky",
                "Crashes once",
                None::<String>,
            ),
        ))
        .with_health(move |_context| {
            health_checks.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok("ok".to_string()) })
        });
        if start > 1 {
            return Box::pin(PluginRuntime::run_with_stream(plugin, stream));
        }
        let crash = crash.clone();
        Box::pin(async move {
            let LocalStream::Memory(mut host_end) = stream else {
                anyhow::bail!("an in-process plugin gets an in-memory stream");
            };
            // The plugin talks to the host through a relay that the crash
            // cuts, so the host sees its connection close and nothing else.
            let (plugin_end, mut relay_end) = tokio::io::duplex(64 * 1024);
            let runtime = tokio::spawn(PluginRuntime::run_with_stream(
                plugin,
                LocalStream::Memory(plugin_end),
            ));
            tokio::select! {
                _ = tokio::io::copy_bidirectional(&mut host_end, &mut relay_end) => {}
                _ = crash.notified() => {}
            }
            drop(host_end);
            runtime.abort();
            anyhow::bail!("gone")
        })
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_plugin_that_goes_away_by_itself_is_started_again() {
    let starts = Arc::new(AtomicUsize::new(0));
    let health_checks = Arc::new(AtomicUsize::new(0));
    let crash = Arc::new(Notify::new());
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
        InProcessPlugins::default().with(
            "flaky",
            crashes_once(starts.clone(), health_checks.clone(), crash.clone()),
        ),
    )
    .await
    .unwrap();
    let plugin = &manager.inner.plugins["flaky"];

    // The supervisor's first round runs as the manager starts. Once its health
    // check has reached the plugin, a round run here returns only after that
    // check's answer was read too: no request is in flight when the plugin
    // dies, so nothing retries it and only a supervisor round can restart it.
    tokio::time::timeout(Duration::from_secs(30), async {
        while health_checks.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the supervisor's first round checks the plugin");
    plugin.supervise().await.unwrap();
    crash.notify_one();

    // On a slow runner the supervisor's next round may already have started
    // it again; it skips a stopped plugin, so that also means it was not.
    let gone = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let summary = plugin.summary().await;
            if summary.status != "running" || starts.load(Ordering::SeqCst) > 1 {
                return summary;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the plugin's exit is noticed");
    if starts.load(Ordering::SeqCst) == 1 {
        assert_eq!(gone.status, "restarting", "not stopped for good");
        assert_eq!(gone.pid, None);
    }

    // A supervisor round starts it again if that has not happened yet.
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
