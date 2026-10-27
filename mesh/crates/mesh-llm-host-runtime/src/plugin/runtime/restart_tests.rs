//! A plugin whose process goes away by itself is started again.

use super::*;
use crate::plugin::{InProcessPluginRunner, InProcessPlugins, PluginManager, config};
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

/// A running `flaky` plugin from [`crashes_once`], with no request in flight.
struct Running {
    manager: PluginManager,
    starts: Arc<AtomicUsize>,
    crash: Arc<Notify>,
}

impl Running {
    async fn start() -> Self {
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

        // The supervisor's first round runs as the manager starts. Once its
        // health check has reached the plugin, a round run here returns only
        // after that check's answer was read too: no request is in flight when
        // the plugin dies, so nothing retries it and only a supervisor round
        // can restart it.
        tokio::time::timeout(Duration::from_secs(30), async {
            while health_checks.load(Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the supervisor's first round checks the plugin");
        manager.inner.plugins["flaky"].supervise().await.unwrap();
        Self {
            manager,
            starts,
            crash,
        }
    }

    fn plugin(&self) -> &ExternalPlugin {
        &self.manager.inner.plugins["flaky"]
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_plugin_that_goes_away_by_itself_is_started_again() {
    let running = Running::start().await;
    let plugin = running.plugin();
    running.crash.notify_one();

    // On a slow runner the supervisor's next round may already have started
    // it again; it skips a stopped plugin, so that also means it was not.
    let gone = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let summary = plugin.summary().await;
            if summary.status != "running" || running.starts.load(Ordering::SeqCst) > 1 {
                return summary;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the plugin's exit is noticed");
    if running.starts.load(Ordering::SeqCst) == 1 {
        assert_eq!(gone.status, "restarting", "not stopped for good");
        assert_eq!(gone.pid, None);
    }

    // A supervisor round starts it again if that has not happened yet.
    plugin.supervise().await.unwrap();
    assert_eq!(running.starts.load(Ordering::SeqCst), 2);
    assert_eq!(plugin.summary().await.status, "running");

    running.manager.shutdown().await;
    assert_eq!(
        plugin.summary().await.status,
        "stopped",
        "a shutdown still stops it for good"
    );
}

/// Lets the plugin go away while `status` is written, as by a shutdown or a
/// disable that runs after the connection loop took the runtime and before
/// its teardown settles the plugin's status: they find no runtime to stop and
/// leave that status. Returns once the teardown has ended.
async fn go_away_while_settled_as(running: &Running, status: &str) {
    let plugin = running.plugin();

    // The connection loop's teardown fails the requests still waiting on the
    // connection once it has settled the plugin's status.
    let (settled_tx, settled_rx) = oneshot::channel();
    plugin
        .runtime
        .lock()
        .await
        .as_ref()
        .expect("the plugin runs")
        .pending
        .lock()
        .await
        .insert(u64::MAX, settled_tx);

    // Holding the summary lets the connection loop take the runtime and then
    // makes it wait.
    let mut summary = plugin.summary.lock().await;
    running.crash.notify_one();
    tokio::time::timeout(Duration::from_secs(30), async {
        while plugin.runtime.lock().await.is_some() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the connection loop takes the runtime");
    summary.status = status.into();
    summary.pid = None;
    drop(summary);

    tokio::time::timeout(Duration::from_secs(30), settled_rx)
        .await
        .expect("the connection loop's teardown ends")
        .expect("the teardown answers the waiting request")
        .expect_err("with the disconnection");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_plugin_disabled_as_it_goes_away_stays_down() {
    let running = Running::start().await;
    go_away_while_settled_as(&running, "disabled").await;
    let plugin = running.plugin();
    assert_eq!(plugin.summary().await.status, "disabled");
    plugin.supervise().await.unwrap();
    assert_eq!(
        running.starts.load(Ordering::SeqCst),
        1,
        "a disabled plugin stays down"
    );
    running.manager.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_plugin_shut_down_as_it_goes_away_stays_stopped() {
    let running = Running::start().await;
    go_away_while_settled_as(&running, "stopped").await;
    assert_eq!(running.plugin().summary().await.status, "stopped");
    running.manager.shutdown().await;
}
