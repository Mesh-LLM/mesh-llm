//! Terminal plugin shutdown is serialized with startup, never transport-replayed.

use super::{ExternalPlugin, Ordering, stop_runtime};
use anyhow::{Result, bail};

impl ExternalPlugin {
    pub(super) fn reject_shutdown(&self) -> Result<()> {
        if self.stopping.load(Ordering::Acquire) {
            bail!("Plugin '{}' is shutting down", self.spec.name);
        }
        Ok(())
    }

    pub(crate) async fn shutdown(&self) {
        // Publish terminal intent before waiting for a startup already in progress.
        self.stopping.store(true, Ordering::Release);
        let _guard = self.restart_lock.lock().await;
        self.authenticated_peer.store(false, Ordering::Release);
        {
            let mut summary = self.summary.lock().await;
            summary.status = "shutting down".into();
            summary.error = None;
        }

        let runtime = self.runtime.lock().await.take();
        if let Some(runtime) = runtime {
            stop_runtime(runtime, "plugin shutting down").await;
        }

        *self.server_info.lock().await = None;
        *self.manifest.lock().await = None;

        let mut summary = self.summary.lock().await;
        summary.status = "stopped".into();
        summary.pid = None;
        summary.version = None;
        summary.capabilities.clear();
        summary.tools.clear();
        summary.error = None;
        drop(summary);
        self.publish_summary().await;
        self.announce_stopped().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::FutureExt;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    #[tokio::test]
    async fn disconnected_status_is_not_terminal_shutdown() {
        let runner: crate::plugin::InProcessPluginRunner = Arc::new(|_| Box::pin(async { Ok(()) }));
        let plugin = crate::plugin::runtime::tests::in_process_plugin("crash-fixture", runner);
        plugin.summary.lock().await.status = "stopped".into();
        assert!(!plugin.is_stopping().await);
        plugin.shutdown().await;
        assert!(plugin.is_stopping().await);
    }

    #[tokio::test]
    async fn shutdown_rejects_startup_already_waiting_for_the_restart_lock() {
        let starts = Arc::new(AtomicUsize::new(0));
        let runner_starts = starts.clone();
        let runner = Arc::new(move |_stream| {
            runner_starts.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(()) }) as _
        });
        let plugin = super::super::tests::in_process_plugin("shutdown-fixture", runner);
        let startup_guard = plugin.restart_lock.lock().await;
        let startup = plugin.ensure_running();
        tokio::pin!(startup);
        assert!(startup.as_mut().now_or_never().is_none());
        let shutdown = plugin.shutdown();
        tokio::pin!(shutdown);
        assert!(shutdown.as_mut().now_or_never().is_none());
        assert!(plugin.stopping.load(Ordering::Acquire));
        drop(startup_guard);
        assert!(startup.await.is_err());
        shutdown.await;
        assert_eq!(starts.load(Ordering::SeqCst), 0);
        assert!(plugin.runtime.lock().await.is_none());
    }
}
