//! Per-instance shutdown, independent of the runtime command queue. The scope
//! belongs to the embedded runtime's orchestration future, not the process or
//! its OS signal handlers. CLI runtimes have no embedded cancellation source.
use super::EmbeddedServeHandle;
use anyhow::{Context, Result};
use std::time::{Duration, Instant};
use tokio::sync::watch;

const EMBEDDED_STARTUP_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

impl EmbeddedServeHandle {
    /// Request cancellation and wait until the runtime thread finishes cleanup.
    /// Dropping the handle also requests cancellation, but does not wait for it.
    pub async fn stop(mut self) -> Result<()> {
        self.request_shutdown("sdk");
        let task = self
            .task
            .take()
            .context("embedded mesh runtime thread handle is unavailable")?;
        join_embedded_runtime_thread(task).await
    }

    fn request_shutdown(&mut self, source: &'static str) {
        // Release the SDK's transport reference BEFORE allowing the worker to
        // exit. The worker's runtime then owns the final endpoint drop and
        // waits for netwatch's blocking OS socket close during runtime teardown.
        // Retaining this clone until after the join defers that close onto the
        // caller's runtime, letting stop return while the QUIC port is bound.
        drop(self.native_node.take());
        self.shutdown.send_replace(true);
        if let Some(tx) = self.control_tx.take() {
            let _ = tx.send(crate::api::RuntimeControlRequest::Shutdown { source });
        }
    }
}

impl Drop for EmbeddedServeHandle {
    fn drop(&mut self) {
        self.request_shutdown("sdk-drop");
    }
}

async fn join_embedded_runtime_thread(task: std::thread::JoinHandle<Result<()>>) -> Result<()> {
    tokio::task::spawn_blocking(move || join_embedded_runtime_thread_blocking(task))
        .await
        .context("join embedded mesh runtime thread")?
}

async fn join_embedded_runtime_thread_with_timeout(
    task: std::thread::JoinHandle<Result<()>>,
    timeout: Duration,
) -> Result<()> {
    tokio::task::spawn_blocking(move || {
        let deadline = Instant::now() + timeout;
        loop {
            if task.is_finished() {
                return join_embedded_runtime_thread_blocking(task);
            }
            if Instant::now() >= deadline {
                anyhow::bail!(
                    "timed out after {:?} waiting for embedded mesh runtime thread to exit",
                    timeout
                );
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    })
    .await
    .context("join embedded mesh runtime thread after startup failure")?
}

fn join_embedded_runtime_thread_blocking(task: std::thread::JoinHandle<Result<()>>) -> Result<()> {
    task.join()
        .map_err(|_| anyhow::anyhow!("embedded mesh runtime thread panicked"))?
}

/// Shut the embedded runtime down after a failed startup, joining it with a
/// bounded timeout, and return `error` annotated when the join itself failed.
pub(super) async fn shutdown_after_startup_error(
    handle: &mut EmbeddedServeHandle,
    error: anyhow::Error,
) -> anyhow::Error {
    handle.request_shutdown("sdk-startup-error");
    if let Err(shutdown_error) = join_embedded_runtime_thread_with_timeout(
        handle.task.take().expect("runtime task"),
        EMBEDDED_STARTUP_SHUTDOWN_TIMEOUT,
    )
    .await
    {
        return error.context(format!(
            "failed to shut down embedded mesh runtime after startup error: {shutdown_error}"
        ));
    }
    error
}

tokio::task_local! {
    pub(crate) static SHUTDOWN: watch::Receiver<bool>;
}

pub(crate) async fn cancelled() {
    let Ok(mut receiver) = SHUTDOWN.try_with(Clone::clone) else {
        return std::future::pending().await;
    };
    loop {
        if *receiver.borrow_and_update() || receiver.changed().await.is_err() {
            return;
        }
    }
}

/// The embedded instance was stopped before it finished starting.
#[derive(Debug)]
pub(crate) struct StoppedDuringStartup;

impl std::fmt::Display for StoppedDuringStartup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("embedded mesh runtime stopped during startup")
    }
}

impl std::error::Error for StoppedDuringStartup {}

/// An error once this embedded instance is being stopped; never for a CLI
/// runtime.
pub(crate) fn ensure_running() -> anyhow::Result<()> {
    if SHUTDOWN
        .try_with(|receiver| *receiver.borrow())
        .unwrap_or(false)
    {
        return Err(StoppedDuringStartup.into());
    }
    Ok(())
}

/// A startup that ended because of a stop is a clean exit.
pub(crate) fn stopped_during_startup_is_ok(result: anyhow::Result<()>) -> anyhow::Result<()> {
    match result {
        Err(error) if error.downcast_ref::<StoppedDuringStartup>().is_some() => Ok(()),
        result => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[tokio::test]
    async fn only_a_stopped_embedded_instance_ends_its_startup() {
        assert!(
            ensure_running().is_ok(),
            "a CLI runtime has no stop to observe"
        );

        let (stop, receiver) = watch::channel(false);
        SHUTDOWN
            .scope(receiver, async move {
                assert!(ensure_running().is_ok());
                stop.send(true).unwrap();
                let error = ensure_running().unwrap_err();
                assert!(error.downcast_ref::<StoppedDuringStartup>().is_some());
            })
            .await;
    }

    #[test]
    fn a_startup_ended_by_a_stop_is_a_clean_exit_and_other_errors_stay() {
        let stopped = Err(anyhow::Error::from(StoppedDuringStartup)).context("start mesh node");
        assert!(stopped_during_startup_is_ok(stopped).is_ok());

        let failed = Err(anyhow::anyhow!("bind failed"));
        assert!(stopped_during_startup_is_ok(failed).is_err());
    }
}
