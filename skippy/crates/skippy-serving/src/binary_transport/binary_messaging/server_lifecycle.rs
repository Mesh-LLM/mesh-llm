//! Own async binary-stage startup, shutdown, and embedded frontend tasks.
use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result};
use skippy_runtime::ActivationBoundaryDesc;

use super::{BinaryStageOptions, run_binary_stage};

pub async fn serve_binary_stage(options: BinaryStageOptions) -> Result<()> {
    serve_binary_stage_with_shutdown(options, std::future::pending::<()>()).await
}

pub async fn serve_binary_stage_with_shutdown(
    options: BinaryStageOptions,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    serve_binary_stage_with_shutdown_and_boundary_observer(options, shutdown, |_, _| {}).await
}

pub(crate) async fn serve_binary_stage_with_shutdown_and_boundary_observer(
    options: BinaryStageOptions,
    shutdown: impl Future<Output = ()> + Send + 'static,
    boundary_observer: impl FnOnce(Option<ActivationBoundaryDesc>, Option<ActivationBoundaryDesc>)
    + Send
    + 'static,
) -> Result<()> {
    run_blocking_stage_with_shutdown(shutdown, move |stop| {
        run_binary_stage(options, stop, boundary_observer)
    })
    .await
}

/// Own the frontend task so startup failures, panics and early worker errors
/// cannot leave a detached listener behind.
pub(super) struct EmbeddedFrontendTask(pub(super) Option<tokio::task::JoinHandle<Result<()>>>);

impl EmbeddedFrontendTask {
    pub(super) fn is_finished(&self) -> bool {
        self.0.as_ref().is_none_or(|task| task.is_finished())
    }

    pub(super) async fn finish(mut self) -> Result<()> {
        if let Some(task) = self.0.as_mut() {
            task.await.context("embedded OpenAI task failed")??;
        }
        self.0.take();
        Ok(())
    }
}

impl Drop for EmbeddedFrontendTask {
    fn drop(&mut self) {
        if let Some(task) = self.0.take() {
            task.abort();
        }
    }
}

pub(super) async fn wait_for_shutdown(requested: Arc<AtomicBool>) {
    while !requested.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Request cooperative shutdown even if the owner drops or aborts its future.
struct BinaryStageStopGuard {
    stop: Arc<AtomicBool>,
    stop_task: tokio::task::JoinHandle<()>,
}

impl Drop for BinaryStageStopGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.stop_task.abort();
    }
}

async fn run_blocking_stage_with_shutdown(
    shutdown: impl Future<Output = ()> + Send + 'static,
    run: impl FnOnce(Arc<AtomicBool>) -> Result<Option<EmbeddedFrontendTask>> + Send + 'static,
) -> Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_task = tokio::spawn({
        let stop = stop.clone();
        async move {
            shutdown.await;
            stop.store(true, Ordering::SeqCst);
        }
    });
    let _stop_guard = BinaryStageStopGuard {
        stop: stop.clone(),
        stop_task,
    };
    // The native load and socket loop block. Keep the async worker available
    // for frontend and shutdown tasks, including on a current-thread runtime.
    let result = tokio::task::spawn_blocking(move || run(stop))
        .await
        .context("binary stage worker failed")?;
    if let Some(frontend) = result? {
        frontend.finish().await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[tokio::test]
    async fn embedded_frontend_failure_and_panic_are_returned_to_the_owner() {
        let failed = super::EmbeddedFrontendTask(Some(tokio::spawn(async {
            Err(anyhow!("frontend bind failed"))
        })));
        assert!(
            failed
                .finish()
                .await
                .unwrap_err()
                .to_string()
                .contains("bind failed")
        );
        let panicked = super::EmbeddedFrontendTask(Some(tokio::spawn(async {
            panic!("frontend panic");
            #[allow(unreachable_code)]
            Ok(())
        })));
        assert!(
            panicked
                .finish()
                .await
                .unwrap_err()
                .to_string()
                .contains("task failed")
        );
    }

    #[tokio::test]
    async fn early_worker_exit_cancels_the_owned_frontend() {
        let (dropped, observed) = tokio::sync::oneshot::channel::<()>();
        let task = super::EmbeddedFrontendTask(Some(tokio::spawn(async move {
            let _dropped = dropped;
            std::future::pending::<()>().await;
            Ok(())
        })));
        drop(task);
        assert!(
            tokio::time::timeout(Duration::from_secs(1), observed)
                .await
                .unwrap()
                .is_err()
        );
    }

    #[tokio::test]
    async fn cancelling_the_join_also_cancels_the_frontend() {
        let (dropped, observed) = tokio::sync::oneshot::channel::<()>();
        let task = super::EmbeddedFrontendTask(Some(tokio::spawn(async move {
            let _dropped = dropped;
            std::future::pending::<()>().await;
            Ok(())
        })));
        let owner = tokio::spawn(task.finish());
        tokio::task::yield_now().await;
        owner.abort();
        let _ = owner.await;
        assert!(
            tokio::time::timeout(Duration::from_secs(1), observed)
                .await
                .unwrap()
                .is_err()
        );
    }

    #[tokio::test]
    async fn frontend_remains_owned_until_inflight_work_finishes_after_stop() {
        let requested = Arc::new(AtomicBool::new(false));
        let (draining, started) = tokio::sync::oneshot::channel();
        let (release, finished) = tokio::sync::oneshot::channel();
        let task = super::EmbeddedFrontendTask(Some(tokio::spawn({
            let requested = requested.clone();
            async move {
                super::wait_for_shutdown(requested).await;
                let _ = draining.send(());
                finished.await.context("drain interrupted")?;
                Ok(())
            }
        })));
        requested.store(true, Ordering::SeqCst);
        tokio::time::timeout(Duration::from_secs(1), started)
            .await
            .unwrap()
            .unwrap();
        assert!(
            !task.is_finished(),
            "stop request must not detach an active frontend"
        );
        release.send(()).unwrap();
        task.finish().await.unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn frontend_and_shutdown_progress_while_stage_loop_blocks() {
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let (ready, observed_ready) = tokio::sync::oneshot::channel();
        let owner = tokio::spawn(run_blocking_stage_with_shutdown(
            async move {
                let _ = stopped.await;
            },
            move |requested| {
                let frontend_stop = requested.clone();
                let frontend = EmbeddedFrontendTask(Some(tokio::spawn(async move {
                    let _ = ready.send(());
                    wait_for_shutdown(frontend_stop).await;
                    Ok(())
                })));
                while !requested.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Ok(Some(frontend))
            },
        ));
        tokio::time::timeout(Duration::from_secs(1), observed_ready)
            .await
            .unwrap()
            .unwrap();
        stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), owner)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn aborting_owner_stops_the_blocking_stage_loop() {
        let (entered, observed_entered) = tokio::sync::oneshot::channel();
        let (exited, observed_exited) = tokio::sync::oneshot::channel();
        let owner = tokio::spawn(run_blocking_stage_with_shutdown(
            std::future::pending::<()>(),
            move |requested| {
                let _ = entered.send(());
                while !requested.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                let _ = exited.send(());
                Ok(None)
            },
        ));
        tokio::time::timeout(Duration::from_secs(1), observed_entered)
            .await
            .unwrap()
            .unwrap();
        owner.abort();
        assert!(owner.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(1), observed_exited)
            .await
            .unwrap()
            .unwrap();
    }
}
