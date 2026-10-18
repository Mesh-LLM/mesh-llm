//! Per-instance shutdown, independent of the runtime command queue. The scope
//! belongs to the embedded runtime's orchestration future, not the process or
//! its OS signal handlers. CLI runtimes have no embedded cancellation source.
use tokio::sync::watch;

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
