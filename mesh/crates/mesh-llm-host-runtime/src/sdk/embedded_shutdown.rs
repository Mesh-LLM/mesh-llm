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
