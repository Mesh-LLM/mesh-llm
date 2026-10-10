use super::{args::Options, http, observer::Observer};
use crate::{
    automation::daemon_readiness::{self as daemon, correlation::RequestId, ports::Ports},
    command::DynResult,
    process::Cancellation,
};
use std::{
    sync::mpsc,
    thread::{Scope, ScopedJoinHandle},
    time::{Duration, Instant},
};

pub(super) struct Workers<'scope> {
    daemon: ScopedJoinHandle<'scope, ()>,
    smoke: ScopedJoinHandle<'scope, ()>,
}

impl<'scope> Workers<'scope> {
    pub(super) fn start<'env>(
        scope: &'scope Scope<'scope, 'env>,
        launch: (&Options, Ports, bool, Instant),
        cancellation: &'env Cancellation,
    ) -> DynResult<(Observer, Self)> {
        let (options, ports, headless, started) = launch;
        let id = RequestId::generate()?;
        let runtime = || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
        };
        let daemon_runtime = runtime()?;
        let smoke_runtime = runtime()?;
        let (daemon_requests, daemon_work) = mpsc::sync_channel(1);
        let (daemon_results, daemon_responses) = mpsc::sync_channel(1);
        let daemon = scope.spawn(move || {
            daemon::http::work(ports, (daemon_work, daemon_results), daemon_runtime)
        });
        let (requests, work) = mpsc::sync_channel(1);
        let (results, responses) = mpsc::sync_channel(1);
        let smoke = scope.spawn(move || {
            while let Ok(request) = work.recv() {
                let result = smoke_runtime.block_on(async {
                    let cancelled = async {
                        while !cancellation.is_cancelled() {
                            tokio::time::sleep(Duration::from_millis(10)).await;
                        }
                    };
                    tokio::select! {
                        biased;
                        () = cancelled => http::ResultBody::Failed,
                        result = http::transfer(request) => result,
                    }
                });
                if results.send(result).is_err() {
                    break;
                }
            }
        });
        Ok((
            Observer {
                daemon: daemon::observer::Observer::new(
                    daemon_requests,
                    daemon_responses,
                    id,
                    options.readiness.as_secs(),
                ),
                requests,
                responses,
                ports,
                headless,
                started,
                gated: false,
                in_flight: false,
                next: Duration::ZERO,
            },
            Self { daemon, smoke },
        ))
    }

    pub(super) fn join(self) -> (bool, bool) {
        let daemon = self.daemon.join().is_ok();
        let smoke = self.smoke.join().is_ok();
        (daemon, smoke)
    }
}
