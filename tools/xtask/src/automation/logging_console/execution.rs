use super::{
    ResultRow, checks,
    coordinator::{Owner, Stage},
    options::Options,
    setup::Prepared,
};
use crate::{
    automation::retained_session,
    process::{self, retained::Report},
};
use std::time::Duration;

pub(super) type Session = Result<Report<String>, retained_session::Error<String>>;

#[derive(thiserror::Error)]
#[error("logging console narrative or cleanup failed; evidence retained")]
pub(super) struct Failure {
    pub report: Session,
    pub joined: bool,
}
impl std::fmt::Debug for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, formatter)
    }
}

pub(super) fn execute(prepared: Prepared, options: &Options) -> (Session, bool, Vec<ResultRow>) {
    let (requests, jobs) = std::sync::mpsc::sync_channel(1);
    let (results, responses) = std::sync::mpsc::sync_channel(1);
    let mut owner = Owner {
        initial: Some(prepared.initial),
        restarted: Some(prepared.restarted),
        browser: Some(prepared.browser),
        requests,
        responses,
        stage: Stage::InitialReady,
        pending: false,
        request_id: String::new(),
        results: Vec::new(),
        wait: prepared.browser_wait,
    };
    std::thread::scope(|scope| {
        let cancellation = process::Cancellation::default();
        let worker_cancellation = cancellation.clone();
        let directory = &prepared.directory;
        let worker = scope.spawn(move || {
            while let Ok(check) = jobs.recv() {
                let result = checks::execute(
                    check,
                    checks::CheckContext {
                        base: options.base_port,
                        root: &directory.join("requests"),
                        wait: options.wait(),
                        cancellation: &worker_cancellation,
                    },
                );
                if results.send(result).is_err() {
                    break;
                }
            }
        });
        let report = retained_session::run(
            &mut owner,
            &process::Limits {
                execution: prepared.browser_wait + options.wait() * 4 + Duration::from_secs(60),
                graceful_shutdown: Duration::from_secs(5),
                forced_shutdown: Duration::from_secs(5),
                retained_bytes_per_stream: 65536,
                readiness: process::Readiness::None,
                completion: process::Completion::Exit,
            },
        );
        let rows = std::mem::take(&mut owner.results);
        cancellation.cancel();
        drop(owner);
        (report, worker.join().is_ok(), rows)
    })
}
