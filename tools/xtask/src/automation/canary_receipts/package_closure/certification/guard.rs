//! Shared tree supervisor plus continuously observed, bounded host admission.
use super::observation::{self, Host};
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessReport, ProcessSpec, Readiness,
    },
};
use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

pub(super) struct Monitored {
    pub(super) process: ProcessReport,
    pub(super) minimum: u64,
    pub(super) guard_error: Option<String>,
}
struct Done<'a>(&'a AtomicBool);
impl Drop for Done<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

pub(super) fn run(
    spec: &ProcessSpec,
    deadline: Instant,
    outer: &Cancellation,
    initial: Host,
    reserve: u64,
    files: OutputFiles,
) -> DynResult<Monitored> {
    monitored(spec, deadline, outer, initial, reserve, files, |cancel| {
        observation::observe(&spec.cwd, cancel).map_err(|error| error.to_string())
    })
}

fn monitored(
    spec: &ProcessSpec,
    deadline: Instant,
    outer: &Cancellation,
    initial: Host,
    reserve: u64,
    files: OutputFiles,
    observe: impl Fn(&Cancellation) -> Result<Host, String> + Sync,
) -> DynResult<Monitored> {
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .ok_or("certification budget expired before process spawn")?;
    let done = AtomicBool::new(false);
    let child = Cancellation::default();
    thread::scope(|scope| {
        let monitor = scope.spawn(|| {
            let mut minimum = initial.available;
            let mut next = Instant::now();
            let mut error = None;
            while !done.load(Ordering::SeqCst) {
                if outer.is_cancelled() {
                    child.cancel();
                    break;
                }
                if Instant::now() >= deadline {
                    error = Some("family certification deadline expired".into());
                    child.cancel();
                    break;
                }
                if Instant::now() >= next {
                    match observe(&child) {
                        Ok(host) => {
                            minimum = minimum.min(host.available);
                            if host.total != initial.total || host.available < reserve {
                                error = Some(
                                    "host memory crossed 10% reserve or physical identity changed"
                                        .into(),
                                );
                                child.cancel();
                                break;
                            }
                        }
                        Err(reason) => {
                            error = Some(format!("host memory observation failed: {reason}"));
                            child.cancel();
                            break;
                        }
                    }
                    next = Instant::now() + Duration::from_secs(1);
                }
                thread::sleep(Duration::from_millis(50));
            }
            (minimum, error)
        });
        let result = {
            let _done = Done(&done);
            process::supervise(
                spec,
                &Limits {
                    execution: remaining,
                    graceful_shutdown: Duration::from_secs(15),
                    forced_shutdown: Duration::from_secs(5),
                    retained_bytes_per_stream: 16 * 1024 * 1024,
                    readiness: Readiness::None,
                    completion: Completion::Exit,
                },
                &child,
                files,
            )
        };
        let (minimum, guard_error) = monitor
            .join()
            .map_err(|_| "family memory observer panicked")?;
        Ok(Monitored {
            process: result?,
            minimum,
            guard_error,
        })
    })
}

#[cfg(test)]
#[path = "guard_tests.rs"]
mod tests;
