//! Names the shutdown step a stop is stuck in.
//!
//! A stop that hung after `node_draining` left nothing behind until a
//! supervisor killed the process (#3067): the steps log nothing on their own,
//! and the timeout warnings some of them carry go through `tracing`, which the
//! default filter drops for their modules. Each step now reports on the output
//! stream once it runs past [`SLOW_SHUTDOWN_STEP`], while it is still running,
//! so a kill that follows still leaves the step's name in the log.

use mesh_llm_events::{OutputEvent, emit_event};
use std::future::Future;
use std::time::Duration;

/// Well past `broadcast_leaving`'s 200 ms flush, and far enough below the
/// client readiness smoke's 15 s shutdown budget that the report lands before
/// a supervisor kills the process.
const SLOW_SHUTDOWN_STEP: Duration = Duration::from_secs(2);

/// Runs one shutdown step, reporting it as a warning on the output stream
/// when it is still running after [`SLOW_SHUTDOWN_STEP`], and again with its
/// duration once it ends.
pub(super) async fn shutdown_step<T>(step: &'static str, work: impl Future<Output = T>) -> T {
    run_shutdown_step(step, SLOW_SHUTDOWN_STEP, work, |message| {
        let _ = emit_event(OutputEvent::Warning {
            message,
            context: Some("shutdown".to_string()),
        });
    })
    .await
}

async fn run_shutdown_step<T>(
    step: &'static str,
    threshold: Duration,
    work: impl Future<Output = T>,
    mut report: impl FnMut(String),
) -> T {
    let started = tokio::time::Instant::now();
    tokio::pin!(work);
    tokio::select! {
        biased;
        output = &mut work => return output,
        () = tokio::time::sleep(threshold) => {}
    }
    report(format!(
        "shutdown step {step} is still running after {:.1} s",
        threshold.as_secs_f64()
    ));
    let output = work.await;
    report(format!(
        "shutdown step {step} finished after {:.1} s",
        started.elapsed().as_secs_f64()
    ));
    output
}

#[cfg(test)]
mod tests {
    use super::run_shutdown_step;
    use std::time::Duration;

    #[tokio::test(start_paused = true)]
    async fn a_quick_step_reports_nothing_and_returns_its_output() {
        let mut reports = Vec::new();
        let output = run_shutdown_step(
            "close_endpoint",
            Duration::from_secs(2),
            async {
                tokio::time::sleep(Duration::from_millis(200)).await;
                7
            },
            |message| reports.push(message),
        )
        .await;

        assert_eq!(output, 7);
        assert!(reports.is_empty(), "{reports:?}");
    }

    /// The first report has to come while the step is still running: a stop
    /// that hangs is killed before the step ends, as in #3067.
    #[tokio::test(start_paused = true)]
    async fn a_slow_step_is_named_before_it_ends_and_timed_when_it_does() {
        let mut reports = Vec::new();
        let output = run_shutdown_step(
            "close_memberships",
            Duration::from_secs(2),
            async {
                tokio::time::sleep(Duration::from_secs(10)).await;
                "closed"
            },
            |message| reports.push(message),
        )
        .await;

        assert_eq!(output, "closed");
        assert_eq!(
            reports,
            [
                "shutdown step close_memberships is still running after 2.0 s",
                "shutdown step close_memberships finished after 10.0 s",
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_step_that_never_ends_is_still_named() {
        let mut reports = Vec::new();
        let step = run_shutdown_step(
            "shutdown_run_auto_services",
            Duration::from_secs(2),
            std::future::pending::<()>(),
            |message| reports.push(message),
        );
        assert!(
            tokio::time::timeout(Duration::from_secs(15), step)
                .await
                .is_err()
        );

        assert_eq!(
            reports,
            ["shutdown step shutdown_run_auto_services is still running after 2.0 s"]
        );
    }
}
