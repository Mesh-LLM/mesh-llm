use super::telemetry::insert_runtime_session_stats;
use crate::{
    frontend::iteration_scheduler::IterationScheduler,
    telemetry::{Telemetry, lifecycle_attrs},
};
use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_protocol::StageConfig;
use std::collections::BTreeSet;

/// Runtime session keys created by one binary stage connection.
///
/// A connection that fails before its graceful `Stop` message would otherwise
/// leave those sessions holding execution lanes indefinitely.
#[derive(Default)]
pub(super) struct ConnectionSessionTracker {
    active: BTreeSet<String>,
}

impl ConnectionSessionTracker {
    pub(super) fn touch(&mut self, session_key: &str) {
        self.active.insert(session_key.to_string());
    }

    pub(super) fn stopped(&mut self, session_key: &str) {
        self.active.remove(session_key);
    }

    fn drain(&mut self) -> Vec<String> {
        std::mem::take(&mut self.active).into_iter().collect()
    }
}

/// Returns lanes held by sessions that never reached a graceful `Stop`.
pub(super) fn release_tracked_connection_sessions(
    config: &StageConfig,
    iteration_scheduler: &IterationScheduler,
    telemetry: &Telemetry,
    session_tracker: &mut ConnectionSessionTracker,
) -> Result<()> {
    let orphaned = session_tracker.drain();
    if orphaned.is_empty() {
        return Ok(());
    }
    let orphaned_count = orphaned.len();
    let scheduler_config = config.clone();
    let scheduler_telemetry = telemetry.clone();
    let failures = iteration_scheduler
        .execute_runtime("binary-orphan-cleanup", move |runtime| {
            let mut failures = Vec::new();
            for session_key in orphaned {
                match runtime.drop_session_timed(&session_key) {
                    Ok(drop_stats) => {
                        let mut attrs = lifecycle_attrs(&scheduler_config);
                        attrs.insert("llama_stage.session_key".to_string(), json!(session_key));
                        attrs.insert(
                            "llama_stage.session_reset".to_string(),
                            json!(drop_stats.reset_session),
                        );
                        attrs.insert(
                            "llama_stage.lane_discarded".to_string(),
                            json!(drop_stats.lane_discarded),
                        );
                        insert_runtime_session_stats(
                            &mut attrs,
                            "llama_stage.runtime_sessions_after",
                            &drop_stats.stats_after,
                        );
                        scheduler_telemetry.emit("stage.binary_session_orphan_reclaimed", attrs);
                    }
                    Err(error) => {
                        failures.push(format!("{session_key}: {error:#}"));
                    }
                }
            }
            Ok(failures)
        })
        .map_err(|error| anyhow::anyhow!(format!("{error:#}")))?;
    if !failures.is_empty() {
        bail!(
            "failed to reclaim {}/{} orphaned binary stage session(s): {}",
            failures.len(),
            orphaned_count,
            failures.join("; ")
        );
    }
    Ok(())
}

pub(super) fn combine_connection_and_cleanup_results(
    connection_result: Result<()>,
    cleanup_result: Result<()>,
) -> Result<()> {
    match (connection_result, cleanup_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(connection_error), Ok(())) => Err(connection_error),
        (Ok(()), Err(cleanup_error)) => Err(cleanup_error),
        (Err(connection_error), Err(cleanup_error)) => Err(connection_error).with_context(|| {
            format!("orphaned binary stage session cleanup also failed: {cleanup_error:#}")
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{ConnectionSessionTracker, combine_connection_and_cleanup_results};
    use anyhow::anyhow;

    #[test]
    fn tracker_drains_sessions_that_never_saw_a_stop() {
        let mut tracker = ConnectionSessionTracker::default();
        tracker.touch("session-a");
        tracker.touch("session-a");
        tracker.touch("session-b");
        tracker.stopped("session-b");

        assert_eq!(tracker.drain(), vec!["session-a"]);
        assert!(tracker.drain().is_empty());
    }

    #[test]
    fn tracker_reclaims_nothing_after_graceful_stop() {
        let mut tracker = ConnectionSessionTracker::default();
        tracker.touch("session-a");
        tracker.stopped("session-a");
        assert!(tracker.drain().is_empty());
    }

    #[test]
    fn cleanup_failure_is_returned_when_connection_succeeded() {
        let error =
            combine_connection_and_cleanup_results(Ok(()), Err(anyhow!("orphan cleanup failed")))
                .expect_err("cleanup failure must reach the connection supervisor");

        assert!(error.to_string().contains("orphan cleanup failed"));
    }

    #[test]
    fn connection_and_cleanup_failures_are_both_preserved() {
        let error = combine_connection_and_cleanup_results(
            Err(anyhow!("connection failed")),
            Err(anyhow!("orphan cleanup failed")),
        )
        .expect_err("combined lifecycle failures must be returned");
        let message = format!("{error:#}");

        assert!(message.contains("connection failed"));
        assert!(message.contains("orphan cleanup failed"));
    }
}
