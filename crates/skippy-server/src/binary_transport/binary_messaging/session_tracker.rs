use super::telemetry::insert_runtime_session_stats;
use crate::{
    runtime_state::RuntimeState,
    telemetry::{Telemetry, lifecycle_attrs},
};
use serde_json::json;
use skippy_protocol::StageConfig;
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};

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
    runtime: &Arc<Mutex<RuntimeState>>,
    telemetry: &Telemetry,
    session_tracker: &mut ConnectionSessionTracker,
) {
    let orphaned = session_tracker.drain();
    if orphaned.is_empty() {
        return;
    }
    let Ok(mut runtime) = runtime.lock() else {
        return;
    };
    for session_key in orphaned {
        match runtime.drop_session_timed(&session_key) {
            Ok(drop_stats) => {
                let mut attrs = lifecycle_attrs(config);
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
                telemetry.emit("stage.binary_session_orphan_reclaimed", attrs);
            }
            Err(error) => {
                eprintln!(
                    "failed to reclaim orphaned binary stage session {session_key}: {error:#}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ConnectionSessionTracker;

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
}
