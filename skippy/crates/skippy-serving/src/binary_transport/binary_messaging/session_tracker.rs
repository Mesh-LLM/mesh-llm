use super::telemetry::insert_runtime_session_stats;
use crate::{
    binary_transport::binary_kv::take_shared_prefill_tokens,
    frontend::iteration_scheduler::IterationScheduler,
    kv_integration::KvStageIntegration,
    telemetry::{Telemetry, lifecycle_attrs},
};
use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_protocol::StageConfig;
use std::{
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex},
};

#[derive(Default)]
pub(super) struct ConnectionSessionOwnership {
    sessions: Mutex<BTreeMap<String, SessionState>>,
    release_completed: Condvar,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SessionState {
    Owned(u64),
    Releasing(u64),
}

pub(super) struct SessionLease {
    ownership: Arc<ConnectionSessionOwnership>,
    session_key: String,
    connection_id: u64,
}

pub(super) struct SessionRelease<'a> {
    lease: &'a SessionLease,
}

impl Drop for SessionRelease<'_> {
    fn drop(&mut self) {
        self.lease
            .ownership
            .finish_release(&self.lease.session_key, self.lease.connection_id);
    }
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        self.ownership
            .remove_lease(&self.session_key, self.connection_id);
    }
}

impl SessionLease {
    pub(super) fn begin_release(&self) -> Option<SessionRelease<'_>> {
        self.ownership
            .begin_release(&self.session_key, self.connection_id)
            .then_some(SessionRelease { lease: self })
    }

    #[cfg(test)]
    fn is_released(&self) -> bool {
        self.ownership.state(&self.session_key).is_none()
    }

    #[cfg(test)]
    fn still_owns(&self) -> bool {
        self.ownership.state(&self.session_key) == Some(SessionState::Owned(self.connection_id))
    }
}

impl ConnectionSessionOwnership {
    pub(super) fn claim(self: &Arc<Self>, session_key: &str, connection_id: u64) -> SessionLease {
        let sessions = self
            .sessions
            .lock()
            .expect("binary session ownership registry lock poisoned");
        let mut sessions = self
            .release_completed
            .wait_while(sessions, |sessions| {
                matches!(sessions.get(session_key), Some(SessionState::Releasing(_)))
            })
            .expect("binary session ownership registry lock poisoned");
        sessions.insert(session_key.to_string(), SessionState::Owned(connection_id));
        drop(sessions);
        SessionLease {
            ownership: self.clone(),
            session_key: session_key.to_string(),
            connection_id,
        }
    }

    fn begin_release(&self, session_key: &str, connection_id: u64) -> bool {
        let mut sessions = self
            .sessions
            .lock()
            .expect("binary session ownership registry lock poisoned");
        let Some(state) = sessions.get_mut(session_key) else {
            return false;
        };
        if *state != SessionState::Owned(connection_id) {
            return false;
        }
        *state = SessionState::Releasing(connection_id);
        true
    }

    fn finish_release(&self, session_key: &str, connection_id: u64) {
        {
            let mut sessions = self
                .sessions
                .lock()
                .expect("binary session ownership registry lock poisoned");
            if sessions.get(session_key) != Some(&SessionState::Releasing(connection_id)) {
                return;
            }
            sessions.remove(session_key);
        }
        self.release_completed.notify_all();
    }

    fn remove_lease(&self, session_key: &str, connection_id: u64) {
        let mut sessions = self
            .sessions
            .lock()
            .expect("binary session ownership registry lock poisoned");
        if sessions.get(session_key) == Some(&SessionState::Owned(connection_id)) {
            sessions.remove(session_key);
        }
    }

    #[cfg(test)]
    fn state(&self, session_key: &str) -> Option<SessionState> {
        self.sessions
            .lock()
            .expect("binary session ownership registry lock poisoned")
            .get(session_key)
            .copied()
    }

    #[cfg(test)]
    fn owner_count(&self) -> usize {
        self.sessions
            .lock()
            .expect("binary session ownership registry lock poisoned")
            .len()
    }
}

/// Runtime session keys created by one binary stage connection.
///
/// A connection that fails before its graceful `Stop` message would otherwise
/// leave those sessions holding execution lanes indefinitely.
pub(super) struct ConnectionSessionTracker {
    pub(super) connection_id: u64,
    ownership: Arc<ConnectionSessionOwnership>,
    active: BTreeMap<String, SessionLease>,
}

impl ConnectionSessionTracker {
    pub(super) fn new(connection_id: u64, ownership: Arc<ConnectionSessionOwnership>) -> Self {
        Self {
            connection_id,
            ownership,
            active: BTreeMap::new(),
        }
    }

    pub(super) fn touch(&mut self, session_key: &str) {
        if self.active.contains_key(session_key) {
            return;
        }
        let lease = self.ownership.claim(session_key, self.connection_id);
        self.active.insert(session_key.to_string(), lease);
    }

    pub(super) fn session_lease(&self, session_key: &str) -> &SessionLease {
        self.active
            .get(session_key)
            .expect("current binary session must be tracked")
    }

    pub(super) fn stopped(&mut self, session_key: &str) {
        self.active.remove(session_key);
    }

    fn drain(&mut self) -> BTreeMap<String, SessionLease> {
        std::mem::take(&mut self.active)
    }
}

fn reclaim_orphaned_session<T>(
    accumulated: Option<&Mutex<BTreeMap<String, Vec<i32>>>>,
    session_key: &str,
    lease: &SessionLease,
    drop_runtime_session: impl FnOnce() -> T,
) -> Option<T> {
    let release = lease.begin_release()?;
    if let Some(accumulated) = accumulated {
        take_shared_prefill_tokens(accumulated, session_key);
    }
    let result = drop_runtime_session();
    drop(release);
    Some(result)
}

/// Returns lanes and buffered prefill tokens held by sessions that never reached a graceful `Stop`.
pub(super) fn release_tracked_connection_sessions(
    config: &StageConfig,
    iteration_scheduler: &IterationScheduler,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    session_tracker: &mut ConnectionSessionTracker,
) -> Result<()> {
    let orphaned = session_tracker.drain();
    if orphaned.is_empty() {
        return Ok(());
    }
    let orphaned_count = orphaned.len();
    let accumulated = kv.map(|kv| kv.split_prefill_tokens.clone());
    let scheduler_config = config.clone();
    let scheduler_telemetry = telemetry.clone();
    let failures = iteration_scheduler
        .execute_runtime("binary-orphan-cleanup", move |runtime| {
            let mut failures = Vec::new();
            for (session_key, lease) in orphaned {
                let Some(drop_result) =
                    reclaim_orphaned_session(accumulated.as_deref(), &session_key, &lease, || {
                        runtime.drop_session_timed(&session_key)
                    })
                else {
                    continue;
                };
                match drop_result {
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
    use super::{
        ConnectionSessionOwnership, ConnectionSessionTracker,
        combine_connection_and_cleanup_results, reclaim_orphaned_session,
    };
    use anyhow::anyhow;
    use std::{
        collections::BTreeMap,
        sync::{Arc, Mutex},
    };

    fn tracker(
        connection_id: u64,
        ownership: &Arc<ConnectionSessionOwnership>,
    ) -> ConnectionSessionTracker {
        ConnectionSessionTracker::new(connection_id, ownership.clone())
    }

    #[test]
    fn tracker_drains_sessions_that_never_saw_a_stop() {
        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let mut tracker = tracker(1, &ownership);
        tracker.touch("session-a");
        tracker.touch("session-a");
        tracker.touch("session-b");
        tracker.stopped("session-b");

        let orphaned = tracker.drain();
        assert_eq!(orphaned.keys().collect::<Vec<_>>(), vec!["session-a"]);
        assert!(tracker.drain().is_empty());
        assert!(orphaned["session-a"].still_owns());
    }

    #[test]
    fn tracker_reclaims_nothing_after_graceful_stop() {
        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let mut tracker = tracker(1, &ownership);
        tracker.touch("session-a");
        tracker.stopped("session-a");
        assert!(tracker.drain().is_empty());
        assert_eq!(ownership.owner_count(), 0);
    }

    #[test]
    fn stopped_session_removes_released_owner_from_registry() {
        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let mut tracker = tracker(1, &ownership);
        tracker.touch("session-a");
        let lease = tracker.session_lease("session-a");
        drop(lease.begin_release().unwrap());

        tracker.stopped("session-a");

        assert_eq!(ownership.owner_count(), 0);
        assert!(tracker.drain().is_empty());
    }

    #[test]
    fn dropping_release_guard_marks_the_lease_released() {
        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let lease = ownership.claim("session-a", 1);

        drop(lease.begin_release().unwrap());

        assert!(lease.is_released());
    }

    #[test]
    fn stale_cleanup_preserves_replacement_tokens_and_runtime_session() {
        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let mut old = tracker(1, &ownership);
        let mut replacement = tracker(2, &ownership);
        let accumulated = Mutex::new(BTreeMap::new());

        old.touch("shared-session");
        accumulated
            .lock()
            .unwrap()
            .insert("shared-session".to_string(), vec![1, 2]);
        replacement.touch("shared-session");
        accumulated
            .lock()
            .unwrap()
            .insert("shared-session".to_string(), vec![7, 8, 9]);

        let orphaned = old.drain();
        let mut runtime_drop_called = false;
        let release = reclaim_orphaned_session(
            Some(&accumulated),
            "shared-session",
            &orphaned["shared-session"],
            || {
                runtime_drop_called = true;
                Ok::<_, ()>(())
            },
        );

        assert!(release.is_none());
        assert!(!runtime_drop_called);
        drop(orphaned);
        assert_eq!(
            accumulated.lock().unwrap().get("shared-session"),
            Some(&vec![7, 8, 9])
        );
        assert!(replacement.session_lease("shared-session").still_owns());
    }

    #[test]
    fn overlap_after_cleanup_preserves_replacement_tokens_and_runtime_session() {
        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let mut old = tracker(1, &ownership);
        let mut replacement = tracker(2, &ownership);
        let accumulated = Mutex::new(BTreeMap::from([("shared-session".to_string(), vec![1, 2])]));
        old.touch("shared-session");
        let orphaned = old.drain();
        let lease = &orphaned["shared-session"];

        let mut old_runtime_drop_called = false;
        assert_eq!(
            reclaim_orphaned_session(Some(&accumulated), "shared-session", lease, || {
                old_runtime_drop_called = true;
                Ok::<_, ()>(())
            }),
            Some(Ok(()))
        );
        assert!(old_runtime_drop_called);
        assert!(lease.is_released());
        drop(orphaned);

        replacement.touch("shared-session");
        accumulated
            .lock()
            .unwrap()
            .insert("shared-session".to_string(), vec![7, 8, 9]);

        assert_eq!(
            accumulated.lock().unwrap().get("shared-session"),
            Some(&vec![7, 8, 9])
        );
        assert!(replacement.session_lease("shared-session").still_owns());
    }

    #[test]
    fn replacement_claim_waits_for_cleanup_and_then_owns_session() {
        use std::sync::{Barrier, mpsc};
        use std::thread;

        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let mut old = tracker(1, &ownership);
        let accumulated = Arc::new(Mutex::new(BTreeMap::from([(
            "shared-session".to_string(),
            vec![1, 2],
        )])));
        old.touch("shared-session");
        let old_lease = old.drain().remove("shared-session").unwrap();
        let release_started = Arc::new(Barrier::new(2));
        let (finish_release, finish_release_rx) = mpsc::channel();
        let cleanup_accumulated = accumulated.clone();
        let cleanup_started = release_started.clone();
        let cleanup = thread::spawn(move || {
            reclaim_orphaned_session(
                Some(cleanup_accumulated.as_ref()),
                "shared-session",
                &old_lease,
                || {
                    cleanup_started.wait();
                    finish_release_rx.recv().unwrap();
                    Ok::<_, ()>(())
                },
            )
        });
        release_started.wait();

        let claim_started = Arc::new(Barrier::new(2));
        let replacement_ownership = ownership.clone();
        let replacement_started = claim_started.clone();
        let claim = thread::spawn(move || {
            replacement_started.wait();
            replacement_ownership.claim("shared-session", 2)
        });
        claim_started.wait();

        finish_release.send(()).unwrap();
        assert_eq!(cleanup.join().unwrap(), Some(Ok(())));
        let replacement = claim.join().unwrap();
        assert!(replacement.still_owns());
        assert!(!accumulated.lock().unwrap().contains_key("shared-session"));
    }

    #[test]
    fn combined_resource_takeover_waits_until_old_cleanup_finishes() {
        use std::sync::{Barrier, mpsc};
        use std::thread;
        use std::time::Duration;

        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let mut old = tracker(1, &ownership);
        let accumulated = Arc::new(Mutex::new(BTreeMap::from([(
            "shared-session".to_string(),
            vec![1, 2],
        )])));
        let runtime_session = Arc::new(Mutex::new(Some("old")));
        old.touch("shared-session");
        let old_lease = old.drain().remove("shared-session").unwrap();
        let cleanup_started = Arc::new(Barrier::new(2));
        let (finish_cleanup, finish_cleanup_rx) = mpsc::channel();
        let cleanup_tokens = accumulated.clone();
        let cleanup_runtime = runtime_session.clone();
        let cleanup_barrier = cleanup_started.clone();
        let cleanup = thread::spawn(move || {
            reclaim_orphaned_session(
                Some(cleanup_tokens.as_ref()),
                "shared-session",
                &old_lease,
                || {
                    cleanup_barrier.wait();
                    finish_cleanup_rx.recv().unwrap();
                    cleanup_runtime.lock().unwrap().take();
                },
            )
        });
        cleanup_started.wait();

        let replacement_ownership = ownership.clone();
        let replacement_tokens = accumulated.clone();
        let replacement_runtime = runtime_session.clone();
        let (replacement_done, replacement_done_rx) = mpsc::channel();
        let replacement = thread::spawn(move || {
            let lease = replacement_ownership.claim("shared-session", 2);
            replacement_tokens
                .lock()
                .unwrap()
                .insert("shared-session".to_string(), vec![7, 8, 9]);
            *replacement_runtime.lock().unwrap() = Some("replacement");
            replacement_done.send(()).unwrap();
            lease
        });

        assert!(
            replacement_done_rx
                .recv_timeout(Duration::from_millis(50))
                .is_err(),
            "replacement must not publish either resource during old cleanup"
        );
        finish_cleanup.send(()).unwrap();
        assert_eq!(cleanup.join().unwrap(), Some(()));
        replacement_done_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        let replacement = replacement.join().unwrap();

        assert!(replacement.still_owns());
        assert_eq!(
            accumulated.lock().unwrap().get("shared-session"),
            Some(&vec![7, 8, 9])
        );
        assert_eq!(*runtime_session.lock().unwrap(), Some("replacement"));
    }

    #[test]
    fn owned_orphan_cleanup_reclaims_tokens_and_runtime_session() {
        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let mut tracker = tracker(1, &ownership);
        let accumulated = Mutex::new(BTreeMap::from([
            ("session-a".to_string(), vec![1, 2, 3]),
            ("other-connection".to_string(), vec![6]),
        ]));
        tracker.touch("session-a");
        let orphaned = tracker.drain();
        let mut runtime_drop_called = false;

        let release = reclaim_orphaned_session(
            Some(&accumulated),
            "session-a",
            &orphaned["session-a"],
            || {
                runtime_drop_called = true;
                Ok::<_, ()>(())
            },
        );

        assert_eq!(release, Some(Ok(())));
        assert!(runtime_drop_called);
        assert!(orphaned["session-a"].is_released());
        assert_eq!(
            *accumulated.lock().unwrap(),
            BTreeMap::from([("other-connection".to_string(), vec![6])])
        );
    }

    #[test]
    fn cleanup_failure_releases_ownership_before_registry_cleanup() {
        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let mut tracker = tracker(1, &ownership);
        let accumulated = Mutex::new(BTreeMap::from([("session-a".to_string(), vec![1, 2, 3])]));
        tracker.touch("session-a");
        let orphaned = tracker.drain();
        let lease = &orphaned["session-a"];

        let release = reclaim_orphaned_session(Some(&accumulated), "session-a", lease, || {
            Err::<(), _>("runtime drop failed")
        });

        assert_eq!(release, Some(Err("runtime drop failed")));
        assert!(lease.is_released());
        assert_eq!(ownership.owner_count(), 0);
        drop(orphaned);
        let replacement = ownership.claim("session-a", 2);
        assert!(replacement.still_owns());
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
