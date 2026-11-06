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
    capacity: Option<usize>,
    runtime_lanes: Option<Arc<crate::runtime_state::lane_admission::RuntimeLaneAdmission>>,
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
    runtime_lane: Mutex<Option<crate::runtime_state::lane_admission::RuntimeLanePermit>>,
}

pub(super) struct SessionRelease<'a> {
    lease: &'a SessionLease,
}

impl Drop for SessionRelease<'_> {
    fn drop(&mut self) {
        self.lease
            .ownership
            .finish_release(&self.lease.session_key, self.lease.connection_id);
        self.lease
            .runtime_lane
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
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
    pub(super) fn with_runtime_lanes(
        lanes: Arc<crate::runtime_state::lane_admission::RuntimeLaneAdmission>,
    ) -> Self {
        Self {
            capacity: Some(lanes.capacity()),
            runtime_lanes: Some(lanes),
            ..Self::default()
        }
    }

    #[cfg(test)]
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: Some(capacity.max(1)),
            ..Self::default()
        }
    }

    #[cfg(test)]
    pub(super) fn claim(
        self: &Arc<Self>,
        session_key: &str,
        connection_id: u64,
    ) -> Result<SessionLease> {
        self.claim_interruptible(session_key, connection_id, || false)
    }

    fn claim_interruptible(
        self: &Arc<Self>,
        session_key: &str,
        connection_id: u64,
        cancelled: impl Fn() -> bool,
    ) -> Result<SessionLease> {
        let mut sessions = self
            .sessions
            .lock()
            .expect("binary session ownership registry lock poisoned");
        loop {
            if cancelled() {
                bail!("binary session admission cancelled");
            }
            let releasing = matches!(sessions.get(session_key), Some(SessionState::Releasing(_)));
            let capacity_full = !sessions.contains_key(session_key)
                && self
                    .capacity
                    .is_some_and(|capacity| sessions.len() >= capacity);
            if capacity_full
                && sessions
                    .values()
                    .any(|state| *state == SessionState::Owned(connection_id))
            {
                // This connection must remain able to consume Stop for its
                // existing sessions; waiting would hide that Stop behind the
                // new session's admission indefinitely.
                bail!("stop an active session before opening another beyond stage capacity");
            }
            if !releasing && !capacity_full {
                break;
            }
            sessions = self
                .release_completed
                .wait_timeout(sessions, std::time::Duration::from_millis(50))
                .expect("binary session ownership registry lock poisoned")
                .0;
        }
        // Never seize ownership from another live connection. Nonzero wire
        // session ids become session keys, so an observing peer could
        // otherwise replay a victim's live session id on its own connection
        // and take over the victim's session state. Reclaiming a session this
        // connection already owns, or claiming one the previous owner fully
        // released, remains allowed.
        let seized_live_owner = matches!(
            sessions.get(session_key),
            Some(SessionState::Owned(owner)) if *owner != connection_id
        );
        if seized_live_owner {
            bail!("stage session is owned by another live connection");
        }
        // HTTP can occupy lanes without an entry in this binary registry. A
        // connection that already owns a session may only add another when a
        // real runtime permit is immediately available: blocking here would
        // prevent it from consuming Stop for its existing session.
        let runtime_lane = if let Some(lanes) = &self.runtime_lanes
            && sessions
                .values()
                .any(|state| *state == SessionState::Owned(connection_id))
        {
            Some(lanes.try_acquire()?.ok_or_else(|| {
                anyhow::anyhow!(
                    "stop an active session before opening another beyond stage capacity"
                )
            })?)
        } else {
            None
        };
        let needs_runtime_lane = runtime_lane.is_none();
        sessions.insert(session_key.to_string(), SessionState::Owned(connection_id));
        drop(sessions);
        let lease = SessionLease {
            ownership: self.clone(),
            session_key: session_key.to_string(),
            connection_id,
            runtime_lane: Mutex::new(runtime_lane),
        };
        if needs_runtime_lane && let Some(lanes) = &self.runtime_lanes {
            let permit = lanes.acquire_blocking(cancelled)?;
            *lease
                .runtime_lane
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(permit);
        }
        Ok(lease)
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
        drop(sessions);
        self.release_completed.notify_all();
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

    #[cfg(test)]
    pub(super) fn touch(&mut self, session_key: &str) -> Result<()> {
        self.touch_interruptible(session_key, || false)
    }

    pub(super) fn touch_interruptible(
        &mut self,
        session_key: &str,
        cancelled: impl Fn() -> bool,
    ) -> Result<()> {
        if self.active.contains_key(session_key) {
            return Ok(());
        }
        let lease =
            self.ownership
                .claim_interruptible(session_key, self.connection_id, cancelled)?;
        self.active.insert(session_key.to_string(), lease);
        Ok(())
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

    #[test]
    fn global_lane_admission_waits_for_another_connections_session_release() {
        use std::{
            sync::atomic::{AtomicBool, Ordering},
            sync::mpsc,
            time::Duration,
        };
        let ownership = Arc::new(ConnectionSessionOwnership::with_capacity(1));
        let first = ownership.claim("first", 1).unwrap();
        let (waiting_tx, waiting_rx) = mpsc::channel();
        let (admitted_tx, admitted_rx) = mpsc::channel();
        let queued_owner = ownership.clone();
        let queued = std::thread::spawn(move || {
            let notified = AtomicBool::new(false);
            let second = queued_owner
                .claim_interruptible("second", 2, || {
                    if !notified.swap(true, Ordering::AcqRel) {
                        waiting_tx.send(()).unwrap();
                    }
                    false
                })
                .unwrap();
            admitted_tx.send(()).unwrap();
            second
        });
        waiting_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(admitted_rx.try_recv().is_err());
        // Native Stop/cleanup marks the ownership released only after the
        // actual runtime session has been dropped.
        drop(first.begin_release().unwrap());
        admitted_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(queued.join().unwrap());
        drop(first);
        assert_eq!(ownership.owner_count(), 0);
    }

    #[test]
    fn shutdown_cancels_capacity_wait_without_claiming_a_lane() {
        let ownership = Arc::new(ConnectionSessionOwnership::with_capacity(1));
        let first = ownership.claim("first", 1).unwrap();
        assert!(ownership.claim_interruptible("second", 2, || true).is_err());
        assert!(first.still_owns());
        assert_eq!(ownership.owner_count(), 1);
    }

    #[test]
    fn same_connection_over_capacity_does_not_hide_its_queued_stop() {
        let ownership = Arc::new(ConnectionSessionOwnership::with_capacity(1));
        let first = ownership.claim("first", 1).unwrap();
        let error = ownership
            .claim("second", 1)
            .err()
            .expect("multiplexed session must not block its own Stop");
        assert!(error.to_string().contains("stop an active session"));
        assert!(first.still_owns());
        drop(first.begin_release().unwrap());
        assert!(ownership.claim("second", 1).is_ok());
    }

    #[tokio::test]
    async fn http_occupancy_cannot_make_a_second_binary_session_hide_stop() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let lanes = crate::runtime_state::lane_admission::RuntimeLaneAdmission::new(2);
        let _http = lanes.acquire().await.unwrap();
        let ownership = Arc::new(ConnectionSessionOwnership::with_runtime_lanes(lanes));
        let first = ownership.claim("first", 1).unwrap();
        let checks = AtomicUsize::new(0);
        // A regressed blocking claim is cancelled on its second poll so this
        // regression fails deterministically instead of hanging the suite.
        let error = ownership
            .claim_interruptible("second", 1, || checks.fetch_add(1, Ordering::AcqRel) > 0)
            .err()
            .expect("an occupied runtime must refuse same-connection multiplexing");
        assert!(error.to_string().contains("stop an active session"));
        assert_eq!(checks.load(Ordering::Acquire), 1);
        assert_eq!(ownership.owner_count(), 1);
        assert!(first.still_owns());
    }

    #[test]
    fn same_binary_connection_can_use_an_available_second_runtime_lane() {
        let lanes = crate::runtime_state::lane_admission::RuntimeLaneAdmission::new(2);
        let ownership = Arc::new(ConnectionSessionOwnership::with_runtime_lanes(lanes));
        let first = ownership.claim("first", 1).unwrap();
        let second = ownership.claim("second", 1).unwrap();
        assert!(first.still_owns());
        assert!(second.still_owns());
        assert_eq!(ownership.owner_count(), 2);
    }

    #[tokio::test]
    async fn binary_session_waits_for_http_runtime_reservation() {
        use std::{
            sync::atomic::{AtomicBool, Ordering},
            sync::mpsc,
            time::Duration,
        };
        let runtime = crate::runtime_state::RuntimeState::new_modelless_for_test(1);
        let lanes = runtime.lane_admission();
        let http = lanes.acquire().await.unwrap();
        let ownership = Arc::new(ConnectionSessionOwnership::with_runtime_lanes(lanes));
        let queued_owner = ownership.clone();
        let (waiting_tx, waiting_rx) = mpsc::channel();
        let (admitted_tx, admitted_rx) = mpsc::channel();
        let queued = std::thread::spawn(move || {
            let notified = AtomicBool::new(false);
            let lease = queued_owner
                .claim_interruptible("binary", 7, || {
                    if !notified.swap(true, Ordering::AcqRel) {
                        waiting_tx.send(()).unwrap();
                    }
                    false
                })
                .unwrap();
            admitted_tx.send(()).unwrap();
            lease
        });
        waiting_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(admitted_rx.try_recv().is_err());
        drop(http);
        admitted_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let lease = queued.join().unwrap();
        drop(lease.begin_release().unwrap());
        drop(lease);
        assert_eq!(ownership.owner_count(), 0);
    }

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
        tracker.touch("session-a").unwrap();
        tracker.touch("session-a").unwrap();
        tracker.touch("session-b").unwrap();
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
        tracker.touch("session-a").unwrap();
        tracker.stopped("session-a");
        assert!(tracker.drain().is_empty());
        assert_eq!(ownership.owner_count(), 0);
    }

    #[test]
    fn stopped_session_removes_released_owner_from_registry() {
        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let mut tracker = tracker(1, &ownership);
        tracker.touch("session-a").unwrap();
        let lease = tracker.session_lease("session-a");
        drop(lease.begin_release().unwrap());

        tracker.stopped("session-a");

        assert_eq!(ownership.owner_count(), 0);
        assert!(tracker.drain().is_empty());
    }

    #[test]
    fn dropping_release_guard_marks_the_lease_released() {
        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let lease = ownership.claim("session-a", 1).unwrap();

        drop(lease.begin_release().unwrap());

        assert!(lease.is_released());
    }

    #[test]
    fn stale_cleanup_preserves_replacement_tokens_and_runtime_session() {
        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let mut old = tracker(1, &ownership);
        let mut replacement = tracker(2, &ownership);
        let accumulated = Mutex::new(BTreeMap::new());

        old.touch("shared-session").unwrap();
        accumulated
            .lock()
            .unwrap()
            .insert("shared-session".to_string(), vec![1, 2]);
        {
            let lease = old.session_lease("shared-session");
            drop(lease.begin_release().unwrap());
        }
        replacement.touch("shared-session").unwrap();
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
        old.touch("shared-session").unwrap();
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

        replacement.touch("shared-session").unwrap();
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
        old.touch("shared-session").unwrap();
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
            replacement_ownership.claim("shared-session", 2).unwrap()
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
        old.touch("shared-session").unwrap();
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
            let lease = replacement_ownership.claim("shared-session", 2).unwrap();
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
        tracker.touch("session-a").unwrap();
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
        tracker.touch("session-a").unwrap();
        let orphaned = tracker.drain();
        let lease = &orphaned["session-a"];

        let release = reclaim_orphaned_session(Some(&accumulated), "session-a", lease, || {
            Err::<(), _>("runtime drop failed")
        });

        assert_eq!(release, Some(Err("runtime drop failed")));
        assert!(lease.is_released());
        assert_eq!(ownership.owner_count(), 0);
        drop(orphaned);
        let replacement = ownership.claim("session-a", 2).unwrap();
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

    #[test]
    fn second_connection_cannot_claim_a_live_owned_session() {
        let ownership = Arc::new(ConnectionSessionOwnership::default());
        let mut owner = tracker(1, &ownership);
        let mut claimant = tracker(2, &ownership);

        owner.touch("7").unwrap();
        assert!(claimant.touch("7").is_err());

        assert!(
            owner.session_lease("7").still_owns(),
            "a second connection must not seize ownership of a live session"
        );
        assert!(ownership.claim("7", 2).is_err());
        assert!(
            claimant.active.is_empty(),
            "denied session must never be tracked for processing"
        );
    }
}
