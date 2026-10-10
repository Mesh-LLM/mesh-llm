//! Recovery of the runtime lock after a panic.
//!
//! A request that panics while it holds the runtime lock poisons the mutex.
//! Treating poison as fatal turned one panic into an outage that lasted until
//! restart. Take the runtime lock through [`lock_runtime`] or
//! [`try_lock_runtime`] instead: they reset the state the panicking request
//! may have left half-updated, clear the poison, and hand out the lock.
//!
//! A request can be between two runtime operations when its session is
//! reset. Its next operation would otherwise start a fresh, empty session
//! under the same ID and carry on without its prompt. Reset IDs are
//! remembered until the request drops its session, and operations on them
//! fail in the meantime.

use std::sync::{Mutex, MutexGuard, PoisonError, TryLockError};

use anyhow::{Result, bail};

use super::RuntimeState;

/// State behind a mutex that can be made consistent again after a thread
/// panicked while holding the lock.
pub trait PanicRecovery {
    /// Discards whatever an interrupted operation may have left half-done.
    fn reset_after_panic(&mut self);
}

impl PanicRecovery for RuntimeState {
    /// Resets every active lane. The lock does not record which lane the
    /// panicking request was using, so all of them are suspect. Requests that
    /// still held a lane fail with a missing-session error; later requests
    /// start on clean lanes.
    fn reset_after_panic(&mut self) {
        let mut session_ids: Vec<String> = self.sessions.keys().cloned().collect();
        session_ids.extend(self.session_token_counts.keys().cloned());
        session_ids.sort_unstable();
        session_ids.dedup();
        for session_id in &session_ids {
            let _ = self.drop_session_timed(session_id);
        }
        self.sessions_reset_by_panic
            .extend(session_ids.iter().cloned());
        let _ = skippy_events::diagnostics::emit(
            skippy_events::diagnostics::ServingDiagnostic::Warning {
                message: "Recovered the Skippy runtime lock after a panic".to_string(),
                context: Some(format!("reset_sessions={}", session_ids.len())),
            },
        );
    }
}

impl RuntimeState {
    /// Fails if panic recovery discarded `session_id` and its request has
    /// not dropped it yet.
    pub(super) fn ensure_not_reset_by_panic(&self, session_id: &str) -> Result<()> {
        if self.sessions_reset_by_panic.contains(session_id) {
            bail!("session {session_id} was reset after a runtime panic; retry the request");
        }
        Ok(())
    }

    /// Forgets a reset session once its request drops it.
    pub(super) fn forget_reset_session(&mut self, session_id: &str) {
        self.sessions_reset_by_panic.remove(session_id);
    }
}

/// Locks the runtime, recovering it if a previous holder panicked.
pub fn lock_runtime(runtime: &Mutex<RuntimeState>) -> MutexGuard<'_, RuntimeState> {
    lock_recovering(runtime)
}

/// Locks the runtime without blocking, recovering it if a previous holder
/// panicked. Returns `None` while another thread holds the lock.
pub fn try_lock_runtime(runtime: &Mutex<RuntimeState>) -> Option<MutexGuard<'_, RuntimeState>> {
    try_lock_recovering(runtime)
}

pub(crate) fn lock_recovering<S: PanicRecovery>(mutex: &Mutex<S>) -> MutexGuard<'_, S> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| recover(mutex, poisoned))
}

pub(crate) fn try_lock_recovering<S: PanicRecovery>(mutex: &Mutex<S>) -> Option<MutexGuard<'_, S>> {
    match mutex.try_lock() {
        Ok(guard) => Some(guard),
        Err(TryLockError::WouldBlock) => None,
        Err(TryLockError::Poisoned(poisoned)) => Some(recover(mutex, poisoned)),
    }
}

fn recover<'a, S: PanicRecovery>(
    mutex: &'a Mutex<S>,
    poisoned: PoisonError<MutexGuard<'a, S>>,
) -> MutexGuard<'a, S> {
    let mut guard = poisoned.into_inner();
    guard.reset_after_panic();
    mutex.clear_poison();
    guard
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;

    fn poison_with_tracked_session(runtime: &Arc<Mutex<RuntimeState>>) {
        let poisoner = Arc::clone(runtime);
        let _ = thread::spawn(move || {
            let mut runtime = poisoner.lock().unwrap();
            runtime.track_session_tokens_for_test("panicked", 3);
            panic!("request panicked while holding the runtime lock");
        })
        .join();
        assert!(runtime.is_poisoned(), "precondition: runtime is poisoned");
    }

    #[test]
    fn lock_runtime_resets_sessions_and_clears_poison() {
        let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
        poison_with_tracked_session(&runtime);

        let guard = lock_runtime(&runtime);
        assert_eq!(guard.session_stats().tracked_token_counts, 0);
        drop(guard);
        assert!(!runtime.is_poisoned());
        assert!(runtime.lock().is_ok());
    }

    #[test]
    fn try_lock_runtime_recovers_a_poisoned_lock() {
        let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
        poison_with_tracked_session(&runtime);

        let guard = try_lock_runtime(&runtime).expect("an unheld poisoned lock is available");
        assert_eq!(guard.session_stats().tracked_token_counts, 0);
        drop(guard);
        assert!(!runtime.is_poisoned());
    }

    #[test]
    fn sessions_reset_by_recovery_fail_instead_of_restarting_empty() {
        // Zero lanes, so any attempt to start a fresh session fails before
        // native code, unless recovery already refuses it.
        let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(0)));
        poison_with_tracked_session(&runtime);

        let mut guard = lock_runtime(&runtime);
        let error = guard.ensure_session_active("panicked").unwrap_err();
        assert!(error.to_string().contains("reset"), "{error:#}");

        // Once the request cleans up, the ID may be used again.
        let _ = guard.drop_session_timed("panicked");
        let error = guard.ensure_session_active("panicked").unwrap_err();
        assert!(!error.to_string().contains("reset"), "{error:#}");
    }

    #[test]
    fn try_lock_runtime_reports_a_held_lock() {
        let runtime = Mutex::new(RuntimeState::new_modelless_for_test(1));
        let _held = runtime.lock().unwrap();
        assert!(try_lock_runtime(&runtime).is_none());
    }
}
