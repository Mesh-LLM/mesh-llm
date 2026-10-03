//! Fail-open persistence writer with recursion guard.
//!
//! The writer ensures request-path completion despite full queue or store worker failure. When the bus is full, drop counters increment and the caller proceeds without blocking. When the underlying sink fails, a sanitized audit record is written via an error fallback path — but this path itself cannot re-enter (recursion guard) to prevent infinite self-logging loops.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Recursion guard preventing the error-audit fallback from entering itself recursively.
pub struct RecursionGuard {
    /// A process-wide fallback write is in flight. This is intentionally shared:
    /// the error path is fail-open, so suppressing a concurrent fallback is safer
    /// than allowing it to recursively log a sink failure from another thread.
    in_error_path: AtomicBool,
}

impl RecursionGuard {
    pub fn new() -> Self {
        Self {
            in_error_path: AtomicBool::new(false),
        }
    }

    /// Try to enter the error-record path. Returns `true` if entry is allowed, `false` if we are already inside an error record (recursion detected). When returning false, no logging should occur — this prevents self-logging loops.
    pub fn try_enter_error_path(&self) -> bool {
        self.in_error_path
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    /// Exit the error-record path. Must be called after every successful `try_enter_error_path()`.
    pub fn exit_error_path(&self) {
        self.in_error_path.store(false, Ordering::Release);
    }

    /// Check if currently inside an error path (for observability / tests).
    #[allow(dead_code)]
    pub fn is_in_error_path(&self) -> bool {
        self.in_error_path.load(Ordering::Acquire)
    }
}

impl std::fmt::Debug for RecursionGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecursionGuard")
            .field("in_error_path", &self.is_in_error_path())
            .finish_non_exhaustive()
    }
}

impl Default for RecursionGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Fail-open writer that ensures request-path completion despite queue or sink failures.
pub struct FailOpenWriter {
    /// Guard preventing recursive self-logging loops in the error-audit fallback path.
    recursion_guard: Arc<RecursionGuard>,

    /// Total number of writes dropped due to full queue (incremented by bus overflow).
    pub write_drops: Arc<AtomicU64>,

    /// Number of fallback attempts suppressed because the process-wide guard
    /// was already held by recursive re-entry or concurrent contention.
    pub fallback_guard_suppressions: Arc<AtomicU64>,

    /// Number of fallback audit persistence failures deliberately suppressed
    /// instead of producing another fallback audit.
    pub fallback_persistence_suppressions: Arc<AtomicU64>,
}

impl FailOpenWriter {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            recursion_guard: Arc::new(RecursionGuard::new()),
            write_drops: Arc::new(AtomicU64::new(0)),
            fallback_guard_suppressions: Arc::new(AtomicU64::new(0)),
            fallback_persistence_suppressions: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Attempt to record an error/audit entry. Returns `true` when the fallback
    /// path was entered and the recorder completed. Returns `false` when the
    /// recursion guard blocks entry or the recorder panics; the caller proceeds
    /// silently in either case. This method never panics.
    pub fn try_record_error<F>(&self, recorder: F) -> bool
    where
        F: FnOnce() + Send + Sync + 'static,
    {
        if !self.recursion_guard.try_enter_error_path() {
            self.fallback_guard_suppressions
                .fetch_add(1, Ordering::Relaxed);
            return false; // Recursion detected — abort silently.
        }

        // Execute the recorder (best-effort). Wrap in catch_unwind to prevent panics from propagating to request paths.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            recorder();
        }));

        self.recursion_guard.exit_error_path();

        // If the recorder panicked, we silently absorb it (fail-open). Neither
        // suppression counter increments here because this was a valid entry
        // that happened to fail.
        result.is_ok()
    }

    /// Record a write drop due to full queue or sink failure. This is called by the service when enqueue fails. Incrementing this counter is itself fail-open (no-op if anything goes wrong).
    pub fn record_drop(&self) {
        self.write_drops.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a failure of the fallback audit itself was deliberately
    /// suppressed. A fallback is an ordinary canonical System event, so its
    /// persistence failure must never create another fallback event.
    pub fn record_fallback_suppressed(&self) {
        self.fallback_persistence_suppressions
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Clone recursion guard for external observation.
    #[allow(dead_code)]
    pub fn recursion_guard_clone(&self) -> Arc<RecursionGuard> {
        self.recursion_guard.clone()
    }

    /// Whether a recursive error path is currently active (for tests).
    #[allow(dead_code)]
    pub fn is_in_error_path(&self) -> bool {
        self.recursion_guard.is_in_error_path()
    }
}

impl Default for FailOpenWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_enter_allows_first_call() {
        let guard = RecursionGuard::new();
        assert!(guard.try_enter_error_path());
        guard.exit_error_path();
    }

    #[test]
    fn try_enter_blocks_second_nested_call() {
        let guard = RecursionGuard::new();
        assert!(guard.try_enter_error_path());

        // Second nested call should be blocked.
        assert!(!guard.try_enter_error_path());

        guard.exit_error_path();
    }

    #[test]
    fn fallback_guard_suppression_counter_increments() {
        let writer = Arc::new(FailOpenWriter::new());
        assert_eq!(
            writer.fallback_guard_suppressions.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            writer
                .fallback_persistence_suppressions
                .load(Ordering::Relaxed),
            0
        );

        let nested_writer = writer.clone();
        assert!(writer.try_record_error(move || {
            assert!(!nested_writer.try_record_error(|| {}));
        }));

        assert_eq!(
            writer.fallback_guard_suppressions.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            writer
                .fallback_persistence_suppressions
                .load(Ordering::Relaxed),
            0
        );
        assert!(writer.try_record_error(|| {}));
    }

    #[test]
    fn fallback_persistence_suppression_counter_is_separate_from_guard_suppression() {
        let writer = FailOpenWriter::new();

        writer.record_fallback_suppressed();

        assert_eq!(
            writer.fallback_guard_suppressions.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            writer
                .fallback_persistence_suppressions
                .load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn try_record_error_catches_panic() {
        let writer = FailOpenWriter::new();

        // Recorder that panics — should be caught, not propagate.
        let result = writer.try_record_error(|| {
            panic!("simulated recorder panic");
        });

        // Returns false (panic was caught).
        assert!(!result);
        assert!(!writer.is_in_error_path());
        assert!(writer.try_record_error(|| {}));
    }

    #[test]
    fn write_drop_counter_increments() {
        let writer = FailOpenWriter::new();
        assert_eq!(writer.write_drops.load(Ordering::Relaxed), 0);

        for _ in 0..10 {
            writer.record_drop();
        }

        assert_eq!(writer.write_drops.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn writer_is_default() {
        let _writer: FailOpenWriter = Default::default();
    }

    #[test]
    fn recursion_guard_is_default() {
        let _guard: RecursionGuard = Default::default();
    }

    #[test]
    fn error_path_allows_re_entry_after_exit() {
        let guard = RecursionGuard::new();

        // First entry.
        assert!(guard.try_enter_error_path());
        guard.exit_error_path();

        // After exit, can enter again.
        assert!(guard.try_enter_error_path());
        guard.exit_error_path();
    }

    #[test]
    fn writer_try_record_success() {
        let writer = FailOpenWriter::new();

        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let c = called.clone();
        assert!(writer.try_record_error(move || {
            c.store(true, Ordering::Release);
        }));
        assert!(called.load(Ordering::Acquire));
    }

    #[test]
    fn concurrent_error_path_is_suppressed_and_restored() {
        use std::sync::Barrier;
        use std::thread;

        let writer = Arc::new(FailOpenWriter::new());
        let guard = writer.recursion_guard_clone();
        let barrier = Arc::new(Barrier::new(2));

        assert!(guard.try_enter_error_path());

        let worker_writer = writer.clone();
        let worker_barrier = barrier.clone();
        let handle = thread::spawn(move || {
            worker_barrier.wait();
            worker_writer.try_record_error(|| {})
        });

        barrier.wait();
        assert!(!handle.join().expect("worker thread panicked"));
        assert_eq!(
            writer.fallback_guard_suppressions.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            writer
                .fallback_persistence_suppressions
                .load(Ordering::Relaxed),
            0
        );

        guard.exit_error_path();
        assert!(writer.try_record_error(|| {}));
    }
}
